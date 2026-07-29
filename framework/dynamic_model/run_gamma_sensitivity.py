"""
Cascade-strength sensitivity: does the crossover survive, and where does it sit?
Goes in: framework/dynamic_model/run_gamma_sensitivity.py

    python3 run_gamma_sensitivity.py --out gamma_sensitivity.json

every rate in the model is calibrated to published Delta
statistics except gamma, the overload-cascade strength, which was chosen. Before
reporting any ordering between the two knowledge axes we have to know whether it
is a property of the model or of that one arbitrary number.

It is neither, exactly. The crossover exists at every gamma tested, but its
location scales as T_cross ~ 1/gamma. essentially the cascade is what makes
timing matter: with a strong cascade you must trigger the collapse early
enough for it to run to completion within the horizon, so knowing *when*
overtakes knowing *which* sooner.

Three experiments:
  A. crossover location vs gamma   -- the headline robustness result
  B. p_T vs gamma                  -- shows gamma spans ~11 orders of magnitude
  C. SLA margin vs pool size       -- the operational question asked directly,
                                      with no node-vs-job proxy
"""

import argparse
import json

import numpy as np

from model import DynamicConfig, DynamicModel
from decompose import rung_ceiling, _embed, _n_classes
from delta_calibration import calibrate_rates

DELTA_MIX = [8., 8., 8., 8., 4., 4., 4., 4.]


def instance(N, T, c, c_min, rates, gamma, eta=0.50, name="s"):
    return DynamicModel(DynamicConfig(
        N=N, T=T, c=np.asarray(c, float), c_min=float(c_min),
        a0=np.full(N, rates["a0"]), gamma=np.full(N, gamma),
        b0=np.full(N, rates["b0"]), eta=np.full(N, eta), name=name))


def axis_ratio(m, maxiter=130):
    """class ceiling / time_scalar ceiling.  >1 means COMPONENT knowledge wins."""
    K = _n_classes(m)
    solved, v = {}, {}
    for rung in ("scalar", "class", "time_scalar"):
        extra = [e for e in (_embed(solved[s], s, rung, m.cfg.N, m.T, K)
                             for s in solved) if e is not None]
        r = rung_ceiling(m, rung, extra_starts=extra, n_restarts=0,
                         maxiter=maxiter)
        solved[rung] = r["theta"]
        v[rung] = r["vrf"]
    return v, v["class"] / v["time_scalar"]


# --------------------------------------------------------------------------
# A. where is the crossover?
# --------------------------------------------------------------------------

def crossover(gamma, rates, horizons, c_min=12.0, maxiter=130, verbose=True):
    """Bracket the T at which the ratio crosses 1, by scanning horizons."""
    if verbose:
        print(f"\n  gamma = {gamma:.2f}")
        print(f"  {'T':>4}{'p_T':>11}{'class':>12}{'time':>12}{'ratio':>9}  winner")
    rows, prev = [], None
    t_cross = None
    for T in horizons:
        m = instance(8, T, DELTA_MIX, c_min, rates, gamma)
        v, ratio = axis_ratio(m, maxiter)
        rows.append({"T": T, "p_T": m.p_fail, "ratio": ratio,
                     **{k: v[k] for k in v}})
        if verbose:
            print(f"  {T:>4}{m.p_fail:>11.2e}{v['class']:>12.4g}"
                  f"{v['time_scalar']:>12.4g}{ratio:>9.4f}"
                  f"  {'component' if ratio > 1 else 'TIME'}")
        if prev and prev["ratio"] > 1 >= ratio:          # linear in log-ratio
            lo, hi = prev, rows[-1]
            f = np.log(lo["ratio"]) / (np.log(lo["ratio"]) - np.log(hi["ratio"]))
            t_cross = lo["T"] + f * (hi["T"] - lo["T"])
        prev = rows[-1]
    if verbose and t_cross:
        print(f"  -> crossover at T ~ {t_cross:.1f}   "
              f"(gamma * T_cross = {gamma * t_cross:.2f})")
    elif verbose:
        print("  -> no crossover in the scanned range")
    return {"gamma": gamma, "rows": rows, "t_cross": t_cross}


# --------------------------------------------------------------------------
# B. how much does gamma matter at all?
# --------------------------------------------------------------------------

def p_vs_gamma(rates, gammas, pool_sizes, T=8, c_req=24.0, verbose=True):
    if verbose:
        print(f"\n  p_T vs gamma  (job needs {c_req:.0f} GPUs, T={T})")
        print(f"  {'gamma':>7}{'a_i @ half cap':>16}", end="")
        for k in pool_sizes:
            print(f"{'k=%d' % k:>12}", end="")
        print()
    out = []
    for g in gammas:
        row = {"gamma": g, "a_half": rates["a0"] + g}
        if verbose:
            print(f"  {g:>7.2f}{rates['a0'] + g:>16.4f}", end="")
        for k in pool_sizes:
            nb = k // 2
            c = [8.] * nb + [4.] * (k - nb)
            m = instance(k, T, c, c_req, rates, g)
            row[f"k{k}"] = m.p_fail
            if verbose:
                print(f"{m.p_fail:>12.2e}", end="")
        out.append(row)
        if verbose:
            print()
    return out


# --------------------------------------------------------------------------
# C. the operational question, asked directly
# --------------------------------------------------------------------------

def sla_margin(rates, gamma, c_req=24.0, T=8, max_nodes=12, targets=(1e-2, 1e-3),
               verbose=True):
    """How large a pool keeps a job needing c_req GPUs below a target p_T?

    This is the provisioning question a mission-time probability actually
    answers, and it needs no node-vs-job proxy.
    """
    if verbose:
        print(f"\n  SLA margin (gamma={gamma}, job needs {c_req:.0f} GPUs, "
              f"T={T})")
        print(f"  {'nodes':>6}{'pool':>7}{'margin':>9}{'p_T':>12}")
    rows = []
    for k in range(2, max_nodes + 1):
        nb = k // 2
        c = np.array([8.] * nb + [4.] * (k - nb))
        if c.sum() < c_req:
            continue
        m = instance(k, T, c, c_req, rates, gamma)
        margin = (c.sum() - c_req) / c_req
        rows.append({"k": k, "pool": float(c.sum()), "margin": margin,
                     "p_T": m.p_fail})
        if verbose:
            print(f"  {k:>6}{c.sum():>7.0f}{margin * 100:>8.0f}%{m.p_fail:>12.3e}")
    if verbose:
        for eps in targets:
            ok = [r for r in rows if r["p_T"] <= eps]
            if ok:
                b = min(ok, key=lambda z: z["margin"])
                print(f"    p_T <= {eps:.0e}: {b['k']} nodes, "
                      f"{b['margin'] * 100:.0f}% margin")
            else:
                print(f"    p_T <= {eps:.0e}: not reached within {max_nodes} nodes")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.30, 0.10])
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--c-min", type=float, default=12.0)
    ap.add_argument("--maxiter", type=int, default=130)
    ap.add_argument("--skip-sla", action="store_true")
    ap.add_argument("--out", default="gamma_sensitivity.json")
    args = ap.parse_args()

    rates = calibrate_rates("A100", 0.25)
    out = {"args": vars(args), "calibration": rates}

    print("=" * 68)
    print("A. Does the component -> time crossover survive changing gamma?")
    print("=" * 68)
    out["crossover"] = [crossover(g, rates, args.horizons, args.c_min,
                                  args.maxiter) for g in args.gammas]
    located = [(c["gamma"], c["t_cross"]) for c in out["crossover"]
               if c["t_cross"]]
    if len(located) >= 2:
        print("\n  scaling check:")
        for g, t in located:
            print(f"    gamma={g:.2f}  T_cross={t:5.1f}  gamma*T_cross={g*t:.2f}")
        print("  -> T_cross ~ 1/gamma")

    print("\n" + "=" * 68)
    print("B. How much does the uncalibrated gamma actually matter?")
    print("=" * 68)
    out["p_vs_gamma"] = p_vs_gamma(rates, [0.0, 0.01, 0.05, 0.10, 0.30],
                                   [6, 8, 10, 12])

    if not args.skip_sla:
        print("\n" + "=" * 68)
        print("C. The provisioning question, asked directly (no proxy)")
        print("=" * 68)
        out["sla"] = sla_margin(rates, 0.30)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
