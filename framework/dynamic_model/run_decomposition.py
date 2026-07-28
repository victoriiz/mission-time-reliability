"""
Headline experiment: decomposing the optimality gap into TIME and COMPONENT.
Goes in: framework/dynamic_model/run_decomposition.py

  python run_decomposition.py --restarts 2 --maxiter 400 --out decomposition.json

run_ceiling_ladder.py runs the time-HOMOGENEOUS ladder
(scalar / per_component / per_comp_repair) and does not touch decompose.py, so
it cannot produce the reported finding. This driver does.

Produces, in the order the write-up reads them:

  0. correctness   recursion vs brute force; time-varying vs homogeneous;
                   DP vs naive simulation; double vs extended precision
  1. gate          h-transform audits to bias ~0, ESS ~1
  2. decomposition scalar / class / time_scalar / time_class at the reference
                   horizon, with monotonicity and symmetry invariants
  3. sweep         the same rungs across mission lengths -- the headline
  4. payoff        p_T -> r_f -> A -> N_prod

IMPORTANT: keep this file next to model.py. Every module here uses flat
imports, so the driver must live in framework/dynamic_model/, not in a
sibling experiments/ directory.
"""

import argparse
import json
import time

import numpy as np

from model import DynamicConfig, DynamicModel
from proposals import build_kernel, rate_matrices, h_transform_sequence
from ceiling import exact_traj_variance, brute_force_variance
from audit import audit_trajectory
from decompose import (RUNGS, RUNG_ORDER, exact_traj_variance_tv,
                       rung_ceiling, _embed, _n_classes, class_map)
from delta_calibration import calibrate_rates, capacity_plan


# ----------------------------------------------------------------------
# correctness gates
# ----------------------------------------------------------------------

def gate_recursions(seed=0):
    """Recursion vs brute force, and time-varying vs time-homogeneous."""
    cfg = DynamicConfig(N=3, T=3, c=np.array([2., 1., 1.]), c_min=2.0,
                        a0=np.full(3, 0.05), gamma=np.full(3, 0.3),
                        b0=np.full(3, 0.3), eta=np.full(3, 0.5), name="gate")
    m = DynamicModel(cfg)
    rates = rate_matrices(m)
    rng = np.random.default_rng(seed)
    worst_bf = worst_tv = 0.0
    for _ in range(5):
        Q = build_kernel(m, np.exp(rng.normal(0, 0.8, 3)), rates=rates)
        v = exact_traj_variance(m, Q)
        worst_bf = max(worst_bf, abs(v - brute_force_variance(m, Q)) / abs(v))
        worst_tv = max(worst_tv,
                       abs(v - exact_traj_variance_tv(m, [Q] * m.T)) / abs(v))
    return {"brute_force_rel_err": worst_bf, "time_varying_rel_err": worst_tv,
            "passed": worst_bf < 1e-9 and worst_tv < 1e-12}


def gate_precision(m, rung, theta):
    """Re-evaluate a ceiling in extended precision to rule out cancellation.

    Var = M_T(start) - p_T^2, and at short horizons p_T can be ~1e-8, so the
    subtraction is worth checking. Returns the relative disagreement between
    float64 and longdouble.
    """
    build, _ = RUNGS[rung]
    Qs = build(m, theta, rate_matrices(m))
    P, in_F, T = m.Pmat, m.in_F, m.T
    fail, safe = in_F == 1, in_F == 0
    out = {}
    for name, dt in (("f64", np.float64), ("f128", np.longdouble)):
        Pd = P.astype(dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            Rs = [np.where(Q.astype(dt) > 0, Pd * Pd / Q.astype(dt), dt(0))
                  for Q in Qs]
        M = np.zeros(m.n_states, dtype=dt)
        for s in range(1, T + 1):
            t = T - s
            M = Rs[t][:, fail].sum(axis=1) + Rs[t][:, safe] @ M[safe]
            M[fail] = 0
        out[name] = float(M[m.start_idx] - dt(m.p_fail) ** 2)
    return abs(out["f64"] - out["f128"]) / abs(out["f128"])


# ----------------------------------------------------------------------
# the decomposition
# ----------------------------------------------------------------------

def decomposition(m, restarts, maxiter, verbose=True):
    K = _n_classes(m)
    solved, rows = {}, {}
    for rung in RUNG_ORDER:
        extra = [e for e in (_embed(solved[s], s, rung, m.cfg.N, m.T, K)
                             for s in solved) if e is not None]
        r = rung_ceiling(m, rung, extra_starts=extra, n_restarts=restarts,
                         maxiter=maxiter)
        solved[rung] = r["theta"]
        rows[rung] = {"n_params": r["n_params"], "ceiling_vrf": r["vrf"],
                      "ceiling_var": r["var"], "theta": r["theta"].tolist()}
        if verbose:
            print(f"    {rung:<13} params={r['n_params']:>3}  "
                  f"ceiling={r['vrf']:>12.4f}")

    v = {k: rows[k]["ceiling_vrf"] for k in RUNG_ORDER}
    monotone = all(v[b] >= v[a] - 1e-9
                   for a, b in zip(RUNG_ORDER, RUNG_ORDER[1:]))
    base = v["scalar"]
    return {
        "rungs": rows, "monotone": monotone,
        "gain_component": v["class"] / base,
        "gain_time": v["time_scalar"] / base,
        "gain_both": v["time_class"] / base,
        "interaction": v["time_class"] * base / (v["class"] * v["time_scalar"]),
        "winner": "component" if v["class"] > v["time_scalar"] else "time",
        "n_classes": K,
    }


def build_instance(N, T, c, c_min, rates, gamma, eta, name):
    return DynamicModel(DynamicConfig(
        N=N, T=T, c=np.asarray(c, float), c_min=float(c_min),
        a0=np.full(N, rates["a0"]), gamma=np.full(N, gamma),
        b0=np.full(N, rates["b0"]), eta=np.full(N, eta), name=name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-type", default="A100")
    ap.add_argument("--step-hours", type=float, default=0.25)
    ap.add_argument("--c-min", type=float, default=12.0)
    ap.add_argument("--ref-T", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--gamma", type=float, default=0.30)
    ap.add_argument("--eta", type=float, default=0.50)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sla-gpus", type=int, default=256)
    ap.add_argument("--out", default="decomposition.json")
    args = ap.parse_args()

    rates = calibrate_rates(args.node_type, args.step_hours)
    c = [8., 8., 8., 8., 4., 4., 4., 4.]      # Delta A100 partition node mix
    out = {"args": vars(args), "calibration": rates}

    print("[0] correctness gates ...")
    g = gate_recursions()
    out["gates"] = g
    print(f"    brute force {g['brute_force_rel_err']:.2e}   "
          f"time-varying {g['time_varying_rel_err']:.2e}   "
          f"{'PASS' if g['passed'] else 'FAIL'}")
    if not g["passed"]:
        raise SystemExit("recursion gates FAILED -- do not report ceilings")

    print(f"[1] reference instance, T={args.ref_T} ...")
    m = build_instance(8, args.ref_T, c, args.c_min, rates, args.gamma,
                       args.eta, f"delta-{args.node_type}-T{args.ref_T}")
    val = m.validate()
    out["instance"] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                           else v) for k, v in val.items()}
    print(f"    p_T={m.p_fail:.4e}  |F|={int(m.in_F.sum())}/{m.n_states}  "
          f"rare={val['is_rare']}  concentrated={val['is_concentrated']}  "
          f"DP=sim {val['dp_matches_sim']}")

    print("[2] h-transform gate (expect bias ~0, ESS 1.0) ...")
    a = audit_trajectory(m, Q_seq=h_transform_sequence(m), n_paths=2000,
                         n_trials=args.trials)
    out["h_transform_audit"] = a
    print(f"    bias {a['bias']:+.3e}   ESS {a['ess']:.4f}")

    print(f"[3] decomposition at T={args.ref_T} ...")
    d = decomposition(m, args.restarts, args.maxiter)
    out["decomposition"] = d
    print(f"    monotone={d['monotone']}  component x{d['gain_component']:.3f}  "
          f"time x{d['gain_time']:.3f}  both x{d['gain_both']:.3f}  "
          f"interaction x{d['interaction']:.3f}")

    print("[4] horizon sweep (the headline) ...")
    sweep = []
    for T in args.horizons:
        mt = build_instance(8, T, c, args.c_min, rates, args.gamma, args.eta,
                            f"T{T}")
        t0 = time.time()
        dt_ = decomposition(mt, args.restarts, args.maxiter, verbose=False)
        prec = gate_precision(mt, "class", np.array(
            dt_["rungs"]["class"]["theta"]))
        row = {"T": T, "job_hours": T * args.step_hours, "p_T": mt.p_fail,
               "scalar": dt_["rungs"]["scalar"]["ceiling_vrf"],
               "class": dt_["rungs"]["class"]["ceiling_vrf"],
               "time_scalar": dt_["rungs"]["time_scalar"]["ceiling_vrf"],
               "time_class": dt_["rungs"]["time_class"]["ceiling_vrf"],
               "winner": dt_["winner"], "monotone": dt_["monotone"],
               "precision_rel_err": prec, "seconds": time.time() - t0}
        sweep.append(row)
        print(f"    T={T:>2}  p_T={mt.p_fail:.2e}  scalar={row['scalar']:>10.3f}"
              f"  class={row['class']:>10.3f}  time={row['time_scalar']:>10.3f}"
              f"  -> {row['winner']:<9} (f64/f128 {prec:.1e})")
    out["horizon_sweep"] = sweep

    print("[5] capacity-planning payoff ...")
    plan = capacity_plan(m.p_fail, m.T, args.step_hours,
                         rates["mttr_hours"], args.sla_gpus, int(c[0]))
    out["capacity_plan"] = plan.__dict__
    print(f"    p_T={plan.p_T:.4e} -> r_f={plan.r_f:.4e}/h -> A={plan.A:.5f} "
          f"-> N_prod={plan.n_prod} ({plan.overprovision*100:.2f}% margin)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
