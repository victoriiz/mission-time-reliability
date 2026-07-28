"""
Diagnostic for the tilt-ordering reversal.
Goes in: framework/dynamic_model/diagnose_tilt.py

THE OBSERVATION. On the reference instance (c = [8,8,8,8,4,4,4,4], C_min = 12):

  time-homogeneous optimum   lambda_8GPU = 2.298  >  lambda_4GPU = 1.475
  time-varying optimum, t=0  lambda_8GPU = 9.982  <  lambda_4GPU = 31.043

Which node class the optimal proposal prefers to drive REVERSES depending on
whether the proposal is allowed to depend on the clock. That is either a real
consequence of the failure geometry or an optimiser artefact, and a poster
cannot carry it unexplained.

THE HYPOTHESIS. The failure set F = {C(x) < C_min} is upward-closed in the set
of failed nodes, so it is characterised by its MINIMAL CUTS: minimal sets of
nodes whose simultaneous failure lands the system in F. At C_min = 12 of
C_nom = 48 the cheapest cut is all four 8-GPU nodes plus two of the four 4-GPU
nodes. The 8-GPU nodes are therefore MANDATORY (4 of 4 required) while the
4-GPU nodes are PARTIALLY required (2 of 4, with 6 ways to choose them). If the
reversal is driven by this geometry, then changing C_min so that the cut
composition changes should move the tilt ordering with it.

THE TEST. Sweep C_min, and for each value report
  (a) the minimal cuts and the per-class requirement fraction,
  (b) the time-homogeneous class tilts,
  (c) the step-0 time-varying class tilts,
and check whether the ordering in (b) and (c) tracks (a).

CONTROL. Repeat with homogeneous capacities (all nodes equal, same C_nom). The
classes collapse to one, so any reversal must vanish. If it does not, it is an
optimiser artefact.
"""

import numpy as np

from model import DynamicConfig, DynamicModel
from decompose import rung_ceiling, _embed, _n_classes, class_map
from proposals import rate_matrices


# --------------------------------------------------------------------------
# failure-set geometry
# --------------------------------------------------------------------------

def minimal_cuts(model):
    """Minimal-by-inclusion sets of failed components that land the system in F.

    F is upward-closed in the failed set: failing more components only lowers
    capacity, so it cannot take you out of F. A state in F is therefore a
    minimal cut iff repairing ANY single failed component takes you out of F.
    """
    cuts = []
    for xi in range(model.n_states):
        if model.in_F[xi] != 1:
            continue
        x = model.states[xi].copy()
        failed = np.flatnonzero(x == 0)
        if failed.size == 0:
            continue
        minimal = True
        for j in failed:                       # repair one component
            y = x.copy(); y[j] = 1
            if (y @ model.cfg.c) < model.cfg.c_min:
                minimal = False                # still failing => not minimal
                break
        if minimal:
            cuts.append(frozenset(failed.tolist()))
    return cuts


def class_requirement(model, cuts):
    """Per class: mean fraction of that class's components required by a cut."""
    inv, K = class_map(model)
    sizes = np.array([(inv == k).sum() for k in range(K)], dtype=float)
    if not cuts:
        return np.zeros(K), sizes
    counts = np.zeros(K)
    for cut in cuts:
        for k in range(K):
            counts[k] += sum(1 for j in cut if inv[j] == k)
    return (counts / len(cuts)) / sizes, sizes


# --------------------------------------------------------------------------
# tilts
# --------------------------------------------------------------------------

def tilts_for(model, maxiter=150):
    """Return (homogeneous class tilts, step-0 time-varying class tilts)."""
    K = _n_classes(model)
    solved = {}
    for rung in ("scalar", "class", "time_scalar", "time_class"):
        extra = [e for e in (_embed(solved[s], s, rung, model.cfg.N, model.T, K)
                             for s in solved) if e is not None]
        r = rung_ceiling(model, rung, extra_starts=extra, n_restarts=0,
                         maxiter=maxiter)
        solved[rung] = r["theta"]
    lam_hom = np.exp(solved["class"])
    lam_tv = np.exp(np.asarray(solved["time_class"]).reshape(model.T, K))
    return lam_hom, lam_tv[0], solved


def build(N, T, c, c_min, a0, b0, gamma=0.30, eta=0.50, name="diag"):
    return DynamicModel(DynamicConfig(
        N=N, T=T, c=np.asarray(c, float), c_min=float(c_min),
        a0=np.full(N, a0), gamma=np.full(N, gamma),
        b0=np.full(N, b0), eta=np.full(N, eta), name=name))


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

def sweep_cmin(c, a0, b0, T=8, cmins=(12, 20, 28, 36, 40), maxiter=150,
               verbose=True):
    """Does the tilt ordering track the minimal-cut composition?"""
    N = len(c)
    rows = []
    if verbose:
        print(f"{'C_min':>6}{'p_T':>11}{'|F|':>6}{'#cuts':>7}"
              f"{'req_big':>9}{'req_sml':>9}"
              f"{'hom_big':>9}{'hom_sml':>9}{'tv0_big':>9}{'tv0_sml':>9}"
              f"{'  hom':>7}{'  tv0':>7}")
    for cm in cmins:
        m = build(N, T, c, cm, a0, b0)
        if m.in_F[m.start_idx] == 1 or m.in_F.sum() == 0:
            continue
        cuts = minimal_cuts(m)
        req, _ = class_requirement(m, cuts)
        hom, tv0, _ = tilts_for(m, maxiter=maxiter)
        # class 0 is the smaller capacity value under np.unique ordering;
        # identify explicitly by capacity so the labels cannot silently swap
        inv, K = class_map(m)
        caps = np.array([m.cfg.c[inv == k][0] for k in range(K)])
        big = int(np.argmax(caps)); sml = int(np.argmin(caps))
        row = dict(c_min=cm, p_T=m.p_fail, nF=int(m.in_F.sum()),
                   n_cuts=len(cuts),
                   req_big=req[big], req_sml=req[sml],
                   hom_big=hom[big], hom_sml=hom[sml],
                   tv0_big=tv0[big], tv0_sml=tv0[sml])
        row["hom_pref"] = "big" if hom[big] > hom[sml] else "small"
        row["tv0_pref"] = "big" if tv0[big] > tv0[sml] else "small"
        rows.append(row)
        if verbose:
            print(f"{cm:>6}{m.p_fail:>11.2e}{row['nF']:>6}{len(cuts):>7}"
                  f"{req[big]:>9.3f}{req[sml]:>9.3f}"
                  f"{hom[big]:>9.3f}{hom[sml]:>9.3f}"
                  f"{tv0[big]:>9.3f}{tv0[sml]:>9.3f}"
                  f"{row['hom_pref']:>7}{row['tv0_pref']:>7}")
    return rows


def control_homogeneous(a0, b0, N=8, T=8, maxiter=150):
    """Control: equal capacities collapse the classes; no reversal is possible."""
    c = np.full(N, 6.0)                       # same C_nom = 48
    m = build(N, T, c, 12.0, a0, b0, name="control-homogeneous")
    K = _n_classes(m)
    print(f"  control: homogeneous c, classes = {K}, p_T = {m.p_fail:.3e}")
    if K == 1:
        print("  -> single class, tilt ordering undefined, as expected")
    return m


if __name__ == "__main__":
    from delta_calibration import calibrate_rates
    r = calibrate_rates("A100", 0.25)
    c = [8., 8., 8., 8., 4., 4., 4., 4.]
    print("Does the tilt ordering track the minimal-cut composition?\n")
    sweep_cmin(c, r["a0"], r["b0"])
    print()
    control_homogeneous(r["a0"], r["b0"])
