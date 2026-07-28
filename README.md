# Mission-Time Rare-Event Estimation with Exact Proposal Ceilings

Exact variance ceilings for importance-sampling proposals on finite-horizon
Markov models of degrading systems, calibrated to Delta/DeltaAI GPU-cluster
operational data.

**The question this answers.** Practitioners report achieved variance-reduction
factors with no reference: a 400x speedup may be near-optimal or may be leaving
four orders of magnitude on the table. This code computes, exactly and without
sampling, the best variance reduction an entire *family* of proposals can ever
achieve. It is a measuring instrument, not an estimator — it never returns an
estimate of the failure probability.

## Layout

```
framework/dynamic_model/
  model.py               exact ground truth: backward DP, h-transform, validation gates
  proposals.py           odds-tilt proposal families + the h-transform sequence
  ceiling.py             exact second-moment recursion; family ceilings; brute-force check
  decompose.py           time-varying recursion; the time-vs-component decomposition
  audit.py               trajectory sampling audit with the stop-at-F rule
  delta_calibration.py   Delta/DeltaAI parameters + the capacity-planning payoff
  diagnose_tilt.py       minimal-cut geometry; the tilt-ordering diagnostic
  run_decomposition.py   HEADLINE driver: decomposition + horizon sweep
  run_ceiling_ladder.py  time-homogeneous ladder only (scalar/per_component/+repair)
```

```bash
pip install -r requirements.txt
cd framework/dynamic_model
python model.py                                   # validation gates
python run_decomposition.py --restarts 2          # headline: decomposition + sweep
python run_ceiling_ladder.py --restarts 6         # time-homogeneous ladder only
```

`run_ceiling_ladder.py` does **not** import `decompose` and therefore cannot
produce the time-vs-component result or the horizon sweep. Use
`run_decomposition.py` for anything reported as the headline finding.

## What is computed

For any Markov proposal `Q`, the single-sample variance of the mission-time IS
estimator obeys a backward second-moment recursion, stopping at first entry to
the failure set `F`:

```
R(x,y) = P(x,y)^2 / Q(x,y)
M_0(x) = 0
M_s(x) = sum_{y in F} R(x,y) + sum_{y not in F} R(x,y) M_{s-1}(y)
Var    = M_T(start) - p_T^2
```

Optimising this over a family's parameters gives the family's ceiling.

**Mission failure is a HITTING event, so reweighting stops at first entry to F.**
Continuing past the hit biases the estimator even for a provably zero-variance
proposal; in development this produced -11.5% bias on the h-transform. The exact
ceiling cannot detect that error. Only the sampling audit can. Keep both.

## Correctness gates

Every one of these runs in the driver and must pass before results are reported:

| gate | expected |
|---|---|
| recursion vs brute-force path enumeration | agreement to ~1e-16 |
| time-varying recursion vs time-homogeneous | exact |
| backward DP vs naive simulation | within 3 sigma |
| h-transform audit | bias +0.0000%, ESS 1.0000 |
| double vs extended precision | agreement to ~1e-15 |
| ladder monotonicity (nested families) | non-decreasing ceilings |
| symmetry (exchangeable components) | equal tilts |

The last two are not decoration. A naive Nelder-Mead run on the 16-parameter
family returned a ceiling *below* the 8-parameter family that contains it —
impossible, and caught only by the monotonicity invariant. Warm-starting each
family from every smaller solved family fixes it by construction.

## Headline results

Instance: N=8 Delta nodes, `c = [8,8,8,8,4,4,4,4]`, `C_min = 12`, 15-minute
steps, calibrated to A100 MTTR 0.88 h and 0.60% node unavailability.

Optimality gap at a two-hour mission (`p_T = 6.7406e-3`), from
`run_decomposition.py`:

| family | params | time | component | exact ceiling |
|---|---|---|---|---|
| scalar | 1 | no | no | 2.6634 |
| class | 2 | no | yes | 3.4548 |
| time_scalar | 8 | yes | no | 6.0419 |
| time_class | 16 | yes | yes | 8.8101 |
| h-transform | — | yes | full joint state | infinite |

The time-homogeneous ladder from `run_ceiling_ladder.py` (`src_results.json`) is
a separate, coarser cut of the same instance: `per_component` reproduces `class`
at 3.4548 exactly, confirming that the symmetry reduction is lossless, and
`per_comp_repair` reaches **>= 4.479** at 6 restarts. That last figure is a
LOWER BOUND, not a ceiling: it still fails the symmetry invariant
(`sym_violation 0.077`), so the optimiser has not converged. Do not quote it as
a ceiling without that caveat.

Collapse with mission length (system fixed, only `T` varies):

| T | job | p_T | scalar | class | time_scalar | winner |
|---|---|---|---|---|---|---|
| 2 | 0.5 h | 7.90e-8 | 5.04e4 | **2.02e6** | 3.08e5 | component, 6.6x |
| 3 | 0.75 h | 1.97e-5 | 146.3 | **1325** | 1121 | component, 1.18x |
| 4 | 1.0 h | 2.09e-4 | 17.53 | 62.56 | **133.9** | time, 2.14x |
| 6 | 1.5 h | 2.08e-3 | 4.45 | 7.35 | **15.70** | time, 2.14x |
| 8 | 2.0 h | 6.74e-3 | 2.66 | 3.46 | **6.04** | time, 1.75x |

Every product-form ceiling collapses monotonically with mission length, and the
two knowledge axes exchange dominance at a crossover between T=3 and T=4.
Production HPC jobs run for hours, i.e. squarely in the regime where
product-form proposals fail.

All rows were re-evaluated in extended precision (`gate_precision` in
`run_decomposition.py`); float64 and longdouble agree to between 2e-16 and
2e-15, so the short-horizon figures are not cancellation artefacts despite
`p_T` reaching 7.9e-8.

The time/component **interaction is also horizon-dependent**, so do not
generalise it: at T=8 the two axes compose super-multiplicatively (predicted
1.297 x 2.268 = 2.942, measured 3.308, +12%), but at T=4 they compose
sub-multiplicatively (measured 23.99 against a predicted 27.24, -12%).

## Open: the tilt-ordering reversal

The time-homogeneous optimum tilts the 8-GPU nodes harder (2.298 vs 1.475); the
time-varying optimum at step 0 tilts the 4-GPU nodes harder (31.04 vs 9.98).

`diagnose_tilt.py` tests the natural hypothesis — that the ordering tracks the
composition of the failure set's minimal cuts — and **refutes it**. At
`C_min = 36` the requirement fractions predict the small nodes should be tilted
harder (0.419 vs 0.290), but the homogeneous optimum still prefers the large
nodes (2.499 vs 1.643), the same direction as at `C_min = 12`.

What the test *does* establish: the time-homogeneous preference for large nodes
is robust across failure-set geometries, consistent with a capacity-per-unit-
likelihood-ratio-cost argument (failing an 8-GPU node destroys twice the
capacity for the same tilt cost). The step-0 reversal under a time-varying
proposal remains unexplained and should not be interpreted on a poster.

## Caveats

- `O(2^N)` throughout; the dense transition matrix is `O(2^{2N})`, putting the
  practical wall near `N ~ 12`, not `N ~ 20`. Toy scale is deliberate — it is
  what permits exact validation and what precludes scaling claims.
- Parameters are **calibrated to** published Delta summary statistics
  (Cui et al., SC '25, arXiv:2503.11901), **not validated against** telemetry.
- The horizon sweep varies `T` and `p_T` together; holding both fixed is
  impossible for a fixed system.
- The capacity-planning chain uses **node** availability while `F` is defined at
  **job** level. State the proxy explicitly or redefine `F`.

## Reproducing the reported numbers

```bash
cd framework/dynamic_model
python run_decomposition.py  --restarts 2 --out decomposition.json   # headline
python run_ceiling_ladder.py --restarts 6 --out src_results.json     # homogeneous ladder
```

Both JSON files are committed. Every table above is a direct read of one of
them; nothing is transcribed by hand. (An earlier draft of this README quoted
`p_T = 2.35e-6` at T=3, a hand-typed value that had never been computed — the
correct figure, 1.97e-5, comes from `run_decomposition.py`. Read numbers out of
the JSON, not out of memory.)

## Not included

`estimators.py` from the original repository is static-model code: its `audit`,
`ideal_proposal` and `tilt_family_ceiling` read `model.p`, `model.failmask` and
`model.component_bits`, which `DynamicModel` does not define. `audit.py` and
`ceiling.py` are the dynamic replacements. The quantum experiments are
deliberately excluded.

## Next

1. Implement AMS and cross-entropy for the trajectory model and price them
   against the same ceilings — currently there is no approximate classical
   opponent.
2. Learn `h_t` by temporal-difference methods on instances where exact `h_t` is
   available from DP, so learning error is measurable exactly.
3. Determine whether bounded-relative-variance guarantees for LSTD rare-event
   prediction survive absorbing, state-dependent, irreversible dynamics.
