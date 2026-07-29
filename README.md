# Exact Variance Ceilings for Mission-Time Rare-Event Estimation

**Victoria Zhang** · Phuong Cao (advisor) · R. Srikant (advisor)
University of Illinois Urbana–Champaign · National Center for Supercomputing Applications
---

## What problem is this?

A large GPU job fails if too many of its nodes go down before it finishes. That's
rare — about 1 in 148 missions on our instance — so measuring it by simulation
needs roughly **1.5 million simulated missions**.

The standard fix is **importance sampling**: deliberately simulate a distorted
world where failures are common, then correct each run with a likelihood-ratio
weight. It works. The unsolved practical problem is different:

> Someone reports "our proposal achieved 400× variance reduction."
> **Is that good?** Maybe the best possible was 420× and they nearly nailed it.
> Maybe it was 10⁶× and they left four orders of magnitude on the table.
> There is no reference.

**This repository computes that missing reference.** For any *family* of
proposals it returns the best variance reduction any member could ever achieve —
exactly, in closed form, without drawing a single sample. It is a measuring
instrument, not an estimator: it never returns a probability.

---

## The three findings

**1. Product-form proposal quality collapses with mission length.**
The most expressive product-form family falls from $2.7\times10^{6}$ to $8.8$ as
missions go from 30 minutes to 2 hours — five orders of magnitude. Short missions
have essentially *one* dominant failure path that a fixed tilt can target; long
missions have many, and no time-homogeneous proposal can cover them. **Production
HPC jobs run for hours, which is exactly the regime where these methods fail.**

**2. What you need to know flips, and the cascade sets when.**
The optimal proposal knows two things: *which* node to drive and *when* to drive
it. Their relative value exchanges at a crossover, and its location scales with
the overload-cascade strength γ:

| γ | crossover T | job length | γ · T_cross |
|---|---|---|---|
| 0.30 | ≈ 3.2 | 0.8 h | 0.95 |
| 0.10 | ≈ 11 | 2.75 h | 1.10 |
| 0.05 | > 14 (predicted ≈ 21) | > 3.5 h | — |

So **T_cross ≈ 1/γ**. The ordering at any single operating point is *not* a law
— but the scaling is, and the mechanism is that the cascade is what makes timing
matter.

**3. The one uncalibrated parameter dominates the operational answer.**
Every rate is calibrated to published Delta figures except γ, which was chosen.
It swings the failure probability over **eleven orders of magnitude**
(8.75e-14 to 5.66e-3 for a 12-node pool). We therefore report sensitivity, not a
point estimate. Calibrating γ against failure-rate-by-scale data is the
highest-value next step.

---

## Run it yourself

```bash
pip install -r requirements.txt          # numpy, scipy
cd framework/dynamic_model
```

**Start here — 5 seconds.** Confirms the model is sound:

```bash
python3 model.py
```

Expect `p_fail: 0.00633466…`, with `rows_sum_to_1` and `dp_matches_sim` both
`True`. (`is_concentrated` is `False` here — this N=3 reference has 3 of 8 states
failing, and concentration only becomes meaningful at the N=8 working instance,
where it is 15 of 256. The gate is checked where it matters.) The second block is
the deliberately *easy* lumpable case, shown for contrast: identical parameters,
homogeneous capacities, and the state space collapses.

**Finding 1 and 2 — about 2 minutes.** The decomposition and horizon sweep:

```bash
python3 run_decomposition.py --restarts 2 --out decomposition.json
```

Watch for: the recursion matching brute-force enumeration to ~1e-16, the
h-transform auditing to bias +0.0000% and ESS 1.0000, then the four-rung ladder
and the collapse across horizons.

**Finding 2 and 3 — about 5 minutes.** Cascade-strength robustness:

```bash
python3 run_gamma_sensitivity.py --out gamma_sensitivity.json
```

Prints the crossover location per γ (with the γ·T_cross scaling check), the
eleven-order-of-magnitude γ sweep, and the provisioning question asked directly.

**Optional — the time-homogeneous ladder** (coarser cut, kept for comparison):

```bash
python3 run_ceiling_ladder.py --restarts 6 --out src_results.json
```

**Optional — figures** (regenerates the poster panels):

```bash
cd ../../poster && python3 make_figures.py && pdflatex poster.tex
```

> ⚠️ **Both drivers must sit beside `model.py`.** Every module uses flat imports,
> so moving them to a sibling `experiments/` directory fails with
> `ModuleNotFoundError: No module named 'model'`.

> ⚠️ **Python 3.14 may not have scipy wheels yet.** If `pip install` fails, use
> `python3.12 -m venv .venv && source .venv/bin/activate`.

---

## How it works

For any Markov proposal `Q`, with `R = P²/Q`, the estimator's second moment obeys
a backward recursion that **stops at first entry to the failure set F**:

```
M₀(x) = 0
Mₛ(x) = Σ_{y∈F} R(x,y) + Σ_{y∉F} R(x,y) Mₛ₋₁(y)
Var   = M_T(start) − p_T²
```

Optimising this over a family's parameters gives that family's **ceiling**.

**Mission failure is a hitting event, so reweighting stops the moment you enter
F.** Getting this wrong previously biased our provably zero-variance proposal by −11.5%.
The exact ceiling *cannot* detect that error, since it assumes the estimator is
implemented correctly. Only the sampling audit caught it.

**Four nested families**, chosen so the optimality gap decomposes:

| family | params | knows *when*? | knows *which node*? |
|---|---|---|---|
| scalar | 1 | no | no |
| + component | 2 | no | yes |
| + time | 8 | yes | no |
| + both | 16 | yes | yes |
| h-transform | — | yes | full joint state (optimal) |

Exchangeable nodes must receive equal tilts at any optimum, so parameterising by
class is **exact**, shrinking the top family from 64
parameters to 16.

---

## Correctness

Every gate runs automatically before any number is reported.

| gate | expected | why it matters |
|---|---|---|
| recursion vs brute-force path enumeration | ~1e-16 | the variance formula is right |
| time-varying vs time-homogeneous recursion | exact | the two code paths agree |
| backward DP vs naive simulation | within 3σ | the ground truth is right |
| h-transform audit | bias +0.0000%, ESS 1.0000 | DP + stopping rule + audit all right *simultaneously* |
| double vs extended precision | ~1e-15 | short-horizon results aren't cancellation artefacts |
| ladder monotonicity | non-decreasing | nested families can't get worse |
| symmetry | equal tilts within a class | the optimiser converged |

note that **naive Monte Carlo's *sampled* VRF is 1.074**,
though it should be exactly 1.000 (it is the baseline, compared to itself). That
7% error is the noise floor of a 20-trial estimate. It is why we report the exact
ceiling and use sampling only to check bias.

---

## Layout

```
framework/dynamic_model/
  model.py                    exact ground truth: backward DP, h-transform, gates
  proposals.py                odds-tilt families + the h-transform sequence
  ceiling.py                  exact second-moment recursion, family ceilings
  decompose.py                time-varying recursion, the time-vs-component split
  audit.py                    trajectory sampling audit (stop-at-F rule)
  delta_calibration.py        Delta/DeltaAI parameters, provisioning arithmetic
  diagnose_tilt.py            minimal-cut geometry, tilt-ordering diagnostic
  run_decomposition.py        DRIVER: decomposition + horizon sweep
  run_gamma_sensitivity.py    DRIVER: cascade robustness + crossover scaling
  run_ceiling_ladder.py       DRIVER: time-homogeneous ladder only
```

`model.py` keeps `_build_transition_matrix_naive` alongside the vectorised
version so the fast path stays checkable; they are bit-identical, and the
vectorised one makes N=12 build in ~1 s instead of minutes.

---

## Scope — what is and isn't claimed

**Not claimed.** The second-moment-as-value-function formulation is Dupuis &
Wang; the inadequacy of state-independent proposals is Glasserman & Kou.

**Claimed.** An exact instantiation on a finite-horizon **hitting-event** chain
(prior work goes asymptotic, or targets terminal-time events); a decomposition of
the optimality gap that is well posed *only* in finite horizon, since the
infinite-horizon committor literature has no clock; and a measured collapse plus
crossover scaling on a model calibrated to production HPC data.

This is a careful transfer and measurement, not a new method.

**Limitations.**

- Everything is O(2^N). The toy scale of this project is deliberate, as it is what lets us know the
  exact answer, and it is what precludes scaling claims.
- Parameters are *calibrated to* published summary statistics, **not validated
  against** telemetry. γ is not calibrated at all.
- The horizon sweep varies T and p_T together; holding both fixed is impossible
  for a fixed system.
- The provisioning identity uses *node* availability while F is *job*-level, so
  that one derived figure uses job-mission failure as an explicit proxy. Nothing
  else in the project depends on it. `run_gamma_sensitivity.py` experiment C asks
  the question directly instead, with no proxy.

**Open.** The time-homogeneous optimum drives 8-GPU nodes harder (2.298 vs
1.475); the time-varying optimum at step 0 drives 4-GPU nodes harder (31.04 vs
9.98). `diagnose_tilt.py` refutes the natural explanation that it tracks the
failure set's minimal cuts, but this is largely unexplained.

---

## Next

1. Implement AMS and cross-entropy for the trajectory model and price them
   against the same ceilings. There is currently no approximate classical
   opponent.
2. Learn h_t by temporal-difference methods on instances where exact h_t is known
   from DP, so learning error is measurable *exactly* — a validation almost
   nobody in that literature can perform.
3. Calibrate γ against failure-rate-by-scale data so the operational conclusions
   stop being sensitivity ranges.

## Key References

1. S. Cui et al. *Story of Two GPUs: Characterizing the Resilience of Hopper H100
   and Ampere A100 GPUs.* SC '25. arXiv:2503.11901 — source of every Delta figure.
2. J. Blanchet, H. Lam. *State-dependent importance sampling for rare-event
   simulation.* Surv. Oper. Res. Manag. Sci., 2011 — covers Dupuis–Wang.
3. P. Glasserman, S.-G. Kou. *Analysis of an importance sampling estimator for
   tandem queues.* ACM TOMACS, 1995.
4. F. Cérou, A. Guyader. *Adaptive multilevel splitting.* Stoch. Anal. Appl., 2007.
5. R. Rubinstein, D. Kroese. *The Cross-Entropy Method.* Springer, 2004.
6. *The surprising efficiency of TD learning for rare event prediction.*
   NeurIPS 2024. arXiv:2405.17638 — infinite-horizon; motivates the next phase.
7. *A unified perspective on exponential tilt and bridge algorithms.*
   arXiv:2307.12597 — closest prior work; terminal-time, not hitting.
8. G. Chennetier et al. *Adaptive IS based on fault tree analysis.* SIAM/ASA JUQ.
   arXiv:2210.16185.

## Acknowledgements

NSF Award No. 2620473 · CRA/NSF DREU · NCSA Delta.