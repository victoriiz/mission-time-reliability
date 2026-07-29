# Attainable Variance Reduction for Mission-Time GPU Cluster Reliability

**Victoria Zhang** · Phuong Cao (advisor) · R. Srikant (advisor)
University of Illinois Urbana–Champaign · National Center for Supercomputing Applications

---

## Problem

Provisioning a GPU cluster requires the probability that a job survives its
mission: that no more than a tolerable number of its nodes fail before it
finishes. Steady-state availability cannot supply this, since it does not
distinguish a 20-minute job from a 20-hour one.

The quantity is rare. On the instance studied here it is about 1 in 148
missions, so naive Monte Carlo needs roughly 1.5 million simulated missions for
1% relative error. Importance sampling removes that cost by simulating a
distorted chain and correcting with likelihood-ratio weights.

Importance sampling introduces a different difficulty. A reported "400×
variance reduction" cannot be interpreted: the best attainable for that family
may have been 420×, or 10⁶×. No reference exists.

**This repository computes that reference.** For any family of proposals it
returns, in closed form and without drawing a sample, the best variance
reduction any member can attain. It is a measuring instrument rather than an
estimator, and never returns a probability.

---

## Findings

### 1. Most of the apparent collapse with mission length is the rarity scale

Raw ceilings fall from 2.7e6 to 8.8 between a 30-minute and a 2-hour mission.
Most of that is not degradation. For any sampler with bounded relative error the
variance scales as `c·p²`, so `VRF ~ 1/(c·p)`, which makes `1/p_T` the natural
scale of any VRF. As the horizon grows the event becomes less rare and `1/p_T`
falls five orders of magnitude on its own. Normalising it out:

| T | scalar | component | time | both |
|---|---|---|---|---|
| 2 | 0.0040 | 0.160 | 0.0243 | 0.216 |
| 3 | 0.0029 | 0.0261 | 0.0221 | 0.112 |
| 4 | 0.0037 | 0.0131 | 0.0280 | 0.0879 |
| 6 | 0.0093 | 0.0153 | 0.0327 | 0.0611 |
| 8 | 0.0180 | 0.0233 | 0.0407 | 0.0594 |

Relative efficiency stays within one order of magnitude, and for the scalar and
time-only families it *increases* with horizon (0.0040 to 0.0180, and 0.0243 to
0.0407). Both figures are reported: quoting the raw collapse alone would
overstate the effect by four orders of magnitude.

### 2. What changes is which knowledge axis leads, subject to a cascade threshold

The crossover is a ratio of two ceilings at identical T and `p_T`, so it is
invariant under the normalisation above and is the part of the horizon story
that survives. Whether the axes exchange dominance at all is governed by the
overload-cascade strength γ:

| γ | crossover T | job length | evidence |
|---|---|---|---|
| 0.30 | ≈ 3.2 | 0.8 h | located |
| 0.10 | ≈ 11 | 2.75 h | located |
| 0.05 | none through T=22 | > 5.5 h | ratio plateaus near 1.5 |

At γ=0.05 the component/time ratio falls 1.7305, 1.6010, 1.5645, 1.5388 across
T = 14, 18, 20, 22. The per-step decrement decays geometrically and extrapolates
to an asymptote near 1.46, so the ratio never reaches 1. This is threshold
behaviour, not a smooth scaling law: below a cascade strength somewhere between
0.05 and 0.10, component knowledge appears to retain its advantage indefinitely.

An earlier draft inferred `T_cross ∝ 1/γ` from the two located points. Extending
the γ=0.05 scan to T=22 refuted it. Two points define a line trivially; the third
determined the shape.

### 3. The one uncalibrated parameter dominates the operational answer

Every rate is calibrated to published Delta figures except γ, which was chosen.
It swings the failure probability over eleven orders of magnitude (8.75e-14 to
5.66e-3 for a 12-node pool). Sensitivity is therefore reported in place of a
point estimate. Calibrating γ against failure-rate-by-scale data is the
highest-value next step.

---

## Reproducing the results

```bash
pip install -r requirements.txt          # numpy, scipy
cd framework/dynamic_model
```

**Validation gates, about 5 seconds:**

```bash
python3 model.py
```

Expect `p_fail: 0.00633466…`, with `rows_sum_to_1` and `dp_matches_sim` both
`True`. `is_concentrated` is `False` for this N=3 reference, which has 3 of 8
states failing; concentration becomes meaningful only at the N=8 working
instance, where it is 15 of 256. The second block is the deliberately easy
lumpable case: identical parameters, homogeneous capacities, collapsed state
space.

**Findings 1 and 2, about 2 minutes:**

```bash
python3 run_decomposition.py --restarts 2 --out decomposition.json
```

Prints the recursion checks, the h-transform gate, the four-rung decomposition at
the reference horizon, and the horizon sweep with an extended-precision check at
each point.

**Findings 2 and 3, about 5 minutes:**

```bash
python3 run_gamma_sensitivity.py --out gamma_sensitivity.json
```

Three experiments. (A) locates the crossover per γ by log-interpolating the
ratio, reporting "no crossover in the scanned range" where none exists. (B)
sweeps γ against pool size to show the eleven-order spread. (C) asks the
provisioning question directly, with no node-versus-job proxy.

To reproduce the γ=0.05 result specifically, which needs long horizons:

```bash
python3 run_gamma_sensitivity.py --gammas 0.05 --horizons 14 18 20 22 --skip-sla
```

**Optional, the time-homogeneous ladder** (a coarser cut, retained for
comparison):

```bash
python3 run_ceiling_ladder.py --restarts 6 --out src_results.json
```

**Optional, figures and poster:**

```bash
cd ../../poster && python3 make_figures.py && pdflatex poster.tex
```

`make_figures.py` emits three panels: `collapse.pdf` (raw ceilings against
mission length, with the `1/p_T` reference line), `efficiency.pdf` (relative
efficiency `VRF·p_T`, the normalised view that motivates finding 1), and
`schedule.pdf` (the optimal tilt schedule at T=8, not currently used on the
poster). All three read from a table of measured values at the top of the file,
so updating results means editing `SWEEP` and rerunning.

> **Both drivers must sit beside `model.py`.** Every module uses flat imports, so
> moving them to a sibling `experiments/` directory fails with
> `ModuleNotFoundError: No module named 'model'`.

> **Python 3.14 may not have scipy wheels yet.** If `pip install` fails, use
> `python3.12 -m venv .venv && source .venv/bin/activate`.

---

## Method

For any Markov proposal `Q`, with `R = P²/Q`, the estimator's second moment obeys
a backward recursion that stops at first entry to the failure set `F`:

```
M₀(x) = 0
Mₛ(x) = Σ_{y∈F} R(x,y) + Σ_{y∉F} R(x,y) Mₛ₋₁(y)
Var   = M_T(start) − p_T²
```

Optimising this over a family's parameters gives that family's ceiling.

Mission failure is a **hitting** event, so reweighting stops the moment the
trajectory enters `F`. Getting this wrong biased the provably zero-variance
h-transform by −11.5%. The exact ceiling cannot detect that error, since it
assumes the estimator is implemented correctly; only the sampling audit caught
it. Both are retained.

**Four nested families**, chosen so the optimality gap decomposes:

| family | params | knows *when* | knows *which node* |
|---|---|---|---|
| scalar | 1 | no | no |
| + component | 2 | no | yes |
| + time | 8 | yes | no |
| + both | 16 | yes | yes |
| h-transform | — | yes | full joint state (optimal) |

Exchangeable nodes receive equal tilts at any optimum, so parameterising by class
is exact rather than approximate. This reduces the largest family from 64 to 16
parameters. Each family is warm-started from every smaller family it contains,
making the ladder monotone by construction.

---

## Correctness

Every gate runs automatically before any number is reported.

| gate | expected | what it establishes |
|---|---|---|
| recursion vs brute-force path enumeration | ~1e-16 | the variance formula is correct |
| time-varying vs time-homogeneous recursion | exact | the two code paths agree |
| backward DP vs naive simulation | within 3σ | the ground truth is correct |
| h-transform audit | bias +0.0000%, ESS 1.0000 | DP, stopping rule and audit are simultaneously correct |
| double vs extended precision | ~1e-15 | short-horizon results are not cancellation artefacts |
| ladder monotonicity | non-decreasing | nested families cannot regress |
| symmetry | equal tilts within a class | the optimiser converged |

The last two are not decorative. A Nelder-Mead run on the 16-parameter family
returned a ceiling below the 8-parameter family containing it, which is
impossible and was caught only by the monotonicity invariant. Warm-starting fixes
it by construction.

One number is worth understanding: naive Monte Carlo's *sampled* VRF is 1.074
where it must be exactly 1.000, since it is the baseline compared against itself.
That 7% error is the noise floor of a 20-trial estimate, and is why exact
ceilings are reported and sampling is used only to check bias.

---

## Layout

```
framework/dynamic_model/
  model.py                    exact ground truth: backward DP, h-transform, gates
  proposals.py                odds-tilt families, h-transform sequence
  ceiling.py                  exact second-moment recursion, family ceilings
  decompose.py                time-varying recursion, time-vs-component split
  audit.py                    trajectory sampling audit (stop-at-F rule)
  delta_calibration.py        Delta parameters, provisioning arithmetic
  diagnose_tilt.py            minimal-cut geometry, tilt-ordering diagnostic
  run_decomposition.py        DRIVER: decomposition and horizon sweep
  run_gamma_sensitivity.py    DRIVER: cascade robustness, crossover location
  run_ceiling_ladder.py       DRIVER: time-homogeneous ladder only
paper/    src_summary_acm.tex (SRC 800-word summary), full working report
poster/   poster.tex, make_figures.py, figs/
EXPLAINER.md                  plain-language walkthrough of every step
```

`model.py` retains `_build_transition_matrix_naive` alongside the vectorised
version so the fast path stays checkable. They are bit-identical, and the
vectorised form builds N=12 in about one second rather than minutes.

---

## Scope

**Not claimed.** The second-moment-as-value-function formulation is due to Dupuis
and Wang; the inadequacy of state-independent proposals is due to Glasserman and
Kou.

**Claimed.** An exact instantiation on a finite-horizon *hitting-event* chain,
where prior work either goes asymptotic or targets terminal-time events; a
decomposition of the optimality gap that is well posed only in finite horizon,
since the infinite-horizon committor literature has no clock; and measurement on
a model calibrated to production HPC data.

**Limitations.**

- Everything is O(2^N). Toy scale is deliberate: it is what permits exact
  validation, and what precludes scaling claims.
- Parameters are *calibrated to* published summary statistics, not *validated
  against* telemetry. γ is not calibrated at all.
- The horizon sweep varies T and `p_T` together; holding both fixed is impossible
  for a fixed system.
- The provisioning identity uses node availability while `F` is job-level, so
  that one derived figure treats job-mission failure as an explicit proxy.
  Nothing else depends on it, and experiment C of `run_gamma_sensitivity.py` asks
  the question directly instead.
- The crossover rests on two located points and one bound.

**Open.** The time-homogeneous optimum drives 8-GPU nodes harder (2.298 against
1.475); the time-varying optimum at step 0 drives 4-GPU nodes harder (31.04
against 9.98). `diagnose_tilt.py` tests the natural explanation, that the
ordering tracks the failure set's minimal cuts, and refutes it. Unexplained.

---

## Next

1. Implement AMS and cross-entropy for the trajectory model and price them
   against the same ceilings. There is currently no approximate classical
   opponent.
2. Learn `h_t` by temporal-difference methods on instances where exact `h_t` is
   known from DP, so learning error is measurable exactly.
3. Calibrate γ against failure-rate-by-scale data so the operational conclusions
   stop being sensitivity ranges.

## References

1. S. Cui et al. *Story of Two GPUs: Characterizing the Resilience of Hopper H100
   and Ampere A100 GPUs.* SC '25. arXiv:2503.11901. Source of every Delta figure.
2. J. Blanchet, H. Lam. *State-dependent importance sampling for rare-event
   simulation.* Surv. Oper. Res. Manag. Sci., 2011. Covers Dupuis and Wang.
3. P. Glasserman, S.-G. Kou. *Analysis of an importance sampling estimator for
   tandem queues.* ACM TOMACS, 1995.
4. F. Cérou, A. Guyader. *Adaptive multilevel splitting.* Stoch. Anal. Appl., 2007.
5. R. Rubinstein, D. Kroese. *The Cross-Entropy Method.* Springer, 2004.
6. Z. Botev, P. L'Ecuyer, G. Rubino, R. Simard, B. Tuffin. *Static network
   reliability estimation via generalized splitting.* INFORMS J. Comput., 2013.
7. G. Chennetier et al. *Adaptive importance sampling based on fault tree
   analysis.* SIAM/ASA JUQ. arXiv:2210.16185.
8. *The surprising efficiency of temporal difference learning for rare event
   prediction.* NeurIPS 2024. arXiv:2405.17638. Infinite-horizon; motivates the
   next phase.
9. *A unified perspective on exponential tilt and bridge algorithms.*
   arXiv:2307.12597. Closest prior work; terminal-time rather than hitting.

## Acknowledgements

NSF Award No. 2620473 · CRA/NSF DREU · NCSA Delta.
