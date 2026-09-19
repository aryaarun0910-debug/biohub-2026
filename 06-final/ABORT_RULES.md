# ABORT_RULES.md — frozen 2026-09-19

Written by day-1 Arya so that day-7 Arya cannot renegotiate.
Nothing in this file may be edited after 2026-09-19 except to record an
outcome in the Outcome column. If a rule fires, it fires.

Final submission deadline: **2026-09-29 23:59 UTC**.

---

## The thesis being tested

Public systems are strong on edges (~0.926 adj_edge) and weak on divisions
(~0.22 division_jaccard, detection ceiling 100%). Therefore: borrow the
detector, attack divisions, spend runtime there instead of on TTA.

This thesis may be wrong. The oracle decomposition on 2026-09-21 is designed
to kill it, not to confirm it.

---

## Pre-registered decision rules

| Date | Test | Continue if | Otherwise |
|---|---|---|---|
| **09-21** | Oracle: `O2 > 1.5 x O1` | O1 wins clearly -> divisions stay primary | Switch primary axis to fragmentation (learned 1-frame stitcher; ~80% of broken tracks are gap-1, ~18% gap-2, <1% gap-5). Divisions become secondary. |
| **09-21** | Oracle: both O1 and O2 small | at least one > 0.010 | Harden the 0.947 repro. Spend remaining submissions on threshold calibration and the two-final hedge. Accept a robust finish. |
| **09-21** | Division probe AUC vs matched controls | >= 0.85 -> use frozen features, skip the CNN | 0.70-0.85 -> build patch model. **< 0.70 -> learned division classifier is no longer the main bet.** |
| **09-21** | Runtime extrapolation to ~200 films | <= 10 h | Clock problem, not a modelling problem. It becomes the priority. Cut components until it fits. |
| **09-22** | Miner sanity vs 151 frozen real events | strict-mined events resemble real positives under frozen features | Stop self-training. Fall back to the 151 real events + geometry. |
| **09-22** | Loose-pool calibration | loose scores broad/intermediate | If loose collapses onto the strict-positive distribution, the miner has leaked its rule into the classifier. Mined supervision is then NOT trusted for threshold selection. |
| **09-25** | Fork null-equivalence test | byte-identical edge set at lambda_fork = inf | **No fork solver.** Ship baseline + second-pass augmentation only. |
| **09-25** | LOEO combined score | `dJ_edge + 0.1*dJ_div > 0` AND `dJ_edge >= 0` | Reject the change regardless of how good division_jaccard looks. |
| **09-26** | Architecture | — | **FREEZE.** No new model families, no new components. Thresholds only. |
| **09-27..28** | — | — | Calibrated threshold sweep only. No architecture changes. |
| **09-29** | Finals | two intentionally different division thresholds | Preserve the hedge. Do not submit two draws from the same distribution. |

---

## Standing prohibitions

1. **The node-count term is forbidden ground.** Published post-mortem:
   +0.013 offline, -0.004 on the board, four submissions burned. The mechanism
   is that one embryo has ~0.8% of nodes labelled and the other ~9.7%, so
   deleting tracks looks free offline. The labels can confirm the reward and
   cannot confirm the price. `N_pred` is measured and reported, never tuned.

2. **Baseline continuation edges are immutable.** Divisions are added by a
   second pass over targets left unmatched by the 1:1 assignment. Pass 1 is
   never re-solved to accommodate a fork.

3. **The daughter gate is learned, not fixed.** A wide fixed radius rebuilds
   `safe_div` and pays false positives across the whole graph. The wide gate is
   admissible only where a learned per-candidate score pays for it.

4. **The 151 real labelled divisions are a test set.** Never trained on.

5. **Every run reports `J_edge`, `J_division` and the node-count multiplier
   separately.** "Multiplier moved, J flat" is the signature of a fake gain.

6. **One change per submission.** The alternative has a post-mortem too.

7. **Per-film adaptation is a frozen deterministic function of label-free film
   statistics**, fitted on training films and frozen before any hidden-set
   inspection. Normalised distances (`d / d_NN3`) rather than raw density. No
   gradient-based or pseudo-label test-time adaptation.

---

## Measurement error — what a delta must clear to be real

| Quantity | Value | Source |
|---|---|---|
| Paired SE on a 24-film hold-out | ~0.006 | hikaggler, controlled one-variable study |
| Movie-to-movie spread, leave-one-embryo-out | +/-0.14 | ramarlina |
| Division event resolution | `0.1 / D` | D = labelled divisions in the ruler |
| ... at D = 26 (sparse-embryo LOEO fold) | 0.0038 per event | |
| ... at D = 151 (all) | 0.00066 per event | |
| Public board share of test | 29% | competition page |
| Identical-submission reproducibility | unmeasured -> **measure it** | GPU nondeterminism changes rows |

**A reported +0.001 is not a measurement.** Most of the public lineage's
"+0.001 steps" are inside this noise.

---

## Submission budget

~50 total. 4-5 spent on measurement, front-loaded:

- [ ] 0.947 repro -> the floor
- [ ] division-ablation pair (divisions on / off; difference / 0.1 = true division_jaccard)
- [ ] runtime probe on the ~200-film clock
- [ ] identical-submission repeat -> board noise floor

Without the ablation pair you are optimising a quantity you cannot observe.
The board shows only the sum: two people on 0.948 can need opposite work.
