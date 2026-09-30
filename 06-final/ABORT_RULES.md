# ABORT_RULES.md

Frozen 2026-09-19. Amended 2026-09-19 (see log at the bottom).

Written by the version of us that is calm, so that the version of us at 3am
cannot renegotiate. Triggers below fire on **evidence, not on elapsed time**.
If a gate fails, the named fallback is taken. Not discussed — taken.

Hard deadline: **2026-09-29 23:59 UTC**.

---

## The thesis being tested

Public systems are strong on edges (~0.926 adj_edge) and weak on divisions
(~0.22 division_jaccard, with a measured detection ceiling of 100%: parent and
both daughters detected 7/7, both arcs linked 0/7). Therefore: borrow the
detector, attack divisions, spend runtime there instead of on TTA.

**This thesis may be wrong.** Gate G3 exists to kill it, not to confirm it.

---

## Gates

Each gate has an entry condition, a test, and a named fallback. You may not
pass a gate by deciding the test was unfair.

### G1 — Floor
**Entry:** weights local, inference notebook runs.
**Test:** the 0.947 repro scores on the board.
**Pass:** you own a floor and the weights-as-private-dataset path works.
**Fail:** nothing else matters until it does. This is plumbing, not modelling.

### G2 — Instrumentation
**Entry:** G1.
**Test:** predicted graphs cached for all 199 films; node-sampled features
cached; runtime extrapolated to a ~200-film hidden set; board noise floor
measured by submitting the same notebook twice; true `division_jaccard` read
off the divisions-on/divisions-off ablation pair.
**Pass:** every later number is interpretable.
**Fail — runtime > 10 h:** this is a clock problem, not a modelling problem.
It becomes the priority. Cut components until it fits. Precedent: a public
pipeline whose veto silently drops films it cannot finish, never scored.

### G3 — THE THESIS TEST (the important one)
**Entry:** G2.
**Test:** oracle headroom decomposition on the frozen LOEO folds.
- **O1** division ceiling — force correct forks, delete wrong ones
- **O2** association ceiling — force every GT edge whose endpoints are already matched
- **O3** detection ceiling — insert unmatched GT nodes, then O2
- **O4** node-count — set `N_pred = N_est`, to know its size, never to tune it

Each reports `dJ_edge` and `0.1*dJ_div` separately, plus the multiplier.

**Read O1 to three decimals at most.** On fold 44b6, D=26, so one event is
0.0038 of total score. O2 rests on ~129k labelled edges and is measured far
more precisely. If O1 and O2 are close, prefer O2 on measurement confidence
alone — you can see changes there and you cannot see them on divisions.

| Outcome | Action |
|---|---|
| `O1 > 1.5 x O2` | Thesis holds. Divisions primary. Proceed. |
| `O2 > 1.5 x O1` | **Switch primary axis to fragmentation.** Learned 1-frame stitcher: ~80% of broken tracks are gap-1, ~18% gap-2, <1% gap-5. Reuses the same feature cache, second-pass solver and null test. Divisions become secondary. |
| neither exceeds 1.5x | Run both off the shared second-pass machinery. Divisions first (uncrowded). |
| **both < 0.010** | Ceiling is in detection or unreachable nodes. **Stop building.** Harden the repro, spend the rest on threshold calibration and the two-final hedge. Accept a robust finish over a heroic one. |

### G3b — Probe (runs alongside G3)
**Test:** L2 logistic probe on frozen encoder features at the 151 labelled
division parents vs **density- and intensity-matched** controls, LOEO, at
t, t-1 and t-2 separately.
- **AUC >= 0.85** — signal is already in the frozen features. Skip the CNN.
- **0.70 - 0.85** — build the patch model, now with evidence it is needed.
- **AUC < 0.70** — a learned division classifier is no longer the main bet.

The matching is the experiment. Unmatched, the probe relearns "this region is
bright" and hands back the AUC 0.73 we already have from raw intensity.
Whichever way it lands, the t / t-1 / t-2 comparison locates the signal: two
independent observers believed anaphase precedes the labelled split and
neither tested it.

### G4 — Supervision is valid
**Entry:** G3 chose divisions.
**Test:** strict-mined events resemble the 151 frozen real positives under
frozen features; the loose pool's score distribution is broad and intermediate,
stratified by *why* it failed strict mining (insufficient persistence /
daughter exits volume / non-monotonic divergence / dense-neighbourhood
ambiguity / fragmentation / near film end).
**Fail — loose collapses onto the strict-positive distribution:** the miner has
leaked its own rule into the classifier. Mined supervision is then **not
trusted for threshold selection**. Fall back to the 151 real events plus geometry.

### G5 — Integration is safe
**Entry:** a trained division (or stitching) model.
**Test 1 — null equivalence.** With `lambda_fork = inf`, the augmented solver
returns a **byte-identical** sorted edge list to baseline. Not similar. Identical.
**Test 2 — first-child immutability.** Baseline continuation edges cannot be
altered. Only targets left unmatched by the 1:1 pass are eligible as second children.
**Test 3 — accounting.** `dJ_edge + 0.1*dJ_div > 0` **and** `dJ_edge >= 0`.

**Fail Test 1:** no fork solver. Ship baseline plus second-pass augmentation only.
**Fail Test 3:** reject the change regardless of how good division_jaccard looks.
You are spending 0.1-weighted points to buy 1.0-weighted ones; a division gain
that costs edge Jaccard is very probably a loss wearing a disguise.

### G6 — FREEZE  *(deadline-anchored, no later than 2026-09-27 00:00 UTC)*
Fires at G5 if G5 lands earlier — and it should. From here: **no new model
families, no new components.** Thresholds only.
This is the one gate that cannot be pulled later by acceleration, because
everything in the corpus that went wrong went wrong late.

### G7 — Finals  *(deadline-anchored)*
Two submissions bracketing the **single exposed division threshold**: best
measured operating point, and a deliberately precision-heavy twin. Not two
draws from the same distribution.
Rationale: on comparable boards that reshuffled, 5-18% of teams finished with
no medal while holding a submission that would have earned one.

---

## Standing prohibitions

1. **The node-count term is forbidden ground.** Published post-mortem:
   +0.013 offline, -0.004 on the board, four submissions burned. One embryo has
   0.77% of nodes labelled and the other 9.71%, so deleting tracks looks free
   offline. The labels can confirm the reward and cannot confirm the price.
   `N_pred` is measured and reported, never tuned.
2. **Baseline continuation edges are immutable.** Divisions are added by a
   second pass over unmatched targets. Pass 1 is never re-solved for a fork.
3. **The daughter gate is learned, not fixed.** A wide fixed radius rebuilds
   `safe_div` and pays false positives across the whole graph.
4. **The 151 real labelled divisions are a test set.** Never trained on.
5. **Every run reports `J_edge`, `J_division` and the multiplier separately.**
   "Multiplier moved, J flat" is the signature of a fake gain.
6. **One change per submission.**
7. **Per-film adaptation is a frozen deterministic function of label-free film
   statistics**, fitted on training films, frozen before any hidden-set
   inspection. Normalised distances (`d / d_NN3`), not raw density. No
   gradient-based or pseudo-label test-time adaptation.
8. **Decide by expected score gain per engineering hour.** A division_jaccard
   that comes back already high means *less* remaining headroom, not license to
   double down: 0.30 -> 0.40 buys +0.01 before edge consequences.

---

## Measurement error — what a delta must clear to be real

| Quantity | Value | Source |
|---|---|---|
| Paired SE, 24-film hold-out | ~0.006 | controlled one-variable study |
| Movie-to-movie spread, LOEO | +/-0.14 | ramarlina |
| Division event resolution, fold 44b6 (D=26) | **0.0038** | measured here |
| Division event resolution, fold 6bba (D=125) | 0.0008 | measured here |
| Division event resolution, dev subset (D=52) | 0.0019 | measured here |
| Public board share of test | 29% | competition page |
| Identical-submission reproducibility | unmeasured -> **measure at G2** | GPU nondeterminism |

**A reported +0.001 is not a measurement.** Most of the public lineage's
"+0.001 steps" sit inside this noise.

---

## Submission budget (~50)

Measurement first, four of them, all at G1-G2:
- [ ] 0.947 repro -> the floor
- [ ] divisions on / divisions off -> true division_jaccard by subtraction
- [ ] runtime probe on the ~200-film clock
- [ ] identical repeat -> board noise floor

Without the ablation pair you are optimising a quantity you cannot observe: the
board shows only the sum, and two people on 0.948 can need opposite work.

---

## Amendment log

- **2026-09-19, initial.** Frozen before any model code.
- **2026-09-19, amendment 1.** Day-numbered schedule replaced by evidence
  gates G1-G7. Reason: a calendar encodes an assumed pace and silently becomes
  a reason to skip a test when ahead, or to panic when behind. Gates fire on
  measurements. **Two exceptions retain a wall-clock anchor** — G6 (freeze) and
  G7 (finals) — because 2026-09-29 is not negotiable and lateness is the
  failure mode this file exists to prevent. Both are "no later than": moving
  faster pulls them earlier, never later.

- **2026-09-19, amendment 2. G3 PASSED — thesis holds, divisions stay primary.**
  Measured on all 199 films, frozen LOEO folds, metric verified against ground
  truth identity (J=1.0000, divisions exactly 2/0/0).

  | | ceiling | deployed now | headroom |
  |---|---|---|---|
  | adj edge Jaccard | 0.9651 (perfect association, current detections) | ~0.926 | +0.039 |
  | division term | 0.9536 x 0.1 = 0.0954 | ~0.19-0.23 -> 0.021 | **+0.074** |

  Divisions are ~1.9x association. Association only wins if current adj_J is
  below 0.852; the deployed pipeline is well above that. Free number that
  decided it: **99.63% of GT nodes matched, 99.34% of GT edges have BOTH
  endpoints matched.** Detection is not the bottleneck -- O3 buys only +0.0108
  over O2. Caveat kept on the record: the 0.9536 ceiling assumes you know which
  forks to create; realistic attainment (divJ 0.20 -> 0.45 = +0.025 against
  association 0.926 -> 0.945 = +0.019) narrows the lead considerably.

- **2026-09-19, amendment 3. Node-count finding: measured, banked, NOT chased.**
  O4 says neutralising the node-count term is worth +0.0232, and isolated-node
  pruning captures only +0.004 of it, so ~+0.024 is genuinely unclaimed. This
  is NOT prohibition #1 (which bans farming a multiplier above 1 by deleting
  true tracks); moving a ratio toward 1.0 without crossing it is precision.

  It is not being built, for a measured reason. `estimated_number_of_nodes`
  lives in the .geff and test films ship .zarr only, so capturing it needs a
  predicted n_est -- and that predictor transfers 1-for-2 across embryos:
  44b6 -> 6bba gives R2 +0.937, 6bba -> 44b6 gives **R2 +0.494 with a median
  17% UNDERestimate and a p10 of 0.43**. Under-predicting n_est sets the
  threshold too high and lands below true n_est: the exact configuration with a
  published -0.004 board result. Two cross-embryo points, extrapolated to an
  unseen third, with an asymmetric tail into the forbidden region.

  Also on the record: the gap to 1st does not require this lever.
  adj_J 0.950 + 0.1 x divJ 0.23 = 0.973, exactly the leader's score.

  **Approved instead:** a global detection threshold raise 0.965 -> 0.99.
  No new component, no n_est prediction, +0.006 of ceiling, and NEITHER embryo
  crosses parity (44b6 -> 1.009, 6bba -> 1.263). One submission to measure.
  If the predictor is ever built, its target is ratio **1.15, not 1.0**, so a
  17% bias still leaves it above parity.

- **2026-09-19, amendment 4. G3b FIRED — learned division classifier dropped.**
  L2 logistic probe on frozen UNet features, division parents vs controls
  matched within the same film and frame, leave-one-embryo-out:

  | features | AUC | 95% CI |
  |---|---|---|
  | raw peak intensity | 0.630 | 0.572-0.687 |
  | frozen features at t-2 | 0.484 | 0.437-0.538 |
  | frozen features at t-1 | 0.510 | 0.464-0.559 |
  | **frozen features at t+0 (labelled split)** | **0.456** | 0.419-0.506 |
  | frozen features at t+1 | 0.663 | 0.611-0.717 |
  | all offsets + intensity + density | 0.581 | 0.535-0.629 |

  Nothing clears the 0.70 floor; at the split frame the encoder is at CHANCE.
  Per the G3b gate, a learned division classifier is no longer the main bet.
  The anaphase hypothesis is also NOT supported: t-1 and t-2 are both chance.
  That question is now tested rather than believed.

  Scope of the experiment, stated but NOT used to overturn the trigger: it
  tests single-voxel LINEAR decodability, not the image at patch scale; the
  matching was imperfect (intensity alone still reads 0.630); and the 44b6 fold
  carries only 11 positives, so only 6bba (123) has weight.

  **The only real signal is at t+1 (0.663) -- AFTER the split, when two
  daughters exist. That is CONSEQUENCE, not cause.**

  **Replacement mechanism (approved):** promote the consequence criterion from
  labeller to DECISION RULE. Geometry already reaches divJ 0.19-0.23 in the
  deployed pipeline; the probe says appearance will not beat it. Divisions are
  scored by trajectory consequence -- both daughters persist, separation grows
  monotonically, symmetry about the parent -- inside the baseline-immutable
  second pass. This is legal at inference: tracking is offline over the whole
  movie, so the predicted graph's future is observable without labels.
