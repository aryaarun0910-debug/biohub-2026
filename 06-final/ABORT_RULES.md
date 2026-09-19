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
