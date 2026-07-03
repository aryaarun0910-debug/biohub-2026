# Phase 1 Design — Matching-aware arbitration + track-conditioned redetection

**Goal:** convert the measured recall deficit (77-80% of missed edges are no-candidate; V3 adjJ
0.632/0.756) into score by proposing more internally and emitting the *right* one representative
per cell, then recovering cells along strong tracks. Expected +0.01-0.03 min-fold. Everything is
gated on the exact numpy metric across BOTH embryo folds with nested CV + bootstrap CIs.

## Guardrails (from the two red-team passes — do not violate)
- **7 um is the evaluator's matching gate, NOT a dedup radius.** Real distinct nuclei coexist
  within 7 um. Conflict sets = SAME-CELL duplicates only; multiple trajectories may sit inside 7 um.
- **Redetection acceptance is an INFERENCE rule** = calibrated image evidence + trajectory
  confidence. "Recovers two edges" is an OFFLINE evaluation of the idea, never used at inference.
- **Selection** = nested grouped crop/time split within-embryo (selector) + bootstrap CI; freeze;
  then cross-embryo (44b6<->6bba) as the unbiased honesty gate. Never tune on the two folds directly.
- Emit only deduplicated, track-supported nodes; over-propose INTERNALLY (FP counts if an edge
  endpoint matches an annotated node).

## Pipeline (each stage is independently ablated; keep only min-fold-positive)

### 1. Over-proposal  (src/biotrack/propose.py)
Generate a HIGH-RECALL candidate set per frame with per-candidate features:
- multi-threshold / multi-scale DoG bank (lower rel-threshold, extra scale pairs) -> ~1.5-2x count;
- per candidate: DoG response, local scale (which band fired), raw intensity, local contrast,
  local density (kNN count).
Target: raise node recall from ~0.83 toward >0.95 on held-out embryo (measure directly).

### 2. Same-cell conflict sets  (src/biotrack/arbitrate.py)
Group candidates that are duplicate detections of ONE nucleus. Edge between two candidates iff:
- physical distance < r_same (a SMALL same-cell radius ~ one nuclear radius, well below 7 um), AND
- they share image support (both on the same local intensity mode: connected above a local
  fraction of the shared peak / monotone intensity ridge between them).
Connected components = same-cell sets. Distinct nuclei within 7 um land in DIFFERENT sets.
VALIDATION: on GT, assert two annotated nuclei <7 um apart are NOT merged into one set (target
merge-rate ~0); this is the guardrail test for the whole phase.

### 3. Matching-aware arbitration  (src/biotrack/arbitrate.py)
Within each same-cell set, pick ONE representative maximizing a score:
  s = w1*image_evidence + w2*temporal_continuity + w3*matching_stability
- image_evidence: DoG response / contrast (subvoxel-refined centroid);
- temporal_continuity: does this position extend a strong track (velocity-consistent neighbor in
  t-1 and t+1)? computed from a first-pass link on the deduped set;
- matching_stability: is it robust to small perturbation (won't be stolen / won't steal)?
Compare against baselines (brightest peak; nearest-motion) on the exact metric.

### 4. Track-conditioned redetection  (src/biotrack/redetect.py)
For strong tracks (>= L consecutive velocity-consistent links) with a 1-frame gap or a weak
endpoint, predict the position at the missing t via inherited velocity and search the local
full-res image. ACCEPT by a calibrated confidence gate:
  accept if image_score(patch) >= tau_img AND trajectory_consistency >= tau_traj
(tau_* calibrated on the held-out embryo to a target added-node precision > 70%). Add at most one
node per predicted gap. NEVER uses GT.

### 5. Linking + emit
Reuse the V3 two-pass velocity Hungarian on the arbitrated+redetected node set; apply the min-
track-length filter; emit deduplicated nodes only. Count calibration (Phase 2) stays out for now.

## Experiment ladder (gate = min(fold) improvement, bootstrap CI excludes 0)
| Step | Change | Success metric |
|---|---|---|
| 1a | Over-proposal recall ceiling | recall on held-out embryo >0.95 at ratio <=2x (headroom exists) |
| 1b | Same-cell conflict sets | GT merge-rate ~0 (guardrail); dedup count sane |
| 1c | Arbitration vs brightest/nearest | min-fold adjJ +>=0.005 over V3 |
| 1d | + track-conditioned redetection | min-fold adjJ +>=0.003; added-node precision >70% |
| 1e | Full Phase-1 vs V3 anchor | min-fold adjJ up; count ratio in 0.95-1.10; no slice regresses |

## Deliverables
- `src/biotrack/propose.py`, `arbitrate.py`, `redetect.py` (+ unit tests incl. the <7um-no-merge guardrail).
- `scripts/run_phase1_ablation.py` -> per-crop CSV (adjJ, recall, ratio, added-node precision, merge-rate)
  under each ladder step, parallel, nested-CV aware.
- Figures from that CSV; journal entry with both-fold bootstrap CIs.
- Commit + push each step.

## First build (this session)
Step 1a-1b: `propose.py` (over-proposal + features) and the same-cell conflict-set builder in
`arbitrate.py`, with the GT guardrail test (two nuclei <7 um must not merge). Measure the recall
ceiling and merge-rate before touching arbitration.
