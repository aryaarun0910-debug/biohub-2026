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
Group candidates that are duplicate detections of ONE nucleus. **Do NOT use transitive connected
components (P0 fix): a chain of nearby proposals can bridge two real nuclei.** Instead:
- primary: assign each candidate to a **watershed / intensity basin** of the DoG (or foreground)
  response; candidates in the same basin are same-cell duplicates. Basins respect the intensity
  saddle between two real nuclei, so they don't bridge.
- fallback/secondary: **complete-linkage** clustering with a hard **diameter cap** (max pairwise
  distance < r_same ~ one nuclear radius) so no cluster can span two nuclei via a chain.
Distinct nuclei within 7 um land in DIFFERENT sets by construction.
GUARDRAIL TEST (phase-critical): on GT, two annotated nuclei <7 um apart must NOT share a set
(target merge-rate ~0); this test gates the whole phase and runs in CI on synthetic + GT crops.

### 3. Matching-aware arbitration  (src/biotrack/arbitrate.py)
Within each same-cell set, pick ONE representative maximizing a score:
  s = w1*image_evidence + w2*temporal_continuity + w3*matching_stability
- image_evidence: DoG response / contrast (subvoxel-refined centroid);
- temporal_continuity: does this position extend a strong track (velocity-consistent neighbor in
  t-1 and t+1)? computed from a first-pass link on the deduped set;
- **matching_stability (P1 fix: operationalized, not vibes):**
  (a) bidirectional link residual = |predicted - observed| for the incoming AND outgoing link;
  (b) best-vs-second-best association margin (how much better is this candidate's chosen partner
      than its runner-up — larger margin = more stable);
  (c) link invariance under small coordinate perturbation (jitter +-epsilon; fraction of links
      unchanged). Each is computed from the first-pass link; combine as a normalized sum.
Compare against baselines (brightest peak; nearest-motion) on the exact metric.

### 4. Track-conditioned redetection  (src/biotrack/redetect.py)
For strong tracks (>= L consecutive velocity-consistent links) with a 1-frame gap or a weak
endpoint, predict the position at the missing t via inherited velocity and search the local
full-res image. ACCEPT by a confidence gate:
  accept if image_score(patch) >= tau_img AND trajectory_consistency >= tau_traj
**Calibration (P0 fix: no leakage). tau_* are calibrated INSIDE the TRAINING embryo's inner
splits (nested CV), FROZEN, then evaluated cross-embryo** — never tuned on the evaluation embryo.
Add at most one node per predicted gap. NEVER uses GT at inference.

### 5. Linking + emit
Reuse the V3 two-pass velocity Hungarian on the arbitrated+redetected node set; apply the min-
track-length filter; emit deduplicated nodes only. Count calibration (Phase 2) stays out for now.

**Redetection value metric (P1 fix): NOT "added-node precision" (ambiguous under sparse labels).**
Gate on **marginal edge contribution**: for the added nodes, report added TP edges, added FP edges,
recovered FN edges, and the exact-score delta. A recovery is good only if exact-score delta > 0.

## Experiment ladder (gate = min(fold) improvement, bootstrap CI excludes 0)
| Step | Change | Success metric |
|---|---|---|
| 1a | Over-proposal recall-vs-count frontier | recall >0.95 achievable at ratio <=2x on TRAIN inner splits |
| 1b | Basin same-cell conflict sets | GT merge-rate ~0 (guardrail); sane dedup count |
| 1c | Arbitration vs brightest/nearest | min-fold adjJ +>=0.005 over V3 |
| 1d | + track-conditioned redetection | min-fold adjJ +>=0.003; positive marginal edge contribution |
| 1e | Full Phase-1 vs V3 anchor | min-fold adjJ up; count ratio 0.95-1.10; no slice regresses |

## Deliverables
- `src/biotrack/propose.py`, `arbitrate.py`, `redetect.py` (+ unit tests incl. the <7um-no-merge guardrail).
- `scripts/run_phase1_ablation.py` -> per-crop CSV (adjJ, recall, ratio, added-node precision, merge-rate)
  under each ladder step, parallel, nested-CV aware.
- Figures from that CSV; journal entry with both-fold bootstrap CIs.
- Commit + push each step.

## Corrected launch order (Codex, adopted)
1. **Run + submit untouched V3 — one anchor submission** (account has none; later gains are
   uninterpretable without it). Notebook-only comp -> push `kaggle_dog_infer.py` as a Kaggle kernel.
2. Build `run_phase1_ablation.py` + candidate caching (cache proposals so ladder steps are fast).
3. Implement `propose.py`; measure the recall-vs-candidate-count frontier.
4. Implement basin-based same-cell conflict sets + guardrail tests.
5. Provisional brightest representatives -> link -> arbitrate -> relink.
6. Track-conditioned redetection ONLY after arbitration shows positive cross-embryo transfer.
7. Submit Phase-1 ONLY if BOTH frozen embryo evaluations improve.

Realistic expectation (Codex): Phase 1 -> +0.01-0.03; **0.880 from a 0.842 anchor likely needs
Phase 1 + tissue-flow/count adaptation (Phase 2) or sparse divisions (Phase 5).** Sequence honestly.
