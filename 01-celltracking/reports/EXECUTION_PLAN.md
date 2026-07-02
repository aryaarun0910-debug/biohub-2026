# Biohub Cell Tracking — Execution Plan (win the private LB)

**Horizon:** 2 Jul 2026 -> final 29 Sep 2026 (~13 weeks). Team-merge/entry deadline 22 Sep.
**North star:** maximize the PRIVATE 71% score. Public LB is a sanity check.
**TARGET (explicit):** current public leader 0.875 (moving up fast; a fresh 0.874 appeared today).
0.84 is COMMODITY = our Phase-0 anchor only. Goal = **>=0.88 on the PRIVATE set**, i.e. beat the
top of the public board with margin that survives the 71% private split. The phase deltas are sized
to stack there; anything that doesn't move min-fold gets cut.

**Validation (CORRECTED - avoid two-fold overfitting):** do NOT repeatedly select on the same two
embryo folds - that overfits them. Instead: (a) tune/select using NESTED grouped crop+time splits
WITHIN each embryo, with bootstrap uncertainty on the score; (b) FREEZE the chosen config; (c) then
evaluate it cross-embryo (44b6<->6bba) as an UNBIASED check, not a tuning knob. Report bootstrap CIs,
not point scores. Cross-embryo is the honesty gate; the nested within-embryo split is the selector.

## The thesis (why we can win, not just medal)
The whole 0.826-0.875 field optimizes DETECTION and LINKING separately. But the evaluator does a
one-to-one bipartite match FIRST (7um, weight 1/(1+d)), THEN judges edges. So a spatially-closer
DUPLICATE can steal a GT node's match from the proposal carrying the correct trajectory. This is a
CORRELATED, structural weakness across the leaderboard that more compute cannot fix.
Measured on our data: 71-79% of missed edges are DETECTION-miss (recall 0.83/0.91, count ratio <1).
=> **Core weapon: over-propose internally -> matching-aware, track-conditioned proposal ARBITRATION
-> gap redetection along strong tracks -> emit ONE deduplicated representative per cell.**
Same image evidence -> more correct edges, fewer FPs, better count. Metric-native, embryo-general.

## Operating model
- **Local (this machine):** DoG sweeps, arbitration/linking logic, EXACT metric scoring (numpy harness,
  validated), full-data EDA, all 129 crops embryo-held-out. MX350 (2GB) CANNOT train nets.
- **Kaggle / cloud GPU:** train Spotiflow/learned scorers; final inference notebook (<12h, offline).
- **Discipline:** exact-metric go/no-go gate BEFORE every submission; one question per submission;
  experiment registry (git commit, params, both-fold scores, count ratio, runtime, hypothesis).

## Infra already built (green)
Exact numpy metric (20/20 == tracksdata) | embryo-held-out split | lossless submission converter |
V3 DoG notebook | edge-error taxonomy | DAXI I/O verified | full 87GB local | Codex intel folded.

---

## PHASE 0 — Anchor & instrument (Days 1-4)  [target: trustworthy ~0.84 local baseline]
1. **Close the V3 reproduction gap.** Use each zarr's precomputed `image_statistics.quantiles` for
   normalization (host does); re-score 20->all 129 crops. Sweep DoG rel-threshold + peak cap for recall.
   GATE: reach ~0.82-0.84 adj-J on both folds, OR document the exact residual cause.
2. **First Kaggle submission = V3 anchor.** Confirm submission format accepted; measure local<->public
   correlation (our both-fold mean vs the 29% public). Establishes the calibration we trust.
3. **Provenance clarification - PRIVATE first (CORRECTED).** Prefer a PRIVATE organizer clarification
   (direct message / private channel) over a public forum post, so we don't advertise the provenance
   angle to the field. Narrow wording: "If a test image is independently found to be a crop of a freely
   public pre-existing embryo volume, may we use that volume's pre-existing public tracking CSV at
   inference?" Do NOT disclose a suspected match. Preserve the written answer. (Gates the provenance edge.)
4. **Error-taxonomy dashboard** stratified by density, z-depth, timepoint, intensity, motion, crop
   boundary. Locates exactly where detection recall dies -> targets Phases 1 & 3.

## PHASE 1 — The core weapon: matching-aware arbitration + gap redetection (Week 1-2)  [+0.01-0.03]
Codex edge #1 (HIGH, PERMITTED), confirmed by our taxonomy.
1. **Over-propose** 1.5-2x count internally (lower threshold / multi-DoG bank) -> high recall candidate set.
2. **Conflict sets = SAME-CELL duplicates, NOT a 7um radius (CRITICAL CORRECTION).** 7um is the
   evaluator's matching gate, NOT a dedup radius - multiple REAL nuclei routinely sit within 7um, so
   collapsing everything in a 7um ball destroys recall. Build conflict sets from shared IMAGE MODES /
   shared track hypotheses (candidates that are duplicate detections of ONE underlying nucleus:
   same local intensity mode, overlapping support, near-identical trajectory). ALLOW multiple distinct
   trajectories to coexist inside the 7um gate. Within a same-cell conflict set only, choose ONE
   representative by joint image evidence + temporal continuity + one-to-one-matching stability.
   Compare vs baselines (brightest peak, nearest-motion) on the EXACT metric.
3. **Track-conditioned redetection - INFERENCE rule (CORRECTED).** Where a strong track predicts a
   cell at t (gap or weak signal), search the local full-res image at the motion-predicted position and
   accept a recovered node using CALIBRATED IMAGE EVIDENCE + trajectory confidence (heatmap/DoG response,
   local contrast, velocity consistency) - NOT "did it recover a GT edge" (that is an offline evaluation
   oracle, unavailable at inference). "Recovers two edges" is only how we SCORE the idea offline.
4. **Full-res local centroid refinement** only around SELECTED peaks (XY/4 discards crowded-region info).
GATE: min-fold +0.005 (nested-CV, bootstrap CI excludes 0); recovered-node precision >70% on held-out
embryo; count ratio stays 0.95-1.10.

## PHASE 2 — Tissue-flow linking + count/adaptive detection (Week 2-4)  [+0.01-0.02]
Codex edges #2,#4,#5. NOTE: public ZSNS priors must be NORMALIZED for frame cadence and imaging scale
before use (ZSNS voxel scale/cadence differ from the competition) - use as shape/soft priors, re-fit
magnitudes on competition data.
1. **Local coherent-motion linking:** subtract kNN-median displacement; score the RESIDUAL. Density-
   adaptive gate (tight in dense/slow, wide in strong-flow regions). (ZSNS003 prior, cadence-normalized:
   step median 1.3um/fr.)
2. **Density/depth/time-adaptive detection thresholds** (not one global DoG threshold).
3. **Count calibration** into r=0.95-1.05 via confidence pruning, temporally smoothed, embryo-held-out.
4. **Preprocessing ablations (factorial, identical seeds):** precomputed-quantile norm, phase-correlation
   registration, white-tophat sigma~20 background subtraction (Ultrack recipe). Keep only min-fold-positive.
GATE: each component min-fold +0.002 and no density/intensity slice regresses >0.01.

## PHASE 3 — Learned residual detector (Week 4-7)  [+0.005-0.02, high variance]
JUSTIFIED by measured det-miss dominance. Codex edge #4/#6. Trains on Kaggle/cloud.
1. **Spotiflow-3D** as a RESIDUAL/ensemble proposal source (NOT a replacement for DoG).
   - Sparse-label handling is an ABLATION, not a mandated design: compare Linajea soft-mask (weight 1
     within nucleus radius, ~0.01-1e-6 else) vs zero-weight/ignore vs trusted-negatives vs teacher
     pseudo-positives, normalizing loss by pos/neg mass ("soft negative is still negative"). Pick by CV.
   - Optional CPV aux head (regress voxel->center) ONLY if crowded-region merges show in the exact metric.
2. **Pretrain on public same-domain data** (ZSNS embryos + Ultrack 522-frame volume + unet-daxi/simview
   weights) for invariance; fine-tune on the 2 competition embryos. Use public data for PRIORS/pretraining,
   never to pick a single threshold.
3. Feed Spotiflow proposals into the Phase-1 arbitration (union with DoG), not straight to submission.
GATE: min-fold +0.005; DoG-only recall must NOT fall; total inference <8h; reject if embryo-specific.

## PHASE 4 — Learned association + global solve (Week 6-9)  [+0.005-0.015]
Codex edge #6, plan #6/#10.
1. **Learned pairwise edge scorer** on FIXED arbitrated nodes: features = displacement, inherited velocity,
   local density, tissue-flow residual, intensity patches, short temporal context. Train on both embryos +
   public lineages (edge/division scorer only; detector frozen).
2. **Global solve - GATED on evidence, not by default:** only pursue ILP if, AFTER better nodes
   (Phase 1) and better edge costs (2.1), residual GLOBAL conflicts remain (measure: how many errors are
   local-greedy-fixable vs require joint optimization?). If they remain: V3 two-pass Hungarian vs learned
   min-cost-flow vs **motile ILP (SCIP, offline)** with `fit_weights` sSVM tuned to the EXACT metric
   (MaxChildren=2/MaxParents=1).
GATE: evidence of residual global conflict first; then min-fold +0.004 over Phase-1/2 linker; solver
p95 <8h; keep ILP only on clear bidirectional gain.

## PHASE 5 — Sparse, structurally-verified divisions (Week 8-11)  [0 to +0.03, high variance]
Codex edge #3. LAST, precision-first. Freeze the ordinary edge graph.
1. Count G per embryo; add forks INCREMENTALLY (one at a time on the exact combined metric).
2. Require: two daughters persisting >=2 frames; near-symmetric residual split; parent-daughter geometry
   consistent with ZSNS003 priors (daughter sep median 5.85um, parent->daughter median 3.07um).
3. Break-even: at J~0.82 edge-only fork needs 45% precision; with the 0.1 term + small G the first true
   division is very valuable. Keep a NO-DIVISION final candidate alongside.
GATE: division precision lower-CI >0.5 AND overall min-fold +0.003, else abstain.

## PHASE 6 — Robustness, ensemble, provenance, final selection (Week 10-13)
1. **Pseudo-domain robustness:** perturb intensity/blur/background/density/motion -> optimize WORST-CASE
   (2 embryos don't estimate the hidden-embryo distribution). Keep a simple robust submission always.
2. **Diversified finals:** the two final submissions must be materially different failure modes
   (e.g. DoG+arbitration+ILP vs Spotiflow-ensemble+learned-association), not two neighboring thresholds.
3. **Provenance (ONLY if host approved in Phase 0):** rigorous fingerprint (multi-projection phash + MI
   registration + temporal recurrence, vanishing false-match rate) -> transform public tracks into crop
   coords -> validate image-centroid agreement -> emit. Never combined with score probing.
4. **ITEC-style split/merge-then-retrack** as a final booster if Phase 1-4 leave crowded-region errors.
5. Lock reproducible offline notebook; dry-run at full test scale vs the 12h ceiling from day one.

---

## Submission strategy (CORRECTED - do NOT burn 5/day)
5/day is a ceiling, not a plan. Submitting daily invites public-LB (29%) fitting. Rule: **submit ONLY
locally-qualified hypotheses** that already cleared the nested-CV + cross-embryo gate. Most days = zero
submissions. Purpose of a submission is to (a) confirm no train/serve skew and (b) build the local<->public
correlation - and correlation needs SEVERAL MATERIALLY DIFFERENT systems over time, not one anchor and not
neighboring thresholds. Log every submission's full config + both-fold CV + count ratio + runtime.
Final 2 = diversified robust systems chosen by cross-embryo min-fold (bootstrap CI), never by public LB.

## Risk register
| Risk | Control |
|---|---|
| Public-LB overfit (29%) | embryo-held-out selection; diversified finals; keep a robust simple model |
| 2-embryo CV instability | pseudo-domain perturbations; optimize worst-case; report BOTH directions |
| "Soft negative is still negative" | compare mask schemes; normalize loss by pos/neg mass; teacher pseudo-labels |
| Detector overinvestment / assignment stealing | arbitration is the product, detector feeds it; gate on exact metric |
| Division FP damage | incremental, precision-first, keep no-division final |
| Runtime >12h | full-scale dry runs from day 1; gate proposals; profile early |
| Solver license/repro | SCIP/CBC only; log all weights/licenses/hashes |
| Provenance rules risk | written host ruling BEFORE any exact-label transfer; else priors/pretraining only |

## Immediate next 72h (Phase 0)
1. Fix normalization to precomputed quantiles -> re-score all 129 crops (nested-CV + cross-embryo).
2. DoG recall/threshold sweep -> quantify over-proposal headroom.
3. Submit V3 anchor to Kaggle (ONE submission) to check train/serve skew; do NOT start LB-fitting.
4. Send the PRIVATE provenance clarification to the organizer.
5. Build the stratified error-taxonomy dashboard + set up nested-CV + bootstrap harness.
Then start Phase 1 - but FIRST lock the corrected same-cell conflict-set definition (NOT 7um dedup).

## Red-team corrections applied (v2, 2026-07-02)
Independent review (8/10, "one potentially destructive detail") -> fixed BEFORE Phase 1:
- **7um is the evaluator gate, NOT a dedup radius** - conflict sets = same-cell duplicates (shared image
  mode/track), multiple real nuclei allowed within 7um. (Was destructive; now corrected in Phase 1.2.)
- **Redetection uses image/trajectory confidence at inference**, not "recovers an edge" (an offline oracle).
- **Validation** = nested grouped crop/time + bootstrap, freeze, then cross-embryo as unbiased check.
- **Submissions** = only locally-qualified; correlation needs several diverse systems, not a daily matrix.
- Linajea mask = ablation not mandate; public priors normalized for cadence/scale; ILP gated on residual
  global conflict; provenance = private clarification first.
