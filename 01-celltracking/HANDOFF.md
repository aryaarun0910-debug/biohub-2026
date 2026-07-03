# HANDOFF — start here for a new session

**Updated:** 2026-07-03. **Repo:** github.com/aryaarun0910-debug/Biohub-CellTracking-2026 (private).
Read this first, then `reports/journal/JOURNAL.md` (full dated trail) and `reports/SPRINT_2026-07-03.md`.

## TL;DR  (STRATEGIC PIVOT 2026-07-03 evening — read this)
Kaggle "Biohub Cell Tracking During Development" (zebrafish 3D+t, deadline 29 Sep 2026). Our classical
DoG line (V3 anchor **0.807 public**; op_bright_smooth) is **median-tier** and NOT the way up. The whole
upper leaderboard runs the **organizer's own learned baseline** = the `tracking_cellmot` package ALREADY
vendored here (`vendor/kaggle-cell-tracking/src/tracking_cellmot/`: TemporalUNet3D detector +
SimpleNodeTransformer edge model + SCIP-backed division-native ILP via tracksdata). Public 50-epoch
weights are attachable OFFLINE (~0.85). Compute = **Kaggle T4x2 (free) only**. See
[[biohub-leaderboard-reframe]], reports/research/gap_competition_intel_2026-07-03.md, and the Codex
review reports/codex_plan_review_response_2026-07-03.md (accepted).

**Two facts that reshape everything (Codex, verified):** (1) the live public LB (top **0.896**) is a
**leakable 4-movie board** — all four visible `test/` IDs also exist labeled under `train/` (public
notebooks call them `TEST4`), so public score is a DEPLOYMENT check, not evidence of hidden-embryo
generalization. (2) The ~0.856 public stack ALREADY contains detector + ILP + gaps + divisions, so those
are NOT additive gains to bolt on. **The sole optimization target is a clean full-metric embryo-held-out
LEARNED OOF baseline**, not the public board.

## Scores / baselines (exact edge metric, embryo-held-out, full 199 crops; min-fold = the gate)
| System | 44b6 (16% edge wt) | 6bba (84% edge wt) | min-fold | edge-weighted | LB |
|---|---|---|---|---|---|
| V3 (anchor) | 0.632 | 0.756 | 0.632 | 0.736 | **0.807** (measured) |
| op_bright (199) | 0.684 | 0.751 | 0.684 | 0.741 | — (never cleanly submitted) |
| op_bright_smooth (199) | 0.700 | 0.762 | 0.700 | 0.752 | — |
NOTE (red-team #2): the earlier "op_bright ~0.812" / "op_bright_smooth 0.734/0.791" numbers were STALE
(20-crop subset / projection). The op_bright Kaggle submission returned **0.727 = a DEPLOYMENT FAILURE**
(isotropic-coords x anisotropic-SCALE bug, now fixed in the notebook, commit 2fa4e64), NOT a model score.
Do NOT trust the local<->LB +0.07 offset (n=1). LB is EDGE-WEIGHTED (6bba dominates) AND public-only =
4-movie leakable (see TL;DR) — treat it as a serve check, report clean OOF separately on every candidate.
(v3 / v3_smooth 199 CSVs being regenerated; a 20-crop screen had clobbered the 199 record — red-team #1.)

## Open async threads (check these on resume)
- **op_bright submission 54301967 resolved to 0.727 = broken artifact** (coord bug). Fixed notebook committed
  (2fa4e64); NOT re-submitted. Classical line is now a fallback, not the main thrust.
- **v3 / v3_smooth --all 199 re-runs** in progress (regenerating clobbered CSVs; settles op_bright's marginal
  value over free smoothing — red-team #5). Low priority: do NOT let it delay the learned stack.
- **Spotiflow zero-shot screen** ERRORED, unresolved. Superseded — the learned lever is the organizer stack.

## The restructured plan (Codex-reviewed 2026-07-03, ACCEPTED — supersedes the old classical ladder)
Order is deliberate; do NOT skip to retraining or bolt-on post-processing.
0. **Rules gate = CLEARED.** Official rules permit External Data + free/equally-accessible public models.
   Record provenance (URL, checksum, license, artifact code, env) for winner reproducibility. Caveats:
   PAC-MAP is NonCommercial (prize risk, avoid without written permission); NIS3D CC-BY is clean; Zebrahub
   needs a data-license check AND is same-lab -> possible private-test leakage (do NOT transfer its labels).
1. **Reproduce the public 50ep organizer stack UNCHANGED** as an offline notebook = infra/calibration only.
   Load with the support pack's OWN pinned repo/config/checkpoint first (it ships older `biohub_tracking`;
   port to vendored `tracking_cellmot` only after tensor/output parity). Smoke test offline install ->
   1 video -> all 4 visible movies; record runtime/peak-mem/checkpoint SHA/counts; locally score those 4
   with `biotrack.metric` and confirm local<->public parity. Submit ONE unchanged reproduction. Do NOT tune
   from its public score. Go-gate: exact artifact loaded, complete output, <9h projected hidden, parity understood.
2. **Use the metric that already exists; HARDEN it (do not build a second evaluator).** `biotrack.metric`
   wraps the organizer `evaluate` WITH the count adjustment (organizer `evaluate_datasets()` omits the count
   penalty -> not the exact gate; the wrapper is safer). Add dense-crop parity (tie/collision frames, red-team
   #10), real division fixtures, pin the evaluator commit independent of the model artifact, and emit
   edge TP/FP/FN + adjusted edge J + count ratio + division TP/FP/FN/J + final score on every run.
3. **Establish a clean learned OOF baseline = THE sole optimization target** (the one change Codex fought for).
   Get fold-specific public checkpoints if they exist; else train short organizer fold models now:
   Fold A train 6bba / eval all 44b6; Fold B train 44b6 / eval all 6bba; freeze inference params before the
   opposite embryo. NEVER gate on the first-10 alphabetical screen (red-team #3) — use full folds or a fixed
   stratified screen (embryo x density x intensity x depth x time). Public LB is only a serve check.
4. **Tune the EXISTING ILP + division path before integrating anything new** (motile is redundant — the
   organizer script already uses SCIP-backed division-native `tracksdata.solvers.ILPSolver`). Factorial
   ablations on FROZEN learned detections: greedy vs ILP; det-threshold/count trajectory; ILP division-weight
   {off,1.0,0.8,0.4,0.2}; 1-frame gap recovery off/on; division geometry filter off/on; only then micro safe
   post-hoc divisions / gap2. Divisions are NOT "free money" (a false fork adds div-FP AND edge-FP); ~82
   annotated divisions in the 2 embryos -> realistic div_J 0.12-0.27 = +0.012-0.027. Measure the learned
   baseline's EXISTING division TP/FP/FN first.
5. **Retrain for score ONLY after the clean-baseline error decomposition**, chosen from measured residuals:
   no-candidate-dominated -> NIS3D pretrain / longer training / detector ensemble; association-dominated ->
   link-cost training / wider temporal window; count-dominated -> image-feature count regressor (est_n is
   HIDDEN at inference); division-dominated -> division-aware loss + ILP cost calibration.
RULE: bank no delta until it improves the CLEAN embryo-held-out full metric; report public_TEST4 and OOF separately.

## DO-NOT (prize/DQ risk)
- **Public LB is a leakable 4-movie board** (visible `test/` IDs = labeled `train/` copies). Do NOT optimize
  against the public score or believe it as generalization evidence — optimize the clean OOF, report both.
- **Quarantined evaluator defect**: a division-scoring loophole (distant unmatched fork in a weakly-
  connected component qualifies a GT division, dodges division-FP). DO NOT submit/build around it without
  written host clearance. (reports/codex_lateral_sweep_results_2026-07-03.md sec "Quarantine".)
- No private-label reconstruction via submissions; no ToS breaches. External Data + free public models ARE
  allowed (rules verified, cleared). PAC-MAP = NonCommercial (avoid in prize submission); NIS3D CC-BY = clean.
- **Do NOT identify a private test crop and transfer public-source trajectory labels** (Zebrahub/March-22
  provenance) without written host approval. Broad pretraining on public data != label transfer into the test.

## How to run (all from repo root; use .venv = Python 3.12)
```
.venv/Scripts/python.exe scripts/build_inventory.py            # per-crop stats from local geffs -> reports/inventory/embryo_stats.csv
.venv/Scripts/python.exe scripts/build_splits.py               # leave-one-embryo-out -> data/dataset_splits.json
.venv/Scripts/python.exe scripts/run_phase1_ablation.py --config <v3|op_bright|v3_smooth|op_bright_smooth> --all --workers 7
.venv/Scripts/python.exe scripts/run_v3_taxonomy.py --all --workers 7   # 3-way edge FN taxonomy -> v3_taxonomy.csv
.venv/Scripts/python.exe -m pytest tests/ -q                   # 13 tests (metric, scoring, arbitrate guardrails)
.venv/Scripts/python.exe scripts/make_figures.py && scripts/make_animation.py  # journal figures/gifs
```
Add a new pipeline: register a config fn in scripts/run_phase1_ablation.py CONFIGS, gate on min-fold.

## Key code (src/biotrack/)
- `metric_numpy.py` — EXACT edge metric in pure numpy (validated == organizer incl. adversarial cases).
  EDGE-ONLY; divisions via `metric.py` (tracksdata harness). `score_sample`, `gt_candidate_within`.
- `propose.py` — multiscale-DoG over-proposal (loose NMS = recall lever) + features + tensorstore reader.
- `arbitrate.py` — same-cell conflict sets (complete-linkage + diameter cap, NO chain-bridging) + guardrails.
- `submission.py` — lossless graph <-> submission.csv. `cache.py` — candidate cache.
- Kaggle notebooks: notebooks/kaggle_dog_infer.py (V3 anchor), notebooks/kaggle_op_bright/ (op_bright, pushed).
  NOTE: Kaggle image has tensorstore, NOT zarr -> use the tensorstore reader; training notebooks may use internet.

## Environment
- `.venv` (Python 3.12): tracking-cellmot(git) + tracksdata + geff + polars + CPU torch + scipy + skimage +
  spotiflow + matplotlib + kaggle. Default `py`=3.14 (too new for tracksdata) -> always use `.venv`.
- Full 87GB data local at data/train (199 crops: 44b6=71, 6bba=128) + data/test. Never re-download.
- Kaggle user `aryaarun07`; kernels: biohub-v3-anchor, biohub-op-bright. GitHub `aryaarun0910-debug`.
- Local GPU MX350 (2GB) cannot train. Commit + push after every change (standing directive).

## Immediate next (resume order — restructured plan steps 1-4)
1. Mount the public 50ep support pack + its pinned repo; offline smoke test -> reproduce ONE unchanged
   submission (plan step 1). Confirm local(4-movie)<->public parity. Do not tune from it.
2. Harden `biotrack.metric` (dense-crop parity + division fixtures; emit full metric breakdown) — step 2.
3. Stand up the clean embryo-held-out learned OOF baseline (fold A/B), make it the sole target — step 3.
4. Factorial ablation of the EXISTING ILP/division/gap/threshold knobs on frozen detections — step 4.
(Classical v3/op_bright_smooth = fallback only; finish the 199 re-runs but don't let them delay the above.)
```
