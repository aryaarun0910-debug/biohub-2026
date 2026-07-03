# HANDOFF — start here for a new session

**Updated:** 2026-07-03. **Repo:** github.com/aryaarun0910-debug/Biohub-CellTracking-2026 (private).
Read this first, then `reports/journal/JOURNAL.md` (full dated trail) and `reports/SPRINT_2026-07-03.md`.

## TL;DR
Kaggle "Biohub Cell Tracking During Development" (zebrafish 3D+t, deadline 29 Sep 2026). We reproduced
the public V3 DoG pipeline (anchor = **0.807 public**), then built a differentiated pipeline
(over-propose -> same-cell dedup -> V3 link -> temporal smoothing) that improves the exact
embryo-held-out metric. Compute is **Kaggle T4x2 (free) only** (no A100 budget). Target 0.88;
honest odds **~15-25%** for 0.88+, realistic **0.85-0.88**. The classical ceiling is ~0.854 (public V11
temporal smoothing); breaking past it needs a learned detector (dense external pretraining).

## Scores / baselines (exact metric, embryo-held-out; min-fold = the gate)
| System | 44b6 (16% edge wt) | 6bba (84% edge wt) | min-fold | edge-weighted | LB |
|---|---|---|---|---|---|
| V3 (anchor) | 0.632 | 0.756 | 0.632 | 0.736 | **0.807** |
| op_bright (199) | 0.684 | 0.751 | 0.684 | 0.741 | ~0.812 (submitted, pending) |
| op_bright_smooth (20-crop) | 0.734 | 0.791 | 0.734 | ~0.78 | 199 pending |
Local<->LB offset ~ +0.07 (weighted-local -> public). LB is EDGE-WEIGHTED, so 6bba (dense) dominates;
gains must move 6bba to move the LB. op_bright helped only 44b6 (small wt); SMOOTHING helps both.

## Open async threads (check these on resume)
- **Submission 54301967 (op_bright)** = PENDING. Check `kaggle competitions submissions -c biohub-cell-tracking-during-development`.
- **op_bright_smooth --all 199** running (bg log reports/phase1_obs_run.log; result -> reports/inventory/phase1_op_bright_smooth.csv).
  This is the first real EDGE-WEIGHTED number for the smoothing win -> compute weighted mean + project LB.
- **Kaggle Spotiflow zero-shot screen** (aryaarun07/biohub-spotiflow-zeroshot-screen) ERRORED (empty log,
  likely numpy-2.0/pip conflict); hardened + re-pushed but unresolved. Low priority (zero-shot out-of-domain).

## The corrected path to 0.88 (Codex-reviewed, honest)
1. op_bright (submitted) ~0.81.
2. **+ temporal smoothing (V11 lever, DONE, confirmed both folds up)** -> ~0.84-0.855. CONFIRM ON 199 next.
3. + non-oracle count calibration (est_n is HIDDEN at inference; unproven; ~+0.003-0.010).
4. Reproduce FULL public V11 config -> validated ~0.854 classical base.
5. **Learned detector over a 0.85 base** (NIS3D 3.3GB dense zebrafish / DAXI, nnPU / Linajea soft-mask,
   Kaggle T4x2 one-fold-per-GPU, fp16, grad-ckpt) -> ~0.865-0.885. THE ceiling-breaker; run GPU concurrently.
6. Sparse precision-gated divisions (0-0.01, not bankable).
RULE: bank no delta until it improves the exact edge-weighted metric on BOTH folds (bootstrap CI).

## DO-NOT (prize/DQ risk)
- **Quarantined evaluator defect**: a division-scoring loophole (distant unmatched fork in a weakly-
  connected component qualifies a GT division, dodges division-FP). DO NOT submit/build around it without
  written host clearance. (reports/codex_lateral_sweep_results_2026-07-03.md sec "Quarantine".)
- No private-label reconstruction via submissions; no ToS breaches. Public external data for model
  development IS allowed (rules verified).
- Exact test-crop -> public-label transfer (March-22 provenance) = NEEDS-RULES-CHECK (private host Q first).

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

## Immediate next (resume order)
1. Read op_bright_smooth 199 result -> weighted mean -> LB projection; if it beats op_bright, push+submit it.
2. Reproduce full V11 config (temporal smoothing + any V11 params) -> validated 0.854 base.
3. Start GPU dense-external pretraining (NIS3D/DAXI) concurrently on Kaggle T4x2.
4. Non-oracle count calibration experiment (both-fold gate).
```
