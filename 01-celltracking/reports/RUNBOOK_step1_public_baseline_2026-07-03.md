# RUNBOOK — Step 1: reproduce the public 50ep organizer stack UNCHANGED (calibration submission)

Goal: put a strong, HONEST calibration point on the board with zero training, using the public
CC0 support pack, loaded with its OWN pinned code. This is an infra/calibration submission — do NOT
tune from its public score (the public board is a leakable 4-movie set; see HANDOFF DO-NOT).
Grounded in the pack's real `repo/scripts/predict_unet_transformer.py` (read 2026-07-03).

## Artifact (verified)
- Kaggle dataset: `pilkwang/biohub-tracking-support-pack-50ep-v1` — **License CC0-1.0** (clean).
- Ships its own `repo/` (`biohub_tracking` package + scripts), offline wheels, and weights:
  - detector+transformer: `weights/unet_transformer/split_0/edge_predictor_best.pth`
    (sha256 `912ae91a4077b65cc1933e6f37b9deb15268e483130b51555a490a8179ab3f53`, ~8.36MB),
  - plus `checkpoint_last.pth` (~25MB) and `config.json`.
- Model config (config.json / manifest): UNet layers [32,64,128], out 32, downsample [1,4,4],
  window_size 2, `pool_kernel_um 5.0`.
- Provenance GAP (Codex, confirmed): the manifest records arch + weight SHA but **no epoch, training
  set, val split, or clean held-out score**. Single `split_0`. So this proves DEPLOYMENT, not
  hidden-embryo generalization — that is what step 3 (clean OOF baseline) is for.

## ⚠ Config subtlety to verify before trusting the score
`predict_unet_transformer.py`'s `PredictConfig.pool_kernel_um` defaults to **3.0** and is NOT read from
`config.json` and NOT exposed on the CLI — but the artifact `config.json` says **5.0**. `pool_kernel_um`
sets the detection local-max suppression distance (µm), so 3.0 vs 5.0 changes detection density/count and
therefore the count-penalty and edge score. Before the calibration submission, confirm which value the
reference submission notebook used (the pack points to `submission/submission_22_*.ipynb`); if it used
5.0, set it (edit the `PredictConfig` default or the notebook) so the reproduction is faithful. Flag this
as the first thing to check — it is the most likely silent train/serve gap.

## CLI (real, from the script)
```
python repo/scripts/predict_unet_transformer.py \
    --split 0 \
    --data-dir <DIR with the .zarr videos> \
    --weights <.../weights/unet_transformer/split_0/edge_predictor_best.pth> \
    --splits  <dataset_splits.json>        # only needed unless --debug-video is used \
    --det-threshold 0.99                    # script default; detector is high-precision on sparse GT \
    --use-ilp --ilp-division-weight 1.0     # SCIP ILP: global, division-native (Codex: already present) \
    --evaluate                              # score vs GT with the pack's own summarise (full breakdown)
```
Notes: `--debug-video <one .zarr>` ignores fold/splits and runs a single video (use for the smoke test).
`--slice :1` runs only the first video of the fold. Without `--use-ilp` it does greedy linking with
max_children=2 (divisions allowed) / max_parents=1. Detection = TemporalUNet3D logits → max-pool local
maxima > `det_threshold`, with flip-XY TTA (Z excluded — anisotropy). Edges = transformer, softmax over
t+1, threshold 0.5. `--use-ilp` replaces greedy with `td.solvers.ILPSolver` (needs pyscipopt/SCIP).

## Execution order (Kaggle, GPU T4x2, internet OFF)
1. **Offline env.** Mount the support pack. Install its deps offline from the bundled wheels, or use the
   given command (NOTE its warning — do not quote `zarr>=3.0.10,<4`):
   `pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy`
2. **Load with the pack's OWN code first** (`repo/src` on `sys.path`), NOT our vendored `tracking_cellmot`.
   Port to the vendored code only AFTER tensor/output parity (Codex).
3. **Smoke test:** `--debug-video <one test .zarr>` → confirms install, weight load, one-video inference.
   Record: runtime, peak GPU mem, checkpoint SHA, node/edge/division counts, artifact-resolution path.
4. **Local↔public parity:** the 4 visible `test/` IDs (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`,
   `6bba_05db0fb1`) also exist under `train/` WITH labels. Run predict on the **train copies** with
   `--evaluate` (or score the saved geffs with our hardened `biotrack.metric.score_many`) → the local
   score should track the Kaggle public score closely. If they diverge → STOP, diagnose train/serve
   (likely pool_kernel_um or normalization) before submitting.
5. **Full test inference:** run on all `test/` `.zarr` → save geffs → convert to `submission.csv`
   (the pack's reference submission notebook does geff→CSV; our `biotrack.submission.graphs_to_submission`
   also works: load each predicted geff with `biotrack.metric.load_graph`, then `graphs_to_submission`).
6. **Submit ONE unchanged reproduction.** Do not tune from the public score.

## Go-gate (Codex) before moving on
Exact artifact loaded (SHA matches), output complete, **<9 h** projected hidden-test inference,
local↔public parity understood. Record dataset URL + checksum + license + artifact code + env for
winner reproducibility.

## Local scoring bridge (already built + hardened this session)
`biotrack.metric.score_many([(pred_geff, gt_geff), ...])` → run summary with the EXACT gate (edge count
adjustment included; organizer `evaluate_datasets()` omits it and is NOT the gate). It now emits the full
breakdown incl. division TP/FP/FN/J (metric hardening committed 21c6f3a; numpy↔authoritative parity
verified on real crops). Use it to score the 4 TEST4 train copies and any OOF fold in step 3.

## After step 1
Go to plan step 2 (metric hardening — DONE) and step 3 (clean embryo-held-out learned OOF baseline =
the sole optimization target). See HANDOFF.md "restructured plan".
