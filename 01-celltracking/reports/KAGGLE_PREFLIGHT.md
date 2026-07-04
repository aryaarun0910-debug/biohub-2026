# Kaggle kernel PRE-FLIGHT — validate locally on CPU before spending GPU quota

Rule (2026-07-04, after burning several GPU sessions on bugs catchable offline): **before pushing any
Kaggle GPU kernel, run its full code path locally on CPU.** Pushing auto-runs and costs a session; each
of these was found and fixed on CPU for free — ModuleNotFoundError (PYTHONPATH), the checkpoint-selection
bug (best saturates at epoch 0), CLI-flag typos, the pool_kernel_um patch target, the corrupt-crop regex.

## What we CAN validate locally (no GPU, no Kaggle)
The local machine has the metric stack (.venv, CPU torch, tracksdata, geff, zarr) and the full 199-crop
data with `image_statistics.quantiles` attrs — enough to actually RUN the pack's train + predict scripts.

Setup: download the full pack once (`kaggle datasets download pilkwang/biohub-tracking-support-pack-50ep-v1
--unzip -p <dir>`), put `repo/src` + `repo/scripts` on `sys.path`, set `BIOHUB_DATA_DIR=data/train`, and
monkeypatch `torch.cuda.synchronize = lambda *a, **k: None` (the trainer calls it unconditionally).

Reference drivers: `scripts/kaggle_preflight/dryrun_train.py`, `scripts/kaggle_preflight/dryrun_predict.py`.

## Checklist (all PASSED for the current train + predict kernels)
1. `py_compile` the two kernel scripts AND the pack scripts AFTER applying our string-patches
   (trainer last-epoch save; predict pool_kernel_um 3.0->5.0). Assert each patch anchor exists.
2. Cross-check every CLI flag we pass vs the script's `argparse`.
3. Run the split-generation logic on the real 199 crops (no overlap, held-out embryo excluded, counts).
4. Run the corrupt-crop regex against the real `corrupt_files.txt`.
5. **Run the patched trainer** (1 epoch, `--max-iters 2`, 3 crops, `max_frames=40`, CPU): imports,
   open_dataset/quantiles, windows, model fwd/bwd, det+edge loss, eval, checkpoint save. Assert BOTH
   `edge_predictor_best.pth` and `edge_predictor_last.pth` are written (our patch) and config is correct.
6. **Run predict** (1 crop, det_threshold 0.99, greedy, CPU): weight load, `pool_kernel_um==5.0`,
   detection/linking/graph-build, geff save.

## Two local-only failures that DO NOT affect Kaggle (confirmed, not hand-waved)
- **Windows `MAX_PATH` (260 char)** on geff save: zarr's atomic `.partial` temp file exceeds the limit
  under a deep scratchpad path -> `FileNotFoundError`. Linux/Kaggle has no such limit and short paths.
  Reproduced + resolved by re-running with a short `PREDICTIONS_PATH` (geff saved fine).
- **`Cannot access accelerator device when none is available`** in the pack's `--evaluate`: the pack's
  eval path needs a GPU present; a no-GPU machine errors. Kaggle has a GPU. (Our `biotrack.metric` scorer
  does NOT need one — it runs the same match on CPU.) The pack's `evaluate_run` also wraps each crop in
  try/except -> a per-crop failure yields a `nan` row, never a crash.
- Both above were triggered together by the 2-iteration UNTRAINED toy model detecting 0 cells (empty
  geff). The real trained model produces non-empty graphs, so neither path is exercised on the real run.

## Scoring the OOF (the robustness win)
Do NOT trust only the kernel's `--evaluate` number. The predict kernel exports the predicted geffs to
`/kaggle/working/pred_geffs_split_{FOLD}`. Download them and score with our hardened, exact,
CPU/GPU-agnostic `biotrack.metric.score_many([(pred_geff, gt_geff), ...])` — that is the authoritative
clean OOF (edge adj-J + 0.1 div-J, count-adjusted, numpy<->authoritative parity verified). Guard empty
pred graphs (charge all-FN) when scoring locally.
