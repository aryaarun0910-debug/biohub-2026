# RUNBOOK — Step 3: clean embryo-held-out learned OOF baseline (THE sole optimization target)

Per Codex (accepted): the public single-`split_0` pack proves deployment, not generalization. To know
what actually transfers to the disjoint hidden embryo, train the organizer model embryo-held-out and
score it with the exact metric. This OOF number — not the leakable public board — is what we optimize.
Grounded in the pack's real `repo/scripts/train_unet_transformer.py` + `predict_unet_transformer.py`.

## Folds (already correct — no work needed)
`data/dataset_splits.json` is exactly the trainer's format (a 2-fold list of `{"train":[...],"test":[...]}`):
- **fold 0 (Fold A):** train = 128 6bba crops, test = 71 44b6 crops.
- **fold 1 (Fold B):** train = 44b6, test = 6bba.
Names are stems (`6bba_05b6850b`); `--data-dir data/train` resolves them to `.zarr`/`.geff`.

## Two VERIFIED traps (both from reading the code — do not skip)
1. **pool_kernel_um: train uses 5.0, predict DEFAULTS TO 3.0.** `train_unet_transformer.py` trains detection
   with `pool_kernel_um=5.0` (its default; also written into `config.json`), but
   `predict_unet_transformer.py`'s `PredictConfig.pool_kernel_um` defaults to **3.0** and is NOT read from
   config or exposed on the CLI. Inference with 3.0 changes detection density vs training → wrong count and
   score. **At prediction you MUST use 5.0** (edit `PredictConfig.pool_kernel_um` or the notebook). This
   confirms the step-1 flag — it is a real train/serve mismatch, not hypothetical.
2. **Best-epoch selection PEEKS at the held-out embryo.** `train()` evaluates on the test set (the held-out
   embryo) each epoch and saves `edge_predictor_best.pth` by `score = test_acc * test_recall` (lines
   1162-1176). With only 2 embryos there is no third split, so "best" is chosen by looking at the very
   embryo we call OOF → the reported OOF is OPTIMISTIC (model-selection leakage; red-team #9 made concrete).
   Also note that proxy (`acc*recall`) is NOT the competition metric.
   **Fix for a clean number:** carve a few crops out of the TRAIN embryo as a validation set for epoch
   selection (leave the held-out embryo untouched until final scoring), OR fix the epoch budget and report
   the last epoch. Then re-score the held-out embryo ONCE with `biotrack.metric` (the exact gate), never the
   `acc*recall` proxy.

## Train (Kaggle T4x2, one fold per session recommended)
```
python repo/scripts/train_unet_transformer.py \
    --split 0 \                     # then --split 1 (second session)
    --data-dir data/train \
    --splits data/dataset_splits.json \
    --epochs 50 --lr 1e-4 --batch-size 16 --num-workers 8 \
    --downsample 1,4,4 --window-size 2 --pool-kernel-um 5.0
    # --data-parallel is ON by default -> set Kaggle accelerator to "GPU T4 x2" (splits the UNet batch)
    # --unet-weights <nis3d_pretrained.pth>  # OPTIONAL, step-5 lever: strict=False UNet init
```
- Saves `WEIGHTS_PATH/unet_transformer/split_{fold}/edge_predictor_best.pth` + `config.json`.
- 50 epochs x 2 folds likely exceeds the 12h notebook limit together → run one fold per session, or reduce
  epochs / use `--max-iters`. Record per-epoch train/test time to project the budget.
- All crops in a fold must share spatial shape for `batch_size>1`; else set `batch_size=1` or group by shape.

## Predict the held-out embryo, then score with the EXACT gate
```
python repo/scripts/predict_unet_transformer.py \
    --split 0 \                     # predicts fold 0's TEST list (44b6) using split_0 weights
    --data-dir data/train --splits data/dataset_splits.json \
    --det-threshold 0.99 --pool-kernel-um-EQUIVALENT 5.0 \   # (set PredictConfig.pool_kernel_um=5.0) \
    --use-ilp --ilp-division-weight 1.0 \
    --evaluate                       # pack's own summarise; OR score the saved geffs with biotrack.metric
```
Then the authoritative OOF (our hardened exact gate, incl. divisions):
```
from biotrack.metric import score_many
res = score_many([(pred_dir/f"{name}.geff", TRAIN/f"{name}.geff") for name in fold0_test])
res["summary"]  # adj_edge_jaccard (weighted) + 0.1*division_jaccard = the OOF number
```
Report **both folds' OOF** and the min-fold. Freeze all inference params before running the opposite fold.

## What this yields
- A clean per-fold OOF (adj edge J + division J) that predicts the disjoint hidden embryo far better than
  the leakable public board. This becomes the selection gate for step 4 (ILP/division/threshold ablation on
  FROZEN detections) and step 5 (retrain from measured residuals: NIS3D pretrain via `--unet-weights`, etc.).
- A first read on whether the organizer architecture even generalizes across our 2 embryos — the question
  the public single-split pack cannot answer.

## Prereqs / env
Same offline deps as step 1 (`tracksdata pyscipopt geff ilpy zarr polars ...`). Load with the pack's own
`repo/` on `sys.path`. `dataspec.py` sets DATASET_PATH/WEIGHTS_PATH/PREDICTIONS_PATH — configure it or pass
`--data-dir/--splits/--weights` explicitly. See HANDOFF "restructured plan" steps 2-5.
