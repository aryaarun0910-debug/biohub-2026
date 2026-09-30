#!/bin/bash
# Generate DEPLOYED-QUALITY ILP graphs for all 199 train films, locally on MPS.
#
# Verified bit-identical to the Kaggle T4 output on 44b6_341df25f: same node
# count, max coordinate difference 0, identical edge sets. So these are not a
# rebuild -- they are the same graphs the kernel produces, at 96 s/film against
# the T4's ~236 s.
#
# Two patches vs the shipped source, both in the scratch copy:
#   1. device: cuda -> mps fallback (the original falls back to CPU)
#   2. Path('/kaggle/working') -> $BIOHUB_WORKING_DIR
#
# Usage: bash scripts/local_predict_199.sh <shard_index> <n_shards>
set -e
SHARD=${1:-0}; NSHARD=${2:-1}
REPO=/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/repo
source /tmp/pipe_env.sh
export BIOHUB_SECONDARY_WEIGHTS="/Users/aryaarun/Developer/biohub26/weights/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"
export BIOHUB_WORKING_DIR="/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/run/work"
export PYTHONPATH="$REPO/src:$REPO/scripts"
cd /private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/run
/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/pipeenv/bin/python $REPO/scripts/predict_unet_transformer.py \
  --data-dir "/Users/aryaarun/Developer/biohub26/data/biohub-cell-tracking-during-development/train" \
  --splits splits_199.json --split 0 \
  --weights "/Users/aryaarun/Developer/biohub26/weights/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth" \
  --unet-batch-size 4 --det-threshold 0.965 \
  --ilp-edge-weight -1.0 --ilp-appearance-weight 0.0 \
  --ilp-disappearance-weight 2.0 --ilp-division-weight 1.2 --use-ilp \
  --method local199 --slice ${SHARD}::${NSHARD}
