#!/bin/bash
# Train a detector with 40 films HELD OUT, so post-processing can be evaluated
# on films the model never saw.
#
# WHY: docs/HANDOFF.md -- every offline tier scored against train/ annotations,
# which are the detector's own TRAINING TARGETS (1.63% of nodes). s05 predicted
# +0.022 and scored 0.945. This is the only honest local instrument available.
#
# The held-out 40 deliberately INCLUDE the 4 scored films, so the instrument can
# be calibrated against the one hard fact we own: relink OFF is worse by ~0.002.
#
# Usage: bash scripts/train_heldout.sh [epochs] [max_iters]
set -e
EPOCHS=${1:-50}; MAXIT=${2:-}
REPO=/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/repo
source /tmp/pipe_env.sh
export BIOHUB_WORKING_DIR="/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/run/work"
export PYTHONPATH="$REPO/src:$REPO/scripts"
cd /private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/run
EXTRA=""
[ -n "$MAXIT" ] && EXTRA="--max-iters $MAXIT"
/private/tmp/claude-501/-Users-aryaarun-Developer-biohub26/f87a5df4-927a-4e79-9911-13f800866a92/scratchpad/pipeenv/bin/python $REPO/scripts/train_unet_transformer.py \
  --data-dir "/Users/aryaarun/Developer/biohub26/data/biohub-cell-tracking-during-development/train" \
  --splits splits_heldout.json --split 0 \
  --epochs $EPOCHS --single-gpu --num-workers ${NW:-4} --batch-size ${BS:-8} --lr ${LR:-1e-4} --window-size 2 --pool-kernel-um 5.0 \
  --method heldout40 $EXTRA
