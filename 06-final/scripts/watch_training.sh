#!/bin/bash
# Watch the held-out detector training.
#   bash scripts/watch_training.sh          one-shot status
#   bash scripts/watch_training.sh -f       follow live
#
# The trainer writes tqdm progress with carriage returns, so a plain `tail -f`
# is unreadable -- everything here pipes through `tr '\r' '\n'` first.
LOG="$(dirname "$0")/../artifacts/training_live/train_heldout.log"
[ -f "$LOG" ] || { echo "no training log at $LOG"; exit 1; }

if [ "$1" = "-f" ]; then
  tail -f "$LOG" | tr '\r' '\n' | grep --line-buffered -E "test_loss=|iters: +[0-9]+%|Best score|Traceback|Error"
  exit 0
fi

echo "=== held-out detector training ==="
pgrep -f train_unet_transformer >/dev/null \
  && echo "status   : RUNNING (pid $(pgrep -f train_unet_transformer | head -1))" \
  || echo "status   : NOT RUNNING"
echo "log      : $LOG"
echo "started  : $(stat -f '%SB' "$LOG" 2>/dev/null)"
echo
echo "--- current epoch progress ---"
tr '\r' '\n' < "$LOG" | grep -oE "iters: +[0-9]+%[^]]*\]" | tail -1
echo
echo "--- completed epochs (held-out: 40 films the model never saw) ---"
tr '\r' '\n' < "$LOG" | grep -E "test_loss=" | tail -20
n=$(tr '\r' '\n' < "$LOG" | grep -cE "test_loss=")
echo "epochs done: ${n:-0}"
echo
echo "--- best checkpoint ---"
find "$(dirname "$LOG")" -name "edge_predictor_best.pth" -exec ls -lh {} \; 2>/dev/null | awk '{print "  "$NF"  "$5"  "$6" "$7" "$8}'
