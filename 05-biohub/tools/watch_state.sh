#!/usr/bin/env bash
# Poll submissions and kernels; exit as soon as anything changes state.
# Exiting is the point -- a background command that finishes re-invokes the session.
set -u
sig() {
  kaggle competitions submissions -c biohub-cell-tracking-during-development 2>/dev/null \
    | grep -oE "SubmissionStatus\.[A-Z]+ +[0-9.]*" | head -3 | tr -d ' \n'
  for k in biohub-bigval; do
    kaggle kernels status "aryaarun07/$k" 2>/dev/null | grep -oE "KernelWorkerStatus\.[A-Z]+" | tr -d '\n'
  done
}
BASE="$(sig)"
echo "baseline: $BASE"
for i in $(seq 1 96); do          # 96 x 5min = 8h
  sleep 300
  NOW="$(sig)"
  if [ "$NOW" != "$BASE" ]; then
    echo "CHANGED after $((i*5)) min"
    echo "  was: $BASE"
    echo "  now: $NOW"
    exit 0
  fi
done
echo "no change in 8h"; exit 0
