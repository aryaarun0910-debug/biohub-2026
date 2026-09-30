#!/usr/bin/env bash
# Poll for a free Kaggle GPU slot, then push ONE kernel and exit.
# Kaggle allows 2 concurrent batch GPU sessions; a push fails with
# "Maximum batch GPU session count of 2 reached" when both are busy.
set -u
SLUG="$1"; DIR="kernels/$1"
for i in $(seq 1 96); do          # 96 x 5min = 8h ceiling
  if (cd "$DIR" && kaggle kernels push -p . 2>&1 | tee /dev/stderr | grep -q "successfully pushed"); then
    echo "PUSHED $SLUG at $(date -Is)"; exit 0
  fi
  sleep 300
done
echo "GAVE UP on $SLUG after 8h"; exit 1
