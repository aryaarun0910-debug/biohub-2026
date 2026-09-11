#!/usr/bin/env bash
# Are any Colab VMs running and burning compute units?
#   ./tools/colab_vms.sh          # report
#   ./tools/colab_vms.sh --kill   # terminate every running session
#
# An idle A100 costs ~15 units/hour. Measured on this account: ~1.4-1.65 units/h
# on 6h sessions. `colab run` tears down automatically; `colab new` does NOT.
set -uo pipefail
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
COLAB="$HOME/.local/bin/colab"
[ -x "$COLAB" ] || { echo "colab CLI not installed"; exit 2; }

out=$("$COLAB" --auth oauth2 sessions </dev/null 2>&1)
echo "$out"
if grep -qi 'No active sessions' <<<"$out"; then
  echo
  echo "  CLEAN - nothing running, nothing billing."
else
  echo
  echo "  *** SESSIONS ARE LIVE AND MAY BE BILLING ***"
  if [ "${1:-}" = "--kill" ]; then
    for s in $("$COLAB" --auth oauth2 sessions </dev/null 2>/dev/null | awk 'NR>2{print $1}'); do
      echo "  stopping $s"; "$COLAB" --auth oauth2 stop -s "$s" </dev/null 2>&1 | tail -1
    done
  else
    echo "  run with --kill to terminate them, or: colab stop -s <name>"
  fi
fi
echo
pgrep -af 'keep.?alive|colab.*daemon' | grep -v pgrep || echo "  no local keep-alive daemon"
