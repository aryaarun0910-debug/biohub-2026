#!/usr/bin/env bash
# Refuse to spend Colab GPU units on code that has not passed the free tiers.
#
#   ./tools/colab_guard.sh preflight  <script.py>     # T0+T1, free, local
#   ./tools/colab_guard.sh parity     <script.py>     # T2, Colab CPU, ~free
#   ./tools/colab_guard.sh run --gpu A100 <script.py> # T3, PAID - blocked without a receipt
#
# The receipt is bound to the script's sha256, so passing preflight on one file cannot
# authorise launching a different one. Receipts expire after 6 hours.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECEIPTS="$ROOT/db/.receipts"; mkdir -p "$RECEIPTS"
TTL=$((6*3600))

die(){ printf '\n  BLOCKED: %s\n\n' "$1" >&2; exit 1; }
sha(){ python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

receipt_ok(){ # <sha> <tier>
  local f="$RECEIPTS/$1.$2"
  [ -f "$f" ] || return 1
  local age=$(( $(date +%s) - $(python3 -c 'import os,sys; print(int(os.stat(sys.argv[1]).st_mtime))' "$f") ))
  [ "$age" -ge 0 ] && [ "$age" -lt "$TTL" ] || return 1
}

case "${1:-}" in
  preflight)
    s="${2:?script required}"; h=$(sha "$s")
    echo "T0/T1 preflight on $(basename "$s")  [${h:0:12}]"
    rm -f "$RECEIPTS/$h.preflight" "$RECEIPTS/$h.parity"
    python3 "$ROOT/tools/preflight.py" "$s" \
      && touch "$RECEIPTS/$h.preflight" \
      && echo "  PASS - receipt valid 6h"
    ;;

  parity)
    s="${2:?script required}"; h=$(sha "$s")
    receipt_ok "$h" preflight || die "run preflight first: ./tools/colab_guard.sh preflight $s"
    command -v colab >/dev/null || die "colab CLI not on PATH"
    echo "T2 parity request: verify the CLI selected CPU and the script actually performs --parity-check"
    # A CPU session is the cheapest possible way to catch the failure class that
    # burned ~36 units across 5 failed jobs on the old relay: environment parity.
    rm -f "$RECEIPTS/$h.parity"
    colab run "$s" --  --parity-check && touch "$RECEIPTS/$h.parity" \
      && echo "  PASS - parity command exited successfully (verify remote report before GPU use)"
    ;;

  run)
    shift
    [ "$#" -ge 1 ] || die "script required"
    command -v colab >/dev/null || die "colab CLI not on PATH"
    s="${!#}"; h=$(sha "$s")
    receipt_ok "$h" preflight || die "no valid preflight receipt for $(basename "$s")"
    receipt_ok "$h" parity    || die "no valid Colab-CPU parity receipt for $(basename "$s"). \
The old relay burned ~36 units on 5 jobs that all failed on environment parity, not compute."
    echo "T3 PAID GPU RUN - both gates passed."
    echo "  Verify actual billing and remote teardown explicitly; receipts cover this script only."
    before=$(date -u +%FT%TZ)
    colab run "$@"
    echo "  started $before, finished $(date -u +%FT%TZ) - record actual units from the Colab UI"
    ;;

  status) ls -la "$RECEIPTS" 2>/dev/null | tail -n +4 || echo "  no receipts" ;;
  *) sed -n '2,9p' "$0" ;;
esac
