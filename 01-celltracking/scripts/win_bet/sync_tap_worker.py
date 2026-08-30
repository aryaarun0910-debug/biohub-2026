#!/usr/bin/env python
"""Keep the Gate-1 replay worker and the kernel patch that ships it from drifting apart.

`scripts/win_bet/assoc_tap_replay.py` is the SINGLE SOURCE OF TRUTH for the worker. On Kaggle the
repository is not available to the notebook, so `scripts/kaggle_edits/assoc_tap_gate.py` must
carry the worker's text inside a string literal. Two copies of anything drift; this makes the copy
mechanical and `tests/test_assoc_feature_tap.py` fails the moment they disagree.

    python scripts/win_bet/sync_tap_worker.py --check     # non-zero if they have drifted
    python scripts/win_bet/sync_tap_worker.py --write     # regenerate the patch from the worker
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "win_bet" / "assoc_tap_replay.py"
GATE = ROOT / "scripts" / "kaggle_edits" / "assoc_tap_gate.py"

_LITERAL = re.compile(r"(_AFTG_WORKER = r" + "'''" + r")(.*?)(" + "'''" + r")", re.S)


def rendered(gate_path: Path = GATE, worker_path: Path = WORKER) -> str:
    """The gate patch as it must look with the worker's current text embedded."""
    worker = worker_path.read_text(encoding="utf-8")
    if "'''" in worker:
        raise SystemExit(
            f"{worker_path} contains a triple single-quote and cannot be embedded in the "
            "r''' literal the kernel patch uses. Use double-quoted docstrings there."
        )
    if worker.endswith("\\"):
        raise SystemExit(f"{worker_path} ends with a backslash and would escape the r''' close.")
    text = gate_path.read_text(encoding="utf-8")
    if len(_LITERAL.findall(text)) != 1:
        raise SystemExit(f"{gate_path}: expected exactly one _AFTG_WORKER literal")
    return _LITERAL.sub(lambda m: m.group(1) + worker + m.group(3), text, count=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate the patch from the worker")
    ap.add_argument("--check", action="store_true", help="exit non-zero if they have drifted")
    args = ap.parse_args(argv)
    want = rendered()
    have = GATE.read_text(encoding="utf-8")
    if args.write:
        if want != have:
            GATE.write_text(want, encoding="utf-8", newline="")
            print(f"synced {GATE} from {WORKER}")
        else:
            print("already in sync")
        return 0
    if want != have:
        print(f"DRIFT: {GATE} does not carry the current {WORKER}. "
              "Run: python scripts/win_bet/sync_tap_worker.py --write", file=sys.stderr)
        return 1
    print("in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
