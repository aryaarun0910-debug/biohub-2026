"""Extract scorer-authoritative per-crop node estimates from GEFF metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import estimated_nodes  # noqa: E402


def extract_estimates(
    gt_dir: Path,
    *,
    reader: Callable[[str], float] = estimated_nodes,
) -> dict[str, int]:
    paths = sorted(gt_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(f"no GEFF files under {gt_dir}")
    estimates: dict[str, int] = {}
    for path in paths:
        if path.stem in estimates:
            raise ValueError(f"duplicate crop stem: {path.stem}")
        value = float(reader(str(path)))
        if not value.is_integer() or value <= 0:
            raise ValueError(f"{path.name}: invalid estimated_number_of_nodes={value!r}")
        estimates[path.stem] = int(value)
    return estimates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    estimates = extract_estimates(args.gt_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(estimates, indent=2) + "\n", encoding="utf-8")
    print(
        f"NODE_ESTIMATES crops={len(estimates)} total={sum(estimates.values())} "
        f"out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
