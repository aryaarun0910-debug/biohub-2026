"""Compare paired transform deltas across old and replacement screening substrates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(old: dict, new: dict, tolerance: float) -> dict:
    if old["params_sha256"] != new["params_sha256"]:
        raise ValueError("sentinel parameter hashes differ")
    if old["n_crops"] != new["n_crops"]:
        raise ValueError("sentinel crop populations differ")
    keys = ("score", "adj_edge_jaccard", "division_jaccard")
    differences = {key: float(new["delta"][key] - old["delta"][key]) for key in keys}
    return {
        "params_sha256": old["params_sha256"],
        "n_crops": int(old["n_crops"]),
        "old_delta": {key: float(old["delta"][key]) for key in keys},
        "new_delta": {key: float(new["delta"][key]) for key in keys},
        "difference": differences,
        "tolerance": float(tolerance),
        "equivalent": all(abs(value) <= tolerance for value in differences.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("old_f0", "new_f0", "old_f1", "new_f1"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {"fold0": compare(json.loads(args.old_f0.read_text()),
                               json.loads(args.new_f0.read_text()), args.tolerance),
              "fold1": compare(json.loads(args.old_f1.read_text()),
                               json.loads(args.new_f1.read_text()), args.tolerance)}
    result["equivalent_both_folds"] = all(row["equivalent"] for row in result.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SUBSTRATE_SENTINEL equivalent_both_folds={result['equivalent_both_folds']} out={args.out}")
    return 0 if result["equivalent_both_folds"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
