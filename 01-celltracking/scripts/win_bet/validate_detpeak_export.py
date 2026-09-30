"""Fail-closed validation for a DetPeak export and its observational graph parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path)


def validate(
    sidecars: list[Path],
    smoke_csv: Path,
    control_csv: Path,
    manifest: Path,
) -> dict:
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("graph_unchanged") is not True:
        raise ValueError("manifest does not declare graph_unchanged=true")
    if not sidecars:
        raise ValueError("no sidecars supplied")

    details = {}
    for path in sidecars:
        with np.load(path, allow_pickle=False) as sidecar:
            required = {"t", "zyx", "logit", "pipeline_threshold", "pipeline_peak_count"}
            missing = required - set(sidecar.files)
            if missing:
                raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
            logits = sidecar["logit"].astype(np.float64)
            threshold = float(sidecar["pipeline_threshold"])
            recorded = int(sidecar["pipeline_peak_count"])
            if len(logits) == 0:
                raise ValueError(f"{path.name}: empty sidecar")
            reconstructed = int(np.count_nonzero(1.0 / (1.0 + np.exp(-logits)) > threshold))
            if reconstructed != recorded:
                raise ValueError(f"{path.name}: pipeline peak count mismatch")
            details[path.stem] = {
                "exported_peaks": int(len(logits)),
                "pipeline_threshold": threshold,
                "pipeline_peak_count": recorded,
            }

    names = sorted(details)
    smoke = _read_csv(smoke_csv).filter(pl.col("dataset").is_in(names)).drop("id")
    control = _read_csv(control_csv).filter(pl.col("dataset").is_in(names)).drop("id")
    order = [column for column in smoke.columns]
    smoke = smoke.sort(order)
    control = control.select(order).sort(order)
    if not smoke.equals(control):
        raise ValueError("observational parity failed: smoke graph differs from P28 control")
    return {
        "schema_version": 1,
        "sidecars": details,
        "graph_rows": smoke.height,
        "observational_graph_parity": True,
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecars", nargs="+", type=Path, required=True)
    parser.add_argument("--smoke-csv", type=Path, required=True)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.sidecars, args.smoke_csv, args.control_csv, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"DETPEAK_EXPORT_VALIDATION pass={result['pass']} sidecars={len(result['sidecars'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
