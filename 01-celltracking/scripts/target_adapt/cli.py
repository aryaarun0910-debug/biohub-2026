"""Adapt candidate-edge scores independently on one unlabeled target embryo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import AdaptConfig, adapt_candidates, read_edge_models, write_adapted, write_diagnostics


def _named_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--input must be NAME=PATH")
    return name, Path(path)


def _triple(value: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected z,y,x comma-separated numbers") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly three voxel sizes")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embryo", required=True, help="audit identifier; one embryo per invocation")
    p.add_argument("--input", action="append", required=True, type=_named_input, help="NAME=CSV (repeat per model)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--diagnostics", required=True, type=Path)
    p.add_argument("--default-p-fp", type=float, help="explicit OOF estimate when inputs lack p_fp")
    p.add_argument("--voxel-size-um", type=_triple, default=(1.625, 0.40625, 0.40625))
    p.add_argument("--high-score", type=float, default=0.72)
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--min-consensus-models", type=int, default=2)
    p.add_argument("--physical-gate", type=float, default=12.0, help="um/frame")
    p.add_argument("--path-gate", type=float, default=4.0, help="um/frame")
    p.add_argument("--min-pseudo", type=int, default=50)
    p.add_argument("--min-pseudo-frames", type=int, default=4)
    p.add_argument("--min-negatives", type=int, default=50)
    p.add_argument("--blend", type=float, default=0.35)
    p.add_argument("--max-score-drift", type=float, default=0.18)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = AdaptConfig(
        high_score=a.high_score,
        margin=a.margin,
        min_consensus_models=a.min_consensus_models,
        physical_gate_um_per_frame=a.physical_gate,
        path_gate_um_per_frame=a.path_gate,
        min_pseudo=a.min_pseudo,
        min_pseudo_frames=a.min_pseudo_frames,
        min_negatives=a.min_negatives,
        blend=a.blend,
        max_score_drift=a.max_score_drift,
    )
    table = read_edge_models(a.input, default_p_fp=a.default_p_fp)
    result = adapt_candidates(table, voxel_size_um=a.voxel_size_um, config=cfg)
    write_adapted(a.output, table, result)
    write_diagnostics(a.diagnostics, a.embryo, result)
    print(json.dumps({"embryo": a.embryo, "enabled": result.enabled, "reason": result.reason, "output": str(a.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
