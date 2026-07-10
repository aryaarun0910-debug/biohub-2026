"""CSV/GEFF adapters for the metric-aligned solver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import CandidateEdges, SolveResult


@dataclass(frozen=True)
class Nodes:
    time_by_id: dict[int, int]


def load_nodes(path: str | Path) -> Nodes:
    """Load node identities/times from GEFF or a ``node_id,t`` CSV."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if rows and not {"node_id", "t"}.issubset(rows[0]):
            raise ValueError("node CSV needs node_id,t columns")
        mapping = {int(row["node_id"]): int(row["t"]) for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("node_id must be unique")
        return Nodes(mapping)

    try:
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("GEFF node input requires tracksdata") from exc
    result = td.graph.IndexedRXGraph.from_geff(path)
    graph = result[0] if isinstance(result, tuple) else result
    key = td.DEFAULT_ATTR_KEYS.NODE_ID
    attrs = graph.node_attrs(attr_keys=[key, "t"])
    return Nodes(dict(zip(attrs[key].to_list(), attrs["t"].to_list(), strict=True)))


def read_candidates(
    path: str | Path,
    *,
    missing_p_fp: str = "error",
    p_fp_constant: float | None = None,
) -> CandidateEdges:
    """Read candidates without silently treating ``p_fp`` as ``1-p_tp``."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "target_id", "p_tp"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"candidate CSV needs columns {sorted(required)}")
    has_fp = not rows or "p_fp" in rows[0]
    p_tp = np.asarray([float(row["p_tp"]) for row in rows], dtype=np.float64)
    if has_fp:
        p_fp = np.asarray([float(row["p_fp"]) for row in rows], dtype=np.float64)
    elif missing_p_fp == "one-minus-p-tp":
        p_fp = 1.0 - p_tp
    elif missing_p_fp == "constant":
        if p_fp_constant is None:
            raise ValueError("p_fp_constant is required for missing_p_fp='constant'")
        p_fp = np.full(len(rows), float(p_fp_constant))
    elif missing_p_fp == "error":
        raise ValueError(
            "candidate CSV has no p_fp; sparse annotations make p_fp != 1-p_tp. "
            "Choose an explicit approximation only for an ablation."
        )
    else:
        raise ValueError(f"unknown missing_p_fp policy {missing_p_fp!r}")
    return CandidateEdges(
        np.asarray([int(row["source_id"]) for row in rows], dtype=np.int64),
        np.asarray([int(row["target_id"]) for row in rows], dtype=np.int64),
        p_tp,
        p_fp,
    )


def validate_candidates(candidates: CandidateEdges, nodes: Nodes) -> None:
    ids = set(nodes.time_by_id)
    endpoints = set(candidates.source_id) | set(candidates.target_id)
    unknown = sorted(endpoints - ids)
    if unknown:
        raise ValueError(f"candidate endpoints absent from nodes: {unknown[:10]}")
    backwards = [
        (int(s), int(t))
        for s, t in zip(candidates.source_id, candidates.target_id, strict=True)
        if nodes.time_by_id[int(t)] <= nodes.time_by_id[int(s)]
    ]
    if backwards:
        raise ValueError(f"candidate edges must move forward in time: {backwards[:5]}")


def write_solution(
    path: str | Path,
    candidates: CandidateEdges,
    result: SolveResult,
    *,
    lambda_: float,
) -> Path:
    """Emit every reweighted candidate plus a selected flag."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_id", "target_id", "p_tp", "p_fp", "metric_weight", "lambda", "selected"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(candidates)):
            writer.writerow(
                {
                    "source_id": int(candidates.source_id[i]),
                    "target_id": int(candidates.target_id[i]),
                    "p_tp": float(candidates.p_tp[i]),
                    "p_fp": float(candidates.p_fp[i]),
                    "metric_weight": float(result.weights[i]),
                    "lambda": float(lambda_),
                    "selected": int(result.selected[i]),
                }
            )
    return path
