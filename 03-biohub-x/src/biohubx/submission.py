"""The competition submission CSV: written from lineage graphs and validated before anything leaves.

The schema is the official one, read from the competition repository at the
pinned commit rather than assumed (research ledger RL-0051 and RL-0052,
``scripts/geffs_to_csv.py`` and ``scripts/csv_to_geffs.py`` at
075fc5f5a52d11077f9dc2b074644618f26939e2): one row per node and one per edge,
columns ``id, dataset, row_type, node_id, t, z, y, x, source_id, target_id``,
``id`` a 0-based row index prepended last, node coordinates rounded to
integers, and ``-1`` in every field a row type does not use. The reader on
Kaggle's side rebuilds each dataset's graph from these rows and matches
spatially, so the ids only need to be consistent within the file.

Nothing here contacts Kaggle. A submission is an act Arya Arun authorises
individually; this module exists so that when that day comes the file has
already been produced, validated and round-tripped on synthetic data through
the same code, and every reason it could be refused is known in advance.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biohubx.contracts.lineage import LineageGraph

COLUMNS: tuple[str, ...] = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
"""Exactly the official writer's order, ``id`` first as ``with_row_index`` places it."""
NODE = "node"
EDGE = "edge"
UNUSED = -1
MAX_OUT_DEGREE = 2
MAX_IN_DEGREE = 1


class SubmissionError(ValueError):
    """The file cannot be written or does not validate; the message names why."""


def rows_from_graph(graph: LineageGraph, dataset: str) -> list[dict[str, int | str]]:
    """Node rows then edge rows for one dataset, deterministic order, integers everywhere."""
    rows: list[dict[str, int | str]] = []
    for node in sorted(graph.nodes, key=lambda n: n.node_id):
        rows.append(
            {
                "dataset": dataset,
                "row_type": NODE,
                "node_id": int(node.node_id),
                "t": int(node.frame),
                "z": round(node.voxel.z),
                "y": round(node.voxel.y),
                "x": round(node.voxel.x),
                "source_id": UNUSED,
                "target_id": UNUSED,
            }
        )
    for edge in sorted(graph.edges, key=lambda e: (e.source, e.target)):
        rows.append(
            {
                "dataset": dataset,
                "row_type": EDGE,
                "node_id": UNUSED,
                "t": UNUSED,
                "z": UNUSED,
                "y": UNUSED,
                "x": UNUSED,
                "source_id": int(edge.source),
                "target_id": int(edge.target),
            }
        )
    return rows


@dataclass
class WriteReport:
    path: str
    rows: int
    bytes: int
    sha256: str
    per_dataset: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "bytes": self.bytes,
            "raw_digest": f"raw_artifact_sha256:sha256:{self.sha256}",
            "per_dataset": self.per_dataset,
            "bytes_per_row": round(self.bytes / self.rows, 3) if self.rows else None,
        }


def write_csv(graphs: Mapping[str, LineageGraph], path: Path) -> WriteReport:
    """One CSV for every dataset, ``id`` contiguous from 0, written atomically."""
    if not graphs:
        raise SubmissionError("no graphs to write; an empty submission is a refusal, not a file")
    per_dataset: dict[str, dict[str, int]] = {}
    staged = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    with staged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for dataset in sorted(graphs):
            rows = rows_from_graph(graphs[dataset], dataset)
            per_dataset[dataset] = {
                "nodes": sum(1 for r in rows if r["row_type"] == NODE),
                "edges": sum(1 for r in rows if r["row_type"] == EDGE),
            }
            for row in rows:
                writer.writerow([index, *(row[column] for column in COLUMNS[1:])])
                index += 1
    staged.replace(path)
    data = path.read_bytes()
    return WriteReport(
        path=str(path),
        rows=index,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        per_dataset=per_dataset,
    )


@dataclass
class ValidationReport:
    path: str
    rows: int
    datasets: list[str]
    refusals: list[str] = field(default_factory=list)
    per_dataset: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.refusals

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "datasets": self.datasets,
            "per_dataset": self.per_dataset,
            "refusals": self.refusals,
            "ok": self.ok,
        }


def _int(value: str, what: str, refusals: list[str], line: int) -> int | None:
    try:
        return int(value)
    except ValueError:
        refusals.append(f"line {line}: {what} is not an integer: {value!r}")
        return None


def validate_csv(
    path: Path, *, expected_datasets: Sequence[str] | None = None, max_refusals: int = 50
) -> ValidationReport:
    """Every reason the file could be refused, found here rather than by the platform.

    Header exact, ten fields per row, ``id`` contiguous from 0, row types known,
    node ids unique per dataset with non-negative coordinates and unused fields
    at ``-1``, edge rows with unused fields at ``-1`` and endpoints that exist in
    the same dataset one frame apart, in-degree at most 1 and out-degree at most
    2, no self loops, and, when the expected dataset list is given, exactly those
    datasets present.
    """
    refusals: list[str] = []
    nodes: dict[str, dict[int, int]] = {}
    edges: dict[str, list[tuple[int, int]]] = {}
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return ValidationReport(str(path), 0, [], ["the file is empty"])
        if tuple(header) != COLUMNS:
            refusals.append(f"header is {header}, not {list(COLUMNS)}")
            return ValidationReport(str(path), 0, [], refusals)
        for line, record in enumerate(reader, start=2):
            if len(refusals) >= max_refusals:
                refusals.append("stopped after the maximum number of refusals")
                break
            if len(record) != len(COLUMNS):
                refusals.append(f"line {line}: {len(record)} fields, expected {len(COLUMNS)}")
                continue
            values = dict(zip(COLUMNS, record, strict=True))
            row_id = _int(values["id"], "id", refusals, line)
            if row_id is not None and row_id != rows:
                refusals.append(f"line {line}: id {row_id} breaks the contiguous index at {rows}")
            rows += 1
            dataset = values["dataset"]
            if not dataset:
                refusals.append(f"line {line}: empty dataset")
                continue
            numbers = {
                k: _int(values[k], k, refusals, line)
                for k in ("node_id", "t", "z", "y", "x", "source_id", "target_id")
            }
            if any(v is None for v in numbers.values()):
                continue
            n = {k: int(v) for k, v in numbers.items() if v is not None}
            if values["row_type"] == NODE:
                if n["source_id"] != UNUSED or n["target_id"] != UNUSED:
                    refusals.append(f"line {line}: a node row carries edge fields")
                if n["node_id"] < 0 or n["t"] < 0 or n["z"] < 0 or n["y"] < 0 or n["x"] < 0:
                    refusals.append(f"line {line}: a node row has a negative id, frame or coordinate")
                table = nodes.setdefault(dataset, {})
                if n["node_id"] in table:
                    refusals.append(f"line {line}: node_id {n['node_id']} repeats within {dataset}")
                table[n["node_id"]] = n["t"]
            elif values["row_type"] == EDGE:
                if any(n[k] != UNUSED for k in ("node_id", "t", "z", "y", "x")):
                    refusals.append(f"line {line}: an edge row carries node fields")
                if n["source_id"] == n["target_id"]:
                    refusals.append(f"line {line}: self loop on {n['source_id']}")
                edges.setdefault(dataset, []).append((n["source_id"], n["target_id"]))
            else:
                refusals.append(f"line {line}: unknown row_type {values['row_type']!r}")
    per_dataset: dict[str, dict[str, int]] = {}
    for dataset in sorted(set(nodes) | set(edges)):
        table = nodes.get(dataset, {})
        out_degree: Counter[int] = Counter()
        in_degree: Counter[int] = Counter()
        for source, target in edges.get(dataset, []):
            if source not in table or target not in table:
                refusals.append(
                    f"{dataset}: edge {source}->{target} references a node that is not in the file"
                )
                continue
            if table[target] != table[source] + 1:
                refusals.append(
                    f"{dataset}: edge {source}->{target} spans frames "
                    f"{table[source]}->{table[target]}, not one"
                )
            out_degree[source] += 1
            in_degree[target] += 1
        for node_id, degree in out_degree.items():
            if degree > MAX_OUT_DEGREE:
                refusals.append(f"{dataset}: node {node_id} has out-degree {degree}")
        for node_id, degree in in_degree.items():
            if degree > MAX_IN_DEGREE:
                refusals.append(f"{dataset}: node {node_id} has in-degree {degree}")
        per_dataset[dataset] = {"nodes": len(table), "edges": len(edges.get(dataset, []))}
    if expected_datasets is not None:
        missing = sorted(set(expected_datasets) - set(per_dataset))
        extra = sorted(set(per_dataset) - set(expected_datasets))
        if missing:
            refusals.append(f"datasets expected and absent: {missing[:5]}")
        if extra:
            refusals.append(f"datasets present and not expected: {extra[:5]}")
    return ValidationReport(str(path), rows, sorted(per_dataset), refusals, per_dataset)


def round_trip(graphs: Mapping[str, LineageGraph], path: Path) -> dict[str, Any]:
    """Rebuild every dataset from the CSV and compare with what was written, node by node and edge by edge."""
    rebuilt_nodes: dict[str, set[tuple[int, int, int, int, int]]] = {}
    rebuilt_edges: dict[str, set[tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            dataset = record["dataset"]
            if record["row_type"] == NODE:
                rebuilt_nodes.setdefault(dataset, set()).add(
                    tuple(int(record[k]) for k in ("node_id", "t", "z", "y", "x"))  # type: ignore[arg-type]
                )
            elif record["row_type"] == EDGE:
                rebuilt_edges.setdefault(dataset, set()).add(
                    (int(record["source_id"]), int(record["target_id"]))
                )
    differences: list[str] = []
    for dataset, graph in graphs.items():
        expected_nodes = {
            (
                int(n.node_id),
                int(n.frame),
                round(n.voxel.z),
                round(n.voxel.y),
                round(n.voxel.x),
            )
            for n in graph.nodes
        }
        expected_edges = {(int(e.source), int(e.target)) for e in graph.edges}
        if expected_nodes != rebuilt_nodes.get(dataset, set()):
            differences.append(f"{dataset}: nodes differ")
        if expected_edges != rebuilt_edges.get(dataset, set()):
            differences.append(f"{dataset}: edges differ")
    for dataset in set(rebuilt_nodes) - set(graphs):
        differences.append(f"{dataset}: in the file and not among the graphs")
    return {"identical": not differences, "differences": differences, "datasets": sorted(graphs)}


def forecast(
    report: WriteReport, *, nodes_per_movie: float, edges_per_node: float, movies: int
) -> dict[str, Any]:
    """Size and row counts a real submission of this shape would have, from measured bytes per row."""
    bytes_per_row = report.bytes / report.rows if report.rows else 0.0
    rows = movies * nodes_per_movie * (1.0 + edges_per_node)
    return {
        "assumed_movies": movies,
        "assumed_nodes_per_movie": nodes_per_movie,
        "assumed_edges_per_node": edges_per_node,
        "measured_bytes_per_row": round(bytes_per_row, 3),
        "projected_rows": int(rows),
        "projected_bytes": int(rows * bytes_per_row),
        "projected_megabytes": round(rows * bytes_per_row / 1e6, 1),
    }
