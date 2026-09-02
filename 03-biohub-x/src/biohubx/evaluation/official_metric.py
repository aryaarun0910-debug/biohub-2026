"""Fail-closed adapter around the pinned authoritative competition metric.

Consumers: ``biohubx evaluate fixture`` and the Phase-1 calibration tests.
The metric mathematics live only in the byte-identical vendored official
source. This module validates Biohub-X contracts, converts them to the official
graph type, and calls the official functions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import polars as pl
import tracksdata as td  # type: ignore[import-untyped]
from pydantic import ValidationError

from biohubx._vendor.official_competition.tracking_cellmot import metrics as authoritative
from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

OFFICIAL_SOURCE_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_MAX_DISTANCE_UM = 7.0


class UnscorablePredictionError(ValueError):
    """The prediction cannot be scored through the official path.

    Raised for a prediction that contains no edges. The official ``evaluate``
    returns early for an edgeless graph without applying its matching, and its
    own ``node_recall`` helper documents that it requires an already-matched
    graph, so calling the two in sequence raises from deep inside the dependency
    instead of producing a number.

    Biohub-X refuses first and says why. A system that abstains everywhere is a
    real possible output, and the person looking at it needs to be told that the
    harness cannot score it, not handed a key error about an attribute name.
    """


class NodeCountProvenance(StrEnum):
    """Where the metric's ``n_total`` came from.

    The official adjustment divides by a coarse estimate of *every* cell, which
    on real data is the GEFF ``estimated_number_of_nodes`` extra. That number is
    not the annotated node count: the annotations are sparse, so the annotated
    count is smaller by an unknown factor. Substituting one for the other
    silently changes what the adjusted Jaccard means, so the substitution has to
    be named.
    """

    OFFICIAL_GEFF_METADATA = "official_geff_metadata"
    """Read from the dataset's own ``estimated_number_of_nodes`` extra."""

    COMPLETE_SYNTHETIC_GROUND_TRUTH = "complete_synthetic_ground_truth"
    """A fixture whose ground truth is complete by construction, so annotated == total."""

    DECLARED = "declared"
    """An experiment-declared value, recorded in that experiment's configuration."""


@dataclass(frozen=True, slots=True)
class EstimatedTotalNodes:
    """The metric's ``n_total``, carrying the provenance of the number.

    There is no default. An adapter that quietly used the annotated count would
    report an adjusted Jaccard that cannot be compared with the competition's.
    """

    value: float
    provenance: NodeCountProvenance

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError(f"estimated total node count must be finite and positive, got {self.value!r}")

    @classmethod
    def from_official_metadata(cls, value: float) -> EstimatedTotalNodes:
        """The real-data path: the dataset's own coarse estimate of all cells."""
        return cls(value=float(value), provenance=NodeCountProvenance.OFFICIAL_GEFF_METADATA)

    @classmethod
    def from_complete_synthetic_ground_truth(cls, ground_truth: LineageGraph) -> EstimatedTotalNodes:
        """A fixture whose ground truth is complete, so the annotated count IS the total.

        Legitimate only because the fixture is constructed that way. It is never
        a stand-in for a missing estimate on real, sparsely annotated data.
        """
        return cls(
            value=float(len(ground_truth.nodes)),
            provenance=NodeCountProvenance.COMPLETE_SYNTHETIC_GROUND_TRUTH,
        )

    @classmethod
    def declared(cls, value: float) -> EstimatedTotalNodes:
        """An explicitly declared estimate, for calibration experiments."""
        return cls(value=float(value), provenance=NodeCountProvenance.DECLARED)


@dataclass(frozen=True, slots=True)
class OfficialMetricResult:
    """Typed result copied from outputs returned by the official implementation."""

    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    annotated_gt_nodes: int
    estimated_total_nodes: float
    node_count_provenance: str
    total_node_ratio: float
    node_recall: float
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_jaccard: float | None
    score: float

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def _to_official_graph(graph: LineageGraph) -> Any:
    converted = td.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        converted.add_node_attr_key(axis, pl.Float64, 0.0)

    official_ids: dict[int, int] = {}
    for node in sorted(graph.nodes, key=lambda item: item.node_id):
        official_ids[node.node_id] = converted.add_node(
            attrs={
                td.DEFAULT_ATTR_KEYS.T: node.frame,
                "z": node.voxel.z,
                "y": node.voxel.y,
                "x": node.voxel.x,
            }
        )
    for edge in sorted(graph.edges, key=lambda item: (item.source, item.target)):
        converted.add_edge(official_ids[edge.source], official_ids[edge.target], {})
    return converted


def evaluate_graph(
    prediction: LineageGraph,
    ground_truth: LineageGraph | None,
    *,
    estimated_total_nodes: EstimatedTotalNodes,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> OfficialMetricResult:
    """Evaluate one graph pair by calling the pinned authoritative source.

    A missing GT artifact is always a refusal. Dataset identity must match, and
    an isotropic scale cannot be constructed by the coordinate contract.

    ``estimated_total_nodes`` is required and has no default. It is the
    denominator of the official node-count adjustment and refers to every cell,
    not to the annotated ones; see :class:`EstimatedTotalNodes`.
    """
    if ground_truth is None:
        raise FileNotFoundError("ground-truth graph artifact is required; evaluated crop set cannot shrink")
    if prediction.dataset != ground_truth.dataset:
        raise ValueError("prediction and ground truth belong to different datasets")
    if not prediction.edges:
        raise UnscorablePredictionError(
            f"prediction for {prediction.dataset.value!r} contains "
            f"{len(prediction.nodes)} nodes and no edges; the official scorer cannot "
            "report node recall for an unmatched graph, so this is refused rather "
            "than reported as a score of zero"
        )
    if not ground_truth.edges:
        raise UnscorablePredictionError(
            f"ground truth for {ground_truth.dataset.value!r} contains no edges; "
            "there is nothing for an edge Jaccard to measure"
        )

    pred_graph = _to_official_graph(prediction)
    gt_graph = _to_official_graph(ground_truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = authoritative.evaluate(
            pred_graph,
            gt_graph,
            scale=scale.values,
            max_distance=OFFICIAL_MAX_DISTANCE_UM,
        )
        recall = authoritative.node_recall(pred_graph, gt_graph)
        row = authoritative.per_sample_metrics(counts, estimated_total_nodes.value, recall)
        summary = authoritative.summarise([row])

    division = float(summary["division_jaccard"])
    return OfficialMetricResult(
        edge_tp=counts.edge_tp,
        edge_fp=counts.edge_fp,
        edge_fn=counts.edge_fn,
        division_tp=counts.division_tp,
        division_fp=counts.division_fp,
        division_fn=counts.division_fn,
        num_pred_nodes=counts.num_pred_nodes,
        annotated_gt_nodes=len(ground_truth.nodes),
        estimated_total_nodes=estimated_total_nodes.value,
        node_count_provenance=estimated_total_nodes.provenance.value,
        total_node_ratio=float(row["total_node_ratio"]),
        node_recall=float(summary["node_recall"]),
        edge_jaccard=float(summary["edge_jaccard"]),
        adjusted_edge_jaccard=float(summary["adj_edge_jaccard"]),
        division_jaccard=None if math.isnan(division) else division,
        score=float(summary["score"]),
    )


def _node(dataset: DatasetIdentity, node_id: int, frame: int, z: float, y: float, x: float) -> LineageNode:
    return LineageNode(
        dataset=dataset,
        node_id=node_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=z, y=y, x=x),
    )


def _edge(dataset: DatasetIdentity, source: int, target: int, kind: EdgeKind) -> LineageEdge:
    return LineageEdge(
        source_dataset=dataset,
        source=source,
        target_dataset=dataset,
        target=target,
        kind=kind,
    )


def _linear_pair() -> tuple[LineageGraph, LineageGraph]:
    dataset = DatasetIdentity(value="synthetic-linear")
    nodes = (
        _node(dataset, 0, 0, 0.0, 0.0, 0.0),
        _node(dataset, 1, 1, 0.0, 0.0, 0.0),
        _node(dataset, 2, 2, 0.0, 0.0, 0.0),
    )
    edges = (
        _edge(dataset, 0, 1, EdgeKind.CONTINUATION),
        _edge(dataset, 1, 2, EdgeKind.CONTINUATION),
    )
    graph = LineageGraph(dataset=dataset, nodes=nodes, edges=edges)
    return graph, graph.model_copy(deep=True)


def _evaluate_complete_fixture(
    prediction: LineageGraph, ground_truth: LineageGraph
) -> dict[str, int | float | str | None]:
    """Score a fixture whose ground truth is complete, so annotated == total.

    Every fixture in this module is built with its ground truth fully specified,
    which is the one situation in which the annotated count is also the total
    count. Real data is sparsely annotated and must supply its own estimate.
    """
    return evaluate_graph(
        prediction,
        ground_truth,
        estimated_total_nodes=EstimatedTotalNodes.from_complete_synthetic_ground_truth(ground_truth),
    ).to_dict()


def _fixture_case(name: str) -> dict[str, object]:
    """Execute one deterministic calibration case and document its behavior."""
    gt, perfect = _linear_pair()
    dataset = gt.dataset
    if name == "perfect_nodes_perfect_edges":
        return {"behavior": "evaluated", "result": _evaluate_complete_fixture(perfect, gt)}
    if name == "missing_edge":
        pred = LineageGraph(dataset=dataset, nodes=perfect.nodes, edges=perfect.edges[:1])
        return {"behavior": "evaluated", "result": _evaluate_complete_fixture(pred, gt)}
    if name == "false_edge":
        nodes = (*perfect.nodes, _node(dataset, 3, 1, 0.0, 20.0, 0.0))
        edges = (
            _edge(dataset, 0, 1, EdgeKind.DIVISION),
            _edge(dataset, 0, 3, EdgeKind.DIVISION),
            _edge(dataset, 1, 2, EdgeKind.CONTINUATION),
        )
        pred = LineageGraph(dataset=dataset, nodes=nodes, edges=edges)
        return {
            "behavior": "evaluated_official_sparse_gt",
            "result": _evaluate_complete_fixture(pred, gt),
        }
    if name == "valid_division":
        div_ds = DatasetIdentity(value="synthetic-division")
        div_nodes = (
            _node(div_ds, 0, 0, 0.0, 0.0, 0.0),
            _node(div_ds, 1, 1, 0.0, 5.0, 0.0),
            _node(div_ds, 2, 1, 0.0, -5.0, 0.0),
        )
        div_edges = (
            _edge(div_ds, 0, 1, EdgeKind.DIVISION),
            _edge(div_ds, 0, 2, EdgeKind.DIVISION),
        )
        graph = LineageGraph(dataset=div_ds, nodes=div_nodes, edges=div_edges)
        mirror = graph.model_copy(deep=True)
        return {"behavior": "evaluated", "result": _evaluate_complete_fixture(graph, mirror)}
    if name == "missing_gt_artifact":
        try:
            evaluate_graph(
                perfect,
                None,
                estimated_total_nodes=EstimatedTotalNodes.declared(float(len(perfect.nodes))),
            )
        except FileNotFoundError as exc:
            return {"behavior": "refused", "reason": str(exc)}
        raise AssertionError("missing GT did not fail closed")
    if name in {"isotropic_coordinate_decoy", "transposed_axis_decoy"}:
        decoy_ds = DatasetIdentity(value=f"synthetic-{name}")
        gt_nodes = (
            _node(decoy_ds, 0, 0, 0.0, 0.0, 0.0),
            _node(decoy_ds, 1, 1, 0.0, 0.0, 0.0),
        )
        gt_edge = (_edge(decoy_ds, 0, 1, EdgeKind.CONTINUATION),)
        decoy_gt = LineageGraph(dataset=decoy_ds, nodes=gt_nodes, edges=gt_edge)
        offset = (5.0, 0.0, 0.0) if name == "isotropic_coordinate_decoy" else (0.0, 0.0, 5.0)
        pred_nodes = (gt_nodes[0], _node(decoy_ds, 1, 1, *offset))
        pred = LineageGraph(dataset=decoy_ds, nodes=pred_nodes, edges=gt_edge)
        return {"behavior": "evaluated", "result": _evaluate_complete_fixture(pred, decoy_gt)}

    invalid_documents = _invalid_fixture_documents(gt)
    try:
        LineageGraph.model_validate(invalid_documents[name])
    except (ValidationError, ValueError) as exc:
        return {"behavior": "refused", "reason": str(exc).splitlines()[0]}
    raise AssertionError(f"malformed fixture {name!r} was accepted")


def _invalid_fixture_documents(gt: LineageGraph) -> dict[str, dict[str, object]]:
    base = gt.model_dump(mode="python")
    other = DatasetIdentity(value="other-dataset")
    return {
        "missing_node_endpoint": {
            **base,
            "edges": [*base["edges"], _edge(gt.dataset, 2, 99, EdgeKind.CONTINUATION)],
        },
        "duplicate_node": {**base, "nodes": [*base["nodes"], base["nodes"][0]]},
        "illegal_in_degree": {
            **base,
            "nodes": [*base["nodes"], _node(gt.dataset, 3, 1, 0.0, 10.0, 0.0)],
            "edges": [
                _edge(gt.dataset, 0, 1, EdgeKind.DIVISION),
                _edge(gt.dataset, 0, 3, EdgeKind.DIVISION),
                _edge(gt.dataset, 1, 2, EdgeKind.CONTINUATION),
                _edge(gt.dataset, 3, 2, EdgeKind.CONTINUATION),
            ],
        },
        "illegal_out_degree": {
            **base,
            "nodes": [
                *base["nodes"],
                _node(gt.dataset, 3, 1, 0.0, 10.0, 0.0),
                _node(gt.dataset, 4, 1, 0.0, 20.0, 0.0),
            ],
            "edges": [
                _edge(gt.dataset, 0, 1, EdgeKind.DIVISION),
                _edge(gt.dataset, 0, 3, EdgeKind.DIVISION),
                _edge(gt.dataset, 0, 4, EdgeKind.DIVISION),
            ],
        },
        "invalid_three_daughter_fork": {
            **base,
            "nodes": [
                base["nodes"][0],
                base["nodes"][1],
                _node(gt.dataset, 3, 1, 0.0, 10.0, 0.0),
                _node(gt.dataset, 4, 1, 0.0, 20.0, 0.0),
            ],
            "edges": [
                _edge(gt.dataset, 0, 1, EdgeKind.DIVISION),
                _edge(gt.dataset, 0, 3, EdgeKind.DIVISION),
                _edge(gt.dataset, 0, 4, EdgeKind.DIVISION),
            ],
        },
        "cross_dataset_edge": {
            **base,
            "edges": [
                {
                    **base["edges"][0],
                    "target_dataset": other,
                }
            ],
        },
        "backward_time_edge": {**base, "edges": [_edge(gt.dataset, 1, 0, EdgeKind.CONTINUATION)]},
        "non_integer_identity": {**base, "nodes": [{**base["nodes"][0], "node_id": 0.5}, *base["nodes"][1:]]},
    }


CALIBRATION_FIXTURES = (
    "perfect_nodes_perfect_edges",
    "missing_edge",
    "false_edge",
    "missing_node_endpoint",
    "duplicate_node",
    "illegal_in_degree",
    "illegal_out_degree",
    "valid_division",
    "invalid_three_daughter_fork",
    "cross_dataset_edge",
    "backward_time_edge",
    "non_integer_identity",
    "missing_gt_artifact",
    "isotropic_coordinate_decoy",
    "transposed_axis_decoy",
)


def evaluate_calibration_fixtures() -> dict[str, dict[str, object]]:
    """Run every required Phase-1 fixture in a stable order."""
    return {name: _fixture_case(name) for name in CALIBRATION_FIXTURES}
