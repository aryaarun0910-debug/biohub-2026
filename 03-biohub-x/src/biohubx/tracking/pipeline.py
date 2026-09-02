"""The Phase-2 vertical slice: one function that owns the whole chain.

volume -> instances -> representation -> candidate graph -> scored options
-> legal lineage graph -> official score

It lives here rather than in the CLI so that the chain is testable without a
process boundary, and so that no stage can be reached except through the one in
front of it. Every stage's output is a validated contract, so a break shows up at
the stage that caused it.

Consumer: ``biohubx infer synthetic`` and ``biohubx evaluate slice``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from biohubx.contracts.candidates import CandidateGraph
from biohubx.contracts.coordinates import VoxelScaleZYX
from biohubx.contracts.instances import InstanceSet
from biohubx.contracts.lineage import DatasetIdentity, LineageGraph
from biohubx.contracts.predictions import AssociationPrediction
from biohubx.contracts.representation import RepresentationSet
from biohubx.data.synthetic import SyntheticTruth, build_synthetic_truth
from biohubx.evaluation.official_metric import (
    EstimatedTotalNodes,
    NodeCountProvenance,
    OfficialMetricResult,
    evaluate_graph,
)
from biohubx.proposals.classical import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_SUPPRESSION_RADIUS_UM,
    detect_instances,
)
from biohubx.representation.geometry import DEFAULT_DENSITY_RADIUS_UM, describe_instances
from biohubx.tracking.candidate_graph import DEFAULT_CANDIDATE_RADIUS_UM, build_candidate_graph
from biohubx.tracking.graph_decoder import DecodeReport, decode
from biohubx.tracking.matcher import MatcherWeights, score_candidates


@dataclass(frozen=True, slots=True)
class SliceConfig:
    """Every knob the slice exposes, in one typed place with no hidden defaults."""

    seed: int = 0
    noise: float = 0.01
    annotated_fraction: float = 1.0
    estimated_total_nodes: float | None = None
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    suppression_radius_um: float = DEFAULT_SUPPRESSION_RADIUS_UM
    density_radius_um: float = DEFAULT_DENSITY_RADIUS_UM
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM
    weights: MatcherWeights = field(default_factory=MatcherWeights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "noise": self.noise,
            "annotated_fraction": self.annotated_fraction,
            "estimated_total_nodes": self.estimated_total_nodes,
            "detection_threshold": self.detection_threshold,
            "suppression_radius_um": self.suppression_radius_um,
            "density_radius_um": self.density_radius_um,
            "candidate_radius_um": self.candidate_radius_um,
            "weights": {
                "displacement_um_penalty": self.weights.displacement_um_penalty,
                "intensity_change_penalty": self.weights.intensity_change_penalty,
                "birth_base": self.weights.birth_base,
                "birth_border_bonus_per_um": self.weights.birth_border_bonus_per_um,
                "birth_border_reference_um": self.weights.birth_border_reference_um,
            },
        }


@dataclass(frozen=True, slots=True)
class ChainOutcome:
    """Everything the chain produced from one volume and one annotated graph.

    Shared by the synthetic fixture and by real competition windows so that the
    two cannot drift apart: a stage that behaves differently on real data
    behaves differently in exactly one place.
    """

    instances: InstanceSet
    representation: RepresentationSet
    candidates: CandidateGraph
    prediction: AssociationPrediction
    emitted: LineageGraph
    decode_report: DecodeReport
    score: OfficialMetricResult


def run_chain(
    volume: Any,
    *,
    dataset: DatasetIdentity,
    annotated: LineageGraph,
    scale: VoxelScaleZYX,
    estimate: EstimatedTotalNodes,
    settings: SliceConfig,
) -> ChainOutcome:
    """volume -> instances -> representation -> candidates -> graph -> official score."""
    instances = detect_instances(
        volume,
        dataset=dataset,
        threshold=settings.detection_threshold,
        suppression_radius_um=settings.suppression_radius_um,
        scale=scale,
    )
    representation = describe_instances(
        instances, volume, density_radius_um=settings.density_radius_um, scale=scale
    )
    candidates = build_candidate_graph(
        instances, radius_um=settings.candidate_radius_um, ground_truth=annotated, scale=scale
    )
    prediction = score_candidates(candidates, representation, weights=settings.weights)
    emitted, decode_report = decode(prediction, instances)
    score = evaluate_graph(emitted, annotated, estimated_total_nodes=estimate)
    return ChainOutcome(
        instances=instances,
        representation=representation,
        candidates=candidates,
        prediction=prediction,
        emitted=emitted,
        decode_report=decode_report,
        score=score,
    )


@dataclass(frozen=True, slots=True)
class SliceResult:
    """Every intermediate the slice produced, kept for attribution."""

    config: SliceConfig
    truth: SyntheticTruth
    instances: InstanceSet
    representation: RepresentationSet
    candidates: CandidateGraph
    prediction: AssociationPrediction
    emitted: LineageGraph
    decode_report: DecodeReport
    score: OfficialMetricResult

    def report(self) -> dict[str, Any]:
        """A machine-readable summary that never restates a number twice."""
        reach = self.candidates.reach
        if reach is None:
            raise ValueError("the slice must measure reach; a missing reach report is a failure")
        return {
            "schema_version": 1,
            "dataset": self.emitted.dataset.value,
            "config": self.config.to_dict(),
            "detection": {
                "instances": len(self.instances.instances),
                "frames": list(self.instances.frames),
            },
            "candidates": {
                "radius_um": self.candidates.radius_um,
                "edges": len(self.candidates.edges),
                "targets": len(self.candidates.targets),
                "reach": {
                    "true_edges": reach.true_edges,
                    "reachable_true_edges": reach.reachable_true_edges,
                    "unreachable_missing_endpoint": reach.unreachable_missing_endpoint,
                    "unreachable_outside_radius": reach.unreachable_outside_radius,
                    "fraction": reach.reach,
                },
            },
            "decode": self.decode_report.to_dict(),
            "truth": {
                "true_nodes": len(self.truth.lineage.nodes),
                "true_edges": len(self.truth.lineage.edges),
                "annotated_nodes": len(self.truth.annotated.nodes),
                "annotated_edges": len(self.truth.annotated.edges),
            },
            "official": self.score.to_dict(),
        }


def run_slice(config: SliceConfig | None = None) -> SliceResult:
    """Run the whole chain on the deterministic fixture.

    Scored against the ANNOTATED graph, because that is what a real dataset
    provides, while the node-count estimate refers to every cell. Keeping those
    two apart is the whole point of the estimate carrying its provenance.
    """
    settings = config if config is not None else SliceConfig()

    truth = build_synthetic_truth(
        seed=settings.seed,
        noise=settings.noise,
        annotated_fraction=settings.annotated_fraction,
        estimated_total_nodes=settings.estimated_total_nodes,
    )
    estimate = (
        EstimatedTotalNodes.from_complete_synthetic_ground_truth(truth.annotated)
        if settings.annotated_fraction == 1.0 and settings.estimated_total_nodes is None
        else EstimatedTotalNodes.declared(truth.estimated_total_nodes)
    )
    outcome = run_chain(
        truth.volume,
        dataset=truth.lineage.dataset,
        annotated=truth.annotated,
        scale=truth.scale,
        estimate=estimate,
        settings=settings,
    )
    if settings.annotated_fraction < 1.0:
        assert outcome.score.node_count_provenance == NodeCountProvenance.DECLARED.value

    return SliceResult(
        config=settings,
        truth=truth,
        instances=outcome.instances,
        representation=outcome.representation,
        candidates=outcome.candidates,
        prediction=outcome.prediction,
        emitted=outcome.emitted,
        decode_report=outcome.decode_report,
        score=outcome.score,
    )
