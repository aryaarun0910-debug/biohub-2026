"""Sparse temporal candidate graph construction and its reach report.

Generation is separate from scoring. This module decides only what the matcher
is allowed to consider, and then measures how much of the true answer survived
that decision, so that a later disappointing score can be attributed to the
right stage.

Consumers: :mod:`biohubx.tracking.matcher`, ``biohubx infer synthetic``.
"""

from __future__ import annotations

import numpy as np

from biohubx.contracts.candidates import CandidateEdge, CandidateGraph, ReachReport
from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelScaleZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet
from biohubx.contracts.lineage import LineageGraph

DEFAULT_CANDIDATE_RADIUS_UM = 12.0
DEFAULT_MATCH_RADIUS_UM = 7.0
"""The official node-matching radius, reused to decide which instance stands for
which true node when measuring reach."""


def build_candidate_graph(
    instances: InstanceSet,
    *,
    radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
    ground_truth: LineageGraph | None = None,
    match_radius_um: float = DEFAULT_MATCH_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> CandidateGraph:
    """Offer every consecutive-frame pair within ``radius_um`` of each other.

    ``ground_truth`` is optional, and its absence produces no reach report rather
    than a reach of zero: not knowing the answer is different from getting none
    of it right.
    """
    if radius_um <= 0:
        raise ValueError(f"candidate radius must be positive, got {radius_um}")

    edges: list[CandidateEdge] = []
    frames = instances.frames
    for frame in frames:
        if frame + 1 not in frames:
            continue
        sources = instances.by_frame(frame)
        targets = instances.by_frame(frame + 1)
        for source in sources:
            for target in targets:
                separation = _separation_um(source, target)
                if separation <= radius_um:
                    edges.append(
                        CandidateEdge(
                            source=source.instance_id,
                            target=target.instance_id,
                            source_frame=frame,
                            target_frame=frame + 1,
                            displacement_um=separation,
                        )
                    )
    if not edges:
        raise ValueError(f"no candidate pair fell within {radius_um} um; the radius or the detector is wrong")

    reach = (
        None
        if ground_truth is None
        else measure_reach(
            instances,
            edges,
            ground_truth=ground_truth,
            match_radius_um=match_radius_um,
            scale=scale,
        )
    )
    return CandidateGraph(dataset=instances.dataset, radius_um=radius_um, edges=tuple(edges), reach=reach)


def measure_reach(
    instances: InstanceSet,
    edges: list[CandidateEdge],
    *,
    ground_truth: LineageGraph,
    match_radius_um: float = DEFAULT_MATCH_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> ReachReport:
    """Count how many true edges the matcher will actually be offered.

    A true edge is reachable when both of its endpoints were detected within the
    matching radius AND the pair of detections standing for them was offered. The
    two failure modes are reported separately, because a missing endpoint is the
    detector's problem and a missing pair is the radius's problem, and confusing
    them sends the next experiment to the wrong component.
    """
    matched = _match_truth_to_instances(instances, ground_truth, match_radius_um=match_radius_um, scale=scale)
    offered = {(edge.source, edge.target) for edge in edges}

    reachable = 0
    missing_endpoint = 0
    outside_radius = 0
    for edge in ground_truth.edges:
        source = matched.get(edge.source)
        target = matched.get(edge.target)
        if source is None or target is None:
            missing_endpoint += 1
        elif (source, target) in offered:
            reachable += 1
        else:
            outside_radius += 1

    return ReachReport(
        true_edges=len(ground_truth.edges),
        reachable_true_edges=reachable,
        unreachable_missing_endpoint=missing_endpoint,
        unreachable_outside_radius=outside_radius,
    )


def _match_truth_to_instances(
    instances: InstanceSet,
    ground_truth: LineageGraph,
    *,
    match_radius_um: float,
    scale: VoxelScaleZYX,
) -> dict[int, int]:
    """One-to-one nearest matching per frame, in micrometres.

    Greedy by ascending distance. This is the fixture's own bookkeeping, used
    only to attribute reach; the official scorer performs its own optimal
    matching and this never substitutes for it.
    """
    matched: dict[int, int] = {}
    by_frame: dict[int, list[int]] = {}
    for node in ground_truth.nodes:
        by_frame.setdefault(node.frame, []).append(node.node_id)

    truth_positions: dict[int, np.ndarray] = {}
    for node in ground_truth.nodes:
        physical = node.voxel.to_physical(scale)
        truth_positions[node.node_id] = np.array(
            [physical.z_um, physical.y_um, physical.x_um], dtype=np.float64
        )

    for frame, node_ids in sorted(by_frame.items()):
        available = list(instances.by_frame(frame))
        pairs: list[tuple[float, int, int]] = []
        for node_id in sorted(node_ids):
            for instance in available:
                position = np.array([instance.physical.z_um, instance.physical.y_um, instance.physical.x_um])
                distance = float(np.linalg.norm(truth_positions[node_id] - position))
                if distance <= match_radius_um:
                    pairs.append((distance, node_id, instance.instance_id))
        pairs.sort()
        taken_truth: set[int] = set()
        taken_instance: set[int] = set()
        for _distance, node_id, instance_id in pairs:
            if node_id in taken_truth or instance_id in taken_instance:
                continue
            matched[node_id] = instance_id
            taken_truth.add(node_id)
            taken_instance.add(instance_id)
    return matched


def _separation_um(source: CandidateInstance, target: CandidateInstance) -> float:
    return float(
        np.linalg.norm(
            np.array([source.physical.z_um, source.physical.y_um, source.physical.x_um])
            - np.array([target.physical.z_um, target.physical.y_um, target.physical.x_um])
        )
    )
