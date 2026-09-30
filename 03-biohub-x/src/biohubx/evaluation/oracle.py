"""The association ceiling over a frozen proposal set, scored officially.

E06's first measurement. Take the frozen proposals, match every annotated node
one-to-one to a proposal within the official radius, keep only the ground-truth
edges whose two endpoints both matched, rewrite those edges onto their matched
proposals, and score the result through the pinned official metric with every
proposal present as a node whether it matched or not.

What this is: the score a perfect linker would get on these proposals. Every edge
emitted is correct, every proposal is charged in the node count, and nothing
about how to find those edges is assumed. What this is not: a method. It uses
ground truth to choose edges and can never be promoted; its only job is to say
whether association is worth working on with these proposals, or whether the
proposals themselves still cap the score.

Consumer: ``biohubx evaluate oracle-ceiling``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.contracts.instances import InstanceSet
from biohubx.contracts.lineage import LineageGraph, LineageNode
from biohubx.data.competition import edges_with_kinds
from biohubx.evaluation.proposals import OFFICIAL_MATCH_RADIUS_UM


@dataclass(frozen=True, slots=True)
class OracleGraph:
    """The ceiling graph and the accounting behind it."""

    graph: LineageGraph | None
    proposals: int
    annotated_nodes: int
    matched_nodes: int
    annotated_edges: int
    retained_edges: int
    retained_divisions: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["graph"] = None if self.graph is None else "present"
        return payload


def match_one_to_one(
    instances: InstanceSet, annotated: LineageGraph, *, radius_um: float = OFFICIAL_MATCH_RADIUS_UM
) -> dict[int, int]:
    """Annotated node id to proposal instance id, per frame, minimum-cost, capped.

    The same rule the official matcher applies to nodes: within a frame, each
    annotated node takes at most one proposal and each proposal at most one
    annotated node, and no pair further apart than the radius counts.
    """
    proposals: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for instance in instances.instances:
        point = instance.physical
        proposals.setdefault(instance.frame, []).append(
            (instance.instance_id, (point.z_um, point.y_um, point.x_um))
        )
    targets: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for node in annotated.nodes:
        point = node.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        targets.setdefault(node.frame, []).append((node.node_id, (point.z_um, point.y_um, point.x_um)))

    matches: dict[int, int] = {}
    for frame, annotated_points in targets.items():
        candidates = proposals.get(frame)
        if not candidates:
            continue
        cost = np.array(
            [[math.dist(target, candidate) for _, candidate in candidates] for _, target in annotated_points]
        )
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            if cost[row, col] <= radius_um:
                matches[annotated_points[row][0]] = candidates[col][0]
    return matches


def oracle_graph(instances: InstanceSet, annotated: LineageGraph) -> OracleGraph:
    """Build the ceiling graph, or report why none can be built.

    A window in which no annotated edge has both endpoints matched yields a
    graph with nodes and no edges, which the pinned scorer cannot score
    ([[F-0009]]). Such a window is returned with ``graph`` absent and its edge
    count intact, so the caller can report what was left unscored rather than
    let it vanish.
    """
    matches = match_one_to_one(instances, annotated)
    pairs = [
        (matches[edge.source], matches[edge.target])
        for edge in annotated.edges
        if edge.source in matches and edge.target in matches
    ]
    out_degree: dict[int, int] = {}
    for source, _ in pairs:
        out_degree[source] = out_degree.get(source, 0) + 1
    divisions = sum(1 for count in out_degree.values() if count >= 2)

    if not pairs:
        return OracleGraph(
            graph=None,
            proposals=len(instances.instances),
            annotated_nodes=len(annotated.nodes),
            matched_nodes=len(matches),
            annotated_edges=len(annotated.edges),
            retained_edges=0,
            retained_divisions=0,
        )

    nodes = tuple(
        LineageNode(
            dataset=instances.dataset,
            node_id=instance.instance_id,
            frame=instance.frame,
            voxel=instance.voxel,
        )
        for instance in instances.instances
    )
    edges = edges_with_kinds(instances.dataset, np.array(pairs, dtype=np.int64).reshape(-1, 2))
    graph = LineageGraph(dataset=instances.dataset, nodes=nodes, edges=edges)
    return OracleGraph(
        graph=graph,
        proposals=len(instances.instances),
        annotated_nodes=len(annotated.nodes),
        matched_nodes=len(matches),
        annotated_edges=len(annotated.edges),
        retained_edges=len(pairs),
        retained_divisions=divisions,
    )
