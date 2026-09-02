"""Greedy constrained decoding into a complete legal lineage graph.

This is the step that makes the system own its output. It takes scored options
and returns a `LineageGraph`, which means every structural rule is enforced here
rather than hoped for: in-degree at most one, out-degree at most two, every
endpoint present, every edge advancing exactly one frame, edge kinds agreeing
with the final topology.

The strategy is the simplest one that can be correct: each target takes its best
option including abstention, then any parent chosen more than twice keeps its two
best children and the rest become births. A min-cost-flow or ILP decoder is a
challenger to this, not a replacement for it, and it cannot be compared against
anything until this one exists and is known to be legal.

Consumer: ``biohubx infer synthetic``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from biohubx.contracts.instances import InstanceSet
from biohubx.contracts.lineage import (
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.contracts.predictions import AssociationPrediction

MAX_CHILDREN = 2
MAX_PARENTS = 1


@dataclass(frozen=True, slots=True)
class DecodeReport:
    """What the decoder did, in terms a later experiment can compare."""

    nodes: int
    edges: int
    continuations: int
    divisions: int
    births: int
    abstentions: int
    """Targets whose no-parent option outscored every offered parent."""

    out_degree_evictions: int
    """Links dropped because a parent was chosen by more than two targets."""

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def decode(
    prediction: AssociationPrediction,
    instances: InstanceSet,
    *,
    max_children: int = MAX_CHILDREN,
) -> tuple[LineageGraph, DecodeReport]:
    """Decode scored options into a legal graph, and report what was decided.

    Every detected instance becomes a node, including instances that end up with
    no edges at all. Dropping isolated detections would quietly change the node
    count, which the official metric charges for, so that decision belongs to a
    calibration experiment rather than to the decoder.
    """
    if max_children < 1:
        raise ValueError(f"a parent must be allowed at least one child, got {max_children}")

    chosen: dict[int, tuple[int, float]] = {}
    abstentions = 0
    for scored in prediction.targets:
        best = scored.best
        if best is None:
            abstentions += 1
            continue
        chosen[scored.target] = (best.source, best.score)

    # In-degree is already at most one: each target selected a single option.
    # Out-degree is enforced here, keeping the strongest children.
    by_parent: dict[int, list[tuple[float, int]]] = {}
    for target_id, (source, score) in chosen.items():
        by_parent.setdefault(source, []).append((score, target_id))

    evictions = 0
    accepted: list[tuple[int, int]] = []
    for source, children in sorted(by_parent.items()):
        # Sort by descending score, breaking ties by ascending target id so the
        # decode is deterministic rather than dictionary-ordered.
        ranked = sorted(children, key=lambda item: (-item[0], item[1]))
        for rank, (_, target_id) in enumerate(ranked):
            if rank < max_children:
                accepted.append((source, target_id))
            else:
                evictions += 1

    out_degree: dict[int, int] = {}
    for source, _ in accepted:
        out_degree[source] = out_degree.get(source, 0) + 1

    nodes = tuple(
        LineageNode(
            dataset=instance.dataset,
            node_id=instance.instance_id,
            frame=instance.frame,
            voxel=instance.voxel,
        )
        for instance in sorted(instances.instances, key=lambda item: item.instance_id)
    )
    edges = tuple(
        LineageEdge(
            source_dataset=instances.dataset,
            source=source,
            target_dataset=instances.dataset,
            target=target,
            kind=EdgeKind.DIVISION if out_degree[source] == 2 else EdgeKind.CONTINUATION,
        )
        for source, target in sorted(accepted)
    )

    graph = LineageGraph(dataset=instances.dataset, nodes=nodes, edges=edges)

    has_parent = {target_id for _, target_id in accepted}
    divisions = sum(1 for count in out_degree.values() if count == 2)
    report = DecodeReport(
        nodes=len(nodes),
        edges=len(edges),
        continuations=sum(1 for edge in edges if edge.kind is EdgeKind.CONTINUATION),
        divisions=divisions,
        births=sum(1 for node in nodes if node.node_id not in has_parent),
        abstentions=abstentions,
        out_degree_evictions=evictions,
    )
    return graph, report
