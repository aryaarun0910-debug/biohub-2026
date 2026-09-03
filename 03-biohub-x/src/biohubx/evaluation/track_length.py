"""What a minimum-track-length filter costs in correct edges, before anything predicts.

[[F-0018]] measured the node-count lever as an oracle and left a bar: a real filter
reaching the annotated node count must retain about 0.94 of the correct edges
reaching it to buy the step from 0.936 to 0.950, and it stops paying below 0.92.
It did not say whether any implementable filter clears that bar.

Three public notebooks all prune short tracks and isolated nodes ([[R-0010]]), which
makes that policy the obvious first candidate. This measures its edge cost on the
annotated graphs themselves, which is the cheapest possible falsification: applied
to ground truth the filter meets a perfect graph, so whatever it destroys here it
destroys at best on a real one. A filter that already fails the bar against perfect
input cannot pass it against a detector's output.

The measurement is exact rather than an upper bound in one direction and a lower
bound in the other: on ground truth every edge is correct by construction, so the
fraction surviving is the retention term itself.

Consumer: ``biohubx evaluate track-length``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from biohubx.contracts.lineage import LineageGraph


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    """One connected lineage component: what it spans and whether it divides."""

    node_ids: frozenset[int]
    edge_count: int
    frame_span: int
    has_division: bool


def components(graph: LineageGraph) -> list[ComponentSummary]:
    """Split a lineage graph into connected components, ignoring edge direction.

    A component is a lineage tree: one founding cell and everything descended from
    it within the annotated window. Direction is ignored deliberately, because two
    daughters belong to their parent's track for the purpose of a filter that asks
    whether a piece of graph is long enough to be worth keeping.
    """
    parent: dict[int, int] = {node.node_id: node.node_id for node in graph.nodes}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for edge in graph.edges:
        left, right = find(edge.source), find(edge.target)
        if left != right:
            parent[left] = right

    frames = {node.node_id: node.frame for node in graph.nodes}
    outgoing: defaultdict[int, int] = defaultdict(int)
    for edge in graph.edges:
        outgoing[edge.source] += 1

    members: defaultdict[int, set[int]] = defaultdict(set)
    for node in graph.nodes:
        members[find(node.node_id)].add(node.node_id)

    edges_by_root: defaultdict[int, int] = defaultdict(int)
    for edge in graph.edges:
        edges_by_root[find(edge.source)] += 1

    summaries: list[ComponentSummary] = []
    for root, node_ids in members.items():
        spanned = {frames[node_id] for node_id in node_ids}
        summaries.append(
            ComponentSummary(
                node_ids=frozenset(node_ids),
                edge_count=edges_by_root[root],
                frame_span=max(spanned) - min(spanned) + 1,
                has_division=any(outgoing[node_id] >= 2 for node_id in node_ids),
            )
        )
    return summaries


@dataclass(frozen=True, slots=True)
class FilterCost:
    """What one threshold destroys, in the two currencies the metric charges."""

    minimum_frames: int
    keep_divisions: bool
    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int

    @property
    def edge_retention(self) -> float:
        return 1.0 if self.edges_before == 0 else self.edges_after / self.edges_before

    @property
    def node_retention(self) -> float:
        return 1.0 if self.nodes_before == 0 else self.nodes_after / self.nodes_before


def filter_cost(graph: LineageGraph, *, minimum_frames: int, keep_divisions: bool) -> FilterCost:
    """Remove components spanning fewer than ``minimum_frames`` and count the loss.

    ``keep_divisions`` mirrors the exemption the public notebooks carry, which
    keeps a short component that contains a division. Divisions are scarce
    ([[F-0014]]: 151 in the whole corpus) and carry their own metric term, so
    pruning one to save node budget is a trade the exemption declines to make.
    """
    summaries = components(graph)
    kept = [
        summary
        for summary in summaries
        if summary.frame_span >= minimum_frames or (keep_divisions and summary.has_division)
    ]
    return FilterCost(
        minimum_frames=minimum_frames,
        keep_divisions=keep_divisions,
        nodes_before=len(graph.nodes),
        nodes_after=sum(len(summary.node_ids) for summary in kept),
        edges_before=len(graph.edges),
        edges_after=sum(summary.edge_count for summary in kept),
    )
