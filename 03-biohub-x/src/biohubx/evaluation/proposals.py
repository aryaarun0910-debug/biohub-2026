"""How many annotated cells a proposal source actually reaches, at a fixed budget.

[[F-0017]] established that reach, not association, was binding: every annotated
edge the chain could not reach lacked a proposed endpoint. The quantity that
matters is therefore whether an annotated node has *any* proposal near enough to
be matched, which is what this measures.

Reachability upper-bounds the official one-to-one node recall, and the gap is
real: the same classical detector reads 0.3846 here and 0.269 through the official
adapter on the same window ([[F-0025]]). It is used anyway, for one reason. An
annotated endpoint with no nearby proposal cannot be matched by any association
method whatsoever, so reachability is the ceiling every later stage works under,
and a source that raises it is the only thing that can raise that ceiling. It is
never called node recall.

The budget comes from official metadata. Estimated cell counts span two orders of
magnitude across this corpus ([[F-0013]]), so a fixed response threshold yields a
node ratio near one on a dense movie and near ten on a sparse one, and any
comparison built on it would confound detector quality with how much it was
allowed to emit. Truncating every source to the same budget removes that.

Consumer: ``biohubx evaluate proposals``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.contracts.instances import CandidateInstance, InstanceSet
from biohubx.contracts.lineage import LineageGraph

OFFICIAL_MATCH_RADIUS_UM = 7.0
"""The radius the official metric matches within, in physical coordinates.

Not a tuned parameter. It is the competition's own matching tolerance, and using
anything else here would measure a question the metric does not ask.
"""


def truncate_to_budget(instances: InstanceSet, budget: int) -> InstanceSet:
    """Keep the highest-confidence proposals up to ``budget``, renumbering densely.

    Ordering is by descending confidence and then by identity, so two runs over
    the same input truncate to the same set rather than to an arbitrary one of
    several equally-confident subsets.
    """
    if budget < 0:
        raise ValueError(f"a node budget cannot be negative, got {budget}")
    ranked = sorted(instances.instances, key=lambda i: (-i.confidence, i.instance_id))[:budget]
    kept: list[CandidateInstance] = [
        instance.model_copy(update={"instance_id": index}) for index, instance in enumerate(ranked)
    ]
    return InstanceSet(dataset=instances.dataset, instances=tuple(kept))


@dataclass(frozen=True, slots=True)
class Reachability:
    """What a proposal set reaches, and what it cost in node budget."""

    annotated_nodes: int
    reached: int
    proposals: int
    estimated_total_nodes: float

    @property
    def reachability(self) -> float:
        return 0.0 if self.annotated_nodes == 0 else self.reached / self.annotated_nodes

    @property
    def node_ratio(self) -> float:
        """The official adjustment's ratio, positive when over-predicting."""
        if self.estimated_total_nodes <= 0:
            raise ValueError("an estimated total of zero cannot be a denominator")
        return (self.proposals - self.estimated_total_nodes) / self.estimated_total_nodes


def measure_reachability(
    instances: InstanceSet,
    annotated: LineageGraph,
    *,
    estimated_total_nodes: float,
    radius_um: float = OFFICIAL_MATCH_RADIUS_UM,
) -> Reachability:
    """Count annotated nodes with a proposal inside the official radius.

    Compared frame by frame, because the metric matches within a frame and a
    proposal one frame away is not a candidate for the same cell however close it
    lies in space.
    """
    by_frame: dict[int, list[tuple[float, float, float]]] = {}
    for instance in instances.instances:
        point = instance.physical
        by_frame.setdefault(instance.frame, []).append((point.z_um, point.y_um, point.x_um))

    reached = 0
    for node in annotated.nodes:
        candidates = by_frame.get(node.frame)
        if not candidates:
            continue
        physical = node.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        target = (physical.z_um, physical.y_um, physical.x_um)
        nearest = min(math.dist(target, candidate) for candidate in candidates)
        if nearest <= radius_um:
            reached += 1
    return Reachability(
        annotated_nodes=len(annotated.nodes),
        reached=reached,
        proposals=len(instances.instances),
        estimated_total_nodes=estimated_total_nodes,
    )


def normalise_for_classical(volume: np.ndarray) -> np.ndarray:
    """Scale a raw volume onto [0, 1], which the classical detector's threshold assumes.

    The classical detector takes an absolute threshold in (0, 1), so it needs a
    normalised volume; the DoG source normalises per frame internally and takes
    the raw one. Feeding the raw volume to the classical detector would make its
    threshold meaningless and the comparison worthless.
    """
    low = float(volume.min())
    high = float(volume.max())
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume.astype(np.float32) - low) / (high - low)).astype(np.float32)
