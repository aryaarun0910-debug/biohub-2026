"""Why a proposal source missed an annotated cell, one cell at a time.

[[F-0026]] left about a fifth of annotated cells without a DoG proposal within the
matching radius. Whether that fifth is dim, small, crowded, at the volume edge or
just outside the scale bank decides whether another detector could plausibly
complement DoG or would only re-find what DoG already finds. Guessing at that is
cheap and wrong; measuring it per node is cheap and informative.

Every feature here is computed from the volume and the annotations alone, on the
same isotropic grid the detector uses, so a miss is described in the detector's
own terms.

Consumer: ``biohubx evaluate proposals --audit-misses``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.contracts.instances import InstanceSet
from biohubx.contracts.lineage import LineageGraph
from biohubx.evaluation.proposals import OFFICIAL_MATCH_RADIUS_UM


@dataclass(frozen=True, slots=True)
class NodeAudit:
    dataset: str
    embryo: str
    frame: int
    reached: bool
    nearest_proposal_um: float
    # Signed displacement from the annotated cell to its nearest proposal, in
    # micrometres, so a miss can be read as along z, where the grid is coarsest,
    # or in the plane. Distance alone cannot tell those apart, and they call for
    # different fixes.
    nearest_dz_um: float
    nearest_dy_um: float
    nearest_dx_um: float
    intensity_percentile: float
    dog_response_percentile: float
    is_local_maximum_on_grid: bool
    crowding_within_10um: int
    boundary_distance_um: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def one_to_one_recall(
    instances: InstanceSet, annotated: LineageGraph, *, radius_um: float = OFFICIAL_MATCH_RADIUS_UM
) -> tuple[int, int]:
    """Annotated nodes matched one-to-one within the radius, per frame.

    A minimum-cost assignment per frame with a hard cap, so one proposal cannot
    satisfy two annotated cells. This is the quantity reachability upper-bounds,
    and the two are reported side by side so their gap is visible rather than
    argued about. It is not the official function, which needs a scored graph;
    it is the same matching rule applied to nodes alone.
    """
    proposals: dict[int, list[tuple[float, float, float]]] = {}
    for instance in instances.instances:
        point = instance.physical
        proposals.setdefault(instance.frame, []).append((point.z_um, point.y_um, point.x_um))
    targets: dict[int, list[tuple[float, float, float]]] = {}
    for node in annotated.nodes:
        point = node.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        targets.setdefault(node.frame, []).append((point.z_um, point.y_um, point.x_um))

    matched = 0
    for frame, annotated_points in targets.items():
        candidates = proposals.get(frame)
        if not candidates:
            continue
        cost = np.array(
            [[math.dist(target, candidate) for candidate in candidates] for target in annotated_points]
        )
        rows, cols = linear_sum_assignment(cost)
        matched += int(np.sum(cost[rows, cols] <= radius_um))
    return matched, len(annotated.nodes)


def audit_nodes(
    *,
    dataset_id: str,
    embryo: str,
    volume: np.ndarray,
    instances: InstanceSet,
    annotated: LineageGraph,
    response_per_frame: dict[int, np.ndarray],
    plane_stride: int,
    first_frame: int,
    radius_um: float = OFFICIAL_MATCH_RADIUS_UM,
) -> list[NodeAudit]:
    """Describe every annotated node in terms the detector can act on.

    ``response_per_frame`` maps a window-relative frame index to the DoG response
    on the strided grid, so the response at a node is read from exactly the map
    the detector thresholded. ``volume`` is the raw window; percentiles are taken
    within the node's own frame, which is invariant to any monotonic intensity
    normalisation ([[F-0020]]).
    """
    proposals: dict[int, list[tuple[float, float, float]]] = {}
    for instance in instances.instances:
        point = instance.physical
        proposals.setdefault(instance.frame, []).append((point.z_um, point.y_um, point.x_um))

    by_frame_targets: dict[int, list[tuple[float, float, float]]] = {}
    for node in annotated.nodes:
        point = node.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        by_frame_targets.setdefault(node.frame, []).append((point.z_um, point.y_um, point.x_um))

    _, depth, height, width = volume.shape
    scale = OFFICIAL_VOXEL_SCALE
    extent_um = (depth * scale.z_um, height * scale.y_um, width * scale.x_um)
    sorted_frames: dict[int, np.ndarray] = {}

    audits: list[NodeAudit] = []
    for node in annotated.nodes:
        local = node.frame - first_frame
        if local < 0 or local >= volume.shape[0]:
            continue
        physical = node.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        target = (physical.z_um, physical.y_um, physical.x_um)

        nearby = proposals.get(node.frame, [])
        nearest = math.inf
        offset = (0.0, 0.0, 0.0)
        for candidate in nearby:
            distance = math.dist(target, candidate)
            if distance < nearest:
                nearest = distance
                offset = (
                    candidate[0] - target[0],
                    candidate[1] - target[1],
                    candidate[2] - target[2],
                )

        frame = volume[local]
        if local not in sorted_frames:
            sorted_frames[local] = np.sort(frame.ravel())
        zi = int(np.clip(round(node.voxel.z), 0, depth - 1))
        yi = int(np.clip(round(node.voxel.y), 0, height - 1))
        xi = int(np.clip(round(node.voxel.x), 0, width - 1))
        value = float(frame[zi, yi, xi])
        ranks = sorted_frames[local]
        intensity_pct = float(np.searchsorted(ranks, value, side="right") / ranks.size)

        response = response_per_frame.get(local)
        response_pct = float("nan")
        is_max = False
        if response is not None:
            rz = int(np.clip(zi, 0, response.shape[0] - 1))
            ry = int(np.clip(yi // plane_stride, 0, response.shape[1] - 1))
            rx = int(np.clip(xi // plane_stride, 0, response.shape[2] - 1))
            at_node = float(response[rz, ry, rx])
            finite = response[np.isfinite(response)]
            response_pct = float(
                np.searchsorted(np.sort(finite.ravel()), at_node, side="right") / finite.size
            )
            z0, z1 = max(0, rz - 1), min(response.shape[0], rz + 2)
            y0, y1 = max(0, ry - 1), min(response.shape[1], ry + 2)
            x0, x1 = max(0, rx - 1), min(response.shape[2], rx + 2)
            is_max = bool(at_node >= float(np.max(response[z0:z1, y0:y1, x0:x1])))

        crowd = sum(
            1
            for other in by_frame_targets.get(node.frame, [])
            if other != target and math.dist(other, target) <= 10.0
        )
        boundary = min(
            target[0],
            extent_um[0] - target[0],
            target[1],
            extent_um[1] - target[1],
            target[2],
            extent_um[2] - target[2],
        )
        audits.append(
            NodeAudit(
                dataset=dataset_id,
                embryo=embryo,
                frame=node.frame,
                reached=nearest <= radius_um,
                nearest_proposal_um=round(nearest, 3) if math.isfinite(nearest) else -1.0,
                nearest_dz_um=round(offset[0], 3),
                nearest_dy_um=round(offset[1], 3),
                nearest_dx_um=round(offset[2], 3),
                intensity_percentile=round(intensity_pct, 4),
                dog_response_percentile=round(response_pct, 4) if math.isfinite(response_pct) else -1.0,
                is_local_maximum_on_grid=is_max,
                crowding_within_10um=crowd,
                boundary_distance_um=round(boundary, 3),
            )
        )
    return audits


def summarise(audits: list[NodeAudit]) -> dict[str, object]:
    """Medians of every feature for reached and missed nodes, per embryo."""
    out: dict[str, object] = {}
    for embryo in sorted({audit.embryo for audit in audits}):
        subset = [audit for audit in audits if audit.embryo == embryo]
        block: dict[str, object] = {}
        for label, group in (
            ("reached", [a for a in subset if a.reached]),
            ("missed", [a for a in subset if not a.reached]),
        ):
            if not group:
                block[label] = {"count": 0}
                continue

            def median(values: list[float]) -> float:
                return float(np.median(values)) if values else float("nan")

            block[label] = {
                "count": len(group),
                "median_intensity_percentile": round(median([a.intensity_percentile for a in group]), 4),
                "median_dog_response_percentile": round(
                    median([a.dog_response_percentile for a in group if a.dog_response_percentile >= 0]), 4
                ),
                "fraction_local_maximum_on_grid": round(
                    sum(a.is_local_maximum_on_grid for a in group) / len(group), 4
                ),
                "median_crowding_within_10um": round(
                    median([float(a.crowding_within_10um) for a in group]), 2
                ),
                "median_boundary_distance_um": round(median([a.boundary_distance_um for a in group]), 2),
                "median_nearest_proposal_um": round(
                    median([a.nearest_proposal_um for a in group if a.nearest_proposal_um >= 0]), 2
                ),
            }
        out[embryo] = block
    return out
