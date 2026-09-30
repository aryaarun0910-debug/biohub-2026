"""Every annotated cell a proposal source failed to reach, described in terms an arm can act on.

A miss is not one kind of thing. A cell can be dim, merged into a neighbour's
response basin, lost at a volume face, found but placed too far away, present
in the frames either side but not this one, sitting on a peak the quantile
threshold discarded, or annotated where nothing in the image supports it. Each
of those motivates a different arm and kills others. This module measures the
features that separate them and applies a fixed, ordered classification, so
the atlas over a whole embryo is reproducible and its classes are auditable.

The classification motivates arms; it is not itself a finding until a
preregistered probe reproduces what it suggests. Consumer: ``biohubx evaluate
miss-atlas`` and the learned-proposal experiment family it feeds.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.contracts.instances import InstanceSet
from biohubx.contracts.lineage import LineageGraph
from biohubx.evaluation.oracle import match_one_to_one
from biohubx.evaluation.proposals import OFFICIAL_MATCH_RADIUS_UM
from biohubx.proposals.dog import dog_response, isotropic_plane_stride, normalise_frame

CLASSES: tuple[str, ...] = (
    "boundary",
    "merged",
    "dim",
    "localization",
    "temporal_dropout",
    "low_response",
    "annotation_ambiguity",
    "unexplained",
)
"""Primary classes in the order they are tested. A miss carries every flag it
earns; its primary class is the first flag in this order, so the order is part
of the instrument and is recorded with every atlas."""


@dataclass(frozen=True)
class Thresholds:
    boundary_um: float = 8.0
    """Closer than this to any face of the volume: a face effect, whatever else is true."""
    dim_percentile: float = 0.80
    """F-0028 split the 6bba residual at this intensity percentile."""
    merge_neighbour_um: float = 8.0
    """Another annotated cell nearer than this in the same frame can share a response basin."""
    localization_um: float = 14.0
    """A proposal nearer than this, beyond the official 7, is a placement error rather than an absence."""
    low_response_um: float = 7.0
    """A strict per-scale maximum within the official radius that the quantile threshold dropped."""

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MissRecord:
    dataset: str
    embryo: str
    frame: int
    node_id: int
    voxel_zyx: tuple[float, float, float]
    physical_um_zyx: tuple[float, float, float]
    intensity_percentile: float
    response_by_scale: dict[str, float]
    response_max: float
    response_percentile: float
    frame_cutoff: float
    nearest_proposal_um: float
    nearest_vector_um_zyx: tuple[float, float, float]
    nearest_proposal_matched_to_other: bool
    boundary_um: float
    neighbour_distances_um: list[float]
    neighbours_within_10um: int
    has_previous: bool
    has_next: bool
    previous_matched: bool | None
    next_matched: bool | None
    local_max_within_radius: bool
    local_max_um: float
    local_max_response: float
    local_max_below_cutoff: bool
    flags: list[str] = field(default_factory=list)
    primary_class: str = "unexplained"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(record: MissRecord, thresholds: Thresholds) -> tuple[list[str], str]:
    """Every flag a miss earns, and the first of them in CLASSES order.

    Pure function of the measured features, so the rule can be tested on
    constructed records and read in one place.
    """
    flags: list[str] = []
    if record.boundary_um < thresholds.boundary_um:
        flags.append("boundary")
    near_neighbour = (
        bool(record.neighbour_distances_um)
        and record.neighbour_distances_um[0] < thresholds.merge_neighbour_um
    )
    within_localisation = OFFICIAL_MATCH_RADIUS_UM < record.nearest_proposal_um <= thresholds.localization_um
    if within_localisation and (record.nearest_proposal_matched_to_other or near_neighbour):
        flags.append("merged")
    if record.intensity_percentile < thresholds.dim_percentile:
        flags.append("dim")
    if within_localisation and "merged" not in flags:
        flags.append("localization")
    neighbours_present = [m for m in (record.previous_matched, record.next_matched) if m is not None]
    if neighbours_present and all(neighbours_present):
        flags.append("temporal_dropout")
    if record.local_max_within_radius and record.local_max_below_cutoff:
        flags.append("low_response")
    if not record.has_previous and not record.has_next and not record.local_max_within_radius:
        flags.append("annotation_ambiguity")
    primary = next((c for c in CLASSES if c in flags), "unexplained")
    return flags, primary


def _strict_maxima(response: np.ndarray) -> np.ndarray:
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = False
    neighbours = maximum_filter(response, footprint=footprint, mode="constant", cval=-np.inf)
    result: np.ndarray = (response > neighbours) & (response > 0.0)
    return result


def audit_window(
    *,
    dataset_id: str,
    embryo: str,
    volume: np.ndarray,
    annotated: LineageGraph,
    instances: InstanceSet,
    first_frame: int,
    radii_um: tuple[float, ...],
    response_quantile: float,
    thresholds: Thresholds | None = None,
    radius_um: float = OFFICIAL_MATCH_RADIUS_UM,
) -> list[MissRecord]:
    """Describe every annotated node in the window that no proposal matched one-to-one."""
    thresholds = thresholds or Thresholds()
    matches = match_one_to_one(instances, annotated, radius_um=radius_um)
    matched_instances = set(matches.values())
    scale = OFFICIAL_VOXEL_SCALE
    stride = isotropic_plane_stride(scale)
    isotropic_um = float(scale.z_um)
    frames, depth, height, width = volume.shape
    extent_um = (depth * scale.z_um, height * scale.y_um, width * scale.x_um)

    predecessors: dict[int, list[int]] = {}
    successors: dict[int, list[int]] = {}
    for edge in annotated.edges:
        successors.setdefault(edge.source, []).append(edge.target)
        predecessors.setdefault(edge.target, []).append(edge.source)

    proposals_by_frame: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for instance in instances.instances:
        p = instance.physical
        proposals_by_frame.setdefault(instance.frame, []).append(
            (instance.instance_id, (p.z_um, p.y_um, p.x_um))
        )
    nodes_by_frame: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for node in annotated.nodes:
        p = node.voxel.to_physical(scale)
        nodes_by_frame.setdefault(node.frame, []).append((node.node_id, (p.z_um, p.y_um, p.x_um)))

    frame_cache: dict[int, dict[str, Any]] = {}

    def frame_features(local: int) -> dict[str, Any]:
        if local in frame_cache:
            return frame_cache[local]
        plane = volume[local][:, ::stride, ::stride]
        normalised = normalise_frame(plane)
        per_scale = {r: dog_response(normalised, radii_um=(r,), voxel_um=isotropic_um) for r in radii_um}
        combined = dog_response(normalised, radii_um=radii_um, voxel_um=isotropic_um)
        finite = combined[np.isfinite(combined)]
        cutoff = float(np.quantile(finite, response_quantile)) if finite.size else float("nan")
        features = {
            "per_scale": per_scale,
            "combined": combined,
            "cutoff": cutoff,
            "maxima": {r: _strict_maxima(per_scale[r]) for r in radii_um},
            "sorted_intensity": np.sort(volume[local].ravel()),
            "sorted_response": np.sort(finite),
        }
        frame_cache[local] = features
        return features

    search = math.ceil(thresholds.low_response_um / isotropic_um)
    records: list[MissRecord] = []
    for node in annotated.nodes:
        if node.node_id in matches:
            continue
        local = node.frame - first_frame
        if local < 0 or local >= frames:
            continue
        physical = node.voxel.to_physical(scale)
        target = (physical.z_um, physical.y_um, physical.x_um)
        features = frame_features(local)

        zi = int(np.clip(round(node.voxel.z), 0, depth - 1))
        yi = int(np.clip(round(node.voxel.y), 0, height - 1))
        xi = int(np.clip(round(node.voxel.x), 0, width - 1))
        gz, gy, gx = (
            zi,
            int(np.clip(yi // stride, 0, features["combined"].shape[1] - 1)),
            int(np.clip(xi // stride, 0, features["combined"].shape[2] - 1)),
        )
        value = float(volume[local][zi, yi, xi])
        ranks = features["sorted_intensity"]
        intensity_pct = float(np.searchsorted(ranks, value, side="right") / ranks.size)
        response_max = float(features["combined"][gz, gy, gx])
        response_pct = (
            float(
                np.searchsorted(features["sorted_response"], response_max, side="right")
                / features["sorted_response"].size
            )
            if features["sorted_response"].size
            else float("nan")
        )
        response_by_scale = {f"{r:g}um": float(features["per_scale"][r][gz, gy, gx]) for r in radii_um}

        nearest = math.inf
        nearest_vector = (0.0, 0.0, 0.0)
        nearest_id: int | None = None
        for instance_id, candidate in proposals_by_frame.get(node.frame, []):
            distance = math.dist(target, candidate)
            if distance < nearest:
                nearest = distance
                nearest_vector = (
                    candidate[0] - target[0],
                    candidate[1] - target[1],
                    candidate[2] - target[2],
                )
                nearest_id = instance_id
        shares_basin = nearest_id is not None and nearest_id in matched_instances

        boundary = min(
            target[0],
            extent_um[0] - target[0],
            target[1],
            extent_um[1] - target[1],
            target[2],
            extent_um[2] - target[2],
        )
        neighbours = sorted(
            math.dist(target, other)
            for other_id, other in nodes_by_frame.get(node.frame, [])
            if other_id != node.node_id
        )
        within_10 = sum(1 for d in neighbours if d <= 10.0)

        prev_ids = predecessors.get(node.node_id, [])
        next_ids = successors.get(node.node_id, [])
        has_previous = bool(prev_ids)
        has_next = bool(next_ids)
        previous_matched = any(p in matches for p in prev_ids) if prev_ids else None
        next_matched = any(n in matches for n in next_ids) if next_ids else None

        best_um = math.inf
        best_response = float("-inf")
        combined = features["combined"]
        z0, z1 = max(0, gz - search), min(combined.shape[0], gz + search + 1)
        y0, y1 = max(0, gy - search), min(combined.shape[1], gy + search + 1)
        x0, x1 = max(0, gx - search), min(combined.shape[2], gx + search + 1)
        for r in radii_um:
            box = features["maxima"][r][z0:z1, y0:y1, x0:x1]
            for dz, dy, dx in zip(*np.nonzero(box), strict=True):
                pz, py, px = z0 + int(dz), y0 + int(dy), x0 + int(dx)
                distance = isotropic_um * math.sqrt((pz - gz) ** 2 + (py - gy) ** 2 + (px - gx) ** 2)
                if distance <= thresholds.low_response_um:
                    value_here = float(combined[pz, py, px])
                    if value_here > best_response:
                        best_response = value_here
                        best_um = distance
        local_max_within = math.isfinite(best_um)

        record = MissRecord(
            dataset=dataset_id,
            embryo=embryo,
            frame=node.frame,
            node_id=int(node.node_id),
            voxel_zyx=(float(node.voxel.z), float(node.voxel.y), float(node.voxel.x)),
            physical_um_zyx=(round(target[0], 3), round(target[1], 3), round(target[2], 3)),
            intensity_percentile=round(intensity_pct, 4),
            response_by_scale={k: round(v, 6) for k, v in response_by_scale.items()},
            response_max=round(response_max, 6),
            response_percentile=round(response_pct, 4),
            frame_cutoff=round(features["cutoff"], 6),
            nearest_proposal_um=round(nearest, 3) if math.isfinite(nearest) else math.inf,
            nearest_vector_um_zyx=(
                round(nearest_vector[0], 3),
                round(nearest_vector[1], 3),
                round(nearest_vector[2], 3),
            ),
            nearest_proposal_matched_to_other=shares_basin,
            boundary_um=round(boundary, 3),
            neighbour_distances_um=[round(d, 3) for d in neighbours[:3]],
            neighbours_within_10um=within_10,
            has_previous=has_previous,
            has_next=has_next,
            previous_matched=previous_matched,
            next_matched=next_matched,
            local_max_within_radius=local_max_within,
            local_max_um=round(best_um, 3) if local_max_within else math.inf,
            local_max_response=round(best_response, 6) if local_max_within else float("nan"),
            local_max_below_cutoff=local_max_within and best_response < features["cutoff"],
        )
        record.flags, record.primary_class = classify(record, thresholds)
        records.append(record)
    return records


def representative(records: list[MissRecord], per_class: int = 3) -> dict[str, list[dict[str, Any]]]:
    """The clearest few misses of each class, chosen by the feature that defines the class."""
    key = {
        "boundary": lambda r: r.boundary_um,
        "merged": lambda r: r.neighbour_distances_um[0] if r.neighbour_distances_um else math.inf,
        "dim": lambda r: r.intensity_percentile,
        "localization": lambda r: r.nearest_proposal_um,
        "temporal_dropout": lambda r: -r.intensity_percentile,
        "low_response": lambda r: (
            -(r.local_max_response if math.isfinite(r.local_max_response) else -math.inf)
        ),
        "annotation_ambiguity": lambda r: r.intensity_percentile,
        "unexplained": lambda r: -r.intensity_percentile,
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for cls in CLASSES:
        members = sorted((r for r in records if r.primary_class == cls), key=key[cls])
        out[cls] = [
            {
                "dataset": r.dataset,
                "frame": r.frame,
                "node_id": r.node_id,
                "voxel_zyx": list(r.voxel_zyx),
                "physical_um_zyx": list(r.physical_um_zyx),
                "intensity_percentile": r.intensity_percentile,
                "nearest_proposal_um": r.nearest_proposal_um,
                "flags": r.flags,
            }
            for r in members[:per_class]
        ]
    return out


def summarise(records: list[MissRecord], annotated_nodes: int) -> dict[str, Any]:
    primary = Counter(r.primary_class for r in records)
    flags = Counter(flag for r in records for flag in r.flags)
    return {
        "annotated_nodes": annotated_nodes,
        "misses": len(records),
        "miss_fraction": round(len(records) / annotated_nodes, 4) if annotated_nodes else 0.0,
        "primary_class_counts": {c: primary.get(c, 0) for c in CLASSES},
        "flag_counts": {c: flags.get(c, 0) for c in CLASSES if c != "unexplained"},
        "class_order": list(CLASSES),
        "median_intensity_percentile": float(np.median([r.intensity_percentile for r in records]))
        if records
        else None,
        "median_nearest_proposal_um": float(
            np.median([r.nearest_proposal_um for r in records if math.isfinite(r.nearest_proposal_um)])
        )
        if any(math.isfinite(r.nearest_proposal_um) for r in records)
        else None,
        "with_local_max_within_radius": sum(1 for r in records if r.local_max_within_radius),
        "with_low_response_local_max": sum(1 for r in records if r.local_max_below_cutoff),
    }
