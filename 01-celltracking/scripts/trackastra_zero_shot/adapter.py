"""Data adapters for evaluating Trackastra on *frozen* Biohub detections.

Trackastra expects instance masks, whereas the competition detector produces
points.  This module rasterises each point as a physically spherical object
(therefore an anisotropic ellipsoid in voxel space), preserves an explicit
``(time, mask_label) -> source node`` mapping, and converts Trackastra's chosen
links back to a GEFF graph at the original point coordinates.

The Trackastra import is intentionally absent from this module.  Unit tests and
``--dry-run`` therefore work before the offline package/model bundle is staged.
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_VOXEL_SIZE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class FrozenDetections:
    """Point detections in the competition's voxel coordinate system."""

    node_ids: np.ndarray
    time: np.ndarray
    zyx: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.node_ids, dtype=np.int64)
        time = np.asarray(self.time, dtype=np.int64)
        zyx = np.asarray(self.zyx, dtype=np.float64)
        if ids.ndim != 1 or time.ndim != 1 or zyx.ndim != 2 or zyx.shape[1:] != (3,):
            raise ValueError("expected node_ids=(N,), time=(N,), zyx=(N,3)")
        if not (len(ids) == len(time) == len(zyx)):
            raise ValueError("node_ids, time and zyx lengths differ")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("node_ids must be unique")
        if np.any(time < 0):
            raise ValueError("time indices must be non-negative")
        if not np.isfinite(zyx).all():
            raise ValueError("coordinates must be finite")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "zyx", zyx)

    def validate_shape(self, shape: Sequence[int]) -> None:
        if len(shape) != 4:
            raise ValueError("image shape must be (T,Z,Y,X)")
        if len(self.node_ids) == 0:
            return
        if int(self.time.max()) >= int(shape[0]):
            raise ValueError(f"detection time {self.time.max()} exceeds T={shape[0]}")
        spatial = np.asarray(shape[1:], dtype=np.float64)
        bad = np.any((self.zyx < 0) | (self.zyx >= spatial), axis=1)
        if np.any(bad):
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"node {self.node_ids[i]} at {self.zyx[i]} is outside {tuple(shape)}")


@dataclass(frozen=True)
class MaskLabel:
    """Identity bridge between one mask instance and one frozen detection."""

    detection_index: int
    source_node_id: int
    time: int
    label: int


def _load_geff(path: Path) -> FrozenDetections:
    try:
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError("GEFF input requires the existing tracksdata environment") from exc

    result = td.graph.IndexedRXGraph.from_geff(path)
    graph = result[0] if isinstance(result, tuple) else result
    attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"]
    )
    key = td.DEFAULT_ATTR_KEYS.NODE_ID
    return FrozenDetections(
        node_ids=np.asarray(attrs[key].to_list(), dtype=np.int64),
        time=np.asarray(attrs["t"].to_list(), dtype=np.int64),
        zyx=np.column_stack(
            [attrs[axis].to_numpy() for axis in ("z", "y", "x")]
        ),
    )


def _load_csv(path: Path) -> FrozenDetections:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)]
    required = {"node_id", "t", "z", "y", "x"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"point CSV needs columns {sorted(required)}")
    return FrozenDetections(
        node_ids=np.asarray([int(r["node_id"]) for r in rows], dtype=np.int64),
        time=np.asarray([int(r["t"]) for r in rows], dtype=np.int64),
        zyx=np.asarray(
            [[float(r["z"]), float(r["y"]), float(r["x"])] for r in rows],
            dtype=np.float64,
        ).reshape(-1, 3),
    )


def load_frozen_detections(path: str | Path) -> FrozenDetections:
    """Load frozen point detections from a predicted GEFF or point CSV.

    Existing GEFF edges are deliberately ignored: this experiment changes only
    association.  The organizer learned pipeline currently exports these files
    as ``pred_geffs_split_<fold>/<crop>.geff``.
    """

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".geff" or path.name.endswith(".geff"):
        return _load_geff(path)
    raise ValueError("detections must be a .geff directory/file or point .csv")


def _mask_dtype(max_labels_per_frame: int) -> np.dtype:
    return np.dtype(np.uint16 if max_labels_per_frame <= np.iinfo(np.uint16).max else np.uint32)


def rasterize_ellipsoid_masks(
    detections: FrozenDetections,
    shape: Sequence[int],
    *,
    radius_um: float = 2.5,
    voxel_size_um: Sequence[float] = DEFAULT_VOXEL_SIZE_UM,
    output_npy: str | Path | None = None,
) -> tuple[np.ndarray, list[MaskLabel]]:
    """Rasterise points into a Trackastra-compatible instance mask.

    Overlaps are resolved by nearest normalised ellipsoid distance.  Only one
    frame-sized float buffer is needed, so a 100x64x256x256 mask can be written
    as a ~0.8 GiB uint16 memmap without allocating a second full volume.
    Labels restart at one in every frame, as expected for instance masks.
    """

    shape = tuple(int(v) for v in shape)
    detections.validate_shape(shape)
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")
    voxel = np.asarray(voxel_size_um, dtype=np.float64)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError("voxel_size_um must contain three positive values")
    radii = float(radius_um) / voxel

    frame_counts = np.bincount(detections.time, minlength=shape[0]) if len(detections.time) else np.zeros(shape[0], int)
    dtype = _mask_dtype(int(frame_counts.max(initial=0)))
    if output_npy is None:
        masks = np.zeros(shape, dtype=dtype)
    else:
        output_npy = Path(output_npy)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        masks = np.lib.format.open_memmap(output_npy, mode="w+", dtype=dtype, shape=shape)
        masks[:] = 0

    mapping: list[MaskLabel] = []
    spatial_shape = np.asarray(shape[1:], dtype=np.int64)
    for t in range(shape[0]):
        indices = np.flatnonzero(detections.time == t)
        if len(indices) == 0:
            continue
        # Stable source-id ordering makes masks reproducible across GEFF readers.
        indices = indices[np.argsort(detections.node_ids[indices], kind="stable")]
        nearest = np.full(shape[1:], np.inf, dtype=np.float32)
        frame = masks[t]

        for label, det_i in enumerate(indices, start=1):
            center = detections.zyx[det_i]
            lo = np.maximum(0, np.floor(center - radii).astype(np.int64))
            hi = np.minimum(spatial_shape, np.ceil(center + radii).astype(np.int64) + 1)
            grids = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
            dist2 = sum(((g - center[axis]) / radii[axis]) ** 2 for axis, g in enumerate(grids))
            inside = dist2 <= 1.0
            sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            local_best = nearest[sl]
            wins = inside & (dist2 < local_best)
            frame[sl][wins] = label
            local_best[wins] = dist2[wins]
            mapping.append(
                MaskLabel(int(det_i), int(detections.node_ids[det_i]), int(t), int(label))
            )

        # A sub-voxel object must survive rasterisation.  Distinct frozen points
        # landing on the same integer voxel are ambiguous as instance masks and
        # are rejected instead of silently dropping a detection.
        centres = np.rint(detections.zyx[indices]).astype(np.int64)
        centres = np.minimum(np.maximum(centres, 0), spatial_shape - 1)
        if len(np.unique(centres, axis=0)) != len(centres):
            raise ValueError(f"two detections in frame {t} round to the same mask voxel")
        for label, centre in enumerate(centres, start=1):
            if frame[tuple(centre)] != label:
                frame[tuple(centre)] = label
        present = np.unique(frame)
        if len(present) - (1 if present[0] == 0 else 0) != len(indices):
            raise RuntimeError(f"rasterisation lost an instance in frame {t}")

    if hasattr(masks, "flush"):
        masks.flush()
    return masks, mapping


def write_mask_mapping(mapping: Iterable[MaskLabel], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["detection_index", "source_node_id", "time", "label"],
        )
        writer.writeheader()
        writer.writerows(m.__dict__ for m in mapping)
    return path


def export_trackastra_result(
    detections: FrozenDetections,
    mapping: Sequence[MaskLabel],
    predictions: Mapping,
    selected_graph,
    *,
    output_geff: str | Path,
    output_edge_scores: str | Path,
) -> tuple[Path, Path]:
    """Export chosen Trackastra links while restoring frozen point coordinates.

    ``predictions`` is the result of Trackastra 0.5.2's ``model._predict`` and
    ``selected_graph`` is returned by ``model._track_from_predictions``.  The
    private methods are used because the public ``track`` API discards candidate
    association scores; runtime validation in ``run.py`` makes API drift loud.
    """

    try:
        import polars as pl
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GEFF export requires the repo's polars/tracksdata environment") from exc

    by_time_label = {(m.time, m.label): m.detection_index for m in mapping}
    pred_nodes = {int(n["id"]): n for n in predictions["nodes"]}
    pred_to_detection: dict[int, int] = {}
    for pred_id, node in pred_nodes.items():
        key = (int(node["time"]), int(node["label"]))
        if key not in by_time_label:
            raise ValueError(f"Trackastra emitted unknown mask instance {key}")
        pred_to_detection[pred_id] = by_time_label[key]

    candidate_scores = {
        (int(edge[0]), int(edge[1])): float(score)
        for edge, score in predictions["weights"]
    }
    selected_edges = [(int(u), int(v)) for u, v in selected_graph.edges()]
    unknown = {n for edge in selected_edges for n in edge if n not in pred_to_detection}
    if unknown:
        raise ValueError(f"selected graph contains unknown node ids: {sorted(unknown)[:5]}")

    graph = td.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Float64, -999999.0)
    internal = graph.bulk_add_nodes(
        [
            {
                "t": int(t),
                "z": float(c[0]),
                "y": float(c[1]),
                "x": float(c[2]),
            }
            for t, c in zip(detections.time, detections.zyx)
        ]
    )
    if selected_edges:
        graph.add_edge_attr_key("trackastra_score", pl.Float64, 0.0)
        graph.bulk_add_edges(
            [
                {
                    "source_id": internal[pred_to_detection[u]],
                    "target_id": internal[pred_to_detection[v]],
                    "trackastra_score": candidate_scores.get((u, v), 0.0),
                }
                for u, v in selected_edges
            ]
        )

    output_geff = Path(output_geff)
    output_geff.parent.mkdir(parents=True, exist_ok=True)
    if output_geff.exists():
        if output_geff.is_dir():
            shutil.rmtree(output_geff)
        else:
            output_geff.unlink()
    graph.to_geff(output_geff)

    output_edge_scores = Path(output_edge_scores)
    output_edge_scores.parent.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected_edges)
    with output_edge_scores.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_node_id", "target_node_id", "score", "selected"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (u, v), score in sorted(candidate_scores.items()):
            if u not in pred_to_detection or v not in pred_to_detection:
                continue
            writer.writerow(
                {
                    "source_node_id": int(detections.node_ids[pred_to_detection[u]]),
                    "target_node_id": int(detections.node_ids[pred_to_detection[v]]),
                    "score": score,
                    "selected": int((u, v) in selected_set),
                }
            )
    return output_geff, output_edge_scores


def estimated_mask_bytes(shape: Sequence[int], detections: FrozenDetections) -> int:
    counts = np.bincount(detections.time, minlength=int(shape[0])) if len(detections.time) else np.zeros(int(shape[0]), int)
    return math.prod(int(v) for v in shape) * _mask_dtype(int(counts.max(initial=0))).itemsize
