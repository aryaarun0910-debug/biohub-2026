"""Read competition datasets into Biohub-X contracts.

A ``.geff`` ground truth and its ``.zarr`` volume are both Zarr stores. They are
read here through ``zarr`` directly rather than through a format helper, so that
every assumption this repository makes about the official layout is written down
and checked: the GEFF version, the coordinate axes, the voxel scale and the
presence of the dataset's own node-count estimate. A layout change fails loudly
at the assertion that no longer holds instead of silently producing a graph.

Nothing here filters, scores or predicts. It converts official bytes into
validated contracts and refuses when it cannot.

Consumers: ``biohubx infer real`` and ``biohubx evaluate retention``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

SUPPORTED_GEFF_VERSION = "1.1"
"""The only GEFF version this reader claims to understand."""

NODE_ESTIMATE_KEY = "estimated_number_of_nodes"
"""The GEFF ``extra`` entry the official metric's denominator comes from."""

# ``zarr.open`` returns a union of array and group handles, and its element
# access is typed just as loosely. Every handle below is narrowed to ``Any``
# at the point it is opened, immediately after an explicit layout check that
# establishes what it actually is. Narrowing once at the boundary keeps the
# checks visible instead of scattering per-line ignores through the reader.


class CompetitionLayoutError(ValueError):
    """The official layout is not what this reader was written against."""


@dataclass(frozen=True, slots=True)
class DatasetGroundTruth:
    """One movie's annotated lineage plus the metadata the metric needs."""

    dataset: DatasetIdentity
    lineage: LineageGraph
    estimated_total_nodes: int
    frames: int
    embryo: str


@dataclass(frozen=True, slots=True)
class WindowSelection:
    """A tiny, explicitly bounded piece of one movie, declared before it is read."""

    dataset_id: str
    first_frame: int
    frames: int
    z_start: int
    z_stop: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "first_frame": self.first_frame,
            "frames": self.frames,
            "crop_z": [self.z_start, self.z_stop],
            "crop_y": [self.y_start, self.y_stop],
            "crop_x": [self.x_start, self.x_stop],
        }


@dataclass(frozen=True, slots=True)
class CompetitionWindow:
    """A window of real data, with its ground truth restricted to the same window."""

    selection: WindowSelection
    volume: np.ndarray
    annotated: LineageGraph
    dataset_estimated_total_nodes: int
    window_estimated_total_nodes: float
    dataset_frames: int
    intensity_low: float
    intensity_high: float
    dropped_nodes: int
    dropped_edges: int
    scale: VoxelScaleZYX

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection.to_dict(),
            "dataset_frames": self.dataset_frames,
            "dataset_estimated_total_nodes": self.dataset_estimated_total_nodes,
            "window_estimated_total_nodes": self.window_estimated_total_nodes,
            "intensity_normalisation_low": self.intensity_low,
            "intensity_normalisation_high": self.intensity_high,
            "annotated_nodes": len(self.annotated.nodes),
            "annotated_edges": len(self.annotated.edges),
            "dropped_nodes_outside_window": self.dropped_nodes,
            "dropped_edges_outside_window": self.dropped_edges,
        }


def competition_root(explicit: Path | None = None) -> Path:
    """Resolve the competition data root, refusing to guess."""
    if explicit is not None:
        root = explicit
    else:
        env = os.environ.get("BIOHUB_DATA_ROOT")
        if env is None:
            raise CompetitionLayoutError(
                "no competition data root given and BIOHUB_DATA_ROOT is unset; "
                "pass --root rather than letting the reader guess"
            )
        root = Path(env)
    if not root.is_dir():
        raise CompetitionLayoutError(f"competition data root does not exist: {root}")
    return root


def _geff_metadata(store: Any) -> dict[str, Any]:
    attrs = dict(store.attrs)
    if "geff" not in attrs:
        raise CompetitionLayoutError("ground truth carries no 'geff' metadata block")
    meta: dict[str, Any] = attrs["geff"]
    version = str(meta.get("geff_version", ""))
    if not version.startswith(SUPPORTED_GEFF_VERSION):
        raise CompetitionLayoutError(
            f"unsupported GEFF version {version!r}; this reader was written against {SUPPORTED_GEFF_VERSION}"
        )
    if not meta.get("directed", False):
        raise CompetitionLayoutError("ground-truth graph is undirected; lineage requires direction")
    return meta


def _axis_scale(meta: dict[str, Any]) -> VoxelScaleZYX:
    axes = {axis["name"]: axis for axis in meta["axes"]}
    missing = {"t", "z", "y", "x"} - set(axes)
    if missing:
        raise CompetitionLayoutError(f"ground truth is missing axes {sorted(missing)}")
    return VoxelScaleZYX(
        z_um=float(axes["z"]["scale"]), y_um=float(axes["y"]["scale"]), x_um=float(axes["x"]["scale"])
    )


def load_ground_truth(root: Path, dataset_id: str, *, split: str = "train") -> DatasetGroundTruth:
    """Read one movie's complete annotated lineage.

    Frames and coordinates stay exactly as the official file records them. The
    node-count estimate is the dataset's own ``estimated_number_of_nodes``; a
    dataset that does not carry one is refused rather than given a substitute,
    because the metric's denominator is not a quantity to invent.
    """
    path = root / split / f"{dataset_id}.geff"
    if not path.is_dir():
        raise CompetitionLayoutError(f"no ground truth at {path}")
    store: Any = zarr.open(str(path), mode="r")
    meta = _geff_metadata(store)
    scale = _axis_scale(meta)
    if scale != OFFICIAL_VOXEL_SCALE:
        raise CompetitionLayoutError(
            f"ground truth declares scale {scale}, official is {OFFICIAL_VOXEL_SCALE}"
        )

    extra = meta.get("extra") or {}
    if NODE_ESTIMATE_KEY not in extra:
        raise CompetitionLayoutError(
            f"{dataset_id} carries no {NODE_ESTIMATE_KEY}; the metric denominator is unknown"
        )

    ids = np.asarray(store["nodes/ids"][:], dtype=np.int64)
    frames = np.asarray(store["nodes/props/t/values"][:], dtype=np.int64)
    z = np.asarray(store["nodes/props/z/values"][:], dtype=np.float64)
    y = np.asarray(store["nodes/props/y/values"][:], dtype=np.float64)
    x = np.asarray(store["nodes/props/x/values"][:], dtype=np.float64)
    edges = np.asarray(store["edges/ids"][:], dtype=np.int64).reshape(-1, 2)

    dataset = DatasetIdentity(value=dataset_id)
    lineage = build_lineage(dataset=dataset, node_ids=ids, frames=frames, z=z, y=y, x=x, edges=edges)
    return DatasetGroundTruth(
        dataset=dataset,
        lineage=lineage,
        estimated_total_nodes=int(extra[NODE_ESTIMATE_KEY]),
        frames=int(frames.max()) + 1 if ids.size else 0,
        embryo=dataset_id.split("_", 1)[0],
    )


def build_lineage(
    *,
    dataset: DatasetIdentity,
    node_ids: np.ndarray,
    frames: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    edges: np.ndarray,
) -> LineageGraph:
    """Assemble a validated lineage, assigning edge kinds from the topology."""
    nodes = tuple(
        LineageNode(
            dataset=dataset,
            node_id=int(node_id),
            frame=int(frame),
            voxel=VoxelCoordinateZYX(z=float(zz), y=float(yy), x=float(xx)),
        )
        for node_id, frame, zz, yy, xx in zip(node_ids, frames, z, y, x, strict=True)
    )
    return LineageGraph(dataset=dataset, nodes=nodes, edges=edges_with_kinds(dataset, edges))


def edges_with_kinds(dataset: DatasetIdentity, edges: np.ndarray) -> tuple[LineageEdge, ...]:
    """Label each edge from its source's out-degree, as the lineage contract requires."""
    out_degree: dict[int, int] = {}
    for source, _ in edges:
        out_degree[int(source)] = out_degree.get(int(source), 0) + 1
    return tuple(
        LineageEdge(
            source_dataset=dataset,
            source=int(source),
            target_dataset=dataset,
            target=int(target),
            kind=EdgeKind.DIVISION if out_degree[int(source)] == 2 else EdgeKind.CONTINUATION,
        )
        for source, target in edges
    )


def load_window(root: Path, selection: WindowSelection, *, split: str = "train") -> CompetitionWindow:
    """Read one preregistered window of volume and its ground truth together.

    The window's node-count estimate is prorated from the movie's official
    estimate by the fraction of frames and the fraction of voxels selected. That
    proration assumes cells are distributed uniformly, which is an assumption
    and not a measurement, so the result is carried as a DECLARED estimate whose
    derivation is recorded rather than as official metadata.
    """
    truth = load_ground_truth(root, selection.dataset_id, split=split)
    volume_path = root / split / f"{selection.dataset_id}.zarr"
    if not volume_path.is_dir():
        raise CompetitionLayoutError(f"no volume at {volume_path}")
    group: Any = zarr.open(str(volume_path), mode="r")
    array: Any = group["0"]
    if array.ndim != 4:
        raise CompetitionLayoutError(f"expected a (T, Z, Y, X) volume, got shape {array.shape}")

    stop = selection.first_frame + selection.frames
    if stop > array.shape[0]:
        raise CompetitionLayoutError(
            f"window frames {selection.first_frame}..{stop - 1} exceed the movie's {array.shape[0]}"
        )
    statistics = dict(group.attrs).get("image_statistics")
    if not statistics or "quantiles" not in statistics:
        raise CompetitionLayoutError(f"{selection.dataset_id} carries no image_statistics to normalise with")
    quantiles = statistics["quantiles"]
    low, high = float(quantiles["0.001"]), float(quantiles["0.999"])
    if not high > low:
        raise CompetitionLayoutError(f"degenerate intensity window [{low}, {high}]")

    raw = np.asarray(
        array[
            selection.first_frame : stop,
            selection.z_start : selection.z_stop,
            selection.y_start : selection.y_stop,
            selection.x_start : selection.x_stop,
        ],
        dtype=np.float32,
    )
    volume = np.clip((raw - low) / (high - low), 0.0, 1.0)

    annotated, dropped_nodes, dropped_edges = _restrict(truth.lineage, selection)
    if not annotated.edges:
        raise CompetitionLayoutError(
            f"window holds no annotated edges for {selection.dataset_id}; nothing could be scored"
        )

    frame_fraction = selection.frames / truth.frames
    cropped_voxels = (
        (selection.z_stop - selection.z_start)
        * (selection.y_stop - selection.y_start)
        * (selection.x_stop - selection.x_start)
    )
    voxel_fraction = cropped_voxels / (array.shape[1] * array.shape[2] * array.shape[3])
    return CompetitionWindow(
        selection=selection,
        volume=volume,
        annotated=annotated,
        dataset_estimated_total_nodes=truth.estimated_total_nodes,
        window_estimated_total_nodes=truth.estimated_total_nodes * frame_fraction * voxel_fraction,
        dataset_frames=truth.frames,
        intensity_low=low,
        intensity_high=high,
        dropped_nodes=dropped_nodes,
        dropped_edges=dropped_edges,
        scale=OFFICIAL_VOXEL_SCALE,
    )


def _restrict(lineage: LineageGraph, selection: WindowSelection) -> tuple[LineageGraph, int, int]:
    """Keep the nodes inside the window, renumbered so the window starts at frame 0."""
    stop = selection.first_frame + selection.frames
    kept: dict[int, LineageNode] = {}
    for node in lineage.nodes:
        if not selection.first_frame <= node.frame < stop:
            continue
        if not selection.z_start <= node.voxel.z < selection.z_stop:
            continue
        if not selection.y_start <= node.voxel.y < selection.y_stop:
            continue
        if not selection.x_start <= node.voxel.x < selection.x_stop:
            continue
        kept[node.node_id] = LineageNode(
            dataset=node.dataset,
            node_id=node.node_id,
            frame=node.frame - selection.first_frame,
            voxel=VoxelCoordinateZYX(
                z=node.voxel.z - selection.z_start,
                y=node.voxel.y - selection.y_start,
                x=node.voxel.x - selection.x_start,
            ),
        )
    surviving = [edge for edge in lineage.edges if edge.source in kept and edge.target in kept]
    pairs = np.array([[edge.source, edge.target] for edge in surviving], dtype=np.int64).reshape(-1, 2)
    restricted = LineageGraph(
        dataset=lineage.dataset,
        nodes=tuple(kept.values()),
        edges=edges_with_kinds(lineage.dataset, pairs),
    )
    return restricted, len(lineage.nodes) - len(kept), len(lineage.edges) - len(surviving)
