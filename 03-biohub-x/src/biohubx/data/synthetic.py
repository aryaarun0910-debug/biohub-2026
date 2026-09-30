"""A deterministic synthetic movie with a known lineage.

This exists so the whole chain can be exercised end to end without competition
data, a GPU, or any external artifact. It is a fixture, not a simulator: it
differs from real microscopy in cost and in realism, but not in kind. It is
anisotropic, it is scored in micrometres, its cells move and one of them
divides, and its ground truth is a legal lineage graph.

What it deliberately does NOT claim: that a system which scores well here will
score well on real embryos. Nothing measured on this fixture is evidence about
the competition. Its job is to prove that the software owns its graph.

Consumers: :mod:`biohubx.proposals.classical`, ``biohubx infer synthetic``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

SYNTHETIC_DATASET = DatasetIdentity(value="synthetic-slice-v1")

# Small enough to run in under a second on a laptop CPU, large enough that the
# anisotropy is real: z voxels are four times the y and x voxels, so a blob that
# looks round in micrometres is squashed along z in voxel space.
VOLUME_SHAPE = (4, 10, 40, 40)
"""(T, Z, Y, X)."""

BLOB_RADIUS_UM = 2.5
BACKGROUND = 0.02


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    """The fixture and everything known to be true about it."""

    volume: np.ndarray
    """Shape (T, Z, Y, X), float32 in [0, 1]. Deterministic for a given seed."""

    lineage: LineageGraph
    """The complete true lineage, in voxel coordinates."""

    annotated: LineageGraph
    """The subset a sparse annotator would have labelled. May equal `lineage`."""

    estimated_total_nodes: float
    """What the dataset's own coarse estimate of all cells would say."""

    scale: VoxelScaleZYX

    @property
    def frames(self) -> int:
        return int(self.volume.shape[0])


def _cell_tracks() -> list[tuple[int, list[tuple[float, float, float]], int | None, float]]:
    """Hand-written tracks: (track id, voxel positions, parent track, brightness).

    Written out rather than generated so that the true lineage is inspectable
    and cannot drift when the generator is refactored. Positions are chosen so
    that no two cells in a frame come within the 7 micrometre matching radius,
    which keeps the fixture's own matching unambiguous.

    Brightness differs per track, and daughters inherit their parent's, so that
    intensity carries real information rather than being a constant channel that
    looks like evidence in every report while affecting nothing.
    """
    # A cell drifting steadily in x.
    drifting = [(5.0, 10.0, 6.0), (5.0, 10.0, 10.0), (5.0, 10.0, 14.0), (5.0, 10.0, 18.0)]
    # A cell drifting in y, well separated from the first.
    crossing = [(5.0, 28.0, 30.0), (5.0, 25.0, 30.0), (5.0, 22.0, 30.0), (5.0, 19.0, 30.0)]
    # A parent that exists for two frames and then divides into two daughters.
    parent = [(3.0, 8.0, 30.0), (3.0, 8.0, 33.0)]
    daughter_a = [(3.0, 4.0, 35.0), (3.0, 3.0, 37.0)]
    daughter_b = [(3.0, 13.0, 35.0), (3.0, 14.0, 37.0)]
    return [
        (0, drifting, None, 1.00),
        (1, crossing, None, 0.72),
        (2, parent, None, 0.86),
        (3, daughter_a, 2, 0.86),
        (4, daughter_b, 2, 0.86),
    ]


def _render(
    volume: np.ndarray,
    frame: int,
    centre: tuple[float, float, float],
    scale: VoxelScaleZYX,
    amplitude: float,
) -> None:
    """Paint one Gaussian blob that is round in micrometres, not in voxels."""
    z_size, y_size, x_size = volume.shape[1:]
    zz, yy, xx = np.meshgrid(
        np.arange(z_size, dtype=np.float64),
        np.arange(y_size, dtype=np.float64),
        np.arange(x_size, dtype=np.float64),
        indexing="ij",
    )
    dz = (zz - centre[0]) * scale.z_um
    dy = (yy - centre[1]) * scale.y_um
    dx = (xx - centre[2]) * scale.x_um
    squared = dz * dz + dy * dy + dx * dx
    volume[frame] += amplitude * np.exp(-squared / (2.0 * (BLOB_RADIUS_UM / 2.0) ** 2))


def build_synthetic_truth(
    *,
    seed: int = 0,
    noise: float = 0.01,
    annotated_fraction: float = 1.0,
    estimated_total_nodes: float | None = None,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> SyntheticTruth:
    """Build the fixture deterministically.

    ``annotated_fraction`` below 1.0 drops nodes from the annotated graph to
    imitate sparse labelling, keeping every node whose removal would break the
    legality of the annotated graph. ``estimated_total_nodes`` defaults to the
    true node count, which is what a dataset's coarse estimate is estimating.
    """
    if not 0.0 < annotated_fraction <= 1.0:
        raise ValueError(f"annotated_fraction must be in (0, 1], got {annotated_fraction}")

    volume = np.full(VOLUME_SHAPE, BACKGROUND, dtype=np.float64)
    tracks = _cell_tracks()

    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    node_ids: dict[tuple[int, int], int] = {}
    next_id = 0

    for track_id, positions, parent_track, amplitude in tracks:
        # The parent occupies frames 0 and 1; its daughters start at frame 2.
        start_frame = 2 if parent_track is not None else 0
        for offset, centre in enumerate(positions):
            frame = start_frame + offset
            _render(volume, frame, centre, scale, amplitude)
            node_ids[(track_id, frame)] = next_id
            nodes.append(
                LineageNode(
                    dataset=SYNTHETIC_DATASET,
                    node_id=next_id,
                    frame=frame,
                    voxel=VoxelCoordinateZYX(z=centre[0], y=centre[1], x=centre[2]),
                )
            )
            next_id += 1

    # Continuations within each track.
    for track_id, positions, parent_track, _ in tracks:
        start_frame = 2 if parent_track is not None else 0
        for offset in range(len(positions) - 1):
            frame = start_frame + offset
            edges.append(
                LineageEdge(
                    source_dataset=SYNTHETIC_DATASET,
                    source=node_ids[(track_id, frame)],
                    target_dataset=SYNTHETIC_DATASET,
                    target=node_ids[(track_id, frame + 1)],
                    kind=EdgeKind.CONTINUATION,
                )
            )
    # The division: the parent's last frame links to both daughters' first.
    for daughter in (3, 4):
        edges.append(
            LineageEdge(
                source_dataset=SYNTHETIC_DATASET,
                source=node_ids[(2, 1)],
                target_dataset=SYNTHETIC_DATASET,
                target=node_ids[(daughter, 2)],
                kind=EdgeKind.DIVISION,
            )
        )

    rng = np.random.default_rng(seed)
    volume = volume + rng.normal(loc=0.0, scale=noise, size=volume.shape)
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)

    lineage = LineageGraph(dataset=SYNTHETIC_DATASET, nodes=tuple(nodes), edges=tuple(edges))
    annotated = (
        lineage if annotated_fraction == 1.0 else _thin_annotations(lineage, annotated_fraction, seed=seed)
    )
    return SyntheticTruth(
        volume=volume,
        lineage=lineage,
        annotated=annotated,
        estimated_total_nodes=(
            float(len(lineage.nodes)) if estimated_total_nodes is None else estimated_total_nodes
        ),
        scale=scale,
    )


def _thin_annotations(lineage: LineageGraph, fraction: float, *, seed: int) -> LineageGraph:
    """Drop whole tracks to imitate an annotator who labelled only some cells.

    Whole connected components are dropped rather than individual nodes, because
    a real sparse annotation is a cell someone followed, not a random scatter of
    detections, and dropping a middle node would leave an illegal graph.
    """
    parents: dict[int, int] = {}

    def find(node: int) -> int:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        parents[find(left)] = find(right)

    for node in lineage.nodes:
        find(node.node_id)
    for edge in lineage.edges:
        union(edge.source, edge.target)

    components: dict[int, list[int]] = {}
    for node in lineage.nodes:
        components.setdefault(find(node.node_id), []).append(node.node_id)

    ordered = [components[key] for key in sorted(components)]
    target_nodes = max(1, math.floor(len(lineage.nodes) * fraction))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))

    kept: set[int] = set()
    for index in order:
        if len(kept) >= target_nodes:
            break
        kept.update(ordered[index])

    nodes = tuple(node for node in lineage.nodes if node.node_id in kept)
    edges = tuple(edge for edge in lineage.edges if edge.source in kept and edge.target in kept)
    if not nodes:
        raise ValueError("thinning removed every annotation; lower the fraction less aggressively")
    return LineageGraph(dataset=lineage.dataset, nodes=nodes, edges=edges)
