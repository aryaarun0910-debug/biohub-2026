"""The ignore contract, and the peak path that turns a heatmap into proposals.

The load-bearing claim is that an unlabelled cell is never supervised as
background. These check it on constructed volumes where the answer is known, and
on the real preflight window where the answer is what justifies the design.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the model-cpu dependency group is not installed")

from biohubx.contracts.coordinates import VoxelCoordinateZYX  # noqa: E402
from biohubx.contracts.lineage import (  # noqa: E402
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.proposals.peaks import HeatmapProposalError, instances_from_heatmap  # noqa: E402
from biohubx.training.targets import (  # noqa: E402
    IGNORE,
    NEGATIVE,
    POSITIVE,
    TargetConstructionError,
    build_detection_target,
    masked_detection_loss,
)

DATASET = DatasetIdentity(value="target-fixture")
DOWNSAMPLE = (1, 4, 4)


def graph(coords: list[tuple[int, float, float, float]]) -> LineageGraph:
    nodes = tuple(
        LineageNode(
            dataset=DATASET,
            node_id=index,
            frame=frame,
            voxel=VoxelCoordinateZYX(z=z, y=y, x=x),
        )
        for index, (frame, z, y, x) in enumerate(coords)
    )
    edges = tuple(
        LineageEdge(
            source_dataset=DATASET,
            source=index,
            target_dataset=DATASET,
            target=index + 1,
            kind=EdgeKind.CONTINUATION,
        )
        for index in range(len(coords) - 1)
        if coords[index + 1][0] == coords[index][0] + 1
    )
    return LineageGraph(dataset=DATASET, nodes=nodes, edges=edges)


def volume_with_bright_spots(spots: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Dim textured background with a few bright spots on it.

    The background is varied on purpose. A uniform volume has no meaningful
    intensity quantile, so every voxel would land in the ignore band and the
    test would be measuring nothing.
    """
    rng = np.random.default_rng(0)
    volume = rng.uniform(0.0, 0.05, size=(2, 4, 16, 16)).astype(np.float32)
    for frame, z, y, x in spots:
        volume[frame, z, y, x] = 1.0
    return volume


def test_a_bright_unlabelled_voxel_is_ignored_never_background() -> None:
    """The whole point. Two bright spots, one annotated; the other must not be a negative."""
    volume = volume_with_bright_spots([(0, 1, 0, 0), (0, 1, 0, 4)])
    annotated = graph([(0, 1, 0, 0), (1, 1, 0, 0)])

    target = build_detection_target(volume, annotated, downsample=DOWNSAMPLE, ignore_quantile=0.90)
    labels = target.labels

    assert labels[0, 1, 0, 0] == POSITIVE, "the annotated cell is a positive"
    assert labels[0, 1, 0, 1] == IGNORE, "the unannotated bright cell is ignored, not background"
    assert labels[0, 0, 3, 3] == NEGATIVE, "empty space is still supervised as background"


def test_ignored_voxels_contribute_no_loss() -> None:
    volume = volume_with_bright_spots([(0, 1, 0, 0), (0, 1, 0, 4)])
    annotated = graph([(0, 1, 0, 0), (1, 1, 0, 0)])
    target = build_detection_target(volume, annotated, downsample=DOWNSAMPLE, ignore_quantile=0.90)

    quiet = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    baseline = masked_detection_loss(quiet, target)

    shouting = quiet.clone()
    shouting[target.labels == IGNORE] = 50.0
    assert torch.isclose(masked_detection_loss(shouting, target), baseline), (
        "an arbitrarily confident prediction on an ignored voxel changed the loss"
    )

    wrong = quiet.clone()
    wrong[target.labels == POSITIVE] = -50.0
    assert masked_detection_loss(wrong, target) > baseline


def test_positives_and_negatives_are_balanced_against_each_other() -> None:
    """26 positives against 350,000 negatives must not be drowned."""
    volume = volume_with_bright_spots([(0, 1, 0, 0)])
    annotated = graph([(0, 1, 0, 0), (1, 1, 0, 0)])
    target = build_detection_target(volume, annotated, downsample=DOWNSAMPLE, ignore_quantile=0.90)

    missing_one_positive = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    missing_one_positive[target.labels == POSITIVE] = -20.0
    positive_cost = float(masked_detection_loss(missing_one_positive, target))

    one_false_positive = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    first_negative = (target.labels == NEGATIVE).nonzero()[0].tolist()
    one_false_positive[tuple(first_negative)] = 20.0
    negative_cost = float(masked_detection_loss(one_false_positive, target))

    assert positive_cost > negative_cost, (
        "missing an annotated cell must cost more than one spurious voxel, or the "
        "detector learns to predict nothing"
    )


def test_a_target_with_no_placeable_node_is_refused() -> None:
    volume = volume_with_bright_spots([(0, 1, 0, 0)])
    outside = graph([(0, 99.0, 99.0, 99.0), (1, 99.0, 99.0, 99.0)])
    with pytest.raises(TargetConstructionError, match="no annotated node landed"):
        build_detection_target(volume, outside, downsample=DOWNSAMPLE)


def test_a_loss_with_only_one_class_is_refused() -> None:
    volume = np.ones((2, 4, 16, 16), dtype=np.float32)
    annotated = graph([(0, 1, 0, 0), (1, 1, 0, 0)])
    # Everything is equally bright, so the quantile puts every voxel in ignore.
    target = build_detection_target(volume, annotated, downsample=DOWNSAMPLE, ignore_quantile=0.001)
    with pytest.raises(TargetConstructionError, match="needs both classes"):
        masked_detection_loss(torch.zeros((2, 4, 4, 4), dtype=torch.float32), target)


def test_a_mismatched_grid_is_refused() -> None:
    volume = volume_with_bright_spots([(0, 1, 0, 0)])
    annotated = graph([(0, 1, 0, 0), (1, 1, 0, 0)])
    target = build_detection_target(volume, annotated, downsample=DOWNSAMPLE)
    with pytest.raises(TargetConstructionError, match="different grids"):
        masked_detection_loss(torch.zeros((2, 4, 8, 8), dtype=torch.float32), target)


# --- heatmap to proposals ---------------------------------------------------


def test_peaks_are_returned_in_full_resolution_coordinates() -> None:
    """A proposal read off the strided grid must be placed on the original one."""
    heatmap = np.zeros((1, 4, 8, 8), dtype=np.float32)
    heatmap[0, 2, 3, 5] = 0.99

    instances = instances_from_heatmap(heatmap, dataset=DATASET, downsample=DOWNSAMPLE, threshold=0.5)

    assert len(instances.instances) == 1
    voxel = instances.instances[0].voxel
    assert (voxel.z, voxel.y, voxel.x) == (2.0, 12.0, 20.0)


def test_adjacent_peaks_are_suppressed_into_one() -> None:
    heatmap = np.zeros((1, 4, 8, 8), dtype=np.float32)
    heatmap[0, 2, 3, 5] = 0.99
    heatmap[0, 2, 3, 6] = 0.98  # 1.625 um away on the strided grid, inside the 4 um radius

    instances = instances_from_heatmap(heatmap, dataset=DATASET, downsample=DOWNSAMPLE, threshold=0.5)
    assert len(instances.instances) == 1
    assert instances.instances[0].confidence == pytest.approx(0.99, abs=1e-6)


def test_an_empty_heatmap_is_a_failure_not_an_empty_result() -> None:
    with pytest.raises(HeatmapProposalError, match="proposed nothing"):
        instances_from_heatmap(
            np.zeros((1, 4, 8, 8), dtype=np.float32),
            dataset=DATASET,
            downsample=DOWNSAMPLE,
            threshold=0.5,
        )


def test_a_threshold_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(HeatmapProposalError, match=r"threshold must lie in \(0, 1\)"):
        instances_from_heatmap(
            np.ones((1, 4, 8, 8), dtype=np.float32),
            dataset=DATASET,
            downsample=DOWNSAMPLE,
            threshold=1.0,
        )


# --- the design claim, on the real window -----------------------------------


def test_the_ignore_band_contains_every_annotated_cell_in_the_preflight_window() -> None:
    """The reason to believe the band protects unannotated cells.

    Instrument for F-0019. If the intensity band that hides unannotated cells did
    not also contain the annotated ones, there would be no argument that the two
    populations look alike, and the ignore mask would be guesswork.
    """
    from biohubx.artifacts import ARTIFACT_REGISTRY_PATH, load_artifact_registry
    from biohubx.data.competition import WindowSelection, load_window

    repo = pathlib.Path(__file__).resolve().parents[2]
    registry = load_artifact_registry(repo / ARTIFACT_REGISTRY_PATH)
    record = next((a for a in registry.artifacts if a.id == "competition.train.6bba_2540cd90.zarr"), None)
    if record is None or record.external_path is None:
        pytest.skip("the preflight dataset is not registered on this machine")
    root = pathlib.Path(record.external_path).parent.parent
    if not root.is_dir():
        pytest.skip("the competition corpus is not on this machine")

    window = load_window(
        root, WindowSelection("6bba_2540cd90", 0, 12, 0, 32, 128, 256, 128, 256), split="train"
    )
    target = build_detection_target(
        window.volume, window.annotated, downsample=DOWNSAMPLE, ignore_quantile=0.90
    )

    assert target.unplaceable_nodes == 0
    dz, dy, dx = DOWNSAMPLE
    grid = window.volume[:, ::dz, ::dy, ::dx]
    above = sum(
        1
        for node in window.annotated.nodes
        if grid[node.frame, int(node.voxel.z) // dz, int(node.voxel.y) // dy, int(node.voxel.x) // dx]
        >= target.ignore_threshold
    )
    assert above == len(window.annotated.nodes), (
        f"only {above} of {len(window.annotated.nodes)} annotated cells sit in the ignore band; "
        "the band cannot be argued to hide unannotated cells"
    )
    assert target.ignored / (target.positives + target.negatives + target.ignored) < 0.15
