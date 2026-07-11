from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import numpy as np

from scripts.trackastra_zero_shot.adapter import (
    FrozenDetections,
    export_trackastra_result,
    export_frozen_edge_selection,
    load_frozen_detections,
    rasterize_ellipsoid_masks,
    write_mask_mapping,
)


def test_csv_load_and_anisotropic_ellipsoid(tmp_path: Path):
    points = tmp_path / "points.csv"
    points.write_text("node_id,t,z,y,x\n7,0,2,4,4\n", encoding="utf-8")
    detections = load_frozen_detections(points)

    masks, mapping = rasterize_ellipsoid_masks(
        detections,
        (1, 5, 9, 9),
        radius_um=2.0,
        voxel_size_um=(2.0, 1.0, 1.0),
    )

    assert mapping[0].source_node_id == 7
    assert masks[0, 2, 4, 4] == 1
    assert masks[0, 1, 4, 4] == 1  # 2 um axial step lies on the ellipsoid
    assert masks[0, 2, 4, 6] == 1  # 2 um lateral step lies on the ellipsoid
    assert masks[0, 0, 4, 4] == 0
    assert masks[0, 2, 4, 7] == 0


def test_overlap_is_nearest_and_each_detection_survives():
    detections = FrozenDetections(
        node_ids=np.array([10, 20]),
        time=np.array([0, 0]),
        zyx=np.array([[1.0, 3.0, 2.0], [1.0, 3.0, 6.0]]),
    )
    masks, mapping = rasterize_ellipsoid_masks(
        detections,
        (1, 3, 7, 9),
        radius_um=3.0,
        voxel_size_um=(3.0, 1.0, 1.0),
    )
    assert [m.source_node_id for m in mapping] == [10, 20]
    assert masks[0, 1, 3, 2] == 1
    assert masks[0, 1, 3, 6] == 2
    assert set(np.unique(masks)) == {0, 1, 2}


def test_memmap_and_duplicate_rounded_voxel_guard(tmp_path: Path):
    good = FrozenDetections(np.array([1]), np.array([0]), np.array([[1.2, 1.2, 1.2]]))
    output = tmp_path / "mask.npy"
    masks, _ = rasterize_ellipsoid_masks(good, (1, 4, 4, 4), output_npy=output)
    assert output.exists()
    assert isinstance(masks, np.memmap)
    assert np.load(output, mmap_mode="r").shape == (1, 4, 4, 4)

    bad = FrozenDetections(
        np.array([1, 2]), np.array([0, 0]), np.array([[1.1, 1.1, 1.1], [1.2, 1.2, 1.2]])
    )
    try:
        rasterize_ellipsoid_masks(bad, (1, 4, 4, 4))
    except ValueError as exc:
        assert "same mask voxel" in str(exc)
    else:
        raise AssertionError("ambiguous point-to-instance mapping was not rejected")


def test_mask_mapping_writer_uses_dataclass_time_field(tmp_path: Path):
    detections = FrozenDetections(np.array([7]), np.array([3]), np.array([[1.0, 1.0, 1.0]]))
    _, mapping = rasterize_ellipsoid_masks(detections, (4, 3, 3, 3))
    output = write_mask_mapping(mapping, tmp_path / "mapping.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row == {"detection_index": "0", "source_node_id": "7", "time": "3", "label": "1"}


def test_export_restores_points_and_writes_candidate_scores(tmp_path: Path):
    detections = FrozenDetections(
        node_ids=np.array([101, 202]),
        time=np.array([0, 1]),
        zyx=np.array([[1.25, 2.5, 3.75], [2.0, 3.0, 4.0]]),
    )
    _, mapping = rasterize_ellipsoid_masks(detections, (2, 5, 7, 8), radius_um=1.0)
    predictions = {
        "nodes": [
            {"id": 0, "time": 0, "label": 1, "coords": (1, 2, 4)},
            {"id": 1, "time": 1, "label": 1, "coords": (2, 3, 4)},
        ],
        "weights": [((0, 1), 0.875)],
    }
    selected = nx.DiGraph()
    selected.add_nodes_from([0, 1])
    selected.add_edge(0, 1)
    geff = tmp_path / "prediction.geff"
    scores = tmp_path / "scores.csv"

    export_trackastra_result(
        detections,
        mapping,
        predictions,
        selected,
        output_geff=geff,
        output_edge_scores=scores,
    )

    loaded = load_frozen_detections(geff)
    np.testing.assert_allclose(loaded.zyx, detections.zyx)
    with scores.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["source_node_id"] == "101"
    assert row["target_node_id"] == "202"
    assert float(row["score"]) == 0.875
    assert row["selected"] == "1"


def test_export_cached_selection_preserves_frozen_points(tmp_path: Path):
    detections = FrozenDetections(
        node_ids=np.array([101, 202]),
        time=np.array([0, 1]),
        zyx=np.array([[1.25, 2.5, 3.75], [2.0, 3.0, 4.0]]),
    )
    geff = export_frozen_edge_selection(
        detections,
        [(101, 202)],
        {(101, 202): 0.875},
        tmp_path / "cached.geff",
    )
    loaded = load_frozen_detections(geff)
    np.testing.assert_allclose(loaded.zyx, detections.zyx)


def test_export_cached_selection_can_prune_isolated_nodes(tmp_path: Path):
    detections = FrozenDetections(
        node_ids=np.array([101, 202, 303]),
        time=np.array([0, 1, 1]),
        zyx=np.array([[1.25, 2.5, 3.75], [2.0, 3.0, 4.0], [9.0, 9.0, 9.0]]),
    )
    geff = export_frozen_edge_selection(
        detections,
        [(101, 202)],
        {(101, 202): 0.875},
        tmp_path / "pruned.geff",
        prune_isolated=True,
    )
    loaded = load_frozen_detections(geff)
    assert len(loaded.node_ids) == 2
    np.testing.assert_allclose(loaded.zyx, detections.zyx[:2])
