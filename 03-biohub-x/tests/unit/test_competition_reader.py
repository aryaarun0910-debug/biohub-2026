"""The competition reader refuses what it does not understand, and windows correctly.

Built on a hand-made GEFF store rather than on competition data, so the reader's
contract is checked wherever the tests run and not only on a machine that holds
the corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from biohubx.contracts.lineage import EdgeKind
from biohubx.data.competition import (
    CompetitionLayoutError,
    WindowSelection,
    competition_root,
    load_ground_truth,
)

OFFICIAL_AXES: list[dict[str, Any]] = [
    {"name": "t", "type": "time", "scale": 1.0},
    {"name": "z", "type": "space", "scale": 1.625},
    {"name": "y", "type": "space", "scale": 0.40625},
    {"name": "x", "type": "space", "scale": 0.40625},
]


def write_geff(
    root: Path,
    dataset_id: str,
    *,
    frames: list[int],
    z: list[float],
    y: list[float],
    x: list[float],
    edges: list[tuple[int, int]],
    estimate: int | None = 1000,
    version: str = "1.1",
    axes: list[dict[str, Any]] | None = None,
) -> None:
    split = root / "train"
    split.mkdir(parents=True, exist_ok=True)
    store: Any = zarr.open(str(split / f"{dataset_id}.geff"), mode="w")
    extra = {} if estimate is None else {"estimated_number_of_nodes": estimate}
    store.attrs["geff"] = {
        "geff_version": version,
        "directed": True,
        "axes": OFFICIAL_AXES if axes is None else axes,
        "extra": extra,
    }
    store.create_array("nodes/ids", shape=(len(frames),), dtype="uint64")[:] = np.arange(len(frames))
    for name, values in (("t", frames), ("z", z), ("y", y), ("x", x)):
        array = store.create_array(f"nodes/props/{name}/values", shape=(len(frames),), dtype="int64")
        array[:] = np.asarray(values, dtype=np.int64)
    pairs = np.asarray(edges, dtype=np.uint64).reshape(-1, 2)
    store.create_array("edges/ids", shape=pairs.shape, dtype="uint64")[:] = pairs


def simple_track(root: Path, dataset_id: str = "6bba_fixture") -> None:
    """Four cells in four frames, one continuing track."""
    write_geff(
        root,
        dataset_id,
        frames=[0, 1, 2, 3],
        z=[10, 10, 10, 10],
        y=[20, 21, 22, 23],
        x=[30, 30, 30, 30],
        edges=[(0, 1), (1, 2), (2, 3)],
    )


def test_a_ground_truth_reads_into_a_validated_lineage(tmp_path: Path) -> None:
    simple_track(tmp_path)
    truth = load_ground_truth(tmp_path, "6bba_fixture")

    assert len(truth.lineage.nodes) == 4
    assert len(truth.lineage.edges) == 3
    assert truth.estimated_total_nodes == 1000
    assert truth.frames == 4
    assert truth.embryo == "6bba"
    assert all(edge.kind is EdgeKind.CONTINUATION for edge in truth.lineage.edges)


def test_a_fork_in_the_ground_truth_is_labelled_a_division(tmp_path: Path) -> None:
    write_geff(
        tmp_path,
        "6bba_fork",
        frames=[0, 1, 1],
        z=[10, 10, 10],
        y=[20, 21, 19],
        x=[30, 30, 30],
        edges=[(0, 1), (0, 2)],
    )
    truth = load_ground_truth(tmp_path, "6bba_fork")
    assert {edge.kind for edge in truth.lineage.edges} == {EdgeKind.DIVISION}


def test_an_unsupported_geff_version_is_refused(tmp_path: Path) -> None:
    simple_track(tmp_path, "6bba_future")
    store: Any = zarr.open(str(tmp_path / "train" / "6bba_future.geff"), mode="r+")
    meta = dict(store.attrs["geff"])
    meta["geff_version"] = "2.0"
    store.attrs["geff"] = meta

    with pytest.raises(CompetitionLayoutError, match="unsupported GEFF version"):
        load_ground_truth(tmp_path, "6bba_future")


def test_a_missing_node_estimate_is_refused_rather_than_substituted(tmp_path: Path) -> None:
    write_geff(
        tmp_path,
        "6bba_noestimate",
        frames=[0, 1],
        z=[1, 1],
        y=[2, 3],
        x=[4, 4],
        edges=[(0, 1)],
        estimate=None,
    )
    with pytest.raises(CompetitionLayoutError, match="denominator is unknown"):
        load_ground_truth(tmp_path, "6bba_noestimate")


def test_a_non_official_voxel_scale_is_refused(tmp_path: Path) -> None:
    write_geff(
        tmp_path,
        "6bba_scaled",
        frames=[0, 1],
        z=[1, 1],
        y=[2, 3],
        x=[4, 4],
        edges=[(0, 1)],
        axes=[
            {"name": "t", "type": "time", "scale": 1.0},
            {"name": "z", "type": "space", "scale": 2.0},
            {"name": "y", "type": "space", "scale": 0.5},
            {"name": "x", "type": "space", "scale": 0.5},
        ],
    )
    with pytest.raises(CompetitionLayoutError, match="official is"):
        load_ground_truth(tmp_path, "6bba_scaled")


def test_a_root_is_never_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOHUB_DATA_ROOT", raising=False)
    with pytest.raises(CompetitionLayoutError, match="BIOHUB_DATA_ROOT is unset"):
        competition_root(None)


def test_a_window_selection_records_its_own_bounds() -> None:
    selection = WindowSelection("6bba_fixture", 4, 8, 0, 32, 128, 256, 128, 256)
    recorded = selection.to_dict()
    assert recorded["first_frame"] == 4
    assert recorded["frames"] == 8
    assert recorded["crop_z"] == [0, 32]
    assert recorded["crop_y"] == [128, 256]
