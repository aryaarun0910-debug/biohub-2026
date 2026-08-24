"""Identity-sidecar contracts for the runnable Zh001r association lane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle_edits"))

import h1r_edge_data as E  # noqa: E402


def _write_pack(root: Path, *, bad: str | None = None) -> tuple[Path, Path]:
    """Two frames: 10 continues; 20 divides into 21 and 22; 30 is a birth."""
    node_rows = {
        "f0": np.array([[7, 1, 2, 3], [7, 4, 5, 6]], dtype=np.float32),
        "f1": np.array([
            [8, 1, 2, 4], [8, 4, 4, 5], [8, 5, 5, 6], [8, 9, 9, 9],
        ], dtype=np.float32),
    }
    identities = {
        "tid_0_0": np.array([10, 20], dtype=np.int64),
        "pid_0_0": np.array([-1, -1], dtype=np.int64),
        "tid_0_1": np.array([10, 21, 22, 30], dtype=np.int64),
        "pid_0_1": np.array([-1, 20, 20, -1], dtype=np.int64),
    }
    if bad == "length":
        identities["tid_0_1"] = identities["tid_0_1"][:-1]
    elif bad == "duplicate":
        identities["tid_0_1"] = np.array([10, 21, 21, 30], dtype=np.int64)
    elif bad == "missing":
        del identities["pid_0_1"]
    elif bad == "unregistered":
        identities["tid_0_1"][2] = -1
    elif bad == "time":
        node_rows["f1"][:, 0] = 10
    nodes = root / "zh001r_nodes.npz"
    identity = root / "zh001r_identity.npz"
    np.savez(nodes, **node_rows)
    np.savez(identity, **identities)
    return nodes, identity


def test_synthetic_identity_join_emits_continuation_and_division_targets(tmp_path):
    nodes, identity = _write_pack(tmp_path)
    data = E.Zh001rEdgeData(nodes, identity, n_frames=2)
    pair = data.pair(0, 0)

    expected = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
    ], dtype=torch.float32)
    assert torch.equal(pair.target, expected)
    assert pair.target.dtype == torch.float32
    assert pair.source_coords.shape == (2, 3)
    assert pair.target_coords.shape == (4, 3)
    assert pair.continuation_links == 1
    assert pair.division_daughter_links == 2
    assert data.counts == {
        "continuation_links": 1,
        "division_daughter_links": 2,
        "association_links": 3,
    }


def test_discovery_allows_separate_kaggle_mounts_and_refuses_ambiguity(tmp_path):
    nodes_root, identity_root = tmp_path / "nodes", tmp_path / "identity"
    nodes_root.mkdir(); identity_root.mkdir()
    nodes, identity = _write_pack(nodes_root)
    identity.replace(identity_root / identity.name)
    assets = E.discover_edge_assets([nodes_root, identity_root])
    assert assets.nodes == nodes.resolve()
    assert assets.identity == (identity_root / identity.name).resolve()
    loaded = E.load_edge_data([nodes_root, identity_root], n_frames=2)
    assert loaded.counts["association_links"] == 3

    duplicate_root = tmp_path / "old-version"
    duplicate_root.mkdir()
    (duplicate_root / identity.name).write_bytes(assets.identity.read_bytes())
    with pytest.raises(E.EdgeDataError, match="ambiguous"):
        E.discover_edge_assets([nodes_root, identity_root, duplicate_root])


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("length", "row mismatch"),
        ("duplicate", "duplicate track_id"),
        ("missing", "does not exactly cover"),
        ("unregistered", "no registered track identity"),
        ("time", "not adjacent"),
    ],
)
def test_corrupt_or_misaligned_sidecars_fail_closed(tmp_path, bad, message):
    nodes, identity = _write_pack(tmp_path, bad=bad)
    with pytest.raises(E.EdgeDataError, match=message):
        E.Zh001rEdgeData(nodes, identity, n_frames=2)


def test_ambiguous_continuation_and_division_parent_is_rejected():
    with pytest.raises(E.EdgeDataError, match="both a continuation parent"):
        E.build_transition_target(
            np.array([10, 20]),
            np.array([10]),
            np.array([20]),
        )


def test_full_pack_published_totals_are_a_load_time_gate(tmp_path, monkeypatch):
    # Exercise the automatic full-corpus gate cheaply by redefining "full" for this
    # synthetic two-frame corpus; the production constants remain pinned above.
    nodes, identity = _write_pack(tmp_path)
    monkeypatch.setattr(E, "EXPECTED_FULL_CROPS", 1)
    monkeypatch.setattr(E, "N_FRAMES_PER_CROP", 2)
    monkeypatch.setattr(E, "EXPECTED_CONTINUATION_LINKS", 999)
    monkeypatch.setattr(E, "EXPECTED_DIVISION_DAUGHTER_LINKS", 999)
    with pytest.raises(E.EdgeDataError, match="registered authority"):
        E.Zh001rEdgeData(nodes, identity, n_frames=2)
