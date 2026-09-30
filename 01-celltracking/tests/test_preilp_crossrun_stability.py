"""Contract tests for the cross-run pre-ILP stability check.

The point of this instrument is to FAIL when node numbering drifts between runs, and to stay
quiet-but-loud when only candidate edges differ. Both behaviours are asserted here, along with
the fail-closed cases, because an audit that can silently compare nothing looks exactly like an
audit that passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import preilp_crossrun_stability as pcs  # noqa: E402


def export(crop="c0", n=4, shift=0.0, edges=((0, 2), (1, 3)), ids=None):
    ids = list(range(n)) if ids is None else ids
    node_rows = {
        "dataset": [crop] * n,
        "row_type": ["node"] * n,
        "node_id": ids,
        "t": [i // 2 for i in range(n)],
        "z": [float(i) + shift for i in range(n)],
        "y": [float(i) * 2 + shift for i in range(n)],
        "x": [float(i) * 3 + shift for i in range(n)],
        "source_id": [-1] * n,
        "target_id": [-1] * n,
    }
    m = len(edges)
    edge_rows = {
        "dataset": [crop] * m,
        "row_type": ["edge"] * m,
        "node_id": [-1] * m,
        "t": [-1] * m, "z": [-1.0] * m, "y": [-1.0] * m, "x": [-1.0] * m,
        "source_id": [s for s, _ in edges],
        "target_id": [t for _, t in edges],
    }
    return pl.concat([pl.DataFrame(node_rows), pl.DataFrame(edge_rows)], how="vertical")


def test_identical_exports_pass_and_are_joinable():
    a = export()
    r = pcs.compare_crop("c0", a, a)
    assert r["passed"] is True
    assert r["candidate_edges_identical"] is True


def test_shifted_coordinates_fail_the_hard_gate():
    """Same ids, different coordinates: the numbering no longer means the same cells."""
    r = pcs.compare_crop("c0", export(), export(shift=0.001))
    assert r["node_ids_match"] is True
    assert r["node_coords_match_exactly"] is False
    assert r["passed"] is False


def test_reordered_ids_fail_the_hard_gate():
    r = pcs.compare_crop("c0", export(), export(ids=[3, 2, 1, 0]))
    assert r["passed"] is False


def test_edge_difference_alone_does_not_fail_the_gate():
    """FACT-0363: a gate that assumes edge determinism discards real results as failures."""
    r = pcs.compare_crop("c0", export(edges=((0, 2), (1, 3))), export(edges=((0, 2),)))
    assert r["node_coords_match_exactly"] is True
    assert r["passed"] is True, "node identity is the hard gate; edges are reported"
    assert r["candidate_edges_identical"] is False
    assert r["edges_only_in_old"] == 1


def test_empty_crop_raises_rather_than_comparing_nothing():
    empty = export().filter(pl.col("row_type") == "edge")
    with pytest.raises(SystemExit, match="refusing to compare nothing"):
        pcs.compare_crop("c0", empty, export())
