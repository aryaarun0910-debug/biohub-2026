"""Software contracts for the LEVER-0035 retention instruments.

Contracts only. Which solver weight to ship is an experiment result, not a unit test.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scripts.win_bet import ilp_replay, ilp_sweep
from scripts.win_bet.detpeak_curve import match_one_to_one, match_one_to_one_pairs
from scripts.win_bet.lost_cell_atlas import summarise as atlas_summarise


def _rows(nodes, edges, crop="c"):
    node_rows = pl.DataFrame({
        "dataset": [crop] * len(nodes), "row_type": ["node"] * len(nodes),
        "node_id": [n for n in nodes], "t": [0] * len(nodes),
        "z": [0.0] * len(nodes), "y": [0.0] * len(nodes), "x": [0.0] * len(nodes),
        "source_id": [-1] * len(nodes), "target_id": [-1] * len(nodes),
    })
    if not edges:
        return node_rows
    edge_rows = pl.DataFrame({
        "dataset": [crop] * len(edges), "row_type": ["edge"] * len(edges),
        "node_id": [-1] * len(edges), "t": [-1] * len(edges),
        "z": [-1.0] * len(edges), "y": [-1.0] * len(edges), "x": [-1.0] * len(edges),
        "source_id": [s for s, _ in edges], "target_id": [t for _, t in edges],
    })
    return pl.concat([node_rows, edge_rows])


def test_deployed_weights_match_the_notebook_env_block():
    """These four constants define the baseline every sweep arm is paired against.

    Verified in the P28 notebook env block and the 2026-08-19 pre-ILP kernel log. If the
    deployed configuration ever changes, every recorded delta silently changes meaning.
    """
    assert ilp_replay.DEPLOYED_WEIGHTS == {
        "edge": -1.0, "appearance": 0.0, "disappearance": 1.5, "division": 1.0,
    }
    assert ilp_sweep.DEPLOYED_MIN_TRACK_LEN == 6


def test_short_track_filter_removes_whole_components_not_loose_nodes():
    """BIOHUB_OUTPUT_MIN_TRACK_LEN cuts by COMPONENT length, so a 3-chain dies entire."""
    rows = _rows([1, 2, 3, 10, 11, 12, 13], [(1, 2), (2, 3), (10, 11), (11, 12), (12, 13)])
    kept = ilp_sweep.apply_short_track_filter(rows, min_len=4)
    kept_nodes = set(kept.filter(pl.col("row_type") == "node")["node_id"].to_list())
    assert kept_nodes == {10, 11, 12, 13}
    # every surviving edge must have both endpoints surviving
    edges = kept.filter(pl.col("row_type") == "edge")
    assert set(edges["source_id"]) <= kept_nodes
    assert set(edges["target_id"]) <= kept_nodes


def test_short_track_filter_handles_forks_as_one_component():
    """A division makes a Y shape; it is ONE weakly-connected component, not two branches."""
    rows = _rows([1, 2, 3, 4], [(1, 2), (2, 3), (2, 4)])
    kept = ilp_sweep.apply_short_track_filter(rows, min_len=4)
    assert kept.filter(pl.col("row_type") == "node").height == 4


def test_short_track_filter_is_a_noop_below_two():
    rows = _rows([1, 2], [(1, 2)])
    assert ilp_sweep.apply_short_track_filter(rows, min_len=1).equals(rows)


def test_paired_bootstrap_detects_a_consistent_shift_and_not_noise():
    base = [{"dataset": f"c{i}", "v": 0.5} for i in range(40)]
    better = [{"dataset": f"c{i}", "v": 0.6} for i in range(40)]
    got = ilp_sweep.paired_bootstrap(base, better, "v")
    assert got["mean"] == pytest.approx(0.1)
    assert got["excludes_zero"] is True

    same = [{"dataset": f"c{i}", "v": 0.5} for i in range(40)]
    got = ilp_sweep.paired_bootstrap(base, same, "v")
    assert got["mean"] == pytest.approx(0.0)


def test_paired_bootstrap_pairs_by_dataset_not_by_position():
    """Arms can emit crops in different orders; pairing by index would silently mismatch."""
    base = [{"dataset": "a", "v": 1.0}, {"dataset": "b", "v": 2.0}]
    arm = [{"dataset": "b", "v": 2.0}, {"dataset": "a", "v": 1.0}]
    assert ilp_sweep.paired_bootstrap(base, arm, "v")["mean"] == pytest.approx(0.0)


def test_matcher_pairs_and_count_agree():
    """FACT-0354/0355/0357 share one matcher; a divergence would desync recall and the atlas."""
    gt = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    pred = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    pairs = match_one_to_one_pairs(pred, gt)
    assert len(pairs) == match_one_to_one(pred, gt) == 2
    assert {g for g, _, _ in pairs} == {0, 1}
    assert all(d <= 7.0 for _, _, d in pairs)


def test_atlas_deciding_split_counts_zero_candidate_cells():
    """The share with no candidate edge is what decides PKT-0025 falsifier (b)."""
    table = pl.DataFrame({
        "lost": [True, True, True, False],
        "cand_support_known": [True, True, True, True],
        "cand_out": [0, 1, 2, 3],
        "cand_in": [0, 0, 1, 2],
        "cand_out_best": [0.0, 0.99, 0.7, 0.99],
        "cand_in_best": [0.0, 0.0, 0.6, 0.98],
        "detector_prob": [0.99, 0.99, 0.99, 0.99],
        "residual_um": [1.0, 1.0, 1.0, 1.0],
        "at_temporal_boundary": [False, False, False, False],
    })
    got = atlas_summarise(table)
    assert got["n_lost"] == 3
    assert got["deciding_split"]["lost_with_no_candidate_edge"] == 1
    assert got["deciding_split"]["lost_with_candidate_edge"] == 2
    assert got["deciding_split"]["share_unrescuable_by_solver"] == pytest.approx(1 / 3)
