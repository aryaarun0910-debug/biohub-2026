from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biotrack import wrapper as W


def _fixture():
    # The main pair is 11.74 um apart across one missing frame: outside the
    # 5.8*2 base gate but inside the published density-adaptive maximum 11.85.
    nodes = {
        1: {"node_id": 1, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        2: {"node_id": 2, "t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        3: {"node_id": 3, "t": 3, "z": 0.0, "y": 0.0, "x": 28.9},
        4: {"node_id": 4, "t": 4, "z": 0.0, "y": 0.0, "x": 28.9},
        5: {"node_id": 5, "t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
        6: {"node_id": 6, "t": 1, "z": 0.0, "y": 200.0, "x": 0.0},
        7: {"node_id": 7, "t": 3, "z": 0.0, "y": 100.0, "x": 28.9},
        8: {"node_id": 8, "t": 3, "z": 0.0, "y": 200.0, "x": 28.9},
    }
    return nodes, [{"source_id": 1, "target_id": 2}, {"source_id": 3, "target_id": 4}]


def _stats():
    return {
        "gap_close_effective_max_gap": 0, "gap_candidates": 0,
        "gap_pairs_selected": 0, "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0, "gap_skipped_node_cap": 0,
        "gap_added_nodes": 0, "gap_added_edges": 0,
        "gap_density_nodes_scored": 0, "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "prefix_density_nodes_blended": 0, "gap_refined_synthetic": 0,
        "gap_refine_failed": 0, "gap_refine_rejected_shift": 0,
    }


def test_density_gate_is_default_off_and_expands_only_when_enabled(monkeypatch):
    nodes, edges = _fixture()
    monkeypatch.setattr(W, "GAP_CLOSE_UM", 5.8)
    monkeypatch.setattr(W, "GAP_CLOSE_MAX_GAP", 2)
    monkeypatch.setattr(W, "GAP_REFINE_SYNTHETIC", False)
    monkeypatch.setattr(W, "GAP_CLOSE_MAX_ADDED_FRAC", 1.0)
    monkeypatch.setattr(W, "PREFIX_DENSITY_PRIOR_UM", {"crop": 40.0})
    monkeypatch.setattr(W, "GLOBAL_DENSITY_MEDIAN_UM", 40.0)

    monkeypatch.setattr(W, "GAP_DENSITY_ADAPTIVE", False)
    _, base_edges = W.close_single_frame_gaps(copy.deepcopy(nodes), copy.deepcopy(edges), _stats(), "crop_x")
    assert len(base_edges) == len(edges)

    monkeypatch.setattr(W, "GAP_DENSITY_ADAPTIVE", True)
    stats = _stats()
    _, adaptive_edges = W.close_single_frame_gaps(copy.deepcopy(nodes), copy.deepcopy(edges), stats, "crop_x")
    assert len(adaptive_edges) > len(edges)
    assert stats["gap_density_candidates_expanded"] > 0
    assert stats["gap_density_selected_outside_base"] > 0
