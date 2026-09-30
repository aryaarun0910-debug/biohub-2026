"""Contract tests for the candidate-gate evaluation harness.

These enforce software contracts only. Whether a given k is worth shipping is an
experiment result, not a unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.win_bet.candidate_gate_eval import (
    DEFAULT_SCALE,
    _candidate_parents,
    _frame_trees,
    _gt_to_pred,
    knn_containment,
)


def _nodes(rows):
    return pd.DataFrame(rows, columns=["dataset", "node_id", "t", "z", "y", "x"])


def test_scale_is_the_deployed_geometry():
    # The harness measures distances in microns; a wrong scale silently changes every
    # kNN ordering. 1.625 z / 0.40625 lateral is the deployed voxel size.
    assert DEFAULT_SCALE == (1.625, 0.40625, 0.40625)


def test_gt_to_pred_skips_unmatched_nodes():
    pred = pd.DataFrame(
        {"dataset": ["a", "a", "a"], "gt_id": [-1, 7, 9], "node_id": [0, 1, 2]}
    )
    mapping = _gt_to_pred(pred)
    assert mapping == {("a", 7): 1, ("a", 9): 2}
    assert ("a", -1) not in mapping


def test_candidate_parents_groups_by_dataset_and_target():
    edges = pd.DataFrame(
        {
            "dataset": ["a", "a", "b"],
            "source_id": [10, 11, 10],
            "target_id": [20, 20, 20],
        }
    )
    cand = _candidate_parents(edges)
    # same target id in a different dataset must not collide
    assert cand[("a", 20)] == {10, 11}
    assert cand[("b", 20)] == {10}


def test_frame_trees_convert_voxels_to_microns():
    nodes = _nodes([("a", 0, 0, 1.0, 0.0, 0.0)])
    _, pos, time = _frame_trees(nodes, DEFAULT_SCALE)
    assert time[("a", 0)] == 0
    np.testing.assert_allclose(pos[("a", 0)], [1.625, 0.0, 0.0])


def test_knn_containment_finds_parent_and_respects_k():
    # frame 0 holds three sources; the true parent is the SECOND nearest to the child,
    # so it must be missed at k=1 and found at k=2.
    nodes = _nodes(
        [
            ("a", 100, 0, 0.0, 0.0, 0.0),   # nearest to child
            ("a", 101, 0, 0.0, 0.0, 10.0),  # true parent, second nearest
            ("a", 102, 0, 0.0, 0.0, 90.0),
            ("a", 200, 1, 0.0, 0.0, 1.0),   # the child
        ]
    )
    trees, pos, time = _frame_trees(nodes, DEFAULT_SCALE)
    gt2pred = {("a", 1): 101, ("a", 2): 200}
    out = knn_containment([("a", 1, 2)], trees, pos, time, gt2pred, [1, 2, 3])
    assert out["n_evaluated"] == 1
    assert out["hits"]["1"] == 0
    assert out["hits"]["2"] == 1
    assert out["recall"]["2"] == 1.0


def test_knn_containment_skips_pairs_with_no_mapping():
    nodes = _nodes([("a", 100, 0, 0.0, 0.0, 0.0), ("a", 200, 1, 0.0, 0.0, 1.0)])
    trees, pos, time = _frame_trees(nodes, DEFAULT_SCALE)
    # child has no GT->pred mapping, so the pair is not evaluable and must not be
    # counted as a miss -- silently counting it would understate recall.
    out = knn_containment([("a", 1, 2)], trees, pos, time, {("a", 1): 100}, [1])
    assert out["n_evaluated"] == 0
    assert out["recall"]["1"] == 0.0


def test_knn_containment_handles_k_larger_than_frame_population():
    nodes = _nodes([("a", 100, 0, 0.0, 0.0, 0.0), ("a", 200, 1, 0.0, 0.0, 1.0)])
    trees, pos, time = _frame_trees(nodes, DEFAULT_SCALE)
    out = knn_containment(
        [("a", 1, 2)], trees, pos, time, {("a", 1): 100, ("a", 2): 200}, [1, 5, 20]
    )
    assert out["n_evaluated"] == 1
    assert out["hits"]["20"] == 1


def test_knn_containment_requires_parent_in_the_previous_frame():
    # parent sits at t=0 but the child at t=5: the harness looks only at t-1, so this
    # pair must be skipped rather than scored against the wrong frame.
    nodes = _nodes([("a", 100, 0, 0.0, 0.0, 0.0), ("a", 200, 5, 0.0, 0.0, 1.0)])
    trees, pos, time = _frame_trees(nodes, DEFAULT_SCALE)
    out = knn_containment(
        [("a", 1, 2)], trees, pos, time, {("a", 1): 100, ("a", 2): 200}, [1]
    )
    assert out["n_evaluated"] == 0
