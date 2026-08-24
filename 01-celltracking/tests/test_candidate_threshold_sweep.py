from __future__ import annotations

import numpy as np

from scripts.win_bet.candidate_threshold_sweep import match_gt_to_pred_one_to_one


def test_candidate_sweep_matching_is_one_to_one_under_collision() -> None:
    pred_ids = np.array([100])
    pred_pos = np.array([[0.0, 0.0, 0.0]])
    gt_ids = np.array([1, 2])
    gt_pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    matched = match_gt_to_pred_one_to_one(pred_ids, pred_pos, gt_ids, gt_pos)
    assert matched == {1: 100}


def test_candidate_sweep_matching_maximizes_scorer_weight() -> None:
    pred_ids = np.array([10, 20])
    pred_pos = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    gt_ids = np.array([1, 2])
    gt_pos = np.array([[0.2, 0.0, 0.0], [5.9, 0.0, 0.0]])
    assert match_gt_to_pred_one_to_one(pred_ids, pred_pos, gt_ids, gt_pos) == {
        1: 10, 2: 20,
    }
