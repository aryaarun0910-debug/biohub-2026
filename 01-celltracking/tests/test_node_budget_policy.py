from __future__ import annotations

import numpy as np
import pytest

from scripts.win_bet import node_budget_policy as policy


def test_exact_budget_uses_strict_scalar_threshold():
    logits = np.asarray([-2.0, 0.1, 1.0, 2.0, 3.0])
    got = policy.threshold_for_budget(logits, 3)
    assert got["selected_count"] == 3
    assert np.count_nonzero(policy.sigmoid(logits) > got["threshold"]) == 3
    assert got["tie_limited"] is False


def test_ties_are_not_split_and_smaller_count_wins_equal_error():
    logits = np.asarray([0.0, 0.0, 2.0, 2.0])
    got = policy.threshold_for_budget(logits, 3)
    assert got["selected_count"] == 2
    assert got["tie_limited"] is True


def test_directory_freeze_checks_pipeline_heartbeat_and_routes_only_on_scores(tmp_path):
    peaks = tmp_path / "peaks"
    peaks.mkdir()
    logits = np.asarray([0.1, 1.0, 2.0], dtype=np.float32)
    threshold = 0.7
    count = int(np.count_nonzero(policy.sigmoid(logits) > threshold))
    np.savez(peaks / "crop_a.npz", t=np.zeros(3), zyx=np.zeros((3, 3)), logit=logits,
             pipeline_threshold=threshold, pipeline_peak_count=count)
    got = policy.freeze_directory(peaks, {"crop_a": 2}, target_multiplier=1.0)
    assert got["forbidden_routes"] == ["dataset", "embryo", "crop_name", "ground_truth"]
    assert got["crops"]["crop_a"]["pipeline_peak_count"] == count

    np.savez(peaks / "crop_a.npz", t=np.zeros(3), zyx=np.zeros((3, 3)), logit=logits,
             pipeline_threshold=threshold, pipeline_peak_count=count + 1)
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        policy.freeze_directory(peaks, {"crop_a": 2}, target_multiplier=1.0)


def test_invalid_and_incomplete_inputs_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        policy.threshold_for_budget(np.asarray([1.0]), -1)
    with pytest.raises(FileNotFoundError, match="no DetPeak"):
        policy.freeze_directory(tmp_path, {}, target_multiplier=1.0)
