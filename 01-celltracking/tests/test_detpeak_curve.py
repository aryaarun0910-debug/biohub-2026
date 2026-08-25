"""Contract tests for the detection-threshold replay harness.

Software contracts only. Which threshold to ship is an experiment result, not a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.win_bet import detpeak_curve as C


def test_matching_radius_and_scale_are_the_official_scorer_values():
    # src/biotrack/metric.py:28-29. A wrong radius silently changes every recall number.
    assert C.MAX_DISTANCE_UM == 7.0
    assert C.SCALE_UM == (1.625, 0.40625, 0.40625)


def test_downsampled_peak_grid_is_isotropic_1625():
    """zyx is written BEFORE coords[:, 1:] *= ds_arr, so um = zyx * downsample * scale.

    That product is isotropic 1.625. Getting this wrong is the 4x-in-z error the forum
    warned about (discussion/733973): a radius expressed in voxels is wrong in one axis by 4x.
    """
    got = C.peaks_to_um(np.array([[1, 1, 1]]))
    np.testing.assert_allclose(got, [[1.625, 1.625, 1.625]])


def test_matching_is_one_to_one_so_duplicates_earn_nothing():
    """"Duplicating a detection costs ~9% and buys nothing" -- discussion/733877.

    Two predictions on top of one GT node must match ONCE. A greedy matcher would return 2
    and hide exactly the penalty the metric imposes.
    """
    gt = np.array([[0.0, 0.0, 0.0]])
    twin = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert C.match_one_to_one(twin, gt) == 1


def test_matching_respects_the_radius():
    gt = np.array([[0.0, 0.0, 0.0]])
    inside = np.array([[6.9, 0.0, 0.0]])
    outside = np.array([[7.1, 0.0, 0.0]])
    assert C.match_one_to_one(inside, gt) == 1
    assert C.match_one_to_one(outside, gt) == 0


def test_matching_pairs_maximally_not_greedily():
    """A greedy nearest-first matcher can strand a GT node the assignment could have served."""
    gt = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    pred = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert C.match_one_to_one(pred, gt) == 2


def test_empty_inputs_are_zero_not_an_error():
    gt = np.array([[0.0, 0.0, 0.0]])
    assert C.match_one_to_one(np.empty((0, 3)), gt) == 0
    assert C.match_one_to_one(gt, np.empty((0, 3))) == 0


def test_curve_is_monotone_in_predicted_count():
    """The peak set at a lower threshold is a strict SUPERSET, so n_pred cannot increase
    with the threshold. If it does, the export or the sigmoid convention is wrong."""
    rng = np.random.default_rng(0)
    zyx = rng.integers(0, 20, size=(200, 3))
    logit = rng.normal(2.5, 2.0, size=200).astype(np.float32)
    peaks = {0: (zyx, logit)}
    gt = {0: C.peaks_to_um(zyx[:50])}
    rows = C.curve(peaks, gt, [0.5, 0.8, 0.96875, 0.99])
    counts = [r["n_pred"] for r in rows]
    assert counts == sorted(counts, reverse=True), counts


def test_projection_penalises_over_prediction_only_at_one_tenth():
    """The score line is 1 - 0.1 * over_prediction, and recall enters SQUARED.

    Encoding the tax at the wrong rate would flip the recommended direction.
    """
    assert C.OVER_PREDICTION_TAX == 0.1
    rng = np.random.default_rng(1)
    zyx = rng.integers(0, 30, size=(100, 3))
    peaks = {0: (zyx, np.full(100, 10.0, dtype=np.float32))}   # everything survives
    gt = {0: C.peaks_to_um(zyx[:10])}                          # 10 GT, 100 predicted
    row = C.curve(peaks, gt, [0.5], n_est=10)[0]
    assert row["n_pred"] == 100
    assert row["over_prediction"] == pytest.approx(9.0)        # 100/10 - 1
    expected = (row["node_recall"] ** 2) * (1.0 - 0.1 * 9.0)
    assert row["projected_edge_index"] == pytest.approx(expected)
