"""Software contracts for scripts/win_bet/dc_subvoxel_refine.py (LEVER-0021 instrument).

These test the refinement arithmetic and its guards on synthetic heatmaps. They say nothing
about whether refinement helps the score - that is PKT-0017's job, not a unit test's.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dc_subvoxel_refine", ROOT / "scripts" / "win_bet" / "dc_subvoxel_refine.py")
dc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dc)


def gaussian_heatmap(shape, centres, sigma=1.0, amp=0.8):
    hm = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)
    for cz, cy, cx in centres:
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        np.maximum(hm, (amp * np.exp(-0.5 * d2 / sigma ** 2)).astype(np.float32), out=hm)
    return hm


def test_com1_recovers_subvoxel_centre_with_target_decode():
    # heatmap centre at pooled (10.3, 20.6, 30.2) -> full-res (10.3, 82.4, 120.8)
    hm = gaussian_heatmap((32, 48, 48), [(10.3, 20.6, 30.2)])
    original = np.array([[10.0, 84.0, 120.0]])  # an integer detection nearby
    res = dc.refine_points(hm, original, pool=4)
    got = res["com1"][0]
    assert res["moved_com1"][0]
    assert abs(got[0] - 10.3) < 0.15
    assert abs(got[1] - 82.4) < 0.6
    assert abs(got[2] - 120.8) < 0.6
    # the diagnostic decode differs by exactly the trainer's (pool-1)/2 offset in xy only
    off = res["com1_off"][0]
    assert off[0] == got[0]
    assert abs((off[1] - got[1]) - 1.5) < 1e-9
    assert abs((off[2] - got[2]) - 1.5) < 1e-9


def test_parabolic_arm_recovers_peak():
    hm = gaussian_heatmap((32, 48, 48), [(10.3, 20.6, 30.2)])
    res = dc.refine_points(hm, np.array([[10.0, 84.0, 120.0]]), pool=4)
    got = res["par1"][0]
    assert res["moved_par1"][0]
    assert abs(got[0] - 10.3) < 0.15
    assert abs(got[1] - 82.4) < 0.8
    assert abs(got[2] - 120.8) < 0.8


def test_no_signal_keeps_original_and_reports_hm_max():
    hm = np.full((32, 48, 48), 0.01, dtype=np.float32)
    original = np.array([[10.0, 84.0, 120.0]])
    res = dc.refine_points(hm, original, pool=4)
    for arm in dc.ARMS:
        assert np.array_equal(res[arm], original)
        assert not res[f"moved_{arm}"][0]
    assert res["hm_max"][0] == pytest.approx(0.01, abs=1e-6)


def test_large_shift_is_rejected_by_guard():
    # a peak 2 pooled voxels away in BOTH z and y (~4.2 um) is beyond the 3.0 um clamp
    hm = gaussian_heatmap((32, 48, 48), [(12.0, 23.0, 30.0)], sigma=0.7)
    original = np.array([[10.0, 84.0, 120.0]])  # pooled (10, 21, 30)
    res = dc.refine_points(hm, original, pool=4, max_shift_um=3.0)
    assert np.array_equal(res["com2"], original)
    assert not res["moved_com2"][0]
    # with a looser clamp the same window does move
    res2 = dc.refine_points(hm, original, pool=4, max_shift_um=10.0)
    assert res2["moved_com2"][0]


def test_border_nodes_do_not_index_out_of_range():
    hm = gaussian_heatmap((8, 8, 8), [(0.2, 0.3, 7.6)])
    original = np.array([[0.0, 0.0, 31.0], [64.0, 255.0, 255.0]])  # z=64 appears in real exports
    res = dc.refine_points(hm, original, pool=4)
    assert np.all(np.isfinite(res["com1"]))
    assert res["com1"].shape == (2, 3)


def test_frozen_match_eval_calibration_gate_and_direction():
    # two matched nodes; arm coordinates halve the residual, so median must drop by 50%
    ref = pd.DataFrame({
        "dataset": ["a", "a"], "node_id": [1, 2], "t": [0, 0],
        "z": [10.0, 20.0], "y": [40.0, 80.0], "x": [40.0, 80.0],
    })
    gt = pd.DataFrame({"dataset": ["a", "a"], "gt_id": [7, 8], "t": [0, 0],
                       "z": [10.0, 20.0], "y": [44.0, 84.0], "x": [40.0, 80.0]})
    for arm in dc.ARMS:
        ref[f"z_{arm}"] = ref["z"]
        ref[f"y_{arm}"] = ref["y"] + 2.0   # halfway to GT
        ref[f"x_{arm}"] = ref["x"]
        ref[f"moved_{arm}"] = True
    atlas_nodes = pd.DataFrame({"dataset": ["a", "a", "a"], "node_id": [1, 2, 3], "gt_id": [7, 8, -1],
                                "z": [10.0, 20.0, 5.0], "y": [40.0, 80.0, 5.0], "x": [40.0, 80.0, 5.0]})
    out = dc.frozen_match_eval(ref, atlas_nodes, gt)
    assert out["calibration_gate"]["ok"]
    assert out["n_matched"] == 2
    assert out["baseline"]["median_um"] == pytest.approx(4 * 0.40625)
    for arm in dc.ARMS:
        assert out["arms"][arm]["median_change_pct"] == pytest.approx(-50.0)


def test_select_crops_stride_and_explicit():
    names = [f"c{i:02d}" for i in range(10)]
    assert dc.select_crops(names, None, 3, None) == ["c00", "c03", "c06", "c09"]
    assert dc.select_crops(names, 2, 3, None) == ["c00", "c03"]
    assert dc.select_crops(names, None, None, "c05,c01") == ["c05", "c01"]
    with pytest.raises(SystemExit):
        dc.select_crops(names, None, None, "zz")
