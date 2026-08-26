"""Software contracts for the peak-snap arms of scripts/win_bet/dc_subvoxel_refine.py (PKT-0017 amendment 1)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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


def test_heatmap_peaks_finds_each_cell_once():
    hm = gaussian_heatmap((32, 48, 48), [(10.3, 20.6, 30.2), (10.0, 26.0, 30.0)])
    peaks = dc.heatmap_peaks(hm, 0.10)
    assert len(peaks) == 2
    assert sorted(map(tuple, peaks.tolist())) == [(10, 21, 30), (10, 26, 30)]


def test_snap_moves_detection_to_nearest_peak_and_resolves_collisions():
    # two cells, pooled centres (10.3, 20.6, 30.2) and (10.0, 26.0, 30.0): 5.4 pooled voxels = 8.8 um apart in y
    hm = gaussian_heatmap((32, 48, 48), [(10.3, 20.6, 30.2), (10.0, 26.0, 30.0)])
    det = np.array([
        [12.0, 84.0, 120.0],   # ~2 slices off cell 1 (3.3 um) -> wants cell 1
        [11.0, 82.0, 120.0],   # nearer to cell 1 -> wins it; the first keeps its original coordinate
        [10.0, 104.0, 120.0],  # cell 2
        [10.0, 160.0, 120.0],  # no peak within any radius -> original
    ])
    res = dc.snap_points(hm, det, pool=4)
    assert res["n_peaks"] == 2
    a = "snap_r4"
    assert res[f"moved_{a}"].tolist() == [False, True, True, False]
    assert np.allclose(res[a][1], [10.3, 82.4, 120.8], atol=[0.15, 0.6, 0.6])
    assert np.allclose(res[a][2], [10.0, 104.0, 120.0], atol=[0.15, 0.6, 0.6])
    assert np.array_equal(res[a][0], det[0])
    assert np.array_equal(res[a][3], det[3])
    # the radius arms differ only by reach: r3 cannot reach the 3.3 um detection, r5 can
    assert not res["moved_snap_r3"][0]


def test_snap_unique_mode_refuses_ambiguous_detection():
    hm = gaussian_heatmap((32, 48, 48), [(10.0, 20.0, 30.0), (10.0, 22.0, 30.0)])  # 3.25 um apart
    det = np.array([[10.0, 84.0, 120.0]])  # pooled y=21: both peaks within 4 um
    res = dc.snap_points(hm, det, pool=4)
    assert res["moved_snap_r4"][0]
    assert not res["moved_snapu_r4"][0]
    assert np.array_equal(res["snapu_r4"][0], det[0])


def test_snap_no_peaks_keeps_everything():
    hm = np.full((16, 16, 16), 0.02, dtype=np.float32)
    det = np.array([[5.0, 20.0, 20.0]])
    res = dc.snap_points(hm, det, pool=4)
    assert res["n_peaks"] == 0
    for a in dc.SNAP_ARMS:
        assert np.array_equal(res[a], det)
        assert not res[f"moved_{a}"][0]
