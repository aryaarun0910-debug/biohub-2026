"""Structural preconditions for the combined P3 + D1 + D1-F export kernel.

These run on CPU with no GPU and no kernel. Every one of them is a claim the export block
depends on, and each has already been either asserted loosely or gotten wrong once:

  * the output grid is isotropic at 1.625 um -- the whole class-B argument rests on it, and my
    first draft of the export block double-multiplied `downsample` and would have inflated
    every y/x radius 4x;
  * the pool kernel is (3,3,3), i.e. +/-1.625 um suppression against ~6.5 um cell spacing;
  * split_0 and split_1 are genuinely different fold-specific checkpoints -- the LOEO retarget
    block warns the support pack ships only split_0, so fold-honesty is not free;
  * `detect_head` is 33 parameters, which is why class D cannot be read as encoder failure;
  * A/B/D partition the non-accepted voxels exactly, with no overlap and no gap;
  * the GT physical -> output-grid mapping round-trips.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))

WEIGHTS = ROOT / "artifacts" / "kaggle" / "weights_dataset"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
DOWNSAMPLE = (1, 4, 4)
POOL_KERNEL_UM = 5.0
DET_THRESHOLD = 0.96875   # BIOHUB_DET_THRESHOLD, deployment env cell02

pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "config_split_0.json").exists(),
    reason="weights_dataset not present locally",
)


def _cfg(split: int) -> dict:
    return json.loads((WEIGHTS / f"config_split_{split}.json").read_text())


def test_shipped_config_matches_assumed_geometry():
    for split in (0, 1):
        c = _cfg(split)
        assert tuple(c["downsample"]) == DOWNSAMPLE
        assert c["unet_out_channels"] == 32
        assert c["pool_kernel_um"] == POOL_KERNEL_UM


def test_output_grid_is_isotropic_at_1_625_um():
    """scale * downsample must be isotropic; the class-B argument depends on it."""
    step = tuple(s * d for s, d in zip(VOXEL_SCALE_UM, DOWNSAMPLE))
    assert step == pytest.approx((1.625, 1.625, 1.625), abs=1e-9), step
    assert len(set(round(v, 9) for v in step)) == 1, "grid is NOT isotropic"


def test_pool_kernel_is_3x3x3_and_suppresses_only_1_625um():
    """The deployed suppression radius must be far tighter than cell spacing (~6.5 um)."""
    from predict_unet_transformer import pool_kernel_from_um

    step = tuple(s * d for s, d in zip(VOXEL_SCALE_UM, DOWNSAMPLE))
    k = pool_kernel_from_um(POOL_KERNEL_UM, step)
    assert tuple(k) == (3, 3, 3), k
    radius_um = max((kk // 2) * st for kk, st in zip(k, step))
    assert radius_um == pytest.approx(1.625, abs=1e-9)
    assert radius_um < 6.5 / 2, "pool could merge distinct cells; class-B argument breaks"


def test_fold_checkpoints_are_genuinely_different():
    """The LOEO retarget block warns the pack ships only split_0. Fold honesty is not free."""
    torch = pytest.importorskip("torch")
    p0, p1 = WEIGHTS / "edge_predictor_best_split_0.pth", WEIGHTS / "edge_predictor_best_split_1.pth"
    h0 = hashlib.sha256(p0.read_bytes()).hexdigest()
    h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    assert h0 != h1, "split_0 and split_1 are the SAME FILE -- fold 1 is not fold-honest"

    a = torch.load(p0, map_location="cpu")
    b = torch.load(p1, map_location="cpu")
    diff = sum(1 for k in a if not torch.equal(a[k].float(), b[k].float()))
    assert diff > 100, f"only {diff} tensors differ; checkpoints are near-identical"
    assert any(k.startswith("unet.") for k in a), "checkpoint lacks U-Net keys"


def test_detect_head_is_33_parameters():
    """Class D cannot be read as encoder failure through a 33-parameter linear map."""
    torch = pytest.importorskip("torch")
    a = torch.load(WEIGHTS / "edge_predictor_best_split_0.pth", map_location="cpu")
    keys = [k for k in a if "detect_head" in k]
    assert set(keys) == {"detect_head.weight", "detect_head.bias"}
    assert tuple(a["detect_head.weight"].shape) == (1, 32, 1, 1, 1)
    assert sum(a[k].numel() for k in keys) == 33


def _classify(logit, pooled, prob, thr):
    is_lm, over = logit == pooled, prob > thr
    if is_lm and over:
        return "accepted"
    return "A" if is_lm else ("B" if over else "D")


def test_abd_partition_is_exclusive_and_exhaustive():
    """Every voxel lands in exactly one of accepted/A/B/D, matching the deployed peak rule."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    lg = torch.from_numpy(rng.normal(0, 3, size=(1, 8, 24, 24)).astype(np.float32))
    pooled = F.max_pool3d(lg.unsqueeze(0), (3, 3, 3), stride=1, padding=1)[0]
    prob = torch.sigmoid(lg)
    # The DEPLOYED threshold, not 0.5. BIOHUB_DET_THRESHOLD = 0.96875 => logit > ln(31) ~ 3.434.
    # At 0.5 the fixture is degenerate: a 3x3x3 local max is the max of 27 draws and is almost
    # always positive, so class A empties. The real operating point is far more aggressive,
    # which is exactly why class A is expected to be populated on real data.
    thr = DET_THRESHOLD

    counts = {"accepted": 0, "A": 0, "B": 0, "D": 0}
    Z, Y, X = lg.shape[1:]
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                counts[_classify(float(lg[0, z, y, x]), float(pooled[0, z, y, x]),
                                 float(prob[0, z, y, x]), thr)] += 1
    assert sum(counts.values()) == Z * Y * X, "partition is not exhaustive"

    # `accepted` must equal the deployed rule exactly
    deployed = int(((lg == pooled) & (prob > thr)).sum())
    assert counts["accepted"] == deployed, "our 'accepted' disagrees with the deployed peak rule"
    assert counts["A"] > 0 and counts["D"] > 0, "degenerate fixture"


def test_gt_physical_to_grid_mapping_round_trips():
    """GT is in ORIGINAL voxel coords; the export maps to the downsampled grid."""
    rng = np.random.default_rng(1)
    for _ in range(500):
        z = float(rng.integers(0, 64))
        y = float(rng.integers(0, 256))
        x = float(rng.integers(0, 256))
        gz, gy, gx = (int(round(z / DOWNSAMPLE[0])), int(round(y / DOWNSAMPLE[1])),
                      int(round(x / DOWNSAMPLE[2])))
        # the grid cell must contain the original point to within half a step
        back = (gz * DOWNSAMPLE[0], gy * DOWNSAMPLE[1], gx * DOWNSAMPLE[2])
        assert abs(back[0] - z) <= DOWNSAMPLE[0] / 2
        assert abs(back[1] - y) <= DOWNSAMPLE[1] / 2
        assert abs(back[2] - x) <= DOWNSAMPLE[2] / 2


def test_export_block_threads_gt_dir_and_does_not_double_scale():
    """Locks the two bugs found by reading the caller rather than assuming."""
    src = (ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py").read_text(encoding="utf-8")
    assert "TEST_DIR" not in src, "TEST_DIR is a notebook global, absent from the predict script"
    assert "_step = tuple(float(v) for v in voxel_size)" in src, (
        "voxel_size is already scale*downsample in predict_video; do not multiply again"
    )
    assert "gt_dir" in src


def test_harmonic_patch_does_not_touch_detection():
    """P3 and the D1 audit may share one encoder pass ONLY if harmonic leaves det_logits alone."""
    spec = json.loads((ROOT / "scripts" / "kaggle_specs" / "p3_harmonic.json").read_text())
    bi = [e for e in spec["edits"] if e["kind"] == "replace" and "_bi_new" in e.get("old", "")]
    assert len(bi) == 1
    for side in ("old", "new"):
        assert "det_logits" not in bi[0][side], "harmonic touches detection; kernels cannot merge"
        assert "edge_logits_pair" in bi[0][side]


def test_deployed_threshold_is_aggressive_and_implies_a_populated_class_A():
    """det_threshold = 0.96875 means logit > ln(31) ~ 3.434 to be accepted.

    Pre-registered consequence: a local maximum only has to fall below that to become class A,
    so A should be well populated on real data. If D1 instead returns almost all D, the loss of
    signal is far below the operating point and is not a thresholding artefact.
    """
    import math
    assert DET_THRESHOLD == 0.96875
    logit_needed = math.log(DET_THRESHOLD / (1.0 - DET_THRESHOLD))
    assert logit_needed == pytest.approx(3.4340, abs=1e-3), logit_needed
