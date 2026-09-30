"""Software-contract tests for Biohub-X's T=2 exact-correspondence sanity substrate."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from biohubx.synthetic_t2 import (  # noqa: E402
    CalibrationTarget,
    SyntheticT2Refusal,
    WarpConfig,
    audit_generated_split,
    displacement_statistics,
    generate_t2,
    known_physical_warp,
    voxel_to_um,
)


def _points(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    # Deliberately different ranges on each index axis.  Treating this as isotropic would change
    # both the warp and the hard-negative order, so the fixture discriminates the conventions.
    i = np.arange(n, dtype=np.float64)
    points = np.column_stack((2 + (i % 12), 4 + (3 * i) % 90, 7 + (7 * i) % 120))
    lineage = np.arange(1000, 1000 + n, dtype=np.int64)
    crops = np.asarray(["44b6_a"] * n)
    return points, lineage, crops, {"44b6_a": "train", "6bba_b": "validation"}


def _target_for(points: np.ndarray, crops: np.ndarray, cfg: WarpConfig) -> CalibrationTarget:
    warp = known_physical_warp(points, crops, cfg)
    stats = displacement_statistics(warp["target_um_zyx"] - warp["source_um_zyx"])
    return CalibrationTarget(
        name="fixture",
        median_um=stats["median_um"],
        p99_um=stats["p99_um"],
        per_axis_median_um_zyx=tuple(stats["per_axis_median_um_zyx"]),
        median_tolerance_um=1e-10,
        p99_relative_tolerance=1e-10,
        per_axis_tolerance_um_zyx=(1e-10, 1e-10, 1e-10),
    )


def _batch(cfg: WarpConfig | None = None) -> tuple[dict, dict[str, str]]:
    points, lineage, crops, manifest = _points()
    cfg = cfg or WarpConfig(seed=17, k_hard_negatives=5, source_dropout=0.2)
    target = _target_for(points, crops, cfg)
    return generate_t2(
        points, lineage, crops, split_name="train", split_by_crop=manifest,
        calibration_target=target, config=cfg), manifest


def test_known_warp_is_physical_anisotropic_and_components_reconstruct_target():
    points, _lineage, crops, _manifest = _points()
    cfg = WarpConfig(
        seed=3, k_hard_negatives=3, source_dropout=0.0,
        affine_zyx=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_um_zyx=(1.625, 0.8125, 0.40625),
        local_control_points=0, local_amplitude_um_zyx=(0.0, 0.0, 0.0),
        jitter_std_um_zyx=(0.0, 0.0, 0.0),
    )
    warp = known_physical_warp(points, crops, cfg)
    delta_um = warp["target_um_zyx"] - warp["source_um_zyx"]
    assert np.allclose(delta_um, [1.625, 0.8125, 0.40625])
    # The same physical translation is 1, 2, 1 voxels because full-res data are 4:1:1.
    assert np.allclose(warp["target_vox_zyx"] - points, [1.0, 2.0, 1.0])
    rebuilt = (warp["source_um_zyx"] + warp["affine_delta_um_zyx"]
               + warp["local_delta_um_zyx"] + warp["jitter_um_zyx"])
    assert np.array_equal(rebuilt, warp["target_um_zyx"])
    assert warp["roundtrip_max_um"] < 1e-9


def test_affine_smooth_field_and_jitter_are_all_live_and_known():
    points, _lineage, crops, _manifest = _points()
    warp = known_physical_warp(points, crops, WarpConfig(seed=9))
    assert np.std(warp["affine_delta_um_zyx"], axis=0).max() > 0
    assert np.std(warp["local_delta_um_zyx"], axis=0).max() > 0
    assert np.std(warp["jitter_um_zyx"], axis=0).max() > 0
    assert warp["local_field"] and "control_points_um_zyx" in warp["local_field"][0]


def test_generation_is_bit_deterministic_for_one_seed():
    a, _ = _batch(WarpConfig(seed=123, k_hard_negatives=4, source_dropout=0.25))
    b, _ = _batch(WarpConfig(seed=123, k_hard_negatives=4, source_dropout=0.25))
    for key in (
        "target_um_zyx", "affine_delta_um_zyx", "local_delta_um_zyx", "jitter_um_zyx",
        "source_visible", "candidate_source_index", "candidate_target_index",
        "candidate_is_positive", "no_parent_label",
    ):
        assert np.array_equal(a[key], b[key]), key


def test_different_seed_changes_the_known_warp_and_dropout():
    a, _ = _batch(WarpConfig(seed=1, k_hard_negatives=4, source_dropout=0.25))
    b, _ = _batch(WarpConfig(seed=2, k_hard_negatives=4, source_dropout=0.25))
    assert not np.array_equal(a["target_um_zyx"], b["target_um_zyx"])
    assert not np.array_equal(a["source_visible"], b["source_visible"])


def test_hard_negatives_are_k_nearest_other_warped_points_in_microns():
    batch, _ = _batch(WarpConfig(seed=22, k_hard_negatives=5, source_dropout=0.0))
    target = batch["target_um_zyx"]
    src_rows = batch["candidate_source_index"]
    tgt_rows = batch["candidate_target_index"]
    pos = batch["candidate_is_positive"]
    for i in range(len(target)):
        got = src_rows[(tgt_rows == i) & ~pos]
        others = np.asarray([j for j in range(len(target)) if j != i])
        dist = np.linalg.norm(target[others] - target[i], axis=1)
        order = np.lexsort((others, batch["lineage_id"][others], dist))[:5]
        assert np.array_equal(got, others[order])


def test_dropout_removes_the_positive_and_sets_one_explicit_abstention_label():
    batch, _ = _batch(WarpConfig(seed=41, k_hard_negatives=4, source_dropout=0.35))
    assert batch["no_parent_target_index"].tolist() == list(range(len(batch["lineage_id"])))
    assert np.array_equal(batch["no_parent_label"], ~batch["source_visible"])
    assert batch["no_parent_label"].any(), "fixture must exercise abstention"
    for i, abstain in enumerate(batch["no_parent_label"]):
        rows = batch["candidate_target_index"] == i
        assert int(batch["candidate_is_positive"][rows].sum()) == int(not abstain)
        assert int((~batch["candidate_is_positive"][rows]).sum()) == 4


def test_split_manifest_is_a_required_pre_generation_capability_boundary():
    points, lineage, crops, _manifest = _points()
    cfg = WarpConfig(seed=5, k_hard_negatives=3)
    target = _target_for(points, crops, cfg)
    with pytest.raises(SyntheticT2Refusal, match="manifest is required before generation"):
        generate_t2(points, lineage, crops, split_name="train", split_by_crop=None,
                    calibration_target=target, config=cfg)
    with pytest.raises(SyntheticT2Refusal, match="another split"):
        generate_t2(points, lineage, crops, split_name="train",
                    split_by_crop={"44b6_a": "validation"},
                    calibration_target=target, config=cfg)


def test_post_generation_split_audit_can_reject_a_deliberately_cross_split_pair():
    batch, manifest = _batch(WarpConfig(seed=7, k_hard_negatives=3, source_dropout=0.1))
    broken = dict(batch)
    broken["source_crop"] = batch["source_crop"].copy()
    source = int(batch["candidate_source_index"][0])
    broken["source_crop"][source] = "6bba_b"
    with pytest.raises(SyntheticT2Refusal, match="crosses source crops|crosses split"):
        audit_generated_split(broken, manifest)


def test_calibration_is_required_before_candidate_rows_can_exist():
    points, lineage, crops, manifest = _points()
    with pytest.raises(SyntheticT2Refusal, match="calibration_target is required"):
        generate_t2(points, lineage, crops, split_name="train", split_by_crop=manifest,
                    calibration_target=None, config=WarpConfig(k_hard_negatives=3))


def test_calibration_gate_fails_closed_without_widening_tolerances():
    points, lineage, crops, manifest = _points()
    cfg = WarpConfig(seed=29, k_hard_negatives=3)
    good = _target_for(points, crops, cfg)
    bad = replace(good, median_um=good.median_um + 2.0)
    with pytest.raises(SyntheticT2Refusal, match="calibration refused.*median"):
        generate_t2(points, lineage, crops, split_name="train", split_by_crop=manifest,
                    calibration_target=bad, config=cfg)


def test_t_greater_than_two_is_refused_before_generation():
    points, _lineage, crops, _manifest = _points()
    with pytest.raises(SyntheticT2Refusal, match="T=2 only"):
        known_physical_warp(points, crops, WarpConfig(temporal_window=4))


def test_candidate_shortage_refuses_instead_of_quietly_reducing_k():
    points, lineage, crops, manifest = _points(n=6)
    cfg = WarpConfig(seed=4, k_hard_negatives=5, source_dropout=0.5)
    target = _target_for(points, crops, cfg)
    with pytest.raises(SyntheticT2Refusal, match="fewer than K=5"):
        generate_t2(points, lineage, crops, split_name="train", split_by_crop=manifest,
                    calibration_target=target, config=cfg)


def test_voxel_converter_refuses_wrong_shape_and_nonfinite_input():
    with pytest.raises(SyntheticT2Refusal):
        voxel_to_um(np.zeros((3, 2)))
    bad = np.zeros((3, 3))
    bad[0, 0] = np.nan
    with pytest.raises(SyntheticT2Refusal):
        voxel_to_um(bad)
