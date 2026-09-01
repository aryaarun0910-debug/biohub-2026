"""Fail-closed T=2 exact-correspondence point substrate for Biohub-X.

This module builds a *point sanity substrate*, not images and not a scientific result.  A source
point and its target are tied by lineage identity through a known physical-space warp.  The warp
is composed of an affine map, a smooth local field and per-point jitter.  Candidate negatives are
nearby *other warped points*, and removing a source creates an explicit no-parent label.

The important boundaries are mechanical:

* coordinates enter and leave in full-resolution ``(z, y, x)`` voxels, but every deformation and
  every distance is computed in anisotropic physical microns;
* a source-movie split manifest is required before generation and is hashed into the result;
* a displacement calibration target is required and checked before candidate rows are built;
* the temporal window is fixed at two and any other value is refused.

Nothing here trains a model, touches a registry, or writes an artifact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contract import SCALE_UM


class SyntheticT2Refusal(RuntimeError):
    """A contract violation.  The generator never repairs or silently drops bad input."""


@dataclass(frozen=True)
class WarpConfig:
    """Known physical-space warp parameters.

    ``affine_zyx`` acts about the per-crop point-cloud centre.  Local-field control points and
    weights, jitter and source dropout are sampled from independent children of ``seed`` and are
    returned in the result so the transformation can be reconstructed exactly.
    """

    seed: int = 314159
    temporal_window: int = 2
    k_hard_negatives: int = 8
    source_dropout: float = 0.15
    affine_zyx: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.002, 0.0),
        (-0.002, 1.0, 0.003),
        (0.0, -0.003, 1.0),
    )
    translation_um_zyx: tuple[float, float, float] = (0.0, 0.55, 0.9)
    local_control_points: int = 8
    local_length_um: float = 24.0
    local_amplitude_um_zyx: tuple[float, float, float] = (0.12, 0.25, 0.25)
    jitter_std_um_zyx: tuple[float, float, float] = (0.025, 0.06, 0.06)


@dataclass(frozen=True)
class CalibrationTarget:
    """Externally supplied fold-specific displacement target.

    The target deliberately contains no built-in fold measurements.  Registry-backed values are
    supplied by the caller, keeping this software contract separate from mutable measurements.
    """

    name: str
    median_um: float
    p99_um: float
    per_axis_median_um_zyx: tuple[float, float, float]
    median_tolerance_um: float = 0.15
    p99_relative_tolerance: float = 0.20
    per_axis_tolerance_um_zyx: tuple[float, float, float] = (0.15, 0.15, 0.15)
    anchor_fact: str = "FACT-0040"


def _matrix3(value: object, where: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        raise SyntheticT2Refusal(f"{where}: expected a finite 3x3 matrix, got {arr.shape}")
    return arr


def _vec3(value: object, where: str, *, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise SyntheticT2Refusal(f"{where}: expected a finite length-3 vector, got {arr.shape}")
    if nonnegative and np.any(arr < 0):
        raise SyntheticT2Refusal(f"{where}: values must be non-negative")
    return arr


def voxel_to_um(points_vox_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_vox_zyx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise SyntheticT2Refusal(
            f"points_vox_zyx: expected finite (N, 3) in (z, y, x), got {points.shape}")
    return points * np.asarray(SCALE_UM, dtype=np.float64)


def um_to_voxel(points_um_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_um_zyx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise SyntheticT2Refusal(
            f"points_um_zyx: expected finite (N, 3) in (z, y, x), got {points.shape}")
    return points / np.asarray(SCALE_UM, dtype=np.float64)


def displacement_statistics(displacement_um_zyx: np.ndarray) -> dict:
    """The quantities required by the T=2 calibration contract."""
    delta = np.asarray(displacement_um_zyx, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 3 or delta.shape[0] < 2:
        raise SyntheticT2Refusal(
            f"displacement calibration needs at least two (z,y,x) rows, got {delta.shape}")
    if not np.isfinite(delta).all():
        raise SyntheticT2Refusal("displacement calibration received non-finite values")
    norm = np.linalg.norm(delta, axis=1)
    axis = np.median(np.abs(delta), axis=0)
    return {
        "n": int(len(delta)),
        "median_um": float(np.median(norm)),
        "p99_um": float(np.percentile(norm, 99)),
        "per_axis_median_um_zyx": axis.astype(float).tolist(),
    }


def displacement_calibration_gate(displacement_um_zyx: np.ndarray,
                                  target: CalibrationTarget) -> dict:
    """Refuse unless norm median, norm p99 and all axis medians meet the external target."""
    got = displacement_statistics(displacement_um_zyx)
    want_axis = _vec3(target.per_axis_median_um_zyx, "calibration axis target",
                      nonnegative=True)
    tol_axis = _vec3(target.per_axis_tolerance_um_zyx, "calibration axis tolerance",
                     nonnegative=True)
    if not np.isfinite(target.median_um) or target.median_um < 0:
        raise SyntheticT2Refusal("calibration median target must be finite and non-negative")
    if not np.isfinite(target.p99_um) or target.p99_um <= 0:
        raise SyntheticT2Refusal("calibration p99 target must be finite and positive")
    if target.median_tolerance_um < 0 or not 0 <= target.p99_relative_tolerance < 1:
        raise SyntheticT2Refusal("calibration tolerances are invalid")

    got_axis = np.asarray(got["per_axis_median_um_zyx"])
    median_error = abs(got["median_um"] - target.median_um)
    p99_error_fraction = abs(got["p99_um"] - target.p99_um) / target.p99_um
    axis_errors = np.abs(got_axis - want_axis)
    checks = {
        "median": bool(median_error <= target.median_tolerance_um),
        "p99": bool(p99_error_fraction <= target.p99_relative_tolerance),
        "axis_z": bool(axis_errors[0] <= tol_axis[0]),
        "axis_y": bool(axis_errors[1] <= tol_axis[1]),
        "axis_x": bool(axis_errors[2] <= tol_axis[2]),
    }
    report = {
        "target": target.name,
        "anchor_fact": target.anchor_fact,
        "observed": got,
        "expected": {
            "median_um": float(target.median_um),
            "p99_um": float(target.p99_um),
            "per_axis_median_um_zyx": want_axis.tolist(),
        },
        "errors": {
            "median_um": float(median_error),
            "p99_relative": float(p99_error_fraction),
            "per_axis_um_zyx": axis_errors.tolist(),
        },
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    if not report["passes"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SyntheticT2Refusal(
            "displacement calibration refused before candidate generation; failed "
            + ", ".join(failed))
    return report


def _split_digest(split_by_crop: Mapping[str, str]) -> str:
    payload = json.dumps(sorted((str(k), str(v)) for k, v in split_by_crop.items()),
                         separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def assert_split_before_generation(crop_ids: np.ndarray, split_name: str,
                                   split_by_crop: Mapping[str, str] | None) -> dict:
    """The generator's capability boundary: it can see only one preassigned split."""
    if not split_name:
        raise SyntheticT2Refusal("split_name is required before generation")
    if split_by_crop is None or not isinstance(split_by_crop, Mapping):
        raise SyntheticT2Refusal("split_by_crop manifest is required before generation")
    crops = np.asarray(crop_ids).astype(str)
    unique = sorted(set(crops.tolist()))
    missing = [crop for crop in unique if crop not in split_by_crop]
    wrong = [crop for crop in unique if split_by_crop.get(crop) != split_name]
    if missing:
        raise SyntheticT2Refusal(f"split manifest has no assignment for {missing}")
    if wrong:
        detail = {crop: split_by_crop.get(crop) for crop in wrong}
        raise SyntheticT2Refusal(
            f"generator for split {split_name!r} was given crop(s) from another split: {detail}")
    return {
        "split": split_name,
        "crops": unique,
        "manifest_sha256": _split_digest(split_by_crop),
        "checked_before_generation": True,
    }


def _smooth_local_field(points_um: np.ndarray, crop_ids: np.ndarray, cfg: WarpConfig,
                        rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    n_controls = int(cfg.local_control_points)
    if n_controls < 0 or cfg.local_length_um <= 0:
        raise SyntheticT2Refusal("local control count must be >=0 and length must be positive")
    amplitude = _vec3(cfg.local_amplitude_um_zyx, "local amplitude", nonnegative=True)
    out = np.zeros_like(points_um)
    records: list[dict] = []
    if n_controls == 0 or not np.any(amplitude):
        return out, records

    for crop in sorted(set(crop_ids.tolist())):
        idx = np.flatnonzero(crop_ids == crop)
        lo, hi = points_um[idx].min(axis=0), points_um[idx].max(axis=0)
        controls = rng.uniform(lo, hi, size=(n_controls, 3))
        weights = rng.normal(size=(n_controls, 3)) * amplitude[None, :]
        sq = ((points_um[idx, None, :] - controls[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-sq / (2.0 * float(cfg.local_length_um) ** 2))
        out[idx] = (kernel @ weights) / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
        records.append({
            "crop": crop,
            "control_points_um_zyx": controls.tolist(),
            "control_weights_um_zyx": weights.tolist(),
        })
    return out, records


def known_physical_warp(source_vox_zyx: np.ndarray, crop_ids: np.ndarray,
                        config: WarpConfig) -> dict:
    """Apply the deterministic known warp and expose every additive component."""
    if config.temporal_window != 2:
        raise SyntheticT2Refusal(
            f"synthetic_t2 is T=2 only; temporal_window={config.temporal_window} is forbidden")
    affine = _matrix3(config.affine_zyx, "affine_zyx")
    translation = _vec3(config.translation_um_zyx, "translation_um_zyx")
    jitter_std = _vec3(config.jitter_std_um_zyx, "jitter_std_um_zyx", nonnegative=True)
    points_um = voxel_to_um(source_vox_zyx)
    crops = np.asarray(crop_ids).astype(str)
    if crops.shape != (len(points_um),):
        raise SyntheticT2Refusal("crop_ids must have one entry per source point")

    seed = np.random.SeedSequence(int(config.seed))
    local_seed, jitter_seed, dropout_seed = seed.spawn(3)
    local, local_records = _smooth_local_field(
        points_um, crops, config, np.random.default_rng(local_seed))
    affine_delta = np.zeros_like(points_um)
    for crop in sorted(set(crops.tolist())):
        idx = np.flatnonzero(crops == crop)
        centre = points_um[idx].mean(axis=0)
        affine_delta[idx] = ((points_um[idx] - centre) @ affine.T
                             - (points_um[idx] - centre)) + translation
    jitter = np.random.default_rng(jitter_seed).normal(size=points_um.shape) * jitter_std
    target_um = points_um + affine_delta + local + jitter
    target_vox = um_to_voxel(target_um)
    roundtrip = np.max(np.abs(voxel_to_um(target_vox) - target_um)) if len(target_um) else 0.0
    if roundtrip > 1e-9:
        raise SyntheticT2Refusal(f"physical/voxel round trip failed at {roundtrip:.3g} um")
    return {
        "source_um_zyx": points_um,
        "target_um_zyx": target_um,
        "target_vox_zyx": target_vox,
        "affine_delta_um_zyx": affine_delta,
        "local_delta_um_zyx": local,
        "jitter_um_zyx": jitter,
        "local_field": local_records,
        "dropout_seed": dropout_seed,
        "roundtrip_max_um": float(roundtrip),
    }


def _validate_identity(lineage_ids: np.ndarray, crop_ids: np.ndarray, n: int) -> np.ndarray:
    lineage = np.asarray(lineage_ids)
    if lineage.shape != (n,) or lineage.dtype.kind not in "iu":
        raise SyntheticT2Refusal("lineage_ids must be one integer id per source point")
    lineage = lineage.astype(np.int64)
    keys = [f"{crop_ids[i]}|{int(lineage[i])}" for i in range(n)]
    if len(keys) != len(set(keys)):
        raise SyntheticT2Refusal("lineage ids must be unique within each crop")
    return lineage


def _build_candidates(target_um: np.ndarray, crop_ids: np.ndarray, lineage: np.ndarray,
                      visible: np.ndarray, k: int) -> dict:
    if k < 1:
        raise SyntheticT2Refusal("k_hard_negatives must be >= 1")
    source_rows: list[int] = []
    target_rows: list[int] = []
    positive: list[bool] = []
    retrieval_dist: list[float] = []

    # The negative basis follows t2_correspondence_v1 exactly: nearest OTHER WARPED points.
    # Each selected warped point maps back to its stable source lineage for the candidate row.
    for target in range(len(target_um)):
        same_crop_visible = np.flatnonzero((crop_ids == crop_ids[target]) & visible)
        others = same_crop_visible[same_crop_visible != target]
        if len(others) < k:
            raise SyntheticT2Refusal(
                f"crop {crop_ids[target]!r} target {target} has {len(others)} visible hard "
                f"negatives, fewer than K={k}; refusing instead of shortening the row")
        dist = np.linalg.norm(target_um[others] - target_um[target], axis=1)
        # Stable tie-break: physical distance, then lineage id, then source row.
        order = np.lexsort((others, lineage[others], dist))[:k]
        negatives = others[order]
        if visible[target]:
            source_rows.append(target)
            target_rows.append(target)
            positive.append(True)
            retrieval_dist.append(0.0)
        for source, d in zip(negatives, dist[order]):
            source_rows.append(int(source))
            target_rows.append(target)
            positive.append(False)
            retrieval_dist.append(float(d))
    return {
        "candidate_source_index": np.asarray(source_rows, dtype=np.int64),
        "candidate_target_index": np.asarray(target_rows, dtype=np.int64),
        "candidate_is_positive": np.asarray(positive, dtype=bool),
        "candidate_retrieval_distance_um": np.asarray(retrieval_dist, dtype=np.float64),
    }


def audit_generated_split(batch: Mapping[str, object], split_by_crop: Mapping[str, str]) -> dict:
    """Recompute endpoint split membership and all label/count invariants from the arrays."""
    required = {
        "generation_split", "source_crop", "target_crop", "source_visible",
        "candidate_source_index", "candidate_target_index", "candidate_is_positive",
        "no_parent_target_index", "no_parent_label", "k_hard_negatives", "lineage_id",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise SyntheticT2Refusal(f"generated split audit missing fields {missing}")
    split = str(batch["generation_split"])
    source_crop = np.asarray(batch["source_crop"]).astype(str)
    target_crop = np.asarray(batch["target_crop"]).astype(str)
    visible = np.asarray(batch["source_visible"], dtype=bool)
    src = np.asarray(batch["candidate_source_index"], dtype=np.int64)
    tgt = np.asarray(batch["candidate_target_index"], dtype=np.int64)
    pos = np.asarray(batch["candidate_is_positive"], dtype=bool)
    null_target = np.asarray(batch["no_parent_target_index"], dtype=np.int64)
    null_label = np.asarray(batch["no_parent_label"], dtype=bool)
    lineage = np.asarray(batch["lineage_id"], dtype=np.int64)
    n, k = len(target_crop), int(batch["k_hard_negatives"])
    if len(source_crop) != n or visible.shape != (n,) or lineage.shape != (n,):
        raise SyntheticT2Refusal("generated point arrays disagree in length")
    if np.any(src < 0) or np.any(src >= n) or np.any(tgt < 0) or np.any(tgt >= n):
        raise SyntheticT2Refusal("candidate endpoint index is out of bounds")
    if not np.all(source_crop[src] == target_crop[tgt]):
        raise SyntheticT2Refusal("candidate pair crosses source crops")
    endpoint_crops = set(source_crop[src].tolist()) | set(target_crop[tgt].tolist())
    wrong = {c: split_by_crop.get(c) for c in endpoint_crops if split_by_crop.get(c) != split}
    if wrong:
        raise SyntheticT2Refusal(f"post-generation pair crosses split {split!r}: {wrong}")
    if not np.array_equal(null_target, np.arange(n, dtype=np.int64)):
        raise SyntheticT2Refusal("every target must carry exactly one ordered no-parent row")
    if not np.array_equal(null_label, ~visible):
        raise SyntheticT2Refusal("explicit no-parent labels disagree with source dropout")

    for target in range(n):
        rows = np.flatnonzero(tgt == target)
        if int((~pos[rows]).sum()) != k:
            raise SyntheticT2Refusal(f"target {target} does not carry exactly K={k} negatives")
        expected_positive = int(visible[target])
        if int(pos[rows].sum()) != expected_positive:
            raise SyntheticT2Refusal(
                f"target {target} has {int(pos[rows].sum())} positives; expected "
                f"{expected_positive} from dropout")
        if expected_positive:
            psrc = src[rows[pos[rows]]]
            if len(psrc) != 1 or lineage[psrc[0]] != lineage[target]:
                raise SyntheticT2Refusal("positive correspondence changed lineage identity")
    return {
        "passes": True,
        "split": split,
        "candidates": int(len(src)),
        "targets": int(n),
        "abstentions": int(null_label.sum()),
        "cross_split_pairs": 0,
    }


def generate_t2(source_vox_zyx: np.ndarray, lineage_ids: np.ndarray, crop_ids: np.ndarray,
                *, split_name: str, split_by_crop: Mapping[str, str] | None,
                calibration_target: CalibrationTarget | None,
                config: WarpConfig = WarpConfig()) -> dict:
    """Generate one deterministic, calibrated T=2 exact-correspondence batch."""
    source_vox = np.asarray(source_vox_zyx, dtype=np.float64)
    crops = np.asarray(crop_ids).astype(str)
    split_report = assert_split_before_generation(crops, split_name, split_by_crop)
    lineage = _validate_identity(lineage_ids, crops, len(source_vox))
    if calibration_target is None:
        raise SyntheticT2Refusal(
            "calibration_target is required; an uncalibrated generator may not build candidates")
    if not 0.0 <= config.source_dropout < 1.0:
        raise SyntheticT2Refusal("source_dropout must be in [0, 1)")

    warp = known_physical_warp(source_vox, crops, config)
    calibration = displacement_calibration_gate(
        warp["target_um_zyx"] - warp["source_um_zyx"], calibration_target)
    dropout_rng = np.random.default_rng(warp.pop("dropout_seed"))
    visible = dropout_rng.random(len(source_vox)) >= float(config.source_dropout)
    candidates = _build_candidates(
        warp["target_um_zyx"], crops, lineage, visible, int(config.k_hard_negatives))

    batch = {
        "schema": "biohubx_synthetic_t2_v1",
        "temporal_window": 2,
        "generation_split": split_name,
        "split_manifest_sha256": split_report["manifest_sha256"],
        "seed": int(config.seed),
        "scale_um_zyx": np.asarray(SCALE_UM, dtype=np.float64),
        "source_vox_zyx": source_vox,
        "source_crop": crops.copy(),
        "target_crop": crops.copy(),
        "lineage_id": lineage,
        "source_visible": visible,
        "no_parent_target_index": np.arange(len(source_vox), dtype=np.int64),
        "no_parent_label": ~visible,
        "k_hard_negatives": int(config.k_hard_negatives),
        "calibration": calibration,
        **warp,
        **candidates,
    }
    batch["split_audit"] = audit_generated_split(batch, split_by_crop or {})
    return batch

