"""M1 augmentation tests.

The central property: every round-one transform is IMAGE-ONLY, so coordinates and lineage
targets are provably unchanged. That is what makes it safe to omit target transformation
and loss masking in round one. These tests enforce it rather than trusting the docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "m1"))

import m1_augment as M  # noqa: E402


@pytest.fixture
def vol():
    rng = np.random.default_rng(0)
    return rng.random((2, 8, 24, 24), dtype=np.float32)


ALL = [M.brightness, M.gamma, M.contrast, M.shot_and_read_noise, M.psf_blur,
       M.intensity_drift]


@pytest.mark.parametrize("fn", ALL)
def test_shape_and_dtype_preserved(vol, fn):
    """Shape preservation is what guarantees coordinate indices stay valid."""
    out = fn(vol, np.random.default_rng(1))
    assert out.shape == vol.shape
    assert out.dtype == np.float32


@pytest.mark.parametrize("fn", ALL)
def test_no_nan_or_inf(vol, fn):
    out = fn(vol, np.random.default_rng(2))
    assert np.isfinite(out).all()


def test_pipeline_output_bounded_and_finite(vol):
    out = M.apply_augmentations(vol, np.random.default_rng(3))
    assert out.shape == vol.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pipeline_is_deterministic_given_seed(vol):
    a = M.apply_augmentations(vol, np.random.default_rng(1234))
    b = M.apply_augmentations(vol, np.random.default_rng(1234))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_output(vol):
    a = M.apply_augmentations(vol, np.random.default_rng(1))
    b = M.apply_augmentations(vol, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_blur_uses_physical_sigma_so_z_is_blurred_less():
    """Voxels are (1.625, 0.40625, 0.40625) um. A physical sigma must blur z ~4x less
    than y/x; a naive voxel-space sigma would over-blur the coarse axis."""
    v = np.zeros((1, 9, 25, 25), np.float32)
    v[0, 4, 12, 12] = 1.0
    out = M.separable_blur(v, sigma_um=0.6)[0]
    # One voxel along z is 1.625 um but only 0.40625 um along y, so at the SAME voxel
    # offset the physical distance is 4x greater in z and the kernel must have decayed
    # far more. (Comparing out[4,12,12] both ways would index the same centre voxel.)
    one_step_z = out[5, 12, 12]
    one_step_y = out[4, 13, 12]
    assert one_step_z < one_step_y, (
        f"z should decay faster per voxel: z={one_step_z:.5f} y={one_step_y:.5f}")
    assert one_step_z < 0.25 * one_step_y      # markedly, not marginally
    assert np.isclose(out.sum(), 1.0, atol=1e-3)   # blur conserves mass


def test_blur_zero_sigma_is_identity(vol):
    np.testing.assert_array_equal(M.separable_blur(vol, 0.0), vol)


def test_intensity_drift_varies_across_frames_only(vol):
    """Drift must scale whole frames, never move structure within a frame."""
    rng = np.random.default_rng(7)
    out = M.intensity_drift(vol, rng)
    ratio = out / np.clip(vol, 1e-6, None)
    for f in range(vol.shape[0]):
        r = ratio[f]
        assert np.allclose(r, r.flat[0], rtol=1e-4), "drift must be uniform within a frame"


def test_noise_preserves_expected_signal_roughly(vol):
    """Poisson noise must not systematically shift the mean."""
    out = np.mean([M.shot_and_read_noise(vol, np.random.default_rng(i)) for i in range(12)], 0)
    assert abs(float(out.mean() - vol.mean())) < 0.02


def test_excluded_transforms_are_absent():
    """Round one must NOT contain coordinate-moving or target-invalidating transforms."""
    names = dir(M)
    for banned in ("rescale", "anisotropy", "jitter", "missing_slice", "dropout_slice"):
        assert not any(banned in n.lower() for n in names), f"{banned} must not be in M1 v1"


def test_config_is_frozen_and_fingerprinted():
    with pytest.raises(Exception):
        M.DEFAULT.p_gamma = 0.9          # frozen dataclass
    assert len(M.config_fingerprint()) == 16


def test_ranges_are_physically_sane():
    c = M.DEFAULT
    assert 0 < c.gamma_range[0] < 1 < c.gamma_range[1] < 2
    assert c.contrast_range[0] < 1 < c.contrast_range[1]
    assert c.blur_sigma_um[0] == 0.0 and c.blur_sigma_um[1] <= 1.5
    assert c.poisson_scale[0] > 0
    assert 0 <= c.drift_max <= 0.2
