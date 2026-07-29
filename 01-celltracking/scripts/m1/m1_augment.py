"""M1 round-one augmentations: IMAGE-ONLY transforms with safe target semantics.

Every transform here changes voxel intensities ONLY. Coordinates, lineage targets and
masks are provably unchanged, so no target transformation or loss masking is required.
That property is enforced by tests/test_m1_augment.py, not merely asserted.

INCLUDED (round one):
  * brightness / gamma / contrast
  * Poisson (shot) + Gaussian read noise
  * separable axial/lateral PSF blur
  * mild frame-to-frame intensity drift
  (the existing coordinate-correct flip stays in the trainer, where it also flips coords)

DELIBERATELY EXCLUDED from round one -- these move or invalidate targets and each needs
explicit target transformation plus dedicated tests before it can be considered:
  * physical rescaling / anisotropy change   (moves coordinates)
  * localization jitter                      (moves coordinates)
  * missing-slice corruption                 (needs loss masking; can delete a GT node)

All ranges are frozen constants below. They are derived from fixed physical priors and
training-family statistics only -- never from the held-out embryo.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------- frozen ranges (M1 v1)
# Voxel scale is (z, y, x) = (1.625, 0.40625, 0.40625) um; z is ~4x coarser than xy, so the
# PSF blur sigma is specified in um and converted per axis, keeping it physically isotropic
# rather than isotropic in voxels.
VOXEL_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class AugmentConfig:
    p_brightness: float = 0.5
    brightness_delta: float = 0.10      # additive, on [0,1]-normalised intensities
    p_gamma: float = 0.5
    gamma_range: tuple = (0.75, 1.35)
    p_contrast: float = 0.5
    contrast_range: tuple = (0.85, 1.20)
    p_noise: float = 0.5
    poisson_scale: tuple = (60.0, 400.0)   # photons at full scale; lower = noisier
    read_sigma: tuple = (0.0, 0.02)        # gaussian read noise sigma on [0,1]
    p_blur: float = 0.4
    blur_sigma_um: tuple = (0.0, 0.9)      # physical PSF sigma, converted per axis
    p_drift: float = 0.3
    drift_max: float = 0.06                # max per-frame multiplicative drift amplitude

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = AugmentConfig()


# ------------------------------------------------------------------------- primitives
def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def separable_blur(vol: np.ndarray, sigma_um: float,
                   voxel_um: tuple = VOXEL_UM) -> np.ndarray:
    """Separable Gaussian blur with a PHYSICAL sigma, applied per axis in voxels.

    vol is (..., Z, Y, X). A physical sigma keeps the PSF isotropic in micrometres, which
    on this anisotropic grid means a much smaller kernel along z than along y/x.
    """
    if sigma_um <= 0:
        return vol
    out = vol.astype(np.float32, copy=True)
    for axis_off, vs in enumerate(voxel_um):          # 0->Z, 1->Y, 2->X
        s = sigma_um / vs
        if s < 1e-3:
            continue
        r = max(1, int(round(3 * s)))
        k = _gauss1d(s, r)
        ax = out.ndim - 3 + axis_off
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        # convolve along `ax` by summing shifted slices (small kernels; avoids scipy dep)
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(i, i + out.shape[ax])
            acc += w * p[tuple(sl)]
        out = acc
    return out


# ------------------------------------------------------------------------- transforms
def brightness(vol, rng, cfg=DEFAULT):
    return vol + np.float32(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta))


def gamma(vol, rng, cfg=DEFAULT):
    g = float(rng.uniform(*cfg.gamma_range))
    return np.clip(vol, 0.0, None) ** np.float32(g)


def contrast(vol, rng, cfg=DEFAULT):
    c = np.float32(rng.uniform(*cfg.contrast_range))
    m = np.float32(vol.mean())
    return (vol - m) * c + m


def shot_and_read_noise(vol, rng, cfg=DEFAULT):
    """Poisson shot noise at a sampled photon scale, plus Gaussian read noise."""
    lam = float(rng.uniform(*cfg.poisson_scale))
    v = np.clip(vol, 0.0, None)
    noisy = rng.poisson(v * lam).astype(np.float32) / np.float32(lam)
    sigma = float(rng.uniform(*cfg.read_sigma))
    if sigma > 0:
        noisy = noisy + rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32)
    return noisy


def psf_blur(vol, rng, cfg=DEFAULT):
    return separable_blur(vol, float(rng.uniform(*cfg.blur_sigma_um)))


def intensity_drift(vol, rng, cfg=DEFAULT):
    """Mild multiplicative drift across the window's frame axis.

    Expects (W, Z, Y, X). A per-frame gain models slow illumination/bleaching change; it
    is intensity-only and leaves every coordinate untouched.
    """
    if vol.ndim < 4:
        return vol
    w = vol.shape[0]
    amp = float(rng.uniform(0.0, cfg.drift_max))
    ramp = np.linspace(-amp, amp, w, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1].copy()
    return vol * (1.0 + ramp).reshape((w,) + (1,) * (vol.ndim - 1))


ORDER = [("p_brightness", brightness), ("p_gamma", gamma), ("p_contrast", contrast),
         ("p_noise", shot_and_read_noise), ("p_blur", psf_blur), ("p_drift", intensity_drift)]


def apply_augmentations(vol: np.ndarray, rng: np.random.Generator,
                        cfg: AugmentConfig = DEFAULT) -> np.ndarray:
    """Apply the frozen M1 image-only pipeline. Coordinates/targets are NEVER touched."""
    out = vol.astype(np.float32, copy=True)
    for prob_attr, fn in ORDER:
        if rng.random() < getattr(cfg, prob_attr):
            out = fn(out, rng, cfg)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def config_fingerprint(cfg: AugmentConfig = DEFAULT) -> str:
    import hashlib
    import json as _json
    return hashlib.sha256(_json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(DEFAULT.to_dict(), indent=2))
    print("fingerprint:", config_fingerprint())


# ---------------------------------------------------------------- trainer adapter
def as_trainer_augmentation(cfg: AugmentConfig = DEFAULT):
    """Adapt the M1 image-only pipeline to the trainer's augmentation contract.

    The trainer calls ``aug(imgs, coords, masks, *, rng) -> (imgs, coords, masks)`` where
    ``imgs`` is a torch tensor of shape (W, *spatial). Coordinates and masks are returned
    UNCHANGED -- that is the property that makes round-one augmentation target-safe.
    """
    def _aug(imgs, coords, masks, *, rng):
        import torch
        arr = imgs.detach().cpu().float().numpy()
        out = apply_augmentations(arr, rng, cfg)
        return torch.from_numpy(out).to(imgs.device, imgs.dtype), coords, masks
    _aug.__name__ = "m1_image_only_augment"
    return _aug


# Source patch for the trainer's UNSEEDED per-sample RNG. FrameWindowDataset.__getitem__
# creates `np.random.default_rng()` with no seed, which is the root reproducibility defect.
# `idx` is in scope there, so the replacement derives the generator from
# (seed, epoch, idx) -- reproducible, yet decorrelated across epochs and samples.
RNG_PATCH_OLD = "            rng = np.random.default_rng()"
RNG_PATCH_NEW = ("            rng = np.random.default_rng("
                 "[M1_SEED, int(M1_EPOCH[0]), int(idx)])")
