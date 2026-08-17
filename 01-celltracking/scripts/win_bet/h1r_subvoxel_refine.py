r"""H1-R sub-voxel refine -- parabolic peak interpolation for detection centroids.

AUDIT (2026-08-17): the deployed detector returns INTEGER voxel coords
(`vendor/.../predict_unet_transformer.py::extract_coords_from_logits`, lines 282-293:
`peak_idx = torch.nonzero(is_peak); coords = peak_idx.float()...astype(np.int16)`) at the
4x-downsampled grid -> localization quantized to ~1.6 um in xy. The 7um-match metric rewards
1/(1+d), and community EDA measured a cliff at sigma~2 um. This adds the standard quadratic
sub-voxel offset from the logit values at each peak's +/-1 neighbours (separable per axis).

INTEGRATION (a kaggle_edit patch on the detector, then a detector re-run to apply):
  after `peak_idx = torch.nonzero(is_peak[0,0])`, compute `off = parabolic_offset(det_logits, peak_idx)`
  and return `peak_idx.float() + off` as **float32** (drop the int16 cast). Downstream that
  integer-indexes UNet features already rounds/clamps (`_index_features` does `.long().clamp`),
  so passing sub-voxel floats through is safe; the geff export scales the float coords by
  voxel_size -> correct physical positions for scoring.

This module is the math + a self-test; it does not modify vendor code.
"""
from __future__ import annotations

import numpy as np


def parabolic_offset(logit_vol: np.ndarray, peaks: np.ndarray, clamp: float = 0.5) -> np.ndarray:
    """Sub-voxel offset (N,3) for integer `peaks` (N,3 z,y,x) in `logit_vol` (Z,Y,X).

    Per axis: delta = 0.5*(L[-1]-L[+1]) / (L[-1] - 2*L[0] + L[+1]); 0 at borders or flat/convex.
    """
    Z, Y, X = logit_vol.shape
    off = np.zeros((len(peaks), 3), dtype=np.float32)
    dims = (Z, Y, X)
    for n, (z, y, x) in enumerate(peaks.astype(int)):
        for a, (c, lo) in enumerate(zip((z, y, x), (0, 0, 0))):
            if c <= 0 or c >= dims[a] - 1:
                continue
            idx_m = [z, y, x]; idx_m[a] = c - 1
            idx_p = [z, y, x]; idx_p[a] = c + 1
            lm = float(logit_vol[tuple(idx_m)])
            l0 = float(logit_vol[z, y, x])
            lp = float(logit_vol[tuple(idx_p)])
            denom = lm - 2.0 * l0 + lp
            if denom >= -1e-9:            # not a strict concave peak along this axis
                continue
            d = 0.5 * (lm - lp) / denom
            off[n, a] = float(np.clip(d, -clamp, clamp))
    return off


def _self_test() -> None:
    """Place 3D Gaussian peaks at known sub-voxel centres; check recovery vs integer argmax."""
    rng = np.random.default_rng(0)
    Z, Y, X = 24, 40, 40
    max_int_err, max_ref_err = 0.0, 0.0
    for _ in range(40):
        cz, cy, cx = rng.uniform(4, Z - 4), rng.uniform(4, Y - 4), rng.uniform(4, X - 4)
        zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
        g = np.exp(-(((zz - cz) / 1.6) ** 2 + ((yy - cy) / 1.6) ** 2 + ((xx - cx) / 1.6) ** 2))
        logit = np.log(np.clip(g, 1e-6, 1) / np.clip(1 - g, 1e-6, 1))  # inverse-sigmoid
        pk = np.array(np.unravel_index(np.argmax(g), g.shape))          # integer argmax
        off = parabolic_offset(logit, pk[None, :])[0]
        ref = pk + off
        true = np.array([cz, cy, cx])
        max_int_err = max(max_int_err, float(np.abs(pk - true).max()))
        max_ref_err = max(max_ref_err, float(np.abs(ref - true).max()))
    print(f"  integer-argmax max per-axis error : {max_int_err:.3f} vox")
    print(f"  parabolic-refined max per-axis err: {max_ref_err:.3f} vox")
    assert max_ref_err < max_int_err * 0.5, "refinement did not halve localization error"
    print("  SELF-TEST OK: parabolic sub-voxel refine reduces localization error > 2x")


if __name__ == "__main__":
    _self_test()
