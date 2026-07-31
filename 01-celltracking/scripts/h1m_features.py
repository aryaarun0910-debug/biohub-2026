"""H1-M shared feature definitions for the MOTHER-DIVISION GATE.

The unit is one (embryo, mother, t) EVENT -- never a candidate daughter pair. The
question answered is P(this mother divides at t -> t+1), which is the question the
competition metric actually pays for and the one a pair-ranker cannot answer.

Everything here is deliberately SCALE-FREE or MICRON-SCALED so that a representation
fitted on Zebrahub (voxel 1.24 / 0.439 / 0.439 um) transfers to the competition crops
(voxel 1.625 / 0.40625 / 0.40625 um):

  * the raw per-node statistics come from `scripts/h1i_node_appearance._node_features`,
    imported verbatim, with the sampling kernel rebuilt for whichever voxel spacing the
    imagery has -- so the 6 um sphere, the 1.5/3/4.5/6 um shells and every moment are
    the same PHYSICAL quantity on both sides;
  * only ratios, log-ratios, differences and micron lengths are exposed to the model.
    H1-I established that every RAW-SCALE intensity feature flips sign between embryo
    families while every RATIO feature transfers, so raw intensity is excluded by
    construction and the two dynamic-range-normalised level terms are isolated in
    their own group so they can be ablated.

No feature in this module reads ground truth, family identity or crop identity.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1i_node_appearance as h1i  # noqa: E402  (kernel maths reused verbatim)

EPS = 1e-6
DTS = (-2, -1, 0, 1, 2)

VOX_COMPETITION = np.array([1.625, 0.40625, 0.40625])   # data/train/*.zarr multiscales
VOX_ZEBRAHUB_L0 = np.array([1.24, 0.439, 0.439])        # <E>.ome.zarr level 0


# ------------------------------------------------------------------ kernel
def build_kernel(vox: np.ndarray):
    """Exact re-derivation of h1i_node_appearance._build_kernel for arbitrary voxel
    spacing. Radii, shells and moment order are unchanged; only the voxel lattice
    the physical sphere is sampled on differs."""
    vox = np.asarray(vox, dtype=np.float64)
    hz = int(np.ceil(h1i.R_OUTER / vox[0]))
    hy = int(np.ceil(h1i.R_OUTER / vox[1]))
    hx = int(np.ceil(h1i.R_OUTER / vox[2]))

    dz = np.arange(-hz, hz + 1) * vox[0]
    dy = np.arange(-hy, hy + 1) * vox[1]
    dx = np.arange(-hx, hx + 1) * vox[2]
    DZ, DY, DX = np.meshgrid(dz, dy, dx, indexing="ij")
    r = np.sqrt(DZ ** 2 + DY ** 2 + DX ** 2)
    keep = r <= h1i.R_OUTER
    order = np.argsort(r[keep], kind="stable")

    dzk = DZ[keep][order].astype(np.float32)
    dyk = DY[keep][order].astype(np.float32)
    dxk = DX[keep][order].astype(np.float32)
    rk = r[keep][order].astype(np.float32)

    iz, iy, ix = np.meshgrid(np.arange(-hz, hz + 1), np.arange(-hy, hy + 1),
                             np.arange(-hx, hx + 1), indexing="ij")
    vox_off = np.stack([iz[keep][order], iy[keep][order], ix[keep][order]], axis=1)

    mat = np.empty((rk.size, h1i.N_MOM), dtype=np.float32)
    for n in (1, 2, 3, 4):
        for c, (i, j, k) in enumerate(h1i.MULTI[n]):
            mat[:, h1i.MULTI_OFF[n] + c] = (dzk ** i) * (dyk ** j) * (dxk ** k)

    bnd = {name: int(np.searchsorted(rk, R, side="right"))
           for name, R in (("inner", h1i.R_INNER), ("s1", 1.5), ("s2", 3.0),
                           ("s3", 4.5), ("core", h1i.R_CORE), ("mid", h1i.R_MID))}
    return dict(HZ=hz, HY=hy, HX=hx, DZK=dzk, DYK=dyk, DXK=dxk, RK=rk,
                VOX_OFF=vox_off, MOMENT_MAT=np.ascontiguousarray(mat), BND=bnd,
                K=int(rk.size), n_core_vox=float(bnd["core"]))


def install_kernel(vox: np.ndarray) -> dict:
    """Point h1i_node_appearance's module globals at a kernel for `vox`, so that
    `h1i._node_features` computes the same PHYSICAL statistics on any imagery.
    Process-local; the file on disk is never modified."""
    k = build_kernel(vox)
    h1i.VOX = np.asarray(vox, dtype=np.float64)
    h1i.HZ, h1i.HY, h1i.HX = k["HZ"], k["HY"], k["HX"]
    h1i.DZK, h1i.DYK, h1i.DXK, h1i.RK = k["DZK"], k["DYK"], k["DXK"], k["RK"]
    h1i.VOX_OFF, h1i.MOMENT_MAT, h1i.BND = k["VOX_OFF"], k["MOMENT_MAT"], k["BND"]
    h1i.K = k["K"]
    return k


# ------------------------------------------------- per-node derived (scale free)
RATIO_FEATS = ["massn", "conc", "contrast", "saddle_self", "peakn"]
DIFF_FEATS = ["rg", "spher", "aniso", "linearity", "planarity", "detn",
              "kurt", "skew", "coff"]
LEVEL_SHAPE = ["conc", "contrast", "saddle_self", "spher", "aniso", "linearity",
               "planarity", "detn", "kurt", "skew", "p10", "p20", "p30", "rg", "coff"]
LEVEL_INTENSITY = ["massn", "peakn"]          # dynamic-range normalised -> ablatable
NODE_FEATS = sorted(set(RATIO_FEATS + DIFF_FEATS + LEVEL_SHAPE + LEVEL_INTENSITY))


def node_derived(nf: pd.DataFrame, n_core_vox: float) -> pd.DataFrame:
    """Dimensionless / self-normalising per-node descriptors.

    Identical in definition to h1i_candidate_eval.per_node_derived, except that the
    core-voxel count is taken from the ACTIVE kernel instead of being hard-coded to
    the competition value (419), so `massn` is the same physical quantity -- mean
    background-subtracted core intensity as a fraction of the frame dynamic range --
    at either voxel spacing.
    """
    d = pd.DataFrame(index=nf.index)
    dyn = np.maximum(nf.hi - nf.bg, 1.0)
    l1 = np.maximum(nf.lam1, EPS)
    tr = np.maximum(nf.lam1 + nf.lam2 + nf.lam3, EPS)
    d["massn"] = nf.mass_core / (dyn * n_core_vox)
    d["conc"] = nf.mass_core / np.maximum(nf.mass_outer, EPS)
    d["contrast"] = nf.peak_inner / np.maximum(nf.shell3, EPS)
    d["saddle_self"] = nf.peak_inner / np.maximum(nf.peak_mid, EPS)
    d["rg"] = np.sqrt(tr)
    d["spher"] = nf.lam3 / l1
    d["aniso"] = (nf.lam1 - nf.lam3) / tr
    d["linearity"] = (nf.lam1 - nf.lam2) / l1
    d["planarity"] = (nf.lam2 - nf.lam3) / l1
    d["detn"] = np.cbrt(np.maximum(nf.lam1 * nf.lam2 * nf.lam3, 0.0)) / (tr / 3.0)
    d["kurt"] = nf.axial_kurt
    d["skew"] = np.abs(nf.axial_skew)
    d["p10"] = nf.shell1 / np.maximum(nf.shell0, EPS)
    d["p20"] = nf.shell2 / np.maximum(nf.shell0, EPS)
    d["p30"] = nf.shell3 / np.maximum(nf.shell0, EPS)
    d["coff"] = nf.centroid_off_um
    d["peakn"] = nf.peak_inner / dyn
    return d


# --------------------------------------------------- mother-event feature vector
def _names() -> tuple[list[str], dict[str, list[str]]]:
    lvl = [f"L_{f}" for f in LEVEL_SHAPE]
    lvl_i = [f"L_{f}" for f in LEVEL_INTENSITY]
    ratio, diff, peak = [], [], []
    for f in RATIO_FEATS:
        ratio += [f"R_{f}_p1", f"R_{f}_p2", f"R_{f}_m1", f"R_{f}_m2"]
    for f in DIFF_FEATS:
        diff += [f"D_{f}_p1", f"D_{f}_p2", f"D_{f}_m1", f"D_{f}_m2"]
    for f in ("conc", "massn", "spher", "kurt"):
        peak.append(f"P_{f}")
    groups = {"level_shape": lvl, "level_intensity": lvl_i, "log_ratio": ratio,
              "delta": diff, "peakedness": peak}
    return lvl + lvl_i + ratio + diff + peak, groups


H1M_FEATURES, H1M_GROUPS = _names()
H1M_NO_INTENSITY = [c for c in H1M_FEATURES if c not in H1M_GROUPS["level_intensity"]]


def mother_event_features(by_dt: dict[int, pd.DataFrame], index) -> pd.DataFrame:
    """Assemble the frozen mother-centred five-frame feature block.

    `by_dt[dt]` is a per-node derived frame already aligned to `index` (one row per
    mother event, NaN where that frame does not exist). Every column is a level of a
    scale-free shape statistic, a log-ratio across time, a difference across time, or
    a three-point peakedness -- nothing raw.
    """
    out = pd.DataFrame(index=index)
    z = by_dt[0]
    for f in LEVEL_SHAPE + LEVEL_INTENSITY:
        out[f"L_{f}"] = z[f].to_numpy(dtype=float)

    def lr(a, b):
        a = np.maximum(np.asarray(a, dtype=float), EPS)
        b = np.maximum(np.asarray(b, dtype=float), EPS)
        return np.log(a) - np.log(b)

    for f in RATIO_FEATS:
        out[f"R_{f}_p1"] = lr(by_dt[1][f], z[f])
        out[f"R_{f}_p2"] = lr(by_dt[2][f], z[f])
        out[f"R_{f}_m1"] = lr(z[f], by_dt[-1][f])
        out[f"R_{f}_m2"] = lr(z[f], by_dt[-2][f])
    for f in DIFF_FEATS:
        out[f"D_{f}_p1"] = by_dt[1][f].to_numpy(float) - z[f].to_numpy(float)
        out[f"D_{f}_p2"] = by_dt[2][f].to_numpy(float) - z[f].to_numpy(float)
        out[f"D_{f}_m1"] = z[f].to_numpy(float) - by_dt[-1][f].to_numpy(float)
        out[f"D_{f}_m2"] = z[f].to_numpy(float) - by_dt[-2][f].to_numpy(float)
    out["P_conc"] = (2 * z["conc"].to_numpy(float) - by_dt[-1]["conc"].to_numpy(float)
                     - by_dt[1]["conc"].to_numpy(float))
    out["P_massn"] = (2 * np.log(np.maximum(z["massn"].to_numpy(float), EPS))
                      - np.log(np.maximum(by_dt[-1]["massn"].to_numpy(float), EPS))
                      - np.log(np.maximum(by_dt[1]["massn"].to_numpy(float), EPS)))
    out["P_spher"] = (2 * z["spher"].to_numpy(float) - by_dt[-1]["spher"].to_numpy(float)
                      - by_dt[1]["spher"].to_numpy(float))
    out["P_kurt"] = (2 * z["kurt"].to_numpy(float) - by_dt[-1]["kurt"].to_numpy(float)
                     - by_dt[1]["kurt"].to_numpy(float))
    return out[H1M_FEATURES]


def config_hash() -> str:
    """Stable hash of the frozen representation (feature names + radii + kernel rule)."""
    import hashlib
    import json
    payload = {"features": H1M_FEATURES, "dts": list(DTS),
               "radii": [h1i.R_INNER, h1i.R_CORE, h1i.R_MID, h1i.R_OUTER],
               "unit": "mother_time_event", "version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
