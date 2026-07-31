"""H4-S alternative lane: is SYNTHETIC mitosis usable as a training signal?

This is the CHEAP KILL TEST that must pass before any model is trained on
synthetic splits.  If synthetic split patches are trivially separable from real
imagery, a model trained on them learns the synthesis artefact and the lane is
dead -- no amount of downstream tuning recovers it.

Synthesis (physically constrained, three constraints, all measured not guessed)
------------------------------------------------------------------------------
1. DAUGHTER SEPARATION.  d = 10.3 um at t+1, the measured competition value
   (H1M_FINDINGS 3.2.1), NOT the 5.7 um Zebrahub value.  Parent-to-midpoint
   offset 3.9 um, also measured.
2. MASS CONSERVATION.  The nucleus field is split into two copies at amplitude
   0.5 each, so total background-subtracted mass is preserved exactly.
3. POINT SPREAD FUNCTION.  No PSF is modelled, and none needs to be.  The source
   patch is ALREADY convolved with the microscope PSF, and convolution commutes
   with translation for a shift-invariant PSF:
        h * (f shifted)  ==  (h * f) shifted
   so translating the real, already-convolved nucleus renders two shifted copies
   of the underlying object EXACTLY right.  This is the one place where synthesis
   is provably faithful, and it is why the test below isolates the parts that are
   not (frozen neighbourhood, no chromatin condensation, no membrane).

Four populations, all scored in the SAME frozen 17-descriptor space and the same
t -> t+1 ratio space that H1-M uses:

    A  real_normal   real t+1 patch of a non-dividing mother
    B  synth_normal  frame-t nucleus translated by that mother's own measured
                     one-step speed; everything else frozen
    C  synth_split   frame-t nucleus split into two half-mass copies at 10.3 um
    D  real_div      real t+1 patch of a TRUE dividing mother

Three tests, in the order that decides the lane:

    T1  ARTEFACT     AUC(A vs B).  Near 0.5 => synthesis is faithful.  Near 1.0
                     => the pipeline stamps a signature on every synthetic patch
                     and any B-vs-C classifier will read that signature instead
                     of mitosis.  THIS IS THE KILL TEST.
    T2  REALISM      AUC(C vs D).  Do synthetic splits sit where real divisions
                     sit?  Near 0.5 is good.
    T3  TRANSFER     Train on pure synthetic (B vs C), evaluate on pure real
                     (A vs D).  This is the only number that says whether
                     synthetic supervision would have worked.  Reference arm:
                     the same classifier trained on real (A vs D) with
                     crop-grouped CV.

Usage:
  .venv\\Scripts\\python.exe scripts\\h4_synth_overlap.py \\
      --comp <comp_mother_events.parquet> --out <synth.json> \\
      --neg-per-frame 40 [--max-crops N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import shift as nd_shift

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1i_node_appearance as h1i  # noqa: E402
import h1m_features as F  # noqa: E402
import h1m_gate as G  # noqa: E402
import h4_ssl_residual as S  # noqa: E402

ROOT = Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")

# MEASURED competition mitotic geometry (H1M_FINDINGS section 3.2.1).
DAUGHTER_SEP_UM = 10.3
PARENT_MIDPOINT_UM = 3.9
R_NUC_UM = 4.5              # radius of the field treated as "the nucleus" and moved

# translation margin, in voxels, for the largest displacement we apply
MARG = np.ceil((PARENT_MIDPOINT_UM + DAUGHTER_SEP_UM / 2 + 1.0) / h1i.VOX).astype(int)


def unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def box_of(vol: np.ndarray, iz: int, iy: int, ix: int) -> np.ndarray:
    """Padded cuboid around a node, large enough to translate inside."""
    hz, hy, hx = h1i.HZ + MARG[0], h1i.HY + MARG[1], h1i.HX + MARG[2]
    Z, Y, X = vol.shape
    z0, z1 = iz - hz, iz + hz + 1
    y0, y1 = iy - hy, iy + hy + 1
    x0, x1 = ix - hx, ix + hx + 1
    pz = (max(0, -z0), max(0, z1 - Z))
    py = (max(0, -y0), max(0, y1 - Y))
    px = (max(0, -x0), max(0, x1 - X))
    sub = vol[max(0, z0):min(Z, z1), max(0, y0):min(Y, y1), max(0, x0):min(X, x1)]
    if any(sum(p) for p in (pz, py, px)):
        sub = np.pad(sub, (pz, py, px), mode="edge")
    return sub.astype(np.float32)


def _sphere_mask(shape: tuple) -> np.ndarray:
    cz, cy, cx = [s // 2 for s in shape]
    zz = (np.arange(shape[0]) - cz) * h1i.VOX[0]
    yy = (np.arange(shape[1]) - cy) * h1i.VOX[1]
    xx = (np.arange(shape[2]) - cx) * h1i.VOX[2]
    r = np.sqrt(zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2)
    return (r <= R_NUC_UM).astype(np.float32)


def centre_patch(box: np.ndarray) -> np.ndarray:
    """Pull the frozen r<=6um radius-sorted sample vector out of a synthesised box,
    centred on the ORIGINAL node position -- exactly how h1i samples a real patch."""
    cz, cy, cx = [s // 2 for s in box.shape]
    idx = (VOFF[:, 0] + cz, VOFF[:, 1] + cy, VOFF[:, 2] + cx)
    return box[idx]


VOFF = h1i.VOX_OFF.astype(np.int64)


def synthesise(box: np.ndarray, mask: np.ndarray, mode: str,
               rng: np.random.Generator, speed_um: float) -> np.ndarray:
    """Return the r<=6um sample vector of a synthetic t+1 patch."""
    nuc = box * mask
    rest = box - nuc
    if mode == "normal":
        # speed_um is NaN for a node with no predecessor (track age 0); such a node
        # has no measured one-step displacement, so it is translated by zero.
        sp = float(speed_um)
        sp = float(np.clip(sp, 0.0, 6.0)) if np.isfinite(sp) else 0.0
        v = unit(rng, 1)[0] * sp
        out = rest + nd_shift(nuc, v / h1i.VOX, order=1, mode="nearest")
    else:                                                  # "split"
        mid = unit(rng, 1)[0] * PARENT_MIDPOINT_UM
        ax = unit(rng, 1)[0] * (DAUGHTER_SEP_UM / 2.0)
        a = nd_shift(nuc, (mid + ax) / h1i.VOX, order=1, mode="nearest")
        b = nd_shift(nuc, (mid - ax) / h1i.VOX, order=1, mode="nearest")
        out = rest + 0.5 * a + 0.5 * b                     # mass conserved exactly
    return centre_patch(out)


def descriptors(samples: np.ndarray, bg: float, hi: float) -> pd.DataFrame:
    """Frozen h1i kernel maths -> the 17 self-normalising descriptors."""
    f = h1i._node_features(samples, bg)
    nf = pd.DataFrame({k: np.asarray(v, float) for k, v in f.items()})
    nf["bg"], nf["hi"] = bg, hi
    return F.node_derived(nf, float(h1i.BND["core"]))


def collect(comp: pd.DataFrame, neg_per_frame: int, max_crops: int | None,
            seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = comp[comp.n_frames_seen == 5]
    div = comp[comp.label == 1]
    crops = (div.groupby(["fold", "crop"]).size().sort_values(ascending=False)
             .reset_index()[["fold", "crop"]].values.tolist())
    if max_crops:
        crops = crops[:max_crops]

    rows = []
    for fold, crop in crops:
        arr = zarr.open_group(str(ROOT / f"data/train/{crop}.zarr"), mode="r")["0"]
        T = arr.shape[0]
        sub = comp[comp.crop == crop]
        d = sub[sub.label == 1]
        for t in sorted(d.t.unique()):
            if t + 1 >= T:
                continue
            pos = d[d.t == t]
            negp = sub[(sub.t == t) & (sub.label == 0)]
            if len(negp) > neg_per_frame:
                negp = negp.iloc[rng.choice(len(negp), neg_per_frame, replace=False)]
            ev = pd.concat([pos, negp])
            f0 = np.asarray(arr[int(t)])
            f1 = np.asarray(arr[int(t) + 1])
            s0 = f0[::2, ::3, ::3]
            bg0, hi0 = float(np.percentile(s0, 20.0)), float(np.percentile(s0, 99.5))
            s1 = f1[::2, ::3, ::3]
            bg1, hi1 = float(np.percentile(s1, 20.0)), float(np.percentile(s1, 99.5))

            mask = None
            for _, e in ev.iterrows():
                iz = int(np.clip(round(e.z_vox), 0, f0.shape[0] - 1))
                iy = int(np.clip(round(e.y_vox), 0, f0.shape[1] - 1))
                ix = int(np.clip(round(e.x_vox), 0, f0.shape[2] - 1))
                b0 = box_of(f0, iz, iy, ix)
                if mask is None or mask.shape != b0.shape:
                    mask = _sphere_mask(b0.shape)
                nb = float(np.maximum(bg0, 0.0))
                b0c = np.maximum(b0 - nb, 0.0)             # split the OBJECT, not the bg
                base = nb

                real_t = centre_patch(b0)
                real_t1 = centre_patch(box_of(f1, iz, iy, ix))
                syn_n = base + synthesise(b0c, mask, "normal", rng, float(e.speed_um))
                syn_s = base + synthesise(b0c, mask, "split", rng, float(e.speed_um))

                d0 = descriptors(real_t[None, :], bg0, hi0).iloc[0]
                for cls, samp, bgu, hiu in (
                        ("real_normal" if e.label == 0 else "real_div", real_t1, bg1, hi1),
                        ("synth_normal", syn_n, bg0, hi0),
                        ("synth_split", syn_s, bg0, hi0)):
                    d1 = descriptors(samp[None, :], bgu, hiu).iloc[0]
                    r = {"cls": cls, "crop": crop, "family": e.family, "t": int(t),
                         "mother": int(e.mother), "label": int(e.label),
                         "realisable": bool(e.realisable)}
                    for c in S.DESC:
                        if c in S.LOG_DESC:
                            r[f"R_{c}_p1"] = float(np.log(max(d1[c], S.EPS))
                                                   - np.log(max(d0[c], S.EPS)))
                        else:
                            r[f"D_{c}_p1"] = float(d1[c] - d0[c])
                        r[f"L_{c}"] = float(d1[c])
                    rows.append(r)
    return pd.DataFrame(rows)


P1 = [f"R_{c}_p1" if c in S.LOG_DESC else f"D_{c}_p1" for c in S.DESC]
LV = [f"L_{c}" for c in S.DESC]


def _fit_eval(tr: pd.DataFrame, te: pd.DataFrame, ytr, yte, feats) -> float:
    st = G.Standardiser(tr[feats].to_numpy(float))
    w = G.fit_logistic(st(tr[feats].to_numpy(float)), np.asarray(ytr, float), l2=10.0)
    s = G.score(w, st(te[feats].to_numpy(float)))
    return G.auc(s[np.asarray(yte) == 1], s[np.asarray(yte) == 0])


def pairwise(df: pd.DataFrame, a: str, b: str, feats) -> dict:
    """Univariate best-feature AUC and a multivariate crop-grouped CV AUC."""
    A, B = df[df.cls == a], df[df.cls == b]
    best, per = 0.5, {}
    for c in feats:
        u = G.auc(A[c].to_numpy(float), B[c].to_numpy(float))
        per[c] = u
        if abs(u - 0.5) > abs(best - 0.5):
            best = u
    d = pd.concat([A, B])
    y = (d.cls == a).astype(int).to_numpy()
    crops = d.crop.to_numpy()
    uc = np.unique(crops)
    rng = np.random.default_rng(0)
    fold = {c: i for c, i in zip(uc, rng.permutation(len(uc)) % 5)}
    fv = np.array([fold[c] for c in crops])
    oof = np.full(len(d), np.nan)
    for k in range(5):
        tr, temask = d[fv != k], fv == k
        if temask.sum() == 0 or len(np.unique(y[fv != k])) < 2:
            continue
        st = G.Standardiser(tr[feats].to_numpy(float))
        w = G.fit_logistic(st(tr[feats].to_numpy(float)),
                           y[fv != k].astype(float), l2=10.0)
        oof[temask] = G.score(w, st(d[temask][feats].to_numpy(float)))
    ok = np.isfinite(oof)
    return {"n_a": int(len(A)), "n_b": int(len(B)),
            "best_single_feature": max(per, key=lambda c: abs(per[c] - 0.5)),
            "best_single_auc": best,
            "multivariate_cv_auc": G.auc(oof[ok & (y == 1)], oof[ok & (y == 0)]),
            "top5": dict(sorted(per.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5])}


def smd(df: pd.DataFrame, a: str, b: str, feats) -> dict:
    A, B = df[df.cls == a][feats].to_numpy(float), df[df.cls == b][feats].to_numpy(float)
    sd = np.sqrt((A.var(0) + B.var(0)) / 2.0)
    v = np.abs(A.mean(0) - B.mean(0)) / np.maximum(sd, 1e-9)
    return {"mean_abs_smd": float(np.nanmean(v)), "median_abs_smd": float(np.nanmedian(v)),
            "max_abs_smd": float(np.nanmax(v)),
            "n_features_over_1sd": int(np.nansum(v > 1.0)), "n_features": len(feats)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--neg-per-frame", type=int, default=40)
    ap.add_argument("--max-crops", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    comp = pd.read_parquet(a.comp)
    df = collect(comp, a.neg_per_frame, a.max_crops, a.seed)
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.dump, compression="zstd", index=False)

    feats = P1 + LV
    res = {"config": {"daughter_sep_um": DAUGHTER_SEP_UM,
                      "parent_midpoint_um": PARENT_MIDPOINT_UM,
                      "r_nucleus_um": R_NUC_UM, "mass_conserved": True,
                      "psf": "implicit: translation of an already-convolved field",
                      "neg_per_frame": a.neg_per_frame, "seed": a.seed,
                      "n_features": len(feats), "cfg_hash": S.cfg_hash()},
           "counts": df.cls.value_counts().to_dict(),
           "n_crops": int(df.crop.nunique()), "n_frames": int(df.groupby(["crop", "t"]).ngroups)}

    # ---- T1 KILL TEST: is synthetic trivially separable from real? ----
    res["T1_artefact_real_vs_synth_normal"] = pairwise(df, "real_normal", "synth_normal", feats)
    res["T1_smd"] = smd(df, "real_normal", "synth_normal", feats)
    res["T1b_realdiv_vs_synthsplit"] = pairwise(df, "real_div", "synth_split", feats)

    # ---- T2 REALISM ----
    res["T2_realism"] = res["T1b_realdiv_vs_synthsplit"]

    # ---- signal presence inside each domain ----
    res["signal_real"] = pairwise(df, "real_div", "real_normal", feats)
    res["signal_synth"] = pairwise(df, "synth_split", "synth_normal", feats)

    # ---- T3 DECISIVE: train pure synthetic -> evaluate pure real ----
    syn = df[df.cls.isin(["synth_normal", "synth_split"])]
    real = df[df.cls.isin(["real_normal", "real_div"])]
    ysyn = (syn.cls == "synth_split").astype(int).to_numpy()
    yreal = (real.cls == "real_div").astype(int).to_numpy()
    res["T3_synth_to_real"] = {
        "n_train": int(len(syn)), "n_test": int(len(real)),
        "n_test_pos": int(yreal.sum()),
        "auc_p1_only": _fit_eval(syn, real, ysyn, yreal, P1),
        "auc_p1_plus_levels": _fit_eval(syn, real, ysyn, yreal, feats),
    }
    # per-family transfer
    res["T3_synth_to_real_by_family"] = {}
    for fam in ("44b6", "6bba"):
        rf = real[real.family == fam]
        if (rf.cls == "real_div").sum() < 3:
            continue
        res["T3_synth_to_real_by_family"][fam] = {
            "n_pos": int((rf.cls == "real_div").sum()),
            "auc_p1_only": _fit_eval(syn, rf, ysyn,
                                     (rf.cls == "real_div").astype(int).to_numpy(), P1)}
    # reference arm: real-trained, crop-grouped CV
    res["T3_reference_real_trained_cv"] = res["signal_real"]["multivariate_cv_auc"]

    # ---- sign agreement of the mitotic signature, real vs synthetic ----
    sign = {}
    for c in P1:
        ur = G.auc(df[df.cls == "real_div"][c].to_numpy(float),
                   df[df.cls == "real_normal"][c].to_numpy(float))
        us = G.auc(df[df.cls == "synth_split"][c].to_numpy(float),
                   df[df.cls == "synth_normal"][c].to_numpy(float))
        sign[c] = {"auc_real": ur, "auc_synth": us,
                   "sign_agrees": bool(np.sign(ur - .5) == np.sign(us - .5))}
    res["signature_sign_agreement"] = sign
    res["n_sign_agree"] = int(sum(v["sign_agrees"] for v in sign.values()))
    res["n_p1_features"] = len(P1)
    res["seconds"] = round(time.time() - t0, 1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items()
                      if k != "signature_sign_agreement"}, indent=2, default=float))


if __name__ == "__main__":
    main()
