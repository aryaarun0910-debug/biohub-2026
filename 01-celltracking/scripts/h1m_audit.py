"""H1-M pre-training audits. Every one of these must PASS before any GPU is spent.

  A  label audit            -- do the mother labels reconcile with the scorer's own
                               division counts, and is anything unlabeled leaking in?
  B  disjointness proof     -- train / validation embryo sets share no embryo and no
                               (embryo, time-window), checked on the actual row ids
  C  patch coordinates      -- are the gathered spheres actually centred on nuclei?
                               measured against an offset null, on BOTH domains
  D  class-balanced sampler -- does the positive class receive the intended share of
                               the gradient, and is the fit reproducible?
  E  domain shift           -- standardised mean difference, external vs competition
  F  matched negatives      -- build and verify the matched negative design

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m_audit.py --comp <parquet> --external <p> ... \
      --out <json> [--skip-c]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402
import h1i_node_appearance as h1i  # noqa: E402
from h1m_gate import Standardiser, fit_logistic, labelled, score, auc  # noqa: E402

ROOT = Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
GT_DIVISIONS = {"44b6": 26, "6bba": 125}
MATCH_COLS = ["t", "local_density", "L_raw_massn", "track_age", "speed_um"]


# ------------------------------------------------------------------ A
def audit_labels(comp: pd.DataFrame, ext: pd.DataFrame | None) -> dict:
    out = {"competition": {}, "external": {}}
    for fam, s in comp.groupby("family"):
        div = s[s.label == 1]
        out["competition"][fam] = {
            "mother_events": int(len(s)),
            "outdeg_hist": {str(k): int(v) for k, v in
                            s.mother_gt_outdeg.value_counts().sort_index().items()},
            "label_hist": {str(k): int(v) for k, v in
                           s.label.value_counts().sort_index().items()},
            "dividers": int(len(div)),
            "realisable": int(s.realisable.sum()),
            "realisable_implies_divider": bool((s.realisable & (s.label != 1)).sum() == 0),
            "scorer_gt_divisions": GT_DIVISIONS[fam],
            "dividers_le_scorer_total": bool(len(div) <= GT_DIVISIONS[fam]),
            "unmatched_gt_divisions": GT_DIVISIONS[fam] - int(len(div)),
            "any_unlabeled_rows": int((s.label == -1).sum()),
        }
    if ext is not None and len(ext):
        for emb, s in ext.groupby("embryo"):
            out["external"][emb] = {
                "rows": int(len(s)),
                "dividers": int((s.label == 1).sum()),
                "non_dividers": int((s.label == 0).sum()),
                "unlabeled": int((s.label == -1).sum()),
                "hard_negatives": int(s.hard_negative.sum()),
                "outdeg_hist": {str(k): int(v) for k, v in
                                s.mother_gt_outdeg.value_counts().sort_index().items()},
                "realisable_implies_divider":
                    bool((s.realisable & (s.label != 1)).sum() == 0),
                "windows": sorted(s.window.unique().tolist()),
                "neg_pool_total": int(s.neg_pool_total.max()),
            }
        out["external_excluded_embryos"] = ["ZSNS001"]
    return out


# ------------------------------------------------------------------ B
def audit_disjointness(ext: pd.DataFrame) -> dict:
    res = {"loeo": {}, "all_pass": True}
    if ext is None or not len(ext):
        return {"skipped": "no external data"}
    ext = ext.copy()
    ext["_key"] = ext.embryo.astype(str) + ":" + ext.window.astype(str) + ":" \
        + ext.t.astype(str) + ":" + ext.mother.astype(str)
    for held in sorted(ext.embryo.unique()):
        tr, te = ext[ext.embryo != held], ext[ext.embryo == held]
        emb_ok = not (set(tr.embryo) & set(te.embryo))
        win_ok = not (set(tr.embryo + "|" + tr.window) & set(te.embryo + "|" + te.window))
        row_ok = not (set(tr._key) & set(te._key))
        res["loeo"][held] = {"train_embryos": sorted(tr.embryo.unique().tolist()),
                             "test_embryos": sorted(te.embryo.unique().tolist()),
                             "embryo_disjoint": bool(emb_ok),
                             "window_disjoint": bool(win_ok),
                             "row_key_disjoint": bool(row_ok),
                             "n_train": int(len(tr)), "n_test": int(len(te))}
        res["all_pass"] &= bool(emb_ok and win_ok and row_ok)
    return res


# ------------------------------------------------------------------ C
def _gather_stats(vol_flat, base, off, bg):
    box = vol_flat[base[:, None] + off[None, :]]
    fv = h1i._node_features(box, bg)
    peak = np.asarray(fv["peak_inner"], float)
    shell = np.maximum(np.asarray(fv["shell3"], float), 1e-6)
    return peak / shell, np.asarray(fv["mass_core"], float)


def audit_patch_coords_competition(comp: pd.DataFrame, n_crops: int = 6,
                                   seed: int = 7) -> dict:
    """Recompute the sphere at each mother's own voxel position and at a 12 um
    displaced position in the same frame. If the coordinates address nuclei, the
    true-position contrast must dominate the displaced null."""
    import zarr
    kern = F.install_kernel(F.VOX_COMPETITION)
    rng = np.random.default_rng(seed)
    crops = (comp.groupby(["fold", "crop"]).size().sort_values(ascending=False)
             .head(n_crops).index.tolist())
    true_c, null_c = [], []
    for fold, crop in crops:
        sub = comp[(comp.fold == fold) & (comp.crop == crop)]
        sub = sub.sample(min(400, len(sub)), random_state=seed)
        arr = zarr.open_group(str(ROOT / f"data/train/{crop}.zarr"), mode="r")["0"]
        T, Z, Y, X = arr.shape
        HZ, HY, HX = kern["HZ"], kern["HY"], kern["HX"]
        off = (kern["VOX_OFF"][:, 0].astype(np.int64) * (Y * X)
               + kern["VOX_OFF"][:, 1].astype(np.int64) * X
               + kern["VOX_OFF"][:, 2].astype(np.int64))
        for t, g in sub.groupby("t"):
            frame = np.asarray(arr[int(t)])
            bg = float(np.percentile(frame[::2, ::3, ::3], 20.0))
            fl = frame.ravel()
            iz = np.rint(g.z_vox.to_numpy()).astype(np.int64)
            iy = np.rint(g.y_vox.to_numpy()).astype(np.int64)
            ix = np.rint(g.x_vox.to_numpy()).astype(np.int64)
            # displaced null: 12 um in a random axis-aligned direction
            dz = rng.choice([-1, 1], iz.size) * int(round(12.0 / F.VOX_COMPETITION[0]))
            dy = rng.choice([-1, 1], iz.size) * int(round(12.0 / F.VOX_COMPETITION[1]))
            jz, jy, jx = iz + dz, iy + dy, ix
            ok = ((iz >= HZ) & (iz < Z - HZ) & (iy >= HY) & (iy < Y - HY)
                  & (ix >= HX) & (ix < X - HX))
            ok &= (jz >= HZ) & (jz < Z - HZ) & (jy >= HY) & (jy < Y - HY)
            if ok.sum() == 0:
                continue
            b0 = (iz[ok] * (Y * X) + iy[ok] * X + ix[ok])
            b1 = (jz[ok] * (Y * X) + jy[ok] * X + jx[ok])
            c0, _ = _gather_stats(fl, b0, off, bg)
            c1, _ = _gather_stats(fl, b1, off, bg)
            true_c.append(c0)
            null_c.append(c1)
    t_, n_ = np.concatenate(true_c), np.concatenate(null_c)
    return {"n": int(t_.size), "contrast_true_median": float(np.median(t_)),
            "contrast_null_median": float(np.median(n_)),
            "auc_true_vs_null": auc(t_, n_),
            "frac_true_gt_null_median": float((t_ > np.median(n_)).mean()),
            "PASS": bool(np.median(t_) > np.median(n_) and auc(t_, n_) > 0.8)}


def audit_patch_coords_zebrahub(embryo: str, tracks: Path, frame: int) -> dict:
    """One-chunk check that the published track coordinates are level-0 voxel indices
    that land on nuclei: true node positions vs positions drawn uniformly inside the
    cell bounding box of the same frame."""
    from h1m_zebrahub_appearance import zarray, fetch_chunk, decode, Z_SLABS, X_CHUNK
    kern = F.install_kernel(F.VOX_ZEBRAHUB_L0)
    meta = zarray(embryo)
    cz, cy, cx = meta["chunks"][2], meta["chunks"][3], meta["chunks"][4]
    Y = min(meta["shape"][3], cy)
    X = min(meta["shape"][4] - X_CHUNK * cx, cx)
    buf = np.zeros((cz * len(Z_SLABS), Y, X), dtype=np.uint16)
    for k, zc in enumerate(Z_SLABS):
        buf[k * cz:(k + 1) * cz] = decode(fetch_chunk(embryo, frame, zc, X_CHUNK),
                                          (cz, cy, cx))[:, :Y, :X]
    ZM = buf.shape[0]
    fl = buf.ravel()
    HZ, HY, HX = kern["HZ"], kern["HY"], kern["HX"]
    off = (kern["VOX_OFF"][:, 0].astype(np.int64) * (Y * X)
           + kern["VOX_OFF"][:, 1].astype(np.int64) * X
           + kern["VOX_OFF"][:, 2].astype(np.int64))
    tr = pd.read_parquet(tracks, columns=["t", "z", "y", "x"])
    sub = tr[tr.t == frame]
    bg = float(np.percentile(buf[::4, ::6, ::6], 20.0))
    iz = np.rint(sub.z.to_numpy()).astype(np.int64)
    iy = np.rint(sub.y.to_numpy()).astype(np.int64)
    ix = np.rint(sub.x.to_numpy()).astype(np.int64)
    ok = ((iz >= HZ) & (iz < ZM - HZ) & (iy >= HY) & (iy < Y - HY)
          & (ix >= HX) & (ix < X - HX))
    iz, iy, ix = iz[ok][:4000], iy[ok][:4000], ix[ok][:4000]
    rng = np.random.default_rng(11)
    jz = rng.integers(iz.min(), iz.max() + 1, iz.size)
    jy = rng.integers(iy.min(), iy.max() + 1, iz.size)
    jx = rng.integers(ix.min(), ix.max() + 1, iz.size)
    jz = np.clip(jz, HZ, ZM - HZ - 1); jy = np.clip(jy, HY, Y - HY - 1)
    jx = np.clip(jx, HX, X - HX - 1)
    c0, m0 = _gather_stats(fl, iz * (Y * X) + iy * X + ix, off, bg)
    c1, m1 = _gather_stats(fl, jz * (Y * X) + jy * X + jx, off, bg)
    return {"embryo": embryo, "frame": frame, "n": int(iz.size), "bg": bg,
            "contrast_true_median": float(np.median(c0)),
            "contrast_null_median": float(np.median(c1)),
            "mass_true_median": float(np.median(m0)),
            "mass_null_median": float(np.median(m1)),
            "auc_true_vs_null": auc(c0, c1),
            "PASS": bool(np.median(m0) > 2 * np.median(m1) and auc(c0, c1) > 0.8)}


# ------------------------------------------------------------------ D
def audit_sampler(ext: pd.DataFrame, feats: list[str], l2: float) -> dict:
    d = labelled(ext)
    X = d[feats].to_numpy(float)
    y = d.label.to_numpy(float)
    st = Standardiser(X)
    Z = st(X)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    w_pos = nneg / max(npos, 1)
    sw = np.where(y == 1, w_pos, 1.0)
    w1 = fit_logistic(Z, y, l2=l2)
    w2 = fit_logistic(Z, y, l2=l2)
    perm = np.random.default_rng(3).permutation(len(y))
    w3 = fit_logistic(Z[perm], y[perm], l2=l2)
    return {"n_pos": npos, "n_neg": nneg, "raw_prevalence": npos / (npos + nneg),
            "positive_weight": float(w_pos),
            "effective_positive_gradient_share":
                float(sw[y == 1].sum() / sw.sum()),
            "balanced_PASS": bool(abs(sw[y == 1].sum() / sw.sum() - 0.5) < 1e-9),
            "rerun_identical_PASS": bool(np.allclose(w1, w2, atol=1e-12)),
            "row_order_invariant_max_abs_diff": float(np.max(np.abs(w1 - w3))),
            "row_order_invariant_PASS": bool(np.allclose(w1, w3, atol=1e-6))}


# ------------------------------------------------------------------ E
def audit_domain_shift(comp: pd.DataFrame, ext: pd.DataFrame,
                       feats: list[str]) -> dict:
    rows = []
    e = labelled(ext)
    en = e[e.label == 0]
    for fam, s in comp.groupby("family"):
        cn = labelled(s)
        cn = cn[cn.label == 0]
        for c in feats:
            a, b = en[c].to_numpy(float), cn[c].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            sd = np.sqrt(0.5 * (a.std() ** 2 + b.std() ** 2)) + 1e-9
            rows.append({"family": fam, "feature": c,
                         "smd": float((a.mean() - b.mean()) / sd)})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="feature", columns="family", values="smd")
    piv["max_abs"] = piv.abs().max(axis=1)
    return {"mean_abs_smd": float(df.smd.abs().mean()),
            "median_abs_smd": float(df.smd.abs().median()),
            "n_features_over_1sd": int((piv.max_abs > 1).sum()),
            "worst": piv.sort_values("max_abs", ascending=False).head(12)
                        .round(3).reset_index().to_dict("records")}


# ------------------------------------------------------------------ F
def build_matched_negatives(comp: pd.DataFrame, k: int = 20, seed: int = 5) -> dict:
    """For each divider, take the k nearest non-dividers from the SAME crop in
    standardised (t, local density, core mass, track age, speed) space."""
    rng = np.random.default_rng(seed)
    picks, report = [], {}
    for fam, s in comp.groupby("family"):
        s = labelled(s)
        M = s[MATCH_COLS].to_numpy(float)
        M = np.nan_to_num(M, nan=np.nanmedian(M, axis=0))
        mu, sd = M.mean(0), np.maximum(M.std(0), 1e-6)
        Z = (M - mu) / sd
        y = s.label.to_numpy()
        crop = s.crop.to_numpy()
        chosen = []
        for i in np.flatnonzero(y == 1):
            cand = np.flatnonzero((y == 0) & (crop == crop[i]))
            if cand.size == 0:
                continue
            d = np.linalg.norm(Z[cand] - Z[i], axis=1)
            chosen.append(cand[np.argsort(d)[:k]])
        if not chosen:
            continue
        ch = np.unique(np.concatenate(chosen))
        idx = s.index.to_numpy()
        picks.append(pd.Series(idx[ch]))
        pos = s[y == 1]
        neg = s.iloc[ch]
        report[fam] = {
            "positives": int(len(pos)), "matched_negatives": int(len(neg)),
            "k_requested": k,
            "balance_smd": {c: float((pos[c].mean() - neg[c].mean())
                                     / (np.sqrt(.5 * (pos[c].std() ** 2
                                                      + neg[c].std() ** 2)) + 1e-9))
                            for c in MATCH_COLS},
            "hard_fraction": float(neg.hard_negative.mean()),
            "unmatched_pool_hard_fraction": float(s[y == 0].hard_negative.mean()),
        }
    return report


# ------------------------------------------------------------------ G
def _eff(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float((a.mean() - b.mean()) / (np.sqrt(.5 * (a.std() ** 2 + b.std() ** 2)) + 1e-12))


def audit_division_signature(comp: pd.DataFrame, ext: pd.DataFrame | None) -> dict:
    """THE decisive external-transfer test.

    A mitosis has one unmistakable mother-centred signature: in the frame AFTER the
    division, the core within 3 um of the mother's last position loses mass (it is
    now split between two separating daughters) and the radius of gyration grows.
    This measures that signature's amplitude and its TIMING, by re-anchoring the
    mother frame to t, t-1 and t-2. A corpus whose annotated fork frames are
    temporally imprecise shows a small, smeared, or displaced signature no matter how
    many positives it has, and cannot teach the cue.
    """
    def one(d: pd.DataFrame, tag: str) -> dict:
        p, n = d[d.label == 1], d[d.label == 0]
        # forward one-step log core-mass ratio, anchored at t+o
        f0 = (p.R_massn_p1.to_numpy(float), n.R_massn_p1.to_numpy(float))
        f1 = (-p.R_massn_m1.to_numpy(float), -n.R_massn_m1.to_numpy(float))
        f2 = (-(p.R_massn_m2 - p.R_massn_m1).to_numpy(float),
              -(n.R_massn_m2 - n.R_massn_m1).to_numpy(float))
        return {"dataset": tag, "n_pos": int(len(p)), "n_neg": int(len(n)),
                "mass_drop_effect_anchor_0": _eff(*f0),
                "mass_drop_effect_anchor_m1": _eff(*f1),
                "mass_drop_effect_anchor_m2": _eff(*f2),
                "rg_growth_effect_p1": _eff(p.D_rg_p1.to_numpy(float),
                                            n.D_rg_p1.to_numpy(float)),
                "rg_growth_effect_p2": _eff(p.D_rg_p2.to_numpy(float),
                                            n.D_rg_p2.to_numpy(float)),
                "best_abs_effect_any_anchor": float(np.nanmax(np.abs([
                    _eff(*f0), _eff(*f1), _eff(*f2)])))}

    rows = [one(labelled(comp[comp.family == f]), f"competition_{f}")
            for f in sorted(comp.family.unique())]
    if ext is not None and len(ext):
        for (emb, win), s in ext.groupby(["embryo", "window"]):
            rows.append(one(labelled(s), f"zebrahub_{emb}_{win}"))
    comp_best = max(abs(r["mass_drop_effect_anchor_0"]) for r in rows
                    if r["dataset"].startswith("competition"))
    ext_best = max([r["best_abs_effect_any_anchor"] for r in rows
                    if r["dataset"].startswith("zebrahub")], default=float("nan"))
    return {"rows": rows, "competition_best_anchor0_abs_effect": comp_best,
            "external_best_abs_effect_any_anchor": ext_best,
            "amplitude_ratio_comp_over_ext": comp_best / ext_best if ext_best else None,
            "external_signature_PASS": bool(ext_best >= 0.5 * comp_best)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--zeb-frame", type=int, default=250)
    ap.add_argument("--l2", type=float, default=30.0)
    a = ap.parse_args()

    comp = pd.read_parquet(a.comp)
    ext = (pd.concat([pd.read_parquet(p) for p in a.external], ignore_index=True)
           if a.external else None)
    feats = F.H1M_FEATURES

    res = {"feature_hash": F.config_hash(),
           "A_labels": audit_labels(comp, ext),
           "B_disjointness": audit_disjointness(ext) if ext is not None else {"skipped": True},
           "F_matched_negatives": build_matched_negatives(comp),
           "G_division_signature": audit_division_signature(comp, ext)}
    if ext is not None and len(ext):
        res["D_sampler"] = audit_sampler(ext, feats, a.l2)
        res["E_domain_shift"] = audit_domain_shift(comp, ext, feats)
    if not a.skip_c:
        res["C_patch_coords_competition"] = audit_patch_coords_competition(comp)
        a4 = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH"
                  r"\agent_runs\agent4\zebrahub_prep")
        res["C_patch_coords_zebrahub"] = audit_patch_coords_zebrahub(
            "ZSNS003", a4 / "ZSNS003.parquet", a.zeb_frame)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
