"""H1-I: join per-node appearance features onto the frozen candidate census and
evaluate bilateral (per embryo-family) AUC on the metric-visible subset.

Nothing here regenerates or retunes the candidate surface: it reads the frozen
h0c_top3 census and only ADDS columns.

Reported per feature:
  AUC on family 44b6 (16 positives vs 54,031 reliable negatives)
  AUC on family 6bba (76 positives vs 199,319 reliable negatives)
  sign consistency  = (auc_a - 0.5) and (auc_b - 0.5) share a sign
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VOX = np.array([1.625, 0.40625, 0.40625])
EPS = 1e-6
BREAKEVEN = {"44b6": 0.0407, "6bba": 0.0638}

# number of kernel voxels inside r<=3um (matches h1i_node_appearance.BND['core'])
N_CORE_VOX = 419.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC, NaN-robust (NaNs dropped)."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return (r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def per_node_derived(nf: pd.DataFrame) -> pd.DataFrame:
    """Dimensionless / self-normalising per-node descriptors."""
    d = pd.DataFrame(index=nf.index)
    dyn = np.maximum(nf.hi - nf.bg, 1.0)
    l1 = np.maximum(nf.lam1, EPS)
    tr = np.maximum(nf.lam1 + nf.lam2 + nf.lam3, EPS)
    d["massn"] = nf.mass_core / (dyn * N_CORE_VOX)
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
    d["peak_raw"] = nf.peak_inner
    for c in ("v1z", "v1y", "v1x"):
        d[c] = nf[c]
    d["inside"] = nf.inside
    return d


NODE_FEATS = ["massn", "conc", "contrast", "saddle_self", "rg", "spher", "aniso",
              "linearity", "planarity", "detn", "kurt", "skew", "p10", "p20",
              "p30", "coff", "peakn", "peak_raw"]


def load_nodes(feat_dir: Path, fold: int, crop: str) -> dict:
    p = feat_dir / str(fold) / f"{crop}.parquet"
    if not p.exists():
        return None
    nf = pq.read_table(p).to_pandas()
    der = per_node_derived(nf)
    der["node_id"] = nf.node_id.values
    der["dt"] = nf.dt.values
    return {int(dt): sub.set_index("node_id") for dt, sub in der.groupby("dt")}


def build_crop(root: Path, feat_dir: Path, mid_dir: Path, fold: int, crop: str):
    cen = pq.read_table(
        root / f"artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3/{fold}/{crop}.parquet"
    ).to_pandas()
    cen = cen[cen.metric_visible & cen.label.isin(["positive", "reliable_negative"])]
    if len(cen) == 0:
        return None
    nd = load_nodes(feat_dir, fold, crop)
    if nd is None:
        return None

    g = pq.read_table(root / f"artifacts/kaggle/e0c_cache/graphs/{fold}/{crop}.parquet",
                      columns=["row_type", "node_id", "t", "z", "y", "x"]).to_pandas()
    g = g[g.row_type == "node"].set_index("node_id")
    P = g[["z", "y", "x"]].to_numpy() * VOX[None, :]
    pos_of = {nid: i for i, nid in enumerate(g.index.to_numpy())}

    out = cen[["cand_id", "crop", "family", "fold", "t", "mother", "d1", "d2",
               "rank", "label"]].copy()

    def take(role, dt, cols):
        tbl = nd.get(dt)
        if tbl is None:
            return pd.DataFrame(np.nan, index=cen.index, columns=cols)
        r = tbl.reindex(cen[role].to_numpy())[cols]
        r.index = cen.index
        return r

    # ---- per-node blocks --------------------------------------------------
    m0 = take("mother", 0, NODE_FEATS + ["v1z", "v1y", "v1x", "inside"])
    mm1 = take("mother", -1, NODE_FEATS)
    mm2 = take("mother", -2, NODE_FEATS)
    mp1 = take("mother", 1, NODE_FEATS + ["v1z", "v1y", "v1x"])
    mp2 = take("mother", 2, NODE_FEATS)
    a0 = take("d1", 0, NODE_FEATS)
    b0 = take("d2", 0, NODE_FEATS)
    am1 = take("d1", -1, NODE_FEATS)
    bm1 = take("d2", -1, NODE_FEATS)

    for c in NODE_FEATS:
        out["m_" + c] = m0[c].values
    # --- mother temporal trajectory ---------------------------------------
    out["m_dconc_m1"] = m0["conc"].values - mm1["conc"].values
    out["m_dspher_m1"] = m0["spher"].values - mm1["spher"].values
    out["m_daniso_m1"] = m0["aniso"].values - mm1["aniso"].values
    out["m_drg_m1"] = m0["rg"].values - mm1["rg"].values
    out["m_massratio_m1"] = m0["massn"].values / np.maximum(mm1["massn"].values, EPS)
    out["m_massratio_m2"] = m0["massn"].values / np.maximum(mm2["massn"].values, EPS)
    out["m_dkurt_p1"] = mp1["kurt"].values - m0["kurt"].values
    out["m_dsaddle_p1"] = mp1["saddle_self"].values - m0["saddle_self"].values
    out["m_drg_p1"] = mp1["rg"].values - m0["rg"].values
    out["m_dspher_p1"] = mp1["spher"].values - m0["spher"].values
    out["m_massratio_p1"] = mp1["massn"].values / np.maximum(m0["massn"].values, EPS)
    out["m_massratio_p2"] = mp2["massn"].values / np.maximum(m0["massn"].values, EPS)
    out["m_kurt_p1"] = mp1["kurt"].values
    out["m_saddle_p1"] = mp1["saddle_self"].values
    out["m_conc_p1"] = mp1["conc"].values
    # condensation profile: peak of concentration at t relative to t-1 and t+1
    out["m_conc_peakedness"] = (2 * m0["conc"].values - mm1["conc"].values - mp1["conc"].values)

    # --- daughter pair symmetry -------------------------------------------
    def sym(u, v):
        return np.abs(u - v) / np.maximum(np.abs(u) + np.abs(v), EPS)

    out["d_massbal"] = sym(a0["massn"].values, b0["massn"].values)
    out["d_peakbal"] = sym(a0["peakn"].values, b0["peakn"].values)
    out["d_rgbal"] = sym(a0["rg"].values, b0["rg"].values)
    out["d_spherbal"] = sym(a0["spher"].values, b0["spher"].values)
    out["d_concbal"] = sym(a0["conc"].values, b0["conc"].values)
    out["d_contrastbal"] = sym(a0["contrast"].values, b0["contrast"].values)
    out["d_mass_mean"] = 0.5 * (a0["massn"].values + b0["massn"].values)
    out["d_conc_mean"] = 0.5 * (a0["conc"].values + b0["conc"].values)
    out["d_spher_mean"] = 0.5 * (a0["spher"].values + b0["spher"].values)
    out["d_contrast_mean"] = 0.5 * (a0["contrast"].values + b0["contrast"].values)
    out["d_rg_mean"] = 0.5 * (a0["rg"].values + b0["rg"].values)

    # --- mass conservation -------------------------------------------------
    out["mass_cons"] = ((a0["massn"].values + b0["massn"].values)
                        / np.maximum(m0["massn"].values, EPS))
    out["mass_cons_log"] = np.log(np.maximum(out["mass_cons"].values, EPS))
    out["mass_cons_dev"] = np.abs(out["mass_cons_log"].values)
    # daughters existed at t-1 already? (a real daughter should not)
    out["d_massratio_m1"] = ((a0["massn"].values + b0["massn"].values)
                             / np.maximum(am1["massn"].values + bm1["massn"].values, EPS))

    # --- axis alignment ----------------------------------------------------
    i1 = np.array([pos_of[n] for n in cen.d1.to_numpy()])
    i2 = np.array([pos_of[n] for n in cen.d2.to_numpy()])
    im = np.array([pos_of[n] for n in cen.mother.to_numpy()])
    sep = P[i2] - P[i1]
    sepn = np.linalg.norm(sep, axis=1)
    su = sep / np.maximum(sepn, EPS)[:, None]
    out["sep_um"] = sepn
    for tag, blk in (("m0", m0), ("mp1", mp1)):
        v = np.stack([blk["v1z"].values, blk["v1y"].values, blk["v1x"].values], axis=1)
        out[f"axis_align_{tag}"] = np.abs(np.einsum("ij,ij->i", v, su))
    # mother displacement axis vs separation axis
    disp = 0.5 * (P[i1] + P[i2]) - P[im]
    out["mid_off_um"] = np.linalg.norm(disp, axis=1)

    # --- mid-line saddle between the two daughters at t+1 -------------------
    mp = mid_dir / str(fold) / f"{crop}.parquet"
    if mp.exists():
        mid = pq.read_table(mp).to_pandas()
        if len(mid):
            mid = mid.drop_duplicates("cand_id").set_index("cand_id")
            mid = mid.reindex(cen.cand_id.to_numpy())
            rmin = mid["ridge_min"].to_numpy()
            rmid = mid["ridge_mid"].to_numpy()
            pk1 = a0["peak_raw"].values
            pk2 = b0["peak_raw"].values
            pk_lo = np.minimum(pk1, pk2)
            pk_mu = 0.5 * (pk1 + pk2)
            out["saddle_ratio"] = rmin / np.maximum(pk_lo, EPS)
            out["saddle_ratio_mean"] = rmin / np.maximum(pk_mu, EPS)
            out["saddle_mid_ratio"] = rmid / np.maximum(pk_lo, EPS)
            # a real division sits at intermediate saddle depth: neither a solid
            # ridge (one blob, ratio ~1) nor empty background (ratio ~0).
            lr = np.log(np.clip(out["saddle_ratio"].values, 1e-3, 10.0))
            out["saddle_band"] = -np.abs(lr - np.log(0.5))
            out["saddle_x_sep"] = out["saddle_ratio"].values / np.maximum(sepn, EPS)

    # --- frozen geometry features (h1g), for like-for-like comparison -------
    gp = root / f"artifacts/kaggle/e0c_cache/fork_candidates/h1g_features/{fold}/{crop}.parquet"
    if gp.exists():
        gf = pq.read_table(gp).to_pandas().drop_duplicates("cand_id").set_index("cand_id")
        gcols = [c for c in gf.columns if c not in
                 ("crop", "fold", "family", "mother", "d1", "d2", "rank", "label",
                  "metric_visible")]
        gf = gf.reindex(cen.cand_id.to_numpy())[gcols]
        for c in gcols:
            out["G_" + c] = gf[c].to_numpy()
    return out


def evaluate(df: pd.DataFrame, feats: list[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    fam = {f: df[df.family == f] for f in ("44b6", "6bba")}
    for c in feats:
        r = {"feature": c}
        for f, sub in fam.items():
            p = sub.loc[sub.label == "positive", c].to_numpy(dtype=float)
            n = sub.loc[sub.label == "reliable_negative", c].to_numpy(dtype=float)
            r[f"auc_{f}"] = auc(p, n)
            r[f"npos_{f}"] = int(np.isfinite(p).sum())
        a, b = r["auc_44b6"], r["auc_6bba"]
        r["sign_consistent"] = bool(np.isfinite(a) and np.isfinite(b)
                                    and np.sign(a - 0.5) == np.sign(b - 0.5))
        r["min_sep"] = min(abs(a - 0.5), abs(b - 0.5)) if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("min_sep", ascending=False)
    res.to_csv(out_csv, index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--mid-dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    feat_dir = Path(args.feat_dir)
    mid_dir = Path(args.mid_dir) if args.mid_dir else Path("___none___")
    parts = []
    for fold in (0, 1):
        for p in sorted(glob.glob(str(feat_dir / str(fold) / "*.parquet"))):
            crop = Path(p).stem
            r = build_crop(root, feat_dir, mid_dir, fold, crop)
            if r is not None:
                parts.append(r)
    df = pd.concat(parts, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")
    print("joined rows", len(df))
    print(df.groupby(["family", "label"]).size().to_string())

    skip = {"cand_id", "crop", "family", "fold", "t", "mother", "d1", "d2",
            "rank", "label"}
    feats = [c for c in df.columns if c not in skip
             and pd.api.types.is_numeric_dtype(df[c])]
    res = evaluate(df, feats, Path(args.out).with_name("auc_table.csv"))
    pd.set_option("display.width", 200)
    print(res.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
