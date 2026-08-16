"""H1-T appendix — does APPEARANCE separate a true fork from its nearest geometric impostor?

The geometry/kinematics critic answers "which pair" (top-1 0.72-0.81) but not "which mother"
(zero true dividers in the top 100 mothers of either family). Everything in that feature space
is geometry, and one frame before a division a dividing mother is geometrically ordinary. The
distinguishing cue, if there is one, is mitotic appearance: rounding, chromatin condensation,
a brighter and more compact nucleus.

Before anyone downloads external imagery or spends GPU on a 3D encoder, this asks the cheap
prior question on data already on disk: taking the 92 competition positives and, for each of
their crops, the highest-scoring reliable-negative candidates from the same crop, do simple
image statistics separate them at all?

Every statistic is deployment-observable (no ground truth, no family identity). Reported per
family with a cross-family logistic check, because same-family separation that does not cross
the family boundary is exactly the failure mode this project keeps hitting.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_appearance_probe.py --per-crop-negatives 20
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from h1t_external_critic import COMP, FEATS  # noqa: E402
from phaseb_d0p_proposer import SCALE, load_e0c_tables  # noqa: E402

SCRATCH = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026\_evidence\agent_runs\agent4")
WEIGHTS = SCRATCH / "h1t_critic_weights.npz"
OUT = SCRATCH / "h1t_appearance.json"
R_NUC_UM = 3.5      # nuclear ball
R_BG_UM = 12.0      # local background ball
STATS = ["nuc_mean", "nuc_max", "nuc_std", "nuc_over_bg", "compactness",
         "d_nuc_mean", "d_over_m", "temporal_rise", "sister_sym"]


def load_mlp():
    z = np.load(WEIGHTS)
    mu, sd = z["mu"], z["sd"]
    W = [(z[f"{i}.weight"], z[f"{i}.bias"]) for i in (0, 2, 4)]

    def fwd(X):
        h = (X - mu) / sd
        for k, (w, b) in enumerate(W):
            h = h @ w.T + b
            if k < len(W) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]
    return fwd


def _ball(shape):
    """Offsets and physical radii for a box that covers R_BG_UM in every direction."""
    rz = int(np.ceil(R_BG_UM / SCALE[0]))
    ry = int(np.ceil(R_BG_UM / SCALE[1]))
    rx = int(np.ceil(R_BG_UM / SCALE[2]))
    dz, dy, dx = np.meshgrid(np.arange(-rz, rz + 1), np.arange(-ry, ry + 1),
                             np.arange(-rx, rx + 1), indexing="ij")
    d = np.sqrt((dz * SCALE[0]) ** 2 + (dy * SCALE[1]) ** 2 + (dx * SCALE[2]) ** 2)
    return (rz, ry, rx), (dz, dy, dx), d


def patch_stats(vol, t, z, y, x, box, off, dist):
    """Intensity statistics in a nuclear ball and a local-background ball at (t, z, y, x)."""
    (rz, ry, rx) = box
    T, Z, Y, X = vol.shape
    if not (0 <= t < T):
        return None
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ry), min(Y, y + ry + 1)
    x0, x1 = max(0, x - rx), min(X, x + rx + 1)
    cube = vol[t, z0:z1, y0:y1, x0:x1].astype(np.float32)
    dz, dy, dx = off
    sub = dist[(z0 - z + rz):(z1 - z + rz), (y0 - y + ry):(y1 - y + ry),
               (x0 - x + rx):(x1 - x + rx)]
    if cube.size == 0 or cube.shape != sub.shape:
        return None
    nuc = cube[sub <= R_NUC_UM]
    bg = cube[(sub > R_NUC_UM) & (sub <= R_BG_UM)]
    if nuc.size < 4 or bg.size < 8:
        return None
    bgm = float(np.median(bg)) + 1e-6
    # intensity-weighted compactness: fraction of nuclear signal inside half the radius
    inner = cube[sub <= R_NUC_UM * 0.5]
    comp = float(inner.sum()) / (float(nuc.sum()) + 1e-6)
    return {"mean": float(nuc.mean()), "max": float(nuc.max()), "std": float(nuc.std()),
            "over_bg": float(nuc.mean()) / bgm, "compact": comp}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-crop-negatives", type=int, default=20)
    a = ap.parse_args()
    import zarr

    fwd = load_mlp()
    comp = (pl.scan_parquet(str(COMP / "*" / "*.parquet")).filter(pl.col("metric_visible"))
            .collect().with_columns(pl.col("steal_required").cast(pl.Int64))
            .filter(pl.col("label").is_in(["positive", "reliable_negative"])))
    comp = comp.with_columns(
        pl.Series("critic", fwd(comp.select(FEATS).to_numpy().astype(np.float32))))

    pos = comp.filter(pl.col("label") == "positive")
    crops = sorted(pos["crop"].unique().to_list())
    print(f"{pos.height} positives across {len(crops)} crops")

    box, off, dist = _ball(None)
    rows = []
    for crop in crops:
        fold = int(comp.filter(pl.col("crop") == crop)["fold"][0])
        sub_ids, t_arr, _pos_um, edges, nd = load_e0c_tables(fold, crop)
        vox = {int(s): (int(z), int(y), int(x)) for s, z, y, x in
               zip(nd["node_id"], nd["z"], nd["y"], nd["x"])}
        tof = {int(s): int(tt) for s, tt in zip(sub_ids, t_arr)}
        vol = zarr.open(str(ROOT / "data" / "train" / f"{crop}.zarr"), mode="r")["0"]
        vol = np.asarray(vol[:])                      # 100x64x256x256 uint16, ~800 MB -> load once

        c = comp.filter(pl.col("crop") == crop)
        take = pl.concat([
            c.filter(pl.col("label") == "positive"),
            (c.filter(pl.col("label") == "reliable_negative")
             .sort("critic", descending=True).head(a.per_crop_negatives)),
        ])
        for r in take.iter_rows(named=True):
            m, d1, d2 = int(r["mother"]), int(r["d1"]), int(r["d2"])
            if m not in vox or d1 not in vox or d2 not in vox:
                continue
            t = tof[m]
            sm = patch_stats(vol, t, *vox[m], box, off, dist)
            sm_prev = patch_stats(vol, t - 1, *vox[m], box, off, dist)
            s1 = patch_stats(vol, t + 1, *vox[d1], box, off, dist)
            s2 = patch_stats(vol, t + 1, *vox[d2], box, off, dist)
            if sm is None or s1 is None or s2 is None:
                continue
            dm = 0.5 * (s1["mean"] + s2["mean"])
            rows.append({
                "crop": crop, "family": r["family"], "fold": fold,
                "label": r["label"], "y": int(r["label"] == "positive"),
                "critic": float(r["critic"]),
                "nuc_mean": sm["mean"], "nuc_max": sm["max"], "nuc_std": sm["std"],
                "nuc_over_bg": sm["over_bg"], "compactness": sm["compact"],
                "d_nuc_mean": dm, "d_over_m": dm / (sm["mean"] + 1e-6),
                "temporal_rise": sm["mean"] / (sm_prev["mean"] + 1e-6) if sm_prev else 1.0,
                "sister_sym": min(s1["mean"], s2["mean"]) / (max(s1["mean"], s2["mean"]) + 1e-6),
            })
        del vol
    df = pl.DataFrame(rows)
    print(f"probe rows={df.height} positives={int(df['y'].sum())}")

    res = {"per_crop_negatives": a.per_crop_negatives, "rows": df.height,
           "positives": int(df["y"].sum()), "per_family": {}}
    for fam in ("44b6", "6bba"):
        s = df.filter(pl.col("family") == fam)
        y = s["y"].to_numpy()
        print(f"\n  {fam}: n={s.height} pos={int(y.sum())} hard-negatives={int((1 - y).sum())}")
        fam_res = {}
        for st in STATS:
            v = s[st].to_numpy()
            au = auc(y, v)
            fam_res[st] = au
            print(f"      {st:<16} AUC={au:.4f}")
        fam_res["critic_on_same_pool"] = auc(y, s["critic"].to_numpy())
        print(f"      {'(geometry critic)':<16} AUC={fam_res['critic_on_same_pool']:.4f}")
        res["per_family"][fam] = fam_res

    # ---- cross-family: fit on one family's appearance stats, test on the other ---------
    from sklearn.linear_model import LogisticRegression
    cross = {}
    for fit_fam, test_fam in (("6bba", "44b6"), ("44b6", "6bba")):
        tr = df.filter(pl.col("family") == fit_fam)
        te = df.filter(pl.col("family") == test_fam)
        Xtr = tr.select(STATS).to_numpy()
        Xte = te.select(STATS).to_numpy()
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit((Xtr - mu) / sd, tr["y"].to_numpy())
        au = auc(te["y"].to_numpy(), lr.decision_function((Xte - mu) / sd))
        cross[test_fam] = au
        print(f"  CROSS-FAMILY appearance logistic {fit_fam} -> {test_fam}: AUC={au:.4f}")
    res["cross_family_appearance_auc"] = cross
    OUT.write_text(json.dumps(res, indent=2, default=float))
    df.write_parquet(SCRATCH / "h1t_appearance_rows.parquet")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
