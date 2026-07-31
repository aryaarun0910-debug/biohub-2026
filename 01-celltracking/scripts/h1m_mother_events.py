"""H1-M: build the competition-side MOTHER-DIVISION event table.

ONE ROW PER (crop, mother node, t) EVENT. There is deliberately no duplication across
candidate daughter pairs -- that inflated n in earlier work and turned a mother-level
question into a pair-ranking question.

Universe        every METRIC-VISIBLE mother on the frozen H0c top-3 shortlist, i.e.
                every prediction node that matches an annotated GT node with
                out-degree >= 1. Only these can ever contribute a division TP or FP,
                so this is the true precision denominator.
Positive        mother_gt_outdeg >= 2  (the GT annotation says this cell divides)
Negative        mother_gt_outdeg == 1  (the GT annotation says this cell continues)
Excluded        out-degree 0 / unmatched -> annotation ends, division cannot be ruled
                out, so these are UNLABELED and never used as negatives.

`realisable` additionally marks the mothers whose TRUE daughter pair survived into the
top-3 shortlist -- the subset H0c can actually convert into division credit.

Matching covariates are carried per event (t, local density, mother core mass, track
age, motion) so a matched negative set can be drawn without re-reading imagery.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m_mother_events.py --feat-dir <dir> --out <parquet>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402

ROOT_DEFAULT = r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026"
CENSUS = "artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3"
GRAPHS = "artifacts/kaggle/e0c_cache/graphs"
DENSITY_UM = 15.0

PROV_COLS = ["embryo", "family", "fold", "crop", "t", "mother", "mother_track_id",
             "z_vox", "y_vox", "z_um", "y_um", "x_vox", "x_um"]


def crop_events(root: Path, feat_dir: Path, fold: int, crop: str,
                n_core_vox: float) -> pd.DataFrame | None:
    cp = root / CENSUS / str(fold) / f"{crop}.parquet"
    fp = feat_dir / str(fold) / f"{crop}.parquet"
    if not (cp.exists() and fp.exists()):
        return None
    cen = pq.read_table(cp, columns=["crop", "family", "fold", "t", "mother", "d1", "d2",
                                     "rank", "flow_midpoint_residual", "label",
                                     "metric_visible", "mother_gt_outdeg",
                                     "steal_required"]).to_pandas()
    cen = cen[cen.metric_visible]
    if len(cen) == 0:
        return None

    # ---- collapse the pair shortlist to one row per mother -------------------
    cen = cen.sort_values(["mother", "rank"])
    g = cen.groupby("mother", sort=False)
    ev = pd.DataFrame({
        "t": g.t.first(),
        "mother_gt_outdeg": g.mother_gt_outdeg.first(),
        "n_cand": g.size(),
        "best_resid_um": g.flow_midpoint_residual.min(),
        "worst_resid_um": g.flow_midpoint_residual.max(),
        "steal_rank0": g.steal_required.first(),
        "realisable": g.label.apply(lambda s: bool((s == "positive").any())),
    }).reset_index()
    ev["crop"] = crop
    ev["family"] = crop.split("_")[0]
    ev["fold"] = fold
    ev["embryo"] = crop                       # crop == acquisition unit here

    # ---- graph geometry: density, track age, motion ---------------------------
    gr = pq.read_table(root / GRAPHS / str(fold) / f"{crop}.parquet",
                       columns=["row_type", "node_id", "t", "z", "y", "x",
                                "source_id", "target_id"]).to_pandas()
    nodes = gr[gr.row_type == "node"]
    edges = gr[gr.row_type == "edge"]
    nid = nodes.node_id.to_numpy()
    P = nodes[["z", "y", "x"]].to_numpy(float) * F.VOX_COMPETITION[None, :]
    tt = nodes.t.to_numpy()
    row_of = {int(n): i for i, n in enumerate(nid)}
    parent = {int(b): int(a) for a, b in zip(edges.source_id.to_numpy(),
                                             edges.target_id.to_numpy())}
    nchild: dict[int, int] = {}
    for a in edges.source_id.to_numpy():
        nchild[int(a)] = nchild.get(int(a), 0) + 1

    from scipy.spatial import cKDTree
    dens = np.zeros(len(nid))
    for f in np.unique(tt):
        sel = np.flatnonzero(tt == f)
        tr = cKDTree(P[sel])
        dens[sel] = np.array([len(x) for x in tr.query_ball_point(P[sel], DENSITY_UM)]) - 1

    mi = np.array([row_of[int(m)] for m in ev.mother.to_numpy()])
    ev["local_density"] = dens[mi]
    ev["z_vox"] = nodes.z.to_numpy()[mi]
    ev["y_vox"] = nodes.y.to_numpy()[mi]
    ev["x_vox"] = nodes.x.to_numpy()[mi]
    ev["z_um"], ev["y_um"], ev["x_um"] = P[mi, 0], P[mi, 1], P[mi, 2]
    ev["pred_outdeg"] = [nchild.get(int(m), 0) for m in ev.mother.to_numpy()]

    age = np.zeros(len(ev), dtype=np.int32)
    speed = np.full(len(ev), np.nan)
    for k, m in enumerate(ev.mother.to_numpy()):
        cur, n = int(m), 0
        p = parent.get(cur)
        if p is not None:
            speed[k] = float(np.linalg.norm(P[row_of[p]] - P[row_of[cur]]))
        while p is not None and n < 100:
            cur, n = p, n + 1
            p = parent.get(cur)
        age[k] = n
    ev["track_age"] = age
    ev["speed_um"] = speed
    ev["mother_track_id"] = -1                # E0c graphs carry no persistent track id

    # ---- appearance ----------------------------------------------------------
    nf = pq.read_table(fp).to_pandas()
    nf = nf[nf.node_id.isin(set(ev.mother.to_numpy()))]
    der = F.node_derived(nf, n_core_vox)
    der["node_id"] = nf.node_id.to_numpy()
    der["dt"] = nf.dt.to_numpy()
    der["inside"] = nf.inside.to_numpy()
    by_dt = {}
    for dt in F.DTS:
        s = der[der.dt == dt].drop_duplicates("node_id").set_index("node_id")
        by_dt[dt] = s.reindex(ev.mother.to_numpy()).reset_index(drop=True)
    feats = F.mother_event_features(by_dt, ev.index)
    ev["inside_volume"] = by_dt[0]["inside"].fillna(False).to_numpy().astype(bool)
    ev["n_frames_seen"] = sum(by_dt[d]["conc"].notna().to_numpy().astype(int)
                              for d in F.DTS)
    ev["L_raw_massn"] = by_dt[0]["massn"].to_numpy()   # kept for matching only
    ev = pd.concat([ev, feats], axis=1)

    ev["label"] = np.where(ev.mother_gt_outdeg >= 2, 1,
                           np.where(ev.mother_gt_outdeg == 1, 0, -1))
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root, feat_dir = Path(a.root), Path(a.feat_dir)
    kern = F.build_kernel(F.VOX_COMPETITION)
    t0, parts = time.time(), []
    for fold in (0, 1):
        for p in sorted((feat_dir / str(fold)).glob("*.parquet")):
            r = crop_events(root, feat_dir, fold, p.stem, kern["n_core_vox"])
            if r is not None and len(r):
                parts.append(r)
    ev = pd.concat(parts, ignore_index=True)

    # hard negatives: a non-divider whose best candidate pair is at least as
    # geometrically plausible as the median true divider (frozen proposer residual)
    for fam, sub in ev.groupby("family"):
        med = sub.loc[sub.label == 1, "best_resid_um"].median()
        ev.loc[sub.index, "hard_negative"] = ((ev.loc[sub.index, "label"] == 0)
                                              & (ev.loc[sub.index, "best_resid_um"] <= med))
        ev.loc[sub.index, "pos_median_resid_um"] = med
    ev["hard_negative"] = ev["hard_negative"].fillna(False).astype(bool)
    ev["feature_hash"] = F.config_hash()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out, compression="zstd", index=False)

    rep = {"feature_hash": F.config_hash(), "n_features": len(F.H1M_FEATURES),
           "seconds": round(time.time() - t0, 1), "rows": int(len(ev)),
           "crops": int(ev.crop.nunique()), "families": {}}
    for fam, s in ev.groupby("family"):
        rep["families"][fam] = {
            "mother_events": int(len(s)),
            "positives_divider": int((s.label == 1).sum()),
            "positives_realisable": int(s.realisable.sum()),
            "negatives": int((s.label == 0).sum()),
            "hard_negatives": int(s.hard_negative.sum()),
            "unlabeled": int((s.label == -1).sum()),
            "prevalence": float((s.label == 1).mean()),
            "complete_5frame": int((s.n_frames_seen == 5).sum()),
        }
    print(json.dumps(rep, indent=2))
    Path(str(out) + ".census.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
