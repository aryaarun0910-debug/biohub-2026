r"""Error atlas part 2/3: per-emitted-edge feature table + candidate-graph comparison.

Builds, for every emitted edge of a LOEO export, every separating feature that can be
computed WITHOUT the model (geometry / topology / motion), plus ``edge_prob`` joined
from the pre-ILP candidate dump when one is supplied.

Label comes from ``ea_atlas.py``:
    scored TP   -> pred_valid & matched           (deleting it costs a point)
    scored FP   -> pred_valid & ~matched          (deleting it gains a point)
    free        -> ~pred_valid                    (deleting it is metric-neutral)

Features (all measured on the emitted graph itself, so no cross-run join needed):
  disp_um           physical source->target displacement
  nn_margin_um      d(target, chosen source) - d(target, nearest OTHER node at t_src)
                    == the geometric analogue of the shipped `logitdiff` pruner
  rev_margin_um     d(source, chosen target) - d(source, nearest OTHER node at t_tgt)
  src_outdeg        out-degree of the source in the emitted graph
  tgt_indeg         in-degree of the target in the emitted graph
  dens15_src        predicted nodes within 15 um of the source, same frame
  disp_ratio        disp_um / median disp_um of the crop-frame (motion-regime normalised)
  accel_um          |disp - disp of the source's own incoming edge| (NaN if none)
  cos_prev          cosine between this edge's vector and the incoming edge's vector
  chain_in/chain_out  how many consecutive frames the track persists back/forward
  s_z, s_t          depth and time
  edge_prob         model probability, from the pre-ILP dump (optional)

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\ea_features.py --tag f1 ^
      --dir c:\temp\error_atlas --csv c:\temp\subvoxel_f1\loeo_split1_strict.csv.gz ^
      --preilp c:\temp\preilp_f1_v2\preilp_split1.parquet
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
BIG = 1e6


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".gz":
        tmp = Path(tempfile.mkdtemp(prefix="eaf_")) / path.name[:-3]
        with gzip.open(path, "rb") as f, tmp.open("wb") as o:
            shutil.copyfileobj(f, o)
        path = tmp
    return pl.read_csv(path).to_pandas()


def crop_features(nd: pd.DataFrame, ed: pd.DataFrame) -> pd.DataFrame:
    """nd: node_id,t,z,y,x for one crop. ed: source_id,target_id for one crop."""
    pos = nd.set_index("node_id")[["t", "z", "y", "x"]]
    P = pos[["z", "y", "x"]].values * SCALE
    tarr = pos["t"].values
    ids = pos.index.values
    pos_of = {int(i): k for k, i in enumerate(ids)}

    # per-frame KD trees
    trees, frame_idx = {}, {}
    for t in np.unique(tarr):
        m = np.flatnonzero(tarr == t)
        frame_idx[int(t)] = m
        trees[int(t)] = cKDTree(P[m])

    si = ed.source_id.map(pos_of).values
    ti = ed.target_id.map(pos_of).values
    vs, vt = P[si], P[ti]
    ts, tt = tarr[si], tarr[ti]
    d = np.linalg.norm(vt - vs, axis=1)

    out = pd.DataFrame({"source_id": ed.source_id.values, "target_id": ed.target_id.values,
                        "disp_um": d, "s_t": ts, "s_z": nd.set_index("node_id").z.reindex(
                            ed.source_id.values).values})

    # --- nn_margin: from the TARGET, how much closer is the chosen source than the
    #     nearest other node in the source frame?  (the logitdiff analogue)
    nn = np.full(len(ed), np.nan)
    for t in np.unique(ts):
        sel = np.flatnonzero(ts == t)
        tr = trees[int(t)]
        # 2 nearest source-frame nodes to each target point
        k = min(2, len(frame_idx[int(t)]))
        dd, ii = tr.query(vt[sel], k=k)
        if k == 1:
            dd = dd[:, None]; ii = ii[:, None]
        gid = frame_idx[int(t)][ii]                       # global row index
        chosen = si[sel][:, None]
        is_chosen = gid == chosen
        other = np.where(is_chosen, np.inf, dd)
        nearest_other = other.min(axis=1)
        nearest_other[~np.isfinite(nearest_other)] = BIG   # only 1 node in frame
        nn[sel] = d[sel] - nearest_other
    out["nn_margin_um"] = nn

    # --- rev_margin: from the SOURCE, how much closer is the chosen target than the
    #     nearest other node in the target frame?
    rv = np.full(len(ed), np.nan)
    for t in np.unique(tt):
        sel = np.flatnonzero(tt == t)
        tr = trees[int(t)]
        k = min(2, len(frame_idx[int(t)]))
        dd, ii = tr.query(vs[sel], k=k)
        if k == 1:
            dd = dd[:, None]; ii = ii[:, None]
        gid = frame_idx[int(t)][ii]
        chosen = ti[sel][:, None]
        other = np.where(gid == chosen, np.inf, dd)
        no = other.min(axis=1)
        no[~np.isfinite(no)] = BIG
        rv[sel] = d[sel] - no
    out["rev_margin_um"] = rv

    # --- degrees
    out["src_outdeg"] = ed.groupby("source_id").source_id.transform("size").values
    out["tgt_indeg"] = ed.groupby("target_id").target_id.transform("size").values

    # --- local density around the source
    dens = np.empty(len(ids))
    for t, m in frame_idx.items():
        dens[m] = trees[t].query_ball_point(P[m], r=15.0, return_length=True) - 1
    out["dens15_src"] = dens[si]

    # --- motion-regime normalisation: median displacement of the crop-frame
    tmp = pd.DataFrame({"t": ts, "d": d})
    med = tmp.groupby("t").d.transform("median").values
    out["disp_ratio"] = d / np.maximum(med, 1e-6)

    # --- acceleration / direction agreement with the source's own incoming edge
    inc = {}
    for s, tg, k in zip(si, ti, range(len(ed))):
        inc.setdefault(int(tg), k)     # target -> the (unique-ish) incoming edge row
    prev = np.array([inc.get(int(s), -1) for s in si])
    has = prev >= 0
    accel = np.full(len(ed), np.nan)
    cosp = np.full(len(ed), np.nan)
    if has.any():
        pv = prev[has]
        v_prev = vt[pv] - vs[pv]
        v_cur = (vt - vs)[has]
        accel[has] = np.abs(np.linalg.norm(v_cur, axis=1) - np.linalg.norm(v_prev, axis=1))
        nrm = np.linalg.norm(v_cur, axis=1) * np.linalg.norm(v_prev, axis=1)
        cosp[has] = np.where(nrm > 1e-9, (v_cur * v_prev).sum(axis=1) / np.maximum(nrm, 1e-9), 0.0)
    out["accel_um"] = accel
    out["cos_prev"] = cosp
    out["has_prev"] = has

    # --- chain persistence (how long the track continues either side)
    has_out = pd.Series(ed.target_id.values).isin(set(ed.source_id.values)).values
    out["tgt_continues"] = has_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv", required=True, help="the export the atlas was run on")
    ap.add_argument("--preilp", help="pre-ILP parquet supplying edge_prob")
    args = ap.parse_args()
    D = Path(args.dir)

    sub = read_any(Path(args.csv))
    nodes = sub[sub.row_type == "node"]
    eds = sub[sub.row_type == "edge"]
    print(f"{sub.dataset.nunique()} crops, {len(nodes)} nodes, {len(eds)} edges")

    feats = []
    for i, (ds, nd) in enumerate(nodes.groupby("dataset"), 1):
        ed = eds[eds.dataset == ds]
        f = crop_features(nd[["node_id", "t", "z", "y", "x"]], ed[["source_id", "target_id"]])
        f.insert(0, "dataset", ds)
        feats.append(f)
        if i % 20 == 0:
            print(f"  {i} crops", flush=True)
    F = pd.concat(feats, ignore_index=True)

    # --- attach the scorer label -------------------------------------------------
    lab = pd.read_parquet(D / f"edges_{args.tag}.parquet")[
        ["dataset", "source_id", "target_id", "matched", "pred_valid"]]
    F = F.merge(lab, on=["dataset", "source_id", "target_id"], how="left")
    F["scored"] = F.pred_valid.fillna(False)
    F["label"] = np.where(~F.scored, "free", np.where(F.matched.fillna(False), "TP", "FP"))
    print(F.label.value_counts())

    # --- attach edge_prob from the pre-ILP dump ---------------------------------
    if args.preilp:
        pre = pd.read_parquet(args.preilp)
        pn = pre[pre.row_type == "node"][["dataset", "node_id", "t", "z", "y", "x"]]
        pe = pre[pre.row_type == "edge"][["dataset", "source_id", "target_id", "edge_prob"]]
        # coordinate-consistency gate: the two runs share an id space only approximately
        m = nodes[["dataset", "node_id", "t", "z", "y", "x"]].merge(
            pn, on=["dataset", "node_id"], how="inner", suffixes=("", "_p"))
        dist = np.sqrt(((m.z - m.z_p) * SCALE[0]) ** 2 + ((m.y - m.y_p) * SCALE[1]) ** 2
                       + ((m.x - m.x_p) * SCALE[2]) ** 2)
        ok = (m.t == m.t_p) & (dist <= 3.5)
        good = m.loc[ok, ["dataset", "node_id"]]
        print(f"id-join: {len(m)}/{len(nodes)} ids shared, {int(ok.sum())} coord-consistent "
              f"({ok.mean():.4f} of shared)")
        gs = set(map(tuple, good.values))
        F["_gs"] = [(d, s) in gs for d, s in zip(F.dataset, F.source_id)]
        F["_gt_"] = [(d, s) in gs for d, s in zip(F.dataset, F.target_id)]
        F = F.merge(pe, on=["dataset", "source_id", "target_id"], how="left")
        F["edge_prob"] = F.edge_prob.where(F._gs & F._gt_)
        F = F.drop(columns=["_gs", "_gt_"])
        print(f"edge_prob attached to {F.edge_prob.notna().mean():.4f} of emitted edges; "
              f"of SCORED edges {F.loc[F.scored, 'edge_prob'].notna().mean():.4f}")
        # source-side probability margin within the candidate graph
        pe = pe.sort_values("edge_prob", ascending=False)
        pe["rank_in_src"] = pe.groupby(["dataset", "source_id"]).cumcount() + 1
        pe["src_best"] = pe.groupby(["dataset", "source_id"]).edge_prob.transform("max")
        pe["src_margin"] = pe.src_best - pe.edge_prob
        F = F.merge(pe[["dataset", "source_id", "target_id", "rank_in_src", "src_margin"]],
                    on=["dataset", "source_id", "target_id"], how="left")

    F.to_parquet(D / f"feats_{args.tag}.parquet")
    print(f"wrote {D}/feats_{args.tag}.parquet  ({len(F)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
