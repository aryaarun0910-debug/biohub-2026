r"""Error atlas: scorer-exact per-edge / per-GT-edge loss classification.

Replays the official matching (``tracking_cellmot.metrics.evaluate`` ->
``tracksdata`` ``DistanceMatching(max_distance=7.0, scale=(1.625,.40625,.40625))``)
for every crop of a LOEO export, then *dumps the intermediate tables the scorer
throws away*:

  nodes_<tag>.parquet    one row per predicted node   -> matched GT id (or -1)
  edges_<tag>.parquet    one row per predicted edge   -> scored?/TP?/free?
  gtedges_<tag>.parquet  one row per ground-truth edge-> TP? both endpoints detected?
  crops_<tag>.parquet    one row per crop             -> the official metric row

Every count in these tables reconciles with ``scripts/core/score_loeo_submission.py``:
  edge_tp  == edges.matched.sum()
  edge_fp  == edges.pred_valid.sum() - edge_tp
  edge_fn  == len(gtedges) - edge_tp

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\ea_atlas.py ^
      --csv c:\temp\subvoxel_f0\loeo_split0_strict.csv.gz --tag f0 ^
      --out-dir c:\temp\error_atlas
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

import tracksdata as td  # noqa: E402

from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph  # noqa: E402
from biotrack.submission import read_submission  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    _evaluate_matched_graph,
    evaluate,
    node_recall,
    per_sample_metrics,
)

K = td.DEFAULT_ATTR_KEYS
NID, MID, MASK = K.NODE_ID, K.MATCHED_NODE_ID, K.MATCHED_EDGE_MASK
ESRC, ETGT, EID = K.EDGE_SOURCE, K.EDGE_TARGET, K.EDGE_ID


def open_csv(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    tmp = Path(tempfile.mkdtemp(prefix="ea_")) / path.name[:-3]
    with gzip.open(path, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return tmp


def build_graph(sub: pl.DataFrame):
    """Exact replica of biotrack.submission.submission_to_graphs for ONE dataset,
    but also returns the submission_node_id -> internal_node_id map."""
    nodes = sub.filter(pl.col("row_type") == "node").sort("node_id")
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in zip(
            nodes["t"].to_list(), nodes["z"].to_list(),
            nodes["y"].to_list(), nodes["x"].to_list(),
        )
    ])
    sub_ids = [int(s) for s in nodes["node_id"].to_list()]
    s2i = dict(zip(sub_ids, internal))
    edges = sub.filter(pl.col("row_type") == "edge")
    if edges.height > 0:
        g.bulk_add_edges([
            {"source_id": s2i[int(s)], "target_id": s2i[int(t)]}
            for s, t in zip(edges["source_id"].to_list(), edges["target_id"].to_list())
        ])
    return g, {v: k for k, v in s2i.items()}  # internal -> submission id


def gt_tables(gt) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.asarray(gt.node_ids())
    na = gt.node_attrs(attr_keys=[NID, "t", "z", "y", "x"]).to_pandas()
    na = na.rename(columns={NID: "gt_id"})
    na["out_degree"] = np.asarray(gt.out_degree(list(ids)))[
        np.searchsorted(np.sort(ids), na.gt_id.values)
    ] if False else np.nan  # filled below
    # degrees, aligned by explicit id order
    deg_out = dict(zip([int(i) for i in ids], [int(d) for d in gt.out_degree(list(ids))]))
    deg_in = dict(zip([int(i) for i in ids], [int(d) for d in gt.in_degree(list(ids))]))
    na["out_degree"] = na.gt_id.map(deg_out)
    na["in_degree"] = na.gt_id.map(deg_in)
    ea = gt.edge_attrs(attr_keys=[ESRC, ETGT]).to_pandas()
    ea = ea.rename(columns={ESRC: "gt_source", ETGT: "gt_target"})
    return na, ea


def run_crop(name: str, sub: pl.DataFrame, gt_geff: Path):
    g, i2s = build_graph(sub)
    gt = load_graph(gt_geff)
    er = evaluate(g, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    rec = node_recall(g, gt) if g.num_edges() and g.num_nodes() else 0.0
    crow = {"dataset": name, **per_sample_metrics(er, estimated_nodes(gt_geff), rec)}

    # ---- predicted nodes -----------------------------------------------------
    na = g.node_attrs(attr_keys=[NID, MID, "t", "z", "y", "x"]).to_pandas()
    na = na.rename(columns={NID: "internal", MID: "gt_id"})
    na["node_id"] = na.internal.map(i2s)
    na["gt_id"] = na.gt_id.fillna(-1).astype("int64")
    na.insert(0, "dataset", name)

    # ---- predicted edges (the SCORED frame, post filter/dedup/outdeg cap) -----
    ef = _evaluate_matched_graph(g, gt).to_pandas()
    ef = ef.rename(columns={ESRC: "internal_src", ETGT: "internal_tgt", MASK: "matched"})
    ef["source_id"] = ef.internal_src.map(i2s)
    ef["target_id"] = ef.internal_tgt.map(i2s)
    m = na.set_index("internal")
    ef["gt_src"] = ef.internal_src.map(m.gt_id)
    ef["gt_tgt"] = ef.internal_tgt.map(m.gt_id)
    for c in ("t", "z", "y", "x"):
        ef[f"s_{c}"] = ef.internal_src.map(m[c])
        ef[f"t_{c}"] = ef.internal_tgt.map(m[c])
    ef.insert(0, "dataset", name)
    keep = ["dataset", "source_id", "target_id", "matched", "pred_valid",
            "gt_src", "gt_tgt", "s_t", "s_z", "s_y", "s_x", "t_t", "t_z", "t_y", "t_x"]
    ef = ef[keep]

    # ---- ground-truth edges ---------------------------------------------------
    gn, ge = gt_tables(gt)
    detected = set(na.loc[na.gt_id != -1, "gt_id"].astype(int).tolist())
    tp_pairs = set(map(tuple, ef.loc[ef.matched, ["gt_src", "gt_tgt"]].astype("int64").values))
    ge["is_tp"] = [(int(a), int(b)) in tp_pairs for a, b in
                   zip(ge.gt_source.values, ge.gt_target.values)]
    ge["src_detected"] = ge.gt_source.isin(detected)
    ge["tgt_detected"] = ge.gt_target.isin(detected)
    gpos = gn.set_index("gt_id")
    for c in ("t", "z", "y", "x"):
        ge[f"s_{c}"] = ge.gt_source.map(gpos[c])
        ge[f"t_{c}"] = ge.gt_target.map(gpos[c])
    ge["src_outdeg"] = ge.gt_source.map(gpos.out_degree)
    ge.insert(0, "dataset", name)

    gn.insert(0, "dataset", name)
    gn["detected"] = gn.gt_id.isin(detected)

    # ---- reconciliation guard -------------------------------------------------
    assert int(ef.matched.sum()) == er.edge_tp, (name, ef.matched.sum(), er.edge_tp)
    assert int(ef.pred_valid.sum()) - int(ef.matched.sum()) == er.edge_fp, name
    assert len(ge) - int(ef.matched.sum()) == er.edge_fn, name
    return crow, na, ef, ge, gn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--parquet", help="pre-ILP dump with the same schema (no 'id' column)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-crops", type=int)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.parquet:
        df = pl.read_parquet(args.parquet).with_columns(
            pl.col("node_id").fill_null(-1).cast(pl.Int64),
            pl.col("t").fill_null(-1).cast(pl.Int64),
            pl.col("z").fill_null(-1.0), pl.col("y").fill_null(-1.0), pl.col("x").fill_null(-1.0),
            pl.col("source_id").fill_null(-1).cast(pl.Int64),
            pl.col("target_id").fill_null(-1).cast(pl.Int64),
        )
    else:
        df = read_submission(open_csv(Path(args.csv)))
    names = sorted(df["dataset"].unique().to_list())
    if args.max_crops:
        names = names[: args.max_crops]
    print(f"{len(names)} crops")

    crows, nas, efs, ges, gns = [], [], [], [], []
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            print(f"  skip {name}")
            continue
        c, na, ef, ge, gn = run_crop(name, df.filter(pl.col("dataset") == name), gt_geff)
        crows.append(c); nas.append(na); efs.append(ef); ges.append(ge); gns.append(gn)
        print(f"  [{i}/{len(names)}] {name} adjJ={c['adj_edge_jaccard']:.4f} "
              f"tp={c['edge_tp']} fp={c['edge_fp']} fn={c['edge_fn']}", flush=True)

    pd.DataFrame(crows).to_parquet(out / f"crops_{args.tag}.parquet")
    pd.concat(nas, ignore_index=True).to_parquet(out / f"nodes_{args.tag}.parquet")
    pd.concat(efs, ignore_index=True).to_parquet(out / f"edges_{args.tag}.parquet")
    pd.concat(ges, ignore_index=True).to_parquet(out / f"gtedges_{args.tag}.parquet")
    pd.concat(gns, ignore_index=True).to_parquet(out / f"gtnodes_{args.tag}.parquet")
    print(f"wrote {out}/*_{args.tag}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
