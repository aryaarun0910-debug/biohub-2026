r"""Colab-vs-Kaggle parity check for LOEO exports (PKT-0017 follow-on; the p21 pilot).

Two submission-format exports of the SAME configuration on the SAME crops should be identical to
the byte where the pipeline is deterministic, and identical in the scorer's per-crop rows where it
is not. This instrument reports both levels, crop by crop, for the crops common to both files:

  1. structure  - node count, edge count, fork count per crop
  2. content    - node table identity ((t,z,y,x) per node_id) and edge-set identity per crop
  3. scorer     - per-crop rows from two ea_atlas dumps (crops_<tag>.parquet), if given

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\colab_parity_check.py ^
      --a C:\temp\colab_runs\p21-parity-f0-a6\working\sweep_pen_off.csv.gz --a-name colab ^
      --b C:\temp\p20_relink_sweep_f0\sweep_pen_off.csv.gz --b-name kaggle ^
      [--crops-a C:\temp\colab_runs\...\atlas\crops_colab.parquet --crops-b C:\temp\p20_relink_sweep_f0\atlas\crops_pen_off.parquet] ^
      --json-out C:\temp\colab_runs\p21-parity-f0-a6\parity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def structure(df: pd.DataFrame) -> pd.DataFrame:
    nodes = df[df.row_type == "node"].groupby("dataset").size().rename("nodes")
    e = df[df.row_type == "edge"]
    edges = e.groupby("dataset").size().rename("edges")
    forks = (e.groupby(["dataset", "source_id"]).size() > 1).groupby("dataset").sum().rename("forks")
    return pd.concat([nodes, edges, forks], axis=1).fillna(0).astype(int)


def content_identity(a: pd.DataFrame, b: pd.DataFrame, crop: str) -> dict:
    an = a[(a.dataset == crop) & (a.row_type == "node")][["node_id", "t", "z", "y", "x"]].sort_values("node_id").reset_index(drop=True)
    bn = b[(b.dataset == crop) & (b.row_type == "node")][["node_id", "t", "z", "y", "x"]].sort_values("node_id").reset_index(drop=True)
    nodes_same = an.shape == bn.shape and bool((an.values == bn.values).all())
    ae = set(map(tuple, a[(a.dataset == crop) & (a.row_type == "edge")][["source_id", "target_id"]].values.tolist()))
    be = set(map(tuple, b[(b.dataset == crop) & (b.row_type == "edge")][["source_id", "target_id"]].values.tolist()))
    # node ids may be renumbered between runs; compare edges by endpoint coordinates when ids differ
    if not nodes_same and len(an) == len(bn):
        pa = an.set_index("node_id")[["t", "z", "y", "x"]].apply(tuple, axis=1).to_dict()
        pb = bn.set_index("node_id")[["t", "z", "y", "x"]].apply(tuple, axis=1).to_dict()
        ae_c = {(pa.get(s), pa.get(t)) for s, t in ae}
        be_c = {(pb.get(s), pb.get(t)) for s, t in be}
        coords_same = set(pa.values()) == set(pb.values())
        edges_same = ae_c == be_c
    else:
        coords_same = nodes_same
        edges_same = ae == be
    return {"nodes_identical": nodes_same, "node_coords_identical_as_set": coords_same,
            "edges_identical": edges_same, "n_nodes": (int(len(an)), int(len(bn))), "n_edges": (len(ae), len(be))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-name", default="a")
    ap.add_argument("--b", required=True); ap.add_argument("--b-name", default="b")
    ap.add_argument("--crops-a"); ap.add_argument("--crops-b")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    a = pd.read_csv(args.a); b = pd.read_csv(args.b)
    common = sorted(set(a.dataset.unique()) & set(b.dataset.unique()))
    sa, sb = structure(a).loc[common], structure(b).loc[common]
    j = sa.join(sb, lsuffix=f"_{args.a_name}", rsuffix=f"_{args.b_name}")
    same_struct = (sa.values == sb.values).all(axis=1)
    result = {"crops": common, "n_common": len(common), "structure_identical_crops": int(same_struct.sum()),
              "totals": {args.a_name: sa.sum().to_dict(), args.b_name: sb.sum().to_dict()}, "content": {}}
    print(f"common crops {len(common)}: structure identical on {int(same_struct.sum())}")
    print(j.to_string())
    ident = {"nodes": 0, "coords": 0, "edges": 0}
    for c in common:
        ci = content_identity(a, b, c)
        result["content"][c] = ci
        ident["nodes"] += ci["nodes_identical"]; ident["coords"] += ci["node_coords_identical_as_set"]; ident["edges"] += ci["edges_identical"]
    result["content_identical_crops"] = ident
    print(f"content identical crops: node tables {ident['nodes']}/{len(common)}, node coordinate sets {ident['coords']}/{len(common)}, edge sets {ident['edges']}/{len(common)}")
    if args.crops_a and args.crops_b:
        ca = pd.read_parquet(args.crops_a).set_index("dataset").loc[common]
        cb = pd.read_parquet(args.crops_b).set_index("dataset").loc[common]
        cols = ["edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes", "adj_edge_jaccard"]
        d = (ca[cols] - cb[cols])
        result["scorer"] = {"identical_rows": int((d.abs().sum(axis=1) == 0).sum()),
                            "max_abs_adj_edge_delta": float(d["adj_edge_jaccard"].abs().max()),
                            "sum_edge_tp_delta": int(d["edge_tp"].sum()), "sum_edge_fp_delta": int(d["edge_fp"].sum())}
        print("scorer rows:", json.dumps(result["scorer"]))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
