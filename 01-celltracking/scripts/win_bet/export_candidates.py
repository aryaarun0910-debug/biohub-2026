"""Phase B — export the full pre-assignment candidate surface + scorer-consistent labels.

For each OOF crop, runs the parity-verified wrapper with the candidate sink to capture
EVERY within-gate motion_relink candidate (source/target, raw, motion, edge_prob,
composite cost, tight/relaxed pass, per-source density+cost-rank, wrapper-selected),
then labels each via the scorer's optimal predicted-node -> GT matching: positive only
if BOTH endpoints match GT nodes AND the corresponding GT edge exists (nearest-GT would
be false supervision). The wrapper's composite decision (`selected`, `cost`) is the
baseline the breadth model must beat at the graph level, not edge_prob alone.

Output: artifacts/kaggle/phaseB_candidates/candidates_split_{0,1}.parquet
"""
from __future__ import annotations

import copy
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402
import tracksdata as td  # noqa: E402
from tracksdata.metrics import DistanceMatching  # noqa: E402

from biotrack import wrapper as W  # noqa: E402
from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph  # noqa: E402

OUT = ROOT / "artifacts" / "kaggle" / "phaseB_candidates"


def build_nodes_edges(graph):
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {"node_id": nid, "t": int(row["t"]),
                            "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    raw_edges: list[dict] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        ep = row.get("edge_prob")
        raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                          "edge_prob": None if ep is None else float(ep)})
    return nodes_by_id, raw_edges


def match_pred_to_gt(nodes_by_id: dict, gt_geff: str):
    """Optimal pred-node -> GT-node matching (scorer-consistent). Returns
    raw_pred_id -> matched_GT_node_id (or None) and the set of GT edges."""
    gt = load_graph(gt_geff)
    g = td.graph.InMemoryGraph()
    for k in ["z", "y", "x"]:
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = sorted(nodes_by_id)
    internal = g.bulk_add_nodes([
        {"t": int(nodes_by_id[i]["t"]), "z": float(nodes_by_id[i]["z"]),
         "y": float(nodes_by_id[i]["y"]), "x": float(nodes_by_id[i]["x"])}
        for i in ids
    ])
    raw_to_internal = {rid: internal[k] for k, rid in enumerate(ids)}
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    internal_to_gt = {}
    for row in na.iter_rows(named=True):
        mid = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        internal_to_gt[int(row[td.DEFAULT_ATTR_KEYS.NODE_ID])] = (
            None if mid is None or int(mid) == -1 else int(mid))
    raw_to_gt = {rid: internal_to_gt.get(raw_to_internal[rid]) for rid in ids}
    gt_edges = set()
    if gt.num_edges() > 0:
        ea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        for r in ea.iter_rows(named=True):
            gt_edges.add((int(r[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]), int(r[td.DEFAULT_ATTR_KEYS.EDGE_TARGET])))
    return raw_to_gt, gt_edges


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # deployment-exact (E0c) config
    W.OUTPUT_MIN_TRACK_LEN = 7
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_REFINE_SYNTHETIC = True
    W.TEST_DIR = ROOT / "data" / "train"

    for S in (0, 1):
        pred_dir = ROOT / f"artifacts/kaggle/oof_clean/pred_geffs_split_{S}"
        rows = []
        crops = sorted(pred_dir.glob("*.geff"))
        for n, pg in enumerate(crops, 1):
            ds = pg.stem
            gt_geff = ROOT / "data" / "train" / f"{ds}.geff"
            if not gt_geff.exists():
                continue
            g = W.graph_from_geff(pg)
            nbi, raw = build_nodes_edges(g)
            raw_to_gt, gt_edges = match_pred_to_gt(nbi, str(gt_geff))
            W._CANDIDATE_SINK = []
            W.filter_output_graph(copy.deepcopy(nbi), raw, dataset=ds)
            cands = W._CANDIDATE_SINK
            W._CANDIDATE_SINK = None

            by_src = defaultdict(list)
            for c in cands:
                by_src[c["source_id"]].append(c)
            for sid, lst in by_src.items():
                for rank, c in enumerate(sorted(lst, key=lambda c: c["cost"])):
                    s_gt, t_gt = raw_to_gt.get(c["source_id"]), raw_to_gt.get(c["target_id"])
                    label = int(s_gt is not None and t_gt is not None and (s_gt, t_gt) in gt_edges)
                    sn, tn = nbi[c["source_id"]], nbi[c["target_id"]]
                    rows.append({
                        "dataset": ds, "t": int(sn["t"]),
                        "source_id": c["source_id"], "target_id": c["target_id"],
                        "raw_um": c["raw_um"], "motion_um": c["motion_um"], "edge_prob": c["edge_prob"],
                        "cost": c["cost"], "pass": c["pass"], "gate_um": c["gate_um"],
                        "selected": c["selected"], "cost_rank": rank, "n_cand": len(lst),
                        "sz": float(sn["z"]), "sy": float(sn["y"]), "sx": float(sn["x"]),
                        "tz": float(tn["z"]), "ty": float(tn["y"]), "tx": float(tn["x"]),
                        "label": label,
                    })
            if n % 20 == 0:
                print(f"  split {S}: {n}/{len(crops)} crops, {len(rows)} cand rows")

        df = pl.DataFrame(rows)
        # dedup a (source,target) that appears in both passes: keep selected / lower cost
        df = df.sort(["source_id", "target_id", "selected", "cost"],
                     descending=[False, False, True, False]).unique(
            subset=["dataset", "source_id", "target_id"], keep="first")
        out = OUT / f"candidates_split_{S}.parquet"
        df.write_parquet(out)

        fam = df["dataset"][0].split("_")[0]
        n = df.height
        npos = int(df["label"].sum())
        # candidate recall: sources whose true target is present in the pool
        src_has_pos = df.group_by("dataset", "source_id").agg(pl.col("label").max().alias("hp"))
        recall = float(src_has_pos["hp"].mean())
        # wrapper composite top-1: among selected edges, precision vs label
        sel = df.filter(pl.col("selected") == 1)
        sel_prec = float(sel["label"].mean()) if sel.height else float("nan")
        print(f"\n=== split {S} held-out {fam}: {n} cands ({npos} pos, {100*npos/n:.1f}%) -> {out.name} ===")
        print(f"  candidate recall (source has true target in pool) = {recall:.4f}")
        print(f"  wrapper-selected edges = {sel.height}, precision vs label = {sel_prec:.4f}")


if __name__ == "__main__":
    main()
