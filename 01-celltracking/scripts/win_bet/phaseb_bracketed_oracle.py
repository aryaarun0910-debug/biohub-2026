"""Bracketed-miss endpoint oracle — size the track-conditioned hypothesis-completion lever.

Read-only. For each crop, match the cached E0c graph to GT, find MISSED GT nodes (no
predicted node within 7um), and classify each by whether an existing predicted track
brackets it:
  bracketed  : the GT node's GT predecessor AND a GT successor are BOTH matched (a gap in
               an otherwise-tracked lineage -> recoverable by interpolation + local image
               query; the highest-value, safest completion mode)
  continuation: exactly one temporal side is matched (endpoint continuation mode)
  isolated   : neither side matched (hard; needs de-novo detection)

Also counts GT edges that would be RESTORED by recovering the bracketed nodes (each
recovered node restores its incident GT edges) -> an upper bound on edge-J recoverable by
bracketed completion. This is the number that says whether hypothesis completion is real.
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"


def cached_crops(split: int):
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def analyze(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from biotrack.submission import submission_to_graphs

    gdf = (pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    pred = submission_to_graphs(gdf)[crop]
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    pred.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched_gt = {int(m) for m in na[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list() if m not in (None, -1)}

    gt_ids = [int(n) for n in gt.node_ids()]
    missed = [g for g in gt_ids if g not in matched_gt]
    n_gt_edges = gt.num_edges()
    br = cont = iso = restore_edges = 0
    for g in missed:
        preds = [int(p) for p in gt.predecessors(g)]
        succs = [int(s) for s in gt.successors(g)]
        p_ok = any(p in matched_gt for p in preds)
        s_ok = any(s in matched_gt for s in succs)
        if preds and succs and p_ok and s_ok:
            br += 1
            # recovering g restores its incident GT edges whose OTHER endpoint is matched
            restore_edges += sum(1 for p in preds if p in matched_gt) + sum(1 for s in succs if s in matched_gt)
        elif (preds and p_ok) or (succs and s_ok):
            cont += 1
        else:
            iso += 1
    return {"split": split, "crop": crop, "gt_nodes": len(gt_ids), "matched": len(gt_ids) - len(missed),
            "missed": len(missed), "bracketed": br, "continuation": cont, "isolated": iso,
            "gt_edges": n_gt_edges, "restorable_edges": restore_edges}


def main() -> None:
    tasks = [(s, c) for s in (0, 1) for c in cached_crops(s)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(analyze, tasks))
    df = pl.DataFrame(rows)
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        d = df.filter(pl.col("split") == fold)
        gt_n = int(d["gt_nodes"].sum()); miss = int(d["missed"].sum())
        br = int(d["bracketed"].sum()); cont = int(d["continuation"].sum()); iso = int(d["isolated"].sum())
        gt_e = int(d["gt_edges"].sum()); rest = int(d["restorable_edges"].sum())
        recall = 1 - miss / max(1, gt_n)
        print(f"=== {fam} ===")
        print(f"  GT nodes {gt_n}, node recall {recall:.4f}, missed {miss}")
        print(f"  missed breakdown: bracketed {br} ({100*br/max(1,miss):.1f}%) | "
              f"continuation {cont} ({100*cont/max(1,miss):.1f}%) | isolated {iso} ({100*iso/max(1,miss):.1f}%)")
        print(f"  GT edges {gt_e}; edges restorable by bracketed completion {rest} "
              f"(<= {100*rest/max(1,gt_e):.2f}% of GT edges = loose edge-recall ceiling)")


if __name__ == "__main__":
    main()
