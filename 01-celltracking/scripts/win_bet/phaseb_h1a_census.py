"""H1a — canonical candidate census over the frozen H0c top-3 shortlist.

Builds a partitioned, provenance-preserving Parquet dataset (one file per fold/crop, never one
enormous CSV) and reports the census the H1 model will be trained on.

The candidate surface is IMMUTABLE and imported from the frozen H0c replay
(`phaseb_h0c_replay.shortlist`, config hash 04eeac97500d): 15/15 geometric generation, kNN flow,
flow-midpoint top-K=3 per mother. Nothing is retuned here.

LABELS — this is a positive/unlabeled problem, not positive/negative. Annotation is sparse, so
an unannotated mother is NOT evidence of "no division":
  positive           mother matches a GT divider AND this pair is its true daughter pair
  reliable_negative  mother matches an annotated GT node that continues (out-degree 1), OR
                     matches a GT divider but this is the wrong pair (truth is known)
  unlabeled          mother unmatched, or matched to a GT node with out-degree 0 (annotation
                     ends, so division cannot be ruled out)
Training and primary calibration must use scorer-reliable labels only.

`metric_visible` marks candidates whose mother matches a GT node with out-degree >= 1 -- only
these can ever contribute a division FP, so they form the true precision denominator.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h1a_census.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import CACHE, cached_crops, load_e0c_tables  # noqa: E402
from phaseb_h0c_replay import CFG, CFG_HASH, shortlist  # noqa: E402

TABLE = CACHE / "fork_candidates" / "h0c_top3"
OUT_JSON = ROOT / "reports/inventory/phaseb_h1a_census.json"
EXPECTED_ROWS = 14_371_002


def census_one(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, n_pre = shortlist(sub, t, pos, edges)              # GT-FREE, frozen
    t_of = {int(s): int(a) for s, a in zip(sub, t)}
    parent_of: dict[int, int] = {}
    for a, b in edges:
        parent_of[int(b)] = int(a)                            # E0c has <=1 parent per node

    # ---- GT only after generation, for labels/visibility only ---------------
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    gt = load_graph(gt_geff)
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                                 for a, z, y, x in zip(nd["t"], nd["z"], nd["y"], nd["x"])])
    s2i = {int(s): internal[i] for i, s in enumerate(nd["node_id"].to_list())}
    if edges:
        g.bulk_add_edges([{"source_id": s2i[a], "target_id": s2i[b]} for a, b in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    i2g = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
           for r in na.iter_rows(named=True)}
    sub_to_gt, gt_to_sub = {}, {}
    for s, iid in s2i.items():
        m = i2g.get(iid)
        if m not in (None, -1):
            sub_to_gt[int(s)] = int(m)
            gt_to_sub[int(m)] = int(s)
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    true_pair: dict[int, tuple[int, int]] = {}
    reachable = 0
    for n in ids:
        if outdeg.get(n, 0) < 2:
            continue
        ch = [int(c) for c in gt.successors(int(n))][:2]
        if len(ch) != 2:
            continue
        pM, p1, p2 = gt_to_sub.get(n), gt_to_sub.get(ch[0]), gt_to_sub.get(ch[1])
        if pM is None or p1 is None or p2 is None or p1 == p2:
            continue
        reachable += 1
        true_pair[pM] = (min(p1, p2), max(p1, p2))

    rows = []
    cnt = Counter()
    for m, ranked in prop.items():
        gm = sub_to_gt.get(m)
        od = outdeg.get(gm, -1) if gm is not None else -1
        visible = gm is not None and od >= 1
        tp = true_pair.get(m)
        for rank, (resid, (d1, d2)) in enumerate(ranked):
            if tp is not None and (d1, d2) == tp:
                lab = "positive"
            elif gm is not None and (od == 1 or tp is not None):
                lab = "reliable_negative"
            else:
                lab = "unlabeled"
            cnt[lab] += 1
            if visible:
                cnt["metric_visible"] += 1
            p1, p2 = parent_of.get(d1, -1), parent_of.get(d2, -1)
            steal = int((p1 not in (-1, m)) or (p2 not in (-1, m)))
            cnt["steal_required"] += steal
            rows.append({
                "crop": crop, "family": crop.split("_")[0], "fold": split,
                "t": t_of[m], "mother": m, "d1": d1, "d2": d2, "rank": rank,
                "cand_id": f"{crop}:{m}:{d1}:{d2}",
                "flow_midpoint_residual": resid,
                "label": lab, "metric_visible": bool(visible),
                "mother_gt_outdeg": int(od),
                "competing_parent_d1": p1, "competing_parent_d2": p2,
                "steal_required": bool(steal),
                "cfg_hash": CFG_HASH,
            })

    (TABLE / str(split)).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(TABLE / str(split) / f"{crop}.parquet")
    pm = np.asarray([len(v) for v in prop.values()]) if prop else np.array([0])
    return {"split": split, "crop": crop, "rows": len(rows), "mothers": len(prop),
            "reachable": reachable, "cand_pre_topk": n_pre,
            "positives": cnt["positive"], "reliable_negatives": cnt["reliable_negative"],
            "unlabeled": cnt["unlabeled"], "metric_visible": cnt["metric_visible"],
            "steal_required": cnt["steal_required"],
            "pm_mean": float(pm.mean()), "pm_max": int(pm.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    print(f"H1a census over frozen H0c surface (cfg hash {CFG_HASH}): {CFG}")
    out, grand = [], Counter()
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(census_one, [(fold, c) for c in cached_crops(fold)]))
        agg = Counter()
        for r in res:
            for k in ("rows", "mothers", "reachable", "positives", "reliable_negatives",
                      "unlabeled", "metric_visible", "steal_required", "cand_pre_topk"):
                agg[k] += r[k]
        grand.update(agg)
        vis = agg["metric_visible"]
        print(f"\n########## H1a census  {fam} ##########")
        print(f"  rows={agg['rows']:,} (pre-topK {agg['cand_pre_topk']:,})  mothers={agg['mothers']:,}  "
              f"crops={len(res)}")
        print(f"  positives={agg['positives']:,}  reliable_negatives={agg['reliable_negatives']:,}  "
              f"unlabeled={agg['unlabeled']:,} ({agg['unlabeled']/max(agg['rows'],1):.4f})")
        print(f"  METRIC-VISIBLE={vis:,} ({vis/max(agg['rows'],1):.5f} of shortlist)  "
              f"-> positive base rate among visible = {agg['positives']/max(vis,1):.6f}")
        print(f"  steal_required={agg['steal_required']:,}  reachable divisions={agg['reachable']}  "
              f"positives/reachable={agg['positives']}/{agg['reachable']}")
        print(f"  candidates per mother: mean={agg['rows']/max(agg['mothers'],1):.2f} "
              f"max={max(r['pm_max'] for r in res)}")
        out.append({"family": fam, "fold": fold, "crops": len(res), **dict(agg)})
    tot = grand["rows"]
    print(f"\n  TOTAL rows={tot:,}  expected={EXPECTED_ROWS:,}  "
          f"parity={'OK' if tot == EXPECTED_ROWS else 'MISMATCH ' + str(tot - EXPECTED_ROWS)}")
    print(f"  TOTAL positives={grand['positives']} (H0c retained 92: 16 + 76)  "
          f"reproduced={'OK' if grand['positives'] == 92 else 'MISMATCH'}")
    print(f"  partitioned table: {TABLE}")
    OUT_JSON.write_text(json.dumps({"config": CFG, "config_hash": CFG_HASH,
                                    "expected_rows": EXPECTED_ROWS, "total_rows": tot,
                                    "row_parity": tot == EXPECTED_ROWS,
                                    "total_positives": grand["positives"],
                                    "h0c_retained_expected": 92,
                                    "table_dir": str(TABLE), "rows": out}, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
