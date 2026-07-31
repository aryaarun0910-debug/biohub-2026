"""H0b analysis B — continuous rank-compression audit on the fixed 15/15 outer surface.

Question: can a fixed top-K-per-mother budget retain the reachable true divisions while
shrinking the shortlist to something computationally manageable?

Ranking uses ONLY deployment-observable quantities, each applied as a CONTINUOUS score, never
as a hard veto, and never selected using the candidate's own label:

  midpoint_residual   |pair midpoint - (mother + local_flow)|   (primary)
  parent_midpoint     |pair midpoint - mother|
  sister_separation   |d1 - d2|                                  (continuous, not an 8.5 um veto)
  persistence         -(daughters surviving into t+2)            (fewer survivors ranks worse)
  fwd_support         -(mean forward association prob on the two mother->daughter edges)

Forward association probabilities come from the raw OOF GEFF where the pair exists as a
transformer edge (0.0 otherwise). Reverse-time association, primary/secondary detection and
DeepCenter evidence are NOT available locally -- reverse-time requires the V18 source that is
still blocked at retrieval -- so those feature families are reported as unavailable rather
than silently omitted.

For every reachable true division the rank and percentile under each ranking is recorded, and
top-K retention curves plus shortlist sizes are reported. No threshold is chosen here.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h0b_rankcompress.py --workers 6
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

from phaseb_d0p_proposer import (  # noqa: E402
    CACHE, SCALE, cached_crops, load_e0c_tables, _continuations,
)

OUT_JSON = ROOT / "reports/inventory/phaseb_h0b_rankcompress.json"
SURFACE = {"parent_um": 15.0, "sister_um": 15.0, "flow": "knn", "knn_k": 16, "knn_min": 4}
RANKINGS = ("midpoint_residual", "parent_midpoint", "sister_separation", "persistence", "fwd_support")
K_GRID = (1, 2, 3, 5, 10, 20, 50, 100)


def forward_probs(split: int, crop: str, nd) -> dict[tuple[int, int], float]:
    """Transformer edge probability for (mother_sub, daughter_sub) where the edge exists."""
    from scipy.spatial import cKDTree
    from biotrack import wrapper as W
    pg = ROOT / f"artifacts/kaggle/oof_clean/pred_geffs_split_{split}/{crop}.geff"
    if not pg.exists():
        return {}
    g = W.graph_from_geff(pg)
    raw = {int(r["node_id"]): (int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
           for r in g.node_attrs().iter_rows(named=True)}
    by_t: dict[int, tuple[list, list]] = {}
    for nid, a, z, y, x in zip(nd["node_id"].to_list(), nd["t"].to_list(),
                               nd["z"].to_list(), nd["y"].to_list(), nd["x"].to_list()):
        e = by_t.setdefault(int(a), ([], []))
        e[0].append(int(nid))
        e[1].append([float(z) * SCALE[0], float(y) * SCALE[1], float(x) * SCALE[2]])
    trees = {a: (np.asarray(i), cKDTree(np.asarray(p))) for a, (i, p) in by_t.items()}

    def to_sub(a, z, y, x):
        e = trees.get(a)
        if e is None:
            return None
        ids, tree = e
        d, j = tree.query([z * SCALE[0], y * SCALE[1], x * SCALE[2]], k=1)
        return int(ids[j]) if d <= 2.0 else None

    out = {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, t_ = int(r["source_id"]), int(r["target_id"])
        if s not in raw or t_ not in raw:
            continue
        ms = to_sub(*raw[s])
        ds = to_sub(*raw[t_])
        p = r.get("edge_prob")
        if ms is not None and ds is not None and p is not None:
            out[(ms, ds)] = float(p)
    return out


def audit_one(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from scipy.spatial import cKDTree
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    idx_of_sub = {int(s): i for i, s in enumerate(sub)}
    by_t: dict[int, list[int]] = {}
    for i, a in enumerate(t):
        by_t.setdefault(int(a), []).append(i)

    # daughter persistence into t+2 and forward probs (deployment-observable)
    has_child = set()
    for a, b in edges:
        has_child.add(int(a))
    fwd = forward_probs(split, crop, nd)

    cont = _continuations(edges, idx_of_sub, t, pos)
    all_d = [d for f in cont.values() for d in f[1]]
    global_med = np.median(np.stack(all_d), axis=0) if all_d else np.zeros(3)

    # ---- GT only after generation, for auditing -----------------------------
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
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
    from gt_collision import gt_maps_from_matches
    gt_to_sub, _s2g, _ = gt_maps_from_matches(s2i, i2g, context=f"h0b_rankcompress {crop}")
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    truth = {}
    for n in ids:
        if outdeg[n] >= 2:
            ch = [int(c) for c in gt.successors(int(n))][:2]
            if len(ch) == 2:
                pM, p1, p2 = gt_to_sub.get(n), gt_to_sub.get(ch[0]), gt_to_sub.get(ch[1])
                if pM is not None and p1 is not None and p2 is not None and p1 != p2:
                    truth[pM] = (min(p1, p2), max(p1, p2))

    ranks = {r: [] for r in RANKINGS}
    shortlist = Counter()
    total = 0
    reachable = len(truth)

    for tt in sorted(by_t):
        mothers, kids = by_t.get(tt, []), by_t.get(tt + 1, [])
        if not mothers or not kids:
            continue
        tree = cKDTree(pos[kids])
        src, disp = cont.get(tt, ([], []))
        frame_med = np.median(np.stack(disp), axis=0) if disp else global_med
        ktree = cKDTree(pos[src]) if len(src) >= SURFACE["knn_min"] else None
        darr = np.stack(disp) if disp else None
        for mi in mothers:
            D = [kids[j] for j in tree.query_ball_point(pos[mi], SURFACE["parent_um"])]
            if len(D) < 2:
                continue
            if ktree is not None:
                _, jj = ktree.query(pos[mi], k=min(SURFACE["knn_k"], len(src)))
                flow = np.median(darr[np.atleast_1d(jj)], axis=0)
            else:
                flow = frame_med
            pc = pos[mi] + flow
            feats = []
            for a in range(len(D)):
                for b in range(a + 1, len(D)):
                    i1, i2 = D[a], D[b]
                    if float(np.linalg.norm(pos[i1] - pos[i2])) > SURFACE["sister_um"]:
                        continue
                    s1, s2 = int(sub[i1]), int(sub[i2])
                    key = (min(s1, s2), max(s1, s2))
                    mid = 0.5 * (pos[i1] + pos[i2])
                    ms = int(sub[mi])
                    feats.append((key, {
                        "midpoint_residual": float(np.linalg.norm(mid - pc)),
                        "parent_midpoint": float(np.linalg.norm(mid - pos[mi])),
                        "sister_separation": float(np.linalg.norm(pos[i1] - pos[i2])),
                        "persistence": -float((s1 in has_child) + (s2 in has_child)),
                        "fwd_support": -0.5 * (fwd.get((ms, s1), 0.0) + fwd.get((ms, s2), 0.0)),
                    }))
            if not feats:
                continue
            total += len(feats)
            for K in K_GRID:
                shortlist[K] += min(K, len(feats))
            ms = int(sub[mi])
            tgt = truth.get(ms)
            if tgt is None:
                continue
            for rname in RANKINGS:
                order = sorted(range(len(feats)), key=lambda i: feats[i][1][rname])
                pos_rank = next((r for r, i in enumerate(order) if feats[i][0] == tgt), None)
                if pos_rank is not None:
                    ranks[rname].append((pos_rank, len(feats)))

    return {"split": split, "crop": crop, "reachable": reachable, "cand_total": total,
            **{f"sl_{K}": v for K, v in shortlist.items()},
            **{f"ranks_{r}": ranks[r] for r in RANKINGS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out_rows = []
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        tasks = [(fold, c) for c in cached_crops(fold)]
        res = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(audit_one, tasks):
                res.append(r)
        total = sum(r["cand_total"] for r in res)
        reachable = sum(r["reachable"] for r in res)
        print(f"\n########## H0b rank-compression  {fam} ##########")
        print(f"  surface 15/15 (no midpoint veto): candidates={total:,} reachable divisions={reachable}")
        row = {"family": fam, "fold": fold, "cand_total": total, "reachable": reachable,
               "shortlist": {}, "rankings": {}}
        for K in K_GRID:
            row["shortlist"][K] = sum(r[f"sl_{K}"] for r in res)
        for rname in RANKINGS:
            rk = [x for r in res for x in r[f"ranks_{rname}"]]
            if not rk:
                continue
            arr = np.asarray([p for p, _ in rk])
            pct = np.asarray([p / max(n - 1, 1) for p, n in rk])
            ret = {K: int((arr < K).sum()) for K in K_GRID}
            row["rankings"][rname] = {"n": len(rk), "rank_median": float(np.median(arr)),
                                      "rank_p90": float(np.percentile(arr, 90)),
                                      "pct_median": float(np.median(pct)),
                                      "retention": ret}
            print(f"  {rname:<20} n={len(rk):>3} rank med={np.median(arr):>6.1f} "
                  f"p90={np.percentile(arr,90):>7.1f} | retention " +
                  " ".join(f"K{K}={ret[K]/len(rk):.2f}" for K in (1, 3, 10, 50)))
        print("  shortlist size: " + " ".join(f"K{K}={row['shortlist'][K]:,}" for K in (1, 3, 10, 50)))
        out_rows.append(row)
    OUT_JSON.write_text(json.dumps({"surface": SURFACE, "rankings": list(RANKINGS),
                                    "unavailable_features": ["reverse_time_association",
                                                             "secondary_detection", "deepcenter"],
                                    "rows": out_rows}, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
