"""H0c — exact CPU replay of the frozen cascade: 15/15 generation -> flow-midpoint top-3
-> suppress-all -> oracle selection (ceiling only) -> add-replace -> exact patched scorer.

FROZEN 2026-07-31 BEFORE EXECUTION. K and the ranking are inherited from the completed H0b
rank-compression audit and are NOT retuned here:
  generation      parent <=15.0 um, sister <=15.0 um (fixed outer surface, no midpoint veto)
  ranking         flow-midpoint residual |midpoint - (mother + local_flow)|, ascending
  K               3 pairs per mother
  flow            identical kNN estimator, k=16, min_support=4, frame-median then global fallback
  suppression     suppress-all, GT-free deterministic lowest-id child retention
  reconstruction  identical add-replace conflict resolver
  scoring         exact patched scorer

Oracle selection is used ONLY to measure the ceiling: among a mother's top-3 shortlist, a GT
division is admitted iff its true pair survived the ranking. This is a GT-informed upper bound
on the shortlist and is never submittable.

Per-crop baseline (unmodified E0c graph) is scored alongside, so node invariance, edge-J
effect and per-crop regressions are measured rather than assumed.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h0c_replay.py --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
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

OUT_JSON = ROOT / "research/06-knowledge-system/inventory/phaseb_h0c_replay.json"
CFG = {"parent_um": 15.0, "sister_um": 15.0, "rank": "flow_midpoint_residual", "topk": 3,
       "flow": "knn", "knn_k": 16, "knn_min": 4, "suppress": "all_lowest_id",
       "reconstruct": "add_replace"}
CFG_HASH = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:12]
E0C = {0: 0.7595, 1: 0.6490}


def shortlist(sub, t, pos, edges):
    """GT-FREE: 15/15 generation, rank by flow-midpoint residual, keep top-3 per mother."""
    from scipy.spatial import cKDTree
    idx_of_sub = {int(s): i for i, s in enumerate(sub)}
    by_t: dict[int, list[int]] = {}
    for i, a in enumerate(t):
        by_t.setdefault(int(a), []).append(i)
    cont = _continuations(edges, idx_of_sub, t, pos)
    alld = [d for f in cont.values() for d in f[1]]
    gmed = np.median(np.stack(alld), axis=0) if alld else np.zeros(3)

    out: dict[int, list[tuple[int, int]]] = {}
    n_pre = 0
    for tt in sorted(by_t):
        mothers, kids = by_t.get(tt, []), by_t.get(tt + 1, [])
        if not mothers or not kids:
            continue
        tree = cKDTree(pos[kids])
        src, disp = cont.get(tt, ([], []))
        fmed = np.median(np.stack(disp), axis=0) if disp else gmed
        ktree = cKDTree(pos[src]) if len(src) >= CFG["knn_min"] else None
        darr = np.stack(disp) if disp else None
        for mi in mothers:
            D = [kids[j] for j in tree.query_ball_point(pos[mi], CFG["parent_um"])]
            if len(D) < 2:
                continue
            if ktree is not None:
                _, jj = ktree.query(pos[mi], k=min(CFG["knn_k"], len(src)))
                flow = np.median(darr[np.atleast_1d(jj)], axis=0)
            else:
                flow = fmed
            pc = pos[mi] + flow
            scored = []
            for a in range(len(D)):
                for b in range(a + 1, len(D)):
                    i1, i2 = D[a], D[b]
                    if float(np.linalg.norm(pos[i1] - pos[i2])) > CFG["sister_um"]:
                        continue
                    mid = 0.5 * (pos[i1] + pos[i2])
                    s1, s2 = int(sub[i1]), int(sub[i2])
                    scored.append((float(np.linalg.norm(mid - pc)), (min(s1, s2), max(s1, s2))))
            if not scored:
                continue
            n_pre += len(scored)
            scored.sort(key=lambda z: z[0])
            # value is [(residual, (d1, d2)), ...] in rank order; selection is unchanged.
            out[int(sub[mi])] = scored[:CFG["topk"]]
    return out, n_pre


def replay_one(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, n_pre = shortlist(sub, t, pos, edges)          # GT-free
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")

    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")

    def score(edge_set):
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in edge_set])
        o = pl.concat([node_rows, er]) if er.height else node_rows
        g = submission_to_graphs(o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    base = score({(int(a), int(b)) for a, b in edges})

    # ---- GT only after generation ------------------------------------------
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
    from gt_collision import gt_maps_from_matches
    gt_to_sub, _s2g, _ = gt_maps_from_matches(s2i, i2g, context=f"h0c_replay {crop}")
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    reachable, retained = 0, {}
    for n in ids:
        if outdeg[n] < 2:
            continue
        ch = [int(c) for c in gt.successors(int(n))][:2]
        if len(ch) != 2:
            continue
        pM, p1, p2 = gt_to_sub.get(n), gt_to_sub.get(ch[0]), gt_to_sub.get(ch[1])
        if pM is None or p1 is None or p2 is None or p1 == p2:
            continue
        reachable += 1
        if (min(p1, p2), max(p1, p2)) in {k for _, k in prop.get(pM, ())}:
            retained[pM] = (min(p1, p2), max(p1, p2))

    # ---- suppress-all then add-replace over the retained shortlist ----------
    es = {(int(a), int(b)) for a, b in edges}
    par, ch_ = {}, {}
    for a, b in es:
        ch_.setdefault(a, set()).add(b)
        par.setdefault(b, set()).add(a)
    for m in [n for n, k in ch_.items() if len(k) >= 2]:
        keep = min(ch_[m])
        for k in list(ch_[m]):
            if k != keep:
                es.discard((m, k)); ch_[m].discard(k); par.get(k, set()).discard(m)
    steals = 0
    used = set()
    for pM, (p1, p2) in retained.items():
        if p1 in used or p2 in used:
            continue
        for k in list(ch_.get(pM, set())):
            if k not in (p1, p2):
                es.discard((pM, k)); ch_[pM].discard(k); par.get(k, set()).discard(pM)
        for pc in (p1, p2):
            for s in list(par.get(pc, set())):
                if s != pM:
                    es.discard((s, pc)); par[pc].discard(s); ch_.get(s, set()).discard(pc)
                    steals += 1
            es.add((pM, pc)); ch_.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
        used |= {p1, p2}
    treat = score(es)

    return {"split": split, "crop": crop, "reachable": reachable, "retained": len(retained),
            "cand_pre_topk": n_pre, "cand_shortlist": sum(len(v) for v in prop.values()),
            "mothers": len(prop), "steals": steals,
            **{f"b_{k}": base[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                           "division_fp", "division_fn", "node_recall",
                                           "num_pred_nodes")},
            **{f"t_{k}": treat[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                            "division_fp", "division_fn", "node_recall",
                                            "num_pred_nodes")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise
    print(f"H0c frozen config hash {CFG_HASH}: {CFG}")
    rows_out = []
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(replay_one, [(fold, c) for c in cached_crops(fold)]))
        b_rows, t_rows, regress = [], [], []
        for r in res:
            ne = estimated_nodes(str(ROOT / "data" / "train" / f"{r['crop']}.geff"))
            b = per_sample_metrics(EvaluationResult(*[r[f"b_{k}"] for k in
                ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                 "num_pred_nodes")]), ne, r["b_node_recall"])
            t_ = per_sample_metrics(EvaluationResult(*[r[f"t_{k}"] for k in
                ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                 "num_pred_nodes")]), ne, r["t_node_recall"])
            b_rows.append(b); t_rows.append(t_)
            if t_["adj_edge_jaccard"] < b["adj_edge_jaccard"] - 1e-9:
                regress.append((r["crop"], round(t_["adj_edge_jaccard"] - b["adj_edge_jaccard"], 5)))
        sb, st = summarise(b_rows), summarise(t_rows)
        pre = sum(r["cand_pre_topk"] for r in res)
        sl = sum(r["cand_shortlist"] for r in res)
        reach = sum(r["reachable"] for r in res)
        ret = sum(r["retained"] for r in res)
        node_ok = all(r["b_num_pred_nodes"] == r["t_num_pred_nodes"] for r in res)
        rec_ok = all(abs(r["b_node_recall"] - r["t_node_recall"]) < 1e-12 for r in res)
        print(f"\n########## H0c  {fam} ##########")
        print(f"  baseline  composite={sb['score']:.4f} adjEdgeJ={sb['adj_edge_jaccard']:.4f} "
              f"divJ={sb['division_jaccard']:.4f} (TP{sb['division_tp']}/FP{sb['division_fp']}/FN{sb['division_fn']})")
        print(f"  H0c       composite={st['score']:.4f} adjEdgeJ={st['adj_edge_jaccard']:.4f} "
              f"divJ={st['division_jaccard']:.4f} (TP{st['division_tp']}/FP{st['division_fp']}/FN{st['division_fn']})")
        print(f"  DELTA composite={st['score']-sb['score']:+.4f}  vs E0c anchor {st['score']-E0C[fold]:+.4f}  "
              f"adjEdgeJ={st['adj_edge_jaccard']-sb['adj_edge_jaccard']:+.4f}")
        print(f"  retention: {ret}/{reach} reachable = {ret/max(reach,1):.3f}")
        print(f"  candidates: pre-topK={pre:,} shortlist={sl:,} mothers={sum(r['mothers'] for r in res):,} "
              f"steals={sum(r['steals'] for r in res):,}")
        print(f"  node invariance: N_pred identical={node_ok} node_recall identical={rec_ok}")
        print(f"  per-crop adjEdgeJ regressions: {len(regress)}" +
              (f" -> {sorted(regress, key=lambda z: z[1])[:5]}" if regress else ""))
        rows_out.append({"family": fam, "fold": fold, "baseline": sb, "h0c": st,
                         "delta_composite": st["score"] - sb["score"],
                         "delta_vs_e0c_anchor": st["score"] - E0C[fold],
                         "delta_adj_edge_jaccard": st["adj_edge_jaccard"] - sb["adj_edge_jaccard"],
                         "reachable": reach, "retained": ret, "retention": ret / max(reach, 1),
                         "cand_pre_topk": pre, "cand_shortlist": sl,
                         "steals": sum(r["steals"] for r in res),
                         "node_count_invariant": node_ok, "node_recall_invariant": rec_ok,
                         "n_regressed_crops": len(regress), "regressed": regress})
    OUT_JSON.write_text(json.dumps({"config": CFG, "config_hash": CFG_HASH, "rows": rows_out},
                                   indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
