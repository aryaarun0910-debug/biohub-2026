"""Division Oracle C — exact topology-constrained, authoritatively-scored division ceiling.

GT-INFORMED OFFLINE UPPER BOUND. Never submittable. Never mutates the canonical E0c cache.
Starts from every cached E0c graph, edits EDGES ONLY (no nodes added -> node-count penalty
unchanged), and scores with the authoritative metric.

Modes:
  baseline    : unmodified reconstruction (sanity: must reproduce E0c 0.7595 / 0.6484).
  add_only    : conflict-free add-only forks (add the two daughter edges only when the
                mother stays <=2 children AND neither daughter has a conflicting parent).
  add_replace : conflict-aware add-or-replace (force the reachable fork: replace the
                mother's wrong children, remove daughters' conflicting parents, resolve
                daughter competition between forks).

For each GT division (mother out_degree>=2): map GT mother + 2 GT daughters to predicted
E0c nodes via scorer-consistent DistanceMatching; construct the fork under global
constraints (<=1 parent/daughter, <=2 children/mother, distinct daughters, forward time,
no cycles, daughter-competition resolution). Authoritative scorer recomputes matching +
division semantics. ±1-frame tolerance is handled by the division metric.

Go/no-go (commander): GREEN exact >= +0.015 both folds; AMBER +0.008..+0.015 either;
RED < +0.008 either.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
ORACLE_DIR = CACHE / "GT_ORACLE_do_not_submit"
E0C = {0: (0.7595, 0.7595), 1: (0.6484, 0.6490)}  # (edge-J, composite)


def cached_crops(split: int):
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def build_forks(nodes_df, edges, gt_to_sub, gt_div, mode):
    """Edit the renumbered-id edge set per mode under global constraints. Returns edited
    edge list + stats. edges: list of (s,t). gt_div: {M: (C1,C2)}."""
    edge_set = set((int(s), int(t)) for s, t in edges)
    parents: dict[int, set] = {}
    children: dict[int, set] = {}
    for s, t in edge_set:
        children.setdefault(s, set()).add(t)
        parents.setdefault(t, set()).add(s)
    st = {"reachable": 0, "added": 0, "replaced": 0, "skipped": 0, "conflicts": 0, "steals": 0}
    if mode == "baseline":
        return list(edge_set), st
    used = set()
    for M, (C1, C2) in gt_div.items():
        pM, pC1, pC2 = gt_to_sub.get(M), gt_to_sub.get(C1), gt_to_sub.get(C2)
        if pM is None or pC1 is None or pC2 is None or pC1 == pC2:
            st["skipped"] += 1; continue
        st["reachable"] += 1
        if pC1 in used or pC2 in used:
            st["conflicts"] += 1; st["skipped"] += 1; continue
        kids = children.get(pM, set())
        if mode == "add_only":
            confl = any(p != pM for pc in (pC1, pC2) for p in parents.get(pc, set()))
            if confl or len(kids | {pC1, pC2}) > 2:
                st["skipped"] += 1; continue
            for pc in (pC1, pC2):
                if (pM, pc) not in edge_set:
                    edge_set.add((pM, pc)); children.setdefault(pM, set()).add(pc); parents.setdefault(pc, set()).add(pM)
            used |= {pC1, pC2}; st["added"] += 1
        else:  # add_replace
            for k in list(kids):
                if k not in (pC1, pC2):
                    edge_set.discard((pM, k)); children[pM].discard(k); parents.get(k, set()).discard(pM); st["replaced"] += 1
            for pc in (pC1, pC2):
                for s in list(parents.get(pc, set())):
                    if s != pM:
                        edge_set.discard((s, pc)); parents[pc].discard(s); children.get(s, set()).discard(pc); st["steals"] += 1
                edge_set.add((pM, pc)); children.setdefault(pM, set()).add(pc); parents.setdefault(pc, set()).add(pM)
            used |= {pC1, pC2}; st["added"] += 1
    return list(edge_set), st


def score_one(args) -> dict:
    split, crop, mode, materialize = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph, score_pred_graph

    df = pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
    nodes = df.filter(pl.col("row_type") == "node").sort("node_id")
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))

    # scorer-consistent match: renumbered sub id -> gt id
    g = td.graph.InMemoryGraph()
    for k in ["z", "y", "x"]:
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                                 for t, z, y, x in zip(nodes["t"], nodes["z"], nodes["y"], nodes["x"])])
    sub_ids = nodes["node_id"].to_list()
    sub_to_int = {int(s): internal[k] for k, s in enumerate(sub_ids)}
    if edges:
        g.bulk_add_edges([{"source_id": sub_to_int[s], "target_id": sub_to_int[t]} for s, t in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    int_to_gt = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID] for r in na.iter_rows(named=True)}
    gt_to_sub = {}
    for s, iid in sub_to_int.items():
        mid = int_to_gt.get(iid)
        if mid not in (None, -1):
            gt_to_sub[int(mid)] = s  # optimal 1-1 -> last write fine

    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    gt_div = {}
    for n in ids:
        if outdeg[n] >= 2:
            ch = [int(c) for c in gt.successors(int(n))][:2]
            if len(ch) == 2:
                gt_div[int(n)] = (ch[0], ch[1])

    new_edges, st = build_forks(nodes, edges, gt_to_sub, gt_div, mode)
    # rebuild submission rows (nodes unchanged) + edited edges -> authoritative score
    node_rows = nodes.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    erows = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                           "x": -1.0, "source_id": s, "target_id": t} for s, t in new_edges]) if new_edges else node_rows.head(0)
    out = pl.concat([node_rows, erows]) if erows.height else node_rows
    if materialize and mode != "baseline":
        (ORACLE_DIR / mode / str(split)).mkdir(parents=True, exist_ok=True)
        out.write_parquet(ORACLE_DIR / mode / str(split) / f"{crop}.parquet")
    from biotrack.submission import submission_to_graphs
    graph = submission_to_graphs(out.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
    row = score_pred_graph(graph, str(ROOT / "data" / "train" / f"{crop}.geff"))
    return {"split": split, "crop": crop, **{k: row[k] for k in
            ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
             "node_recall", "num_pred_nodes")}, **{f"fk_{k}": v for k, v in st.items()}}


def aggregate(res, mode):
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise
    print(f"\n########## ORACLE-C  mode={mode} ##########")
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        rows, fk = [], {"reachable": 0, "added": 0, "replaced": 0, "skipped": 0, "conflicts": 0, "steals": 0}
        npred = 0
        for c in cached_crops(fold):
            r = res[(fold, c)]
            er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"], r["division_tp"],
                                  r["division_fp"], r["division_fn"], r["num_pred_nodes"])
            n_est = estimated_nodes(str(ROOT / "data" / "train" / f"{c}.geff"))
            rows.append(per_sample_metrics(er, n_est, r["node_recall"]))
            npred += r["num_pred_nodes"]
            for k in fk:
                fk[k] += r[f"fk_{k}"]
        s = summarise(rows)
        e0e, e0c = E0C[fold]
        print(f"  {fam}: edgeJ={s['edge_jaccard']:.4f} (dvs {e0e:+.4f}) divJ={s['division_jaccard']:.4f} "
              f"(TP{s['division_tp']}/FP{s['division_fp']}/FN{s['division_fn']}) "
              f"composite={s['score']:.4f} (dvs E0c {s['score']-e0c:+.4f})  N_pred={npred}")
        print(f"     forks: reachable={fk['reachable']} added={fk['added']} replaced={fk['replaced']} "
              f"skipped={fk['skipped']} conflicts={fk['conflicts']} steals={fk['steals']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--materialize", action="store_true")
    a = ap.parse_args()
    (ORACLE_DIR).mkdir(parents=True, exist_ok=True)
    (ORACLE_DIR / "README.txt").write_text(
        "GT-INFORMED ORACLE GRAPHS — OFFLINE UPPER BOUND ONLY. NEVER SUBMIT / NEVER PACKAGE.\n")
    for mode in ("baseline", "add_only", "add_replace"):
        tasks = [(s, c, mode, a.materialize) for s in (0, 1) for c in cached_crops(s)]
        res = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(score_one, tasks):
                res[(r["split"], r["crop"])] = r
        aggregate(res, mode)


if __name__ == "__main__":
    main()
