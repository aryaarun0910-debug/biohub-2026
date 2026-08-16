"""D0' — composed fork-suppression + reconstruction division ceiling (GT-INFORMED ORACLE).

OFFLINE UPPER BOUND. Never submittable. Never mutates the canonical E0c cache. Starts from
every cached E0c graph, edits EDGES ONLY (nodes untouched -> count penalty unchanged), and
scores with the authoritative patched metric.

Oracle C measured add-only / add-replace but never REMOVED an existing false fork, so its
division Jaccard denominator kept all 93 / 584 false forks. This script measures the
composed operation across parity-controlled arms:

  baseline                          reproduce E0c exactly (parity gate)
  suppress_all                      reproduce the patched no-fork ablation (parity gate)
  suppress_all_then_add_replace     reset the fork layer, then rebuild reachable true forks
  selective_suppress_then_add_replace
                                    oracle-remove only FALSE forks, retain true ones, then add
  add_replace_then_selective_suppress
                                    reverse order (operations may interact)

Suppression must reduce a fork to out-degree 1. The retained child is chosen explicitly:
  1. a GT-consistent continuation (mother->child is a real GT edge) when one exists;
  2. otherwise a frozen deterministic rule: lowest renumbered target id.
Both counts are reported, so a GT-assisted retention is never presented as ordinary
suppression. `--fallback-only` disables rule 1 entirely for a GT-free suppression control.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_oracle_d0prime.py --workers 4
"""
from __future__ import annotations

import argparse
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
ORACLE_DIR = CACHE / "GT_ORACLE_do_not_submit"
OUT_JSON = ROOT / "reports/inventory/phaseb_oracle_d0prime.json"

# Historical patched anchors (journal 2026-07-19): (edge_jaccard, composite)
E0C = {0: (0.7595, 0.7595), 1: (0.6484, 0.6490)}
NO_FORK = {0: 0.7595, 1: 0.6482}  # patched no-fork composite

ARMS = (
    "baseline",
    "suppress_all",
    "suppress_all_then_add_replace",
    "selective_suppress_then_add_replace",
    "add_replace_then_selective_suppress",
)

STAT_KEYS = (
    "reachable", "added", "replaced", "skipped", "conflicts", "steals",
    "forks_removed", "fork_edges_removed", "true_forks_lost",
    "retained_gt_consistent", "retained_by_fallback", "readded_after_suppression",
    "forks_retained_true",
)


def cached_crops(split: int):
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def _index(edge_set):
    parents, children = {}, {}
    for s, t in edge_set:
        children.setdefault(s, set()).add(t)
        parents.setdefault(t, set()).add(s)
    return parents, children


def suppress(edge_set, parents, children, ctx, st, *, selective: bool, fallback_only: bool):
    """Reduce forks to out-degree 1. Returns the set of removed (s,t) edges.

    selective=True removes only forks whose mother does not match a GT dividing node.
    """
    sub_to_gt, gt_edges, gt_div = ctx["sub_to_gt"], ctx["gt_edges"], ctx["gt_div"]
    removed = set()
    for m in [n for n, ch in children.items() if len(ch) >= 2]:
        kids = children.get(m, set())
        if len(kids) < 2:
            continue
        gm = sub_to_gt.get(m)
        is_true_fork = gm is not None and gm in gt_div
        if selective and is_true_fork:
            st["forks_retained_true"] += 1
            continue
        # choose the retained child
        keep = None
        if not fallback_only:
            gt_ok = sorted(k for k in kids
                           if gm is not None and sub_to_gt.get(k) is not None
                           and (gm, sub_to_gt[k]) in gt_edges)
            if gt_ok:
                keep = gt_ok[0]
                st["retained_gt_consistent"] += 1
        if keep is None:
            keep = min(kids)  # frozen deterministic rule
            st["retained_by_fallback"] += 1
        for k in list(kids):
            if k == keep:
                continue
            edge_set.discard((m, k))
            children[m].discard(k)
            parents.get(k, set()).discard(m)
            removed.add((m, k))
            st["fork_edges_removed"] += 1
        st["forks_removed"] += 1
        if is_true_fork:
            st["true_forks_lost"] += 1
    return removed


def add_replace(edge_set, parents, children, ctx, st, removed_earlier):
    """Oracle C add-replace: force every reachable GT fork with conflict resolution."""
    gt_to_sub, gt_div = ctx["gt_to_sub"], ctx["gt_div"]
    used = set()
    for M, (C1, C2) in gt_div.items():
        pM, pC1, pC2 = gt_to_sub.get(M), gt_to_sub.get(C1), gt_to_sub.get(C2)
        if pM is None or pC1 is None or pC2 is None or pC1 == pC2:
            st["skipped"] += 1
            continue
        st["reachable"] += 1
        if pC1 in used or pC2 in used:
            st["conflicts"] += 1
            st["skipped"] += 1
            continue
        for k in list(children.get(pM, set())):
            if k not in (pC1, pC2):
                edge_set.discard((pM, k))
                children[pM].discard(k)
                parents.get(k, set()).discard(pM)
                st["replaced"] += 1
        for pc in (pC1, pC2):
            for s in list(parents.get(pc, set())):
                if s != pM:
                    edge_set.discard((s, pc))
                    parents[pc].discard(s)
                    children.get(s, set()).discard(pc)
                    st["steals"] += 1
            if (pM, pc) not in edge_set and (pM, pc) in removed_earlier:
                st["readded_after_suppression"] += 1
            edge_set.add((pM, pc))
            children.setdefault(pM, set()).add(pc)
            parents.setdefault(pc, set()).add(pM)
        used |= {pC1, pC2}
        st["added"] += 1


def build_arm(edges, ctx, arm, fallback_only):
    edge_set = {(int(s), int(t)) for s, t in edges}
    parents, children = _index(edge_set)
    st = dict.fromkeys(STAT_KEYS, 0)
    if arm == "baseline":
        return list(edge_set), st
    if arm == "suppress_all":
        suppress(edge_set, parents, children, ctx, st, selective=False, fallback_only=fallback_only)
    elif arm == "suppress_all_then_add_replace":
        rm = suppress(edge_set, parents, children, ctx, st, selective=False, fallback_only=fallback_only)
        add_replace(edge_set, parents, children, ctx, st, rm)
    elif arm == "selective_suppress_then_add_replace":
        rm = suppress(edge_set, parents, children, ctx, st, selective=True, fallback_only=fallback_only)
        add_replace(edge_set, parents, children, ctx, st, rm)
    elif arm == "add_replace_then_selective_suppress":
        add_replace(edge_set, parents, children, ctx, st, set())
        suppress(edge_set, parents, children, ctx, st, selective=True, fallback_only=fallback_only)
    else:
        raise ValueError(arm)
    return list(edge_set), st


def score_one(args) -> dict:
    split, crop, arm, materialize, fallback_only = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    df = pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
    nodes = df.filter(pl.col("row_type") == "node").sort("node_id")
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    gt = load_graph(gt_geff)

    # scorer-consistent match: renumbered sub id <-> gt id
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                                 for t, z, y, x in zip(nodes["t"], nodes["z"], nodes["y"], nodes["x"])])
    sub_ids = nodes["node_id"].to_list()
    sub_to_int = {int(s): internal[k] for k, s in enumerate(sub_ids)}
    if edges:
        g.bulk_add_edges([{"source_id": sub_to_int[s], "target_id": sub_to_int[t]} for s, t in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    int_to_gt = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
                 for r in na.iter_rows(named=True)}
    from gt_collision import gt_maps_from_matches
    gt_to_sub, sub_to_gt, _ = gt_maps_from_matches(
        sub_to_int, int_to_gt, context=f"oracle_d0prime {crop}")

    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    gt_div = {}
    for n in ids:
        if outdeg[n] >= 2:
            ch = [int(c) for c in gt.successors(int(n))][:2]
            if len(ch) == 2:
                gt_div[int(n)] = (ch[0], ch[1])
    gt_edges = {(int(r[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]), int(r[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]))
                for r in gt.edge_attrs().iter_rows(named=True)}
    ctx = {"gt_to_sub": gt_to_sub, "sub_to_gt": sub_to_gt, "gt_edges": gt_edges, "gt_div": gt_div}

    new_edges, st = build_arm(edges, ctx, arm, fallback_only)

    node_rows = nodes.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    erows = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                           "x": -1.0, "source_id": s, "target_id": t} for s, t in new_edges]) \
        if new_edges else node_rows.head(0)
    out = pl.concat([node_rows, erows]) if erows.height else node_rows
    if materialize and arm != "baseline":
        (ORACLE_DIR / arm / str(split)).mkdir(parents=True, exist_ok=True)
        out.write_parquet(ORACLE_DIR / arm / str(split) / f"{crop}.parquet")
    graph = submission_to_graphs(out.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
    row = score_pred_graph(graph, gt_geff)
    return {"split": split, "crop": crop,
            **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                                   "division_fn", "node_recall", "num_pred_nodes")},
            **{f"fk_{k}": v for k, v in st.items()}}


def aggregate(res, arm) -> list[dict]:
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise
    print(f"\n########## D0'  arm={arm} ##########")
    out = []
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        rows, fk, npred = [], dict.fromkeys(STAT_KEYS, 0), 0
        for c in cached_crops(fold):
            r = res[(fold, c)]
            er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"], r["division_tp"],
                                  r["division_fp"], r["division_fn"], r["num_pred_nodes"])
            rows.append(per_sample_metrics(
                er, estimated_nodes(str(ROOT / "data" / "train" / f"{c}.geff")), r["node_recall"]))
            npred += r["num_pred_nodes"]
            for k in fk:
                fk[k] += r[f"fk_{k}"]
        s = summarise(rows)
        e0e, e0c = E0C[fold]
        delta = s["score"] - e0c
        print(f"  {fam}: adjEdgeJ={s['adj_edge_jaccard']:.4f} edgeJ={s['edge_jaccard']:.4f} "
              f"divJ={s['division_jaccard']:.4f} (TP{s['division_tp']}/FP{s['division_fp']}/"
              f"FN{s['division_fn']}) composite={s['score']:.4f} (dvs E0c {delta:+.4f}) "
              f"recall={s['node_recall']:.4f} N_pred={npred}")
        print(f"     forks: removed={fk['forks_removed']} edges_removed={fk['fork_edges_removed']} "
              f"true_lost={fk['true_forks_lost']} true_retained={fk['forks_retained_true']} "
              f"reachable={fk['reachable']} added={fk['added']} replaced={fk['replaced']} "
              f"steals={fk['steals']} skipped={fk['skipped']} readded={fk['readded_after_suppression']}")
        print(f"     retention: gt_consistent={fk['retained_gt_consistent']} "
              f"fallback={fk['retained_by_fallback']}")
        out.append({"arm": arm, "fold": fold, "family": fam,
                    "adj_edge_jaccard": s["adj_edge_jaccard"], "edge_jaccard": s["edge_jaccard"],
                    "division_jaccard": s["division_jaccard"], "division_tp": s["division_tp"],
                    "division_fp": s["division_fp"], "division_fn": s["division_fn"],
                    "composite": s["score"], "delta_vs_e0c": delta,
                    "node_recall": s["node_recall"], "num_pred_nodes": npred,
                    "forks": dict(fk)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--fallback-only", action="store_true",
                    help="disable GT-consistent retention; deterministic rule only (GT-free control)")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    (ORACLE_DIR / "README.txt").write_text(
        "GT-INFORMED ORACLE GRAPHS - OFFLINE UPPER BOUND ONLY. NEVER SUBMIT / NEVER PACKAGE.\n")

    all_rows = []
    for arm in a.arms.split(","):
        tasks = [(s, c, arm, a.materialize, a.fallback_only) for s in (0, 1) for c in cached_crops(s)]
        res = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(score_one, tasks):
                res[(r["split"], r["crop"])] = r
        all_rows.extend(aggregate(res, arm))

    OUT_JSON.write_text(json.dumps(
        {"fallback_only": a.fallback_only, "e0c_anchor": E0C, "no_fork_anchor": NO_FORK,
         "rows": all_rows}, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")

    # parity gates
    for row in all_rows:
        if row["arm"] == "baseline":
            exp = E0C[row["fold"]][1]
            ok = abs(row["composite"] - exp) < 1e-4
            print(f"PARITY baseline {row['family']}: {row['composite']:.4f} vs {exp:.4f} "
                  f"-> {'OK' if ok else 'MISMATCH'}")
        if row["arm"] == "suppress_all":
            exp = NO_FORK[row["fold"]]
            print(f"PARITY suppress_all {row['family']}: {row['composite']:.4f} vs historical "
                  f"no-fork {exp:.4f} -> diff {row['composite']-exp:+.4f}")


if __name__ == "__main__":
    main()
