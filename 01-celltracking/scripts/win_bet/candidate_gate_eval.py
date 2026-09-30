r"""Candidate-gate evaluation harness -- size the fix BEFORE building it.

WHY THIS EXISTS
---------------
The measured bottleneck of this pipeline is candidate GENERATION, not the solver: a
perfect solver over today's candidates is worth +0.0012, while 15.65% of detectable GT
edges are never nominated. Every proposal to widen the candidate set therefore has to
answer the same three questions, and this session answered them by hand twice:

  1. CONTAINMENT  -- what fraction of GT edges does the proposed gate contain?
  2. BY CLASS     -- separately for continuations and DIVISIONS, because divisions
                     behave differently (measured 2026-08-25: at k=3 continuations
                     reach 92.7% but division daughters only 75.2%).
  3. COST         -- what does it do to candidate-set size, which the ILP pays for?

Reporting containment pooled across classes hides the division gap, and the division
term is where the largest untouched metric headroom sits (+0.0993 oracle). So this
harness always reports the split.

WHAT IT DOES NOT DO
-------------------
It measures CONTAINMENT (an oracle ceiling), not score. A gate that contains a GT edge
only makes that edge *reachable*; whether it is then selected depends on the ranker and
the solver. Two measured facts bound the interpretation:

  * The deployed nominator emits exactly ONE candidate parent per target
    (measured: in-degree histogram is {1: 2,162,040}), so widening changes the
    reachable set, not the selection.
  * `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"` means the ILP pays nothing to leave a target
    unlinked, so it accepts every feasible candidate. Until that is restored, a wider
    gate is NOT pruned by the solver.

Treat a containment gain as a necessary condition, never as a predicted score gain.

USAGE
-----
    .venv\Scripts\python.exe scripts\win_bet\candidate_gate_eval.py ^
        --preilp C:/temp/preilp_f1_v2/preilp_split1.parquet ^
        --atlas  C:/temp/error_atlas ^
        --suffix pre1 --k 1,2,3,5,10 --out C:/tmp/gate_eval.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# z, y, x microns per voxel -- the deployed geometry.
DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


def _load(preilp: pathlib.Path, atlas: pathlib.Path, suffix: str):
    pre = pd.read_parquet(preilp)
    edges = pre[pre.row_type != "node"]
    nodes = pre[pre.row_type == "node"][["dataset", "node_id", "t", "z", "y", "x"]]
    gt_edges = pd.read_parquet(atlas / f"gtedges_{suffix}.parquet")
    gt_nodes = pd.read_parquet(atlas / f"gtnodes_{suffix}.parquet")
    pred_nodes = pd.read_parquet(atlas / f"nodes_{suffix}.parquet")
    return edges, nodes, gt_edges, gt_nodes, pred_nodes


def _gt_to_pred(pred_nodes: pd.DataFrame) -> dict:
    m = pred_nodes[pred_nodes.gt_id != -1][["dataset", "gt_id", "node_id"]]
    return {(d, int(g)): int(n) for d, g, n in zip(m.dataset, m.gt_id, m.node_id)}


def _candidate_parents(edges: pd.DataFrame) -> dict:
    cand: dict = {}
    for d, s, t in zip(edges.dataset, edges.source_id, edges.target_id):
        cand.setdefault((d, int(t)), set()).add(int(s))
    return cand


def _frame_trees(nodes: pd.DataFrame, scale) -> tuple[dict, dict, dict]:
    """Per (dataset, t) KD-tree over predicted node positions, in microns."""
    sz, sy, sx = scale
    buckets: dict = {}
    pos: dict = {}
    time: dict = {}
    for d, nid, t, z, y, x in zip(
        nodes.dataset, nodes.node_id, nodes.t, nodes.z, nodes.y, nodes.x
    ):
        nid = int(nid)
        p = (z * sz, y * sy, x * sx)
        pos[(d, nid)] = np.asarray(p)
        time[(d, nid)] = int(t)
        buckets.setdefault((d, int(t)), []).append((nid, p))
    trees = {}
    for key, vals in buckets.items():
        ids = np.asarray([v[0] for v in vals])
        pts = np.asarray([v[1] for v in vals])
        trees[key] = (cKDTree(pts), ids, set(ids.tolist()))
    return trees, pos, time


def knn_containment(pairs, trees, pos, time, gt2pred, ks) -> dict:
    """For each (dataset, gt_parent, gt_child): is the parent within the child's kNN?"""
    hits = {k: 0 for k in ks}
    n = 0
    kmax_req = max(ks)
    for dataset, gt_parent, gt_child in pairs:
        pp = gt2pred.get((dataset, gt_parent))
        cp = gt2pred.get((dataset, gt_child))
        if pp is None or cp is None:
            continue
        key = (dataset, time.get((dataset, cp), -1) - 1)
        entry = trees.get(key)
        if entry is None:
            continue
        tree, ids, id_set = entry
        if pp not in id_set:
            continue
        n += 1
        _, idx = tree.query(pos[(dataset, cp)], k=min(kmax_req, len(ids)))
        order = ids[np.atleast_1d(idx)]
        for k in ks:
            if pp in order[:k]:
                hits[k] += 1
    return {
        "n_evaluated": n,
        "hits": {str(k): hits[k] for k in ks},
        "recall": {str(k): round(hits[k] / max(1, n), 4) for k in ks},
    }


def evaluate(preilp, atlas, suffix, ks, scale=DEFAULT_SCALE, control_cap=40000) -> dict:
    edges, nodes, gt_edges, gt_nodes, pred_nodes = _load(preilp, atlas, suffix)
    gt2pred = _gt_to_pred(pred_nodes)
    cand = _candidate_parents(edges)
    trees, pos, time = _frame_trees(nodes, scale)

    indeg = edges.groupby(["dataset", "target_id"]).size().value_counts().sort_index()
    res = {
        "candidate_edges": int(len(edges)),
        "predicted_nodes": int(len(nodes)),
        "indegree_histogram": {int(k): int(v) for k, v in indeg.items()},
        "gt_divisions": int((gt_nodes.out_degree == 2).sum()),
    }

    mothers = set(
        zip(
            gt_nodes[gt_nodes.out_degree == 2].dataset,
            gt_nodes[gt_nodes.out_degree == 2].gt_id.astype(int),
        )
    )
    kids: dict = {}
    for d, s, t in zip(gt_edges.dataset, gt_edges.gt_source, gt_edges.gt_target):
        kids.setdefault((d, int(s)), []).append(int(t))

    div_need, ctrl_need = [], []
    for dataset, gt_mother in mothers:
        daughters = kids.get((dataset, gt_mother), [])
        if len(daughters) != 2:
            continue
        mp = gt2pred.get((dataset, gt_mother))
        if mp is None:
            continue
        for gt_child in daughters:
            cp = gt2pred.get((dataset, gt_child))
            if cp is not None and mp not in cand.get((dataset, cp), set()):
                div_need.append((dataset, gt_mother, gt_child))

    for d, s, t in zip(gt_edges.dataset, gt_edges.gt_source, gt_edges.gt_target):
        if (d, int(s)) in mothers:
            continue
        pp, cp = gt2pred.get((d, int(s))), gt2pred.get((d, int(t)))
        if pp is not None and cp is not None and pp not in cand.get((d, cp), set()):
            ctrl_need.append((d, int(s), int(t)))

    res["division_daughters_needing_nomination_fix"] = len(div_need)
    res["continuation_edges_needing_nomination_fix"] = len(ctrl_need)
    res["divisions"] = knn_containment(div_need, trees, pos, time, gt2pred, ks)
    res["continuations"] = knn_containment(
        ctrl_need[:control_cap], trees, pos, time, gt2pred, ks
    )
    eligible = int(len(nodes))
    res["size_multiplier_vs_today"] = {
        str(k): round(k * eligible / max(1, len(edges)), 2) for k in ks
    }
    return res


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preilp", required=True, type=pathlib.Path)
    ap.add_argument("--atlas", required=True, type=pathlib.Path)
    ap.add_argument("--suffix", default="pre1")
    ap.add_argument("--k", default="1,2,3,5,10")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    res = evaluate(args.preilp, args.atlas, args.suffix, ks)

    print(f"candidate edges     : {res['candidate_edges']:,}")
    print(f"in-degree histogram : {res['indegree_histogram']}")
    print(f"GT divisions        : {res['gt_divisions']:,}")
    print(
        "\nneeding a nomination fix: "
        f"{res['division_daughters_needing_nomination_fix']} division daughters, "
        f"{res['continuation_edges_needing_nomination_fix']:,} continuations"
    )
    print(f"\n{'k':>4} {'divisions':>12} {'continuations':>15} {'size x':>8}")
    for k in ks:
        print(
            f"{k:>4} {res['divisions']['recall'][str(k)]:>12.4f}"
            f" {res['continuations']['recall'][str(k)]:>15.4f}"
            f" {res['size_multiplier_vs_today'][str(k)]:>8.2f}"
        )
    print(
        "\nNOTE: containment is an ORACLE CEILING, not a score. The deployed nominator "
        "emits one parent per target and the ILP appearance weight is 0.0, so a wider "
        "gate is not pruned by the solver -- see module docstring."
    )
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
