"""The competition metric, implemented locally.

score = adjusted_edge_jaccard + 0.1 * division_jaccard
adj    = max(0, J * (1 - 0.1 * (N_pred - N_est) / N_est))

Division TP follows the connected-component definition: for each true division
source, some anchor component must be hit by BOTH daughter lineages and must
itself contain a predicted fork. An unmatched predicted EDGE is dropped from
consideration; an unmatched predicted DIVISION is penalised. That asymmetry is
why deleting nodes is nearly free on the edge axis and is not free on divisions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

RADIUS_UM, A, DIV_W, GRID_UM = 7.0, 0.1, 0.1, 1.625


def match_nodes(gt_grid, gt_t, det_zyx, det_t):
    p2g, g2p = {}, {}
    for t in np.unique(gt_t):
        gi = np.where(gt_t == t)[0]
        pi = np.where(det_t == t)[0]
        if not len(gi) or not len(pi):
            continue
        d = np.linalg.norm((gt_grid[gi][:, None] - det_zyx[pi][None]) * GRID_UM, axis=-1)
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] <= RADIUS_UM:
                g2p[int(gi[i])] = int(pi[j]); p2g[int(pi[j])] = int(gi[i])
    return p2g, g2p


def _adj(out_map):
    return out_map


def edge_confusion(pred_edges, gt_edges_idx, p2g):
    """gt_edges_idx: (M,2) GT edges as INDEX pairs into the gt arrays."""
    gt_set = {(int(a), int(b)) for a, b in gt_edges_idx}
    gt_out, gt_in = set(), set()
    for a, b in gt_edges_idx:
        gt_out.add(int(a)); gt_in.add(int(b))
    tp, fp, hit = 0, 0, set()
    for s, t in pred_edges:
        ms, mt = p2g.get(int(s)), p2g.get(int(t))
        if ms is not None and mt is not None and (ms, mt) in gt_set:
            tp += 1; hit.add((ms, mt))
        elif (mt is not None and mt in gt_in) or (ms is not None and ms in gt_out):
            fp += 1
    return tp, fp, len(gt_set) - len(hit)


def _components(nodes, edges):
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, t in edges:
        s, t = int(s), int(t)
        if s in parent and t in parent:
            ra, rb = find(s), find(t)
            if ra != rb: parent[ra] = rb
    return {n: find(n) for n in nodes}


def division_confusion(pred_nodes, pred_edges, gt_edges_idx, p2g, g2p):
    gt_out, gt_in = {}, {}
    for a, b in gt_edges_idx:
        gt_out.setdefault(int(a), []).append(int(b))
        gt_in[int(b)] = int(a)
    pred_out = {}
    for s, t in pred_edges:
        pred_out.setdefault(int(s), []).append(int(t))

    comp = _components(list(pred_nodes), pred_edges)
    fork_comps = {comp[n] for n, ch in pred_out.items() if len(ch) >= 2 and n in comp}

    def lineage(g):
        seen, stack = set(), [g]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); stack.extend(gt_out.get(x, []))
        return seen

    tp_sources, fn = set(), 0
    for g, ch in gt_out.items():
        if len(ch) < 2:
            continue
        anchors = {comp[g2p[a]] for a in ([g] + ([gt_in[g]] if g in gt_in else []))
                   if a in g2p and g2p[a] in comp}
        hits = []
        for c in ch[:2]:
            hits.append({comp[g2p[x]] for x in lineage(c) if x in g2p and g2p[x] in comp})
        ok = anchors & hits[0] & hits[1] & fork_comps
        if ok:
            tp_sources.add(g)
        else:
            fn += 1

    fp = 0
    for n, ch in pred_out.items():
        if len(ch) < 2:
            continue
        mg = p2g.get(int(n))
        if mg is not None and mg in gt_out and mg not in tp_sources:
            fp += 1
    return len(tp_sources), fp, fn


def score_film(pred_nodes, pred_edges, det_zyx, det_t, gt_grid, gt_t,
               gt_edges_idx, n_est, n_pred=None):
    p2g, g2p = match_nodes(gt_grid, gt_t, det_zyx, det_t)
    etp, efp, efn = edge_confusion(pred_edges, gt_edges_idx, p2g)
    dtp, dfp, dfn = division_confusion(pred_nodes, pred_edges, gt_edges_idx, p2g, g2p)
    J = etp / max(etp + efp + efn, 1)
    n_pred = len(pred_nodes) if n_pred is None else n_pred
    mult = 1.0 - A * (n_pred - n_est) / n_est
    return {"J_edge": J, "multiplier": mult, "adj_J_edge": max(0.0, J * mult),
            "edge_tp": etp, "edge_fp": efp, "edge_fn": efn,
            "div_tp": dtp, "div_fp": dfp, "div_fn": dfn,
            "n_pred": n_pred, "n_est": float(n_est), "weight": etp + efp + efn}


def aggregate(rows):
    """Edge: weighted by TP+FP+FN. Divisions: micro-averaged (pooled, then ratio)."""
    w = np.array([r["weight"] for r in rows], float)
    adj = float(np.average([r["adj_J_edge"] for r in rows], weights=w)) if w.sum() else 0.0
    tp = sum(r["div_tp"] for r in rows); fp = sum(r["div_fp"] for r in rows)
    fn = sum(r["div_fn"] for r in rows)
    dj = tp / max(tp + fp + fn, 1)
    return {"adj_J_edge": adj, "div_jaccard": dj, "div_tp": tp, "div_fp": fp,
            "div_fn": fn, "score": adj + DIV_W * dj,
            "J_edge": float(np.average([r["J_edge"] for r in rows], weights=w)) if w.sum() else 0.0,
            "multiplier": float(np.average([r["multiplier"] for r in rows], weights=w)) if w.sum() else 1.0}
