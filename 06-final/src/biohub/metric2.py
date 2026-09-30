"""Competition metric on ORIGINAL-voxel coordinates (anisotropic scale).

metric.py assumes the downsampled 64^3 grid, which is isotropic at 1.625 um.
The kernel's prediction graphs use full-resolution voxels, where the scale is
(1.625, 0.40625, 0.40625) -- a 4:1 anisotropy. Using the wrong one silently
inflates every distance in y and x by 4x.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625])
RADIUS_UM, A, DIV_W = 7.0, 0.1, 0.1


def match(pred_t, pred_zyx, gt_t, gt_zyx):
    p2g, g2p = {}, {}
    pu, gu = pred_zyx * SCALE, gt_zyx * SCALE
    for t in np.unique(gt_t):
        gi, pi = np.where(gt_t == t)[0], np.where(pred_t == t)[0]
        if not len(gi) or not len(pi):
            continue
        d = np.linalg.norm(gu[gi][:, None] - pu[pi][None], axis=-1)
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] <= RADIUS_UM:
                g2p[int(gi[i])] = int(pi[j]); p2g[int(pi[j])] = int(gi[i])
    return p2g, g2p


def _components(n, edges):
    par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for s, t in edges:
        ra, rb = find(int(s)), find(int(t))
        if ra != rb:
            par[ra] = rb
    return [find(i) for i in range(n)]


def score(pred_t, pred_zyx, pred_edges, gt_t, gt_zyx, gt_edges, n_est):
    p2g, g2p = match(pred_t, pred_zyx, gt_t, gt_zyx)
    gt_set = {(int(a), int(b)) for a, b in gt_edges}
    gt_out, gt_in = {}, set()
    for a, b in gt_edges:
        gt_out.setdefault(int(a), []).append(int(b)); gt_in.add(int(b))
    tp = fp = 0; hit = set()
    for s, t in pred_edges:
        ms, mt = p2g.get(int(s)), p2g.get(int(t))
        if ms is not None and mt is not None and (ms, mt) in gt_set:
            tp += 1; hit.add((ms, mt))
        elif (mt is not None and mt in gt_in) or (ms is not None and ms in gt_out):
            fp += 1
    fn = len(gt_set) - len(hit)
    J = tp / max(tp + fp + fn, 1)
    mult = 1.0 - A * (len(pred_t) - n_est) / n_est

    comp = _components(len(pred_t), pred_edges)
    pout = {}
    for s, t in pred_edges:
        pout.setdefault(int(s), []).append(int(t))
    forks = {comp[n] for n, ch in pout.items() if len(ch) >= 2}

    def lineage(g):
        seen, st = set(), [g]
        while st:
            x = st.pop()
            if x in seen: continue
            seen.add(x); st.extend(gt_out.get(x, []))
        return seen

    gt_par = {int(b): int(a) for a, b in gt_edges}
    dtp, dfn, tp_src = 0, 0, set()
    for g, ch in gt_out.items():
        if len(ch) < 2: continue
        anch = {comp[g2p[a]] for a in ([g] + ([gt_par[g]] if g in gt_par else [])) if a in g2p}
        hits = [{comp[g2p[x]] for x in lineage(c) if x in g2p} for c in ch[:2]]
        if anch & hits[0] & hits[1] & forks:
            dtp += 1; tp_src.add(g)
        else:
            dfn += 1
    dfp = sum(1 for n, ch in pout.items()
              if len(ch) >= 2 and p2g.get(n) is not None
              and p2g[n] in gt_out and p2g[n] not in tp_src)
    return dict(J_edge=J, multiplier=mult, adj=max(0.0, J * mult),
                etp=tp, efp=fp, efn=fn, dtp=dtp, dfp=dfp, dfn=dfn,
                n_pred=len(pred_t), n_est=n_est, weight=tp + fp + fn)


def aggregate(rows):
    w = np.array([r["weight"] for r in rows], float)
    adj = float(np.average([r["adj"] for r in rows], weights=w)) if w.sum() else 0.0
    dtp = sum(r["dtp"] for r in rows); dfp = sum(r["dfp"] for r in rows)
    dfn = sum(r["dfn"] for r in rows)
    dj = dtp / max(dtp + dfp + dfn, 1)
    return dict(adj=adj, divJ=dj, dtp=dtp, dfp=dfp, dfn=dfn, proxy=adj + DIV_W * dj,
                J=float(np.average([r["J_edge"] for r in rows], weights=w)) if w.sum() else 0.0,
                mult=float(np.average([r["multiplier"] for r in rows], weights=w)) if w.sum() else 1.0)
