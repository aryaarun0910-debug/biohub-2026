"""Graph repair: the stages between the raw assignment and the submission.

All of these shrink the unparented-node pool, which is the thing poisoning the
division candidate pool (9.2% of nodes unparented against ~0.76 true divisions
per film). None of them touch the fork machinery.

Watch the node count: removing ISOLATED nodes and fragments is precision and is
what the deployed pipeline does. Pushing n_pred below n_est is prohibition #1.
Every function here reports how many nodes it removed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

GRID_UM = 1.625


def _components(n_nodes, edges):
    parent = list(range(n_nodes))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, t in edges:
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_nodes)])


def close_gaps(edges, det_t, det_zyx, max_um=10.0, reuse_um=3.2,
               allow_synthetic=False, max_synth_frac=0.05):
    """Rejoin a track ending at t with one starting at t+2.

    Prefers REUSING an unlinked node near the midpoint over inventing one:
    reuse removes an orphan and adds no node, so it helps on both axes.
    """
    edges = [(int(s), int(t)) for s, t in edges]
    pos = det_zyx.astype(np.float64) * GRID_UM
    outd, ind = {}, {}
    for s, t in edges:
        outd[s] = outd.get(s, 0) + 1; ind[t] = ind.get(t, 0) + 1
    by_t = {}
    for i, t in enumerate(det_t):
        by_t.setdefault(int(t), []).append(i)

    new_nodes, added, reused = [], [], 0
    n_base = len(det_t)
    for t in sorted(by_t):
        if t + 2 not in by_t:
            continue
        ends = [i for i in by_t[t] if outd.get(i, 0) == 0]
        starts = [j for j in by_t[t + 2] if ind.get(j, 0) == 0]
        mids = [k for k in by_t.get(t + 1, []) if ind.get(k, 0) == 0 and outd.get(k, 0) == 0]
        if not ends or not starts:
            continue
        D = np.linalg.norm(pos[starts][None] - pos[ends][:, None], axis=-1)
        cost = D.copy(); forbid = D > max_um
        cost[forbid] = 1e6
        r, c = linear_sum_assignment(cost)
        used_mid = set()
        for a, bb in zip(r, c):
            if forbid[a, bb]:
                continue
            e, s = ends[a], starts[bb]
            mid = (pos[e] + pos[s]) / 2
            pick = None
            if mids:
                free = [k for k in mids if k not in used_mid]
                if free:
                    d = np.linalg.norm(pos[free] - mid, axis=1)
                    if d.min() <= reuse_um:
                        pick = free[int(np.argmin(d))]; used_mid.add(pick); reused += 1
            if pick is None:
                if not allow_synthetic or len(new_nodes) >= max_synth_frac * n_base:
                    continue
                pick = n_base + len(new_nodes)
                new_nodes.append((t + 1, mid / GRID_UM))
            added += [(e, pick), (pick, s)]
            outd[e] = 1; ind[s] = 1
            ind[pick] = ind.get(pick, 0) + 1; outd[pick] = outd.get(pick, 0) + 1

    if new_nodes:
        det_t = np.concatenate([det_t, np.array([t for t, _ in new_nodes], det_t.dtype)])
        det_zyx = np.concatenate([det_zyx, np.array([p for _, p in new_nodes], det_zyx.dtype)])
    return edges + added, det_t, det_zyx, {"gap_reused": reused,
                                           "gap_synth": len(new_nodes),
                                           "gap_edges": len(added)}


def prune_and_filter(edges, det_t, det_zyx, min_track_len=6, keep_forks=True,
                     prune_isolated=True):
    """Drop isolated nodes and short weakly-connected components.

    Returns remapped (edges, det_t, det_zyx, keep_index, stats). keep_index lets
    the caller carry any parallel array (probabilities, features) along.
    """
    n = len(det_t)
    edges = [(int(s), int(t)) for s, t in edges]
    if not edges:
        return edges, det_t, det_zyx, np.arange(n), {"removed": 0}
    comp = _components(n, edges)
    sizes = np.bincount(comp, minlength=n)
    outdeg = {}
    for s, _ in edges:
        outdeg[s] = outdeg.get(s, 0) + 1
    fork_comps = {comp[s] for s, d in outdeg.items() if d >= 2} if keep_forks else set()
    incident = np.zeros(n, bool)
    for s, t in edges:
        incident[s] = incident[t] = True

    keep = np.ones(n, bool)
    if prune_isolated:
        keep &= incident
    keep &= (sizes[comp] >= min_track_len) | np.isin(comp, list(fork_comps))
    if not keep.any():
        return edges, det_t, det_zyx, np.arange(n), {"removed": 0}

    idx = np.where(keep)[0]
    remap = -np.ones(n, np.int64); remap[idx] = np.arange(len(idx))
    e2 = [(int(remap[s]), int(remap[t])) for s, t in edges if keep[s] and keep[t]]
    return e2, det_t[idx], det_zyx[idx], idx, {"removed": int(n - len(idx))}
