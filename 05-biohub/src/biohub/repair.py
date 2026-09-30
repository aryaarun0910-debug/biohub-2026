"""Stage 5 — repair: close gaps and prune debris, WITHOUT destroying forks.

This stage is where the public field loses its division term. Its motion-relink rebuilds the
edge list from a 1:1 assignment AFTER safe_div has inserted forks, so every fork upstream is
discarded — 111 safe divisions across 8 embryos yielding 3 TP. Nothing here may rebuild the
edge list, and `assert_forks_preserved` makes that a runtime guarantee rather than a comment.

Gap recovery inserts a NODE, never a longer edge. All 7,998 ground-truth edges span exactly one
frame and the scorer DROPS anything else before counting, so a t->t+2 edge is invisible, not
penalised. Gaps are bounded: ~20 one-frame, ~3 two-frame, ~2 longer per sample.
"""
from __future__ import annotations
import numpy as np
from .contracts import Graph, Config, SCALE

def _adj(g: Graph):
    children, parents = {}, {}
    for s, d in g.edges:
        children.setdefault(int(s), []).append(int(d)); parents[int(d)] = int(s)
    return children, parents

def close_gaps(g: Graph, cfg: Config) -> Graph:
    """Bridge a track that ends at t with one that starts at t+k (k<=gap_max_frames) by
    INTERPOLATING the missing node(s), then emitting only one-frame edges."""
    if not len(g.edges): return g
    children, parents = _adj(g)
    P, t = g.um(), g.t
    ends   = [i for i in range(len(t)) if i not in children]          # no outgoing edge
    starts = [i for i in range(len(t)) if i not in parents]           # no incoming edge
    if not ends or not starts: return g

    new_zyx, new_t, new_edges = [], [], []
    used_start = set()
    S = np.array(starts)
    for e in ends:
        dt = t[S] - t[e]
        ok = np.flatnonzero((dt >= 2) & (dt <= cfg.gap_max_frames + 1))
        if not len(ok): continue
        cand = S[ok]
        d = np.linalg.norm(P[cand] - P[e], axis=1)
        # a gap of k frames permits k times the per-frame motion budget
        budget = cfg.motion_gate_um * (t[cand] - t[e])
        viable = np.flatnonzero((d <= budget) & ~np.isin(cand, list(used_start)))
        if not len(viable): continue
        j = cand[viable[np.argmin(d[viable])]]
        used_start.add(int(j))
        k = int(t[j] - t[e])
        prev = int(e)
        for step in range(1, k):                                      # insert k-1 nodes
            frac = step / k
            new_zyx.append(g.zyx[e] + frac * (g.zyx[j] - g.zyx[e]))
            new_t.append(int(t[e]) + step)
            nid = len(t) + len(new_t) - 1
            new_edges.append((prev, nid)); prev = nid
        new_edges.append((prev, int(j)))

    if not new_edges: return g
    zyx = np.vstack([g.zyx, np.array(new_zyx)]) if new_zyx else g.zyx
    tt  = np.concatenate([t, np.array(new_t, int)]) if new_t else t
    sc  = np.concatenate([g.score, np.zeros(len(new_t))]) if g.score is not None and new_t else g.score
    E   = np.vstack([g.edges, np.array(new_edges, int)])
    return Graph(t=tt, zyx=zyx, score=sc, edges=E, edge_prob=None, dataset=g.dataset)

def prune_short(g: Graph, cfg: Config) -> Graph:
    """Drop components shorter than min_track_len — but NEVER one containing a fork.

    EXP-1: 112 of 199 datasets contain no division at all, so a blanket length filter is a
    cheap way to delete the rare thing you are being scored on."""
    if not len(g.edges): return g
    n = len(g.t)
    parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, d in g.edges:
        ra, rb = find(int(s)), find(int(d))
        if ra != rb: parent[ra] = rb
    comp = {}
    for i in range(n): comp.setdefault(find(i), []).append(i)
    od = g.out_degree()
    keep = np.zeros(n, bool)
    for members in comp.values():
        span = g.t[members].max() - g.t[members].min() + 1
        has_fork = bool((od[members] == 2).any())
        if span >= cfg.min_track_len or has_fork:                     # division exemption
            keep[members] = True
    if keep.all(): return g
    remap = -np.ones(n, int); remap[keep] = np.arange(keep.sum())
    m = keep[g.edges[:, 0]] & keep[g.edges[:, 1]]
    return Graph(t=g.t[keep], zyx=g.zyx[keep],
                 score=None if g.score is None else g.score[keep],
                 edges=remap[g.edges[m]], edge_prob=None, dataset=g.dataset)

def assert_forks_preserved(before: Graph, after: Graph, stage: str) -> None:
    """Runtime guarantee. A fork lost in repair is the public field's defining bug."""
    b, a = before.forks(), after.forks()
    if a < b:
        raise AssertionError(f"{stage} destroyed {b - a} fork(s) ({b} -> {a}). "
                             f"repair may not rebuild the edge list.")

def repair(g: Graph, cfg: Config) -> Graph:
    out = close_gaps(g, cfg);  assert_forks_preserved(g, out, "close_gaps")
    pruned = prune_short(out, cfg); assert_forks_preserved(out, pruned, "prune_short")
    return pruned
