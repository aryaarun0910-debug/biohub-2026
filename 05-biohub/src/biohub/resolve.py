"""Stage 4 — resolve: choose edges from the candidate set, and be ALLOWED TO FORK.

EXP-5, oracle detection over all 199 datasets:
    strict 1:1 Hungarian   division_jaccard 0.0000   tp/fp/fn   0/  0/151
    forking permitted      division_jaccard 0.3727   tp/fp/fn 142/230/  9
    edge-term cost of forking: -0.0002

A 1:1 assignment misses every division BY CONSTRUCTION, and permitting a second child is
essentially free on the edge term. Proposal is therefore solved; the binding constraint is
fork PRECISION (230 FP against 142 TP), which is what `_accept_fork` exists to tighten.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import Graph, Config

BIG = 1e6

def _accept_fork(P, parent, first, second, nxt, cfg) -> bool:
    """Discriminators from EXP-1, measured over all 151 ground-truth divisions.

    cos(angle) between the two parent->daughter arcs: median -0.746, IQR -0.914..-0.486.
    Sisters keep diverging: 10.570um at birth -> 13.656um by t+2. False forks do neither.
    """
    a, b = P[first] - P[parent], P[second] - P[parent]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return False
    if float(np.dot(a, b) / (na * nb)) > cfg.fork_cos_max:      # not opposed enough
        return False
    if np.linalg.norm(P[first] - P[second]) > cfg.fork_sister_um:
        return False
    if max(na, nb) > cfg.fork_parent_um:
        return False
    # divergence persistence: the pair must still be separating one frame later
    g1, g2 = nxt.get(first), nxt.get(second)
    if g1 is not None and g2 is not None:
        sis = np.linalg.norm(P[first] - P[second])
        if np.linalg.norm(P[g1] - P[g2]) < sis + cfg.fork_divergence_min_um:
            return False
    return True

def resolve(g: Graph, cfg: Config, allow_fork: bool = True) -> Graph:
    if not len(g.edges):
        return g
    P = g.um()
    prob = g.edge_prob if g.edge_prob is not None else np.ones(len(g.edges))
    cand: dict[int, dict[int, float]] = {}
    for (s, d), p in zip(g.edges, prob):
        cand.setdefault(int(g.t[s]), {}).setdefault((int(s), int(d)), float(p))

    chosen: list[tuple[int, int]] = []
    for t in sorted(cand):
        pairs = cand[t]
        srcs = sorted({s for s, _ in pairs}); dsts = sorted({d for _, d in pairs})
        si = {s: i for i, s in enumerate(srcs)}; di = {d: i for i, d in enumerate(dsts)}
        C = np.full((len(srcs), len(dsts)), BIG)
        for (s, d), p in pairs.items():
            C[si[s], di[d]] = 1.0 - p                       # maximise probability
        ri, ci = linear_sum_assignment(C)
        first: dict[int, int] = {}
        taken: set[int] = set()
        for r, c in zip(ri, ci):
            if C[r, c] < BIG:
                s, d = srcs[r], dsts[c]
                chosen.append((s, d)); first[s] = d; taken.add(d)
        if not allow_fork:
            continue
        # second pass: an unclaimed target may become a parent's SECOND child
        nxt = dict(first)
        for d in dsts:
            if d in taken:
                continue
            best, best_p = None, 0.0
            for (s, dd), p in pairs.items():
                if dd != d or s not in first or p <= best_p:
                    continue
                if _accept_fork(P, s, first[s], d, nxt, cfg):
                    best, best_p = s, p
            if best is not None:
                chosen.append((best, d)); taken.add(d)

    E = np.array(sorted(set(chosen)), dtype=int) if chosen else np.empty((0, 2), int)
    return Graph(t=g.t, zyx=g.zyx, score=g.score, edges=E,
                 edge_prob=None, dataset=g.dataset)
