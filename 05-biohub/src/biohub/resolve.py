"""Stage 4 — resolve: choose edges from the candidate set, and be ALLOWED TO FORK.

EXP-5, oracle detection over all 199 datasets:
    strict 1:1 Hungarian   division_jaccard 0.0000   tp/fp/fn   0/  0/151
    forking permitted      division_jaccard 0.3727   tp/fp/fn 142/230/  9
    edge-term cost of forking: -0.0002

A 1:1 assignment misses every division BY CONSTRUCTION, and permitting a second child is
essentially free on the edge term. Proposal is therefore solved, and acceptance is what
`_accept_fork` / `_diverges` exist to tighten.

EXP-7 (2026-09-11) swept the two discriminators LEAVE-ONE-EMBRYO-OUT — fit on one embryo,
report on the other, take the min — because 6bba carries 125 of the 151 divisions and the
test set is embryo-disjoint, so a pooled sweep would just fit 6bba:

    cos_max         monotone LOSS on BOTH embryos. Tightening -0.10 -> -0.30 costs 6 TP for
                    0 FP on 6bba. Default is now 1.0, i.e. the angle gate is OFF.
    divergence      the gate that works: >= 0.0um (sisters must merely not RE-CONVERGE)
                    takes 6bba 28 FP -> 18 and 44b6 to zero FP. Tighter (1.0, 2.0um) sheds
                    true divisions far faster than false ones.

    picked on 44b6 -> held-out 6bba 0.3791 | picked on 6bba -> held-out 44b6 0.7308
    chosen by min across embryos: cos<=1.0, div>=0.0  ->  divJ 0.4438 over all 199

RECALL, not precision, is the binding constraint: with every discriminator at its loosest
6bba still shows 67 FN against 28 FP. Those divisions are never PROPOSED as candidates, so
the remaining division headroom lives in score_edges/detect, not here.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import Graph, Config

BIG = 1e6

def _accept_fork(P, parent, first, second, cfg) -> bool:
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
    return True


def _diverges(P, first, second, succ, cfg) -> bool:
    """Sisters must still be separating one frame on. Needs the NEXT frame's assignment,
    so it can only run once every frame is resolved -- not inside the per-frame loop."""
    g1, g2 = succ.get(first), succ.get(second)
    if g1 is None or g2 is None:
        return True                                  # unobservable, do not reject
    sis = np.linalg.norm(P[first] - P[second])
    return np.linalg.norm(P[g1] - P[g2]) >= sis + cfg.fork_divergence_min_um

def resolve(g: Graph, cfg: Config, allow_fork: bool = True) -> Graph:
    if not len(g.edges):
        return g
    P = g.um()
    prob = g.edge_prob if g.edge_prob is not None else np.ones(len(g.edges))
    cand: dict[int, dict[int, float]] = {}
    for (s, d), p in zip(g.edges, prob):
        cand.setdefault(int(g.t[s]), {}).setdefault((int(s), int(d)), float(p))

    chosen: list[tuple[int, int]] = []
    forks: list[tuple[int, int, int]] = []   # (parent, first_child, second) awaiting divergence
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
        forked: set[int] = set()        # a cell divides ONCE: out-degree is capped at 2
        for d in dsts:
            if d in taken:
                continue
            best, best_p = None, 0.0
            for (s, dd), p in pairs.items():
                if dd != d or s not in first or s in forked or p <= best_p:
                    continue
                if _accept_fork(P, s, first[s], d, cfg):
                    best, best_p = s, p
            if best is not None:
                chosen.append((best, d)); taken.add(d); forked.add(best)
                forks.append((best, first[best], d))

    # divergence persistence, now that every frame's successors are known
    if forks:
        succ: dict[int, int] = {}
        for s, d in chosen:
            succ.setdefault(s, d)
        drop = {(p, b) for p, a, b in forks if not _diverges(P, a, b, succ, cfg)}
        if drop:
            chosen = [e for e in chosen if e not in drop]

    E = np.array(sorted(set(chosen)), dtype=int) if chosen else np.empty((0, 2), int)
    return Graph(t=g.t, zyx=g.zyx, score=g.score, edges=E,
                 edge_prob=None, dataset=g.dataset)
