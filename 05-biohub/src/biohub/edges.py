"""Stage 3 — score_edges: propose CANDIDATES and give each a probability.

Deliberately does not select. Measured: accumulating candidates to a 0.99 cutoff reaches
0.966 edge recall while greedy top-1 gets 0.942, so the right edges are already in the
candidate set and it is the selector that loses them. Fusing proposal with selection makes
that +0.024 unreachable by construction.

The default scorer here is geometric (a stand-in for the learned pair transformer). It exists
so `resolve` can be developed, gated and measured before any model is trained.
"""
from __future__ import annotations
import numpy as np
from .contracts import Graph, Config

def candidates(g: Graph, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (E, prob) for every (t, t+1) pair inside the motion gate.

    Edges are always t -> t+1: all 7,998 ground-truth edges span exactly one frame and the
    scorer DROPS anything else before counting, so a longer edge is not penalised, it is
    invisible. Gap recovery therefore has to insert a node, never a longer edge.
    """
    P = g.um()
    by_t: dict[int, np.ndarray] = {}
    for t in np.unique(g.t):
        by_t[int(t)] = np.flatnonzero(g.t == t)

    E, pr = [], []
    for t in sorted(by_t):
        a, b = by_t.get(t), by_t.get(t + 1)
        if a is None or b is None or not len(a) or not len(b):
            continue
        D = np.linalg.norm(P[a][:, None, :] - P[b][None, :, :], axis=2)
        ai, bi = np.nonzero(D <= cfg.motion_gate_um)
        if not len(ai):
            continue
        d = D[ai, bi]
        # Monotone decreasing in distance, 1.0 at zero and ~0 at the gate. A learned scorer
        # replaces this wholesale; `resolve` only ever sees a probability.
        p = np.clip(1.0 - (d / cfg.motion_gate_um) ** 2, 1e-6, 1.0)
        E.append(np.stack([a[ai], b[bi]], axis=1)); pr.append(p)

    if not E:
        return np.empty((0, 2), int), np.empty(0)
    return np.concatenate(E), np.concatenate(pr)

def score_edges(g: Graph, cfg: Config) -> Graph:
    E, p = candidates(g, cfg)
    keep = p >= cfg.edge_prob_min
    return Graph(t=g.t, zyx=g.zyx, score=g.score,
                 edges=E[keep], edge_prob=p[keep], dataset=g.dataset)
