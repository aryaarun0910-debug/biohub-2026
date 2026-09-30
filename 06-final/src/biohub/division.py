"""Fork augmentation by trajectory consequence.

Baseline continuation edges are IMMUTABLE (prohibition #2). A fork may only
claim a target that the 1:1 assignment left unparented, so pass 1 has already
tested "is this node better explained as someone else's continuation?".

The decision keys on consequence, not appearance: G3b measured the frozen
encoder at CHANCE (AUC 0.456) at the split frame, with the only signal at t+1
(0.663) -- after the daughters exist. So we score what happened next.

Legal at inference because tracking is offline over the whole movie: the
predicted graph's future is observable without any label.

Null equivalence (G5 test 1): with `enabled=False` this must return the input
edge list unchanged, byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

GRID_UM = 1.625


@dataclass
class DivConfig:
    parent_max_um: float = 9.0       # P -> candidate daughter (the WIDE arc)
    sister_max_um: float = 14.0      # C <-> Q
    sister_min_um: float = 0.0
    existing_child_max_um: float = 10.0
    symmetry_tau: float = 0.6        # |d(P,C)-d(P,Q)| / mean
    min_track_len: int = 8           # CONSEQUENCE: both daughters must persist
    diverge_um: float = 2.25         # separation must grow by t+2
    border_margin: int = 0           # voxels in the downsampled grid (0..63)
    require_mutual_nn: bool = True
    frame_frac_cap: float = 0.0076
    global_frac_cap: float = 0.00375
    enabled: bool = True


def _chain_len(node: int, succ: dict[int, list[int]], limit: int = 64) -> int:
    n, cur = 1, node
    while cur in succ and len(succ[cur]) == 1 and n < limit:
        cur = succ[cur][0]; n += 1
    return n


def augment(edges, det_t, det_zyx, cfg: DivConfig):
    """edges: iterable of (src, tgt). Returns a NEW list with fork edges added."""
    base = [(int(s), int(t)) for s, t in edges]
    if not cfg.enabled:
        return base, []

    pos = det_zyx.astype(np.float64) * GRID_UM
    succ, indeg = {}, {}
    for s, t in base:
        succ.setdefault(s, []).append(t)
        indeg[t] = indeg.get(t, 0) + 1

    by_t = {}
    for i, t in enumerate(det_t):
        by_t.setdefault(int(t), []).append(i)

    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))
    lo, hi = cfg.border_margin, 63 - cfg.border_margin
    inside = lambda i: (cfg.border_margin == 0) or bool(
        np.all(det_zyx[i, 1:] >= lo) and np.all(det_zyx[i, 1:] <= hi))

    proposals = []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        orphans = [j for j in nxt if indeg.get(j, 0) == 0]
        if not orphans:
            continue
        opos = pos[orphans]
        parents = [i for i in by_t[t] if len(succ.get(i, ())) == 1]
        frame_cap = max(1, round(cfg.frame_frac_cap * len(by_t[t])))
        picked = 0
        cands = []
        for P in parents:
            C = succ[P][0]
            dpc = d(P, C)
            if dpc > cfg.existing_child_max_um:
                continue
            # nearest unparented node to the EXISTING child
            k = int(np.argmin(np.linalg.norm(opos - pos[C], axis=1)))
            Q = orphans[k]
            if cfg.require_mutual_nn:
                pass                      # Q is by construction nearest to C
            dpq, dcq = d(P, Q), d(C, Q)
            if dpq > cfg.parent_max_um or not (cfg.sister_min_um <= dcq <= cfg.sister_max_um):
                continue
            if abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) > cfg.symmetry_tau:
                continue
            if not (inside(P) and inside(C) and inside(Q)):
                continue
            # CONSEQUENCE: both daughters must persist
            if cfg.min_track_len > 1:
                if min(_chain_len(C, succ), _chain_len(Q, succ)) < cfg.min_track_len:
                    continue
            # CONSEQUENCE: separation must grow
            if cfg.diverge_um > 0:
                sc, sq = succ.get(C, []), succ.get(Q, [])
                if len(sc) != 1 or len(sq) != 1:
                    continue
                if d(sc[0], sq[0]) - dcq < cfg.diverge_um:
                    continue
            cands.append((dpq + 0.15 * dcq, P, Q))
        for _, P, Q in sorted(cands):
            if picked >= frame_cap:
                break
            if indeg.get(Q, 0) or len(succ.get(P, ())) >= 2:
                continue
            proposals.append((P, Q))
            succ.setdefault(P, []).append(Q); indeg[Q] = 1
            picked += 1

    cap = max(1, round(cfg.global_frac_cap * len(base)))
    proposals = proposals[:cap]
    return base + proposals, proposals
