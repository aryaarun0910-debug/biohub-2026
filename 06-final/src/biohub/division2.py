"""Fork augmentation with CONTESTED targets, and correct stage ordering.

Two findings drove this, both measured on all 151 labelled divisions:

  44.4%  fail because the 1:1 assignment already gave the true daughter to
         another parent -- and 53.5% of those incumbents are edges with no
         corresponding ground-truth edge, i.e. simply wrong.
  16%    are destroyed by the short-track filter BEFORE the fork stage runs.
         Same bug class as the public pipeline's motion-relink wiping the ILP's
         forks. Fork FIRST, prune second.

Q22's immutability is relaxed, not discarded: a fork may CONTEST a target only
by strictly out-scoring the incumbent -- shorter arc by a margin AND an
incumbent whose learned probability is weak. This is a challenge with a
strict-improvement test, not joint reassignment. G5 test 3 (dJ_edge >= 0) still
gates acceptance, and null equivalence still holds because with forks disabled
nothing contests anything.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

GRID_UM = 1.625


@dataclass
class DivConfig2:
    parent_max_um: float = 12.0        # the WIDE daughter arc (autopsy: 9.0 was cutting real ones)
    sister_max_um: float = 16.0
    existing_child_max_um: float = 10.0
    symmetry_tau: float = 1.0          # autopsy: 0.6 killed 9.3% of true divisions
    min_track_len: int = 4
    diverge_um: float = 0.0
    n_candidates: int = 3              # consider the k nearest to C, not only the nearest
    # contest
    contest: bool = True
    contest_margin_um: float = 1.0     # challenger must be this much shorter
    contest_max_prob: float = 0.90     # and the incumbent must be this unsure
    frame_frac_cap: float = 0.02
    global_frac_cap: float = 0.01
    enabled: bool = True


def _chain_len(node, succ, limit=64):
    n, cur = 1, node
    while cur in succ and len(succ[cur]) == 1 and n < limit:
        cur = succ[cur][0]; n += 1
    return n


def augment(edges, det_t, det_zyx, cfg: DivConfig2, edge_meta=None):
    """edges: (src, tgt) pairs. edge_meta: {(s,t): (prob, dist_um)} for contests.

    Returns (new_edges, added, dropped).
    """
    base = [(int(s), int(t)) for s, t in edges]
    if not cfg.enabled:
        return base, [], []

    meta = edge_meta or {}
    pos = det_zyx.astype(np.float64) * GRID_UM
    succ, par = {}, {}
    for s, t in base:
        succ.setdefault(s, []).append(t)
        par[t] = s
    by_t = {}
    for i, t in enumerate(det_t):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))

    added, dropped = [], []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        nxt_arr = np.array(nxt)
        npos = pos[nxt_arr]
        parents = [i for i in by_t[t] if len(succ.get(i, ())) == 1]
        if not parents:
            continue
        cands = []
        for P in parents:
            C = succ[P][0]
            dpc = d(P, C)
            if dpc > cfg.existing_child_max_um:
                continue
            # the k nearest nodes to the existing child, not just the nearest
            dist_to_C = np.linalg.norm(npos - pos[C], axis=1)
            for k in np.argsort(dist_to_C)[:cfg.n_candidates + 1]:
                Q = int(nxt_arr[k])
                if Q == C:
                    continue
                dpq, dcq = d(P, Q), d(C, Q)
                if dpq > cfg.parent_max_um or dcq > cfg.sister_max_um:
                    continue
                if abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) > cfg.symmetry_tau:
                    continue
                incumbent = par.get(Q)
                if incumbent is not None:
                    if not cfg.contest or incumbent == P:
                        continue
                    p_inc, d_inc = meta.get((incumbent, Q), (1.0, d(incumbent, Q)))
                    # strict improvement: shorter arc AND an unsure incumbent
                    if not (dpq + cfg.contest_margin_um < d_inc and p_inc < cfg.contest_max_prob):
                        continue
                if cfg.min_track_len > 1 and min(_chain_len(C, succ), _chain_len(Q, succ)) < cfg.min_track_len:
                    continue
                if cfg.diverge_um > 0:
                    sc, sq = succ.get(C, []), succ.get(Q, [])
                    if len(sc) != 1 or len(sq) != 1 or d(sc[0], sq[0]) - dcq < cfg.diverge_um:
                        continue
                cands.append((dpq + 0.15 * dcq, P, Q, incumbent))
        cap = max(1, round(cfg.frame_frac_cap * len(by_t[t])))
        picked = 0
        for _, P, Q, inc in sorted(cands):
            if picked >= cap:
                break
            if len(succ.get(P, ())) >= 2 or par.get(Q) == P:
                continue
            if inc is not None:
                if par.get(Q) != inc:
                    continue
                dropped.append((inc, Q))
                succ[inc] = [x for x in succ.get(inc, []) if x != Q]
            added.append((P, Q))
            succ.setdefault(P, []).append(Q); par[Q] = P
            picked += 1

    cap = max(1, round(cfg.global_frac_cap * len(base)))
    if len(added) > cap:
        added, dropped = added[:cap], [x for x in dropped if x[1] in {q for _, q in added[:cap]}]
    drop_set = set(dropped)
    return [e for e in base if e not in drop_set] + added, added, dropped
