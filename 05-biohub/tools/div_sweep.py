#!/usr/bin/env python3
"""Sweep division SELECTION on the 0.947 model's own linked graphs.

Their add_safe_divisions_postlink proposes divisions, ranks them, and truncates by a cap. The
cap is not conservative -- 0.004 of edges against a true biological rate of 0.00391 measured on
Zebrahub -- so RANKING decides division Jaccard. Their ranking key is one line:

    score = parent_dist + 0.15 * sister_dist          (ascending; lowest wins)

which EXP-19 measured at pooled AUC 0.776, WORSE than parent_dist alone (0.806), because
sister_dist ranks at 0.551 -- near noise -- and drags the signal down.

This reimplements their selector faithfully with the ranker made pluggable, so alternatives can
be measured against ground truth offline instead of costing a two-hour submission each.
The DeepCenter image veto is omitted: it needs the volumes, and it gates candidates rather than
ordering them, so it is orthogonal to the ranking question.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def _pos(n) -> np.ndarray:
    return np.array([float(n["z"]) * VOXEL_SCALE_UM[0],
                     float(n["y"]) * VOXEL_SCALE_UM[1],
                     float(n["x"]) * VOXEL_SCALE_UM[2]])


def _dist(a, b) -> float:
    return float(np.linalg.norm(_pos(a) - _pos(b)))


@dataclass
class DivCfg:
    """Their published settings (biohub-lf-dctta cell 1)."""
    max_um: float = 9.0              # BIOHUB_SAFE_DIV_MAX_UM
    sister_max_um: float = 14.0      # BIOHUB_SAFE_DIV_SISTER_MAX_UM
    existing_child_max_um: float = 10.0
    symmetry_tau: float = 0.6        # BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU
    diverge_um: float = 2.25         # BIOHUB_SAFE_DIV_DIVERGE_UM
    require_divergence: bool = True
    require_mutual_nn: bool = True
    frame_frac_cap: float = 0.0076
    global_frac_cap: float = 0.00375


def theirs(f: dict) -> float:
    """Their ranker. Lower is better, so negate for a uniform 'higher wins' convention."""
    return -(f["parent_dist"] + 0.15 * f["sister_dist"])


def parent_only(f: dict) -> float:
    return -f["parent_dist"]


def _propose(nodes_by_id, edges, cfg: DivCfg):
    """Shared gate logic. Returns [(features, source_id, candidate_id, frame), ...]."""
    out_by_source, incoming = {}, set()
    for e in edges:
        out_by_source.setdefault(int(e["source_id"]), []).append(e)
        incoming.add(int(e["target_id"]))
    ids_by_t = {}
    for nid, n in nodes_by_id.items():
        ids_by_t.setdefault(int(n["t"]), []).append(nid)
    existing = {(int(e["source_id"]), int(e["target_id"])) for e in edges}

    props = []
    for t in sorted(ids_by_t):
        kids_frame = ids_by_t.get(t + 1, [])
        if not kids_frame:
            continue
        srcs = [i for i in ids_by_t[t] if len(out_by_source.get(i, [])) == 1]
        cands = [i for i in kids_frame if i not in incoming]
        if not srcs or not cands:
            continue
        tree = cKDTree(np.stack([_pos(nodes_by_id[c]) for c in cands])) if cfg.require_mutual_nn else None
        for sid in srcs:
            src = nodes_by_id[sid]
            ce = out_by_source[sid][0]
            cid = int(ce["target_id"]); child = nodes_by_id.get(cid)
            if child is None or int(child["t"]) != t + 1:
                continue
            child_dist = _dist(src, child)
            if child_dist > cfg.existing_child_max_um:
                continue
            mutual = cands[int(tree.query(_pos(child))[1])] if tree is not None else None
            for qid in cands:
                if (sid, qid) in existing:
                    continue
                q = nodes_by_id[qid]
                parent_dist = _dist(src, q)
                if parent_dist > cfg.max_um:
                    continue
                sister_dist = _dist(child, q)
                if sister_dist > cfg.sister_max_um:
                    continue
                if cfg.require_mutual_nn and qid != mutual:
                    continue
                diverge = 0.0
                if cfg.require_divergence or True:
                    cs, qs = out_by_source.get(cid, []), out_by_source.get(qid, [])
                    if len(cs) == 1 and len(qs) == 1:
                        cg = nodes_by_id.get(int(cs[0]["target_id"]))
                        qg = nodes_by_id.get(int(qs[0]["target_id"]))
                        if (cg is not None and qg is not None
                                and int(cg["t"]) == t + 2 and int(qg["t"]) == t + 2):
                            diverge = _dist(cg, qg) - sister_dist
                        elif cfg.require_divergence:
                            continue
                    elif cfg.require_divergence:
                        continue
                    if cfg.require_divergence and diverge < cfg.diverge_um:
                        continue
                if cfg.symmetry_tau > 0.0:
                    den = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / den > cfg.symmetry_tau:
                        continue
                va, vb = _pos(child) - _pos(src), _pos(q) - _pos(src)
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                cos = float(va @ vb / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
                props.append((dict(parent_dist=parent_dist, sister_dist=sister_dist,
                                   child_dist=child_dist, cos=cos, diverge=diverge,
                                   arc_max=max(na, nb), arc_min=min(na, nb),
                                   arc_asym=abs(na - nb), arc_sum=na + nb),
                              sid, qid, t))
    return props


def collect_proposals(nodes_by_id, edges, cfg: DivCfg):
    """Every (source, candidate) fork the gates admit, with its features. No ranking, no cap.

    Split out of add_safe_divisions so the trainer can harvest LABELLED candidates under widened
    gates using exactly the code path that runs in production -- a feature computed two ways is a
    bug waiting for a deadline.
    """
    return _propose(nodes_by_id, edges, cfg)


def add_safe_divisions(nodes_by_id, edges, cfg: DivCfg, rank=theirs):
    """Faithful reimplementation of add_safe_divisions_postlink with a pluggable ranker."""
    out_by_source, incoming = {}, set()
    for e in edges:
        out_by_source.setdefault(int(e["source_id"]), []).append(e)
        incoming.add(int(e["target_id"]))
    ids_by_t = {}
    for nid, n in nodes_by_id.items():
        ids_by_t.setdefault(int(n["t"]), []).append(nid)
    existing = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * cfg.global_frac_cap)))

    added, used_t, used_s = [], set(), set()
    for t in sorted(ids_by_t):
        kids_frame = ids_by_t.get(t + 1, [])
        if not kids_frame:
            continue
        srcs = [i for i in ids_by_t[t] if len(out_by_source.get(i, [])) == 1]
        cands = [i for i in kids_frame if i not in incoming and i not in used_t]
        if not srcs or not cands:
            continue
        tree = cKDTree(np.stack([_pos(nodes_by_id[c]) for c in cands])) if cfg.require_mutual_nn else None
        frame_cap = max(1, int(round(len(srcs) * cfg.frame_frac_cap)))
        props = []
        for sid in srcs:
            src = nodes_by_id[sid]
            ce = out_by_source[sid][0]
            cid = int(ce["target_id"]); child = nodes_by_id.get(cid)
            if child is None or int(child["t"]) != t + 1:
                continue
            child_dist = _dist(src, child)
            if child_dist > cfg.existing_child_max_um:
                continue
            mutual = cands[int(tree.query(_pos(child))[1])] if tree is not None else None
            for qid in cands:
                if (sid, qid) in existing:
                    continue
                q = nodes_by_id[qid]
                parent_dist = _dist(src, q)
                if parent_dist > cfg.max_um:
                    continue
                sister_dist = _dist(child, q)
                if sister_dist > cfg.sister_max_um:
                    continue
                if cfg.require_mutual_nn and qid != mutual:
                    continue
                diverge = None
                if cfg.require_divergence:
                    cs, qs = out_by_source.get(cid, []), out_by_source.get(qid, [])
                    if len(cs) != 1 or len(qs) != 1:
                        continue
                    cg = nodes_by_id.get(int(cs[0]["target_id"]))
                    qg = nodes_by_id.get(int(qs[0]["target_id"]))
                    if cg is None or qg is None or int(cg["t"]) != t + 2 or int(qg["t"]) != t + 2:
                        continue
                    diverge = _dist(cg, qg) - sister_dist
                    if diverge < cfg.diverge_um:
                        continue
                if cfg.symmetry_tau > 0.0:
                    den = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / den > cfg.symmetry_tau:
                        continue
                va, vb = _pos(child) - _pos(src), _pos(q) - _pos(src)
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                cos = float(va @ vb / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
                props.append((dict(parent_dist=parent_dist, sister_dist=sister_dist,
                                   child_dist=child_dist, cos=cos,
                                   diverge=(0.0 if diverge is None else diverge),
                                   arc_max=max(na, nb), arc_min=min(na, nb),
                                   arc_asym=abs(na - nb), arc_sum=na + nb),
                              sid, qid))
        if not props:
            continue
        props.sort(key=lambda p: -rank(p[0]))          # higher rank wins
        n_frame = 0
        for f, sid, qid in props:
            if len(added) >= global_cap or n_frame >= frame_cap:
                break
            if qid in used_t or qid in incoming or sid in used_s:
                continue
            added.append({"source_id": sid, "target_id": qid, "edge_prob": None,
                          "distance_um": f["parent_dist"], "safe_division": 1})
            used_t.add(qid); used_s.add(sid); n_frame += 1
    return [*edges, *added]
