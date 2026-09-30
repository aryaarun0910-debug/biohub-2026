r"""GT-free daughter-steal rule - LEVER-0025 stage C instrument (PKT-0021).

WHY THIS EXISTS
---------------
Stage A (`div_reach_steal.py census`) showed that most metric-legal-but-unscored GT divisions on
the champion controls die because the second daughter ``b`` already has a parent ``P`` that is not
the mother ``M`` (FACT-0295: that edge cannot be a GT edge). Stage B is the GT-guided oracle. This
file is stage C: a rule that picks the same steals WITHOUT ground truth, from the export alone.

The rule is a re-ranking of the deployed proposer's candidate set (coupled_division_transplant.py)
with its orphan condition (C2) relaxed under explicit geometric conditions. Every threshold is
fit on one fold and tested on the other (both directions), never on the fold it is reported on.

Stages
------
  features : enumerate the GT-free candidate population (M, a, b, P) on an export, compute
             features, and LABEL each candidate from the oracle plans + atlas (labels are for
             evaluation only; the rule never sees them). Also tabulates the oracle's planned
             steals/orphans with the same features.
  fit      : grid-search the rule thresholds on one fold's features (train) and report
             precision/recall on the other (test).
  run      : apply a rule (explicit parameters) GT-free to an export, write the modified CSV,
             score control vs rule per crop with the official metrics, paired crop bootstrap.
  econ     : the price of a FALSE add, measured: oracle plans + N random non-oracle candidates of
             one kind (steal | orphan), scored per crop on the touched crops. Calibrates `search`.
  search   : scorer-faithful proxy (linearised pooled-Jaccard deltas from the control totals) per
             candidate, univariate lift table, and a greedy conjunction fit on one fold reported
             on the other. Labels evaluate; conditions are GT-free.

Discrepancy resolved 2026-08-27 (why most oracle adds were "not in the population"): the
population is enumerated at --parent-max/--sister-max (12/14 in the recorded run) while the old
`in_population` flag used the deployed 8/11 constants, so the two summary counts measured
different things. The substantive finding is that every oracle add has d(M,a) <= 3.8 um and
d(M,b) of 5-20 um (median ~10): the unscored residual is precisely the geometry the deployed
8 um parent gate excludes (survivorship - the in-gate divisions are the ones already scored).

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\div_steal_rule.py features ^
      --csv C:/temp/p20_relink_sweep_f0/sweep_pen_off.csv.gz --tag f0 ^
      --plans C:/temp/div_reach/f0/plans_f0.parquet --atlas-dir C:/temp/p20_relink_sweep_f0/atlas ^
      --out-dir C:/temp/div_reach/rule
  .\.venv\Scripts\python.exe scripts\win_bet\div_steal_rule.py fit --train f0 --test f1 --out-dir C:/temp/div_reach/rule
  .\.venv\Scripts\python.exe scripts\win_bet\div_steal_rule.py run ^
      --csv C:/temp/p19_relink_sweep_f1/sweep_pen_off.csv.gz --tag f1 --params C:/temp/div_reach/rule/params_fit_f0.json ^
      --out-dir C:/temp/div_reach/rule
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=float)   # z, y, x um per level-0 voxel

# Deployed proposer constants (coupled_division_transplant.py), verified 2026-08-27 in the built
# notebook notebooks/kaggle_p20_relink_sweep_f0/biohub-p20-relink-sweep-loeo-f0.ipynb: cell2 sets
# the 4.66/8.5/7.65 preset and then the "factory env overrides" set 8.0/11.0/10.0 in the same
# cell, before cell3 reads them into SAFE_DIV_*; DEEPCENTER_SAFE_DIV_VETO is "0".
PARENT_MAX_UM = 8.0
SISTER_MAX_UM = 11.0
CHILD_MAX_UM = 10.0
DIVERGE_UM = 2.25
FRAME_FRAC_CAP = 0.0076
GLOBAL_FRAC_CAP = 0.00375
RELINK_TIGHT_UM = 6.0
RELINK_RELAXED_UM = 10.0

EPS = 1e-6


# ============================================================================ graph arrays
class Crop:
    """Array view of one crop's lineage graph (a plain class so the module imports by path)."""

    def __init__(self, ids, t, pos, parent, child1, child2, outdeg, by_t, len_before, len_after):
        self.ids = ids              # submission node ids, sorted
        self.t = t                  # int frame per node
        self.pos = pos              # (n, 3) um
        self.parent = parent        # index of parent or -1
        self.child1 = child1        # first child index or -1
        self.child2 = child2        # second child index or -1
        self.outdeg = outdeg
        self.by_t = by_t
        self.len_before = len_before   # number of consecutive predecessors
        self.len_after = len_after     # longest chain of successors

    def index_of(self, sub_ids) -> np.ndarray:
        sub_ids = np.asarray(sub_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, sub_ids)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        ok = self.ids[pos] == sub_ids
        return np.where(ok, pos, -1)


def build_crop(node_ids, t, zyx_vox, src, tgt) -> Crop:
    """Arrays for one crop. Coordinates are voxels; ``pos`` is in um at the anisotropic scale."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    order = np.argsort(node_ids, kind="stable")
    ids = node_ids[order]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise RuntimeError("duplicate node ids in crop")
    t = np.asarray(t, dtype=np.int64)[order]
    pos = np.asarray(zyx_vox, dtype=float)[order] * SCALE[None, :]
    n = len(ids)
    parent = np.full(n, -1, dtype=np.int64)
    child1 = np.full(n, -1, dtype=np.int64)
    child2 = np.full(n, -1, dtype=np.int64)
    outdeg = np.zeros(n, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    si = np.searchsorted(ids, src)
    ti = np.searchsorted(ids, tgt)
    if len(src) and (np.any(ids[np.clip(si, 0, n - 1)] != src) or np.any(ids[np.clip(ti, 0, n - 1)] != tgt)):
        raise RuntimeError("edge references an unknown node")
    for s, d in zip(si, ti):
        if parent[d] != -1:
            raise RuntimeError(f"node {ids[d]} has two parents")
        parent[d] = s
        if outdeg[s] == 0:
            child1[s] = d
        elif outdeg[s] == 1:
            child2[s] = d
        else:
            raise RuntimeError(f"node {ids[s]} has more than two children")
        outdeg[s] += 1
    by_t = {int(k): np.flatnonzero(t == k) for k in np.unique(t)}
    len_before = np.zeros(n, dtype=np.int64)
    for k in sorted(by_t):
        idx = by_t[k]
        p = parent[idx]
        has = p >= 0
        len_before[idx[has]] = len_before[p[has]] + 1
    len_after = np.zeros(n, dtype=np.int64)
    for k in sorted(by_t, reverse=True):
        idx = by_t[k]
        c1, c2 = child1[idx], child2[idx]
        l1 = np.where(c1 >= 0, len_after[np.maximum(c1, 0)] + 1, 0)
        l2 = np.where(c2 >= 0, len_after[np.maximum(c2, 0)] + 1, 0)
        len_after[idx] = np.maximum(l1, l2)
    return Crop(ids, t, pos, parent, child1, child2, outdeg, by_t, len_before, len_after)


def crop_from_frame(sub: pl.DataFrame) -> Crop:
    nodes = sub.filter(pl.col("row_type") == "node")
    edges = sub.filter(pl.col("row_type") == "edge")
    return build_crop(nodes["node_id"].to_numpy(), nodes["t"].to_numpy(),
                      np.stack([nodes["z"].to_numpy(), nodes["y"].to_numpy(), nodes["x"].to_numpy()], axis=1),
                      edges["source_id"].to_numpy(), edges["target_id"].to_numpy())


# ============================================================================ features
def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v * v).sum(axis=1))


def _cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    nu, nv = _norm(u), _norm(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (u * v).sum(axis=1) / (nu * nv)
    return np.where((nu > 0) & (nv > 0), c, np.nan)


def _safe_pos(crop: Crop, idx: np.ndarray) -> np.ndarray:
    """Position rows for ``idx`` with NaN where idx == -1."""
    out = crop.pos[np.maximum(idx, 0)].astype(float).copy()
    out[idx < 0] = np.nan
    return out


def _nn_excluding_self(tree: cKDTree, frame_idx: np.ndarray, q_idx: np.ndarray, crop: Crop) -> tuple[np.ndarray, np.ndarray]:
    """Nearest node in the frame to each query node, excluding the query node itself."""
    if len(q_idx) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    k = min(3, len(frame_idx))
    d, j = tree.query(crop.pos[q_idx], k=k)
    d = np.atleast_2d(d).reshape(len(q_idx), k)
    j = np.atleast_2d(j).reshape(len(q_idx), k)
    cand = frame_idx[np.clip(j, 0, len(frame_idx) - 1)]
    cand = np.where(j < len(frame_idx), cand, -1)
    out_idx = np.full(len(q_idx), -1, dtype=np.int64)
    out_d = np.full(len(q_idx), np.inf)
    for col in range(k - 1, -1, -1):
        ok = (cand[:, col] != q_idx[:, None][:, 0]) & (cand[:, col] >= 0)
        out_idx = np.where(ok, cand[:, col], out_idx)
        out_d = np.where(ok, d[:, col], out_d)
    return out_idx, out_d


def candidate_features(crop: Crop, M: np.ndarray, A: np.ndarray, B: np.ndarray,
                       parent_max: float = PARENT_MAX_UM) -> pd.DataFrame:
    """GT-free features for explicit (M, a, b) triples (indices into ``crop``), any frame.

    M is the prospective mother, ``a`` its existing (only) child at t+1, ``b`` the prospective second
    daughter at t+1; ``P`` is b's current parent (or none). All distances in um.
    """
    M = np.asarray(M, dtype=np.int64)
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    n = len(M)
    tM = crop.t[M]
    pM, pA, pB = crop.pos[M], crop.pos[A], crop.pos[B]
    P = crop.parent[B]
    has_p = P >= 0
    pP = _safe_pos(crop, P)
    d_mb = _norm(pB - pM)
    d_ab = _norm(pB - pA)
    d_ma = _norm(pA - pM)
    d_pb = _norm(pB - pP)                      # nan when no P
    Mpred = crop.parent[M]
    pMpred = _safe_pos(crop, Mpred)
    Ppred = np.where(has_p, crop.parent[np.maximum(P, 0)], -1)
    pPpred = _safe_pos(crop, Ppred)

    # successors at t+2 (single-successor rule of the deployed C3)
    a_succ = np.where(crop.outdeg[A] == 1, crop.child1[A], -1)
    b_succ = np.where(crop.outdeg[B] == 1, crop.child1[B], -1)
    a_succ = np.where((a_succ >= 0) & (crop.t[np.maximum(a_succ, 0)] == tM + 2), a_succ, -1)
    b_succ = np.where((b_succ >= 0) & (crop.t[np.maximum(b_succ, 0)] == tM + 2), b_succ, -1)
    pAs, pBs = _safe_pos(crop, a_succ), _safe_pos(crop, b_succ)
    divergence = _norm(pAs - pBs) - d_ab       # nan unless both continue

    mid = 0.5 * (pA + pB)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mb_pb = d_mb / np.where(has_p, d_pb, np.nan)
    feat = {
        "t": tM,
        "d_mb": d_mb, "d_ab": d_ab, "d_ma": d_ma,
        "gate_parent": d_mb <= PARENT_MAX_UM, "gate_sister": d_ab <= SISTER_MAX_UM,
        "gate_child": d_ma <= CHILD_MAX_UM,
        "c1_has_pred": Mpred >= 0,
        "a_continues": a_succ >= 0, "b_continues": b_succ >= 0,
        "divergence": divergence,
        "c3_ok": np.nan_to_num(divergence, nan=-np.inf) >= DIVERGE_UM,
        "b_has_parent": has_p,
        "p_outdeg": np.where(has_p, crop.outdeg[np.maximum(P, 0)], 0),
        "p_has_pred": Ppred >= 0,
        "p_len_before": np.where(has_p, crop.len_before[np.maximum(P, 0)], -1),
        "b_len_after": crop.len_after[B],
        "a_len_after": crop.len_after[A],
        "m_len_before": crop.len_before[M],
        "d_pb": d_pb,
        "steal_gain": d_pb - d_mb,                              # >0: M is nearer to b than P is
        "ratio_mb_pb": ratio_mb_pb,
        "asym": np.abs(d_ma - d_mb),
        "d_m_mid": _norm(pM - mid),
        "cos_daughters": _cos(pA - pM, pB - pM),
        "d_mpred_mid": _norm(pMpred + (pM - pMpred) - mid),     # M's constant-velocity prediction vs midpoint
        "d_ppred_b": _norm(pP + (pP - pPpred) - pB),            # P's constant-velocity prediction vs b
        "cos_pb_next": _cos(pB - pP, pBs - pB),
        "cos_mb_next": _cos(pB - pM, pBs - pB),
        "cos_ab_next": _cos(pBs - pB, pAs - pA),
    }

    # frame-local neighbourhood features (per frame to keep the trees small)
    nn_a_is_b = np.zeros(n, dtype=bool)
    nn_b_is_a = np.zeros(n, dtype=bool)
    n_t1_closer_than_b = np.zeros(n, dtype=np.int64)
    n_t_closer_than_m = np.zeros(n, dtype=np.int64)
    n_t_closer_than_p = np.full(n, -1, dtype=np.int64)
    nearest_t_of_b = np.full(n, -1, dtype=np.int64)
    d_p_orphan = np.full(n, np.nan)
    d_ppred_orphan = np.full(n, np.nan)
    d_m_orphan_excl_a = np.full(n, np.nan)
    n_t1_within_parent_max = np.zeros(n, dtype=np.int64)
    n_orphans_t1_within_parent_max = np.zeros(n, dtype=np.int64)
    d_p_nn_t1_excl_b = np.full(n, np.nan)
    for tt in np.unique(tM):
        sel = np.flatnonzero(tM == tt)
        f1 = crop.by_t.get(int(tt) + 1)
        f0 = crop.by_t.get(int(tt))
        if f1 is None or len(f1) == 0:
            continue
        tree1 = cKDTree(crop.pos[f1])
        tree0 = cKDTree(crop.pos[f0])
        a_s, b_s, m_s, p_s = A[sel], B[sel], M[sel], P[sel]
        nn_a, _ = _nn_excluding_self(tree1, f1, a_s, crop)
        nn_b, _ = _nn_excluding_self(tree1, f1, b_s, crop)
        nn_a_is_b[sel] = nn_a == b_s
        nn_b_is_a[sel] = nn_b == a_s
        r = np.maximum(d_mb[sel] - EPS, 0.0)
        n_t1_closer_than_b[sel] = tree1.query_ball_point(pM[sel], r=r, return_length=True)
        n_t_closer_than_m[sel] = tree0.query_ball_point(pB[sel], r=r, return_length=True)
        n_t1_within_parent_max[sel] = tree1.query_ball_point(pM[sel], r=parent_max, return_length=True)
        _d0, j0 = tree0.query(pB[sel], k=1)
        nearest_t_of_b[sel] = f0[np.asarray(j0).reshape(-1)]
        hp = has_p[sel]
        if hp.any():
            rp = np.maximum(d_pb[sel][hp] - EPS, 0.0)
            n_t_closer_than_p[sel[hp]] = tree0.query_ball_point(pB[sel][hp], r=rp, return_length=True)
            # nearest t+1 node to P other than b
            k = min(3, len(f1))
            dd, jj = tree1.query(pP[sel][hp], k=k)
            dd = np.atleast_2d(dd).reshape(hp.sum(), k)
            jj = np.atleast_2d(jj).reshape(hp.sum(), k)
            cand = np.where(jj < len(f1), f1[np.clip(jj, 0, len(f1) - 1)], -1)
            best = np.full(hp.sum(), np.nan)
            for col in range(k - 1, -1, -1):
                ok = (cand[:, col] != b_s[hp]) & (cand[:, col] >= 0)
                best = np.where(ok, dd[:, col], best)
            d_p_nn_t1_excl_b[sel[hp]] = best
        orph = f1[crop.parent[f1] < 0]
        if len(orph):
            treeo = cKDTree(crop.pos[orph])
            n_orphans_t1_within_parent_max[sel] = treeo.query_ball_point(pM[sel], r=parent_max, return_length=True)
            if hp.any():
                d_p_orphan[sel[hp]] = treeo.query(pP[sel][hp], k=1)[0]
                pp = Ppred[sel][hp] >= 0
                if pp.any():
                    pred_pos = (pP[sel][hp] + (pP[sel][hp] - pPpred[sel][hp]))[pp]
                    d_ppred_orphan[sel[hp][pp]] = treeo.query(pred_pos, k=1)[0]
            # nearest orphan to M other than a (a is never an orphan: it has parent M) and other than b
            k = min(2, len(orph))
            dd, jj = treeo.query(pM[sel], k=k)
            dd = np.atleast_2d(dd).reshape(len(sel), k)
            jj = np.atleast_2d(jj).reshape(len(sel), k)
            cand = np.where(jj < len(orph), orph[np.clip(jj, 0, len(orph) - 1)], -1)
            best = np.full(len(sel), np.nan)
            for col in range(k - 1, -1, -1):
                ok = (cand[:, col] != b_s) & (cand[:, col] >= 0)
                best = np.where(ok, dd[:, col], best)
            d_m_orphan_excl_a[sel] = best
    feat.update({
        "nn_a_is_b": nn_a_is_b, "nn_b_is_a": nn_b_is_a,
        "mutual_nn_all": nn_a_is_b & nn_b_is_a,
        "n_t1_closer_than_b": n_t1_closer_than_b,     # nodes at t+1 nearer to M than b (a counts)
        "n_t_closer_than_m": n_t_closer_than_m,       # nodes at t nearer to b than M
        "n_t_closer_than_p": n_t_closer_than_p,       # nodes at t nearer to b than P (-1: no P)
        "nearest_t_is_m": nearest_t_of_b == M,
        "nearest_t_is_p": has_p & (nearest_t_of_b == P),
        "d_p_orphan": d_p_orphan,                     # nearest orphan at t+1 to P
        "d_ppred_orphan": d_ppred_orphan,             # nearest orphan at t+1 to P's motion prediction
        "p_alt_tight": np.nan_to_num(d_p_orphan, nan=np.inf) <= RELINK_TIGHT_UM,
        "p_alt_relaxed": np.nan_to_num(d_p_orphan, nan=np.inf) <= RELINK_RELAXED_UM,
        "assign_gain": d_pb - (d_mb + np.nan_to_num(d_p_orphan, nan=0.0)),   # current cost - steal cost (P->orphan)
        "d_p_nn_t1_excl_b": d_p_nn_t1_excl_b,
        "d_m_orphan_excl_ab": d_m_orphan_excl_a,
        "n_t1_within_parent_max": n_t1_within_parent_max,
        "n_orphans_t1_within_parent_max": n_orphans_t1_within_parent_max,
        "rank_score": d_mb + 0.15 * d_ab,             # the deployed ranking key
    })
    out = pd.DataFrame(feat)
    out.insert(0, "P_id", np.where(has_p, crop.ids[np.maximum(P, 0)], -1))
    out.insert(0, "b_id", crop.ids[B])
    out.insert(0, "a_id", crop.ids[A])
    out.insert(0, "M_id", crop.ids[M])
    return out


def enumerate_candidates(crop: Crop, parent_max: float = PARENT_MAX_UM,
                         sister_max: float = SISTER_MAX_UM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (M, a, b): M out-degree 1 with child a at t+1; b another t+1 node within ``parent_max``
    of M and ``sister_max`` of a. b may be an orphan or owned by P != M."""
    Ms, As, Bs = [], [], []
    for tt in sorted(crop.by_t):
        f1 = crop.by_t.get(tt + 1)
        if f1 is None or len(f1) == 0:
            continue
        f0 = crop.by_t[tt]
        m = f0[crop.outdeg[f0] == 1]
        if len(m) == 0:
            continue
        a = crop.child1[m]
        ok = crop.t[a] == tt + 1
        m, a = m[ok], a[ok]
        if len(m) == 0:
            continue
        tree1 = cKDTree(crop.pos[f1])
        lists = tree1.query_ball_point(crop.pos[m], r=parent_max)
        counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        if counts.sum() == 0:
            continue
        mrep = np.repeat(m, counts)
        arep = np.repeat(a, counts)
        b = f1[np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])]
        keep = b != arep
        keep &= _norm(crop.pos[b] - crop.pos[arep]) <= sister_max
        Ms.append(mrep[keep]); As.append(arep[keep]); Bs.append(b[keep])
    if not Ms:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return np.concatenate(Ms), np.concatenate(As), np.concatenate(Bs)


# ============================================================================ the rule
DEFAULT_PARAMS = {
    "parent_max": PARENT_MAX_UM, "sister_max": SISTER_MAX_UM, "child_max": CHILD_MAX_UM,
    "require_c1": True, "require_c3": True, "diverge_min": DIVERGE_UM,
    "require_mutual_nn": True,      # b and a are each other's nearest t+1 node among ALL t+1 nodes
    "require_m_nearest_t": True,    # M is the nearest t-node to b (so P is not)
    "min_steal_gain": 0.0,          # d(P,b) - d(M,b)
    "max_cos_daughters": 1.0,       # cos of the two daughter vectors at M
    "max_asym": np.inf,             # |d(M,a) - d(M,b)|
    "min_d_pb": 0.0,                # the stolen edge's own length
    "steal": True,                  # allow b with a parent (the steal); False = orphan-only
    "orphan": False,                # allow b without a parent (re-proposing what the proposer missed)
    "max_p_outdeg": 2,
}


def rule_mask(feat: pd.DataFrame, params: dict | None = None) -> np.ndarray:
    p = {**DEFAULT_PARAMS, **(params or {})}
    f = feat
    m = (f["d_mb"].to_numpy() <= p["parent_max"]) & (f["d_ab"].to_numpy() <= p["sister_max"]) \
        & (f["d_ma"].to_numpy() <= p["child_max"])
    if p["require_c1"]:
        m &= f["c1_has_pred"].to_numpy()
    if p["require_c3"]:
        m &= np.nan_to_num(f["divergence"].to_numpy(dtype=float), nan=-np.inf) >= p["diverge_min"]
    if p["require_mutual_nn"]:
        m &= f["mutual_nn_all"].to_numpy()
    if p["require_m_nearest_t"]:
        m &= f["nearest_t_is_m"].to_numpy()
    has_p = f["b_has_parent"].to_numpy()
    kind_ok = np.zeros(len(f), dtype=bool)
    if p["steal"]:
        steal_ok = has_p & (np.nan_to_num(f["steal_gain"].to_numpy(dtype=float), nan=-np.inf) >= p["min_steal_gain"])
        steal_ok &= np.nan_to_num(f["d_pb"].to_numpy(dtype=float), nan=-np.inf) >= p["min_d_pb"]
        steal_ok &= f["p_outdeg"].to_numpy() <= p["max_p_outdeg"]
        kind_ok |= steal_ok
    if p["orphan"]:
        kind_ok |= ~has_p
    m &= kind_ok
    cos = np.nan_to_num(f["cos_daughters"].to_numpy(dtype=float), nan=1.0)
    m &= cos <= p["max_cos_daughters"]
    m &= f["asym"].to_numpy(dtype=float) <= p["max_asym"]
    return m


def select_edits(feat: pd.DataFrame, mask: np.ndarray) -> list[dict]:
    """One edit per daughter and per mother: rank accepted candidates by the deployed key
    (d_mb + 0.15 d_ab, ascending), take the best per b, then the best per M. Returns plans in the
    div_reach_steal layout: {"fork", "add": [(M, b)], "remove": [(P, b)] or []}."""
    acc = feat.loc[mask].sort_values(["rank_score", "M_id", "b_id"], kind="stable")
    acc = acc.drop_duplicates("b_id", keep="first").drop_duplicates("M_id", keep="first")
    plans = []
    for r in acc.itertuples(index=False):
        plans.append({"route": "rule", "fork": int(r.M_id), "keep": [int(r.a_id)],
                      "add": [(int(r.M_id), int(r.b_id))],
                      "remove": [(int(r.P_id), int(r.b_id))] if r.b_has_parent else [],
                      "daughter_kinds": ["rule"], "removed_edge_is_full_tp": False})
    return plans


# ============================================================================ glue
def _drs():
    spec = importlib.util.spec_from_file_location("div_reach_steal", ROOT / "scripts" / "win_bet" / "div_reach_steal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_export(csv: Path):
    drs = _drs()
    ea = drs._ea_atlas()
    from biotrack.submission import read_submission
    return drs, ea, read_submission(ea.open_csv(csv))


def _plans_table(plans_path: Path) -> pd.DataFrame:
    p = pd.read_parquet(plans_path)
    rows = []
    for r in p.itertuples(index=False):
        adds = json.loads(r.add)
        rems = json.loads(r.remove)
        for (m, b) in adds:
            rem = [x for x in rems if x[1] == b]
            rows.append({"dataset": r.dataset, "divider": int(r.divider), "route": r.route,
                         "daughter_kind": r.daughter_kinds, "M_id": int(m), "b_id": int(b),
                         "P_id": int(rem[0][0]) if rem else -1,
                         "plan_kind": "steal" if rem else "orphan"})
    return pd.DataFrame(rows)


def _labels_from_atlas(atlas_dir: Path, arm: str = "pen_off"):
    nodes = pd.read_parquet(atlas_dir / f"nodes_{arm}.parquet", columns=["dataset", "node_id", "gt_id"])
    edges = pd.read_parquet(atlas_dir / f"edges_{arm}.parquet", columns=["dataset", "source_id", "target_id", "matched", "pred_valid"])
    gte = pd.read_parquet(atlas_dir / f"gtedges_{arm}.parquet", columns=["dataset", "gt_source", "gt_target"])
    gtn = pd.read_parquet(atlas_dir / f"gtnodes_{arm}.parquet", columns=["dataset", "gt_id", "out_degree"])
    return nodes, edges, gte, gtn


def label_candidates(cands: pd.DataFrame, plans: pd.DataFrame, atlas) -> pd.DataFrame:
    """Attach evaluation-only labels: oracle positive, GT-edge truth of M->b, TP-ness of P->b."""
    nodes, edges, gte, gtn = atlas
    out = cands.copy()
    key = pd.MultiIndex.from_frame(out[["dataset", "M_id", "b_id"]])
    pk = pd.MultiIndex.from_frame(plans[["dataset", "M_id", "b_id"]]) if len(plans) else pd.MultiIndex.from_tuples([], names=["dataset", "M_id", "b_id"])
    out["oracle_pos"] = key.isin(pk)
    pkind = plans.drop_duplicates(["dataset", "M_id", "b_id"]).set_index(["dataset", "M_id", "b_id"])["plan_kind"] \
        if len(plans) else pd.Series(dtype=object)
    out["plan_kind"] = pd.Series(pkind.reindex(key).to_numpy(), index=out.index).fillna("")
    gt_of = nodes.set_index(["dataset", "node_id"])["gt_id"]
    out["gt_M"] = gt_of.reindex(pd.MultiIndex.from_frame(out[["dataset", "M_id"]])).to_numpy()
    out["gt_b"] = gt_of.reindex(pd.MultiIndex.from_frame(out[["dataset", "b_id"]])).to_numpy()
    out["gt_a"] = gt_of.reindex(pd.MultiIndex.from_frame(out[["dataset", "a_id"]])).to_numpy()
    out["gt_P"] = gt_of.reindex(pd.MultiIndex.from_frame(out[["dataset", "P_id"]])).to_numpy()
    for c in ("gt_M", "gt_b", "gt_a", "gt_P"):
        out[c] = out[c].fillna(-1).astype("int64")
    gte_key = pd.MultiIndex.from_frame(gte[["dataset", "gt_source", "gt_target"]])
    out["mb_is_gt_edge"] = pd.MultiIndex.from_arrays([out["dataset"], out["gt_M"], out["gt_b"]]).isin(gte_key) & (out["gt_M"] >= 0) & (out["gt_b"] >= 0)
    out["pb_is_gt_edge"] = pd.MultiIndex.from_arrays([out["dataset"], out["gt_P"], out["gt_b"]]).isin(gte_key) & (out["gt_P"] >= 0) & (out["gt_b"] >= 0)
    e = edges.set_index(["dataset", "source_id", "target_id"])
    ek = pd.MultiIndex.from_frame(out[["dataset", "P_id", "b_id"]])
    out["pb_matched"] = e["matched"].reindex(ek).fillna(False).to_numpy().astype(bool)
    out["pb_pred_valid"] = e["pred_valid"].reindex(ek).fillna(False).to_numpy().astype(bool)
    od = gtn.set_index(["dataset", "gt_id"])["out_degree"]
    out["gt_M_outdeg"] = od.reindex(pd.MultiIndex.from_frame(out[["dataset", "gt_M"]])).fillna(-1).to_numpy().astype(int)
    return out


def planned_in_population(pf: pd.DataFrame, parent_max: float, sister_max: float) -> np.ndarray:
    """Whether a planned add falls inside the ENUMERATION gates (not the deployed 8/11 constants)."""
    return (pf["d_mb"].to_numpy(dtype=float) <= parent_max) & (pf["d_ab"].to_numpy(dtype=float) <= sister_max)


def cmd_features(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drs, ea, df = _load_export(Path(args.csv))
    plans = _plans_table(Path(args.plans))
    atlas = _labels_from_atlas(Path(args.atlas_dir))
    names = drs.select_names(sorted(df["dataset"].unique().to_list()), args.max_crops, args.stride)
    print(f"{len(names)} crops, {len(plans)} planned adds", flush=True)
    t0 = time.time()
    cand_frames: list[pd.DataFrame] = []
    plan_frames: list[pd.DataFrame] = []
    for i, name in enumerate(names, 1):
        sub = df.filter(pl.col("dataset") == name)
        crop = crop_from_frame(sub)
        M, A, B = enumerate_candidates(crop, args.parent_max, args.sister_max)
        cf = candidate_features(crop, M, A, B, args.parent_max)
        cf.insert(0, "dataset", name)
        cand_frames.append(cf)
        pc = plans[plans.dataset == name]
        if len(pc):
            mi = crop.index_of(pc["M_id"].to_numpy())
            bi = crop.index_of(pc["b_id"].to_numpy())
            ok = (mi >= 0) & (bi >= 0) & (crop.outdeg[np.maximum(mi, 0)] == 1)
            if not ok.all():
                print(f"  {name}: {int((~ok).sum())} planned adds not reproducible on the export (fork out-degree != 1?)")
            mi, bi = mi[ok], bi[ok]
            ai = crop.child1[mi]
            pf = candidate_features(crop, mi, ai, bi, args.parent_max)
            pf.insert(0, "dataset", name)
            pf = pf.merge(pc[ok][["M_id", "b_id", "divider", "route", "daughter_kind", "plan_kind"]], on=["M_id", "b_id"], how="left")
            plan_frames.append(pf)
        print(f"  [{i}/{len(names)}] {name} nodes={len(crop.ids)} cands={len(cf)} plans={len(pc)} ({time.time() - t0:.0f}s)", flush=True)
    cands = pd.concat(cand_frames, ignore_index=True)
    cands = label_candidates(cands, plans, atlas)
    cands.to_parquet(out / f"cands_{args.tag}.parquet")
    pf_all = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
    if len(pf_all):
        pf_all = label_candidates(pf_all, plans, atlas)
        pf_all["in_population"] = planned_in_population(pf_all, args.parent_max, args.sister_max)
        pf_all["in_deployed_gates"] = pf_all["gate_parent"] & pf_all["gate_sister"]
        pf_all.to_parquet(out / f"plans_features_{args.tag}.parquet")
    summary = {
        "tag": args.tag, "crops": len(names), "population": int(len(cands)),
        "population_with_parent": int(cands.b_has_parent.sum()),
        "population_orphan": int((~cands.b_has_parent).sum()),
        "oracle_positives_in_population": int(cands.oracle_pos.sum()),
        "oracle_positives_steal": int((cands.oracle_pos & cands.b_has_parent).sum()),
        "oracle_positives_orphan": int((cands.oracle_pos & ~cands.b_has_parent).sum()),
        "planned_adds_total": int(len(plans)),
        "enumeration_gates_um": [args.parent_max, args.sister_max],
        "planned_in_population": int(pf_all.in_population.sum()) if len(pf_all) else 0,
        "planned_in_deployed_gates": int(pf_all.in_deployed_gates.sum()) if len(pf_all) else 0,
        "mb_gt_edge_in_population": int(cands.mb_is_gt_edge.sum()),
        "pb_matched_in_population": int(cands.pb_matched.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (out / f"features_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# ============================================================================ fit / evaluate
def evaluate_rule(cands: pd.DataFrame, params: dict) -> dict:
    """Precision/recall of a rule on a labelled candidate table, after per-b/per-M selection."""
    mask = rule_mask(cands, params)
    # per-crop selection (node ids are only unique within a crop): same order as select_edits
    sel = np.zeros(len(cands), dtype=bool)
    if mask.any():
        for _name, g in cands.loc[mask].groupby("dataset", sort=False):
            gsel = g.sort_values(["rank_score", "M_id", "b_id"], kind="stable").drop_duplicates("b_id").drop_duplicates("M_id")
            sel[cands.index.get_indexer(gsel.index)] = True
    c = cands.loc[sel]
    pos_total = int(cands.oracle_pos.sum())
    pos_steal_total = int((cands.oracle_pos & cands.b_has_parent).sum())
    tp = int(c.oracle_pos.sum())
    fp = int((~c.oracle_pos).sum())
    return {
        "fired": int(sel.sum()), "tp_oracle": tp, "fp_oracle": fp,
        "precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall": (tp / pos_total) if pos_total else float("nan"),
        "recall_steal": (int((c.oracle_pos & c.b_has_parent).sum()) / pos_steal_total) if pos_steal_total else float("nan"),
        "fired_steal": int(c.b_has_parent.sum()), "fired_orphan": int((~c.b_has_parent).sum()),
        "false_mb_is_gt_edge": int((~c.oracle_pos & c.mb_is_gt_edge).sum()),   # 'false' by the oracle but a GT edge anyway
        "false_breaks_matched_pb": int((~c.oracle_pos & c.pb_matched).sum()),  # the real cost: a scored TP edge broken
        "false_breaks_gt_pb": int((~c.oracle_pos & c.pb_is_gt_edge).sum()),
        "crops_touched": int(c.dataset.nunique()),
    }


def _grid(base: dict) -> list[dict]:
    grids = {
        "min_steal_gain": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        "max_cos_daughters": [1.0, 0.5, 0.0, -0.3, -0.5],
        "max_asym": [np.inf, 4.0, 3.0, 2.0],
        "min_d_pb": [0.0, 2.0, 3.0, 4.0, 5.0],
    }
    import itertools
    keys = list(grids)
    out = []
    for combo in itertools.product(*[grids[k] for k in keys]):
        out.append({**base, **dict(zip(keys, combo))})
    return out


def cmd_fit(args) -> int:
    out = Path(args.out_dir)
    tr = pd.read_parquet(out / f"cands_{args.train}.parquet")
    te = pd.read_parquet(out / f"cands_{args.test}.parquet")
    base = {**DEFAULT_PARAMS, "steal": True, "orphan": bool(args.orphan)}
    results = []
    for params in _grid(base):
        r_tr = evaluate_rule(tr, params)
        obj = r_tr["tp_oracle"] - args.fp_weight * r_tr["fp_oracle"]
        results.append((obj, params, r_tr))
    results.sort(key=lambda x: (-x[0], x[2]["fp_oracle"]))
    best_obj, best_params, best_tr = results[0]
    r_te = evaluate_rule(te, best_params)
    fixed = evaluate_rule(te, base)      # the un-fitted base rule (no threshold chosen on either fold)
    fixed_tr = evaluate_rule(tr, base)
    summary = {
        "train": args.train, "test": args.test, "fp_weight": args.fp_weight,
        "best_params": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in best_params.items()},
        "train_objective": best_obj, "train_result": best_tr, "test_result": r_te,
        "base_params": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in base.items()},
        "base_train_result": fixed_tr, "base_test_result": fixed,
        "top5_train": [{"objective": o, "params": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in p.items() if k in ("min_steal_gain", "max_cos_daughters", "max_asym", "min_d_pb")}, "train": r} for o, p, r in results[:5]],
    }
    tag = f"fit_{args.train}"
    (out / f"params_{tag}.json").write_text(json.dumps({k: (1e9 if isinstance(v, float) and np.isinf(v) else v) for k, v in best_params.items()}, indent=2), encoding="utf-8")
    (out / f"fit_summary_{args.train}_to_{args.test}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


# ============================================================================ run (score)
def apply_rule_to_crop(sub: pl.DataFrame, params: dict, drs) -> tuple[pl.DataFrame, list[dict], pd.DataFrame]:
    crop = crop_from_frame(sub)
    M, A, B = enumerate_candidates(crop, params.get("parent_max", PARENT_MAX_UM), params.get("sister_max", SISTER_MAX_UM))
    feat = candidate_features(crop, M, A, B, params.get("parent_max", PARENT_MAX_UM))
    mask = rule_mask(feat, params)
    plans = select_edits(feat, mask)
    edges_df = sub.filter(pl.col("row_type") == "edge")
    edges = set(zip(edges_df["source_id"].to_list(), edges_df["target_id"].to_list()))
    new_edges, applied = drs.apply_plans(edges, plans)
    return drs.rewrite_crop_edges(sub, new_edges), applied, feat.loc[mask]


def cmd_run(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {**DEFAULT_PARAMS, **json.loads(Path(args.params).read_text(encoding="utf-8"))} if args.params else dict(DEFAULT_PARAMS)
    for k, v in params.items():
        if isinstance(v, (int, float)) and v >= 1e9:
            params[k] = np.inf
    drs, ea, df = _load_export(Path(args.csv))
    sc = drs._scorer()
    summarise = sc[-1]
    names = drs.select_names(sorted(df["dataset"].unique().to_list()), args.max_crops, args.stride)
    print(f"{len(names)} crops; params={params}", flush=True)
    rows_c, rows_r, applied_all, frames, fired = [], [], [], [], []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            continue
        sub = df.filter(pl.col("dataset") == name)
        sub_r, applied, fired_df = apply_rule_to_crop(sub, params, drs)
        fired_df = fired_df.copy()
        fired_df.insert(0, "dataset", name)
        fired.append(fired_df)
        applied_all += [{**p, "dataset": name} for p in applied]
        frames.append(sub_r)
        rc = {"dataset": name, **drs.score_crop(sub, gt_geff, ea, sc)}
        rr = {"dataset": name, **drs.score_crop(sub_r, gt_geff, ea, sc)}
        rows_c.append(rc)
        rows_r.append(rr)
        print(f"  [{i}/{len(names)}] {name} applied={len(applied)} "
              f"div c={rc['division_tp']}/{rc['division_fp']}/{rc['division_fn']} r={rr['division_tp']}/{rr['division_fp']}/{rr['division_fn']} "
              f"edge c={rc['edge_tp']}/{rc['edge_fp']}/{rc['edge_fn']} r={rr['edge_tp']}/{rr['edge_fp']}/{rr['edge_fn']} ({time.time() - t0:.0f}s)", flush=True)
    tag = args.tag
    rule_df = pl.concat(frames, how="vertical").with_columns(pl.Series("id", np.arange(sum(f.height for f in frames), dtype=np.int64)))
    rule_csv = out / f"rule_{tag}.csv.gz"
    import gzip
    import shutil
    rule_df.write_csv(out / f"rule_{tag}.csv")
    with open(out / f"rule_{tag}.csv", "rb") as fin, gzip.open(rule_csv, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    (out / f"rule_{tag}.csv").unlink()
    pd.DataFrame(rows_c).to_parquet(out / f"crops_control_{tag}.parquet")
    pd.DataFrame(rows_r).to_parquet(out / f"crops_rule_{tag}.parquet")
    if fired:
        pd.concat(fired, ignore_index=True).to_parquet(out / f"fired_{tag}.parquet")
    pd.DataFrame(applied_all).to_parquet(out / f"applied_{tag}.parquet") if applied_all else None
    sc_c = summarise(rows_c)
    sc_r = summarise(rows_r)
    boot = drs.paired_bootstrap(rows_c, rows_r, summarise)
    keys = ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard", "division_tp", "division_fp", "division_fn")
    counts = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    delta = {k: (sc_r[k] - sc_c[k]) if isinstance(sc_r.get(k), (int, float)) and sc_r[k] == sc_r[k] and sc_c[k] == sc_c[k] else None for k in keys}
    delta.update({k: int(sum(r[k] for r in rows_r) - sum(r[k] for r in rows_c)) for k in counts})
    summary = {
        "tag": tag, "csv": str(args.csv), "params_file": str(args.params), "n_crops": len(rows_c),
        "params": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in params.items()},
        "control": sc_c, "rule": sc_r,
        "control_counts": {k: int(sum(r[k] for r in rows_c)) for k in counts},
        "rule_counts": {k: int(sum(r[k] for r in rows_r)) for k in counts},
        "delta": delta,
        "paired_bootstrap": boot,
        "applied": len(applied_all), "applied_steals": int(sum(1 for p in applied_all if p["remove"])),
        "applied_orphans": int(sum(1 for p in applied_all if not p["remove"])),
        "rule_csv": str(rule_csv),
    }
    (out / f"run_summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


# ============================================================================ econ (price of a false add)
def _oracle_plans_by_crop(plans_path: Path) -> dict[str, list[dict]]:
    pdf = pd.read_parquet(plans_path)
    out: dict[str, list[dict]] = {}
    for r in pdf.to_dict("records"):
        out.setdefault(r["dataset"], []).append({**r, "add": [tuple(x) for x in json.loads(r["add"])],
                                                 "remove": [tuple(x) for x in json.loads(r["remove"])]})
    return out


def random_false_plans(cands: pd.DataFrame, kind: str, n: int, seed: int = 0) -> dict[str, list[dict]]:
    """N random non-oracle candidates of one kind as plans, one per daughter and per mother."""
    pool = cands[(cands.b_has_parent == (kind == "steal")) & ~cands.oracle_pos]
    samp = pool.sample(min(n, len(pool)), random_state=seed).drop_duplicates("b_id").drop_duplicates("M_id")
    out: dict[str, list[dict]] = {}
    for r in samp.itertuples(index=False):
        out.setdefault(r.dataset, []).append({
            "route": "rand", "fork": int(r.M_id), "keep": [int(r.a_id)], "add": [(int(r.M_id), int(r.b_id))],
            "remove": [(int(r.P_id), int(r.b_id))] if r.b_has_parent else [],
            "daughter_kinds": ["rand"], "removed_edge_is_full_tp": False})
    return out


def cmd_econ(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    odir = Path(args.oracle_dir)
    cands = pd.read_parquet(Path(args.cands_dir) / f"cands_{args.tag}.parquet",
                            columns=["dataset", "M_id", "a_id", "b_id", "P_id", "b_has_parent", "oracle_pos",
                                     "pb_matched", "mb_is_gt_edge", "gt_M", "gt_b"])
    rand = random_false_plans(cands, args.kind, args.n, args.seed)
    oracle = _oracle_plans_by_crop(odir / f"plans_{args.tag}.parquet")
    rows_c = pd.read_parquet(odir / f"crops_control_{args.tag}.parquet").to_dict("records")
    rows_o = pd.read_parquet(odir / f"crops_oracle_{args.tag}.parquet").to_dict("records")
    drs, ea, df = _load_export(Path(args.csv))
    sc = drs._scorer()
    summarise = sc[-1]
    rows_r, n_applied, t0 = [], 0, time.time()
    for ro in rows_o:
        name = ro["dataset"]
        if name not in rand:
            rows_r.append(dict(ro))
            continue
        sub = df.filter(pl.col("dataset") == name)
        edges_df = sub.filter(pl.col("row_type") == "edge")
        edges = set(zip(edges_df["source_id"].to_list(), edges_df["target_id"].to_list()))
        edges, _ = drs.apply_plans(edges, oracle.get(name, []))
        edges, applied = drs.apply_plans(edges, rand[name])
        n_applied += len(applied)
        rr = {"dataset": name, **drs.score_crop(drs.rewrite_crop_edges(sub, edges), Path(args.gt_dir) / f"{name}.geff", ea, sc)}
        rows_r.append(rr)
        print(f"  {name} applied={len(applied)} div o={ro['division_tp']}/{ro['division_fp']}/{ro['division_fn']} "
              f"r={rr['division_tp']}/{rr['division_fp']}/{rr['division_fn']} ({time.time() - t0:.0f}s)", flush=True)
    so, sr, scc = drs._arm_summary(rows_o, summarise), drs._arm_summary(rows_r, summarise), drs._arm_summary(rows_c, summarise)
    keys = ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard", "division_tp", "division_fp", "division_fn", "edge_tp", "edge_fp", "edge_fn")
    n_true = sum(len(v) for v in oracle.values())
    res = {"tag": args.tag, "kind": args.kind, "n_sampled": int(sum(len(v) for v in rand.values())), "n_applied": n_applied,
           "control": scc, "oracle": so, "oracle_plus_random": sr,
           "delta_vs_oracle": {k: sr[k] - so[k] for k in keys},
           "per_false_add_score": (sr["score"] - so["score"]) / max(1, n_applied),
           "per_true_add_score": (so["score"] - scc["score"]) / max(1, n_true),
           "paired_bootstrap_vs_oracle": drs.paired_bootstrap(rows_o, rows_r, summarise),
           "note": "measured ON TOP of the oracle arm: the FP-fork price scales with the arm's division "
                   "Jaccard, so this overstates the cost relative to the control state by oracle_divJ/control_divJ"}
    (out / f"econ_{args.tag}_{args.kind}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("control", "oracle", "oracle_plus_random")}, indent=2, default=float))
    return 0


# ============================================================================ search (proxy + greedy)
def control_totals(oracle_summary: Path) -> dict:
    c = json.loads(Path(oracle_summary).read_text(encoding="utf-8"))["control"]
    De = c["edge_tp"] + c["edge_fp"] + c["edge_fn"]
    Dd = c["division_tp"] + c["division_fp"] + c["division_fn"]
    return {"De": De, "Je": c["edge_tp"] / De, "Dd": Dd, "Jd": c["division_tp"] / Dd}


def proxy_values(cands: pd.DataFrame, ctrl: dict, division_weight: float = 0.1) -> np.ndarray:
    """Linearised score delta of adding each candidate alone, from its evaluation labels:
      +w/Dd            oracle_pos (FN division -> TP)
      -w*Jd/Dd         annotated M, not oracle_pos (a charged FP fork)
      +1/De            M->b is a GT edge (FN -> TP);   -Je/De if not GT and M OR b annotated (FP edge)
      -1/De            P->b was matched (TP -> FN);    +Je/De if unmatched and P OR b annotated (FP removed)
    A predicted edge is charged when EITHER endpoint is matched (vendor metrics.py:194,
    ``pred_valid = out_valid | in_valid``); an edge with both endpoints unannotated is metric-free,
    and a fork is charged only at an annotated mother (FACT-0294)."""
    annM, annb, annP = cands.gt_M.to_numpy() >= 0, cands.gt_b.to_numpy() >= 0, cands.gt_P.to_numpy() >= 0
    has_p = cands.b_has_parent.to_numpy().astype(bool)
    op, mbgt, pbm = cands.oracle_pos.to_numpy(), cands.mb_is_gt_edge.to_numpy(), cands.pb_matched.to_numpy()
    gain_div, cost_fork = division_weight / ctrl["Dd"], division_weight * ctrl["Jd"] / ctrl["Dd"]
    gain_edge, cost_fpe = 1.0 / ctrl["De"], ctrl["Je"] / ctrl["De"]
    return (gain_div * op - cost_fork * (annM & ~op) + gain_edge * mbgt - cost_fpe * ((annM | annb) & ~mbgt)
            - gain_edge * pbm + cost_fpe * (has_p & (annP | annb) & ~pbm)).astype(np.float64)


def condition_menu(t: pd.DataFrame) -> dict[str, np.ndarray]:
    """GT-free threshold conditions the greedy search may conjoin."""
    m: dict[str, np.ndarray] = {}
    f = lambda c: t[c].to_numpy()  # noqa: E731
    fl = lambda c, nan: np.nan_to_num(t[c].to_numpy(dtype=float), nan=nan)  # noqa: E731
    for c in ("c1_has_pred", "mutual_nn_all", "nn_a_is_b", "nn_b_is_a", "nearest_t_is_m", "p_has_pred"):
        m[c] = f(c).astype(bool)
    m["both_continue"] = f("a_continues") & f("b_continues")
    for x in (0.0, 1.0, 1.5, 2.25): m[f"divergence>={x}"] = fl("divergence", -np.inf) >= x
    for x in (0.5, 0.0, -0.3, -0.5, -0.7): m[f"cos_daughters<={x}"] = fl("cos_daughters", 1.0) <= x
    for x in (3.0, 4.0, 5.0): m[f"d_m_mid<={x}"] = f("d_m_mid") <= x
    for x in (1, 2): m[f"n_t1_closer_than_b<={x}"] = f("n_t1_closer_than_b") <= x
    for x in (1, 2, 3): m[f"n_t_closer_than_m<={x}"] = f("n_t_closer_than_m") <= x
    for x in (2.0, 3.0, 4.0): m[f"d_ma<={x}"] = f("d_ma") <= x
    for x in (8.0, 9.0, 10.0, 11.0): m[f"d_mb<={x}"] = f("d_mb") <= x
    for x in (6.0, 8.0): m[f"d_mb>={x}"] = f("d_mb") >= x
    for x in (9.0, 10.0, 11.0, 12.0): m[f"d_ab<={x}"] = f("d_ab") <= x
    for x in (1.0, 2.0, 3.0): m[f"d_pb<={x}"] = fl("d_pb", 0.0) <= x
    for x in (1.0, 2.0): m[f"d_pb>={x}"] = fl("d_pb", 0.0) >= x
    for x in (4.0, 6.0, 10.0): m[f"ratio_mb_pb<={x}"] = fl("ratio_mb_pb", np.inf) <= x
    for x in (2, 5, 10): m[f"p_len_before>={x}"] = f("p_len_before") >= x
    for x in (5, 10): m[f"p_len_before<={x}"] = f("p_len_before") <= x
    for x in (2, 5, 10, 20): m[f"m_len_before>={x}"] = f("m_len_before") >= x
    for x in (2, 5, 10): m[f"b_len_after>={x}"] = f("b_len_after") >= x
    for x in (2, 5, 10): m[f"a_len_after>={x}"] = f("a_len_after") >= x
    for x in (4, 5, 6): m[f"n_t1_within_parent_max<={x}"] = f("n_t1_within_parent_max") <= x
    for x in (0.0, 0.3, 0.5): m[f"cos_mb_next>={x}"] = fl("cos_mb_next", -np.inf) >= x
    for x in (0.5, 0.2, 0.0): m[f"cos_ab_next<={x}"] = fl("cos_ab_next", np.inf) <= x
    for x in (6.0, 8.0): m[f"asym<={x}"] = f("asym") <= x
    for x in (1.5, 2.0, 2.5): m[f"d_ppred_b<={x}"] = fl("d_ppred_b", np.inf) <= x
    m["p_outdeg==1"] = f("p_outdeg") == 1
    m["p_no_alt_orphan10"] = fl("d_p_orphan", np.inf) > 10.0
    for x in (12.0, 13.0): m[f"rank_score<={x}"] = f("rank_score") <= x
    return m


def mask_stats(t: pd.DataFrame, v: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    op = int(t.oracle_pos.to_numpy()[mask].sum())
    return {"fired": n, "tp_oracle": op, "precision": (op / n) if n else float("nan"),
            "recall": op / max(1, int(t.oracle_pos.sum())), "proxy_dscore": float(v[mask].sum()),
            "mb_gt": int(t.mb_is_gt_edge.to_numpy()[mask].sum()), "pb_matched": int(t.pb_matched.to_numpy()[mask].sum()),
            "M_ann": int((t.gt_M.to_numpy() >= 0)[mask].sum())}


def greedy_conjunction(tr: pd.DataFrame, vtr: np.ndarray, te: pd.DataFrame, vte: np.ndarray, kind: str,
                       max_steps: int = 6, min_fired: int = 20) -> list[dict]:
    """Add the condition that most raises the TRAIN proxy at each step; report train and test at every step."""
    mtr, mte = condition_menu(tr), condition_menu(te)
    cur_tr = tr.b_has_parent.to_numpy() == (kind == "steal")
    cur_te = te.b_has_parent.to_numpy() == (kind == "steal")
    chosen: list[str] = []
    hist = [{"rule": [f"kind={kind}"], "train": mask_stats(tr, vtr, cur_tr), "test": mask_stats(te, vte, cur_te)}]
    for _ in range(max_steps):
        best = None
        for name, cond in mtr.items():
            if name in chosen:
                continue
            m = cur_tr & cond
            if m.sum() < min_fired:
                continue
            obj = vtr[m].sum()
            if best is None or obj > best[0]:
                best = (obj, name)
        if best is None or best[0] <= vtr[cur_tr].sum() + 1e-12:
            break
        chosen.append(best[1])
        cur_tr &= mtr[best[1]]
        cur_te &= mte[best[1]]
        hist.append({"rule": [f"kind={kind}"] + list(chosen), "train": mask_stats(tr, vtr, cur_tr), "test": mask_stats(te, vte, cur_te)})
    return hist


SEARCH_COLUMNS = ["dataset", "M_id", "b_id", "d_mb", "d_ab", "d_ma", "divergence", "d_pb", "ratio_mb_pb", "asym", "d_m_mid",
                  "cos_daughters", "d_ppred_b", "cos_mb_next", "cos_ab_next", "n_t1_closer_than_b", "n_t_closer_than_m",
                  "d_p_orphan", "n_t1_within_parent_max", "rank_score", "p_outdeg", "p_len_before", "b_len_after",
                  "a_len_after", "m_len_before", "c1_has_pred", "a_continues", "b_continues", "b_has_parent", "p_has_pred",
                  "nn_a_is_b", "nn_b_is_a", "mutual_nn_all", "nearest_t_is_m",
                  "oracle_pos", "gt_M", "gt_b", "gt_P", "mb_is_gt_edge", "pb_matched"]


def cmd_search(args) -> int:
    out = Path(args.out_dir)
    tabs, vals = {}, {}
    for tag, summ in ((args.train, args.control_train), (args.test, args.control_test)):
        t = pd.read_parquet(out / f"cands_{tag}.parquet", columns=SEARCH_COLUMNS)
        tabs[tag] = t
        vals[tag] = proxy_values(t, control_totals(Path(summ)))
    res = {}
    for tag in (args.train, args.test):
        t, v = tabs[tag], vals[tag]
        pos, hp = t.oracle_pos.to_numpy(), t.b_has_parent.to_numpy()
        res[f"ceiling_{tag}"] = {"all_positives_only": float(v[pos].sum()), "n_pos": int(pos.sum()),
                                 "steal_pos_value": float(v[pos & hp].sum()), "orphan_pos_value": float(v[pos & ~hp].sum())}
        for kind in ("steal", "orphan"):
            base = hp == (kind == "steal")
            b = mask_stats(t, v, base)
            rows = []
            for name, cond in condition_menu(t).items():
                s = mask_stats(t, v, base & cond)
                rows.append({"cond": name, **s, "lift": (s["precision"] / b["precision"]) if b["precision"] and s["fired"] else float("nan")})
            u = pd.DataFrame(rows).sort_values("proxy_dscore", ascending=False)
            u.to_csv(out / f"univariate_{tag}_{kind}.csv", index=False)
            res[f"univariate_top_{tag}_{kind}"] = u.head(8).to_dict("records")
    for kind in ("steal", "orphan"):
        h = greedy_conjunction(tabs[args.train], vals[args.train], tabs[args.test], vals[args.test], kind, args.max_steps, args.min_fired)
        res[f"greedy_{kind}_train{args.train}_test{args.test}"] = h
        print(f"=== greedy {kind} train {args.train} -> test {args.test}")
        for step in h:
            print("  " + " & ".join(step["rule"]) + f"\n     train {step['train']}\n     test  {step['test']}")
    (out / f"search_{args.train}_to_{args.test}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.startswith("ceiling")}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("econ")
    p.add_argument("--csv", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--kind", choices=["steal", "orphan"], required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle-dir", required=True, help="dir holding plans_/crops_control_/crops_oracle_<tag>.parquet")
    p.add_argument("--cands-dir", required=True)
    p.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_econ)
    p = sub.add_parser("search")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--control-train", required=True, help="oracle_summary_<train>.json (control totals)")
    p.add_argument("--control-test", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--min-fired", type=int, default=20)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("features")
    p.add_argument("--csv", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--plans", required=True)
    p.add_argument("--atlas-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parent-max", type=float, default=PARENT_MAX_UM)
    p.add_argument("--sister-max", type=float, default=SISTER_MAX_UM)
    p.add_argument("--max-crops", type=int)
    p.add_argument("--stride", type=int)
    p.set_defaults(func=cmd_features)
    p = sub.add_parser("fit")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fp-weight", type=float, default=1.0)
    p.add_argument("--orphan", action="store_true")
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("run")
    p.add_argument("--csv", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--params", help="json of rule parameters (from fit); default = base rule")
    p.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-crops", type=int)
    p.add_argument("--stride", type=int)
    p.set_defaults(func=cmd_run)
    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
