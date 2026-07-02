"""Pure-numpy reimplementation of the exact competition metric (edge term).

Faithful to the host algorithm (reverse-engineered from tracksdata):
- per-timepoint node matching = one-to-one assignment maximizing sum of weights
  w = 1/(1+d), over pairs within max_distance (physical um), scale-corrected;
- a predicted edge u->v is a TP iff u,v match GT nodes g_u,g_v AND GT has edge
  g_u->g_v (directed);
- a predicted edge is "valid" (counts as FP if not TP) iff its matched source has
  GT out-degree>0 OR its matched target has GT in-degree>0 (FP only in annotated regions);
- edge_fp = valid_pred - tp; edge_fn = gt_edges - tp; J = tp/(tp+fp+fn);
- adjusted J = max(0, J * (1 - 0.1 * (N_pred - N_est)/N_est)).

Runs with NO tracksdata/geff -> usable inside a Kaggle notebook for OOF scoring.
Validate against biotrack.metric (the tracksdata harness) with scripts/validate_numpy_metric.py
and tests/test_numpy_metric.py (adversarial assignment-conflict cases).

CONTRACT / SCOPE (important):
- This module implements the EDGE term ONLY (adjusted edge Jaccard + count penalty),
  validated EXACT vs the organizer implementation including adversarial matching.
- It does NOT implement the division term. For ANY division-sensitive go/no-go, use the
  authoritative tracksdata harness `biotrack.metric` (which computes divisions exactly).
  A validated numpy division term is a Phase-5 prerequisite (see reports/EXECUTION_PLAN.md).
- Since Phases 0-4 run divisions-OFF, the edge-only numpy metric is the correct fast gate there;
  `score()['score']` returns adj_edge_jaccard only and must not be read as the full LB score
  once divisions are enabled.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE = (1.625, 0.40625, 0.40625)
MAX_DISTANCE = 7.0
ALPHA = 0.1  # count-adjustment coefficient


@dataclass
class Sample:
    """One embryo's graph as flat arrays. node_ids are arbitrary unique ints."""
    node_ids: np.ndarray      # (N,) int
    t: np.ndarray             # (N,) int
    zyx: np.ndarray           # (N,3) float, raw voxel coords
    edges: np.ndarray         # (E,2) int, [source_id, target_id]


def _match_timepoint(pred_zyx: np.ndarray, gt_zyx: np.ndarray,
                     scale: np.ndarray, max_distance: float) -> list[tuple[int, int]]:
    """Return list of (pred_local_idx, gt_local_idx) matched within max_distance.

    One-to-one assignment maximizing sum of 1/(1+d) over in-gate pairs (host's
    min_weight_full_bipartite_matching(maximize=True), replicated via dense LSA).
    """
    if len(pred_zyx) == 0 or len(gt_zyx) == 0:
        return []
    d = np.linalg.norm(pred_zyx[:, None, :] * scale - gt_zyx[None, :, :] * scale, axis=2)
    ingate = d <= max_distance
    if not ingate.any():
        return []
    w = np.where(ingate, 1.0 / (1.0 + d), 0.0)
    ri, ci = linear_sum_assignment(w, maximize=True)
    return [(int(r), int(c)) for r, c in zip(ri, ci) if ingate[r, c]]


def match_nodes(pred: Sample, gt: Sample,
                scale=SCALE, max_distance=MAX_DISTANCE) -> dict[int, int]:
    """Match pred node_ids -> gt node_ids, per timepoint. Unmatched pred omitted."""
    scale = np.asarray(scale, float)
    matched: dict[int, int] = {}
    times = np.unique(np.concatenate([pred.t, gt.t])) if len(pred.t) and len(gt.t) else []
    for tt in times:
        pi = np.where(pred.t == tt)[0]
        gi = np.where(gt.t == tt)[0]
        if len(pi) == 0 or len(gi) == 0:
            continue
        for pl, gl in _match_timepoint(pred.zyx[pi], gt.zyx[gi], scale, max_distance):
            matched[int(pred.node_ids[pi[pl]])] = int(gt.node_ids[gi[gl]])
    return matched


def gt_candidate_within(pred: Sample, gt: Sample,
                        scale=SCALE, max_distance=MAX_DISTANCE) -> dict[int, bool]:
    """For each GT node, does ANY pred node at the same timepoint lie within max_distance?

    Independent of the one-to-one assignment. Used to separate 'no candidate detected'
    (a true detection miss) from 'candidate exists but lost the assignment' (arbitration).
    """
    scale = np.asarray(scale, float)
    out: dict[int, bool] = {}
    for tt in np.unique(gt.t):
        gi = np.where(gt.t == tt)[0]
        pi = np.where(pred.t == tt)[0]
        if len(pi) == 0:
            for g in gi:
                out[int(gt.node_ids[g])] = False
            continue
        d = np.linalg.norm(gt.zyx[gi][:, None, :] * scale - pred.zyx[pi][None, :, :] * scale, axis=2)
        near = (d <= max_distance).any(axis=1)
        for k, g in enumerate(gi):
            out[int(gt.node_ids[g])] = bool(near[k])
    return out


def _degrees(sample: Sample) -> tuple[dict[int, int], dict[int, int]]:
    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    for s, t in sample.edges:
        out_deg[int(s)] = out_deg.get(int(s), 0) + 1
        in_deg[int(t)] = in_deg.get(int(t), 0) + 1
    return out_deg, in_deg


def score_sample(pred: Sample, gt: Sample, n_est: float,
                 scale=SCALE, max_distance=MAX_DISTANCE) -> dict:
    """Per-sample edge metrics (edge_tp/fp/fn, jaccard, adjusted, node_recall, count ratio)."""
    matched = match_nodes(pred, gt, scale, max_distance)  # pred_id -> gt_id
    gt_edge_set = {(int(s), int(t)) for s, t in gt.edges}
    gt_out, gt_in = _degrees(gt)

    n_pred_nodes = len(pred.node_ids)
    gt_num_edges = len(gt.edges)

    # dedup predicted edges, keeping a matched copy if duplicates disagree (host behavior)
    seen: dict[tuple[int, int], bool] = {}
    for s, t in pred.edges:
        s, t = int(s), int(t)
        gs, gt_ = matched.get(s), matched.get(t)
        is_tp = gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set
        # valid = FP-eligible: matched src has GT out>0 OR matched tgt has GT in>0
        is_valid = (gs is not None and gt_out.get(gs, 0) > 0) or \
                   (gt_ is not None and gt_in.get(gt_, 0) > 0)
        prev = seen.get((s, t))
        # keep matched=True if any duplicate is matched; valid if any is valid
        seen[(s, t)] = (bool(prev[0]) or is_tp, bool(prev[1]) or is_valid) if prev else (is_tp, is_valid)

    edge_tp = sum(1 for tp, _ in seen.values() if tp)
    valid_pred = sum(1 for _, v in seen.values() if v)
    edge_fp = valid_pred - edge_tp
    edge_fn = gt_num_edges - edge_tp
    denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denom if denom > 0 else float("nan")

    if n_est and n_est == n_est and n_est > 0:
        node_ratio = (n_pred_nodes - n_est) / n_est
    else:
        node_ratio = float("nan")
    if edge_jaccard == edge_jaccard and node_ratio == node_ratio:
        adj = max(0.0, edge_jaccard * (1 - ALPHA * node_ratio))
    else:
        adj = float("nan")

    matched_gt_ids = {g for g in matched.values()}
    node_recall = len(matched_gt_ids) / len(gt.node_ids) if len(gt.node_ids) else float("nan")

    return {
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": edge_fn,
        "num_pred_nodes": n_pred_nodes, "total_node_ratio": node_ratio,
        "edge_jaccard": edge_jaccard, "adj_edge_jaccard": adj, "node_recall": node_recall,
    }


def summarise(rows: list[dict]) -> dict:
    """Run-level summary: adj_edge_jaccard weighted by w_i = tp+fp+fn; micro edge Jaccard."""
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {"n": 0, "edge_jaccard": float("nan"), "adj_edge_jaccard": float("nan"), "score": float("nan")}
    tp = sum(r["edge_tp"] for r in valid)
    fp = sum(r["edge_fp"] for r in valid)
    fn = sum(r["edge_fn"] for r in valid)
    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    tw = sum(weights)
    adj = sum(w * r["adj_edge_jaccard"] for w, r in zip(weights, adj_rows)) / tw if tw else float("nan")
    return {
        "n": len(valid),
        "edge_jaccard": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
        "adj_edge_jaccard": adj,
        "node_recall": sum(r["node_recall"] for r in valid) / len(valid),
        "score": adj,  # + 0.1 * division_jaccard once divisions are added
    }
