"""Metric-faithful edge ceilings and a small track-conditioned redetection audit.

The edge metric delegates matching and scoring to ``biotrack.metric_numpy``, which
is parity-tested against the organizer implementation.  Coordinates are raw ZYX
voxels and all gates/distances are evaluated in physical microns.

The redetection audit is deliberately diagnostic, not an inference pipeline: GT
is used to identify which dangling predicted endpoints correspond to a missed GT
endpoint, while the extrapolation and image peak selection use prediction/image
information only.  Consequently its rescue rate is conditional recall and says
nothing about the false-positive cost of querying every dangling track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from biotrack.metric_numpy import (
    MAX_DISTANCE,
    SCALE,
    Sample,
    gt_candidate_within,
    match_nodes,
    score_sample,
)


def graph_to_sample(graph) -> Sample:
    """Convert a tracksdata graph to the numpy metric's flat representation."""
    import tracksdata as td

    keys = td.DEFAULT_ATTR_KEYS
    na = graph.node_attrs(attr_keys=[keys.NODE_ID, "t", "z", "y", "x"])
    ids = np.asarray(na[keys.NODE_ID].to_list(), dtype=np.int64)
    times = np.asarray(na["t"].to_list(), dtype=np.int64)
    zyx = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], axis=1).astype(float)
    if graph.num_edges():
        ea = graph.edge_attrs(attr_keys=[keys.EDGE_SOURCE, keys.EDGE_TARGET])
        edges = np.stack(
            [np.asarray(ea[keys.EDGE_SOURCE].to_list()), np.asarray(ea[keys.EDGE_TARGET].to_list())], axis=1
        ).astype(np.int64)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    return Sample(ids, times, zyx, edges)


def load_sample(path: str | Path) -> Sample:
    from biotrack.metric import load_graph

    return graph_to_sample(load_graph(path))


def with_edges(sample: Sample, edges: Iterable[tuple[int, int]] | np.ndarray) -> Sample:
    arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64).reshape(-1, 2)
    return Sample(sample.node_ids, sample.t, sample.zyx, arr)


def _edge_set(sample: Sample) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in sample.edges}


def oracle_edges(
    pred: Sample,
    gt: Sample,
    candidate_edges: Iterable[tuple[int, int]] | np.ndarray | None = None,
    *,
    scale=SCALE,
    max_distance: float = MAX_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return (candidate-supported TP edges, endpoint-oracle edges, matching).

    Candidate-supported edges are the subset of the supplied candidate support
    whose two endpoints match a directed GT edge.  Endpoint-oracle edges connect
    the uniquely matched prediction endpoints for every recoverable GT edge,
    whether or not that edge was proposed.  Both therefore have zero edge FP.
    """
    matched = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    gt_edges = _edge_set(gt)
    candidates = pred.edges if candidate_edges is None else np.asarray(candidate_edges, dtype=np.int64).reshape(-1, 2)
    pred_ids = {int(x) for x in pred.node_ids}
    invalid = [(int(a), int(b)) for a, b in candidates if int(a) not in pred_ids or int(b) not in pred_ids]
    if invalid:
        raise ValueError(f"candidate edges reference unknown prediction node ids; first={invalid[0]}")

    candidate_tp = sorted(
        {
            (int(a), int(b))
            for a, b in candidates
            if matched.get(int(a)) is not None
            and matched.get(int(b)) is not None
            and (matched[int(a)], matched[int(b)]) in gt_edges
        }
    )
    gt_to_pred = {g: p for p, g in matched.items()}
    endpoint = sorted(
        (gt_to_pred[a], gt_to_pred[b]) for a, b in gt_edges if a in gt_to_pred and b in gt_to_pred
    )
    return (
        np.asarray(candidate_tp, dtype=np.int64).reshape(-1, 2),
        np.asarray(endpoint, dtype=np.int64).reshape(-1, 2),
        matched,
    )


def measure_oracles(
    pred: Sample,
    gt: Sample,
    n_est: float,
    candidate_edges: Iterable[tuple[int, int]] | np.ndarray | None = None,
    *,
    scale=SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Measure baseline, candidate-support, and endpoint edge ceilings.

    ``endpoint_oracle`` is an attainable edge score for the fixed node positions
    under the metric's actual one-to-one assignment. ``existential_endpoint`` is
    looser: it only asks whether any detection lies in each endpoint gate and can
    overstate attainability when detections compete in the assignment.
    """
    candidate_tp, endpoint_edges, matched = oracle_edges(
        pred, gt, candidate_edges, scale=scale, max_distance=max_distance
    )
    baseline = score_sample(pred, gt, n_est, scale=scale, max_distance=max_distance)
    candidate = score_sample(
        with_edges(pred, candidate_tp), gt, n_est, scale=scale, max_distance=max_distance
    )
    endpoint = score_sample(
        with_edges(pred, endpoint_edges), gt, n_est, scale=scale, max_distance=max_distance
    )

    near = gt_candidate_within(pred, gt, scale=scale, max_distance=max_distance)
    gt_edges = [(int(a), int(b)) for a, b in gt.edges]
    matched_gt = set(matched.values())
    existential = sum(bool(near.get(a)) and bool(near.get(b)) for a, b in gt_edges)
    assignment = sum(a in matched_gt and b in matched_gt for a, b in gt_edges)
    n_edges = len(gt_edges)
    no_candidate = sum(not (bool(near.get(a)) and bool(near.get(b))) for a, b in gt_edges)
    lost_assignment = existential - assignment
    candidate_gap = assignment - len(candidate_tp)

    return {
        "baseline": baseline,
        "candidate_oracle": candidate,
        "endpoint_oracle": endpoint,
        "gt_edges": n_edges,
        "candidate_tp_edges": len(candidate_tp),
        "assignment_endpoint_edges": assignment,
        "existential_endpoint_edges": existential,
        "candidate_edge_gap": candidate_gap,
        "lost_assignment_edges": lost_assignment,
        "no_candidate_edges": no_candidate,
        "assignment_endpoint_recall": assignment / n_edges if n_edges else float("nan"),
        "existential_endpoint_recall": existential / n_edges if n_edges else float("nan"),
    }


@dataclass(frozen=True)
class RedetectQuery:
    gt_node_id: int
    t: int
    center_zyx: np.ndarray
    truth_zyx: np.ndarray
    direction: str
    motion_model: str


@dataclass(frozen=True)
class BlindQuery:
    """An inference-valid search request generated only from a predicted graph."""

    anchor_node_id: int
    t: int
    center_zyx: np.ndarray
    direction: str
    motion_model: str


@dataclass(frozen=True)
class PeakCandidate:
    zyx: np.ndarray
    intensity: float
    confidence: float


@dataclass(frozen=True)
class RedetectProposal:
    query: BlindQuery
    zyx: np.ndarray
    intensity: float
    confidence: float


def track_conditioned_queries(
    pred: Sample,
    gt: Sample,
    *,
    scale=SCALE,
    max_distance: float = MAX_DISTANCE,
) -> list[RedetectQuery]:
    """Build a conditional audit set for no-candidate endpoints beside a matched track.

    A forward query extrapolates a matched predicted source into its GT successor's
    time; a backward query is symmetric.  Velocity comes only from the adjacent
    predicted track edge when available, otherwise the model is stationary.
    """
    matched = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    gt_to_pred = {g: p for p, g in matched.items()}
    near = gt_candidate_within(pred, gt, scale=scale, max_distance=max_distance)
    pidx = {int(n): i for i, n in enumerate(pred.node_ids)}
    gidx = {int(n): i for i, n in enumerate(gt.node_ids)}
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    for a, b in pred.edges:
        outgoing.setdefault(int(a), []).append(int(b))
        incoming.setdefault(int(b), []).append(int(a))

    def extrapolate(anchor: int, target_t: int, backward: bool) -> tuple[np.ndarray, str]:
        ai = pidx[anchor]
        at = int(pred.t[ai])
        center = pred.zyx[ai].copy()
        neighbours = outgoing.get(anchor, []) if backward else incoming.get(anchor, [])
        valid = []
        for other in neighbours:
            oi = pidx.get(other)
            if oi is None:
                continue
            ot = int(pred.t[oi])
            if (backward and ot > at) or ((not backward) and ot < at):
                valid.append((abs(ot - at), oi))
        if valid:
            _, oi = min(valid)
            dt = at - int(pred.t[oi])
            velocity = (pred.zyx[ai] - pred.zyx[oi]) / dt
            center = pred.zyx[ai] + velocity * (target_t - at)
            return center, "constant_velocity"
        return center, "stationary"

    best: dict[int, RedetectQuery] = {}
    for ga, gb in gt.edges:
        ga, gb = int(ga), int(gb)
        ta, tb = int(gt.t[gidx[ga]]), int(gt.t[gidx[gb]])
        if ga in gt_to_pred and not near.get(gb, False) and tb > ta:
            center, model = extrapolate(gt_to_pred[ga], tb, backward=False)
            q = RedetectQuery(gb, tb, center, gt.zyx[gidx[gb]].copy(), "forward", model)
            if gb not in best or (best[gb].motion_model == "stationary" and model == "constant_velocity"):
                best[gb] = q
        if gb in gt_to_pred and not near.get(ga, False) and ta < tb:
            center, model = extrapolate(gt_to_pred[gb], ta, backward=True)
            q = RedetectQuery(ga, ta, center, gt.zyx[gidx[ga]].copy(), "backward", model)
            if ga not in best or (best[ga].motion_model == "stationary" and model == "constant_velocity"):
                best[ga] = q
    return sorted(best.values(), key=lambda q: (q.t, q.gt_node_id))


def blind_dangling_queries(
    pred: Sample,
    n_frames: int,
    *,
    directions: tuple[str, ...] = ("forward", "backward"),
    allow_stationary: bool = True,
) -> list[BlindQuery]:
    """Generate next/previous-frame searches from predicted track ends/starts.

    No GT, estimated counts, or image intensities enter query generation. An end
    with an incoming temporal edge uses constant velocity; otherwise it is a
    stationary query when ``allow_stationary`` is enabled. Starts are symmetric.
    """
    allowed = set(directions)
    if not allowed <= {"forward", "backward"}:
        raise ValueError(f"invalid directions: {sorted(allowed - {'forward', 'backward'})}")
    pidx = {int(n): i for i, n in enumerate(pred.node_ids)}
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    for a, b in pred.edges:
        a, b = int(a), int(b)
        if a not in pidx or b not in pidx:
            continue
        ta, tb = int(pred.t[pidx[a]]), int(pred.t[pidx[b]])
        if tb > ta:
            outgoing.setdefault(a, []).append(b)
            incoming.setdefault(b, []).append(a)

    queries: list[BlindQuery] = []
    for node in sorted(pidx):
        i = pidx[node]
        at = int(pred.t[i])
        anchor = pred.zyx[i]
        if "forward" in allowed and not outgoing.get(node) and at + 1 < n_frames:
            priors = [p for p in incoming.get(node, []) if int(pred.t[pidx[p]]) < at]
            if priors:
                prior = max(priors, key=lambda p: int(pred.t[pidx[p]]))
                pi = pidx[prior]
                dt = at - int(pred.t[pi])
                center = anchor + (anchor - pred.zyx[pi]) / dt
                model = "constant_velocity"
            else:
                center, model = anchor.copy(), "stationary"
            if allow_stationary or model != "stationary":
                queries.append(BlindQuery(node, at + 1, center, "forward", model))
        if "backward" in allowed and not incoming.get(node) and at - 1 >= 0:
            futures = [p for p in outgoing.get(node, []) if int(pred.t[pidx[p]]) > at]
            if futures:
                future = min(futures, key=lambda p: int(pred.t[pidx[p]]))
                fi = pidx[future]
                dt = int(pred.t[fi]) - at
                center = anchor - (pred.zyx[fi] - anchor) / dt
                model = "constant_velocity"
            else:
                center, model = anchor.copy(), "stationary"
            if allow_stationary or model != "stationary":
                queries.append(BlindQuery(node, at - 1, center, "backward", model))
    return sorted(queries, key=lambda q: (q.t, q.anchor_node_id, q.direction))


def local_peaks(
    frame: np.ndarray,
    center_zyx: np.ndarray,
    *,
    scale=SCALE,
    search_radius_um: float = MAX_DISTANCE,
    smooth_sigma_um: float = 0.8,
    top_k: int = 5,
    nms_um: float = 2.0,
) -> np.ndarray:
    """Return the strongest local maxima inside a physical-radius search ball."""
    peaks = local_peak_candidates(
        frame,
        center_zyx,
        scale=scale,
        search_radius_um=search_radius_um,
        smooth_sigma_um=smooth_sigma_um,
        top_k=top_k,
        nms_um=nms_um,
    )
    return np.asarray([p.zyx for p in peaks], dtype=float).reshape(-1, 3)


def local_peak_candidates(
    frame: np.ndarray,
    center_zyx: np.ndarray,
    *,
    scale=SCALE,
    search_radius_um: float = MAX_DISTANCE,
    smooth_sigma_um: float = 0.8,
    top_k: int = 5,
    nms_um: float = 2.0,
) -> list[PeakCandidate]:
    """Return physically NMSed local maxima with a patch-robust confidence.

    Confidence is ``(peak - patch median) / (patch p99 - patch median)``. It is
    intentionally simple and calibration-free; OOF sweeps must choose its gate.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"expected a 3D ZYX frame, got shape {frame.shape}")
    scale = np.asarray(scale, dtype=float)
    center = np.asarray(center_zyx, dtype=float)
    radius_vox = np.ceil(search_radius_um / scale).astype(int)
    lo = np.maximum(0, np.floor(center).astype(int) - radius_vox)
    hi = np.minimum(np.asarray(frame.shape), np.ceil(center).astype(int) + radius_vox + 1)
    if np.any(lo >= hi):
        return []
    patch = frame[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].astype(np.float32)
    smoothed = gaussian_filter(patch, sigma=np.maximum(0.01, smooth_sigma_um / scale), mode="nearest")
    footprint_size = np.maximum(1, 2 * np.floor((nms_um / 2) / scale).astype(int) + 1)
    is_peak = smoothed == maximum_filter(smoothed, size=tuple(int(x) for x in footprint_size), mode="nearest")
    coords = np.argwhere(is_peak).astype(float) + lo
    if not len(coords):
        return []
    inside = np.linalg.norm((coords - center) * scale, axis=1) <= search_radius_um
    coords = coords[inside]
    if not len(coords):
        return []
    values = smoothed[tuple((coords - lo).astype(int).T)]
    order = np.argsort(values)[::-1]
    kept: list[PeakCandidate] = []
    median = float(np.median(smoothed))
    p99 = float(np.percentile(smoothed, 99))
    denom = max(p99 - median, np.finfo(np.float32).eps)
    for i in order:
        c = coords[i]
        if all(np.linalg.norm((c - k.zyx) * scale) >= nms_um for k in kept):
            value = float(values[i])
            kept.append(PeakCandidate(c, value, (value - median) / denom))
            if len(kept) >= top_k:
                break
    return kept


def blind_redetection_proposals(
    pred: Sample,
    n_frames: int,
    frame_reader: Callable[[int], np.ndarray],
    *,
    directions: tuple[str, ...] = ("forward", "backward"),
    allow_stationary: bool = True,
    search_radius_um: float = MAX_DISTANCE,
    smooth_sigma_um: float = 0.8,
    top_k_per_query: int = 3,
    peak_nms_um: float = 2.0,
    max_queries: int | None = None,
    scale=SCALE,
) -> tuple[list[BlindQuery], list[RedetectProposal]]:
    """Run image peak search for blind dangling-track queries, grouped by frame."""
    queries = blind_dangling_queries(
        pred, n_frames, directions=directions, allow_stationary=allow_stationary
    )
    if max_queries is not None and len(queries) > max_queries:
        # Deterministically cover the whole movie instead of biasing the guard to
        # early frames (queries are time-sorted).
        keep = np.linspace(0, len(queries) - 1, max_queries, dtype=int)
        queries = [queries[int(i)] for i in keep]
    by_time: dict[int, list[BlindQuery]] = {}
    for query in queries:
        by_time.setdefault(query.t, []).append(query)
    proposals: list[RedetectProposal] = []
    for t, frame_queries in sorted(by_time.items()):
        frame = np.asarray(frame_reader(t))
        for query in frame_queries:
            peaks = local_peak_candidates(
                frame,
                query.center_zyx,
                scale=scale,
                search_radius_um=search_radius_um,
                smooth_sigma_um=smooth_sigma_um,
                top_k=top_k_per_query,
                nms_um=peak_nms_um,
            )
            proposals.extend(
                RedetectProposal(query, peak.zyx, peak.intensity, peak.confidence) for peak in peaks
            )
    return queries, proposals


def select_redetection_proposals(
    pred: Sample,
    proposals: Iterable[RedetectProposal],
    *,
    min_confidence: float = 1.0,
    max_added_nodes: int | None = 500,
    max_added_ratio: float | None = 0.02,
    dedup_um: float = 2.0,
    scale=SCALE,
) -> list[RedetectProposal]:
    """Globally rank candidates, enforcing confidence, count budget, and 3D dedup."""
    proposals = list(proposals)
    budgets = []
    if max_added_nodes is not None:
        budgets.append(max(0, max_added_nodes))
    if max_added_ratio is not None:
        budgets.append(max(0, int(np.floor(len(pred.node_ids) * max_added_ratio))))
    budget = min(budgets) if budgets else len(proposals)
    if budget == 0:
        return []
    pidx_by_t: dict[int, np.ndarray] = {}
    for t in np.unique(pred.t):
        pidx_by_t[int(t)] = pred.zyx[pred.t == t]
    selected: list[RedetectProposal] = []
    for proposal in sorted(proposals, key=lambda p: p.confidence, reverse=True):
        if proposal.confidence < min_confidence:
            continue
        existing = pidx_by_t.get(proposal.query.t, np.empty((0, 3)))
        if len(existing) and np.min(np.linalg.norm((existing - proposal.zyx) * np.asarray(scale), axis=1)) < dedup_um:
            continue
        same_t = [p for p in selected if p.query.t == proposal.query.t]
        if same_t and min(np.linalg.norm((p.zyx - proposal.zyx) * np.asarray(scale)) for p in same_t) < dedup_um:
            continue
        selected.append(proposal)
        if len(selected) >= budget:
            break
    return selected


def apply_redetection_proposals(pred: Sample, proposals: Iterable[RedetectProposal]) -> Sample:
    """Add one node and one anchor edge per selected proposal."""
    selected = list(proposals)
    if not selected:
        return with_edges(pred, pred.edges.copy())
    next_id = int(np.max(pred.node_ids)) + 1 if len(pred.node_ids) else 0
    ids = list(map(int, pred.node_ids))
    times = list(map(int, pred.t))
    positions = [np.asarray(p, dtype=float) for p in pred.zyx]
    edges = [(int(a), int(b)) for a, b in pred.edges]
    for offset, proposal in enumerate(selected):
        node_id = next_id + offset
        ids.append(node_id)
        times.append(proposal.query.t)
        positions.append(np.asarray(proposal.zyx, dtype=float))
        if proposal.query.direction == "forward":
            edges.append((proposal.query.anchor_node_id, node_id))
        elif proposal.query.direction == "backward":
            edges.append((node_id, proposal.query.anchor_node_id))
        else:
            raise ValueError(f"unknown direction: {proposal.query.direction}")
    return Sample(
        np.asarray(ids, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
        np.asarray(positions, dtype=float).reshape(-1, 3),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def count_multiplier(n_pred: int, n_est: float) -> float:
    if not n_est or not np.isfinite(n_est) or n_est <= 0:
        return float("nan")
    return max(0.0, 1.0 - 0.1 * (n_pred - n_est) / n_est)


def query_rescue(query: RedetectQuery, peaks: np.ndarray, *, scale=SCALE, gate_um=MAX_DISTANCE) -> bool:
    if not len(peaks):
        return False
    return bool(np.min(np.linalg.norm((np.asarray(peaks) - query.truth_zyx) * np.asarray(scale), axis=1)) <= gate_um)
