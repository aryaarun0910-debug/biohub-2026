"""Conservative test-time calibration of association candidates.

The module deliberately has no access to labels or images.  It uses only candidate
scores and geometry from one embryo, then discards all fitted state.  High-confidence
mutual links provide pseudo-positive anchors; their displacement field and two-step
paths calibrate the remaining candidate scores.  If the anchors are too sparse or
incoherent, adaptation is disabled and the unadapted ensemble is emitted unchanged.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SOURCE_ALIASES = ("source_id", "source_node_id")
TARGET_ALIASES = ("target_id", "target_node_id")
SCORE_ALIASES = ("p_tp", "score", "probability")


@dataclass(frozen=True)
class EdgeTable:
    source_id: np.ndarray
    target_id: np.ndarray
    source_t: np.ndarray
    target_t: np.ndarray
    source_zyx: np.ndarray
    target_zyx: np.ndarray
    model_names: tuple[str, ...]
    scores: np.ndarray  # (E, M), NaN when a model did not propose the edge
    p_fp: np.ndarray

    def __len__(self) -> int:
        return len(self.source_id)


@dataclass(frozen=True)
class AdaptConfig:
    high_score: float = 0.72
    margin: float = 0.08
    min_consensus_models: int = 2
    physical_gate_um_per_frame: float = 12.0
    path_gate_um_per_frame: float = 4.0
    min_pseudo: int = 50
    min_pseudo_frames: int = 4
    min_negatives: int = 50
    blend: float = 0.35
    ridge: float = 0.1
    max_score_drift: float = 0.18
    max_anchor_median_residual_um: float = 4.0
    flow_anchor_gate_um_per_frame: float = 2.0
    flow_anchor_margin_um_per_frame: float = 0.5
    seed: int = 20260710


@dataclass(frozen=True)
class AdaptResult:
    p_tp: np.ndarray
    base_p_tp: np.ndarray
    pseudo_positive: np.ndarray
    path_support: np.ndarray
    motion_residual_um: np.ndarray
    enabled: bool
    reason: str
    diagnostics: Mapping[str, object]


def _column(row: Mapping[str, str], aliases: Sequence[str], description: str) -> str:
    for name in aliases:
        if name in row and row[name] != "":
            return name
    raise ValueError(f"candidate CSV needs {description}; accepted columns: {list(aliases)}")


def _probability(value: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("candidate scores must be finite")
    # Trackastra versions differ between probabilities and logits.  Auto-sigmoid
    # only when the value cannot already be a probability.
    if x < 0.0 or x > 1.0:
        x = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))
    return min(1.0 - 1e-6, max(1e-6, x))


def _read_one(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    s_col = _column(rows[0], SOURCE_ALIASES, "source node id")
    d_col = _column(rows[0], TARGET_ALIASES, "target node id")
    score_col = _column(rows[0], SCORE_ALIASES, "association score")
    required_geometry = {"source_t", "target_t", "source_z", "source_y", "source_x", "target_z", "target_y", "target_x"}
    if not required_geometry.issubset(rows[0]):
        missing = sorted(required_geometry - set(rows[0]))
        raise ValueError(f"candidate CSV needs endpoint time/ZYX columns; missing {missing}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "key": (int(row[s_col]), int(row[d_col])),
                "score": _probability(row[score_col]),
                "p_fp": None if "p_fp" not in row or row["p_fp"] == "" else _probability(row["p_fp"]),
                "source_t": int(row["source_t"]),
                "target_t": int(row["target_t"]),
                "source_zyx": np.asarray([float(row[f"source_{a}"]) for a in "zyx"], dtype=np.float64),
                "target_zyx": np.asarray([float(row[f"target_{a}"]) for a in "zyx"], dtype=np.float64),
            }
        )
    keys = [r["key"] for r in parsed]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate source/target candidate in {path}")
    return parsed


def read_edge_models(
    inputs: Sequence[tuple[str, str | Path]],
    *,
    default_p_fp: float | None = None,
) -> EdgeTable:
    """Union model candidate CSVs and retain an explicit metric-FP estimate.

    Geometry is canonicalised from the first model containing an edge and checked
    across other models.  ``default_p_fp`` must be explicitly supplied if no input
    provides an OOF-calibrated p_fp; it is never inferred as ``1-p_tp``.
    """

    if not inputs:
        raise ValueError("at least one named model input is required")
    names = tuple(name for name, _ in inputs)
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    rows_by_model = [_read_one(Path(path)) for _, path in inputs]
    keys = sorted({r["key"] for rows in rows_by_model for r in rows})
    if not keys:
        empty = np.asarray([], dtype=np.int64)
        return EdgeTable(empty, empty, empty, empty, np.empty((0, 3)), np.empty((0, 3)), names, np.empty((0, len(names))), np.asarray([]))
    index = {key: i for i, key in enumerate(keys)}
    scores = np.full((len(keys), len(names)), np.nan, dtype=np.float64)
    geometry: dict[tuple[int, int], dict[str, object]] = {}
    fp_values: list[list[float]] = [[] for _ in keys]
    for m, rows in enumerate(rows_by_model):
        for row in rows:
            key = row["key"]
            i = index[key]
            scores[i, m] = float(row["score"])
            if row["p_fp"] is not None:
                fp_values[i].append(float(row["p_fp"]))
            if key in geometry:
                old = geometry[key]
                if old["source_t"] != row["source_t"] or old["target_t"] != row["target_t"]:
                    raise ValueError(f"models disagree on endpoint time for edge {key}")
                if not np.allclose(old["source_zyx"], row["source_zyx"], atol=1e-4) or not np.allclose(old["target_zyx"], row["target_zyx"], atol=1e-4):
                    raise ValueError(f"models disagree on endpoint coordinates for edge {key}")
            else:
                geometry[key] = row
    if default_p_fp is not None and not 0 <= default_p_fp <= 1:
        raise ValueError("default_p_fp must be in [0,1]")
    missing_fp = [keys[i] for i, v in enumerate(fp_values) if not v]
    if missing_fp and default_p_fp is None:
        raise ValueError(
            f"{len(missing_fp)} candidates have no p_fp. Supply an OOF-calibrated --default-p-fp explicitly; "
            "sparse annotation means p_fp != 1-p_tp."
        )
    fp = np.asarray([np.mean(v) if v else default_p_fp for v in fp_values], dtype=np.float64)
    return EdgeTable(
        source_id=np.asarray([k[0] for k in keys], dtype=np.int64),
        target_id=np.asarray([k[1] for k in keys], dtype=np.int64),
        source_t=np.asarray([geometry[k]["source_t"] for k in keys], dtype=np.int64),
        target_t=np.asarray([geometry[k]["target_t"] for k in keys], dtype=np.int64),
        source_zyx=np.asarray([geometry[k]["source_zyx"] for k in keys], dtype=np.float64),
        target_zyx=np.asarray([geometry[k]["target_zyx"] for k in keys], dtype=np.float64),
        model_names=names,
        scores=scores,
        p_fp=fp,
    )


def _mean_scores(scores: np.ndarray) -> np.ndarray:
    count = np.sum(np.isfinite(scores), axis=1)
    if np.any(count == 0):
        raise ValueError("candidate with no model score")
    return np.nansum(scores, axis=1) / count


def _mutual_votes(table: EdgeTable, cfg: AdaptConfig) -> tuple[np.ndarray, np.ndarray]:
    votes = np.zeros(len(table), dtype=np.int64)
    mutual_fraction = np.zeros(len(table), dtype=np.float64)
    for m in range(len(table.model_names)):
        valid = np.isfinite(table.scores[:, m])
        outgoing: dict[int, list[int]] = {}
        incoming: dict[int, list[int]] = {}
        for i in np.flatnonzero(valid):
            outgoing.setdefault(int(table.source_id[i]), []).append(int(i))
            incoming.setdefault(int(table.target_id[i]), []).append(int(i))
        model_mutual = np.zeros(len(table), dtype=bool)
        for i in np.flatnonzero(valid):
            out = outgoing[int(table.source_id[i])]
            inc = incoming[int(table.target_id[i])]
            out_rank = sorted((table.scores[j, m], -j, j) for j in out)
            inc_rank = sorted((table.scores[j, m], -j, j) for j in inc)
            if out_rank[-1][2] != i or inc_rank[-1][2] != i:
                continue
            out_second = out_rank[-2][0] if len(out_rank) > 1 else 0.0
            inc_second = inc_rank[-2][0] if len(inc_rank) > 1 else 0.0
            score = table.scores[i, m]
            if score >= cfg.high_score and score - out_second >= cfg.margin and score - inc_second >= cfg.margin:
                model_mutual[i] = True
        votes += model_mutual
        mutual_fraction += model_mutual.astype(float)
    mutual_fraction /= max(1, len(table.model_names))
    return votes, mutual_fraction


def _fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(z.shape[1], dtype=np.float64)
    # Deterministic full-batch Adam; tiny data and no dependency on sklearn/scipy.
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    rng = np.random.default_rng(seed)
    w[1:] = rng.normal(0, 1e-4, size=len(w) - 1)
    for step in range(1, 801):
        logits = np.clip(z @ w, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = z.T @ (pred - y) / len(y)
        grad[1:] += ridge * w[1:]
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1 - 0.9**step)
        vh = v / (1 - 0.999**step)
        w -= 0.03 * mh / (np.sqrt(vh) + 1e-8)
    return w, mean, scale


def adapt_candidates(
    table: EdgeTable,
    *,
    voxel_size_um: Sequence[float] = (1.625, 0.40625, 0.40625),
    config: AdaptConfig | None = None,
) -> AdaptResult:
    cfg = config or AdaptConfig()
    if len(table) == 0:
        empty = np.asarray([], dtype=np.float64)
        return AdaptResult(empty, empty, empty.astype(bool), empty, empty, False, "empty candidate set", {"enabled": False})
    voxel = np.asarray(voxel_size_um, dtype=np.float64)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError("voxel_size_um must contain three positive values")
    dt = table.target_t - table.source_t
    if np.any(dt <= 0):
        raise ValueError("all candidates must move forward in time")
    base = _mean_scores(table.scores)
    displacement = (table.target_zyx - table.source_zyx) * voxel / dt[:, None]
    speed = np.linalg.norm(displacement, axis=1)
    physical = speed <= cfg.physical_gate_um_per_frame
    votes, mutual_fraction = _mutual_votes(table, cfg)
    needed = min(max(1, cfg.min_consensus_models), len(table.model_names))
    pseudo = (votes >= needed) & physical

    unique_frames = np.unique(table.source_t[pseudo])
    diagnostics: dict[str, object] = {
        "candidate_count": len(table),
        "model_count": len(table.model_names),
        "pseudo_positive_count": int(pseudo.sum()),
        "pseudo_frames": int(len(unique_frames)),
        "consensus_models_required": int(needed),
    }
    fallback_residual = np.full(len(table), np.inf)
    fallback_path = np.zeros(len(table), dtype=np.float64)
    if pseudo.sum() < cfg.min_pseudo:
        reason = f"only {int(pseudo.sum())} pseudo positives; require {cfg.min_pseudo}"
        diagnostics.update(enabled=False, reason=reason)
        return AdaptResult(base, base, pseudo, fallback_path, fallback_residual, False, reason, diagnostics)
    if len(unique_frames) < cfg.min_pseudo_frames:
        reason = f"pseudo positives span {len(unique_frames)} frames; require {cfg.min_pseudo_frames}"
        diagnostics.update(enabled=False, reason=reason)
        return AdaptResult(base, base, pseudo, fallback_path, fallback_residual, False, reason, diagnostics)

    global_flow = np.median(displacement[pseudo], axis=0)
    flow_by_t: dict[int, np.ndarray] = {}
    for t in np.unique(table.source_t):
        local = pseudo & (table.source_t == t)
        # Shrink sparse time slices to the embryo-wide robust flow.
        flow_by_t[int(t)] = np.median(displacement[local], axis=0) if local.sum() >= 3 else global_flow
    expected = np.asarray([flow_by_t[int(t)] for t in table.source_t])
    motion_residual = np.linalg.norm(displacement - expected, axis=1)

    # Consensus scores can confidently prefer a source-domain motion pattern.
    # Once a coherent target flow is established, replace those provisional
    # anchors with unambiguous, flow-consistent outgoing candidates. This is the
    # actual target-domain adaptation step; merely calibrating the original
    # consensus labels cannot correct a systematic motion shift.
    flow_pseudo = np.zeros(len(table), dtype=bool)
    for source in np.unique(table.source_id):
        ids = np.flatnonzero((table.source_id == source) & physical)
        if not len(ids):
            continue
        order = ids[np.argsort(motion_residual[ids])]
        best = int(order[0])
        second = float(motion_residual[order[1]]) if len(order) > 1 else math.inf
        if (
            motion_residual[best] <= cfg.flow_anchor_gate_um_per_frame
            and second - motion_residual[best] >= cfg.flow_anchor_margin_um_per_frame
        ):
            flow_pseudo[best] = True
    pseudo = flow_pseudo
    unique_frames = np.unique(table.source_t[pseudo])
    diagnostics["flow_refined_pseudo_count"] = int(pseudo.sum())
    diagnostics["pseudo_frames"] = int(len(unique_frames))
    if pseudo.sum() < cfg.min_pseudo or len(unique_frames) < cfg.min_pseudo_frames:
        reason = "target flow did not yield enough unambiguous anchors"
        diagnostics.update(enabled=False, reason=reason)
        return AdaptResult(base, base, pseudo, fallback_path, motion_residual, False, reason, diagnostics)
    anchor_residual = float(np.median(motion_residual[pseudo]))
    diagnostics["anchor_median_residual_um"] = anchor_residual
    diagnostics["global_flow_zyx_um_per_frame"] = global_flow.tolist()
    if anchor_residual > cfg.max_anchor_median_residual_um:
        reason = f"incoherent anchors: median residual {anchor_residual:.3f} um"
        diagnostics.update(enabled=False, reason=reason)
        return AdaptResult(base, base, pseudo, fallback_path, motion_residual, False, reason, diagnostics)

    # Two-step (t -> t+1 -> t+2) consistency.  For each candidate, compare its
    # displacement with pseudo-positive links immediately before and after it.
    incoming_pseudo: dict[int, list[int]] = {}
    outgoing_pseudo: dict[int, list[int]] = {}
    for i in np.flatnonzero(pseudo):
        outgoing_pseudo.setdefault(int(table.source_id[i]), []).append(int(i))
        incoming_pseudo.setdefault(int(table.target_id[i]), []).append(int(i))
    path_residual = np.full(len(table), np.inf)
    path_pairs = 0
    for i in range(len(table)):
        neighbours = incoming_pseudo.get(int(table.source_id[i]), []) + outgoing_pseudo.get(int(table.target_id[i]), [])
        if neighbours:
            path_residual[i] = min(float(np.linalg.norm(displacement[i] - displacement[j])) for j in neighbours)
            path_pairs += len(neighbours)
    path_support = np.exp(-np.minimum(path_residual, 30.0) / max(1e-6, cfg.path_gate_um_per_frame))
    path_support[~np.isfinite(path_residual)] = 0.0
    diagnostics["two_step_path_pairs"] = int(path_pairs)

    # Competing edges touching an anchor endpoint are hard pseudo-negatives.
    positive_keys_source = set(int(v) for v in table.source_id[pseudo])
    positive_keys_target = set(int(v) for v in table.target_id[pseudo])
    negative = (~pseudo) & physical & np.asarray(
        [int(s) in positive_keys_source or int(t) in positive_keys_target for s, t in zip(table.source_id, table.target_id)],
        dtype=bool,
    )
    if negative.sum() < cfg.min_negatives:
        reason = f"only {int(negative.sum())} competing pseudo negatives; require {cfg.min_negatives}"
        diagnostics.update(enabled=False, reason=reason, pseudo_negative_count=int(negative.sum()))
        return AdaptResult(base, base, pseudo, path_support, motion_residual, False, reason, diagnostics)
    diagnostics["pseudo_negative_count"] = int(negative.sum())

    eps = 1e-5
    logit_base = np.log(np.clip(base, eps, 1 - eps) / np.clip(1 - base, eps, 1 - eps))
    agreement = np.sum(np.isfinite(table.scores), axis=1) / len(table.model_names)
    features = np.column_stack(
        [
            logit_base,
            agreement,
            mutual_fraction,
            np.exp(-motion_residual / max(1e-6, cfg.path_gate_um_per_frame)),
            path_support,
        ]
    )
    train = pseudo | negative
    y = pseudo[train].astype(np.float64)
    w, mean, scale = _fit_logistic(features[train], y, cfg.ridge, cfg.seed)
    z = (features - mean) / scale
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(w[0] + z @ w[1:], -30, 30)))
    evidence = min(1.0, pseudo.sum() / max(cfg.min_pseudo * 2.0, 1.0))
    alpha = cfg.blend * evidence
    adapted = (1.0 - alpha) * base + alpha * calibrated
    drift = float(np.mean(np.abs(adapted - base)))
    diagnostics.update(
        blend_used=float(alpha),
        mean_absolute_score_drift=drift,
        logistic_coefficients=w.tolist(),
        enabled=True,
    )
    if not np.isfinite(adapted).all() or drift > cfg.max_score_drift:
        reason = f"score-drift safeguard fired ({drift:.3f} > {cfg.max_score_drift:.3f})"
        diagnostics.update(enabled=False, reason=reason)
        return AdaptResult(base, base, pseudo, path_support, motion_residual, False, reason, diagnostics)
    reason = "adapted from coherent per-embryo consensus and two-step paths"
    diagnostics["reason"] = reason
    return AdaptResult(np.clip(adapted, 1e-6, 1 - 1e-6), base, pseudo, path_support, motion_residual, True, reason, diagnostics)


def write_adapted(path: str | Path, table: EdgeTable, result: AdaptResult) -> Path:
    """Write metric-solver-compatible candidates plus audit columns."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id", "target_id", "p_tp", "p_fp", "base_p_tp", "pseudo_positive",
        "path_support", "motion_residual_um", "adaptation_enabled",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(table)):
            writer.writerow(
                {
                    "source_id": int(table.source_id[i]),
                    "target_id": int(table.target_id[i]),
                    "p_tp": float(result.p_tp[i]),
                    "p_fp": float(table.p_fp[i]),
                    "base_p_tp": float(result.base_p_tp[i]),
                    "pseudo_positive": int(result.pseudo_positive[i]),
                    "path_support": float(result.path_support[i]),
                    "motion_residual_um": float(result.motion_residual_um[i]),
                    "adaptation_enabled": int(result.enabled),
                }
            )
    return path


def write_diagnostics(path: str | Path, embryo: str, result: AdaptResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embryo": embryo, "state_scope": "this embryo only; reset after invocation", **result.diagnostics}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
