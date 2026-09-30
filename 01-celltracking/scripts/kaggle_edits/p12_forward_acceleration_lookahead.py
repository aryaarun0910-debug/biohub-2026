"""P12: bounded three-frame forward-acceleration evidence for motion assignment.

This file overrides only ``motion_relink_edges``.  P9's detections, learned edge
probabilities, two-pass Hungarian assignment, geometry, downstream repair, coupled
divisions, and output contract are unchanged.  For each already-admissible t->t+1
candidate, the sole treatment is a bounded cost bonus when at least one physical
t+2 continuation has low acceleration residual.

The treatment is deliberately fail-closed and fully instrumented.  With
``BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD=0`` the override is semantically identical
to P9's original function.
"""

import math
import os

import numpy as np
from scipy.optimize import linear_sum_assignment


P12_USE_FORWARD_ACCELERATION_LOOKAHEAD = (
    os.environ.get("BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD", "0") != "0"
)
P12_FORWARD_LOOKAHEAD_NEXT_STEP_UM = float(
    os.environ.get("BIOHUB_FORWARD_LOOKAHEAD_NEXT_STEP_UM", "10.0")
)
P12_FORWARD_LOOKAHEAD_MAX_ACCEL_UM = float(
    os.environ.get("BIOHUB_FORWARD_LOOKAHEAD_MAX_ACCEL_UM", "4.0")
)
P12_FORWARD_LOOKAHEAD_MAX_BONUS = float(
    os.environ.get("BIOHUB_FORWARD_LOOKAHEAD_MAX_BONUS", "0.20")
)


def p12_forward_acceleration_lookahead(
    source_id: int,
    target_id: int,
    node_time: dict[int, int],
    ids_by_t: dict[int, list[int]],
    position_um: dict[int, np.ndarray],
    next_step_gate_um: float,
) -> tuple[float | None, int]:
    """Return the smallest t->t+1->t+2 acceleration residual and support count."""
    source_pos = np.asarray(position_um[source_id], dtype=np.float64)
    target_pos = np.asarray(position_um[target_id], dtype=np.float64)
    target_t = int(node_time[target_id])
    current_velocity = target_pos - source_pos
    residuals: list[float] = []
    for next_id in ids_by_t.get(target_t + 1, []):
        next_pos = np.asarray(position_um[next_id], dtype=np.float64)
        next_velocity = next_pos - target_pos
        if float(np.linalg.norm(next_velocity)) > float(next_step_gate_um):
            continue
        residual = float(np.linalg.norm(next_velocity - current_velocity))
        if np.isfinite(residual):
            residuals.append(residual)
    if not residuals:
        return None, 0
    return min(residuals), len(residuals)


def p12_forward_acceleration_bonus(
    residual_um: float | None,
    max_accel_um: float,
    max_bonus: float,
) -> float:
    """Map residual 0..max_accel linearly onto max_bonus..0, bounded at both ends."""
    if (
        residual_um is None
        or not np.isfinite(residual_um)
        or max_accel_um <= 0
        or max_bonus <= 0
    ):
        return 0.0
    support = max(0.0, 1.0 - float(residual_um) / float(max_accel_um))
    return float(max_bonus) * support


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    """P9 motion relink plus exactly one bounded forward-lookahead cost term."""
    for key in (
        "forward_lookahead_evaluated_edges",
        "forward_lookahead_candidates",
        "forward_lookahead_supported_edges",
        "forward_lookahead_bonus_edges",
        "forward_lookahead_bonus_milli_sum",
    ):
        stats.setdefault(key, 0)

    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    node_time: dict[int, int] = {}
    for node_id, node in nodes_by_id.items():
        node_t = int(node["t"])
        node_time[node_id] = node_t
        ids_by_t.setdefault(node_t, []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (
                    source_pos - prev_pos
                )
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                candidate_cost = (
                    motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
                )

                if P12_USE_FORWARD_ACCELERATION_LOOKAHEAD:
                    stats["forward_lookahead_evaluated_edges"] += 1
                    residual_um, continuation_count = p12_forward_acceleration_lookahead(
                        source_id,
                        target_id,
                        node_time,
                        ids_by_t,
                        position_um,
                        P12_FORWARD_LOOKAHEAD_NEXT_STEP_UM,
                    )
                    stats["forward_lookahead_candidates"] += int(continuation_count)
                    if continuation_count > 0:
                        stats["forward_lookahead_supported_edges"] += 1
                    bonus = p12_forward_acceleration_bonus(
                        residual_um,
                        P12_FORWARD_LOOKAHEAD_MAX_ACCEL_UM,
                        P12_FORWARD_LOOKAHEAD_MAX_BONUS,
                    )
                    if bonus > 0.0:
                        candidate_cost -= bonus
                        stats["forward_lookahead_bonus_edges"] += 1
                        stats["forward_lookahead_bonus_milli_sum"] += int(
                            round(bonus * 1000.0)
                        )
                cost[i, j] = candidate_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for row, column in zip(row_ind, col_ind):
            if cost[row, column] >= big:
                continue
            matches.append(
                (
                    source_ids[int(row)],
                    target_ids[int(column)],
                    float(raw_dist[row, column]),
                    float(motion_dist[row, column]),
                    float(prob_matrix[row, column]),
                )
            )
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (
            ("tight", MOTION_RELINK_TIGHT_UM),
            ("relaxed", MOTION_RELINK_RELAXED_UM),
        ):
            pass_sources = [
                node_id for node_id in source_ids if node_id in unmatched_sources
            ]
            pass_targets = [
                node_id for node_id in target_ids if node_id in unmatched_targets
            ]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append(
                    (source_id, target_id, raw, motion, pass_name, prob)
                )
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_prob": prob,
                    "distance_um": raw,
                    "motion_distance_um": motion,
                    "motion_relinked": 1,
                    "motion_pass": pass_name,
                }
            )
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges
