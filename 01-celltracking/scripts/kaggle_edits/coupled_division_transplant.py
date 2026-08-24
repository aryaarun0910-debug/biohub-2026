"""Coupled safe-division transplant for the P3 harmonic deployment.

This is a clean-room port of the mechanism present in both public 0.926 kernels:

* ``kunaldesale2408/biohub-cell-tracking`` (kernel id 130667075)
* ``rockerritesh/0-926-biohub-divsub`` (kernel id 131694028)

The public ``add_safe_divisions_postlink`` function is byte-identical between those
notebooks (SHA-256 ``b0d50a39e07624bc5a29207215fb4ec11908b49a8b60662e3527a513b1b35a5d``).
The mechanism is kept atomic: wider 8/11/10 um geometry, a mid-track-parent gate,
the public cKDTree nearest-orphan sister constraint, and t+2 divergence >= 2.25 um.

The only deliberate hardening is preservation of the deployed lineage contract:
every target has at most one parent and every source at most two children.  The
public function lacks the explicit source-degree guard that P3 already carries.

This file is inserted after P3's original ``add_safe_divisions_postlink`` definition,
so this later definition is the one called by ``filter_output_graph``.
"""

import os

import numpy as np
from scipy.spatial import cKDTree


COUPLED_DIV_DIVERGE_UM = float(os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25"))


def _coupled_inc(stats: dict[str, int], key: str, amount: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + int(amount)


def _coupled_assert_degrees(edges: list[dict[str, object]], stage: str) -> None:
    """Enforce the lineage-degree contract even outside the full P3 notebook."""
    out_degree: dict[int, int] = {}
    in_degree: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        out_degree[source_id] = out_degree.get(source_id, 0) + 1
        in_degree[target_id] = in_degree.get(target_id, 0) + 1
    bad_out = sorted(node_id for node_id, degree in out_degree.items() if degree > 2)
    bad_in = sorted(node_id for node_id, degree in in_degree.items() if degree > 1)
    if bad_out or bad_in:
        raise RuntimeError(
            f"degree invariant violated after {stage}: out-degree>2 on {bad_out[:8]} "
            f"(n={len(bad_out)}), in-degree>1 on {bad_in[:8]} (n={len(bad_in)})"
        )


def _coupled_point_um(node: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(node["z"]) * float(VOXEL_SCALE_UM[0]),
            float(node["y"]) * float(VOXEL_SCALE_UM[1]),
            float(node["x"]) * float(VOXEL_SCALE_UM[2]),
        ],
        dtype=float,
    )


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    """Add a second daughter only when all coupled 0.926-lineage gates agree."""
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {
        (int(edge["source_id"]), int(edge["target_id"])) for edge in edges
    }
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    out_degree_now = {
        source_id: len(source_edges) for source_id, source_edges in out_by_source.items()
    }

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [
            node_id
            for node_id in ids_by_t[t]
            if len(out_by_source.get(node_id, [])) == 1
        ]
        candidate_ids = [
            node_id
            for node_id in child_frame_ids
            if node_id not in incoming and node_id not in used_targets
        ]
        if not source_ids or not candidate_ids:
            continue

        # C2, matching the public implementation: for each linked child c1, retain
        # only its nearest orphan q.  Parent-q and c1-q are then independently gated.
        candidate_positions = np.asarray(
            [_coupled_point_um(nodes_by_id[candidate_id]) for candidate_id in candidate_ids],
            dtype=float,
        )
        candidate_tree = cKDTree(candidate_positions)

        def one_successor(node_id: int) -> int | None:
            node_edges = out_by_source.get(node_id, [])
            return int(node_edges[0]["target_id"]) if len(node_edges) == 1 else None

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                _coupled_inc(stats, "safe_division_rejected_existing_child")
                continue

            # C1: a proposed mother must already have a predecessor, excluding a
            # track-start association error from being promoted into a fork.
            if source_id not in incoming:
                _coupled_inc(stats, "safe_division_rejected_track_start")
                continue

            near_parent = candidate_tree.query_ball_point(
                _coupled_point_um(source), r=SAFE_DIV_MAX_UM
            )
            nearest_dist, nearest_index = candidate_tree.query(
                _coupled_point_um(existing_child)
            )
            mutual_candidate = (
                candidate_ids[int(nearest_index)]
                if float(nearest_dist) <= SAFE_DIV_SISTER_MAX_UM
                else None
            )

            for candidate_index in near_parent:
                candidate_id = candidate_ids[int(candidate_index)]
                if (source_id, candidate_id) in existing_edges:
                    continue
                if candidate_id != mutual_candidate:
                    _coupled_inc(stats, "safe_division_rejected_not_mutual")
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue

                # C3: both prospective daughters must continue to t+2 and their
                # separation must grow by at least 2.25 um (configurable, pinned by spec).
                existing_successor = one_successor(existing_child_id)
                candidate_successor = one_successor(candidate_id)
                if existing_successor is None or candidate_successor is None:
                    _coupled_inc(stats, "safe_division_rejected_no_t2")
                    continue
                next_existing = nodes_by_id.get(existing_successor)
                next_candidate = nodes_by_id.get(candidate_successor)
                if next_existing is None or next_candidate is None:
                    _coupled_inc(stats, "safe_division_rejected_no_t2")
                    continue
                if (
                    int(next_existing["t"]) != t + 2
                    or int(next_candidate["t"]) != t + 2
                ):
                    _coupled_inc(stats, "safe_division_rejected_no_t2")
                    continue
                divergence = edge_distance_um(next_existing, next_candidate) - sister_dist
                if divergence < COUPLED_DIV_DIVERGE_UM:
                    _coupled_inc(stats, "safe_division_rejected_divergence")
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append(
                    (score, source_id, candidate_id, parent_dist, sister_dist)
                )

        _coupled_inc(stats, "safe_division_candidates", len(proposals))
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                _coupled_inc(stats, "safe_division_skipped_cap")
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if out_degree_now.get(source_id, 0) >= 2:
                _coupled_inc(stats, "safe_division_skipped_outdegree")
                continue
            added.append(
                {
                    "source_id": source_id,
                    "target_id": candidate_id,
                    "edge_prob": None,
                    "distance_um": parent_dist,
                    "safe_division": 1,
                }
            )
            used_targets.add(candidate_id)
            out_degree_now[source_id] = out_degree_now.get(source_id, 0) + 1
            added_this_frame += 1

    if not added:
        return edges
    _coupled_inc(stats, "safe_divisions_added", len(added))
    result = [*edges, *added]
    _coupled_assert_degrees(result, "coupled division transplant")
    return result
