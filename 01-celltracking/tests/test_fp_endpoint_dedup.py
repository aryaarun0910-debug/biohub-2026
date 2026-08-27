from __future__ import annotations

import inspect

import pandas as pd
import polars as pl
import pytest

from scripts.win_bet import fp_endpoint_dedup as dedup


def _nodes(rows):
    return pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])


def _supported_pair(*, candidate_xyz=(0, 8, 0), keeper_xyz=(0, 0, 0)):
    # Keeper 12 is in a five-node continuation; candidate 90 is an interior leaf.
    nodes = _nodes([
        (10, 0, 0, 0, 0),
        (11, 1, 0, 0, 0),
        (12, 2, *keeper_xyz),
        (13, 3, 0, 0, 0),
        (14, 4, 0, 0, 0),
        (90, 2, *candidate_xyz),
        (91, 3, 0, 8, 0),
    ])
    edges = {(10, 11), (11, 12), (12, 13), (13, 14), (90, 91)}
    return nodes, edges


def _submission(nodes: pd.DataFrame, edges: set[tuple[int, int]]) -> pl.DataFrame:
    dataset = "toy"
    node_rows = pl.DataFrame({
        "id": list(range(len(nodes))),
        "dataset": [dataset] * len(nodes),
        "row_type": ["node"] * len(nodes),
        "node_id": nodes.node_id.tolist(),
        "t": nodes.t.tolist(),
        "z": nodes.z.tolist(),
        "y": nodes.y.tolist(),
        "x": nodes.x.tolist(),
        "source_id": [-1] * len(nodes),
        "target_id": [-1] * len(nodes),
    }).cast({c: pl.Int64 for c in ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id")})
    src, tgt = zip(*sorted(edges)) if edges else ((), ())
    edge_rows = pl.DataFrame({
        "id": [0] * len(src),
        "dataset": [dataset] * len(src),
        "row_type": ["edge"] * len(src),
        "node_id": [-1] * len(src),
        "t": [-1] * len(src),
        "z": [-1] * len(src),
        "y": [-1] * len(src),
        "x": [-1] * len(src),
        "source_id": list(src),
        "target_id": list(tgt),
    }).cast({c: pl.Int64 for c in ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id")})
    return pl.concat([node_rows, edge_rows], how="vertical")


def test_selector_is_gt_free_and_selects_only_the_supported_leaf():
    assert set(inspect.signature(dedup.select_duplicate_endpoints).parameters) == {"nodes", "edges", "params"}
    nodes, edges = _supported_pair()
    got = dedup.select_duplicate_endpoints(nodes, edges)
    assert got.candidate_id.tolist() == [90]
    assert got.keeper_id.tolist() == [12]
    assert got.reason.unique().tolist() == ["interior_leaf_mutual_nn_to_supported_continuation"]
    assert got.keeper_back_steps.tolist() == [2]
    assert got.keeper_forward_steps.tolist() == [2]


def test_physical_anisotropy_is_used_for_the_duplicate_radius():
    # 3 z voxels and 12 xy voxels are both 4.875 um and pass radius 5.
    z_nodes, edges = _supported_pair(candidate_xyz=(3, 0, 0))
    xy_nodes, _ = _supported_pair(candidate_xyz=(0, 12, 0))
    assert len(dedup.select_duplicate_endpoints(z_nodes, edges)) == 1
    assert len(dedup.select_duplicate_endpoints(xy_nodes, edges)) == 1
    far_nodes, _ = _supported_pair(candidate_xyz=(4, 0, 0))
    assert dedup.select_duplicate_endpoints(far_nodes, edges).empty


def test_boundary_leaf_and_division_keeper_are_protected():
    nodes, edges = _supported_pair()
    # Move the candidate to the crop's first frame; its missing parent is then expected.
    boundary = nodes.copy()
    boundary.loc[boundary.node_id.eq(90), "t"] = 0
    boundary.loc[boundary.node_id.eq(91), "t"] = 1
    assert dedup.select_duplicate_endpoints(boundary, edges).empty

    # A second child makes the keeper a division node; nearby daughters are not deduplicated.
    division_nodes = pd.concat([nodes, _nodes([(15, 3, 0, -8, 0)])], ignore_index=True)
    assert dedup.select_duplicate_endpoints(division_nodes, edges | {(12, 15)}).empty


def test_outgoing_leaf_at_max_frame_is_protected():
    # Synthetic topology isolates the max-frame branch of the boundary guard.  Turning that
    # guard off demonstrates the rest of the candidate conditions are satisfied.
    nodes = _nodes([
        (10, 0, 0, 0, 0), (11, 1, 0, 0, 0), (12, 2, 0, 0, 0),
        (13, 2, 0, 30, 0), (14, 2, 0, 60, 0),
        (91, 1, 0, 8, 0), (90, 2, 0, 8, 0),
    ])
    edges = {(10, 11), (11, 12), (12, 13), (13, 14), (91, 90)}
    assert dedup.select_duplicate_endpoints(nodes, edges).empty
    got = dedup.select_duplicate_endpoints(nodes, edges, {"protect_crop_boundaries": False})
    assert got.candidate_id.tolist() == [90]


def test_mutual_nearest_neighbour_gate_blocks_a_third_closer_detection():
    nodes, edges = _supported_pair(candidate_xyz=(0, 8, 0))
    nodes = pd.concat([nodes, _nodes([(92, 2, 0, 2, 0)])], ignore_index=True)
    # Candidate's nearest node remains keeper 12; keeper's nearest is now 92.
    assert dedup.select_duplicate_endpoints(nodes, edges).empty


def test_selector_is_row_order_and_node_id_invariant():
    nodes, edges = _supported_pair()
    baseline = dedup.select_duplicate_endpoints(nodes, edges)
    shuffled = dedup.select_duplicate_endpoints(nodes.sample(frac=1, random_state=7), set(reversed(sorted(edges))))
    pd.testing.assert_frame_equal(baseline, shuffled)

    remap = {old: old + 1000 for old in nodes.node_id}
    renumbered = nodes.assign(node_id=nodes.node_id.map(remap))
    renumbered_edges = {(remap[s], remap[t]) for s, t in edges}
    got = dedup.select_duplicate_endpoints(renumbered, renumbered_edges)
    assert got.candidate_id.tolist() == [remap[90]]
    assert got.keeper_id.tolist() == [remap[12]]
    assert got.distance_um.tolist() == baseline.distance_um.tolist()


def test_fork_child_is_never_deleted():
    nodes, edges = _supported_pair()
    # Turn candidate 90 into a leaf daughter of a separate fork at t=1.
    edges.remove((90, 91))
    nodes = pd.concat([nodes, _nodes([(50, 1, 0, 8, 0), (92, 2, 0, 30, 0)])], ignore_index=True)
    edges.add((50, 90))
    edges.add((50, 92))
    assert dedup.select_duplicate_endpoints(nodes, edges).empty


def test_apply_normalizes_edges_then_reconciles_telemetry():
    nodes, edges = _supported_pair()
    sub = _submission(nodes, edges)
    decisions = dedup.select_duplicate_endpoints(nodes, edges)
    out, removed, kept = dedup.apply_decisions(sub, decisions)
    assert removed == {(90, 91)}
    assert (90, 91) not in kept
    assert 90 not in out.filter(pl.col("row_type") == "node")["node_id"].to_list()
    assert out.filter(pl.col("row_type") == "edge").height == sub.filter(pl.col("row_type") == "edge").height - 1
    assert out["id"].to_list() == list(range(out.height))


def test_bad_graph_and_parameter_files_fail_loudly(tmp_path):
    nodes, edges = _supported_pair()
    with pytest.raises(RuntimeError, match="missing node"):
        dedup.select_duplicate_endpoints(nodes, edges | {(999, 12)})
    p = tmp_path / "params.json"
    p.write_text('{"radius_um": 8}', encoding="utf-8")
    with pytest.raises(ValueError, match="radius_um"):
        dedup.load_params(p)
    p.write_text('{"oracle_gt_id": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        dedup.load_params(p)
