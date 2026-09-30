"""Adversarial validation of the numpy EDGE metric vs the organizer (tracksdata).

The earlier 20-case check used only 4 early 44b6 crops and no assignment conflicts.
These hand-built cases stress the exact behaviors that decide our strategy: multiple
real nuclei within the 7um gate, assignment stealing, ties, anisotropic separation,
duplicate edges, and empty predictions. numpy edge_tp/fp/fn must match tracksdata exactly.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_cellmot.metrics import evaluate as td_evaluate  # noqa: E402

from biotrack.metric_numpy import SCALE, Sample, score_sample  # noqa: E402

S = np.asarray(SCALE)


def build(nodes, edges):
    """nodes: list of (node_id, t, z, y, x). edges: list of (src_id, tgt_id).
    Returns (Sample, tracksdata graph) with matching identities."""
    ids = np.array([n[0] for n in nodes], np.int64)
    t = np.array([n[1] for n in nodes], np.int64)
    zyx = np.array([[n[2], n[3], n[4]] for n in nodes], float)
    e = np.array(edges, np.int64).reshape(-1, 2)
    sample = Sample(node_ids=ids, t=t, zyx=zyx, edges=e)

    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(tt), "z": float(z), "y": float(y), "x": float(x)}
                                 for tt, (z, y, x) in zip(t, zyx)])
    idmap = {int(i): internal[k] for k, i in enumerate(ids)}
    if len(e):
        g.bulk_add_edges([{"source_id": idmap[int(a)], "target_id": idmap[int(b)]} for a, b in e])
    return sample, g, idmap


def assert_match(pred_nodes, pred_edges, gt_nodes, gt_edges, n_est=1000.0):
    ps, pg, _ = build(pred_nodes, pred_edges)
    gs, gg, _ = build(gt_nodes, gt_edges)
    ref = td_evaluate(pg, gg, scale=SCALE, max_distance=7.0)   # mutates pg
    got = score_sample(ps, gs, n_est)
    assert (got["edge_tp"], got["edge_fp"], got["edge_fn"]) == \
           (ref.edge_tp, ref.edge_fp, ref.edge_fn), \
        f"numpy {(got['edge_tp'], got['edge_fp'], got['edge_fn'])} != td {(ref.edge_tp, ref.edge_fp, ref.edge_fn)}"


def test_two_real_nuclei_within_7um_both_matched():
    # two GT nuclei 5um apart in y (both inside the 7um gate) -> both should match distinct preds
    gt_n = [(1, 0, 10, 100, 100), (2, 0, 10, 112.3, 100),  # 12.3*0.40625=5.0um apart
            (3, 1, 10, 100, 100), (4, 1, 10, 112.3, 100)]
    gt_e = [(1, 3), (2, 4)]
    pred_n = [(1, 0, 10, 100, 100), (2, 0, 10, 112.3, 100),
              (3, 1, 10, 100, 100), (4, 1, 10, 112.3, 100)]
    pred_e = [(1, 3), (2, 4)]
    assert_match(pred_n, pred_e, gt_n, gt_e)


def test_assignment_stealing_duplicate_closer():
    # GT node at y=100; correct-track pred 2um away, wrong duplicate 0.5um away.
    # matcher must pick the closer duplicate -> the correct edge can go unmatched.
    gt_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100)]
    gt_e = [(1, 2)]
    pred_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100),      # correct track
              (3, 1, 10, 101.23, 100)]                          # duplicate ~0.5um from GT#2
    pred_e = [(1, 2)]
    assert_match(pred_n, pred_e, gt_n, gt_e)


def test_crowded_cluster_permuted():
    # 3 GT and 3 pred all within 7um, shuffled -> optimal assignment must agree
    gt_n = [(i + 1, 0, 10, 100 + 4 * i, 100) for i in range(3)] + \
           [(i + 4, 1, 10, 100 + 4 * i, 100) for i in range(3)]
    gt_e = [(1, 4), (2, 5), (3, 6)]
    pred_n = [(1, 0, 10, 101, 100), (2, 0, 10, 108.5, 100), (3, 0, 10, 104.2, 100),
              (4, 1, 10, 100.6, 100), (5, 1, 10, 107.9, 100), (6, 1, 10, 104.6, 100)]
    pred_e = [(1, 4), (3, 6), (2, 5)]
    assert_match(pred_n, pred_e, gt_n, gt_e)


def test_anisotropic_z_separation():
    # two nuclei 3 z-voxels apart = 4.875um (inside gate). A single pred midway must match ONE.
    gt_n = [(1, 0, 6, 50, 50), (2, 0, 9, 50, 50), (3, 1, 6, 50, 50), (4, 1, 9, 50, 50)]
    gt_e = [(1, 3), (2, 4)]
    pred_n = [(1, 0, 6, 50, 50), (2, 1, 6, 50, 50)]  # only the z=6 lineage detected
    pred_e = [(1, 2)]
    assert_match(pred_n, pred_e, gt_n, gt_e)


def test_out_of_gate_no_match():
    # pred 10um from GT (beyond 7um) -> no match, edge is FN
    gt_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100)]
    gt_e = [(1, 2)]
    pred_n = [(1, 0, 10, 124.6, 100), (2, 1, 10, 124.6, 100)]  # 24.6*0.40625=10um
    pred_e = [(1, 2)]
    assert_match(pred_n, pred_e, gt_n, gt_e)


def test_empty_prediction_all_fn():
    gt_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100)]
    gt_e = [(1, 2)]
    assert_match([], [], gt_n, gt_e)


def test_duplicate_pred_edges_no_tp_inflation():
    gt_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100)]
    gt_e = [(1, 2)]
    pred_n = [(1, 0, 10, 100, 100), (2, 1, 10, 100, 100)]
    pred_e = [(1, 2), (1, 2)]  # duplicate edge must not count twice
    assert_match(pred_n, pred_e, gt_n, gt_e)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
