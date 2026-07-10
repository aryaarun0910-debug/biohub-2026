import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric_numpy import Sample
from scripts.oracle_redetect.core import (
    BlindQuery,
    RedetectProposal,
    RedetectQuery,
    apply_redetection_proposals,
    blind_dangling_queries,
    blind_redetection_proposals,
    local_peaks,
    measure_oracles,
    query_rescue,
    select_redetection_proposals,
    track_conditioned_queries,
)


def sample(ids, times, positions, edges=()):
    return Sample(
        np.asarray(ids, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
        np.asarray(positions, dtype=float).reshape(-1, 3),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_candidate_and_endpoint_oracles_separate_linking_from_detection():
    gt = sample(
        [10, 11, 12], [0, 1, 2],
        [[2, 10, 10], [2, 11, 10], [2, 12, 10]],
        [(10, 11), (11, 12)],
    )
    pred = sample(
        [100, 101, 102], [0, 1, 2],
        [[2, 10, 10], [2, 11, 10], [2, 12, 10]],
        [(100, 101)],
    )
    result = measure_oracles(pred, gt, n_est=3)
    assert result["baseline"]["edge_tp"] == 1
    assert result["candidate_oracle"]["edge_tp"] == 1
    assert result["endpoint_oracle"]["edge_tp"] == 2
    assert result["candidate_edge_gap"] == 1
    assert result["assignment_endpoint_recall"] == 1.0
    assert result["endpoint_oracle"]["adj_edge_jaccard"] == 1.0


def test_endpoint_oracle_respects_actual_assignment_and_count_adjustment():
    gt = sample([1, 2], [0, 1], [[0, 0, 0], [0, 0, 0]], [(1, 2)])
    # Third node is unmatched but still enters the exact count adjustment.
    pred = sample([10, 20, 30], [0, 1, 1], [[0, 0, 0], [0, 0, 0], [20, 20, 20]])
    result = measure_oracles(pred, gt, n_est=2)
    assert result["endpoint_oracle"]["edge_jaccard"] == 1.0
    assert result["endpoint_oracle"]["adj_edge_jaccard"] == 0.95
    assert result["existential_endpoint_recall"] == 1.0


def test_no_candidate_endpoint_lowers_both_endpoint_ceilings():
    gt = sample([1, 2], [0, 1], [[0, 0, 0], [0, 0, 0]], [(1, 2)])
    pred = sample([10], [0], [[0, 0, 0]])
    result = measure_oracles(pred, gt, n_est=2)
    assert result["no_candidate_edges"] == 1
    assert result["assignment_endpoint_edges"] == 0
    assert result["endpoint_oracle"]["edge_tp"] == 0


def test_existential_ceiling_exposes_one_to_one_assignment_conflict():
    gt = sample(
        [1, 2, 3, 4], [0, 0, 1, 1],
        [[0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 2, 0]],
        [(1, 3), (2, 4)],
    )
    # A single t=0 prediction is within 7 um of both GT nodes but can match only one.
    pred = sample(
        [10, 30, 40], [0, 1, 1],
        [[0, 1, 0], [0, 0, 0], [0, 2, 0]],
    )
    result = measure_oracles(pred, gt, n_est=3)
    assert result["existential_endpoint_edges"] == 2
    assert result["assignment_endpoint_edges"] == 1
    assert result["lost_assignment_edges"] == 1
    assert result["endpoint_oracle"]["edge_tp"] == 1


def test_track_query_uses_prediction_velocity_then_peak_rescues():
    gt = sample(
        [1, 2, 3], [0, 1, 2],
        [[4, 8, 8], [4, 10, 8], [4, 12, 8]],
        [(1, 2), (2, 3)],
    )
    # The last GT endpoint is absent; the two predicted nodes provide velocity.
    pred = sample(
        [10, 20], [0, 1],
        [[4, 8, 8], [4, 10, 8]],
        [(10, 20)],
    )
    queries = track_conditioned_queries(pred, gt)
    assert len(queries) == 1
    q = queries[0]
    assert q.gt_node_id == 3
    assert q.motion_model == "constant_velocity"
    np.testing.assert_allclose(q.center_zyx, [4, 12, 8])

    frame = np.zeros((9, 24, 24), dtype=np.float32)
    frame[4, 12, 8] = 100
    peaks = local_peaks(frame, q.center_zyx, top_k=3)
    assert query_rescue(q, peaks[:1])


def test_local_peak_search_obeys_physical_ball():
    frame = np.zeros((8, 32, 32), dtype=np.float32)
    frame[2, 10, 10] = 50
    frame[7, 31, 31] = 1000  # brighter but outside the local search ball
    query = RedetectQuery(1, 0, np.array([2.0, 10.0, 10.0]), np.array([2.0, 10.0, 10.0]), "forward", "stationary")
    peaks = local_peaks(frame, query.center_zyx, top_k=1)
    assert len(peaks) == 1
    assert np.linalg.norm((peaks[0] - np.array([2, 10, 10])) * np.array([1.625, 0.40625, 0.40625])) < 2.0
    assert query_rescue(query, peaks)


def test_blind_queries_use_only_predicted_track_ends_and_starts():
    pred = sample(
        [10, 20], [1, 2],
        [[4, 8, 8], [4, 10, 8]],
        [(10, 20)],
    )
    queries = blind_dangling_queries(pred, n_frames=4)
    assert len(queries) == 2
    backward = next(q for q in queries if q.direction == "backward")
    forward = next(q for q in queries if q.direction == "forward")
    assert backward.t == 0 and backward.anchor_node_id == 10
    assert forward.t == 3 and forward.anchor_node_id == 20
    assert backward.motion_model == forward.motion_model == "constant_velocity"
    np.testing.assert_allclose(backward.center_zyx, [4, 6, 8])
    np.testing.assert_allclose(forward.center_zyx, [4, 12, 8])


def test_blind_peak_then_reconnect_improves_exact_metric_without_gt_in_query():
    gt = sample(
        [1, 2, 3], [0, 1, 2],
        [[4, 8, 8], [4, 10, 8], [4, 12, 8]],
        [(1, 2), (2, 3)],
    )
    pred = sample(
        [10, 20], [0, 1],
        [[4, 8, 8], [4, 10, 8]],
        [(10, 20)],
    )
    frames = np.zeros((3, 9, 24, 24), dtype=np.float32)
    frames[2, 4, 12, 8] = 100
    queries, proposals = blind_redetection_proposals(
        pred,
        n_frames=3,
        frame_reader=lambda t: frames[t],
        directions=("forward",),
        top_k_per_query=1,
    )
    assert len(queries) == len(proposals) == 1
    selected = select_redetection_proposals(
        pred, proposals, min_confidence=0.5, max_added_nodes=1, max_added_ratio=1.0
    )
    assert len(selected) == 1
    modified = apply_redetection_proposals(pred, selected)
    before = measure_oracles(pred, gt, n_est=3)["baseline"]
    after = measure_oracles(modified, gt, n_est=3)["baseline"]
    assert (before["edge_tp"], before["edge_fp"], before["edge_fn"]) == (1, 0, 1)
    assert (after["edge_tp"], after["edge_fp"], after["edge_fn"]) == (2, 0, 0)
    assert after["adj_edge_jaccard"] > before["adj_edge_jaccard"]


def test_blind_selection_enforces_confidence_budget_and_existing_node_dedup():
    pred = sample([10], [0], [[0, 0, 0]])
    q1 = BlindQuery(10, 1, np.array([0.0, 0.0, 0.0]), "forward", "stationary")
    proposals = [
        RedetectProposal(q1, np.array([0.0, 0.0, 0.0]), 10, 3.0),
        RedetectProposal(q1, np.array([0.0, 10.0, 0.0]), 9, 2.0),
        RedetectProposal(q1, np.array([0.0, 20.0, 0.0]), 8, 0.5),
    ]
    # Existing node is t=0, so it must not suppress a valid same-position t=1 peak.
    selected = select_redetection_proposals(
        pred,
        proposals,
        min_confidence=1.0,
        max_added_nodes=1,
        max_added_ratio=None,
        dedup_um=2.0,
    )
    assert len(selected) == 1
    assert selected[0].confidence == 3.0


def test_blind_selection_deduplicates_against_existing_same_time_node():
    pred = sample([10, 20], [0, 1], [[0, 0, 0], [0, 0, 0]])
    query = BlindQuery(10, 1, np.array([0.0, 0.0, 0.0]), "forward", "stationary")
    proposal = RedetectProposal(query, np.array([0.0, 1.0, 0.0]), 10, 3.0)
    selected = select_redetection_proposals(
        pred, [proposal], max_added_nodes=1, max_added_ratio=None, dedup_um=2.0
    )
    assert selected == []
