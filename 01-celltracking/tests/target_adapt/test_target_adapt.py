from __future__ import annotations

import csv

import numpy as np

from scripts.target_adapt.core import AdaptConfig, EdgeTable, adapt_candidates, read_edge_models, write_adapted
from scripts.metric_solver.io import read_candidates


def shifted_motion_table(frames: int = 14) -> tuple[EdgeTable, set[tuple[int, int]]]:
    """Target moves +3 in Y; source model often prefers its old +1 motion."""
    source, target, st, tt, sz, tz, scores, truth = [], [], [], [], [], [], [], set()
    # Five independent trajectories provide enough anchors and two-step paths.
    for t in range(frames - 1):
        for track in range(5):
            s = t * 100 + track
            true = (t + 1) * 100 + track
            wrong = (t + 1) * 100 + 50 + track
            p_true = 0.93 if (t + track) % 3 else 0.58
            p_wrong = 0.42 if p_true > 0.9 else 0.82
            for dst, dy, p1, p2 in [(true, 3.0, p_true, p_true - 0.02), (wrong, 1.0, p_wrong, p_wrong + 0.01)]:
                source.append(s); target.append(dst); st.append(t); tt.append(t + 1)
                sz.append([0.0, float(track * 10 + 3 * t), 0.0])
                tz.append([0.0, float(track * 10 + 3 * t + dy), 0.0])
                scores.append([p1, p2])
            truth.add((s, true))
    n = len(source)
    return EdgeTable(
        np.asarray(source), np.asarray(target), np.asarray(st), np.asarray(tt),
        np.asarray(sz), np.asarray(tz), ("organizer", "trackastra"), np.asarray(scores), np.full(n, 0.2)
    ), truth


def top1_accuracy(table: EdgeTable, score: np.ndarray, truth: set[tuple[int, int]]) -> float:
    correct = total = 0
    for s in np.unique(table.source_id):
        ids = np.flatnonzero(table.source_id == s)
        i = ids[np.argmax(score[ids])]
        correct += (int(table.source_id[i]), int(table.target_id[i])) in truth
        total += 1
    return correct / total


def test_adaptation_recovers_target_motion_shift():
    table, truth = shifted_motion_table()
    cfg = AdaptConfig(min_pseudo=20, min_pseudo_frames=4, min_negatives=15, blend=0.75, max_score_drift=0.5)
    result = adapt_candidates(table, voxel_size_um=(1, 1, 1), config=cfg)
    assert result.enabled, result.reason
    before = top1_accuracy(table, result.base_p_tp, truth)
    after = top1_accuracy(table, result.p_tp, truth)
    assert before < 0.8
    assert after > before + 0.15
    assert result.diagnostics["two_step_path_pairs"] > 0


def test_sparse_consensus_falls_back_bit_for_bit():
    table, _ = shifted_motion_table(frames=3)
    result = adapt_candidates(table, voxel_size_um=(1, 1, 1), config=AdaptConfig(min_pseudo=100))
    assert not result.enabled
    np.testing.assert_array_equal(result.p_tp, result.base_p_tp)
    assert "pseudo positives" in result.reason


def test_incoherent_consensus_falls_back():
    table, _ = shifted_motion_table()
    # Make every high-confidence anchor follow a different displacement.
    target = table.target_zyx.copy()
    for i in range(len(target)):
        target[i, 1] += (i % 9) * 7
    incoherent = EdgeTable(table.source_id, table.target_id, table.source_t, table.target_t, table.source_zyx,
                           target, table.model_names, table.scores, table.p_fp)
    cfg = AdaptConfig(min_pseudo=20, min_pseudo_frames=4, min_negatives=10, max_anchor_median_residual_um=0.5)
    result = adapt_candidates(incoherent, voxel_size_um=(1, 1, 1), config=cfg)
    assert not result.enabled
    np.testing.assert_array_equal(result.p_tp, result.base_p_tp)


def test_csv_requires_explicit_fp_and_output_is_solver_compatible(tmp_path):
    fields = ["source_id", "target_id", "p_tp", "source_t", "target_t", "source_z", "source_y", "source_x", "target_z", "target_y", "target_x"]
    path = tmp_path / "model.csv"
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields); w.writeheader()
        w.writerow(dict(zip(fields, [1, 2, .8, 0, 1, 0, 0, 0, 0, 1, 0])))
    try:
        read_edge_models([("m", path)])
    except ValueError as exc:
        assert "p_fp" in str(exc)
    else:
        raise AssertionError("missing p_fp must not be silently inferred")
    table = read_edge_models([("m", path)], default_p_fp=0.15)
    result = adapt_candidates(table, voxel_size_um=(1, 1, 1), config=AdaptConfig(min_pseudo=5))
    out = write_adapted(tmp_path / "out.csv", table, result)
    candidates = read_candidates(out)
    assert candidates.p_tp.tolist() == result.base_p_tp.tolist()
    assert candidates.p_fp.tolist() == [0.15]
