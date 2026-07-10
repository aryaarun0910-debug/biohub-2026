from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts.metric_solver.core import (
    CandidateEdges,
    LogitCalibration,
    dinkelbach_solve,
    reweight_candidates,
    solve_lineage,
)
from scripts.metric_solver.io import Nodes, read_candidates, validate_candidates


def candidates(source, target, tp, fp) -> CandidateEdges:
    return CandidateEdges(np.array(source), np.array(target), np.array(tp), np.array(fp))


@pytest.mark.parametrize("backend", ["scip", "scipy"])
def test_lineage_capacities_and_positive_weights(backend: str) -> None:
    edges = candidates(
        source=[1, 1, 1, 2, 3],
        target=[10, 11, 12, 10, 13],
        tp=[0.9, 0.8, 0.7, 0.85, 0.1],
        fp=[0.1, 0.1, 0.1, 0.1, 0.9],
    )
    result = solve_lineage(edges, lambda_=0.5, backend=backend)
    # The global optimum yields contested target 10 to source 2, allowing source
    # 1 to spend its two-child capacity on targets 11 and 12.
    assert set(np.flatnonzero(result.selected)) == {1, 2, 3}
    assert result.objective == pytest.approx((0.8 - 0.05) + (0.7 - 0.05) + (0.85 - 0.05))


def test_greedy_fallback_is_feasible_but_can_be_suboptimal() -> None:
    edges = candidates(
        source=[1, 1, 1, 2],
        target=[10, 11, 12, 10],
        tp=[0.9, 0.8, 0.7, 0.85],
        fp=[0.1, 0.1, 0.1, 0.1],
    )
    greedy = solve_lineage(edges, lambda_=0.5, backend="greedy")
    exact = solve_lineage(edges, lambda_=0.5, backend="scipy")
    assert set(np.flatnonzero(greedy.selected)) == {0, 1}
    assert exact.objective > greedy.objective


def test_lambda_rejects_fp_risk_independently_of_tp() -> None:
    edges = candidates([1, 2], [10, 11], [0.8, 0.7], [0.9, 0.05])
    assert np.allclose(reweight_candidates(edges, 1.0), [-0.1, 0.65])
    result = solve_lineage(edges, lambda_=1.0, backend="scipy")
    assert result.selected.tolist() == [False, True]


def test_dinkelbach_finds_better_fraction_than_selecting_everything() -> None:
    # Both edges fit the lineage constraints. The risky edge has positive TP,
    # yet lowers T/(G+F), so fractional optimization must drop it.
    edges = candidates([1, 2], [10, 11], [0.9, 0.2], [0.05, 0.9])
    solved = dinkelbach_solve(edges, ground_truth_edges=1.0, backend="scipy")
    assert solved.converged
    assert solved.solution.selected.tolist() == [True, False]
    assert solved.lambda_ == pytest.approx(0.9 / 1.05)
    select_all_ratio = 1.1 / 1.95
    assert solved.solution.ratio(1.0) > select_all_ratio


def test_fp_is_not_inferred_without_explicit_policy(tmp_path) -> None:
    path = tmp_path / "edges.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "target_id", "p_tp"])
        writer.writeheader()
        writer.writerow({"source_id": 1, "target_id": 2, "p_tp": 0.8})
    with pytest.raises(ValueError, match="p_fp != 1-p_tp"):
        read_candidates(path)
    inferred = read_candidates(path, missing_p_fp="one-minus-p-tp")
    assert inferred.p_fp.tolist() == pytest.approx([0.2])


def test_nodes_validate_forward_time() -> None:
    nodes = Nodes({1: 0, 2: 1, 3: 0})
    validate_candidates(candidates([1], [2], [0.9], [0.1]), nodes)
    with pytest.raises(ValueError, match="forward in time"):
        validate_candidates(candidates([2], [3], [0.9], [0.1]), nodes)


def test_tp_and_fp_calibration_can_differ() -> None:
    calibration = LogitCalibration(temperature=1.0, bias=1.0)
    assert calibration(np.array([0.5]))[0] == pytest.approx(1 / (1 + np.exp(-1)))
