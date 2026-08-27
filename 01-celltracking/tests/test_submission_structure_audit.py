"""Fail-closed numeric and row contracts for submission.csv structural audits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "d1"))
import audit_submission_structure as audit_mod  # noqa: E402


def valid_frame() -> pd.DataFrame:
    rows = []
    for dataset in audit_mod.EXPECTED_DATASETS:
        rows.extend([
            {
                "dataset": dataset, "row_type": "node", "node_id": 0,
                "t": 0, "z": 1, "y": 2, "x": 3, "source_id": -1,
                "target_id": -1,
            },
            {
                "dataset": dataset, "row_type": "node", "node_id": 1,
                "t": 1, "z": 2, "y": 3, "x": 4, "source_id": -1,
                "target_id": -1,
            },
            {
                "dataset": dataset, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0,
                "target_id": 1,
            },
        ])
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    return frame[audit_mod.COLUMNS]


def chain_frame(nodes_per_dataset: int | dict[str, int]) -> pd.DataFrame:
    """Structurally valid lineage chains with configurable per-dataset population."""
    rows = []
    for dataset in audit_mod.EXPECTED_DATASETS:
        count = (
            nodes_per_dataset[dataset]
            if isinstance(nodes_per_dataset, dict)
            else nodes_per_dataset
        )
        for node_id in range(count):
            rows.append({
                "dataset": dataset, "row_type": "node", "node_id": node_id,
                "t": node_id, "z": 1, "y": 2, "x": 3,
                "source_id": -1, "target_id": -1,
            })
        for source_id in range(count - 1):
            rows.append({
                "dataset": dataset, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": source_id, "target_id": source_id + 1,
            })
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    return frame[audit_mod.COLUMNS]


def run_audit(tmp_path: Path, frame: pd.DataFrame) -> dict:
    path = tmp_path / "submission.csv"
    frame.to_csv(path, index=False)
    return audit_mod.audit(path)


def check(report: dict, name: str) -> dict:
    return next(row for row in report["checks"] if row["check"] == name)


def test_valid_integer_and_sentinel_contract_passes(tmp_path: Path) -> None:
    report = run_audit(tmp_path, valid_frame())
    assert report["verdict"] == "PASS"
    assert check(report, "A1 numeric")["pass"]
    assert check(report, "A1 row_contract")["pass"]


@pytest.mark.parametrize(
    ("row_type", "column"),
    [
        ("node", "node_id"),
        ("node", "t"),
        ("node", "z"),
        ("node", "y"),
        ("node", "x"),
        ("edge", "source_id"),
        ("edge", "target_id"),
    ],
)
def test_fractional_graph_values_fail_numeric_contract(
    tmp_path: Path, row_type: str, column: str,
) -> None:
    frame = valid_frame()
    index = frame.index[frame["row_type"].eq(row_type)][0]
    frame[column] = frame[column].astype(float)
    frame.loc[index, column] += 0.5

    report = run_audit(tmp_path, frame)
    assert report["verdict"] == "FAIL"
    assert not check(report, "A1 numeric")["pass"]
    assert "fractional=1" in check(report, "A1 numeric")["detail"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "not-a-number"])
def test_non_numeric_or_nonfinite_values_fail_without_truncation(
    tmp_path: Path, bad: object,
) -> None:
    frame = valid_frame()
    frame["t"] = frame["t"].astype(object)
    frame.loc[frame.index[frame["row_type"].eq("node")][0], "t"] = bad

    report = run_audit(tmp_path, frame)
    assert report["verdict"] == "FAIL"
    assert not check(report, "A1 numeric")["pass"]
    assert "non_numeric_or_nonfinite=1" in check(report, "A1 numeric")["detail"]


@pytest.mark.parametrize(
    ("row_type", "column", "bad"),
    [
        ("node", "source_id", 0),
        ("node", "target_id", 0),
        ("edge", "node_id", 0),
        ("edge", "t", 0),
        ("edge", "z", 0),
        ("edge", "y", 0),
        ("edge", "x", 0),
        ("edge", "source_id", -1),
        ("edge", "target_id", -1),
    ],
)
def test_row_type_sentinel_misuse_fails(
    tmp_path: Path, row_type: str, column: str, bad: int,
) -> None:
    frame = valid_frame()
    index = frame.index[frame["row_type"].eq(row_type)][0]
    frame.loc[index, column] = bad

    report = run_audit(tmp_path, frame)
    assert report["verdict"] == "FAIL"
    assert not check(report, "A1 row_contract")["pass"]


@pytest.mark.parametrize(
    ("column", "bad"),
    [("dataset", None), ("dataset", ""), ("row_type", None), ("row_type", "vertex")],
)
def test_missing_or_unknown_row_identity_fails(
    tmp_path: Path, column: str, bad: object,
) -> None:
    frame = valid_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = bad

    report = run_audit(tmp_path, frame)
    assert report["verdict"] == "FAIL"
    assert not check(report, "A1 row_contract")["pass"] or not check(
        report, "A1 schema"
    )["pass"]


def test_no_baseline_keeps_original_audit_behavior(tmp_path: Path) -> None:
    report = run_audit(tmp_path, chain_frame(1))
    assert report["verdict"] == "PASS"
    assert "baseline_plausibility" not in report
    assert not any(row["check"].startswith("B") for row in report["checks"])


def test_explicit_baseline_passes_when_retention_is_above_bounds(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    chain_frame(10).to_csv(baseline_path, index=False)
    chain_frame(7).to_csv(candidate_path, index=False)

    report = audit_mod.audit(candidate_path, baseline=baseline_path)
    assert report["verdict"] == "PASS"
    assert check(report, "B0 baseline")["pass"]
    assert check(report, "B1 total_retention")["pass"]
    assert check(report, "B2 dataset_coverage")["pass"]
    assert report["baseline_plausibility"]["node_retention"] == pytest.approx(0.7)
    assert report["baseline_plausibility"]["edge_retention"] == pytest.approx(2 / 3)


def test_global_population_collapse_fails_baseline_retention(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    chain_frame(10).to_csv(baseline_path, index=False)
    chain_frame(4).to_csv(candidate_path, index=False)

    report = audit_mod.audit(candidate_path, baseline=baseline_path)
    assert report["verdict"] == "FAIL"
    assert not check(report, "B1 total_retention")["pass"]
    assert "nodes=16/40=0.400000" in check(report, "B1 total_retention")["detail"]


def test_one_dataset_collapse_fails_even_when_global_retention_passes(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    chain_frame(10).to_csv(baseline_path, index=False)
    counts = {dataset: 10 for dataset in audit_mod.EXPECTED_DATASETS}
    collapsed = audit_mod.EXPECTED_DATASETS[0]
    counts[collapsed] = 1
    chain_frame(counts).to_csv(candidate_path, index=False)

    report = audit_mod.audit(candidate_path, baseline=baseline_path)
    assert report["verdict"] == "FAIL"
    assert check(report, "B1 total_retention")["pass"]
    assert not check(report, "B2 dataset_coverage")["pass"]
    assert collapsed in check(report, "B2 dataset_coverage")["detail"]
    collapsed_report = report["baseline_plausibility"]["per_dataset"][collapsed]
    assert collapsed_report["node_retention"] == pytest.approx(0.1)
    assert collapsed_report["edge_retention"] == pytest.approx(0.0)


def test_invalid_baseline_fails_closed(tmp_path: Path) -> None:
    baseline = chain_frame(10)
    baseline["t"] = baseline["t"].astype(float)
    baseline.loc[baseline.index[baseline["row_type"].eq("node")][0], "t"] = 0.5
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    baseline.to_csv(baseline_path, index=False)
    chain_frame(10).to_csv(candidate_path, index=False)

    report = audit_mod.audit(candidate_path, baseline=baseline_path)
    assert report["verdict"] == "FAIL"
    assert not check(report, "B0 baseline")["pass"]
    assert "structural_verdict=FAIL" in check(report, "B0 baseline")["detail"]


@pytest.mark.parametrize("bound", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_retention_bound_is_rejected(tmp_path: Path, bound: float) -> None:
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    chain_frame(2).to_csv(baseline_path, index=False)
    chain_frame(2).to_csv(candidate_path, index=False)

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        audit_mod.audit(
            candidate_path, baseline=baseline_path, min_node_retention=bound,
        )


def _float_coord_frame() -> pd.DataFrame:
    """A structurally valid frame whose node z/y/x carry fractional values (an icom export)."""
    frame = valid_frame().copy()
    for col in ("z", "y", "x"):
        frame[col] = frame[col].astype(float)   # float dtype, else pandas truncates on assign
    node = frame["row_type"] == "node"
    frame.loc[node, "z"] = frame.loc[node, "z"] + 0.128
    frame.loc[node, "y"] = frame.loc[node, "y"] + 0.907
    frame.loc[node, "x"] = frame.loc[node, "x"] + 0.673
    return frame


def test_float_coordinates_fail_by_default_but_pass_with_allow_flag(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    _float_coord_frame().to_csv(path, index=False)
    # default: the integer contract flags the fractional coordinates
    assert audit_mod.audit(path)["verdict"] == "FAIL"
    # icom float mode: fractional z/y/x accepted, everything else still checked
    rep = audit_mod.audit(path, allow_float_coords=True)
    assert rep["verdict"] == "PASS", [c for c in rep["checks"] if not c["pass"]]
    assert check(rep, "A1 numeric")["pass"]
    assert check(rep, "A1 row_contract")["pass"]
    assert check(rep, "A3 volume")["pass"]


def test_allow_float_coords_still_rejects_fractional_ids_and_time(tmp_path: Path) -> None:
    # a fractional t or node_id is ALWAYS a defect, float mode or not
    for col in ("t", "node_id"):
        frame = _float_coord_frame()
        frame[col] = frame[col].astype(float)
        node = frame["row_type"] == "node"
        frame.loc[node, col] = frame.loc[node, col] + 0.5
        path = tmp_path / f"sub_{col}.csv"
        frame.to_csv(path, index=False)
        assert audit_mod.audit(path, allow_float_coords=True)["verdict"] == "FAIL", col


def test_allow_float_coords_still_enforces_volume_bounds(tmp_path: Path) -> None:
    frame = _float_coord_frame()
    node = frame["row_type"] == "node"
    frame.loc[node, "z"] = float(audit_mod.Z_MAX) + 5.5   # out of the acquisition volume
    path = tmp_path / "sub_oob.csv"
    frame.to_csv(path, index=False)
    rep = audit_mod.audit(path, allow_float_coords=True)
    assert rep["verdict"] == "FAIL"
    assert not check(rep, "A3 volume")["pass"]
