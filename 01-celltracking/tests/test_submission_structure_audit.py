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
