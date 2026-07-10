from __future__ import annotations

import csv
import json

from scripts.metric_solver.cli import main


def _write(path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_writes_reweighted_candidates(tmp_path, capsys) -> None:
    nodes = tmp_path / "nodes.csv"
    edges = tmp_path / "edges.csv"
    output = tmp_path / "selected.csv"
    _write(
        nodes,
        ["node_id", "t"],
        [{"node_id": 1, "t": 0}, {"node_id": 2, "t": 1}, {"node_id": 3, "t": 1}],
    )
    _write(
        edges,
        ["source_id", "target_id", "p_tp", "p_fp"],
        [
            {"source_id": 1, "target_id": 2, "p_tp": 1.0, "p_fp": 0.0},
            {"source_id": 1, "target_id": 3, "p_tp": 0.1, "p_fp": 0.9},
        ],
    )
    assert main(
        [
            "--candidates", str(edges),
            "--nodes", str(nodes),
            "--output", str(output),
            "--lambda", "0.5",
            "--backend", "scipy",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected"] == 1
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["selected"] for row in rows] == ["1", "0"]
    # Default calibration is a true identity, including exact endpoints.
    assert rows[0]["p_tp"] == "1.0"
    assert rows[0]["p_fp"] == "0.0"
