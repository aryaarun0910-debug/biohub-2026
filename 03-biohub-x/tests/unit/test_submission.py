"""The submission writer and validator hold to the official schema and to each other.

The columns and sentinels come from the competition's own scripts at the
pinned commit; a file that drifts from them is refused by Kaggle after the
fact, which is the expensive place to learn it. These tests keep the writer on
the schema, make the validator name every refusal it exists for, and prove the
round trip: what was written is what is read back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import DatasetIdentity, EdgeKind, LineageEdge, LineageGraph, LineageNode
from biohubx.submission import (
    COLUMNS,
    SubmissionError,
    forecast,
    round_trip,
    rows_from_graph,
    validate_csv,
    write_csv,
)


def graph(name: str, *, division: bool = False) -> LineageGraph:
    dataset = DatasetIdentity(value=name)

    def node(node_id: int, frame: int, z: float, y: float, x: float) -> LineageNode:
        return LineageNode(
            dataset=dataset, node_id=node_id, frame=frame, voxel=VoxelCoordinateZYX(z=z, y=y, x=x)
        )

    def edge(source: int, target: int, kind: EdgeKind = EdgeKind.CONTINUATION) -> LineageEdge:
        return LineageEdge(
            source_dataset=dataset, source=source, target_dataset=dataset, target=target, kind=kind
        )

    nodes = (node(0, 0, 3.4, 10.6, 20.2), node(1, 1, 3.6, 11.0, 21.0), node(2, 1, 5.0, 30.0, 40.0))
    edges = (edge(0, 1), edge(0, 2, EdgeKind.DIVISION)) if division else (edge(0, 1),)
    if division:
        edges = (edge(0, 1, EdgeKind.DIVISION), edge(0, 2, EdgeKind.DIVISION))
    return LineageGraph(dataset=dataset, nodes=nodes, edges=edges)


def test_rows_follow_the_official_schema_with_sentinels_and_rounded_coordinates() -> None:
    rows = rows_from_graph(graph("a"), "a")
    assert [r["row_type"] for r in rows] == ["node", "node", "node", "edge"]
    first = rows[0]
    assert (first["node_id"], first["t"], first["z"], first["y"], first["x"]) == (0, 0, 3, 11, 20)
    assert (first["source_id"], first["target_id"]) == (-1, -1)
    edge = rows[-1]
    assert (edge["node_id"], edge["t"], edge["z"], edge["y"], edge["x"]) == (-1, -1, -1, -1, -1)
    assert (edge["source_id"], edge["target_id"]) == (0, 1)


def test_written_file_has_the_exact_header_contiguous_ids_and_round_trips(tmp_path: Path) -> None:
    graphs = {"b": graph("b", division=True), "a": graph("a")}
    report = write_csv(graphs, tmp_path / "submission.csv")
    text = (tmp_path / "submission.csv").read_text(encoding="utf-8").splitlines()
    assert text[0] == ",".join(COLUMNS)
    assert [line.split(",")[0] for line in text[1:]] == [str(i) for i in range(report.rows)]
    assert text[1].startswith("0,a,node,0,0,3,11,20,-1,-1")
    assert report.per_dataset == {"a": {"nodes": 3, "edges": 1}, "b": {"nodes": 3, "edges": 2}}
    validation = validate_csv(tmp_path / "submission.csv", expected_datasets=["a", "b"])
    assert validation.ok, validation.refusals
    assert round_trip(graphs, tmp_path / "submission.csv")["identical"]


def test_validator_names_each_refusal(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "\n".join(
            [
                ",".join(COLUMNS),
                "0,d,node,0,0,1,1,1,-1,-1",
                "1,d,node,0,1,1,1,1,-1,-1",  # repeated node id
                "3,d,node,2,3,1,1,1,-1,-1",  # id breaks the index; node at frame 3
                "4,d,edge,-1,-1,-1,-1,-1,0,2",  # spans frames 0 -> 3
                "5,d,edge,-1,-1,-1,-1,-1,0,9",  # dangling target
                "6,d,edge,-1,-1,-1,-1,-1,7,7",  # self loop
                "7,d,cell,1,1,1,1,1,-1,-1",  # unknown row type
                "8,d,node,5,1,1,1,x,-1,-1",  # non-integer coordinate
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_csv(path, expected_datasets=["d", "e"])
    joined = "\n".join(report.refusals)
    for expected in (
        "repeats within d",
        "breaks the contiguous index",
        "spans frames 1->3",  # the repeated id left node 0 at frame 1
        "references a node that is not in the file",
        "self loop",
        "unknown row_type",
        "is not an integer",
        "datasets expected and absent: ['e']",
    ):
        assert expected in joined, expected
    assert not report.ok


def test_validator_refuses_a_wrong_header_and_degree_violations(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("dataset,row_type\n", encoding="utf-8")
    assert validate_csv(wrong).refusals[0].startswith("header is")
    degrees = tmp_path / "degrees.csv"
    degrees.write_text(
        "\n".join(
            [
                ",".join(COLUMNS),
                "0,d,node,0,0,1,1,1,-1,-1",
                "1,d,node,1,1,1,1,1,-1,-1",
                "2,d,node,2,1,2,2,2,-1,-1",
                "3,d,node,3,1,3,3,3,-1,-1",
                "4,d,node,4,0,4,4,4,-1,-1",
                "5,d,edge,-1,-1,-1,-1,-1,0,1",
                "6,d,edge,-1,-1,-1,-1,-1,0,2",
                "7,d,edge,-1,-1,-1,-1,-1,0,3",
                "8,d,edge,-1,-1,-1,-1,-1,4,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    joined = "\n".join(validate_csv(degrees).refusals)
    assert "out-degree 3" in joined and "in-degree 2" in joined


def test_empty_submission_is_refused_and_forecast_scales_with_measured_bytes(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError):
        write_csv({}, tmp_path / "empty.csv")
    report = write_csv({"a": graph("a")}, tmp_path / "one.csv")
    projection = forecast(report, nodes_per_movie=1000, edges_per_node=0.9, movies=10)
    assert projection["projected_rows"] == 19000
    assert projection["projected_bytes"] == int(19000 * report.bytes / report.rows)
