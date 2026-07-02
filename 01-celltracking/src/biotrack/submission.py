"""Submission plumbing: tracksdata graphs <-> Kaggle ``sample_submission.csv``.

The competition CSV schema (one row per node and per edge, concatenated over
all datasets):

    id,dataset,row_type,node_id,t,z,y,x,source_id,target_id

- node rows: row_type="node", node_id>=1 (unique within a dataset), t,z,y,x set,
  source_id=target_id=-1.
- edge rows: row_type="edge", node_id=-1, t,z,y,x=-1, source_id/target_id refer
  to node_id values of the same dataset.

These converters let us turn per-dataset predicted graphs into a submission and,
crucially, reconstruct graphs from a submission so we can score it with the exact
local metric (``biotrack.metric``).
"""

from pathlib import Path

import polars as pl
import tracksdata as td

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]


def graph_to_rows(dataset: str, graph: td.graph.BaseGraph) -> list[dict]:
    """Emit submission rows for one dataset's graph.

    Node ids are renumbered 1..N (submission-local); edges are remapped onto them.
    """
    na = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    internal = na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    sub_id = {iid: i + 1 for i, iid in enumerate(internal)}  # 1-based, stable order

    rows: list[dict] = []
    for iid, t, z, y, x in zip(
        internal, na["t"].to_list(), na["z"].to_list(), na["y"].to_list(), na["x"].to_list()
    ):
        rows.append({
            "dataset": dataset, "row_type": "node", "node_id": sub_id[iid],
            "t": int(t), "z": float(z), "y": float(y), "x": float(x),
            "source_id": -1, "target_id": -1,
        })

    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=[
            td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        ])
        for s, tt in zip(
            ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
            ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(),
        ):
            rows.append({
                "dataset": dataset, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": sub_id[s], "target_id": sub_id[tt],
            })
    return rows


def graphs_to_submission(graphs: dict[str, td.graph.BaseGraph]) -> pl.DataFrame:
    """Concatenate per-dataset graphs into a submission DataFrame with global ``id``."""
    rows: list[dict] = []
    for dataset in sorted(graphs):
        rows.extend(graph_to_rows(dataset, graphs[dataset]))
    df = pl.DataFrame(rows)
    df = df.with_row_index("id")
    return df.select(SUBMISSION_COLUMNS)


def write_submission(graphs: dict[str, td.graph.BaseGraph], path: str | Path) -> Path:
    path = Path(path)
    graphs_to_submission(graphs).write_csv(path)
    return path


def submission_to_graphs(df: pl.DataFrame) -> dict[str, td.graph.BaseGraph]:
    """Reconstruct one :class:`InMemoryGraph` per dataset from a submission frame.

    Inverse of :func:`graphs_to_submission` (coordinates and connectivity preserved;
    submission-local node ids are remapped onto fresh internal ids).
    """
    graphs: dict[str, td.graph.BaseGraph] = {}
    for dataset, sub in df.group_by("dataset"):
        ds = dataset[0] if isinstance(dataset, tuple) else dataset
        nodes = sub.filter(pl.col("row_type") == "node").sort("node_id")
        g = td.graph.InMemoryGraph()
        for key in ["z", "y", "x"]:
            g.add_node_attr_key(key, pl.Float64, -999999.0)
        internal = g.bulk_add_nodes([
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for t, z, y, x in zip(
                nodes["t"].to_list(), nodes["z"].to_list(),
                nodes["y"].to_list(), nodes["x"].to_list(),
            )
        ])
        sub_to_internal = {int(sid): internal[i] for i, sid in enumerate(nodes["node_id"].to_list())}

        edges = sub.filter(pl.col("row_type") == "edge")
        if edges.height > 0:
            g.bulk_add_edges([
                {"source_id": sub_to_internal[int(s)], "target_id": sub_to_internal[int(t)]}
                for s, t in zip(edges["source_id"].to_list(), edges["target_id"].to_list())
            ])
        graphs[ds] = g
    return graphs


def read_submission(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path)
