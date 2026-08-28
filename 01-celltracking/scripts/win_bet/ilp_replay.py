r"""CPU replay of the deployed ILP linking stage, and its fail-closed parity gate.

WHY THIS EXISTS
---------------
``FACT-0353`` located the node deficit at the ILP solver: the detector already emits at or above
``estimated_number_of_nodes`` on both folds and the solver then discards 15-20% of its peaks.
``FACT-0356`` sizes it - the ILP is 79-83% of all node removal. ``LEVER-0035`` therefore has to
vary the solver's own terms, and doing that on GPU would cost one full run per setting.

It does not have to. The solver stage consumes only the pre-ILP candidate graph (nodes plus
edges with ``edge_prob``) and runs on CPU, so a single exported pre-ILP graph makes the whole
weight sweep a CPU replay - exactly what the DetPeak sidecars did for the detection threshold.

THIS IS THE SAME SOLVER, NOT A RE-IMPLEMENTATION
------------------------------------------------
``build_graph`` and the solver call are replicated from
``vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:114-146`` and ``:554-564``,
and the solve goes through the same ``tracksdata`` ``ILPSolver``. Re-deriving the linking would
have made every sweep a measurement of the re-derivation. Note the deployed stack has no Gurobi
licence and falls back to SCIP; the Kaggle kernels install ``pyscipopt`` too, so both sides solve
with SCIP.

THE PARITY GATE, AND WHY IT IS NOT SYMMETRIC
-------------------------------------------
``PKT-0025`` STEP 0: before any sweep is trusted, the replayer must reproduce the source run's own
recorded post-ILP graph. ``run_stats.csv`` records that as ``raw_nodes`` / ``raw_edges`` per crop -
the counts of the geff the predictor saved immediately after ``solver.solve``.

The gate treats those two differently, because measurement showed they are different kinds of
target (``FACT-0363``). NODE selection is deterministic: this replayer reproduces ``raw_nodes``
exactly on all 128 fold-1 crops against all three independently recorded runs, and node retention
is precisely what ``LEVER-0035`` varies. The EDGE solution is not - the ILP has multiple optima, and
three identical local solves of one crop returned 5,035 / 5,036 / 5,035 edges while holding nodes at
5,385 every time. An exact edge target therefore does not exist to be met.

So the node gate is hard and is never relaxed; the edge comparison takes a tolerance and an explicit
``--edge-gate advisory`` escape, and the residual degeneracy - at most 2 edges on about 5% of crops,
roughly 1 ppm of the fold's edges - is quoted as the noise floor for any offline sweep rather than
absorbed silently.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

# Deployed solver configuration, verified at
# notebooks/kaggle_p28_champion_control_f1/biohub-p28-champion-control-f1.ipynb (env block) and
# in the 2026-08-19 pre-ILP kernel log.
DEPLOYED_WEIGHTS = {
    "edge": -1.0,
    "appearance": 0.0,
    "disappearance": 1.5,
    "division": 1.0,
}


def build_graph_from_frame(nodes: pl.DataFrame, edges: pl.DataFrame):
    """Replica of predict_unet_transformer.build_graph, driven by a pre-ILP export.

    ``edge_dist`` is present in the deployed graph but is not read by the solver (the objective
    uses ``edge_prob`` only), so it is carried as 0.0 to keep the attribute schema identical.
    """
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in zip(nodes["t"], nodes["z"], nodes["y"], nodes["x"])
    ])
    remap = {int(old): int(new) for old, new in zip(nodes["node_id"], node_ids)}

    if edges.height:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": remap[int(s)],
                "target_id": remap[int(t)],
                "edge_prob": float(p),
                "edge_dist": 0.0,
            }
            for s, t, p in zip(edges["source_id"], edges["target_id"], edges["edge_prob"])
        ])
    return graph


def solve(graph, weights: dict[str, float], timeout: float | None = None):
    import tracksdata as td

    solver = td.solvers.ILPSolver(
        edge_weight=weights["edge"] * td.EdgeAttr("edge_prob"),
        appearance_weight=weights["appearance"],
        disappearance_weight=weights["disappearance"],
        division_weight=weights["division"],
        timeout=timeout,
    )
    return solver.solve(graph)


def solved_to_rows(graph, crop: str) -> pl.DataFrame:
    """Solved graph -> the submission node/edge row schema, for scoring."""
    nodes = graph.node_attrs().to_pandas()
    edges = graph.edge_attrs().to_pandas()
    node_rows = pl.DataFrame({
        "dataset": [crop] * len(nodes),
        "row_type": ["node"] * len(nodes),
        "node_id": nodes["node_id"].astype("int64").to_list(),
        "t": nodes["t"].astype("int64").to_list(),
        "z": nodes["z"].astype("float64").to_list(),
        "y": nodes["y"].astype("float64").to_list(),
        "x": nodes["x"].astype("float64").to_list(),
        "source_id": [-1] * len(nodes),
        "target_id": [-1] * len(nodes),
    })
    if len(edges):
        edge_rows = pl.DataFrame({
            "dataset": [crop] * len(edges),
            "row_type": ["edge"] * len(edges),
            "node_id": [-1] * len(edges),
            "t": [-1] * len(edges),
            "z": [-1.0] * len(edges),
            "y": [-1.0] * len(edges),
            "x": [-1.0] * len(edges),
            "source_id": edges["source_id"].astype("int64").to_list(),
            "target_id": edges["target_id"].astype("int64").to_list(),
        })
        return pl.concat([node_rows, edge_rows])
    return node_rows


def _one_crop(job: tuple) -> dict:
    preilp, crop, weights, timeout, out_dir = job
    frame = pl.read_parquet(preilp).filter(pl.col("dataset") == crop)
    nodes = frame.filter(pl.col("row_type") == "node").sort("node_id")
    edges = frame.filter(pl.col("row_type") == "edge")
    graph = build_graph_from_frame(nodes, edges)
    pre_nodes, pre_edges = graph.num_nodes(), graph.num_edges()
    solved = solve(graph, weights, timeout)
    result = {
        "crop": crop,
        "pre_nodes": int(pre_nodes),
        "pre_edges": int(pre_edges),
        "post_nodes": int(solved.num_nodes()),
        "post_edges": int(solved.num_edges()),
    }
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        solved_to_rows(solved, crop).write_parquet(Path(out_dir) / f"{crop}.parquet")
    return result


def replay(
    preilp: Path,
    crops: list[str],
    weights: dict[str, float],
    workers: int,
    timeout: float | None,
    out_dir: Path | None,
) -> list[dict]:
    jobs = [(str(preilp), c, weights, timeout, str(out_dir) if out_dir else None) for c in crops]
    rows: list[dict] = []
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_one_crop(job))
            print(f"  [{i}/{len(jobs)}] {rows[-1]['crop']} -> "
                  f"{rows[-1]['post_nodes']}n/{rows[-1]['post_edges']}e", flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_one_crop, jobs), 1):
            rows.append(row)
            print(f"  [{i}/{len(jobs)}] {row['crop']} -> "
                  f"{row['post_nodes']}n/{row['post_edges']}e", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--expect-stats", type=Path,
                    help="run_stats.csv of the SOURCE run; enables the fail-closed parity gate")
    ap.add_argument("--edge-weight", type=float, default=DEPLOYED_WEIGHTS["edge"])
    ap.add_argument("--appearance-weight", type=float, default=DEPLOYED_WEIGHTS["appearance"])
    ap.add_argument("--disappearance-weight", type=float, default=DEPLOYED_WEIGHTS["disappearance"])
    ap.add_argument("--division-weight", type=float, default=DEPLOYED_WEIGHTS["division"])
    ap.add_argument("--edge-tolerance", type=int, default=0,
                    help="per-crop |raw_edges| slack allowed by the parity gate")
    ap.add_argument("--edge-gate", choices=["hard", "advisory"], default="hard",
                    help="FACT-0363: the solver has multiple optima and its EDGE count is not "
                         "reproducible, so an exact edge target does not exist. `advisory` reports "
                         "edge parity without failing on it. The NODE gate is always hard and is "
                         "never relaxed - node selection IS deterministic and is what LEVER-0035 varies.")
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--save-graphs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = {
        "edge": args.edge_weight,
        "appearance": args.appearance_weight,
        "disappearance": args.disappearance_weight,
        "division": args.division_weight,
    }
    crops = sorted(
        pl.read_parquet(args.preilp, columns=["dataset"])["dataset"].unique().to_list()
    )
    if args.max_crops:
        crops = crops[: args.max_crops]

    rows = replay(args.preilp, crops, weights, args.workers, args.timeout, args.save_graphs)
    table = pl.DataFrame(rows)

    parity = None
    if args.expect_stats:
        expect = (
            pl.read_csv(args.expect_stats)
            .select(["dataset", "raw_nodes", "raw_edges"])
            .rename({"dataset": "crop"})
        )
        joined = table.join(expect, on="crop", how="inner").with_columns([
            (pl.col("post_nodes") - pl.col("raw_nodes")).alias("node_delta"),
            (pl.col("post_edges") - pl.col("raw_edges")).alias("edge_delta"),
        ])
        node_bad = joined.filter(pl.col("node_delta") != 0)
        edge_bad = joined.filter(pl.col("edge_delta").abs() > args.edge_tolerance)
        parity = {
            "crops_compared": joined.height,
            "node_parity_exact": node_bad.height == 0,
            "edge_parity_within_tolerance": edge_bad.height == 0,
            "edge_tolerance": args.edge_tolerance,
            "n_node_mismatch": node_bad.height,
            "n_edge_mismatch": edge_bad.height,
            "edge_delta_max_abs": int(joined["edge_delta"].abs().max()) if joined.height else 0,
            "node_mismatches": node_bad.select(["crop", "post_nodes", "raw_nodes"]).to_dicts()[:20],
            "edge_mismatches": edge_bad.select(["crop", "post_edges", "raw_edges"]).to_dicts()[:20],
        }

    result = {
        "schema_version": 1,
        "preilp": str(args.preilp),
        "weights": weights,
        "deployed_weights": DEPLOYED_WEIGHTS,
        "is_deployed_configuration": weights == DEPLOYED_WEIGHTS,
        "n_crops": len(rows),
        "totals": {
            "pre_nodes": int(table["pre_nodes"].sum()),
            "pre_edges": int(table["pre_edges"].sum()),
            "post_nodes": int(table["post_nodes"].sum()),
            "post_edges": int(table["post_edges"].sum()),
        },
        "parity": parity,
        "edge_gate": args.edge_gate,
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if parity is None:
        print(f"ILP_REPLAY crops={len(rows)} nodes={result['totals']['post_nodes']:,} "
              f"edges={result['totals']['post_edges']:,} (no parity gate requested)")
        return 0
    print(
        f"ILP_REPLAY_PARITY crops={parity['crops_compared']} "
        f"node_exact={parity['node_parity_exact']} "
        f"edge_within_tol={parity['edge_parity_within_tolerance']} "
        f"(tol={args.edge_tolerance}, max|d|={parity['edge_delta_max_abs']}, "
        f"n_node_bad={parity['n_node_mismatch']}, n_edge_bad={parity['n_edge_mismatch']})"
    )
    if not parity["node_parity_exact"]:
        raise SystemExit("PARITY FAILED on nodes - no sweep from this replayer is trustworthy")
    if not parity["edge_parity_within_tolerance"] and args.edge_gate == "hard":
        raise SystemExit("PARITY FAILED on edges beyond the declared tolerance")
    if not parity["edge_parity_within_tolerance"]:
        print("  edge parity outside tolerance, reported as ADVISORY (FACT-0363 degeneracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
