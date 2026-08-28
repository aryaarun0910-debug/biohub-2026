"""Replay the exact P28 post-ILP notebook chain on CPU.

The deployed notebook remains the source: this instrument extracts its constants,
classes, and functions with ``ast`` instead of maintaining a second implementation.
Only the prediction/submit top-level statements are excluded.  A positive source
fingerprint and required-symbol heartbeat make notebook drift fail closed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import blosc2
import numpy as np
import polars as pl
import torch
import tracksdata as td
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.win_bet.ilp_replay import build_graph_from_frame, solve  # noqa: E402
DEFAULT_NOTEBOOK = (
    ROOT / "notebooks/kaggle_p28_champion_control_f1/biohub-p28-champion-control-f1.ipynb"
)
REQUIRED = {
    "filter_output_graph",
    "motion_relink_edges",
    "close_single_frame_gaps",
    "recover_strict_gap2",
    "add_safe_divisions_postlink",
    "filter_short_track_components",
    "linefit_smooth_output_graph",
    "load_deepcenter_veto_detector",
}

P28_ENV = {
    "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1",
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
    "BIOHUB_ILP_APPEARANCE_WEIGHT": "0.0",
    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": "1.5",
    "BIOHUB_GAP_CLOSE_MAX_GAP": "2",
    "BIOHUB_GAP_CLOSE_UM": "5.8",
    "BIOHUB_GAP_DENSITY_ADAPTIVE": "1",
    "BIOHUB_GAP_DENSITY_REFERENCE_UM": "6.5",
    "BIOHUB_GAP_DENSITY_GAIN": "0.040",
    "BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM": "0.125",
    "BIOHUB_GAP_DENSITY_NEIGHBORS": "3",
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "1",
    "BIOHUB_OUTPUT_GAP2_RECOVERY": "0",
    "BIOHUB_SAFE_DIV_MAX_UM": "8.0",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "11.0",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "10.0",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "2.25",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "0",
    "BIOHUB_USE_DEEPCENTER_VETO": "1",
    "BIOHUB_REQUIRE_DEEPCENTER_VETO": "1",
    "BIOHUB_DEEPCENTER_EXPECTED_EPOCH": "2",
    "BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM": "8.5",
    "BIOHUB_DEEPCENTER_GAP_VETO": "1",
    "BIOHUB_DEEPCENTER_GAP_THRESHOLD": "0.25",
    "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1",
}


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _upper_assignment(node: ast.AST) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(
        isinstance(target, ast.Name)
        and target.id[:1].isalpha()
        and target.id.upper() == target.id
        for target in targets
    )


def load_p28_module(notebook: Path, train_dir: Path, checkpoint: Path) -> ModuleType:
    """Extract the executable post-processing surface from a built P28 notebook."""
    source = _code_source(notebook)
    tree = ast.parse(source, filename=str(notebook))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
            127 <= node.lineno <= 210 or node.lineno in {1410, 1411, 1412, 2516}
        ) and _upper_assignment(node):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and 1415 <= node.lineno < 3172:
            selected.append(node)
        elif isinstance(node, ast.If) and 1618 <= node.lineno <= 1624:
            selected.append(node)

    module = ModuleType("p28_notebook_chain")
    module.__dict__.update(
        {
            "__file__": str(notebook),
            "os": os,
            "json": json,
            "math": math,
            "Path": Path,
            "np": np,
            "blosc2": blosc2,
            "td": td,
            "torch": torch,
            "linear_sum_assignment": linear_sum_assignment,
            "cKDTree": cKDTree,
        }
    )
    old = {key: os.environ.get(key) for key in P28_ENV}
    old_checkpoint = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT")
    try:
        os.environ.update(P28_ENV)
        os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(checkpoint)
        executable = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(executable, str(notebook), "exec"), module.__dict__)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_checkpoint is None:
            os.environ.pop("BIOHUB_DEEPCENTER_CHECKPOINT", None)
        else:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = old_checkpoint

    missing = sorted(REQUIRED - module.__dict__.keys())
    if missing:
        raise RuntimeError(f"P28 extraction missing required symbols: {missing}")
    module.TEST_DIR = train_dir
    module.DEEPCENTER_CHECKPOINT_DEFAULT = str(checkpoint)
    module._source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    module._selected_nodes = len(selected)
    return module


def solved_graph_inputs(graph) -> tuple[dict[int, dict], list[dict]]:
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edges.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": float(row["edge_prob"]),
            }
        )
    return nodes, edges


def submission_frame(crop: str, nodes: dict[int, dict], edges: list[dict]) -> pl.DataFrame:
    node_rows = [
        {
            "dataset": crop, "row_type": "node", "node_id": node_id,
            "t": int(node["t"]), "z": float(node["z"]), "y": float(node["y"]),
            "x": float(node["x"]), "source_id": -1, "target_id": -1,
        }
        for node_id, node in sorted(nodes.items())
    ]
    edge_rows = [
        {
            "dataset": crop, "row_type": "edge", "node_id": -1, "t": -1,
            "z": -1.0, "y": -1.0, "x": -1.0,
            "source_id": int(edge["source_id"]), "target_id": int(edge["target_id"]),
        }
        for edge in edges
    ]
    return pl.DataFrame(node_rows + edge_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--preilp", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expect-stats", type=Path, required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--disappearance-weight", type=float, default=1.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    module = load_p28_module(args.notebook, args.train_dir, args.checkpoint)
    frame = pl.read_parquet(args.preilp).filter(pl.col("dataset") == args.crop)
    if frame.is_empty():
        raise SystemExit(f"crop absent from pre-ILP export: {args.crop}")
    candidates = build_graph_from_frame(
        frame.filter(pl.col("row_type") == "node").sort("node_id"),
        frame.filter(pl.col("row_type") == "edge"),
    )
    solved = solve(
        candidates,
        {"edge": -1.0, "appearance": 0.0, "disappearance": args.disappearance_weight, "division": 1.0},
    )
    nodes, edges = solved_graph_inputs(solved)
    raw_nodes, raw_edges = len(nodes), len(edges)

    # The loader restores process environment, so bind the explicit local checkpoint
    # while the notebook's fail-closed model loader runs.
    previous = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT")
    os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(args.checkpoint)
    try:
        detector = module.load_deepcenter_veto_detector()
    finally:
        if previous is None:
            os.environ.pop("BIOHUB_DEEPCENTER_CHECKPOINT", None)
        else:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = previous
    final_nodes, final_edges, stats = module.filter_output_graph(
        nodes, edges, dataset=args.crop, deepcenter_bundle=detector
    )
    final_frame = submission_frame(args.crop, final_nodes, final_edges)
    from scripts.win_bet import div_reach_steal as drs

    score = drs.score_crop(
        final_frame,
        args.train_dir / f"{args.crop}.geff",
        drs._ea_atlas(),
        drs._scorer(),
    )

    expected_rows = pl.read_csv(args.expect_stats).filter(pl.col("dataset") == args.crop)
    if expected_rows.height != 1:
        raise SystemExit(f"expected exactly one control stats row for {args.crop}")
    expected = expected_rows.row(0, named=True)
    compare_keys = [
        "raw_nodes", "raw_edges", "nodes", "edges", "motion_relink_edges",
        "gap_added_nodes", "safe_divisions_added", "deepcenter_gap_checked",
        "deepcenter_gap_accepted", "deepcenter_gap_rejected",
        "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted",
        "deepcenter_safe_div_rejected", "short_track_nodes_removed",
        "linefit_smoothed_nodes",
    ]
    got = {**stats, "raw_nodes": raw_nodes, "raw_edges": raw_edges, "nodes": len(final_nodes), "edges": len(final_edges)}
    comparison = {
        key: {"got": int(got[key]), "expected": int(expected[key]), "delta": int(got[key]) - int(expected[key])}
        for key in compare_keys
    }
    exact = all(item["delta"] == 0 for item in comparison.values())
    result = {
        "schema_version": 1,
        "heartbeat": "P28_FULL_CHAIN_REPLAY_COMPLETE",
        "crop": args.crop,
        "disappearance_weight": args.disappearance_weight,
        "notebook_sha256": module._source_sha256,
        "selected_ast_nodes": module._selected_nodes,
        "required_symbols": sorted(REQUIRED),
        "control_exact": exact,
        "score": {key: float(value) for key, value in score.items()},
        "comparison": comparison,
        "stats": {key: int(value) for key, value in stats.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"P28_FULL_CHAIN_REPLAY_COMPLETE crop={args.crop} control_exact={exact}")
    if args.disappearance_weight == 1.5 and not exact:
        raise SystemExit("P28 full-chain control parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
