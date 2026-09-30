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


# The exact post-processing surface, selected by NAME so the extraction is fold-agnostic.
# Derived from the fold-1 control notebook and asserted present in whichever notebook is
# loaded, so a rebuild that renames or drops a stage fails closed instead of silently
# replaying a different chain.
CONFIG_NAMES = {
    "OUTPUT_EDGE_MAX_UM",
    "OUTPUT_ENFORCE_NEXT_FRAME",
    "OUTPUT_SINGLE_PARENT_REPAIR",
    "OUTPUT_SINGLE_CHILD_REPAIR",
    "OUTPUT_PRUNE_ISOLATED",
    "OUTPUT_MOTION_RELINK",
    "MOTION_RELINK_TIGHT_UM",
    "MOTION_RELINK_RELAXED_UM",
    "MOTION_RELINK_VELOCITY_WEIGHT",
    "MOTION_RELINK_LEARNED_BONUS",
    "MOTION_RELINK_MAX_FRAME_NODES",
    "OUTPUT_DIVISION_GEOMETRY_FILTER",
    "DIV_PARENT_MAX_UM",
    "DIV_SISTER_MAX_UM",
    "DIV_DROP_TO_SINGLE_IF_BAD",
    "OUTPUT_GAP_CLOSE",
    "GAP_CLOSE_MAX_GAP",
    "GAP_CLOSE_UM",
    "GAP_DENSITY_ADAPTIVE",
    "GAP_DENSITY_REFERENCE_UM",
    "GAP_DENSITY_GAIN",
    "GAP_DENSITY_MAX_STEP_DELTA_UM",
    "GAP_DENSITY_NEIGHBORS",
    "GAP_CLOSE_REUSE_EXISTING",
    "GAP_CLOSE_REUSE_UM",
    "GAP_CLOSE_MAX_ADDED_FRAC",
    "GAP_CLOSE_MAX_ADDED_ABS",
    "GAP_REFINE_SYNTHETIC",
    "GAP_REFINE_WIN_Z",
    "GAP_REFINE_WIN_YX",
    "GAP_REFINE_MAX_SHIFT_UM",
    "OUTPUT_FILTER_SHORT_TRACKS",
    "OUTPUT_MIN_TRACK_LEN",
    "OUTPUT_KEEP_DIVISION_COMPONENTS",
    "ADAPTIVE_SHORT_TRACK_RESCUE",
    "SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC",
    "SHORT_TRACK_RESCUE_MIN_LEN",
    "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB",
    "SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM",
    "SHORT_TRACK_RESCUE_MAX_NODES_FRAC",
    "SHORT_TRACK_RESCUE_MAX_NODES_ABS",
    "OUTPUT_LINEFIT_SMOOTH",
    "OUTPUT_LINEFIT_WEIGHT",
    "OUTPUT_LINEFIT_WINDOW",
    "OUTPUT_GAP2_RECOVERY",
    "GAP2_MAX_TOTAL_UM",
    "GAP2_MAX_STEP_UM",
    "GAP2_MAX_LINKS_FRAC",
    "GAP2_MAX_LINKS_ABS",
    "GAP2_REQUIRE_CONTEXT",
    "GAP2_FRAME_FRAC_CAP",
    "OUTPUT_SAFE_DIVISIONS",
    "SAFE_DIV_MAX_UM",
    "SAFE_DIV_SISTER_MAX_UM",
    "SAFE_DIV_EXISTING_CHILD_MAX_UM",
    "SAFE_DIV_FRAME_FRAC_CAP",
    "SAFE_DIV_GLOBAL_FRAC_CAP",
    "USE_DEEPCENTER_VETO",
    "REQUIRE_DEEPCENTER_VETO",
    "DEEPCENTER_MANIFEST_DEFAULT",
    "DEEPCENTER_CHECKPOINT_DEFAULT",
    "DEEPCENTER_RELATIVE",
    "DEEPCENTER_GAP_VETO",
    "DEEPCENTER_SAFE_DIV_VETO",
    "DEEPCENTER_GAP_THRESHOLD",
    "DEEPCENTER_EXPECTED_EPOCH",
    "DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM",
    "DEEPCENTER_SAFE_DIV_THRESHOLD",
    "DEEPCENTER_SCORE_WIN_Z",
    "DEEPCENTER_SCORE_WIN_YX",
    "DEEPCENTER_SCORE_CACHE_MAX_FRAMES",
    "SUBMISSION_COLUMNS",
    "CSV_COLUMNS",
    "VOXEL_SCALE_UM",
    "COUPLED_DIV_DIVERGE_UM",
}

CHAIN_FUNCTIONS = {
    "graph_from_geff",
    "edge_distance_um",
    "point_distance_um",
    "node_point",
    "edge_sort_key",
    "_next_node_id",
    "read_test_frame",
    "refine_synthetic_midpoint",
    "_dc_pool_frame_xy",
    "_dc_normalize_dynamic_range",
    "_dc_manifest_weight_paths",
    "_dc_checkpoint_candidates",
    "load_deepcenter_veto_detector",
    "_dc_cache_trim",
    "deepcenter_heatmap_for_frame",
    "deepcenter_score_point",
    "deepcenter_accept_repair_point",
    "_position_um",
    "motion_relink_edges",
    "close_single_frame_gaps",
    "_single_successor_map",
    "_single_predecessor_map",
    "recover_strict_gap2",
    "add_safe_divisions_postlink",
    "_coupled_inc",
    "_coupled_assert_degrees",
    "_coupled_point_um",
    "add_safe_divisions_postlink",
    "filter_short_track_components",
    "linefit_smooth_output_graph",
    "assert_degree_invariants",
    "filter_output_graph",
}


def _is_torch_model_block(node: ast.If) -> bool:
    """The `if torch is not None:` block defining the DeepCenter model classes."""
    test = node.test
    if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "torch" and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)):
        return False
    return any(isinstance(b, ast.ClassDef) and b.name == "_DCConvBlock3d" for b in node.body)

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
    # Selection is by NAME and STRUCTURE, never by line number. The fold-0 and fold-1 control
    # notebooks are structurally identical but offset by two lines, and the original
    # line-range selection silently mis-selected on fold 0 - it pulled a config assignment
    # whose dependency fell outside the window and died on `EXPERIMENT_TAG`. Names are stable
    # across folds and across rebuilds; line numbers are not.
    selected: list[ast.stmt] = []
    seen_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _upper_assignment(node):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if names & CONFIG_NAMES:
                selected.append(node)
                seen_names |= names & CONFIG_NAMES
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in CHAIN_FUNCTIONS:
            selected.append(node)
            seen_names.add(node.name)
        elif isinstance(node, ast.If) and _is_torch_model_block(node):
            selected.append(node)
            seen_names.add("_torch_model_block")

    missing_selection = (CONFIG_NAMES | CHAIN_FUNCTIONS | {"_torch_model_block"}) - seen_names
    if missing_selection:
        raise RuntimeError(
            f"P28 extraction did not find {len(missing_selection)} expected symbols in "
            f"{notebook.name}: {sorted(missing_selection)[:12]}"
        )

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



def expanded_edge_frame(crop: str, sidecar: Path, floor: float, rank: int) -> pl.DataFrame:
    """The LEVER-0037 candidate set: sidecar pairs above `floor`, at most `rank` per target.

    Returns the same schema `build_graph_from_frame` consumes from the pre-ILP export, so the
    solver sees a widened candidate graph and nothing else about the replay changes. Node ids
    are the sidecar's own, which FACT-0373 check D verified are the pre-ILP graph's ids.
    """
    import numpy as np

    with np.load(sidecar, allow_pickle=False) as z:
        src = z["source_id"].astype(np.int64)
        tgt = z["target_id"].astype(np.int64)
        prob = z["edge_prob"].astype(np.float64)
    keep = prob > floor
    src, tgt, prob = src[keep], tgt[keep], prob[keep]
    # Rank within target, best first - the same rule the export used, re-applied at `rank`.
    order = np.lexsort((-prob, tgt))
    ranks = np.empty(len(src), dtype=np.int64)
    current, seen = None, 0
    for pos in order:
        if tgt[pos] != current:
            current, seen = tgt[pos], 0
        seen += 1
        ranks[pos] = seen
    keep = ranks <= rank
    return pl.DataFrame({
        "dataset": [crop] * int(keep.sum()),
        "row_type": ["edge"] * int(keep.sum()),
        "node_id": [-1] * int(keep.sum()),
        "t": [-1] * int(keep.sum()),
        "z": [-1.0] * int(keep.sum()),
        "y": [-1.0] * int(keep.sum()),
        "x": [-1.0] * int(keep.sum()),
        "source_id": src[keep].tolist(),
        "target_id": tgt[keep].tolist(),
        "edge_prob": prob[keep].tolist(),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--preilp", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expect-stats", type=Path, required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--disappearance-weight", type=float, default=1.5)
    # LEVER-0037 treatment: widen the candidate set the solver is offered. Off unless a
    # sidecar is given, so the LEVER-0036 arm is unaffected and the two never mix.
    parser.add_argument("--ecb-sidecar", type=Path,
                        help="P30 candidate sidecar for this crop; enables candidate expansion")
    parser.add_argument("--candidate-floor", type=float, default=0.1)
    parser.add_argument("--candidate-rank", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    # PKT-0027 rule (5): a small NET conversion cannot distinguish "few edges gained" from
    # "many gained and nearly as many displaced". Persisting the final graph makes that
    # attribution possible; it is off by default so ordinary replays stay cheap.
    parser.add_argument("--save-graph", type=Path,
                        help="write the final per-crop graph as parquet for identity-level analysis")
    args = parser.parse_args()

    module = load_p28_module(args.notebook, args.train_dir, args.checkpoint)
    frame = pl.read_parquet(args.preilp).filter(pl.col("dataset") == args.crop)
    if frame.is_empty():
        raise SystemExit(f"crop absent from pre-ILP export: {args.crop}")
    edge_frame = frame.filter(pl.col("row_type") == "edge")
    deployed_edges = edge_frame.height
    expansion = None
    if args.ecb_sidecar is not None:
        edge_frame = expanded_edge_frame(
            args.crop, args.ecb_sidecar, args.candidate_floor, args.candidate_rank,
        )
        deployed_pairs = set(zip(
            frame.filter(pl.col("row_type") == "edge")["source_id"].to_list(),
            frame.filter(pl.col("row_type") == "edge")["target_id"].to_list(),
        ))
        offered = set(zip(edge_frame["source_id"].to_list(), edge_frame["target_id"].to_list()))
        expansion = {
            "floor": args.candidate_floor,
            "rank": args.candidate_rank,
            "deployed_edges": deployed_edges,
            "offered_edges": edge_frame.height,
            "multiple": edge_frame.height / max(deployed_edges, 1),
            # A widened set must CONTAIN the deployed one, or this is not a superset
            # experiment and any comparison against the control is confounded.
            "deployed_edges_dropped": len(deployed_pairs - offered),
        }
        if expansion["deployed_edges_dropped"]:
            raise SystemExit(
                f"candidate expansion DROPPED {expansion['deployed_edges_dropped']} deployed "
                "edges - the widened set must be a superset of what the pipeline used"
            )
    candidates = build_graph_from_frame(
        frame.filter(pl.col("row_type") == "node").sort("node_id"),
        edge_frame,
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
    if args.save_graph is not None:
        args.save_graph.mkdir(parents=True, exist_ok=True)
        final_frame.write_parquet(args.save_graph / f"{args.crop}.parquet")
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
        "candidate_expansion": expansion,
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
