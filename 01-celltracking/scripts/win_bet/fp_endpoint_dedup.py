r"""GT-free conservative duplicate-endpoint pruning for LEVER-0026 (PKT-0022).

The deployed selector in this file reads only submission node coordinates and graph topology.
Ground truth is used only by ``run`` after selection, to audit the frozen rule through the exact
``ea_atlas.run_crop`` scorer round-trip.  The initial safe subset is deliberately narrow: an
interior one-edge track endpoint may be removed only when its nearest same-frame detection is a
mutual nearest neighbour on a well-supported, non-division track.

Example (parameters are loaded from a committed/frozen JSON, not supplied per crop)::

  .\.venv\Scripts\python.exe scripts\win_bet\fp_endpoint_dedup.py run ^
      --csv C:/temp/p20_relink_sweep_f0/sweep_pen_off.csv.gz ^
      --atlas-dir C:/temp/p20_relink_sweep_f0/atlas --tag f0 ^
      --params C:/temp/fp_endpoint_dedup/safe_v1.json ^
      --out-dir C:/temp/fp_endpoint_dedup/f0

``run`` records every selected node and incident edge, asserts telemetry against the emitted CSV,
and reports event-level lost/gained scored GT edges and division TPs.  A silent no-op is reported
as a failed gate, not as a successful run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
DEFAULT_PARAMS = {
    "radius_um": 5.0,
    "min_keeper_back_steps": 2,
    "min_keeper_forward_steps": 2,
    "require_mutual_nn": True,
    "protect_crop_boundaries": True,
}


def load_params(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    params = {**DEFAULT_PARAMS, **raw}
    if not 0.0 < float(params["radius_um"]) <= 7.0:
        raise ValueError("radius_um must be in (0, 7]")
    for key in ("min_keeper_back_steps", "min_keeper_forward_steps"):
        if int(params[key]) < 1:
            raise ValueError(f"{key} must be >= 1")
    return params


def _node_edge_tables(sub: pl.DataFrame) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    nodes = (sub.filter(pl.col("row_type") == "node")
             .select(["node_id", "t", "z", "y", "x"])
             .sort("node_id").to_pandas())
    ef = sub.filter(pl.col("row_type") == "edge")
    edges = set(zip(ef["source_id"].to_list(), ef["target_id"].to_list()))
    if len(edges) != ef.height:
        raise RuntimeError("duplicate edge rows in input export")
    return nodes, edges


def _walk_steps(start: int, adjacency: dict[int, list[int]], cap: int) -> int:
    """Number of unambiguous steps from ``start``, capped for bounded runtime."""
    cur = int(start)
    seen = {cur}
    steps = 0
    while steps < cap and len(adjacency.get(cur, ())) == 1:
        nxt = int(adjacency[cur][0])
        if nxt in seen:
            raise RuntimeError("cycle in temporal graph")
        seen.add(nxt)
        cur = nxt
        steps += 1
    return steps


def select_duplicate_endpoints(
    nodes: pd.DataFrame,
    edges: set[tuple[int, int]],
    params: dict | None = None,
) -> pd.DataFrame:
    """Return GT-free delete decisions, one row per node.

    This function intentionally has no atlas/GT argument.  Its deterministic tie-breaking is
    ``distance, candidate_id, keeper_id`` so a frozen rule replays identically across folds.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    required = {"node_id", "t", "z", "y", "x"}
    if set(nodes.columns) != required:
        missing = required - set(nodes.columns)
        if missing:
            raise ValueError(f"missing node columns: {sorted(missing)}")
    if nodes.node_id.duplicated().any():
        raise RuntimeError("duplicate node ids")

    ids = set(int(x) for x in nodes.node_id)
    parents: dict[int, list[int]] = defaultdict(list)
    children: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        source, target = int(source), int(target)
        if source not in ids or target not in ids:
            raise RuntimeError(f"edge references missing node: {(source, target)}")
        parents[target].append(source)
        children[source].append(target)
    if any(len(v) > 1 for v in parents.values()):
        raise RuntimeError("input graph violates one-parent invariant")
    if any(len(v) > 2 for v in children.values()):
        raise RuntimeError("input graph violates two-child invariant")

    indeg = {n: len(parents.get(n, ())) for n in ids}
    outdeg = {n: len(children.get(n, ())) for n in ids}
    t_of = dict(zip(nodes.node_id.astype(int), nodes.t.astype(int)))
    min_t, max_t = int(nodes.t.min()), int(nodes.t.max())
    xyz = nodes[["z", "y", "x"]].to_numpy(np.float64) * SCALE_UM
    nid = nodes.node_id.to_numpy(np.int64)

    # Nearest same-frame node for every node.  k=2 includes self; stable id sorting resolves
    # exact-coordinate ties deterministically after the distance query.
    nearest: dict[int, tuple[int, float]] = {}
    for _t, frame in nodes.assign(_row=np.arange(len(nodes))).groupby("t", sort=True):
        rows = frame._row.to_numpy(np.int64)
        if len(rows) < 2:
            continue
        pts = xyz[rows]
        tree = cKDTree(pts)
        # Query a few neighbours so exact-coordinate ties can be resolved by node id.
        k = min(8, len(rows))
        dd, ii = tree.query(pts, k=k)
        if k == 1:
            dd, ii = dd[:, None], ii[:, None]
        frame_ids = nid[rows]
        for q, node in enumerate(frame_ids):
            choices = [(float(d), int(frame_ids[j])) for d, j in zip(dd[q], ii[q]) if int(frame_ids[j]) != int(node)]
            if choices:
                dist, other = min(choices, key=lambda x: (x[0], x[1]))
                nearest[int(node)] = (other, dist)

    decisions: list[dict] = []
    radius = float(p["radius_um"])
    back_need = int(p["min_keeper_back_steps"])
    fwd_need = int(p["min_keeper_forward_steps"])
    cap = max(back_need, fwd_need)
    for candidate in sorted(ids):
        ci, co = indeg[candidate], outdeg[candidate]
        if ci + co != 1:
            continue
        if p["protect_crop_boundaries"]:
            if ci == 0 and t_of[candidate] == min_t:
                continue
            if co == 0 and t_of[candidate] == max_t:
                continue
        # Never remove a daughter endpoint from a fork.  It may be a real newborn lineage even
        # when another detection is spatially close.
        if ci == 1 and outdeg[int(parents[candidate][0])] == 2:
            continue
        near = nearest.get(candidate)
        if near is None:
            continue
        keeper, distance = near
        if distance > radius:
            continue
        ki, ko = indeg[keeper], outdeg[keeper]
        # A strict non-fork continuation is materially better-connected than a leaf.  Forks are
        # excluded because division daughters can legitimately be close.
        if ki != 1 or ko != 1:
            continue
        if p["require_mutual_nn"] and nearest.get(keeper, (None,))[0] != candidate:
            continue
        back = _walk_steps(keeper, parents, cap)
        forward = _walk_steps(keeper, children, cap)
        if back < back_need or forward < fwd_need:
            continue
        incident = sorted((int(s), int(t)) for s, t in edges if s == candidate or t == candidate)
        if len(incident) != 1:
            raise AssertionError("leaf candidate must have exactly one incident edge")
        decisions.append({
            "candidate_id": candidate,
            "keeper_id": keeper,
            "t": t_of[candidate],
            "distance_um": distance,
            "candidate_indegree": ci,
            "candidate_outdegree": co,
            "keeper_indegree": ki,
            "keeper_outdegree": ko,
            "keeper_back_steps": back,
            "keeper_forward_steps": forward,
            "removed_edges": json.dumps(incident),
            "reason": "interior_leaf_mutual_nn_to_supported_continuation",
        })
    out = pd.DataFrame(decisions)
    if len(out) and (out.candidate_id.duplicated().any() or out.keeper_id.duplicated().any()):
        raise AssertionError("selector must choose at most one candidate per node/keeper")
    return out


def apply_decisions(
    sub: pl.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pl.DataFrame, set[tuple[int, int]], set[tuple[int, int]]]:
    """Delete selected nodes and normalize edges before telemetry and scoring."""
    nodes, input_edges = _node_edge_tables(sub)
    removed_nodes = set(int(x) for x in decisions.get("candidate_id", pd.Series(dtype=int)))
    kept_nodes = set(int(x) for x in nodes.node_id) - removed_nodes
    output_edges = {(int(s), int(t)) for s, t in input_edges if int(s) in kept_nodes and int(t) in kept_nodes}
    removed_edges = input_edges - output_edges

    original_nodes = sub.filter(pl.col("row_type") == "node")
    kept = original_nodes.filter(~pl.col("node_id").is_in(sorted(removed_nodes)))
    dataset = str(original_nodes["dataset"][0])
    src, tgt = zip(*sorted(output_edges)) if output_edges else ((), ())
    edge_rows = pl.DataFrame({
        "id": pl.Series([0] * len(src), dtype=pl.Int64),
        "dataset": [dataset] * len(src),
        "row_type": ["edge"] * len(src),
        "node_id": pl.Series([-1] * len(src), dtype=pl.Int64),
        "t": pl.Series([-1] * len(src), dtype=pl.Int64),
        "z": pl.Series([-1] * len(src), dtype=kept["z"].dtype),
        "y": pl.Series([-1] * len(src), dtype=kept["y"].dtype),
        "x": pl.Series([-1] * len(src), dtype=kept["x"].dtype),
        "source_id": pl.Series(list(src), dtype=pl.Int64),
        "target_id": pl.Series(list(tgt), dtype=pl.Int64),
    })
    out = pl.concat([kept.select(edge_rows.columns), edge_rows], how="vertical")
    out = out.with_columns(pl.Series("id", np.arange(out.height, dtype=np.int64)))
    emitted_edges = out.filter(pl.col("row_type") == "edge").height
    input_edge_rows = sub.filter(pl.col("row_type") == "edge").height
    assert len(removed_edges) == input_edge_rows - emitted_edges
    assert all(s in kept_nodes and t in kept_nodes for s, t in output_edges)
    return out, removed_edges, output_edges


def _ea_atlas():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ea_atlas_dedup", ROOT / "scripts" / "win_bet" / "ea_atlas.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _division_tp_ids(sub: pl.DataFrame, gt_geff: Path, ea) -> set[int]:
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from tracking_cellmot import division_metrics as dm
    graph, _ = ea.build_graph(sub)
    gt = load_graph(gt_geff)
    scores = dm.score_divisions(graph, gt, DEFAULT_SCALE, MAX_DISTANCE).scores
    return {int(k) for k, value in scores.items() if int(value) == 1}


def _arm_summary(rows: list[dict], summarise) -> dict:
    out = dict(summarise(rows))
    for key in ("edge_tp", "edge_fp", "edge_fn"):
        out[key] = int(sum(int(row[key]) for row in rows))
    return out


def _paired_bootstrap(rows_c: list[dict], rows_r: list[dict], summarise, n: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    delta = []
    for _ in range(n):
        take = rng.integers(0, len(rows_c), len(rows_c))
        c = summarise([rows_c[i] for i in take])["score"]
        r = summarise([rows_r[i] for i in take])["score"]
        delta.append(float(r - c))
    a = np.asarray(delta)
    return {"mean": float(a.mean()), "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], "resamples": n}


def cmd_run(args) -> int:
    from biotrack.submission import read_submission
    from tracking_cellmot.metrics import summarise

    params = load_params(Path(args.params))
    params_bytes = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    params_sha256 = hashlib.sha256(params_bytes).hexdigest()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ea = _ea_atlas()
    df = read_submission(ea.open_csv(Path(args.csv)))
    names = sorted(df["dataset"].unique().to_list())
    if args.stride:
        names = names[::args.stride]
    if args.max_crops:
        names = names[:args.max_crops]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_nodes = pd.read_parquet(Path(args.atlas_dir) / "nodes_pen_off.parquet")
    atlas_edges = pd.read_parquet(Path(args.atlas_dir) / "edges_pen_off.parquet")
    atlas_crops = pd.read_parquet(Path(args.atlas_dir) / "crops_pen_off.parquet").set_index("dataset")

    rows_c: list[dict] = []
    rows_r: list[dict] = []
    decision_rows: list[pd.DataFrame] = []
    event_rows: list[dict] = []
    emitted: list[pl.DataFrame] = []
    all_lost_edges: set[tuple[int, int]] = set()
    all_lost_divisions: set[int] = set()
    direct_tp_deleted = 0
    total_removed_edges = 0
    t0 = time.time()
    # This GT-free JSONL is written and flushed before every scorer call.  A killed run therefore
    # leaves its complete processed candidate population on disk instead of losing it at summary.
    ledger_path = out_dir / f"candidate_ledger_{args.tag}.jsonl"
    ledger_file = ledger_path.open("w", encoding="utf-8")

    try:
        for index, name in enumerate(names, 1):
            crop = df.filter(pl.col("dataset") == name)
            nodes, edges = _node_edge_tables(crop)
            decisions = select_duplicate_endpoints(nodes, edges, params)
            decisions.insert(0, "dataset", name)
            result, removed_edges, _ = apply_decisions(crop, decisions)
            emitted.append(result)
            total_removed_edges += len(removed_edges)
            ledger_payload = {
                "dataset": name,
                "input_nodes": int(len(nodes)),
                "input_edges": int(len(edges)),
                "selected": decisions.to_dict("records"),
                "normalized_removed_edges": sorted(removed_edges),
                "params_sha256": params_sha256,
                "source_sha256": source_sha256,
            }
            ledger_file.write(json.dumps(ledger_payload, separators=(",", ":")) + "\n")
            ledger_file.flush()

            gt_geff = Path(args.gt_dir) / f"{name}.geff"
            control_row, _cn, control_edges, _cg, _cgn = ea.run_crop(name, crop, gt_geff)
            result_row, _rn, result_edges, _rg, _rgn = ea.run_crop(name, result, gt_geff)
            for key in ("edge_tp", "edge_fp", "edge_fn", "num_pred_nodes"):
                if int(control_row[key]) != int(atlas_crops.loc[name, key]):
                    raise RuntimeError(f"{name}: control/atlas mismatch for {key}")

            ctrl_pairs = set(map(tuple, control_edges.loc[control_edges.matched, ["gt_src", "gt_tgt"]].astype(int).values))
            rule_pairs = set(map(tuple, result_edges.loc[result_edges.matched, ["gt_src", "gt_tgt"]].astype(int).values))
            lost_pairs, gained_pairs = ctrl_pairs - rule_pairs, rule_pairs - ctrl_pairs
            ctrl_div = _division_tp_ids(crop, gt_geff, ea)
            rule_div = _division_tp_ids(result, gt_geff, ea)
            lost_div, gained_div = ctrl_div - rule_div, rule_div - ctrl_div
            all_lost_edges |= lost_pairs
            all_lost_divisions |= lost_div

            ae = atlas_edges[atlas_edges.dataset.eq(name)].set_index(["source_id", "target_id"])
            crop_direct_tp = 0
            for edge in removed_edges:
                if edge not in ae.index:
                    raise RuntimeError(f"{name}: removed edge absent from frozen atlas: {edge}")
                crop_direct_tp += int(bool(ae.loc[edge, "matched"]))
            direct_tp_deleted += crop_direct_tp

            an = atlas_nodes[atlas_nodes.dataset.eq(name)].set_index("node_id")
            if len(decisions):
                decisions["atlas_gt_id_audit_only"] = decisions.candidate_id.map(an.gt_id).astype(int)
                decisions["direct_scored_tp_edges_deleted"] = [
                    sum(int(bool(ae.loc[tuple(e), "matched"])) for e in json.loads(raw))
                    for raw in decisions.removed_edges
                ]
                decision_rows.append(decisions)
            rows_c.append(control_row)
            rows_r.append(result_row)
            event_rows.append({
            "dataset": name,
            "nodes_removed": int(len(decisions)),
            "edges_removed": int(len(removed_edges)),
            "direct_scored_tp_edges_deleted": crop_direct_tp,
            "lost_gt_edge_pairs": json.dumps(sorted((int(a), int(b)) for a, b in lost_pairs)),
            "gained_gt_edge_pairs": json.dumps(sorted((int(a), int(b)) for a, b in gained_pairs)),
            "lost_division_tp_ids": json.dumps(sorted(lost_div)),
            "gained_division_tp_ids": json.dumps(sorted(gained_div)),
            **{f"control_{k}": control_row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "adj_edge_jaccard", "division_tp", "division_fp", "division_fn")},
            **{f"rule_{k}": result_row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "adj_edge_jaccard", "division_tp", "division_fp", "division_fn")},
            })
            print(
            f"[{index}/{len(names)}] {name} selected={len(decisions)} removed_edges={len(removed_edges)} "
            f"direct_tp_deleted={crop_direct_tp} lost_gt_edges={len(lost_pairs)} lost_divisions={len(lost_div)} "
            f"adj={control_row['adj_edge_jaccard']:.6f}->{result_row['adj_edge_jaccard']:.6f} "
            f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    finally:
        ledger_file.close()

    result_csv = out_dir / f"dedup_{args.tag}.csv.gz"
    full_result = pl.concat(emitted, how="vertical").with_columns(
        pl.Series("id", np.arange(sum(x.height for x in emitted), dtype=np.int64))
    )
    plain = out_dir / f"dedup_{args.tag}.csv"
    full_result.write_csv(plain)
    with plain.open("rb") as source, gzip.open(result_csv, "wb") as target:
        shutil.copyfileobj(source, target)
    plain.unlink()

    decisions_out = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    decisions_out.to_parquet(out_dir / f"decisions_{args.tag}.parquet", index=False)
    pd.DataFrame(event_rows).to_parquet(out_dir / f"events_{args.tag}.parquet", index=False)
    pd.DataFrame(rows_c).to_parquet(out_dir / f"crops_control_{args.tag}.parquet", index=False)
    pd.DataFrame(rows_r).to_parquet(out_dir / f"crops_rule_{args.tag}.parquet", index=False)

    control = _arm_summary(rows_c, summarise)
    rule = _arm_summary(rows_r, summarise)
    delta = {key: float(rule[key] - control[key]) for key in ("score", "adj_edge_jaccard", "division_jaccard")}
    total_selected = int(len(decisions_out))
    full_run = args.max_crops is None and args.stride is None
    safety_gate = {
        "nonzero_heartbeat": total_selected > 0,
        "zero_direct_scored_tp_edge_deletions": direct_tp_deleted == 0,
        "zero_event_level_gt_edge_losses": len(all_lost_edges) == 0,
        "zero_event_level_division_tp_losses": len(all_lost_divisions) == 0,
    }
    packet_gate = {
        "score_delta_at_least_0_003": delta["score"] >= 0.003,
        "adj_delta_at_least_neg_0_0005": delta["adj_edge_jaccard"] >= -0.0005,
    } if full_run else None
    gate_pass = all(safety_gate.values()) and (all(packet_gate.values()) if packet_gate is not None else True)
    summary = {
        "tag": args.tag,
        "csv": str(args.csv),
        "params": params,
        "params_sha256": params_sha256,
        "source_sha256": source_sha256,
        "n_crops": len(names),
        "evaluation_scope": "full" if full_run else "pilot",
        "selected_nodes": total_selected,
        "removed_edges": total_removed_edges,
        "direct_scored_tp_edges_deleted": direct_tp_deleted,
        "event_level_lost_gt_edge_pairs": len(all_lost_edges),
        "event_level_lost_division_tps": len(all_lost_divisions),
        "control": control,
        "rule": rule,
        "delta": delta,
        "paired_bootstrap": _paired_bootstrap(rows_c, rows_r, summarise),
        "safety_gate": safety_gate,
        "packet_gate": packet_gate,
        "gate_pass": gate_pass,
        "result_csv": str(result_csv),
        "candidate_ledger": str(ledger_path),
    }
    (out_dir / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"params_{args.tag}.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--csv", required=True)
    run.add_argument("--atlas-dir", required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--params", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    run.add_argument("--max-crops", type=int)
    run.add_argument("--stride", type=int)
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
