r"""Does a change made ONLY at pre-ILP candidate ranking survive into the emitted graph?

THE QUESTION (PKT-0040, category 5 of the host lost-edge ledger)
----------------------------------------------------------------
Every association number the campaign holds (FACT-0413, FACT-0414) is measured at
``stage: pre_ILP_candidate_ranking``. FACT-0364 says ``motion_relink_edges`` replaces the
solver's whole edge list at a median 99.9% coverage. So a learned parent score consumed only
at candidate ranking may be discarded before the final graph exists. That is a measurable
claim, not an argument, and this module measures it.

THE EXPERIMENT
--------------
For one crop, take the deployed pre-ILP candidate export. FACT-0369: at the deployed floor
each target has AT MOST ONE candidate parent, so "re-ranking" a target can only mean swapping
which source it points at. For a sample of targets we do exactly what a learned re-ranker on
the widened surface would do - promote the RUNNER-UP source from the P30/P34 candidate sidecar
to be the target's parent, carrying the deployed edge's own probability so the ILP and the
relink bonus see an equally confident commitment. Nothing else changes. Then the COMPLETE
deployed chain is replayed (``p28_full_chain_replay.load_p28_module`` - the notebook's own
``filter_output_graph``) and the final graph is inspected.

Each forced choice lands in exactly one bucket, first match wins:

  node_gone   the target is not in the final node set at all      (ledger category 2)
  survived    the final graph contains the forced edge (s2, t)
  reverted    the final graph restored the ORIGINAL parent (s, t) (ledger category 5)
  reassigned  the final parent is some third node                 (ledger category 5)
  no_parent   the target survives with no parent at all           (ledger category 5)

The survival rate is ``survived / n_forced``. Its complement, less ``node_gone``, is the mass
that a pre-ILP-only scorer cannot reach.

CONTROL ARM
-----------
The same crop is replayed unperturbed and compared against the kernel's own ``run_stats.csv``
on the columns ``p28_full_chain_replay`` compares. A control that is not exact is refused,
because a chain that does not reproduce the deployed run is not evidence about it. The control
graph also yields the BASELINE agreement rate - how often the deployed pre-ILP parent is the
final parent with no perturbation at all - which is the number the survival rate must be read
against.

CPU only. Reads committed assets; writes only its own JSON payload and cached graphs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from scripts.win_bet.ilp_replay import build_graph_from_frame, solve  # noqa: E402
from scripts.win_bet.p28_full_chain_replay import (  # noqa: E402
    load_p28_module,
    solved_graph_inputs,
    submission_frame,
)

HEARTBEAT = "ASSOC_PREILP_SURVIVAL_COMPLETE"

# The exact columns p28_full_chain_replay compares against the kernel run_stats.
COMPARE_KEYS = [
    "raw_nodes", "raw_edges", "nodes", "edges", "motion_relink_edges",
    "gap_added_nodes", "safe_divisions_added", "deepcenter_gap_checked",
    "deepcenter_gap_accepted", "deepcenter_gap_rejected",
    "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted",
    "deepcenter_safe_div_rejected", "short_track_nodes_removed",
    "linefit_smoothed_nodes",
]


def _node_table(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("row_type") == "node").sort("node_id")


def _edge_table(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("row_type") == "edge")


def _positions_um(nodes: pl.DataFrame, scale: tuple[float, float, float]) -> dict[int, np.ndarray]:
    return {
        int(n): np.array([z * scale[0], y * scale[1], x * scale[2]], dtype=np.float64)
        for n, z, y, x in zip(nodes["node_id"], nodes["z"], nodes["y"], nodes["x"])
    }


def run_chain(module, crop: str, nodes_frame: pl.DataFrame, edges_frame: pl.DataFrame,
              detector, disappearance: float):
    """Candidate graph -> ILP -> the notebook's own filter_output_graph -> submission rows."""
    candidates = build_graph_from_frame(nodes_frame, edges_frame)
    solved = solve(
        candidates,
        {"edge": -1.0, "appearance": 0.0, "disappearance": disappearance, "division": 1.0},
    )
    nodes, edges = solved_graph_inputs(solved)
    raw_nodes, raw_edges = len(nodes), len(edges)
    final_nodes, final_edges, stats = module.filter_output_graph(
        nodes, edges, dataset=crop, deepcenter_bundle=detector
    )
    stats = {**stats, "raw_nodes": raw_nodes, "raw_edges": raw_edges,
             "nodes": len(final_nodes), "edges": len(final_edges)}
    return submission_frame(crop, final_nodes, final_edges), stats


def parent_map(frame: pl.DataFrame) -> dict[int, int]:
    edges = _edge_table(frame)
    return {int(t): int(s) for s, t in zip(edges["source_id"], edges["target_id"])}


def node_set(frame: pl.DataFrame) -> set[int]:
    return {int(n) for n in _node_table(frame)["node_id"]}


def build_perturbation(nodes_frame: pl.DataFrame, edges_frame: pl.DataFrame, sidecar: Path,
                       n_target: int, seed: int, scale: tuple[float, float, float],
                       relaxed_um: float):
    """Promote the sidecar runner-up to be the parent, for a random sample of targets."""
    with np.load(sidecar, allow_pickle=False) as z:
        s_all = z["source_id"].astype(np.int64)
        t_all = z["target_id"].astype(np.int64)
        p_all = z["edge_prob"].astype(np.float64)

    deployed = {int(t): (int(s), float(p)) for s, t, p in
                zip(edges_frame["source_id"], edges_frame["target_id"], edges_frame["edge_prob"])}
    frame_of = {int(n): int(t) for n, t in zip(nodes_frame["node_id"], nodes_frame["t"])}
    pos = _positions_um(nodes_frame, scale)

    best_alt: dict[int, tuple[float, int]] = {}
    for s, t, p in zip(s_all, t_all, p_all):
        t = int(t); s = int(s)
        cur = deployed.get(t)
        if cur is None or s == cur[0]:
            continue
        if frame_of.get(s) is None or frame_of.get(t) is None:
            continue
        if frame_of[s] + 1 != frame_of[t]:
            continue                      # never offer a non-consecutive pair
        prev = best_alt.get(t)
        if prev is None or p > prev[0]:
            best_alt[t] = (float(p), s)

    eligible = sorted(best_alt)
    rng = np.random.default_rng(seed)
    if n_target > 0 and len(eligible) > n_target:
        picked = sorted(rng.choice(np.array(eligible), size=n_target, replace=False).tolist())
    else:
        picked = eligible

    forced: dict[int, dict] = {}
    for t in picked:
        alt_prob, s2 = best_alt[t]
        s0, p0 = deployed[t]
        forced[int(t)] = {
            "target": int(t),
            "deployed_parent": int(s0),
            "deployed_prob": float(p0),
            "forced_parent": int(s2),
            "forced_alt_native_prob": float(alt_prob),
            "deployed_dist_um": float(np.linalg.norm(pos[t] - pos[s0])),
            "forced_dist_um": float(np.linalg.norm(pos[t] - pos[s2])),
        }
        forced[int(t)]["forced_within_relink_gate"] = bool(
            forced[int(t)]["forced_dist_um"] <= relaxed_um
        )

    src = edges_frame["source_id"].to_list()
    tgt = edges_frame["target_id"].to_list()
    new_src = [forced[int(t)]["forced_parent"] if int(t) in forced else int(s)
               for s, t in zip(src, tgt)]
    perturbed = edges_frame.with_columns(pl.Series("source_id", new_src, dtype=pl.Int64))
    return perturbed, forced


def attribute(forced: dict[int, dict], final: pl.DataFrame) -> dict:
    parents = parent_map(final)
    nodes = node_set(final)
    buckets = {"node_gone": 0, "survived": 0, "reverted": 0, "reassigned": 0, "no_parent": 0}
    gated = {k: 0 for k in buckets}
    n_gated = 0
    for t, rec in forced.items():
        if t not in nodes:
            key = "node_gone"
        else:
            p = parents.get(t)
            if p is None:
                key = "no_parent"
            elif p == rec["forced_parent"]:
                key = "survived"
            elif p == rec["deployed_parent"]:
                key = "reverted"
            else:
                key = "reassigned"
        buckets[key] += 1
        rec["outcome"] = key
        if rec["forced_within_relink_gate"]:
            gated[key] += 1
            n_gated += 1
    return {"n_forced": len(forced), "buckets": buckets,
            "n_forced_within_relink_gate": n_gated, "buckets_within_relink_gate": gated}


def baseline(edges_frame: pl.DataFrame, control: pl.DataFrame) -> dict:
    """Unperturbed agreement between the pre-ILP parent and the final parent."""
    pre = {int(t): int(s) for s, t in zip(edges_frame["source_id"], edges_frame["target_id"])}
    parents = parent_map(control)
    nodes = node_set(control)
    out = {"node_gone": 0, "same": 0, "different": 0, "no_parent": 0}
    for t, s in pre.items():
        if t not in nodes:
            out["node_gone"] += 1
        elif t not in parents:
            out["no_parent"] += 1
        elif parents[t] == s:
            out["same"] += 1
        else:
            out["different"] += 1
    return {"n_preilp_parents": len(pre), **out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", type=Path, required=True)
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--ecb-dir", type=Path, required=True)
    ap.add_argument("--expect-stats", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, default=ROOT / "data/train")
    ap.add_argument("--crop", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--n-targets", type=int, default=400)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--disappearance-weight", type=float, default=1.5)
    ap.add_argument("--control-graph-dir", type=Path,
                    help="reuse a persisted control graph for this crop if present")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    module = load_p28_module(args.notebook, args.train_dir, args.checkpoint)
    scale = tuple(float(v) for v in module.VOXEL_SCALE_UM)
    relaxed_um = float(module.MOTION_RELINK_RELAXED_UM)

    frame = pl.read_parquet(args.preilp).filter(pl.col("dataset") == args.crop)
    if frame.is_empty():
        raise SystemExit(f"crop absent from pre-ILP export: {args.crop}")
    nodes_frame = _node_table(frame)
    edges_frame = _edge_table(frame)

    import os
    previous = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT")
    os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(args.checkpoint)
    try:
        detector = module.load_deepcenter_veto_detector()
    finally:
        if previous is None:
            os.environ.pop("BIOHUB_DEEPCENTER_CHECKPOINT", None)
        else:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = previous

    cached = None
    if args.control_graph_dir is not None:
        candidate = args.control_graph_dir / f"{args.crop}.parquet"
        if candidate.exists():
            cached = pl.read_parquet(candidate)

    control_exact = None
    comparison = None
    if cached is not None:
        control = cached
        control_source = str(args.control_graph_dir / f"{args.crop}.parquet")
    else:
        control, cstats = run_chain(module, args.crop, nodes_frame, edges_frame,
                                    detector, args.disappearance_weight)
        control_source = "recomputed"
        rows = pl.read_csv(args.expect_stats).filter(pl.col("dataset") == args.crop)
        if rows.height != 1:
            raise SystemExit(f"expected exactly one control stats row for {args.crop}")
        expected = rows.row(0, named=True)
        comparison = {k: {"got": int(cstats[k]), "expected": int(expected[k]),
                          "delta": int(cstats[k]) - int(expected[k])} for k in COMPARE_KEYS}
        control_exact = all(v["delta"] == 0 for v in comparison.values())
        if not control_exact:
            raise SystemExit(f"control parity failed for {args.crop}; refusing to report")
        if args.control_graph_dir is not None:
            args.control_graph_dir.mkdir(parents=True, exist_ok=True)
            control.write_parquet(args.control_graph_dir / f"{args.crop}.parquet")

    perturbed_edges, forced = build_perturbation(
        nodes_frame, edges_frame, args.ecb_dir / f"{args.crop}.npz",
        args.n_targets, args.seed, scale, relaxed_um,
    )
    treated, tstats = run_chain(module, args.crop, nodes_frame, perturbed_edges,
                               detector, args.disappearance_weight)

    result = {
        "schema_version": 1,
        "heartbeat": HEARTBEAT,
        "crop": args.crop,
        "fold": args.fold,
        "notebook": str(args.notebook),
        "notebook_code_sha256": module._source_sha256,
        "seed": args.seed,
        "n_targets_requested": args.n_targets,
        "motion_relink_relaxed_um": relaxed_um,
        "control_source": control_source,
        "control_exact": control_exact,
        "control_comparison": comparison,
        "baseline_unperturbed": baseline(edges_frame, control),
        "attribution": attribute(forced, treated),
        "control_final_edges": _edge_table(control).height,
        "treated_final_edges": _edge_table(treated).height,
        "treated_stats": {k: int(v) for k, v in tstats.items()},
        "forced": sorted(forced.values(), key=lambda r: r["target"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    a = result["attribution"]
    print(f"{HEARTBEAT} crop={args.crop} fold={args.fold} "
          f"n={a['n_forced']} survived={a['buckets']['survived']} "
          f"reverted={a['buckets']['reverted']} reassigned={a['buckets']['reassigned']} "
          f"no_parent={a['buckets']['no_parent']} node_gone={a['buckets']['node_gone']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
