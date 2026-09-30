r"""One-factor solver sweeps on the CPU ILP replay, scored by the official metric.

WHAT IT DOES
------------
For each requested solver setting it replays the ILP over the same pre-ILP candidate graph,
optionally applies the deployed short-track filter, writes the result in submission form and
scores it with the official ``tracking_cellmot`` metric. Settings differ from the baseline in
exactly ONE term unless the caller asks for an interaction, so an effect can be attributed.

RAW AND ADJUSTED JACCARD ARE BOTH REPORTED, ALWAYS
--------------------------------------------------
The composite is ``adj_edge_jaccard = max(0, J * (1 - alpha * (N_pred - N_est)/N_est))``
(``tracking_cellmot/metrics.py:447``). Any change that moves the node count moves the second
factor whether or not the tracking improved, so a solver sweep - which moves the node count by
construction - can post a gain that is pure count arithmetic. ``edge_jaccard`` (raw) is therefore
reported beside ``adj_edge_jaccard`` for every setting and every crop, and the summary flags any
setting whose adjusted gain is not accompanied by a raw gain.

WHAT IT DOES NOT MEASURE
------------------------
The scored object is ILP output plus, optionally, the short-track filter. It is NOT the deployed
champion configuration, which also runs motion relink, gap closure, DeepCenter vetoes and line
fitting after the solver. Absolute scores here are therefore below the champion's and are not
comparable to it. Only the DELTA between settings, measured on this same reduced pipeline, is the
signal - and even that delta has to survive the full stack before it means anything on the
leaderboard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ilp_replay import DEPLOYED_WEIGHTS, replay  # noqa: E402

# Deployed short-track filter, verified in the P28 notebook env block.
DEPLOYED_MIN_TRACK_LEN = 6


def apply_short_track_filter(rows: pl.DataFrame, min_len: int) -> pl.DataFrame:
    """Drop weakly-connected components with fewer than ``min_len`` nodes.

    Mirrors BIOHUB_OUTPUT_MIN_TRACK_LEN, which removes short components after linking.
    """
    if min_len <= 1:
        return rows
    nodes = rows.filter(pl.col("row_type") == "node")
    edges = rows.filter(pl.col("row_type") == "edge")
    parent: dict[int, int] = {int(n): int(n) for n in nodes["node_id"]}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in zip(edges["source_id"], edges["target_id"]):
        s, t = int(s), int(t)
        if s in parent and t in parent:
            ra, rb = find(s), find(t)
            if ra != rb:
                parent[ra] = rb

    sizes: dict[int, int] = {}
    for n in parent:
        r = find(n)
        sizes[r] = sizes.get(r, 0) + 1
    keep = {n for n in parent if sizes[find(n)] >= min_len}
    return pl.concat([
        nodes.filter(pl.col("node_id").is_in(list(keep))),
        edges.filter(
            pl.col("source_id").is_in(list(keep)) & pl.col("target_id").is_in(list(keep))
        ),
    ])


def assemble_csv(graph_dir: Path, crops: list[str], min_len: int, out_csv: Path) -> None:
    frames = []
    for crop in crops:
        rows = pl.read_parquet(graph_dir / f"{crop}.parquet")
        frames.append(apply_short_track_filter(rows, min_len))
    table = pl.concat(frames)
    table = table.with_row_index("id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_csv)


def score_csv(csv: Path, gt_dir: Path) -> tuple[dict, list[dict]]:
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph
    from biotrack.submission import read_submission, submission_to_graphs
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

    graphs = submission_to_graphs(read_submission(csv))
    rows = []
    for name in sorted(graphs):
        gt_geff = gt_dir / f"{name}.geff"
        if not gt_geff.exists():
            continue
        pred = graphs[name]
        gt = load_graph(gt_geff)
        er = evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt) if pred.num_edges() and pred.num_nodes() else 0.0
        rows.append({"dataset": name, **per_sample_metrics(er, estimated_nodes(gt_geff), recall)})
    return summarise(rows), rows


def paired_bootstrap(base: list[dict], arm: list[dict], key: str, n: int = 2000) -> dict:
    b = {r["dataset"]: r[key] for r in base}
    a = {r["dataset"]: r[key] for r in arm}
    shared = sorted(set(b) & set(a))
    d = np.array([a[k] - b[k] for k in shared], dtype=np.float64)
    d = d[~np.isnan(d)]
    if not len(d):
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(0)
    draws = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--factor", required=True,
                    choices=["appearance", "disappearance", "division", "edge", "min_track_len"])
    ap.add_argument("--values", required=True, help="comma-separated values for the swept factor")
    ap.add_argument("--min-track-len", type=int, default=DEPLOYED_MIN_TRACK_LEN)
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--crop-stride", type=int, default=1,
                    help="take every Nth crop, for a size-spread pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    crops = sorted(pl.read_parquet(args.preilp, columns=["dataset"])["dataset"].unique().to_list())
    crops = crops[:: args.crop_stride]
    if args.max_crops:
        crops = crops[: args.max_crops]
    values = [float(v) for v in args.values.split(",")]

    baseline_key = (
        DEPLOYED_MIN_TRACK_LEN if args.factor == "min_track_len"
        else DEPLOYED_WEIGHTS[args.factor]
    )
    if baseline_key not in values:
        values = [baseline_key] + values
        print(f"baseline {args.factor}={baseline_key} prepended so every arm has a paired control")

    arms: list[dict] = []
    base_rows: list[dict] | None = None
    for value in values:
        weights = dict(DEPLOYED_WEIGHTS)
        min_len = args.min_track_len
        if args.factor == "min_track_len":
            min_len = int(value)
        else:
            weights[args.factor] = value
        tag = f"{args.factor}_{value}"
        graph_dir = args.work_dir / tag
        print(f"\n=== arm {tag}  weights={weights} min_track_len={min_len}", flush=True)
        if args.factor != "min_track_len" or not arms:
            replay(args.preilp, crops, weights, args.workers, args.timeout, graph_dir)
        else:
            # min_track_len does not change the solve; reuse the first arm's graphs.
            graph_dir = args.work_dir / f"{args.factor}_{values[0]}"
        csv = args.work_dir / f"{tag}.csv"
        assemble_csv(graph_dir, crops, min_len, csv)
        summary, rows = score_csv(csv, args.gt_dir)
        arm = {
            "tag": tag,
            "factor": args.factor,
            "value": value,
            "is_baseline": value == baseline_key,
            "min_track_len": min_len,
            "weights": weights,
            "summary": {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in summary.items()
            },
        }
        if arm["is_baseline"]:
            base_rows = rows
        arms.append(arm)
        arm["_rows"] = rows
        print(f"  {tag}: score={summary.get('score')} adjJ={summary.get('adj_edge_jaccard')} "
              f"rawJ={summary.get('edge_jaccard')} recall={summary.get('node_recall')}", flush=True)

    for arm in arms:
        rows = arm.pop("_rows")
        if base_rows is None or arm["is_baseline"]:
            arm["paired_vs_baseline"] = None
            continue
        arm["paired_vs_baseline"] = {
            "adj_edge_jaccard": paired_bootstrap(base_rows, rows, "adj_edge_jaccard"),
            "edge_jaccard_raw": paired_bootstrap(base_rows, rows, "edge_jaccard"),
            "node_recall": paired_bootstrap(base_rows, rows, "node_recall"),
        }
        adj = arm["paired_vs_baseline"]["adj_edge_jaccard"]["mean"]
        raw = arm["paired_vs_baseline"]["edge_jaccard_raw"]["mean"]
        # The masking check the packet's falsifier (d) asks for.
        arm["adjusted_gain_without_raw_gain"] = bool(
            adj is not None and raw is not None and adj > 0 >= raw
        )

    result = {
        "schema_version": 1,
        "preilp": str(args.preilp),
        "factor": args.factor,
        "n_crops": len(crops),
        "crops": crops,
        "deployed_weights": DEPLOYED_WEIGHTS,
        "deployed_min_track_len": DEPLOYED_MIN_TRACK_LEN,
        "arms": arms,
        "caveat": (
            "ILP + short-track filter only; the deployed champion also runs motion relink, gap "
            "closure, DeepCenter vetoes and line fitting. Deltas between arms are the signal; "
            "absolute scores are not comparable to the champion."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    print(f"\nILP_SWEEP factor={args.factor} crops={len(crops)}")
    for arm in arms:
        p = arm["paired_vs_baseline"]
        mark = " (baseline)" if arm["is_baseline"] else ""
        line = (f"  {arm['tag']:<28} score={arm['summary'].get('score')}"
                f" adjJ={arm['summary'].get('adj_edge_jaccard')}"
                f" rawJ={arm['summary'].get('edge_jaccard')}{mark}")
        if p:
            line += (f"\n      dAdjJ={p['adj_edge_jaccard']['mean']:+.5f} "
                     f"CI{p['adj_edge_jaccard']['ci95']} | "
                     f"dRawJ={p['edge_jaccard_raw']['mean']:+.5f} "
                     f"CI{p['edge_jaccard_raw']['ci95']}")
            if arm["adjusted_gain_without_raw_gain"]:
                line += "\n      WARNING adjusted gain with NO raw gain - count-adjustment artifact"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
