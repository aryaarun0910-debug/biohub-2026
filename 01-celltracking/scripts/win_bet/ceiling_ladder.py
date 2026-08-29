r"""The ceiling ladder: how much score each stage could give if it were PERFECT.

WHY THIS EXISTS
---------------
The campaign's target was reframed on 2026-08-29 from "beat the top-3 boundary" to "make this
machine reach its own ceiling". That is a different and better-posed question, and it is
answerable offline: for a fixed substrate, replace one stage at a time with an oracle that uses
the ground truth, score the result through the OFFICIAL metric, and read off what that stage
could contribute at most.

THE LADDER
----------
Each rung keeps everything below it fixed and perfects one more thing. Every rung is scored by
``tracking_cellmot`` exactly as the leaderboard would.

  0  deployed          the champion export, as shipped
  1  oracle_edges      KEEP the deployed node set; replace the edge set with the best one
                       possible for those nodes - every GT edge whose endpoints are both matched,
                       and nothing else. Ceiling on ASSOCIATION with today's nodes.
  2  oracle_nodes      KEEP the deployed pipeline's candidate peaks; select exactly those that
                       match a GT cell, then perfect the edges. Ceiling on NODE SELECTION given
                       what the detector proposed.
  3  oracle_detection  use the GT nodes themselves with perfect edges. Ceiling of the whole
                       architecture; the residual to 1.0 is metric structure, not modelling.

Rung 3 is also the instrument's own sanity check: if perfect nodes and perfect edges do not score
essentially 1.0, the harness is wrong and no other rung means anything.

READ THE RAW JACCARD, NOT THE ADJUSTED SCORE
--------------------------------------------
Rungs 2 and 3 restrict the prediction to cells that are ANNOTATED, which is far fewer nodes than
the crop actually contains - `estimated_number_of_nodes` counts all cells, annotated or not. The
adjusted score multiplies by ``1 - 0.1 * total_node_ratio`` with no upper cap (``FACT-0191``), so
those rungs collect a large under-production bonus and can report an adjusted score ABOVE 1.0.
That bonus is an artifact of the oracle knowing which cells are annotated; no real method has it.

So the ladder is read on ``edge_jaccard`` (raw), which the node count does not touch, and each
rung reports its node count and a ``count_bonus_inflated`` flag. Rung 1 is the exception: it holds
the deployed node set fixed, so its adjusted number is directly comparable to the deployed one.

READ THESE AS CEILINGS, NOT FORECASTS
-------------------------------------
An oracle says what is reachable if a stage were solved, not what any real method will get.
``FACT-0334`` measured a GT-guided division oracle at +0.0192/+0.0133 and ``FACT-0337`` then found
the GT-free version scored NEGATIVE - the oracle was the ground truth doing the choosing. So the
ladder is for ALLOCATING EFFORT between stages, and every rung must be labelled a ceiling wherever
it is quoted.

WHAT THE GAPS MEAN
------------------
  deployed -> rung 1   what better linking alone can buy on today's node set
  rung 1  -> rung 2    what better node selection can buy from today's candidate peaks
  rung 2  -> rung 3    what the detector's candidate set cannot reach at all
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

from detpeak_curve import (  # noqa: E402
    DOWNSAMPLE,
    MAX_DISTANCE_UM,
    SCALE_UM,
    match_one_to_one_pairs,
    sigmoid,
)

SCALE = np.asarray(SCALE_UM, dtype=np.float64)
DOWN = np.asarray(DOWNSAMPLE, dtype=np.float64)
RUNGS = ("deployed", "oracle_edges", "oracle_nodes", "oracle_detection")


def _rows(crop: str, coords: np.ndarray, edges: list[tuple[int, int]]) -> pl.DataFrame:
    """(t,z,y,x) array plus index-pair edges -> submission rows."""
    n = len(coords)
    node_rows = pl.DataFrame({
        "dataset": [crop] * n, "row_type": ["node"] * n,
        "node_id": list(range(n)),
        "t": coords[:, 0].astype("int64").tolist(),
        "z": coords[:, 1].astype("float64").tolist(),
        "y": coords[:, 2].astype("float64").tolist(),
        "x": coords[:, 3].astype("float64").tolist(),
        "source_id": [-1] * n, "target_id": [-1] * n,
    })
    if not edges:
        return node_rows
    m = len(edges)
    edge_rows = pl.DataFrame({
        "dataset": [crop] * m, "row_type": ["edge"] * m,
        "node_id": [-1] * m, "t": [-1] * m,
        "z": [-1.0] * m, "y": [-1.0] * m, "x": [-1.0] * m,
        "source_id": [int(a) for a, _ in edges],
        "target_id": [int(b) for _, b in edges],
    })
    return pl.concat([node_rows, edge_rows])


def match_gt_to_pred(gt: np.ndarray, pred: np.ndarray) -> dict[int, int]:
    """GT row index -> predicted row index, official one-to-one 7 um, per frame."""
    out: dict[int, int] = {}
    for frame in np.unique(gt[:, 0]).astype(np.int64):
        gm = np.nonzero(gt[:, 0] == frame)[0]
        pm = np.nonzero(pred[:, 0] == frame)[0]
        if not len(pm):
            continue
        pairs = match_one_to_one_pairs(pred[pm, 1:] * SCALE, gt[gm, 1:] * SCALE, MAX_DISTANCE_UM)
        for g, p, _ in pairs:
            out[int(gm[g])] = int(pm[p])
    return out


def oracle_edges_for(gt_edges: list[tuple[int, int]], mapping: dict[int, int]) -> list[tuple[int, int]]:
    """Best edge set achievable on a node set: every GT edge whose BOTH endpoints are matched.

    This maximises edge_tp and sets edge_fp to zero simultaneously, so no other edge set on the
    same nodes can score higher. Degrees stay legal because GT lineage degrees are legal.
    """
    out = []
    for u, v in gt_edges:
        if u in mapping and v in mapping:
            out.append((mapping[u], mapping[v]))
    return out


def score_rows(rows: pl.DataFrame, gt_geff: Path) -> dict:
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph
    from biotrack.submission import read_submission, submission_to_graphs
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        tmp = Path(fh.name)
    # The deployed export already carries `id`; the oracle rungs do not. Normalise either way.
    rows.drop("id", strict=False).with_row_index("id").write_csv(tmp)
    try:
        graphs = submission_to_graphs(read_submission(tmp))
        name = next(iter(graphs))
        pred = graphs[name]
        gt = load_graph(gt_geff)
        er = evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt) if pred.num_edges() and pred.num_nodes() else 0.0
        return per_sample_metrics(er, estimated_nodes(gt_geff), recall)
    finally:
        tmp.unlink(missing_ok=True)


def crop_ladder(
    crop: str,
    gt_geff: Path,
    deployed: pl.DataFrame,
    peaks_npz: Path | None,
) -> dict[str, dict]:
    from biotrack.metric import load_graph

    gt_graph = load_graph(gt_geff)
    gtn = gt_graph.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_index = {int(v): i for i, v in enumerate(gt_ids)}
    gt = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt_graph.edge_attrs().to_pandas()
    gt_edges = [
        (gt_index[int(s)], gt_index[int(t)])
        for s, t in zip(gte["source_id"], gte["target_id"])
        if int(s) in gt_index and int(t) in gt_index
    ]

    dep_nodes = deployed.filter(pl.col("row_type") == "node").sort("node_id")
    dep = dep_nodes.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)

    out: dict[str, dict] = {}
    out["deployed"] = score_rows(deployed, gt_geff)

    # RUNG 1 - deployed nodes, perfect edges.
    mapping = match_gt_to_pred(gt, dep)
    out["oracle_edges"] = score_rows(
        _rows(crop, dep, oracle_edges_for(gt_edges, mapping)), gt_geff
    )

    # RUNG 2 - the detector's candidate peaks, perfectly selected, perfect edges.
    if peaks_npz is not None and peaks_npz.exists():
        with np.load(peaks_npz, allow_pickle=False) as z:
            times = z["t"].astype(np.int64)
            zyx = z["zyx"].astype(np.float64) * DOWN
            keep = sigmoid(z["logit"].astype(np.float64)) > float(z["pipeline_threshold"])
        cand = np.column_stack([times[keep], zyx[keep]])
        cmap = match_gt_to_pred(gt, cand)
        chosen = sorted(set(cmap.values()))
        reindex = {c: i for i, c in enumerate(chosen)}
        sel_map = {g: reindex[p] for g, p in cmap.items()}
        out["oracle_nodes"] = score_rows(
            _rows(crop, cand[chosen], oracle_edges_for(gt_edges, sel_map)), gt_geff
        )

    # RUNG 3 - GT nodes, GT edges. Also the harness's own sanity check.
    ident = {i: i for i in range(len(gt))}
    out["oracle_detection"] = score_rows(
        _rows(crop, gt, oracle_edges_for(gt_edges, ident)), gt_geff
    )
    return out


def summarise_rung(rows: list[dict], deployed_nodes: int | None = None) -> dict:
    from tracking_cellmot.metrics import summarise

    s = summarise(rows)
    out = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in s.items()}
    nodes = int(sum(r["num_pred_nodes"] for r in rows))
    out["num_pred_nodes"] = nodes
    # A rung predicting materially fewer nodes than the deployed export is collecting an
    # under-production bonus the oracle earned by knowing which cells are annotated.
    out["count_bonus_inflated"] = bool(
        deployed_nodes is not None and nodes < 0.95 * deployed_nodes
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deployed-csv", type=Path, required=True)
    ap.add_argument("--peaks-dir", type=Path, help="DetPeak sidecars; enables the oracle_nodes rung")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--crop-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = pl.read_csv(args.deployed_csv)
    crops = sorted(sub["dataset"].unique().to_list())[:: args.crop_stride]
    if args.max_crops:
        crops = crops[: args.max_crops]

    per_rung: dict[str, list[dict]] = {r: [] for r in RUNGS}
    per_crop: list[dict] = []
    for i, crop in enumerate(crops, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {crop}")
        peaks = (args.peaks_dir / f"{crop}.npz") if args.peaks_dir else None
        got = crop_ladder(crop, gt_geff, sub.filter(pl.col("dataset") == crop), peaks)
        row = {"crop": crop}
        for rung, metrics in got.items():
            per_rung[rung].append({"dataset": crop, **metrics})
            row[rung] = metrics["adj_edge_jaccard"]
        per_crop.append(row)
        print(f"  [{i}/{len(crops)}] {crop} " + " ".join(
            f"{r}={got[r]['adj_edge_jaccard']:.4f}" for r in RUNGS if r in got
        ), flush=True)

    deployed_nodes = int(sum(r["num_pred_nodes"] for r in per_rung["deployed"]))
    ladder = {r: summarise_rung(v, deployed_nodes) for r, v in per_rung.items() if v}
    result = {
        "schema_version": 1,
        "deployed_csv": str(args.deployed_csv),
        "n_crops": len(crops),
        "ladder": ladder,
        "per_crop": per_crop,
        "caveat": (
            "Every rung above `deployed` uses the ground truth to make a choice. These are "
            "CEILINGS for allocating effort, not forecasts - FACT-0334/FACT-0337 is the standing "
            "example of a GT-guided oracle whose GT-free version scored negative."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    print(f"\nCEILING_LADDER crops={len(crops)}   (read rawJ; adjJ is inflated where flagged)")
    prev = None
    for rung in RUNGS:
        if rung not in ladder:
            continue
        s = ladder[rung]
        gap = "" if prev is None else f"  (+{s['edge_jaccard'] - prev:.4f} rawJ over previous)"
        flag = "  [adjJ INFLATED by count bonus]" if s["count_bonus_inflated"] else ""
        print(f"  {rung:<18} rawJ={s['edge_jaccard']:.4f}  adjJ={s['adj_edge_jaccard']:.4f}  "
              f"recall={s['node_recall']:.4f}  nodes={s['num_pred_nodes']:,}{gap}{flag}")
        prev = s["edge_jaccard"]
    if "oracle_detection" in ladder:
        raw = ladder["oracle_detection"]["edge_jaccard"]
        if raw < 0.999:
            print(f"  WARNING rung 3 rawJ={raw:.4f}, not ~1.0 - the harness is suspect, "
                  "do not read the other rungs")
        else:
            print(f"  harness check OK: rung 3 rawJ={raw:.4f}")
        total = raw - ladder["deployed"]["edge_jaccard"]
        if total > 0:
            share = {
                "association": (ladder["oracle_edges"]["edge_jaccard"]
                                - ladder["deployed"]["edge_jaccard"]) / total,
            }
            if "oracle_nodes" in ladder:
                share["node_selection"] = (ladder["oracle_nodes"]["edge_jaccard"]
                                           - ladder["oracle_edges"]["edge_jaccard"]) / total
                share["detection"] = (raw - ladder["oracle_nodes"]["edge_jaccard"]) / total
            print("  share of the raw-Jaccard gap: " + "  ".join(
                f"{k}={v:.1%}" for k, v in share.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
