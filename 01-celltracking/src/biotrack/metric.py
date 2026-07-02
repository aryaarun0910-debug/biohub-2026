"""Exact local metric harness.

Wraps the organizer's ``tracking_cellmot`` scoring so we can score predictions
offline with embryo-grouped folds and trust it more than the 29% public LB.

The leaderboard score is
    score = weighted_avg(adj_edge_jaccard) + 0.1 * division_jaccard
where adj_edge_jaccard per sample = max(0, J * (1 - 0.1 * (N_pred - N_est)/N_est))
and the weighting is by per-sample edge volume (TP+FP+FN). See
``tracking_cellmot.metrics.summarise``.

Only GEFF graphs and the fixed voxel scale are needed -- NOT the image volumes.
"""

from pathlib import Path

import tracksdata as td
from geff import GeffMetadata

from tracking_cellmot.metrics import (
    evaluate as _evaluate,
    node_recall as _node_recall,
    per_sample_metrics,
    summarise,
)

# Fixed competition voxel scale (z, y, x) in microns; matches io.DEFAULT_SCALE.
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
MAX_DISTANCE: float = 7.0  # microns; organizer default


def load_graph(geff_path: str | Path) -> td.graph.BaseGraph:
    res = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    return res[0] if isinstance(res, tuple) else res


def estimated_nodes(geff_path: str | Path) -> float:
    try:
        meta = GeffMetadata.read(Path(geff_path))
    except Exception:
        return float("nan")
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def score_pred_graph(
    pred: td.graph.BaseGraph,
    gt_geff: str | Path,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Per-sample metric row for an in-memory predicted graph vs a GT geff.

    ``gt_geff`` supplies the ground-truth graph and the
    ``estimated_number_of_nodes`` used by the count adjustment.
    """
    gt = load_graph(gt_geff)
    er = _evaluate(pred, gt, scale=scale, max_distance=max_distance)
    if pred.num_edges() > 0 and pred.num_nodes() > 0:
        recall = _node_recall(pred, gt)
    else:
        recall = 0.0
    return {"dataset": Path(gt_geff).stem, **per_sample_metrics(er, estimated_nodes(gt_geff), recall)}


def score_one(
    pred_geff: str | Path,
    gt_geff: str | Path,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Per-sample metric row for a single (pred geff, gt geff) embryo pair."""
    return score_pred_graph(load_graph(pred_geff), gt_geff, scale, max_distance)


def score_many(
    pairs: list[tuple[str | Path, str | Path]],
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Run-level summary (LB-faithful) over a list of (pred_geff, gt_geff) pairs."""
    rows = [score_one(p, g, scale, max_distance) for p, g in pairs]
    return {"rows": rows, "summary": summarise(rows)}


def _empty_graph() -> "td.graph.BaseGraph":
    """An empty predicted graph (0 nodes/edges) -> scores as all-FN vs any GT."""
    import polars as pl
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    return g


def score_submission(
    submission_path: str | Path,
    gt_dir: str | Path,
    expected: list[str] | None = None,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Score a Kaggle ``submission.csv`` against GT ``.geff`` files in *gt_dir*.

    Scores over the FULL EXPECTED dataset set, not just datasets present in the CSV:
    a dataset that is expected but MISSING from the submission is charged as an empty
    prediction (all its GT edges become FN, score 0). This prevents silently inflating
    the score by omitting hard datasets.

    ``expected``: dataset stems that must be scored. Defaults to every ``*.geff`` in
    *gt_dir* (i.e. score the whole fold/split). Pass the split's test list to score a fold.
    """
    from biotrack.submission import read_submission, submission_to_graphs

    gt_dir = Path(gt_dir)
    graphs = submission_to_graphs(read_submission(submission_path))
    if expected is None:
        expected = sorted(p.stem for p in gt_dir.glob("*.geff"))
    rows = []
    for dataset in sorted(expected):
        gt_geff = gt_dir / f"{dataset}.geff"
        if not gt_geff.exists():
            continue  # no GT to score against; genuinely not part of the set
        pred = graphs.get(dataset, _empty_graph())  # missing prediction -> all-FN
        rows.append(score_pred_graph(pred, gt_geff, scale, max_distance))
    # warn if the submission contains datasets that aren't in the expected set
    extra = sorted(set(graphs) - set(expected))
    result = {"rows": rows, "summary": summarise(rows)}
    if extra:
        result["unexpected_datasets"] = extra
    return result
