"""Tests for submission scoring: missing datasets and empty predictions.

Reproduces the score-inflation bug (a one-dataset submission scoring ~1.10 while an
omitted expected dataset silently vanished) and confirms omitted datasets are charged
as all-FN. Uses two real local train GEFFs.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import estimated_nodes, load_graph, score_submission  # noqa: E402
from biotrack.submission import graphs_to_submission  # noqa: E402

TRAIN = ROOT / "data" / "train"


def _two_datasets():
    geffs = sorted(TRAIN.glob("*.geff"))
    if len(geffs) < 2:
        pytest.skip("need >=2 local train geffs")
    return geffs[0].stem, geffs[1].stem


def test_missing_dataset_charged_as_all_fn(tmp_path):
    a, b = _two_datasets()
    # submission contains a PERFECT prediction for A only (GT-as-pred), B omitted
    ga = load_graph(str(TRAIN / f"{a}.geff"))
    df = graphs_to_submission({a: ga})
    sub = tmp_path / "submission.csv"
    df.write_csv(sub)

    res = score_submission(sub, TRAIN, expected=[a, b])
    by = {r["dataset"]: r for r in res["rows"]}

    # both datasets must be scored
    assert set(by) == {a, b}, "omitted dataset must still be scored"
    # A is perfect on edges
    assert by[a]["edge_fp"] == 0 and by[a]["edge_fn"] == 0
    # B is all-FN: every GT edge of B is a false negative, zero pred nodes
    gb = load_graph(str(TRAIN / f"{b}.geff"))
    assert by[b]["edge_tp"] == 0
    assert by[b]["edge_fn"] == gb.num_edges()
    assert by[b]["num_pred_nodes"] == 0
    assert by[b]["adj_edge_jaccard"] == 0.0
    # run score must be dragged well below A-only (weighted by edge volume)
    assert res["summary"]["adj_edge_jaccard"] < by[a]["adj_edge_jaccard"]


def test_scoring_only_present_would_inflate(tmp_path):
    """Sanity: scoring A alone (expected=[a]) reports A's high score -- the old bug path."""
    a, _ = _two_datasets()
    ga = load_graph(str(TRAIN / f"{a}.geff"))
    sub = tmp_path / "s.csv"
    graphs_to_submission({a: ga}).write_csv(sub)
    res = score_submission(sub, TRAIN, expected=[a])
    assert res["summary"]["adj_edge_jaccard"] > 1.0  # count-undercount reward on a perfect single set


def test_misspelled_expected_gt_fails_closed(tmp_path):
    a, _ = _two_datasets()
    ga = load_graph(str(TRAIN / f"{a}.geff"))
    sub = tmp_path / "s.csv"
    graphs_to_submission({a: ga}).write_csv(sub)
    with pytest.raises(FileNotFoundError, match="definitely-not-a-crop"):
        score_submission(sub, TRAIN, expected=["definitely-not-a-crop"])


def test_empty_gt_directory_fails_closed(tmp_path):
    a, _ = _two_datasets()
    ga = load_graph(str(TRAIN / f"{a}.geff"))
    sub = tmp_path / "s.csv"
    graphs_to_submission({a: ga}).write_csv(sub)
    gt_dir = tmp_path / "empty-gt"
    gt_dir.mkdir()
    with pytest.raises(ValueError, match="no expected GT datasets"):
        score_submission(sub, gt_dir)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
