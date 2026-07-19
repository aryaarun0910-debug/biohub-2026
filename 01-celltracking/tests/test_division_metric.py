"""Division-term fixtures (Codex metric hardening): a synthetic fork + a real division-bearing crop.

The numpy gate is EDGE-only; divisions route through the authoritative tracksdata harness
(biotrack.metric -> tracking_cellmot.division_metrics). These pin that path:
  - a hand-built mother->2-daughters graph is exactly one division;
  - GT-as-pred on a crop with GT divisions scores division_jaccard == 1.0 (tp=all, fp=fn=0).
"""
import sys
from pathlib import Path

import polars as pl
import pytest
import tracksdata as td

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "train"
DIV_CROP = "6bba_09961292"  # 4 annotated GT divisions


def _fork_graph() -> td.graph.BaseGraph:
    """One mother at t=0 -> two daughters at t=1: a single division event."""
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([
        {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
        {"t": 1, "z": 0.0, "y": -1.0, "x": 0.0},
    ])
    g.bulk_add_edges([
        {"source_id": ids[0], "target_id": ids[1]},
        {"source_id": ids[0], "target_id": ids[2]},
    ])
    return g


def _off_volume_hub_exploit_graph() -> td.graph.BaseGraph:
    """Old-metric exploit: a remote fork weakly connects two real branches.

    The real nodes reproduce a GT division window without a local fork.  A
    synthetic node far outside space and time connects their roots, which the
    pre-2026-07-17 evaluator incorrectly accepted as the division event.
    """
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([
        {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},       # parent
        {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},       # divider position
        {"t": 2, "z": 0.0, "y": 1.0, "x": 0.0},       # daughter 1
        {"t": 2, "z": 0.0, "y": -1.0, "x": 0.0},      # daughter 2
        {"t": -1000, "z": -10000.0, "y": -10000.0, "x": -10000.0},
    ])
    g.bulk_add_edges([
        {"source_id": ids[0], "target_id": ids[1]},
        {"source_id": ids[1], "target_id": ids[2]},
        # The second daughter is connected only through the fake remote hub.
        {"source_id": ids[4], "target_id": ids[0]},
        {"source_id": ids[4], "target_id": ids[3]},
    ])
    return g


def test_synthetic_fork_is_one_division():
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE
    from tracking_cellmot.metrics import evaluate

    er = evaluate(_fork_graph(), _fork_graph(), scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    assert (er.division_tp, er.division_fp, er.division_fn) == (1, 0, 0)


def test_off_volume_weak_component_hub_cannot_score_division():
    """Regression for the patched public division-metric exploit."""
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE
    from tracking_cellmot.metrics import evaluate

    er = evaluate(
        _off_volume_hub_exploit_graph(),
        _fork_graph(),
        scale=DEFAULT_SCALE,
        max_distance=MAX_DISTANCE,
    )
    assert er.division_tp == 0
    assert er.division_fn == 1


@pytest.mark.skipif(not (TRAIN / f"{DIV_CROP}.geff").exists(), reason="needs local train geff")
def test_gt_as_pred_scores_perfect_divisions():
    from biotrack.metric import load_graph, score_pred_graph

    gt_path = str(TRAIN / f"{DIV_CROP}.geff")
    row = score_pred_graph(load_graph(gt_path), gt_path)
    assert row["division_fp"] == 0 and row["division_fn"] == 0
    assert row["division_tp"] >= 1  # this crop has annotated GT divisions


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
