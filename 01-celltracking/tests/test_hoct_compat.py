"""Contracts for the HOCT compatibility evaluation (PKT-0029 STEP 2).

Software contracts only. Whether the published head beats the deployed one is an experiment
result and is not decidable here.

The assertions that matter are the ones that would let a WRONG answer look like a right one:
an empty parity channel reporting a pass, a candidate graph that has quietly collapsed into our
one-parent deployed rule, and a score table joined so that some candidates carry no score.
"""
from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

from scripts.win_bet.hoct_compat import (
    _binomial_p,
    _crop_nodes,
    calibration,
    parity_summary,
)
from scripts.win_bet.hoct_head import GATE_UM, candidate_graph


def test_empty_parity_channel_fails_closed() -> None:
    """A gate that compared nothing must never report a pass."""
    summary = parity_summary(np.empty(0), np.empty(0))
    assert summary["passed"] is False
    assert summary["pairs"] == 0
    assert "reason" in summary


def test_parity_criterion_is_argmax_agreement_not_probability_equality() -> None:
    """Levels may drift; the measured quantity is within-target RANKING, so that is the bar."""
    mine = np.array([0.9, 0.1, 0.8, 0.2])
    recorded = np.array([0.5, 0.4, 0.45, 0.44])          # same ordering, very different levels
    passing = parity_summary(mine, recorded, rank_targets=100, rank_agree=99)
    assert passing["passed"] is True
    assert passing["argmax_agreement"] == pytest.approx(0.99)
    failing = parity_summary(mine, recorded, rank_targets=100, rank_agree=90)
    assert failing["passed"] is False


def test_parity_reports_the_sub_threshold_band_separately() -> None:
    """FACT-0382: every contested fold-0 error has its true parent BELOW the deployed 0.5, so a
    parity channel populated only above it validates the band the task does not use."""
    recorded = np.array([0.9, 0.2, 0.05])
    summary = parity_summary(recorded.copy(), recorded, rank_targets=10, rank_agree=10)
    assert summary["band_a_deployed_above_0p5"]["n"] == 1
    assert summary["band_b_sub_threshold"]["n"] == 2


def test_candidate_graph_is_uncapped_unlike_our_deployed_rule() -> None:
    """FACT-0392 is the whole reason this experiment rebuilds the graph. If the rebuilt surface
    admitted at most one parent per target it would BE our deployed rule and the head would be
    evaluated on competitors it was never trained to discriminate."""
    source = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 6.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
    edge_source, edge_target = candidate_graph(source, target, GATE_UM)
    assert len(edge_source) == 3
    assert int(np.bincount(edge_target).max()) == 3   # three parents offered for one target


def test_crop_nodes_rejects_non_positional_ids() -> None:
    """Every index in this module assumes the export's ids are positions in ``coords_so_far``."""
    frame = pl.DataFrame({
        "row_type": ["node", "node"], "node_id": [7, 9],
        "t": [0, 0], "z": [0.0, 0.0], "y": [0.0, 4.0], "x": [0.0, 4.0],
    })
    with pytest.raises(RuntimeError, match="contiguous"):
        _crop_nodes(frame)


def test_binomial_p_matches_a_hand_computable_case() -> None:
    assert _binomial_p(0, 0) is None
    assert _binomial_p(0, 5) == pytest.approx(2 * (1 / 32))
    assert _binomial_p(2, 4) == pytest.approx(1.0)      # a tie cannot be significant


def test_calibration_reports_the_base_rate_it_is_measured_against() -> None:
    """An ECE without its base rate is unreadable: HOCT's abstain mass makes its LEVEL
    incomparable with ours (FACT-0392), so only the shape against the base rate is informative."""
    table = pl.DataFrame({
        "hoct_prob": [0.05, 0.05, 0.95, 0.95],
        "is_true_parent": [0, 0, 1, 1],
    })
    result = calibration(table, "hoct_prob", bins=10)
    assert result["base_rate"] == pytest.approx(0.5)
    assert result["ece"] == pytest.approx(0.05, abs=1e-6)


ROOT = pathlib.Path(__file__).resolve().parents[1]
PREILP = pathlib.Path("C:/temp/p30_f0/preilp_split0.parquet")
GEFF = ROOT / "data" / "train" / "44b6_0113de3b.geff"


@pytest.mark.skipif(not (PREILP.exists() and GEFF.exists()),
                    reason="needs the fold-0 pre-ILP export and its GT")
def test_label_contract_reproduces_the_shared_surface_row_for_row() -> None:
    """THE assertion that makes the fast path legitimate.

    ``hoct_compat.label_contract`` exists only because ``assoc_parent_dataset.build_crop``
    decides ``true_parent_is_candidate`` with a per-row scan over every candidate edge, which
    does not terminate on HOCT's uncapped 15 um graph. It must be the SAME contract, not a
    similar one, so it is checked against the shared surface on real data.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
    import assoc_parent_dataset as APD
    from scripts.win_bet.hoct_compat import label_contract

    crop = "44b6_0113de3b"
    pre = pl.read_parquet(PREILP).filter(pl.col("dataset") == crop)
    nodes = pre.filter(pl.col("row_type") == "node")
    edges = pre.filter(pl.col("row_type") == "edge").head(3000)
    reference = pl.DataFrame(APD.build_crop(
        crop, GEFF, pl.concat([nodes, edges]), None, 0.0, 10 ** 9,
    ))
    mine = label_contract(
        crop, GEFF, nodes,
        edges["source_id"].to_numpy().astype(np.int64),
        edges["target_id"].to_numpy().astype(np.int64),
        edges["edge_prob"].to_numpy().astype(np.float64),
    )
    assert mine.height == reference.height > 0
    for column in reference.columns:
        left, right = mine[column].to_numpy(), reference[column].to_numpy()
        if left.dtype.kind == "f":
            assert np.allclose(left, right), column
        else:
            assert (left == right).all(), column


def test_deployed_contested_stratum_needs_two_candidates(tmp_path) -> None:
    """The stratum is defined by OUR surface being contested, so a single-candidate target on
    that surface must never enter it - otherwise the stratified number stops being comparable
    to FACT-0381 and FACT-0386, which is the only reason it is reported."""
    from scripts.win_bet.hoct_compat import deployed_contested_targets

    np.savez(tmp_path / "cropA.npz",
             source_id=np.array([0, 1, 2]), target_id=np.array([9, 9, 8]),
             edge_prob=np.array([0.9, 0.4, 0.7]))
    hard = deployed_contested_targets(["cropA"], tmp_path, pl.DataFrame())
    assert hard == {("cropA", 9)}          # target 8 has one candidate, target 9 has two
    below_floor = deployed_contested_targets(["cropA"], tmp_path, pl.DataFrame(), floor=0.5)
    assert below_floor == set()            # only one of target 9's candidates clears 0.5
