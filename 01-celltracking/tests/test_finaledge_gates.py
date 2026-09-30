"""Software contracts for PKT-0042's gate driver. Every one is exercised by PLANTING the
violation, because a fail-closed rule no test can reach is a claim rather than a guard.

Scientific promotion is not decided here (CLAUDE.md rule 4). These check only that the
instrument refuses what it says it refuses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from scripts.win_bet.finaledge_gates import (  # noqa: E402
    BONUS_GRID,
    GateRefusal,
    declare_widened_surface,
    select_bonus,
)


def _declared(pairs):
    return pl.DataFrame({"source": [p[0] for p in pairs], "target": [p[1] for p in pairs]},
                        schema={"source": pl.Int64, "target": pl.Int64})


def _scores(pairs, values):
    return pl.DataFrame(
        {"crop": ["c"] * len(pairs), "source": [p[0] for p in pairs],
         "target": [p[1] for p in pairs], "prob": [0.9] * len(pairs), "score": list(values)},
        schema={"crop": pl.Utf8, "source": pl.Int64, "target": pl.Int64,
                "prob": pl.Float64, "score": pl.Float64})


PAIRS = [(1, 10), (2, 10), (3, 11), (4, 11)]
NODES = {1, 2, 3, 4, 10, 11}


def test_full_cover_is_accepted_and_reports_its_counts():
    frame, cov = declare_widened_surface(
        _declared(PAIRS), _scores(PAIRS, [0.9, 0.2, 0.4, 0.8]), "c", 1, {(1, 10)}, NODES)
    assert cov["missing_pairs"] == 0
    assert cov["declared_pairs_expected"] == 4
    assert cov["scored_pairs"] == 4
    assert cov["coverage_of_declared_surface"] == 1.0
    # rank 1 keeps the learned argmax per target: (1,10) at 0.9 and (4,11) at 0.8
    assert set(zip(frame["source_id"], frame["target_id"])) == {(1, 10), (4, 11)}
    assert frame["edge_prob"].to_list() == [0.9, 0.8]


def test_a_declared_pair_with_no_learned_score_is_an_instrument_failure():
    """An uncovered pair would inherit 0.0 at built-notebook 1832 - the WORST score, not a
    neutral one - so Gate B would fail on coverage and be misread as the mechanism failing."""
    with pytest.raises(GateRefusal, match="INSTRUMENT FAILURE"):
        declare_widened_surface(_declared(PAIRS), _scores(PAIRS[:3], [0.9, 0.2, 0.4]),
                                "c", 1, {(1, 10)}, NODES)


def test_the_declaration_is_not_derived_from_the_cover():
    """The regression this guards: deriving `declared` from the score table would make `missing`
    zero by construction and the whole contract vacuous."""
    with pytest.raises(GateRefusal, match="missing|INSTRUMENT FAILURE"):
        declare_widened_surface(_declared(PAIRS + [(5, 12)]),
                                _scores(PAIRS, [0.9, 0.2, 0.4, 0.8]),
                                "c", 1, {(1, 10)}, NODES | {5, 12})


def test_duplicate_pair_keys_fail():
    with pytest.raises(GateRefusal, match="DUPLICATE"):
        declare_widened_surface(_declared(PAIRS),
                                _scores(PAIRS + [(1, 10)], [0.9, 0.2, 0.4, 0.8, 0.5]),
                                "c", 1, {(1, 10)}, NODES)


def test_unknown_node_ids_fail():
    with pytest.raises(GateRefusal, match="absent from the pre-ILP node set"):
        declare_widened_surface(_declared(PAIRS), _scores(PAIRS, [0.9, 0.2, 0.4, 0.8]),
                                "c", 1, {(1, 10)}, NODES - {4})


def test_a_score_outside_0_1_fails_rather_than_being_silently_sigmoided():
    with pytest.raises(GateRefusal, match="SILENTLY SIGMOID"):
        declare_widened_surface(_declared(PAIRS), _scores(PAIRS, [0.9, 0.2, 0.4, 3.7]),
                                "c", 1, {(1, 10)}, NODES)


# ---------------------------------------------------------------------------------------------
# Plateau selection
# ---------------------------------------------------------------------------------------------

def _panel(win_by_bonus: dict[float, bool]) -> dict:
    def block(win: bool) -> dict:
        d = 0.001 if win else -0.001
        return {
            "report": {
                "channels": {
                    "edge_jaccard_raw": {"delta": d},
                    "count_adjustment": {"delta_adj": d, "count_adjustment_artifact": False},
                    "score": {"delta": d},
                    "final_graph_edges": {"delta_tp": 5 if win else -5},
                    "division_counts": {"delta_tp": 0},
                    "node_recall": {"delta": 0.0},
                },
                "paired_bootstrap": {"ci95": [d, d], "favourable": win},
            },
            "assignment_churn": {"changed_fraction": 0.02, "changed_parent": 100},
        }
    arms = {}
    for b in BONUS_GRID:
        arms[f"gateA_b{b:g}"] = {f"fold{f}": block(win_by_bonus.get(b, False)) for f in (0, 1)}
    return {"arms": arms, "complete": {0: True, 1: True}, "n_crops": {0: 71, 1: 128}}


def test_an_unfinished_fold_is_not_a_fold_that_lost():
    """The most dangerous string this module can emit is a premature "NEGATIVE": the lever's
    amendment exists precisely because a negative Gate A must not be over-read."""
    panel = _panel({2.0: True, 4.0: True, 8.0: True})
    panel["complete"] = {0: True, 1: False}
    sel = select_bonus(panel)
    assert sel["selected_bonus"] is None
    assert sel["provisional_selection_on_incomplete_data"] == 2.0
    assert "INCOMPLETE" in sel["gate_a_verdict"]
    assert "NEGATIVE" not in sel["gate_a_verdict"]


def test_a_fold_with_no_measured_arm_blocks_the_verdict():
    panel = _panel({2.0: True, 4.0: True})
    for b in BONUS_GRID:
        panel["arms"][f"gateA_b{b:g}"].pop("fold1")
    sel = select_bonus(panel)
    assert sel["selected_bonus"] is None
    assert "INCOMPLETE" in sel["gate_a_verdict"]


def test_an_isolated_winning_grid_point_is_refused_as_a_selection():
    sel = select_bonus(_panel({2.0: True}))
    assert sel["selected_bonus"] is None
    assert sel["isolated_wins_refused"] == [2.0]
    assert "ISOLATED" in sel["gate_a_verdict"]


def test_the_smallest_value_on_the_plateau_is_selected_not_the_best_point():
    sel = select_bonus(_panel({2.0: True, 4.0: True, 8.0: True}))
    assert sel["selected_bonus"] == 2.0


def test_a_win_on_one_fold_only_is_not_a_win():
    panel = _panel({2.0: True, 4.0: True})
    for b in (2.0, 4.0):
        ch = panel["arms"][f"gateA_b{b:g}"]["fold0"]["report"]["channels"]
        ch["edge_jaccard_raw"]["delta"] = -0.001
    sel = select_bonus(panel)
    assert sel["selected_bonus"] is None
    assert "NEGATIVE" in sel["gate_a_verdict"]


def test_an_adjusted_gain_with_flat_raw_association_does_not_win():
    """The shape that killed LEVER-0036 (FACT-0375) and, mirrored, LEVER-0037 (FACT-0376)."""
    panel = _panel({2.0: True, 4.0: True})
    for b in (2.0, 4.0):
        for f in (0, 1):
            ch = panel["arms"][f"gateA_b{b:g}"][f"fold{f}"]["report"]["channels"]
            ch["edge_jaccard_raw"]["delta"] = 0.0
            ch["count_adjustment"]["delta_adj"] = 0.01
    sel = select_bonus(panel)
    assert sel["selected_bonus"] is None


# ---------------------------------------------------------------------------------------------
# Control parity: exact everywhere, with FACT-0363's tolerance on raw_edges ALONE
# ---------------------------------------------------------------------------------------------

from scripts.win_bet.finaledge_gates import (  # noqa: E402
    COMPARE_KEYS,
    RAW_EDGE_JITTER_TOLERANCE,
    _control_parity,
)

BASE = {k: 100 for k in COMPARE_KEYS}


def _cfg(tmp_path):
    import csv
    q = tmp_path / "run_stats.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset"] + COMPARE_KEYS)
        w.writerow(["c"] + [BASE[k] for k in COMPARE_KEYS])
    return {"run_stats": q}


def test_an_exact_control_reproduces_and_uses_no_tolerance(tmp_path):
    _cmp, v = _control_parity(_cfg(tmp_path), "c", dict(BASE))
    assert v["exact_on_all_fifteen"] and not v["failing_columns"]
    assert v["raw_edges_tolerance_used"] is False


def test_raw_edges_jitter_inside_fact_0363_reproduces_but_is_recorded(tmp_path):
    stats = dict(BASE, raw_edges=BASE["raw_edges"] + RAW_EDGE_JITTER_TOLERANCE)
    _cmp, v = _control_parity(_cfg(tmp_path), "c", stats)
    assert not v["failing_columns"]                    # reproduces
    assert v["exact_on_all_fifteen"] is False          # but is NOT called exact
    assert v["raw_edges_tolerance_used"] is True
    assert v["raw_edges_delta"] == RAW_EDGE_JITTER_TOLERANCE


def test_raw_edges_jitter_beyond_the_measured_envelope_refuses(tmp_path):
    stats = dict(BASE, raw_edges=BASE["raw_edges"] + RAW_EDGE_JITTER_TOLERANCE + 1)
    _cmp, v = _control_parity(_cfg(tmp_path), "c", stats)
    assert v["failing_columns"] == ["raw_edges"]


def test_the_node_gate_is_never_relaxed(tmp_path):
    """FACT-0363: the ILP's NODE solution is deterministic. A node mismatch is a defect."""
    _cmp, v = _control_parity(_cfg(tmp_path), "c", dict(BASE, raw_nodes=BASE["raw_nodes"] + 1))
    assert "raw_nodes" in v["failing_columns"]


def test_jitter_reaching_the_FINAL_graph_refuses(tmp_path):
    """The emitted graph is the object being scored, so it is exact or nothing."""
    for col in ("nodes", "edges"):
        _cmp, v = _control_parity(_cfg(tmp_path), "c", dict(BASE, **{col: BASE[col] + 1}))
        assert col in v["failing_columns"], col
