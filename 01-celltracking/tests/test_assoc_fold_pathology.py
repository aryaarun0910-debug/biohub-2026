"""Contracts of the PKT-0038 fold-transfer pathology instruments.

SOFTWARE contracts only - no promotion decision lives here (CLAUDE.md rule 4). Each test plants
the violation it guards against, so a test that passes without the guard is not written.

Two instruments are covered:
  ``scripts/win_bet/assoc_fold_pathology.py``   the 2x2, the strata and the ablations
  ``scripts/win_bet/assoc_lost_edge_ledger.py`` the unified lost-edge ledger

The ledger's contracts are the ones that matter most, because the failure mode the coordinator
named - one category silently absorbing another's mass - is invisible in a payload and only a
planted test can catch it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import assoc_fold_pathology as P  # noqa: E402
import assoc_lost_edge_ledger as L  # noqa: E402
import assoc_tournament as T  # noqa: E402
from assoc_train_harness import HarnessRefusal  # noqa: E402


# =============================================================================================
# fixtures - a six-edge scenario, one edge per ledger outcome
# =============================================================================================

class _FakeGraph:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        self._n, self._e = nodes, edges

    def node_attrs(self):
        return _Frame(self._n)

    def edge_attrs(self):
        return _Frame(self._e)


class _Frame:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


# Every node sits on its own y lane, 100 voxels (40.6 um) apart, so the 7 um one-to-one matcher
# can never cross lanes and the scenario means exactly what it says.
def _lane(i: int) -> float:
    return 100.0 * i


SCENARIO = [
    # (name, expected category, parent lane, child lane, in_peaks, in_final, offered,
    #  child_gets_other_parent)
    ("recovered", "recovered", 0, 1, True, True, True, False),
    ("cat1", "cat1_endpoint_absent_from_peaks", 2, 3, False, False, False, False),
    ("cat2", "cat2_endpoint_removed_by_node_selection", 4, 5, True, False, True, False),
    ("cat3", "cat3_true_edge_never_offered", 6, 7, True, True, False, False),
    ("cat4", "cat4_wrong_parent_selected", 8, 9, True, True, True, True),
    ("cat5", "cat5_selected_then_overwritten_provisional", 10, 11, True, True, True, False),
]


def _build(scenario=SCENARIO):
    """GT graph, pre-ILP frame and final frame realising the scenario exactly."""
    gt_nodes, gt_edges = [], []
    pre_nodes, pre_edges = [], []
    fin_nodes, fin_edges = [], []
    nid = 0
    for name, _cat, plane, clane, in_peaks, in_final, offered, other_parent in scenario:
        p, c = nid, nid + 1
        nid += 2
        gt_nodes += [{"node_id": p, "t": 0, "z": 0.0, "y": _lane(plane), "x": 0.0},
                     {"node_id": c, "t": 1, "z": 0.0, "y": _lane(clane), "x": 0.0}]
        gt_edges.append({"source_id": p, "target_id": c})
        if not in_peaks:
            continue
        pre_nodes += [{"row_type": "node", "node_id": p, "t": 0, "z": 0.0,
                       "y": _lane(plane), "x": 0.0, "source_id": -1, "target_id": -1},
                      {"row_type": "node", "node_id": c, "t": 1, "z": 0.0,
                       "y": _lane(clane), "x": 0.0, "source_id": -1, "target_id": -1}]
        if offered:
            pre_edges.append({"row_type": "edge", "node_id": -1, "t": -1, "z": 0.0,
                              "y": 0.0, "x": 0.0, "source_id": p, "target_id": c})
        if not in_final:
            continue
        fin_nodes += [{"row_type": "node", "node_id": p, "t": 0, "z": 0.0,
                       "y": _lane(plane), "x": 0.0, "source_id": -1, "target_id": -1},
                      {"row_type": "node", "node_id": c, "t": 1, "z": 0.0,
                       "y": _lane(clane), "x": 0.0, "source_id": -1, "target_id": -1}]
        if other_parent:
            # a DECOY parent in the child's own lane, so the child has a parent that is not the
            # true one - the only observable that separates category 4 from category 5
            decoy = 10_000 + p
            fin_nodes.append({"row_type": "node", "node_id": decoy, "t": 0, "z": 0.0,
                              "y": _lane(clane), "x": 0.0, "source_id": -1, "target_id": -1})
            fin_edges.append({"row_type": "edge", "node_id": -1, "t": -1, "z": 0.0, "y": 0.0,
                              "x": 0.0, "source_id": decoy, "target_id": c})
        elif name == "recovered":
            fin_edges.append({"row_type": "edge", "node_id": -1, "t": -1, "z": 0.0, "y": 0.0,
                              "x": 0.0, "source_id": p, "target_id": c})
    graph = _FakeGraph(pd.DataFrame(gt_nodes), pd.DataFrame(gt_edges))
    cols = ["row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    pre = pl.DataFrame(pre_nodes + pre_edges).select(cols)
    fin = pl.DataFrame(fin_nodes + fin_edges).select(cols)
    return graph, pre, fin


@pytest.fixture()
def scenario(monkeypatch):
    graph, pre, fin = _build()
    import biotrack.metric as metric
    monkeypatch.setattr(metric, "load_graph", lambda _p: graph)
    return pre, fin


# =============================================================================================
# 1. the ledger's categories are exhaustive, exclusive, and in the declared order
# =============================================================================================

def test_every_category_is_reached_exactly_once(scenario):
    pre, fin = scenario
    got = L.crop_ledger("fixture", Path("unused.geff"), pre, fin)
    assert got["gt_edges"] == len(SCENARIO)
    for _name, cat, *_ in SCENARIO:
        assert got[cat] == 1, f"{cat} was not reached exactly once: {got}"


def test_exhaustiveness_is_asserted_not_reported():
    """Plant the violation the guard exists for: a category that quietly stops counting."""
    honest = {k: 1 for k in L.CATEGORIES}
    honest["recovered"] = 4
    L.assert_exhaustive(honest, 10, "fixture")            # sums, so it must pass
    leaky = dict(honest)
    leaky["cat4_wrong_parent_selected"] = 0                # one edge silently unaccounted for
    with pytest.raises(L.LedgerRefusal, match="does not account for every GT edge"):
        L.assert_exhaustive(leaky, 10, "fixture")


def test_the_pooled_total_is_guarded_by_the_same_contract_as_a_crop():
    """A per-crop guard that the pooled total does not share is a guard with a hole in it."""
    src = (ROOT / "scripts" / "win_bet" / "assoc_lost_edge_ledger.py").read_text(encoding="utf-8")
    assert src.count("assert_exhaustive(") >= 3, (
        "the exhaustiveness contract must be applied per crop AND to the pooled total"
    )


def test_category_five_mass_is_not_folded_into_category_four(scenario):
    """The coordinator's named failure mode: a category absorbing another's mass."""
    pre, fin = scenario
    got = L.crop_ledger("fixture", Path("unused.geff"), pre, fin)
    assert got["cat4_wrong_parent_selected"] == 1
    assert got["cat5_selected_then_overwritten_provisional"] == 1
    assert got["cat4_wrong_parent_selected"] != 2, (
        "category 4 has swallowed category 5 - the child with NO parent must not be reported as "
        "a wrong-parent selection"
    )


def test_first_match_wins_ordering_is_literal():
    """An edge that qualifies for several categories is counted ONCE, by the earliest.

    The planted edge has both endpoints absent from the peaks AND no candidate AND no presence in
    the final graph - categories 1, 2, 3 and 5 all describe it, and the declared order says 1. A
    second, ordinary recovered edge is included only so the frames are non-degenerate.
    """
    graph, pre, fin = _build([
        ("recovered", "recovered", 0, 1, True, True, True, False),
        ("multi", "cat1_endpoint_absent_from_peaks", 2, 3, False, False, False, False),
    ])
    import biotrack.metric as metric
    real = metric.load_graph
    metric.load_graph = lambda _p: graph
    try:
        got = L.crop_ledger("fixture", Path("unused.geff"), pre, fin)
    finally:
        metric.load_graph = real
    assert got["cat1_endpoint_absent_from_peaks"] == 1
    assert got["cat2_endpoint_removed_by_node_selection"] == 0
    assert got["cat3_true_edge_never_offered"] == 0
    assert got["cat5_selected_then_overwritten_provisional"] == 0
    assert got["recovered"] == 1


def test_category_six_is_zero_by_ordering_and_divisions_have_their_own_denominator(scenario):
    pre, fin = scenario
    got = L.crop_ledger("fixture", Path("unused.geff"), pre, fin)
    assert got["cat6_division_topology_failure"] == 0
    assert "divisions" in got and "gt_division_events" in got["divisions"]


def test_the_fact_0370_reconciliation_split_is_computed_from_the_same_matching(scenario):
    pre, fin = scenario
    got = L.crop_ledger("fixture", Path("unused.geff"), pre, fin)
    split = got["fact_0370_split"]
    assert sum(split.values()) == got["gt_edges"]
    # node selection is IGNORED by that split, so the cat2 edge is counted reachable there
    assert split["undetected_endpoint"] == 1
    assert split["unreachable_no_candidate"] == 1
    assert split["reachable"] == 4


# =============================================================================================
# 2. the pathology module's declared classes and its refusals
# =============================================================================================

def test_the_ablation_classes_are_what_they_claim_to_be():
    assert P.ABLATIONS["raw_prob_only"] == ["prob"]
    for name, feats in P.ABLATIONS.items():
        if name == "raw_prob_only":
            continue
        assert "prob" not in feats, (
            f"ablation {name} claims to exclude the raw probability scale and does not"
        )
    assert set(T.BIDIR).issubset(set(P.ABLATIONS["prob_free_context"]))
    assert not (set(P.ABLATIONS["geometry_only"]) & set(T.BIDIR)), (
        "the geometry-only class must carry no probability-derived column"
    )


def test_the_declared_head_refuses_when_no_feature_list_is_bound():
    saved = list(P._PATHOLOGY_FEATURES)
    P._PATHOLOGY_FEATURES = []
    try:
        with pytest.raises(HarnessRefusal, match="no feature list bound"):
            P._pathology_head()
    finally:
        P._PATHOLOGY_FEATURES = saved


def test_bind_features_reaches_every_live_copy_of_this_module():
    """The FACT-0060 class of defect: the harness imports the head BY NAME, so a scripted run has
    two module objects. Binding one of two is a guaranteed silent wrong-feature fit."""
    import importlib

    P._bind_features(["prob", "dist_um"])
    other = importlib.import_module("assoc_fold_pathology")
    assert other._PATHOLOGY_FEATURES == ["prob", "dist_um"]
    head = P._pathology_head()
    assert head.feature_names == ["prob", "dist_um"]


def test_published_assertion_targets_match_the_committed_payload():
    """The instrument checks are only worth having if their targets are the payload's own."""
    p = ROOT / "_evidence" / "assoc" / "tournament" / "gate_fold0.json"
    if not p.is_file():                       # evidence is gitignored; skip rather than lie
        pytest.skip("gate_fold0.json not present in this checkout")
    d = json.loads(p.read_text(encoding="utf-8"))
    by_tag = {m["tag"]: m for m in d["models"]}
    a = next(v for k, v in by_tag.items() if k.startswith("GATE_A"))
    b = next(v for k, v in by_tag.items() if k.startswith("GATE_B"))
    assert a["surface"]["contested"]["top1"] == P.PUBLISHED_GATE["gate_a"]["contested_top1"]
    assert a["conversions"]["contested"]["gained"] == P.PUBLISHED_GATE["gate_a"]["gained"]
    assert a["conversions"]["contested"]["lost"] == P.PUBLISHED_GATE["gate_a"]["lost"]
    assert b["surface"]["contested"]["top1"] == P.PUBLISHED_GATE["gate_b"]["contested_top1"]
    assert (d["deployed_baseline"]["contested"]["top1"]
            == P.PUBLISHED_GATE["deployed_contested_top1_f0"])


def test_the_deployed_probability_scored_through_outcome_returns_the_deployed_decision():
    """The arithmetic self-check the whole module rests on: scoring by ``prob`` must reproduce the
    deployed argmax with churn EXACTLY 0. Anything else means the decision path has drifted."""
    table = T.bidirectional_columns(_small_surface())
    surf = {
        "dec": None, "baseline_map": None, "contested_keys": None, "bar": None,
    }
    from assoc_parent_dataset import evaluate

    base = evaluate(table, "prob")
    baseline_map = base.pop("per_target_correct")
    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    surf = {"dec": dec, "baseline_map": baseline_map,
            "contested_keys": {(r[0], int(r[1])) for r in
                               counts.filter(pl.col("n_dec") > 1)
                               .select(["crop", "target"]).rows()},
            "bar": base["contested"]["top1"]}
    row = P._outcome(surf, dec["prob"].to_numpy().astype(np.float64), "selfcheck")
    assert row["churn"] == 0, "scoring by the deployed probability moved a within-target argmax"
    assert row["delta_vs_deployed"] == 0.0


def test_target_frame_marks_division_involvement_on_the_deployed_decision():
    """``division_involved`` must be the FACT-0371 failure mode - the winning source already wins
    another target - and not merely 'the source has out-degree > 1 somewhere in the table'."""
    rows = []
    for t, (p_hi, src_hi) in enumerate([(0.9, 77), (0.8, 77)]):
        rows += [
            {"crop": "c", "target": t, "source": src_hi, "prob": p_hi, "rank": 1,
             "margin_to_best": 0.0, "n_candidates": 2, "dist_um": 1.0, "dz_um": 0.0,
             "dy_um": 0.0, "dx_um": 0.0, "src_out_degree": 2, "is_true_parent": 1,
             "target_has_true_parent": 1, "true_parent_is_candidate": 1, "target_matched_gt": 1},
            {"crop": "c", "target": t, "source": 90 + t, "prob": 0.2, "rank": 2,
             "margin_to_best": p_hi - 0.2, "n_candidates": 2, "dist_um": 5.0, "dz_um": 0.0,
             "dy_um": 0.0, "dx_um": 0.0, "src_out_degree": 1, "is_true_parent": 0,
             "target_has_true_parent": 1, "true_parent_is_candidate": 1, "target_matched_gt": 1},
        ]
    table = T.bidirectional_columns(pl.DataFrame(rows))
    dec = table
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    got = P.target_frame({"dec": dec})
    assert got["division_involved"].sum() == 2, (
        "source 77 wins BOTH targets, so both are division-involved under the deployed argmax"
    )


def _small_surface(n_crops: int = 4, per_crop: int = 25, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_crops):
        crop = f"emb_{c:03d}"
        for t in range(per_crop):
            n_cand = 1 if (t % 4 == 0) else int(rng.integers(2, 4))
            probs = np.sort(rng.uniform(0.1, 0.99, size=n_cand))[::-1]
            true_idx = 0 if (t % 5) else min(1, n_cand - 1)
            pool = rng.choice(per_crop, size=n_cand, replace=False)
            for i in range(n_cand):
                d = float(rng.uniform(0.5, 9.0)) - (3.0 if i == true_idx else 0.0)
                rows.append({
                    "crop": crop, "target": 1000 * c + t, "source": 500000 + int(pool[i]),
                    "prob": float(probs[i]), "rank": i + 1,
                    "margin_to_best": float(probs[0] - probs[i]),
                    "n_candidates": int(n_cand), "dist_um": abs(d),
                    "dz_um": d / 3, "dy_um": d / 3, "dx_um": d / 3,
                    "src_out_degree": int(rng.integers(1, 3)),
                    "is_true_parent": int(i == true_idx),
                    "target_has_true_parent": 1, "true_parent_is_candidate": 1,
                    "target_matched_gt": 1,
                })
    return pl.DataFrame(rows)
