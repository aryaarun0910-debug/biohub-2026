"""The division-verifier substrate must hold its own leakage contract mechanically.

These are SOFTWARE contract tests (CLAUDE.md rule 4). They assert that the leakage guard fires,
that no identifier can enter the feature space, that the feature function cannot encode daughter
order or how a row was constructed, and that the metric adapter's accounting arithmetic is the
one written in its docstring. They decide nothing scientific - the adapter's scientific proof is
reproducing the deployed fold-0 division counts, which needs the real export and lives in
``scripts/win_bet/divverify_adapter.py prove``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import divverify_dataset as dv  # noqa: E402


def _rows(fold, n, label=0, source="pipeline_fork", embryo=None):
    embryo = embryo or dv.FOLD_EMBRYO.get(fold, "ZSNS003")
    return pl.DataFrame({
        "fold": [fold] * n,
        "embryo": [embryo] * n,
        "crop": [f"{embryo}_c{i}" for i in range(n)],
        "source": [source] * n,
        "label": [label] * n,
        **{f: [0.0] * n for f in dv.FEATURES},
    })


# ------------------------------------------------------------------ the leakage contract
def test_features_contain_no_identifier():
    """Guard (2) of the leakage contract: the closed feature list may not carry fold identity."""
    assert not (set(dv.FEATURES) & set(dv.IDENTIFIER_COLUMNS))


def test_training_rows_exclude_the_judged_fold():
    table = pl.concat([_rows(0, 3), _rows(1, 4)], how="vertical")
    for judged in (0, 1):
        train = dv.training_rows_for(table, judged)
        assert judged not in set(train["fold"].to_list())
        dv.assert_no_leak(train, judged)


def test_assert_no_leak_raises_on_a_judged_fold_row():
    """A written rule is not a lock. The lock must actually fire."""
    table = pl.concat([_rows(1, 4), _rows(0, 1)], how="vertical")
    with pytest.raises(RuntimeError, match="LEAK"):
        dv.assert_no_leak(table, 0)


def test_assert_no_leak_raises_on_the_held_out_embryo_even_if_the_fold_column_lies():
    """Fold is a label; the embryo is the physical fact. Both are checked."""
    mislabelled = _rows(1, 3, embryo=dv.FOLD_EMBRYO[0])
    with pytest.raises(RuntimeError, match="held-out embryo"):
        dv.assert_no_leak(mislabelled, 0)


def test_counterfactuals_carry_the_fold_of_the_crop_they_were_built_from():
    """A counterfactual built from a fold-0 crop IS a fold-0 row - falsifier (b) turns on this."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0))
    assert rows, "the counterfactual constructor produced nothing - a silent no-op"
    assert {r["fold"] for r in rows} == {0}
    table = pl.DataFrame(rows).select(["fold", "embryo", "crop", "label"])
    with pytest.raises(RuntimeError, match="LEAK"):
        dv.assert_no_leak(table, 0)


# ------------------------------------------------------------------ the feature function
def _toy_context():
    """A tiny crop: one mother at t=0 with two daughters at t=1, plus filler."""
    ids = np.array([1, 2, 3, 10, 11, 12, 13, 20, 21], dtype=np.int64)
    t = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    pos = np.array([
        [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0],
        [0.0, -4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 24.0, 0.0], [0.0, 0.0, 24.0],
        [0.0, -6.0, 0.0], [0.0, 6.0, 0.0],
    ], dtype=np.float64)
    edges = [(1, 10), (1, 11), (2, 12), (3, 13), (10, 20), (11, 21)]
    return dv.CropContext(ids, t, pos, edges)


def test_triple_features_is_symmetric_in_daughter_order():
    """d1/d2 order must be inert, or the label could be read off the ordering convention."""
    ctx = _toy_context()
    a = dv.triple_features(ctx, 1, 10, 11)
    b = dv.triple_features(ctx, 1, 11, 10)
    for f in dv.FEATURES:
        assert a[f] == pytest.approx(b[f]), f"{f} depends on daughter order"


def test_triple_features_rejects_a_degenerate_triple():
    ctx = _toy_context()
    assert dv.triple_features(ctx, 1, 10, 10) is None
    assert dv.triple_features(ctx, 1, 1, 11) is None
    assert dv.triple_features(ctx, 99, 10, 11) is None


def test_triple_features_does_not_depend_on_how_the_row_was_produced():
    """The same triple must produce the same vector whether it arrived as a pipeline fork or as a
    constructed counterfactual - otherwise 'is this a real physics or a generator artifact' is
    unanswerable by construction."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=4)
    assert rows
    for r in rows:
        direct = dv.triple_features(ctx, r["mother"], r["d1"], r["d2"])
        for f in dv.FEATURES:
            assert r[f] == pytest.approx(direct[f]), f"{f} encodes the construction route"


def test_temporal_counterfactuals_are_actually_temporally_incoherent():
    """The construction must produce what it claims: a skip fork spans two frames, a same-frame
    fork spans none. A mislabelled negative teaches the wrong physics."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=3)
    skip = [r for r in rows if r["neg_kind"] == "cf_temporal_skip"]
    same = [r for r in rows if r["neg_kind"] == "cf_temporal_same"]
    assert skip and same, "temporal counterfactuals were not constructed"
    assert all(max(r["dt_d1"], r["dt_d2"]) == 2 for r in skip)
    assert all(min(r["dt_d1"], r["dt_d2"]) == 0 for r in same)


def test_wrong_parent_counterfactual_keeps_both_daughters():
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=2)
    wp = [r for r in rows if r["neg_kind"] == "cf_wrong_parent"]
    assert wp
    for r in wp:
        assert {r["d1"], r["d2"]} == {10, 11}
        assert r["mother"] != 1


def test_wrong_sister_counterfactual_keeps_the_mother_and_one_daughter():
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=2)
    ws = [r for r in rows if r["neg_kind"] == "cf_wrong_sister"]
    assert ws
    for r in ws:
        assert r["mother"] == 1
        assert len({r["d1"], r["d2"]} & {10, 11}) == 1


# ------------------------------------------------------------------ the external contract
def test_external_shared_space_drops_the_constant_columns():
    """dt_* and out_degree are constants in the external table; keeping them would hand a mixed
    model a perfect in-domain/external discriminator."""
    for f in ("dt_d1", "dt_d2", "out_degree"):
        assert f not in dv.EXTERNAL_SHARED
    assert set(dv.EXTERNAL_SHARED) < set(dv.FEATURES)


# ------------------------------------------------------------------ the metric adapter
def test_adapter_accounting_arithmetic():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.base = {"division_tp": 5, "division_fp": 67, "division_fn": 21}
    ad.tp_forks = {1, 2}
    ad.fp_forks = {10, 11, 12}
    ad.ignored_forks = {90, 91}

    assert ad.predict_counts(set()) == {
        "division_tp": 5, "division_fp": 67, "division_fn": 21,
        "rejections_that_cut_a_charged_fp": 0,
        "rejections_that_cost_a_true_positive": 0,
        "rejections_the_metric_ignores": 0,
    }
    # rejecting a fork the metric ignores must move nothing - the point the adapter exists to make
    inert = ad.predict_counts({90, 91})
    assert (inert["division_tp"], inert["division_fp"], inert["division_fn"]) == (5, 67, 21)
    assert inert["rejections_the_metric_ignores"] == 2
    # rejecting an FP cuts FP; rejecting a TP moves it to FN
    cut = ad.predict_counts({10, 11})
    assert (cut["division_tp"], cut["division_fp"], cut["division_fn"]) == (5, 65, 21)
    cost = ad.predict_counts({1})
    assert (cost["division_tp"], cost["division_fp"], cost["division_fn"]) == (4, 67, 22)


def test_adapter_expectation_matches_the_registry_scope():
    """The one number written outside the registry is a TEST EXPECTATION, so a registry change
    fails loudly here instead of passing on a stale constant (FACT-0375 scope, control arm)."""
    import divverify_adapter as da

    assert da.DEPLOYED_F0_DIVISION == {"division_tp": 5, "division_fp": 67, "division_fn": 21}


def test_adapter_weakest_child_is_deterministic_and_prefers_a_dead_branch():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.name = "toy"
    ad.pos = {1: np.zeros(3), 10: np.array([0.0, 1.0, 0.0]), 11: np.array([0.0, 9.0, 0.0])}
    ad.children_of = {1: [10, 11], 10: [99]}
    # 11 does not persist, so it goes first even though 10 is nearer
    assert ad.weakest_child(1) == 11
    ad.children_of = {1: [10, 11], 10: [99], 11: [98]}
    # both persist: the LARGER displacement is dropped
    assert ad.weakest_child(1) == 11
    assert ad.weakest_child(1) == 11  # deterministic


def test_adapter_refuses_a_node_that_is_not_a_fork():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.name = "toy"
    ad.pos = {1: np.zeros(3), 10: np.ones(3)}
    ad.children_of = {1: [10]}
    with pytest.raises(RuntimeError, match="not a fork"):
        ad.weakest_child(1)
