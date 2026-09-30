"""Contracts of the PKT-0038 association tournament driver.

These are SOFTWARE contracts only - no promotion decision lives here (CLAUDE.md rule 4). Each test
plants the violation it guards against, so a test that passes without the guard is not written.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import assoc_tournament as T  # noqa: E402
from assoc_train_harness import HarnessRefusal, ModelSpec, make_estimator  # noqa: E402


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------

def _surface(n_crops: int = 10, per_crop: int = 40, seed: int = 7) -> pl.DataFrame:
    """A synthetic frozen-surface table with the columns assoc_parent_dataset emits.

    The true parent is made LEARNABLE but not trivially so: it usually carries the higher
    probability, and in a fixed minority of contested targets it does not, which is the population
    a ranker has to win.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_crops):
        crop = f"emb_{c:04d}"
        for t in range(per_crop):
            n_cand = 1 if (t % 3 == 0) else rng.integers(2, 4)
            probs = np.sort(rng.uniform(0.1, 0.99, size=n_cand))[::-1]
            true_idx = 0 if (t % 7) else min(1, n_cand - 1)
            # Sources are drawn from a SHARED per-crop pool. A pool of unique-per-target sources
            # would make every forward best trivially a mutual best and would make the
            # division-producing category unreachable, i.e. the fixture would hide the two things
            # the bidirectional arm exists to measure.
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


@pytest.fixture()
def surface() -> pl.DataFrame:
    return T.bidirectional_columns(_surface())


# ---------------------------------------------------------------------------------------------
# 1. the derived columns are label-free, deterministic, and never overwrite the surface
# ---------------------------------------------------------------------------------------------

def test_bidirectional_columns_do_not_use_the_label():
    base = _surface()
    scrambled = base.with_columns(pl.col("is_true_parent").shuffle(seed=99))
    a = T.bidirectional_columns(base).select(T.BIDIR + T.META)
    b = T.bidirectional_columns(scrambled).select(T.BIDIR + T.META)
    assert a.equals(b), "a derived column moved when only the LABEL changed - that is leakage"


def test_bidirectional_columns_are_deterministic(surface):
    again = T.bidirectional_columns(_surface())
    assert surface.select(T.BIDIR).equals(again.select(T.BIDIR))


def test_bidirectional_columns_refuse_to_overwrite(surface):
    with pytest.raises(HarnessRefusal, match="refusing to overwrite"):
        T.bidirectional_columns(surface)


def test_bidirectional_columns_refuse_a_surface_missing_a_key_column():
    with pytest.raises(HarnessRefusal, match="missing"):
        T.bidirectional_columns(_surface().drop("prob"))


def test_mutual_best_is_a_real_two_sided_test(surface):
    """A candidate can be its target's best and NOT its source's best. If the column were just the
    forward argmax the tournament's bidirectional arm would carry no new information at all."""
    fwd_best = surface.filter(pl.col("rank") == 1)
    assert fwd_best["is_mutual_best"].min() == 0, "mutual-best collapsed to the forward argmax"


# ---------------------------------------------------------------------------------------------
# 2. mining finds the categories it claims to
# ---------------------------------------------------------------------------------------------

def _mining_fixture():
    #                     g  src  dist  y   score
    spec = [(0, 10, 5.0, 1, 0.9),      # true parent, far
            (0, 11, 2.0, 0, 0.3),      # wrong_nearby_parent + motion_consistent
            (1, 12, 1.0, 1, 0.8),      # true parent, near
            (1, 10, 4.0, 0, 0.95),     # high_conf_substitution + division_producing (10 wins g0)
            (2, 13, 3.0, 1, 0.7),
            (2, 14, 9.0, 0, 0.1)]      # far, low - not hard by any rule
    groups = np.array([s[0] for s in spec])
    srcs = np.array([s[1] for s in spec])
    feat = np.array([[s[2]] for s in spec], dtype=float)
    y = np.array([s[3] for s in spec])
    scores = np.array([s[4] for s in spec])
    return feat, y, groups, srcs, scores, ["dist_um"]


def test_mining_finds_each_declared_category():
    got = T.mine_hard_negatives(*_mining_fixture())
    m = got["masks"]
    assert m["wrong_nearby_parent"][1] and not m["wrong_nearby_parent"][5]
    assert m["high_conf_substitution"][3] and not m["high_conf_substitution"][1]
    assert m["division_producing"][3], "a source that already wins another target is a fork risk"
    assert not m["division_producing"][1]
    assert got["counts"]["wrong_nearby_parent"] >= 1
    assert not got["union"][0] and not got["union"][2], "a positive was mined as a hard negative"


def test_mining_needs_the_distance_feature():
    feat, y, g, s, sc, _ = _mining_fixture()
    with pytest.raises(HarnessRefusal, match="dist_um"):
        T.mine_hard_negatives(feat, y, g, s, sc, ["not_distance"])


# ---------------------------------------------------------------------------------------------
# 3. the mined ranker strips identity and records its curriculum
# ---------------------------------------------------------------------------------------------

def test_mined_ranker_refuses_a_mis_declared_feature_width():
    r = T.MinedRanker("tree", ["a", "b"], tag="x")
    with pytest.raises(HarnessRefusal, match="does not describe this X"):
        r.fit(np.zeros((10, 9)), np.zeros(10, dtype=int))


def test_mined_ranker_records_every_round(surface):
    T.MINING_LOG.clear()
    spec = ModelSpec.from_dict(T.ARM_BY_TAG["A2m.tree.nine.mined"])
    x = surface.select(spec.features).to_numpy().astype(float)
    y = surface["is_true_parent"].to_numpy().astype(int)
    T.make_mined_tree().fit(x, y)
    assert len(T.MINING_LOG) == 1
    rounds = T.MINING_LOG[0]["rounds"]
    assert len(rounds) == T.MINING_ROUNDS
    assert rounds[-1]["terminal"] and not rounds[0]["terminal"]
    assert rounds[0]["categories"]["wrong_nearby_parent"] >= 0
    assert rounds[1]["hard_cumulative"] >= rounds[0]["hard_cumulative"], "curriculum shrank"


def test_curriculum_is_collected_across_both_module_copies():
    """The defect that produced an EMPTY `mining_curriculum` in a payload that claimed to record it.

    `assoc_train_harness.resolve_callable` imports this module BY NAME, so a run started as a
    script has the estimator factories in `sys.modules['assoc_tournament']` and `run_fold` in
    `sys.modules['__main__']` - two module objects with two independent `MINING_LOG` lists. The
    violation is planted directly: a second module object is installed under `__main__` and the
    append is made there, exactly as a scripted run does it.
    """
    import types

    T.clear_curriculum()
    shadow = types.ModuleType("__main__")
    shadow.MINING_LOG = []
    shadow.HEAD = T.HEAD                 # what marks a module as a copy of THIS one
    shadow.FrozenTransferHead = type("FrozenTransferHead", (), {"_PICKLE": None, "_SHA": None})
    prior = sys.modules.get("__main__")
    sys.modules["__main__"] = shadow
    try:
        shadow.MINING_LOG.append({"tag": "planted", "rounds": []})
        assert [e["tag"] for e in T.collected_curriculum()] == ["planted"], (
            "the curriculum written by the scripted module copy was not collected"
        )
        T.clear_curriculum()
        assert shadow.MINING_LOG == [], "clear_curriculum missed a module copy"
        # The SECOND bite of the same defect: bind_frozen_model must reach both class objects, or
        # `apply` refuses with "constructed with no frozen model bound".
        T.bind_frozen_model(Path("nowhere.pkl"), "deadbeef")
        assert shadow.FrozenTransferHead._SHA == "deadbeef", (
            "bind_frozen_model bound only one module copy's class"
        )
        assert T.FrozenTransferHead._SHA == "deadbeef"
    finally:
        if prior is not None:
            sys.modules["__main__"] = prior
        else:
            del sys.modules["__main__"]


def test_run_fold_refuses_an_empty_curriculum_when_a_mining_arm_ran(surface, monkeypatch, tmp_path):
    """A mining arm that records nothing must CRASH, not emit a payload with an empty list."""
    monkeypatch.setattr(T, "collected_curriculum", lambda: [])
    with pytest.raises(HarnessRefusal, match="curriculum is EMPTY"):
        T.run_fold(surface, 1, [T.ARM_BY_TAG["A2m.tree.nine.mined"]],
                   tmp_path / "x.json", "test")


def test_mined_ranker_never_sees_the_identity_columns(surface):
    """The meta columns are stripped, so two tables differing ONLY in identity must score alike."""
    spec = ModelSpec.from_dict(T.ARM_BY_TAG["A2m.tree.nine.mined"])
    x = surface.select(spec.features).to_numpy().astype(float)
    y = surface["is_true_parent"].to_numpy().astype(int)
    a = T.make_mined_tree().fit(x, y).predict_proba(x)[:, 1]
    x2 = x.copy()
    x2[:, 0] += 1_000_000        # group_uid
    x2[:, 1] += 7_000_000        # src_uid
    b = T.make_mined_tree().fit(x2, y).predict_proba(x2)[:, 1]
    assert np.allclose(a, b), "identity reached the model - the meta strip is not working"


def test_secondary_seed_differs_from_the_primary(surface):
    """A consensus of two identical fits is not a consensus (FACT-0378 is why arm 5 exists)."""
    spec = ModelSpec.from_dict(T.ARM_BY_TAG["A5.secondary.seed"])
    x = surface.select(spec.features).to_numpy().astype(float)
    y = surface["is_true_parent"].to_numpy().astype(int)
    p = T.make_bidir_tree().fit(x, y).predict_proba(x)[:, 1]
    s = T.make_secondary_seed().fit(x, y).predict_proba(x)[:, 1]
    assert not np.allclose(p, s), "the secondary reproduced the primary exactly"


# ---------------------------------------------------------------------------------------------
# 4. the frozen transfer head genuinely refuses to learn
# ---------------------------------------------------------------------------------------------

def _freeze_a_model(tmp_path: Path, surface: pl.DataFrame) -> tuple[Path, str]:
    import hashlib

    spec = ModelSpec.from_dict(T.ARM_BY_TAG["A4.bidir.tree"])
    x = surface.select(spec.features).to_numpy().astype(float)
    y = surface["is_true_parent"].to_numpy().astype(int)
    model = T.make_bidir_tree().fit(x, y)
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps(model))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_head_predictions_are_unchanged_by_fit(tmp_path, surface):
    p, sha = _freeze_a_model(tmp_path, surface)
    T.bind_frozen_model(p, sha)
    spec = ModelSpec.from_dict(T.ARM_BY_TAG["A4.bidir.tree"])
    x = surface.select(spec.features).to_numpy().astype(float)
    y = surface["is_true_parent"].to_numpy().astype(int)
    h = T.make_frozen_transfer()
    before = h.predict_proba(x)[:, 1].copy()
    h.fit(x, 1 - y)                      # inverted labels: a learner would move a long way
    after = h.predict_proba(x)[:, 1]
    assert np.array_equal(before, after), "the frozen head learned - transfer is not frozen"
    assert h.n_fit_calls_ == 1, "a refit went unrecorded"


def test_frozen_head_refuses_tampered_bytes(tmp_path, surface):
    p, sha = _freeze_a_model(tmp_path, surface)
    T.bind_frozen_model(p, sha)
    p.write_bytes(p.read_bytes() + b"\x00")
    with pytest.raises(HarnessRefusal, match="bytes changed"):
        T.make_frozen_transfer()


def test_frozen_head_refuses_when_nothing_is_bound():
    T.FrozenTransferHead._PICKLE = None
    with pytest.raises(HarnessRefusal, match="no frozen model bound"):
        T.make_frozen_transfer()


# ---------------------------------------------------------------------------------------------
# 5. the harness's additive `declared` class
# ---------------------------------------------------------------------------------------------

def test_declared_class_requires_a_head():
    with pytest.raises(HarnessRefusal, match="requires a `head`"):
        make_estimator("declared", None)


def test_declared_class_does_not_claim_a_cache():
    for tag, arm in T.ARM_BY_TAG.items():
        if arm["class"] == "declared":
            assert not ModelSpec.from_dict(arm).needs_cache(), tag


def test_every_declared_head_resolves():
    for arm in T.ARMS:
        if arm.get("head") and "frozen_transfer" not in arm["head"]:
            assert make_estimator("declared", arm["head"]) is not None


# ---------------------------------------------------------------------------------------------
# 6. selection discipline
# ---------------------------------------------------------------------------------------------

def _payload(rows: list[tuple[str, float, bool]]) -> dict:
    return {
        "fold": 1,
        "deployed_baseline": {"contested": {"top1": 0.70}},
        "models": [{
            "tag": tag,
            "surface": {"contested": {"top1": top1}, "abstentions": {}},
            "conversions": {"contested": {"gained": 1, "lost": 0, "net": 1, "churn": 1,
                                          "crop_paired_bootstrap": {"favourable": fav,
                                                                    "excludes_zero": fav,
                                                                    "ci95": [0.01, 0.02]}}},
            "single_candidate": {"n_regressions": 0, "n_shadow_regressions": 0},
            "verdict": {"promotable": False},
        } for tag, top1, fav in rows],
    }


def test_winner_is_the_best_favourable_arm():
    sel = T.select_winner(_payload([("a", 0.80, True), ("b", 0.90, False), ("c", 0.75, True)]))
    assert sel["winner"] == "a", "an unfavourable arm outranked a favourable one"
    assert sel["favourable_on_discovery"] is True


def test_no_favourable_arm_means_no_winner_but_still_transfers():
    sel = T.select_winner(_payload([("a", 0.80, False), ("b", 0.90, False)]))
    assert sel["winner"] == "b"
    assert sel["favourable_on_discovery"] is False, (
        "a tournament with no favourable arm must not present a winner as one"
    )


def test_freeze_refuses_the_gate_fold(tmp_path, surface):
    p = tmp_path / "d.json"
    bad = _payload([("A4.bidir.tree", 0.8, True)])
    bad["fold"] = 0
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(HarnessRefusal, match="never swapped"):
        T.freeze(p, surface, tmp_path)


def test_apply_refuses_a_record_that_is_not_a_freeze(tmp_path, surface):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"heartbeat": "SOMETHING_ELSE"}), encoding="utf-8")
    with pytest.raises(HarnessRefusal, match="not a frozen-selection record"):
        T.apply_to_gate(p, surface, tmp_path)


def test_apply_refuses_when_the_frozen_model_moved(tmp_path, surface):
    pkl, sha = _freeze_a_model(tmp_path, surface)
    rec = tmp_path / "frozen_selection.json"
    rec.write_text(json.dumps({
        "heartbeat": "ASSOC_TOURNAMENT_FROZEN", "frozen_on_fold": 1, "applies_to_fold": 0,
        "model_pickle": str(pkl), "model_pickle_sha256": sha,
        "arm": T.ARM_BY_TAG["A4.bidir.tree"], "abstain_frozen": {"kind": "none", "tau": None},
        "selection": {"favourable_on_discovery": False},
    }, default=str), encoding="utf-8")
    pkl.write_bytes(pkl.read_bytes() + b"\x00")
    with pytest.raises(HarnessRefusal, match="bytes changed"):
        T.apply_to_gate(rec, surface, tmp_path)


# ---------------------------------------------------------------------------------------------
# 7. end to end: one payload, one set of channels, nothing promotable
# ---------------------------------------------------------------------------------------------

def test_freeze_then_apply_round_trip(tmp_path, surface):
    out = tmp_path / "out"
    payload = T.run_fold(surface, 1, T.ARMS, out / "discovery_fold1.json", "test-discovery")

    channels = [set(m.keys()) for m in payload["models"]]
    assert all(c == channels[0] for c in channels), "PKT-0034 falsifier (d): channels diverged"
    assert {m["tag"] for m in payload["models"]} == {a["tag"] for a in T.ARMS}
    assert all(not m["verdict"]["promotable"] for m in payload["models"]), (
        "an arm was promotable without a full-chain measurement (FACT-0364/FACT-0376)"
    )
    assert any("full-chain" in b for m in payload["models"] for b in m["verdict"]["blockers"])
    assert payload["tournament"]["mining_curriculum"], "the curriculum was not recorded"
    assert payload["tournament"]["full_chain"]["measured"] is False

    rec = T.freeze(out / "discovery_fold1.json", surface, out)
    assert rec["frozen_on_fold"] == 1 and rec["applies_to_fold"] == 0
    assert Path(rec["model_pickle"]).is_file()

    gate = T.apply_to_gate(out / "frozen_selection.json", T.bidirectional_columns(_surface(seed=11)),
                           out)
    tags = {m["tag"] for m in gate["models"]}
    assert any(t.startswith("GATE_A.") for t in tags)
    assert any(t.startswith("GATE_B.") for t in tags)
    assert all(not m["verdict"]["promotable"] for m in gate["models"])


def test_consensus_is_marked_unpromotable(surface):
    cons = T.consensus_diagnostic(surface, "A4.bidir.tree", "A5.secondary.seed", 1)
    assert cons["promotable"] is False
    assert 0.0 <= cons["contested"]["agreement_rate"] <= 1.0
    assert cons["contested"]["oracle_of_the_two"] >= cons["contested"]["top1_where_they_agree"] - 1


def test_blocked_arm_is_declared_with_its_conditions():
    assert T.BLOCKED_ARM["class"] == "contextual"
    assert ModelSpec.from_dict(T.BLOCKED_ARM).needs_cache()
    assert len(T.BLOCKED_ARM["blocked_on"]) >= 3


def test_every_fold_wires_a_dual_trunk_pair_to_its_own_cache():
    """Host constraint (2): both plausible trunks are cached per fold, and they are never one cache.

    The failure this guards is silent and total. If two trunk roles shared a cache directory,
    licence or receipt, the pair would be ONE trunk written twice - `audit_dual_trunk` would have
    nothing to compare and a null result could still be blamed on trunk provenance, which is
    exactly the FACT-0392 risk the pair exists to remove.
    """
    for fold, roles in T.TRUNK_ROLES_BY_FOLD.items():
        assert set(T.DUAL_TRUNK_PAIR) <= set(roles), f"fold {fold} does not cache both trunks"
        specs = [T.context_spec(fold, r, "t.parquet", "p.parquet", "ecb") for r in roles]
        for key in ("dir", "licence", "receipt"):
            paths = [sp["cache"][key] for sp in specs]
            assert len(set(paths)) == len(paths), f"fold {fold} shares cache {key} across trunks"
        assert len({sp["out_dir"] for sp in specs}) == len(specs)
        assert len({sp["trunk"]["role"] for sp in specs}) == len(specs)
        # Exactly one role per fold is the LOEO-legitimate one we own (FACT-0345, FACT-0378).
        assert sum(sp["trunk"]["fold_legitimate"] for sp in specs) == 1
        for sp in specs:
            assert int(sp["fold"]) == fold and sp["no_claim"] is True
            assert sp["allow_degenerate"] is False
            tags = [m["tag"] for m in sp["models"]]
            assert len(set(tags)) == len(tags), "an arm tag repeats inside one spec"
            assert any(m["abstain"]["kind"] != "none" for m in sp["models"]), (
                "the contextual arm has no abstaining twin, so side (b) is inert on the one arm "
                "carrying the richest representation"
            )
            assert all(ModelSpec.from_dict(m).needs_cache() for m in sp["models"])
