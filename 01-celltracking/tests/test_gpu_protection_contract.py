"""Contracts for the GPU PROTECTION CONTRACT instrument (PKT-0036), schema 2.

Every test plants the violation it guards. The point is not that today's artifacts pass - that is a
fact about today's artifacts - but that each clause REJECTS the defect it exists for, and that the
two PROFILES cannot launder each other.

The three FALSE PASSES that motivated the profile system are pinned as regression tests, because a
false pass is worse than a fail: nobody looks at it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import gpu_protection_contract as G  # noqa: E402

PASS, FAIL, OOS = G.PASS, G.FAIL, G.OOS


def ctx(nb_text="", spec=None, trunk=None, nodes=None, repo=None):
    return {"spec": spec or {}, "spec_path": Path("x.json"), "spec_rel": "x.json",
            "repo": repo or ROOT, "trunk": trunk, "nodes": nodes, "nb_text": nb_text,
            "all_specs": []}


def status(fn, **kw):
    return fn(ctx(**kw))["status"]


SUB_NB = G.GOOD_SUB_NB
SUB_SPEC = G.GOOD_SUB_SPEC
CACHE_NB = G.GOOD_CACHE_NB
CACHE_SPEC = G.GOOD_CACHE_SPEC


# ------------------------------------------------------------------ the profile machinery
def test_an_unknown_profile_refuses_rather_than_defaulting():
    """Not a skip, not a default. A typo must not silently disable the contract."""
    with pytest.raises(G.ContractRefusal, match="unknown gpu_contract_profile"):
        G.select_profile({"gpu_contract_profile": "definitely_not_a_profile"})


def test_a_spec_cannot_exempt_an_individual_clause():
    """The spec SELECTS a profile. It may never decide clause applicability - that would let a
    submitter exempt itself from the clause it is about to violate."""
    spec = {"expects_submission": True, "gpu_contract_skip": ["SART-1"],
            "clauses": [], "SART-1": False}
    assert G.select_profile(spec) == "submission_run_v1"
    assert "SART-1" in G.PROFILES["submission_run_v1"]["clauses"]


def test_every_non_applicable_clause_has_a_contract_owned_oos_reason():
    """An OOS reason is written in the contract, never supplied by the submitting agent."""
    missing = [(p, c) for p in G.PROFILES for c in G.CLAUSES
               if c not in G.PROFILES[p]["clauses"] and (p, c) not in G.OOS_REASONS]
    assert not missing, f"clauses would be dropped with no recorded reason: {missing}"


def test_signing_refuses_when_the_built_notebook_is_absent(tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"slug": "s", "out_dir": "nope", "code_file": "missing.ipynb",
                                "edits": []}), encoding="utf-8")
    with pytest.raises(G.ContractRefusal, match="built notebook absent"):
        G.sign(spec, tmp_path)


# ------------------------------------------------------------------ the three v1 false passes
def test_a_wheel_glob_no_longer_satisfies_the_cache_reconstruction_clause():
    """V1 ART-3 passed on the bare token 'tar.gz', which in the P38 submission notebook is
    ("*.whl", "*.tar.gz", "*.zip") - a wheel-install glob - while naming audit_feature_cache.py as
    the auditor. It reported green on an artifact with no cache at all."""
    assert status(G.c_art_3, nb_text='patterns = ("*.whl", "*.tar.gz", "*.zip")') == FAIL


def test_an_ilp_node_budget_no_longer_satisfies_the_time_budget_clause():
    """V1 PROC-4 passed on the word BUDGET, matching short_track_rescue_budget - a rescue NODE
    budget with no temporal meaning."""
    nb = 't = time.time()\nbudget = min(3, n)\nstats["short_track_rescue_budget"] = budget'
    assert status(G.c_proc_4, nb_text=nb) == FAIL


def test_an_inapplicable_trunk_clause_is_out_of_scope_and_not_a_pass():
    """V1 DATA-2/DATA-3 reported PASS while their evidence said 'no trunk declared'."""
    assert status(G.c_data_2, spec=CACHE_SPEC, trunk=None) == OOS
    assert status(G.c_data_3, spec=CACHE_SPEC, trunk=None) == OOS


def test_a_build_with_no_generated_patch_is_out_of_scope_and_not_a_silent_pass():
    assert status(G.c_code_2, spec={"edits": []}) == OOS


# ------------------------------------------------------------------ cross-class laundering
SUB_CLAUSES = ("SPROC-2", "SPROC-3", "SART-1", "SART-2", "SART-3", "SART-4")
CACHE_CLAUSES = ("PROC-2", "PROC-3", "ART-3", "ART-4")


@pytest.mark.parametrize("cid", SUB_CLAUSES)
def test_a_submission_cannot_pass_using_cache_tokens(cid):
    """The cache notebook is full of AFT/cache tokens and satisfies its own class. It must not
    satisfy a single submission clause."""
    assert status(G.CLAUSES[cid][1], nb_text=CACHE_NB, spec=SUB_SPEC) == FAIL


@pytest.mark.parametrize("cid", CACHE_CLAUSES)
def test_a_cache_run_cannot_pass_using_submission_tokens(cid):
    """And the mirror: a submission notebook must not satisfy a cache clause."""
    assert status(G.CLAUSES[cid][1], nb_text=SUB_NB, spec=SUB_SPEC) == FAIL


# ------------------------------------------------------------------ accept controls
def test_a_correct_submission_notebook_passes_its_own_clauses():
    """A guard that refuses everything distinguishes nothing."""
    bad = [c for c in ("SDATA-1", "PROC-1", "SPROC-1", "SPROC-2", "SPROC-3",
                       "SART-1", "SART-2", "SART-3", "SART-4")
           if status(G.CLAUSES[c][1], nb_text=SUB_NB, spec=SUB_SPEC) == FAIL]
    assert not bad, bad


def test_a_correct_cache_notebook_passes_its_own_clauses():
    bad = [c for c in ("PROC-1", "PROC-2", "PROC-3", "PROC-4", "ART-1", "ART-3", "ART-4")
           if status(G.CLAUSES[c][1], nb_text=CACHE_NB, spec=CACHE_SPEC) != PASS]
    assert not bad, bad


# ------------------------------------------------------------------ submission clauses
def test_a_submission_that_also_claims_a_loeo_fold_is_rejected():
    """A hidden-test submission and a held-out fold are different evaluation scopes."""
    spec = dict(SUB_SPEC, edits=[{"kind": "env", "vars": {"BIOHUB_LOEO_FOLD": "0"}}])
    assert status(G.c_sdata_1, nb_text=SUB_NB, spec=spec) == FAIL
    nb = SUB_NB + '\nos.environ["BIOHUB_LOEO_ARM"] = "champion"\n'
    assert status(G.c_sdata_1, nb_text=nb, spec=SUB_SPEC) == FAIL


def test_an_unbound_base_hash_is_rejected():
    spec = dict(SUB_SPEC, base_notebook="pyproject.toml", base_sha256="deadbeef")
    assert status(G.c_sdata_2, spec=spec) == FAIL


def test_the_live_treatment_heartbeat_rejects_a_treatment_that_is_not_live():
    """The spec's value must WIN the last write, be read back, and be RECORDED into run_stats."""
    assert status(G.c_sproc_1, nb_text=SUB_NB.replace("= '2.0'", "= '1.0'"),
                  spec=SUB_SPEC) == FAIL
    assert status(G.c_sproc_1,
                  nb_text=SUB_NB.replace('"motion_relink_learned_bonus"', '"other"'),
                  spec=SUB_SPEC) == FAIL


def test_a_submission_declaring_no_treatment_is_rejected():
    """An untreated resubmission is not a calibration; there is nothing to prove live."""
    assert status(G.c_sproc_1, nb_text=SUB_NB, spec=dict(SUB_SPEC, edits=[])) == FAIL


def test_a_missing_completion_heartbeat_is_rejected():
    assert status(G.c_sproc_2, nb_text=SUB_NB.replace("SUBMISSION_COMPLETE", "done")) == FAIL


def test_a_missing_runtime_reconciliation_is_rejected():
    """Not a predetermined crop count - the hidden test set's size is unknowable before the run."""
    nb = SUB_NB.replace("if len(geffs) != len(test_stems):", "")
    assert status(G.c_sproc_3, nb_text=nb) == FAIL


def test_a_direct_truncating_write_of_the_submission_is_rejected():
    nb = SUB_NB.replace("os.replace(SUBMISSION_TMP, SUBMISSION_PATH)", 'SUBMISSION_PATH.open("w")')
    assert status(G.c_sart_1, nb_text=nb) == FAIL


def test_a_missing_output_size_guard_is_rejected():
    assert status(G.c_sart_2, nb_text=SUB_NB.replace("row_id == total_nodes + total_edges", "")) == FAIL


def test_the_expected_output_is_rederived_as_submission_csv_from_the_notebook():
    """FACT-0417's shape: a receipt field hardcoded to the wrong worker would FAIL a correct run
    and PASS a run that recorded nothing."""
    c = G.c_sart_3(ctx(nb_text=SUB_NB, spec=SUB_SPEC))
    assert c["status"] == PASS and c["evidence"]["receipt_trusted"] is False
    nb = SUB_NB.replace("submission.csv", "other.csv").replace("_guard_submission", "_g")
    assert status(G.c_sart_3, nb_text=nb, spec=SUB_SPEC) == FAIL


def test_a_digest_that_is_only_a_variable_name_does_not_satisfy_the_receipt_clause():
    """Found by the mutation test: matching `_guard_digest` would let `_guard_digest = 0` pass."""
    nb = SUB_NB.replace("hashlib.sha256(_guard_submission.read_bytes())", "0")
    assert status(G.c_sart_4, nb_text=nb) == FAIL


# ------------------------------------------------------------------ cache clauses retained
def test_the_generated_patch_matches_its_generator():
    import sync_tap_worker

    on_disk = (ROOT / "scripts" / "kaggle_edits" / "assoc_tap_gate.py").read_text(encoding="utf-8")
    assert sync_tap_worker.rendered() == on_disk


def test_the_fact_0418_pair_with_no_legitimate_arm_is_rejected():
    trunk = {"role": "official", "dual_trunk_pair": list(G.INVALID_PAIR)}
    c = G.c_data_3(ctx(spec=CACHE_SPEC, trunk=trunk))
    assert c["status"] == FAIL and c["evidence"]["is_the_fact_0418_invalid_pair"] is True


def test_an_illegitimate_role_declared_legitimate_is_rejected():
    trunk = {"role": "official", "fold_legitimate": True}
    assert status(G.c_data_2, spec={"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "1"}}]},
                  trunk=trunk) == FAIL


def test_the_superseded_storage_figure_would_have_passed_a_run_the_rederived_one_refuses():
    big = 14_000_000
    assert big * G.SUPERSEDED_BYTES_PER_NODE * 2 / 1024**3 <= 20.0, "the premise of this test moved"
    assert status(G.c_art_2, nodes=big) == FAIL


def test_a_capacity_guard_with_no_denominator_is_rejected():
    assert status(G.c_art_2, nodes=None) == FAIL


# ------------------------------------------------------------------ driver
def test_the_selftest_catches_every_mutation_it_plants(tmp_path):
    assert G._selftest(tmp_path) == 0


def test_a_bare_token_does_not_satisfy_the_atomic_submission_clause():
    """Found by mutating the REAL P38 candidate: the earlier alternation accepted the bare token
    SUBMISSION_TMP, so a comment plus `x = 'SUBMISSION_TMP'` passed. Matching a token rather than a
    mechanism is exactly how v1's ART-3 passed on a wheel-install glob."""
    good = 'os.replace(_SUB_TMP, SUBMISSION_PATH)\n'
    assert status(G.c_sart_1, nb_text=good) == PASS
    for bad in ("# os.replace mentioned\nx = 'SUBMISSION_TMP'\n",
                '_SUB_TMP = SUBMISSION_PATH.with_suffix(".csv.part")\n',
                'with SUBMISSION_PATH.open("w") as f:\n'):
        assert status(G.c_sart_1, nb_text=bad) == FAIL, bad
