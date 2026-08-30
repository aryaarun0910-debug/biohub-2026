"""Contracts for the GPU PROTECTION CONTRACT instrument (PKT-0036).

Every test here plants the violation it guards. The point is not that the four sections pass on
today's artifact - that is a fact about today's artifact - but that each clause REJECTS the defect
it exists for. A clause that has never rejected anything has never been tested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import gpu_protection_contract as G  # noqa: E402


def _clause(clauses, cid):
    return [c for c in clauses if c["id"] == cid][0]


# ---------------------------------------------------------------------------- CODE
def test_the_generated_patch_matches_its_generator():
    """THE TRAP THAT PRODUCED THE CLAUSE. assoc_tap_gate.py is RENDERED from assoc_tap_replay.py,
    so committing the file a check names can still leave HEAD failing its own drift lock."""
    import sync_tap_worker

    on_disk = (ROOT / "scripts" / "kaggle_edits" / "assoc_tap_gate.py").read_text(encoding="utf-8")
    assert sync_tap_worker.rendered() == on_disk, (
        "the shipped generated patch is stale against its generator; run "
        "python scripts/win_bet/sync_tap_worker.py --write"
    )


def test_a_duplicate_slug_or_out_dir_is_rejected(tmp_path):
    spec = {"slug": "s", "out_dir": "d", "code_file": "n.ipynb", "edits": []}
    (tmp_path / "a.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(spec), encoding="utf-8")
    cl = G.section_code(spec, tmp_path / "a.json", tmp_path,
                        [tmp_path / "a.json", tmp_path / "b.json"])
    assert not _clause(cl, "CODE-4")["passed"]


def test_the_vendored_predictor_does_not_satisfy_the_deployed_pin():
    """FACT-0408: the vendored copy has no fusion code and cannot exhibit the defect under test."""
    ven = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
    if not ven.is_file():
        pytest.fail("the vendored tree is absent, so this contrast cannot be exercised - that is a "
                    "FAILURE and not a skip, because the pin's whole purpose is the contrast")
    import hashlib

    assert hashlib.sha256(ven.read_bytes()).hexdigest() != G.DEPLOYED_PREDICTOR_SHA256


# ---------------------------------------------------------------------------- DATA
F1 = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "1"}}]}
HONEST_F1 = {"role": "oof_split1", "fold_legitimate": True,
             "dual_trunk_pair": ["oof_split1", "stabledet"],
             "contamination": "none", "checkpoint_sha256": "2e4ebf616b3d4fb5"}


def test_the_honest_fold1_trunk_pair_is_accepted():
    """THE ACCEPT CONTROL. A guard that refuses everything proves nothing."""
    assert all(c["passed"] for c in G.section_data(F1, HONEST_F1))


def test_the_fact_0418_pair_with_no_legitimate_arm_is_rejected():
    bad = dict(HONEST_F1, dual_trunk_pair=list(G.INVALID_PAIR))
    c = _clause(G.section_data(F1, bad), "DATA-3")
    assert not c["passed"]
    assert c["evidence"]["is_the_fact_0418_invalid_pair"] is True


def test_an_illegitimate_trunk_role_is_rejected_even_when_the_spec_declares_it_legitimate():
    """FACT-0418 retracted 'official' to UNVERIFIED: it is byte-identical to our own split_0, which
    is LEAKY on fold 1. A spec asserting fold_legitimate true does not make it so."""
    lying = dict(HONEST_F1, role="official", fold_legitimate=True)
    assert not _clause(G.section_data(F1, lying), "DATA-2")["passed"]


def test_a_missing_checkpoint_hash_is_rejected():
    no_hash = dict(HONEST_F1)
    no_hash.pop("checkpoint_sha256")
    assert not _clause(G.section_data(F1, no_hash), "DATA-4")["passed"]


def test_an_unresolvable_fold_is_rejected():
    assert not _clause(G.section_data({"edits": []}, None), "DATA-1")["passed"]


# ---------------------------------------------------------------------------- PROCESS
GOOD_NB = ('env = dict(os.environ)\nenv["PYTHONPATH"] = "scripts"\nprint("AFT_GATE ok")\n'
           'start = time.time()\nelapsed = time.time() - start\nif elapsed > BUDGET: abort()\n')
GOOD_SPEC = {"edits": [{"vars": {"BIOHUB_AFT_EXPECT_CROPS": "2", "BIOHUB_LOEO_LIMIT": "2"}}]}


def test_a_correct_notebook_passes_every_process_clause():
    assert all(c["passed"] for c in G.section_process(GOOD_NB, GOOD_SPEC))


def test_an_env_that_does_not_reach_the_subprocess_is_rejected():
    """FACT-0060 / FACT-0399: parent-process state does not reach the child."""
    nb = GOOD_NB.replace("env = dict(os.environ)", "")
    assert not _clause(G.section_process(nb, GOOD_SPEC), "PROC-1")["passed"]


def test_a_stripped_heartbeat_is_rejected():
    nb = GOOD_NB.replace("AFT_GATE", "quiet")
    assert not _clause(G.section_process(nb, GOOD_SPEC), "PROC-2")["passed"]


@pytest.mark.parametrize("spec", [
    {"edits": [{"vars": {"BIOHUB_AFT_EXPECT_CROPS": "2", "BIOHUB_LOEO_LIMIT": "1"}}]},
    {"edits": []},
])
def test_a_crop_count_that_is_absent_or_disagrees_with_the_limit_is_rejected(spec):
    """The kernel's own all_passed cannot see a crop that never ran, so the count is asserted
    externally or it is not asserted at all."""
    assert not _clause(G.section_process(GOOD_NB, spec), "PROC-3")["passed"]


def test_a_missing_timing_probe_is_rejected():
    assert not _clause(G.section_process("print('AFT_GATE')\nenv = dict()\nPYTHONPATH",
                                         GOOD_SPEC), "PROC-4")["passed"]


# ---------------------------------------------------------------------------- ARTIFACT
GOOD_ART = ('os.replace(tmp, out)\n_LOEO_KEEP |= {"aft_cache.tar.gz"}\ncache_manifest\n'
            'aft_gate.json\n')


def test_a_correct_notebook_passes_every_artifact_clause():
    assert all(c["passed"] for c in G.section_artifact(GOOD_ART, {}, 2332346))


def test_an_artifact_dropped_from_the_export_sweep_is_rejected():
    nb = GOOD_ART.replace("_LOEO_KEEP", "x")
    assert not _clause(G.section_artifact(nb, {}, 1), "ART-1")["passed"]


def test_a_capacity_guard_with_no_denominator_is_rejected():
    """'Non-empty' is insufficient; a guard with no node count has verified nothing."""
    assert not _clause(G.section_artifact(GOOD_ART, {}, None), "ART-2")["passed"]


def test_the_superseded_storage_figure_would_have_passed_a_run_the_rederived_one_refuses():
    """FACT-0416: the earlier estimate was 12.3x LOW. This is the whole reason the clause pins a
    value rather than a method."""
    big = 14_000_000
    assert big * G.SUPERSEDED_BYTES_PER_NODE * 2 / 1024**3 <= 20.0, "the premise of this test moved"
    assert not _clause(G.section_artifact(GOOD_ART, {}, big), "ART-2")["passed"]


def test_expected_outputs_are_rederived_from_the_notebook_and_not_from_a_receipt():
    """FACT-0417: the signed receipt's expected_gpu_outputs was hardcoded to the wrong worker,
    which would have FAILED a correct run and PASSED a run that recorded nothing."""
    c = _clause(G.section_artifact(GOOD_ART, {}, 1), "ART-4")
    assert c["passed"] and c["evidence"]["receipt_trusted"] is False
    assert c["evidence"]["expected_outputs_rederived_from_notebook"]
    blind = _clause(G.section_artifact("os.replace(a,b)\n_LOEO_KEEP\ncache_manifest", {}, 1), "ART-4")
    assert not blind["passed"]


# ---------------------------------------------------------------------------- driver
def test_the_selftest_catches_every_mutation_it_plants(tmp_path):
    assert G._selftest(tmp_path) == 0


def test_signing_refuses_when_the_built_notebook_is_absent(tmp_path):
    """FAIL CLOSED. A contract that cannot see the artifact must refuse, never pass by default."""
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"slug": "s", "out_dir": "nope", "code_file": "missing.ipynb",
                                "edits": []}), encoding="utf-8")
    with pytest.raises(G.ContractRefusal, match="built notebook absent"):
        G.sign(spec, tmp_path)
