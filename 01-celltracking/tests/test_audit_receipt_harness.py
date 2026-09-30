"""The audit harness must catch a defect that the artifact's own `all_passed` would not.

These are software contracts, not scientific promotions (CLAUDE.md rule 4): each test plants a
defect and asserts the auditor rejects it. Without them, "the receipt harness is fail-closed" is
a docstring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import audit_p33_gate1 as gate1  # noqa: E402
import audit_result_completeness as completeness  # noqa: E402

PATCH = ROOT / "scripts" / "kaggle_edits" / "assoc_feature_parity.py"


def _conds(report, log, expected=2):
    src = PATCH.read_text(encoding="utf-8")
    return {c["condition"]: c["passed"] for c in
            gate1.check_report(report, log, expected, gate1.TOL, src)}


def test_gate1_selftest_mutations_are_all_caught():
    assert gate1.selftest(PATCH) == 0


def test_gate1_accepts_a_clean_synthetic_run():
    report, log = gate1._synth(2)
    assert all(_conds(report, log).values())


def test_gate1_rejects_a_single_crop_run_the_kernel_would_pass():
    """The kernel's all_passed is TRUE here: it never asserts the crop count."""
    report, log = gate1._synth(1)
    assert report["all_passed"] is True
    assert _conds(report, log, expected=2)["1_both_crop_heartbeats"] is False


def test_gate1_rejects_a_vacuous_band_a_the_kernel_would_pass():
    """Band A comparing nothing makes conditions (3) and (4) vacuous; the kernel has no floor."""
    report, log = gate1._synth(2)
    report["crops"][0]["band_a"] = {"recorded": 0, "reproduced": 0, "missing": 0, "extra": 0,
                                    "max_abs_prob_delta": 0.0}
    conds = _conds(report, log)
    assert conds["3_node_ordering_and_identity_transitive"] is False
    assert conds["4_deployed_candidates_exact"] is False


def test_gate1_missing_report_is_a_fail_receipt_not_an_exception(tmp_path):
    receipt = gate1.build_receipt(tmp_path / "nope.json", tmp_path / "nolog.txt", 2, gate1.TOL,
                                  PATCH, "", "k", 1)
    assert receipt["verdict"] == "FAIL"
    assert "fail_reason" in receipt


def test_completeness_selftest_mutations_are_all_caught():
    import assoc_report

    assert completeness.selftest(assoc_report) == 0


@pytest.mark.parametrize("channel", ["edge_jaccard_raw", "count_adjustment", "node_recall",
                                     "division_counts"])
def test_completeness_rejects_each_required_channel(channel):
    import assoc_report

    report = completeness._synthetic()
    report["channels"].pop(channel)
    assert completeness.check_one(report, assoc_report)["accepted"] is False


def test_completeness_rejects_a_bare_net_conversion_figure():
    import assoc_report

    report = completeness._synthetic()
    report["channels"]["parent_conversions"] = {"net": 28}
    out = completeness.check_one(report, assoc_report)
    assert out["accepted"] is False
    assert any("bare net" in m for m in out["missing"])


def test_completeness_rejects_a_pooled_headline():
    import assoc_report

    report = completeness._synthetic()
    report["fold"] = "pooled"
    assert completeness.check_one(report, assoc_report)["accepted"] is False


def test_version_coherence_rebuild_binds_patch_source_drift(tmp_path):
    """The manifest hashes the EDIT SPEC, so only a rebuild catches a patch-source edit."""
    import audit_version_coherence as vc

    spec_path = ROOT / "scripts" / "kaggle_specs" / "p34_acquisition_f1.json"
    row = vc.audit_spec(spec_path, None)
    assert row["checks"]["rebuild_reproduces_built_sha256"]["passed"] is True
    manifest = json.loads(
        (ROOT / json.loads(spec_path.read_text(encoding="utf-8"))["out_dir"]
         / "build_manifest.json").read_text(encoding="utf-8"))
    # every edit payload hash is a hash of the edit SPEC, never of the code_file bytes
    assert all("payload_sha256" in e for e in manifest["edits"])
