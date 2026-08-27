"""Software contracts for LEVER-0023 stage-1 prerequisites (PKT-0019, 2026-08-27).

FACT-0329 exposed the hole: the S5 smoke initialised from the pack weights, and a run judged on a
LOEO fold must initialise from the out-of-fold weights for that fold or the transfer gate is not
clean. These tests pin (a) the runner's fail-closed override, (b) the full spec's use of it for a
fold-0-judged run, and (c) the reciprocal S5 -> LOEO consumer that gives gate 3 an instrument.
They say nothing about whether the weights help - that is the packet's job.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "kaggle_edits" / "h1r_edge_kernel_run.py"
S5 = ROOT / "scripts" / "kaggle_specs" / "h1r_edge_s5.json"
S5_SMOKE = ROOT / "scripts" / "kaggle_specs" / "h1r_edge_s5_smoke.json"
CONSUMER = ROOT / "scripts" / "kaggle_specs" / "deploy_h1r_edge_s5_loeo_f0.json"
P20 = ROOT / "scripts" / "kaggle_specs" / "p20_relink_sweep_f0.json"


def _env(spec: dict) -> dict:
    out: dict = {}
    for e in spec.get("edits", []):
        if e.get("kind") == "env":
            out.update(e.get("vars", {}))
    return out


def test_runner_override_fails_closed_and_stages_config_next_to_weights():
    src = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("H1R_EDGE_INIT_WEIGHTS_GLOB"' in src
    assert 'os.environ.get("H1R_EDGE_INIT_CONFIG_GLOB"' in src
    # both or neither
    assert "must be set together" in src
    # exactly one hit each, else RuntimeError (no silent fallback to the pack)
    assert "exactly one hit each is required" in src
    # the loader contract needs config.json NEXT TO the weights file
    assert '_stage / "config.json"' in src and '_stage / "edge_predictor_best.pth"' in src
    # the resolved initialisation is printed with its sha256 so the kernel log is evidence
    assert '"H1R_INIT_WEIGHTS"' in src and "sha256" in src
    # the bounded depth ladder, never a recursive walk of /kaggle/input
    assert "**" not in src.split("def _find_input")[1].split("return sorted")[0]


def test_runner_exports_uniquely_named_copies_for_the_loeo_consumer():
    src = RUNNER.read_text(encoding="utf-8")
    assert '"edge_predictor_best_h1r_s5.pth"' in src
    assert '"config_h1r_s5.json"' in src


def test_full_s5_spec_initialises_from_oof_split_0_for_a_fold0_judged_run():
    spec = json.loads(S5.read_text(encoding="utf-8"))
    env = _env(spec)
    assert env["H1R_EDGE_INIT_WEIGHTS_GLOB"].endswith("edge_predictor_best_split_0.pth")
    assert env["H1R_EDGE_INIT_CONFIG_GLOB"].endswith("config_split_0.json")
    assert "split_1" not in env["H1R_EDGE_INIT_WEIGHTS_GLOB"]
    assert "aryaarun07/biohub-oof-weights" in spec["datasets"]
    assert spec["artifact_role"] == "training"
    consumers = {row["artifact"]: row for row in spec["deploy_consumers"]}
    assert consumers["edge_predictor_best.pth"]["spec"] == "scripts/kaggle_specs/deploy_h1r_edge_s5_loeo_f0.json"
    # the appearance projector is declared an undeployed by-product, honestly, with a reason
    assert "spec" not in consumers["appearance_projector_best.pth"]
    assert len(consumers["appearance_projector_best.pth"]["byproduct_reason"]) >= 12


def test_smoke_spec_keeps_the_pack_default_so_its_gate_1_evidence_stays_comparable():
    env = _env(json.loads(S5_SMOKE.read_text(encoding="utf-8")))
    assert "H1R_EDGE_INIT_WEIGHTS_GLOB" not in env


def test_full_s5_first_run_does_not_pretend_to_resume():
    """DG-005: /kaggle/working starts empty, so a first run must not enable resume without resume_sources."""
    spec = json.loads(S5.read_text(encoding="utf-8"))
    assert _env(spec)["H1R_EDGE_RESUME"] == "0" or spec.get("resume_sources")


def test_consumer_is_reciprocal_and_globs_the_unique_basename():
    c = json.loads(CONSUMER.read_text(encoding="utf-8"))
    assert c["artifact_role"] == "deployment"
    assert c["kernel_sources"] == ["aryaarun07/biohub-h1r-edge-s5"]
    assert {"producer_spec": "scripts/kaggle_specs/h1r_edge_s5.json", "artifact": "edge_predictor_best.pth"} in c["consumes_artifacts"]
    env = _env(c)
    assert env["BIOHUB_LOEO_FOLD"] == "0"
    assert env["BIOHUB_LOEO_ARM"] == "strict"
    assert env["BIOHUB_LOEO_WEIGHTS_GLOB"] == "/kaggle/input/*/edge_predictor_best_h1r_s5.pth"
    assert env["BIOHUB_LOEO_CONFIG_GLOB"] == "/kaggle/input/*/config_h1r_s5.json"
    assert c["expects_submission"] is False


def test_consumer_is_the_champion_configuration_minus_the_relink_sweep():
    """Gate 3 pairs this export against the EXP-0022 control; both must be the same pipeline."""
    c = json.loads(CONSUMER.read_text(encoding="utf-8"))
    p20 = json.loads(P20.read_text(encoding="utf-8"))
    assert c["base_notebook"] == p20["base_notebook"] and c["base_sha256"] == p20["base_sha256"]
    ce, pe = _env(c), _env(p20)
    for key in ("BIOHUB_SAFE_DIV_MAX_UM", "BIOHUB_SAFE_DIV_SISTER_MAX_UM",
                "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "BIOHUB_SAFE_DIV_DIVERGE_UM", "BIOHUB_LOEO_STEMS"):
        assert ce[key] == pe[key], key
    assert "BIOHUB_RELINK_DIVISION_SWEEP" not in ce
    code_files = {e.get("code_file") for e in c["edits"]}
    assert "scripts/kaggle_edits/coupled_division_transplant.py" in code_files
    assert "scripts/kaggle_edits/loeo_retarget.py" in code_files
    assert "scripts/kaggle_edits/loeo_export.py" in code_files
    assert "scripts/kaggle_edits/relink_sweep.py" not in code_files
    assert len(json.loads(ce["BIOHUB_LOEO_STEMS"])) == 71
    assert all(re.match(r"^44b6_", s) for s in json.loads(ce["BIOHUB_LOEO_STEMS"]))
