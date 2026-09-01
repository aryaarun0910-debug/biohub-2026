"""THE RECEIPT ENVELOPE'S CONTRACTS.

Raw release receipts are gitignored (`.gitignore:20`, `notebooks/**/_out/`), so a fresh clone has
none of them - measured 2026-09-01, 4 of 74 notebook directories carry one and all four are
ignored. The envelope makes the BINDING durable without publishing the raw audit product.

Three properties carry the whole design, and each is a way it could go wrong:

  1. NOTHING MACHINE-LOCAL REACHES A TRACKED FILE. The envelope is allow-listed, not redacted:
     fields are copied by name, so an unreviewed field cannot arrive by default.
  2. AN ENVELOPE NEVER UPGRADES PROVENANCE. Generating one in September for an artifact audited
     in July does not make that artifact better bound. Binding is derived from the registry.
  3. A CLONE CAN TELL THE FOUR STATES APART - fully bound with the raw receipt here, bound by
     envelope alone, historical and unbound, or unknown.

Software contracts only (CLAUDE.md rule 4).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

import receipt_envelope as RE  # noqa: E402

ENV_FILE = ROOT / "research" / "00-system" / "registry" / "generated" / "receipts.json"
ALLOWED = {
    "receipt_id", "notebook_dir", "raw_receipt_location", "raw_receipt_available_here",
    "raw_receipt_tracked", "schema_version", "redaction_status", "notebook_sha256",
    "manifest_sha256", "artifact_sha256", "spec_sha256", "audit_tool_version",
    "audit_tool_sha256", "kernel_slug", "kernel_version", "submission_reference",
    "experiments", "facts", "state", "binding_strength", "binding_basis", "verdict",
    # Phase 1.5: the envelope now carries BOTH identities and says which is which.
    "notebook_canonical_sha256", "canonicalization_version", "hash_kinds",
    "receipt_recorded_manifest_sha256", "receipt_recorded_notebook_sha256",
    "git_reproducible",
}
STATES = {"fully_bound_raw_available", "bound_envelope_only", "historical_unbound", "unknown"}


@pytest.fixture(scope="module")
def payload():
    assert ENV_FILE.is_file(), "run scripts/core/receipt_envelope.py"
    return json.loads(ENV_FILE.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------
# 1. NOTHING MACHINE-LOCAL, AND NOTHING UNREVIEWED
# ------------------------------------------------------------------------------------------
def test_no_machine_local_path_reaches_the_tracked_envelope():
    text = ENV_FILE.read_text(encoding="utf-8")
    hits = re.findall(r"[A-Za-z]:[\\/]|/home/|/Users/|\\\\\\\\", text)
    assert not hits, f"the tracked envelope carries machine-local paths: {sorted(set(hits))[:5]}"


def test_only_allow_listed_fields_are_present(payload):
    for name, env in payload["envelopes"].items():
        extra = sorted(set(env) - ALLOWED)
        assert not extra, (
            f"{name} carries un-allow-listed field(s) {extra}. The envelope copies fields BY "
            f"NAME so that an unreviewed field cannot arrive by default; adding one requires "
            f"reviewing it and adding it here.")


def test_the_allow_list_refuses_a_local_path():
    """The refusal must be demonstrated, not asserted."""
    with pytest.raises(RE.EnvelopeRefusal, match="machine-local"):
        RE._clean(r"C:\\Users\\someone\\out\\submission.csv", "submission.path")
    assert RE._clean("aryaarun07/biohub-p35", "kernel.slug") == "aryaarun07/biohub-p35"


def test_the_envelope_is_tracked_and_the_raw_receipts_are_not(payload):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                              ENV_FILE.relative_to(ROOT).as_posix()],
                             cwd=ROOT, capture_output=True).returncode == 0
    assert tracked, "the envelope is not tracked, so a clone still has no binding"
    for name, env in payload["envelopes"].items():
        assert env["raw_receipt_tracked"] is False
        ignored = subprocess.run(["git", "check-ignore", "-q", env["raw_receipt_location"]],
                                 cwd=ROOT, capture_output=True).returncode == 0
        assert ignored, f"{env['raw_receipt_location']} is NOT ignored - raw receipts must stay so"


# ------------------------------------------------------------------------------------------
# 2. NO RETROSPECTIVE UPGRADE
# ------------------------------------------------------------------------------------------
def test_binding_comes_from_the_registry_not_from_the_envelopes_existence(payload):
    doc = yaml.safe_load(
        (ROOT / "research" / "00-system" / "registry" / "experiments.yaml").read_text("utf-8"))
    exps = doc["experiments"] if isinstance(doc, dict) and "experiments" in doc else doc
    known = {e["id"] for e in exps}
    for name, env in payload["envelopes"].items():
        for eid in env["experiments"]:
            assert eid in known, f"{name} claims a non-existent experiment {eid}"
        if env["state"] == "historical_unbound":
            assert not env["experiments"] and not env["facts"], (
                f"{name} is historical_unbound yet carries bindings - an envelope must never "
                f"upgrade an artifact's provenance")
            assert env["binding_strength"] == "none"


def test_every_envelope_states_one_of_the_four_states_and_says_why(payload):
    for name, env in payload["envelopes"].items():
        assert env["state"] in STATES, f"{name} has state {env['state']!r}"
        assert env["binding_basis"], f"{name} states no basis for its binding"


def test_the_champion_binding_survives_without_the_raw_receipt(payload):
    """The point of the whole exercise: a clone can still verify what the champion was."""
    roles = yaml.safe_load(
        (ROOT / "research" / "00-system" / "registry" / "baseline_roles.yaml").read_text("utf-8"))
    champ_dir = Path(roles["leaderboard_champion"]["notebook"]).parent.name
    env = payload["envelopes"][champ_dir]
    assert env["experiments"] == [roles["leaderboard_champion"]["experiment"]]
    assert env["submission_reference"] == roles["leaderboard_champion"]["submission"]
    assert env["notebook_sha256"] == roles["leaderboard_champion"]["notebook_sha256"]
    assert env["artifact_sha256"], "the submitted artifact's digest is not carried"
    assert env["kernel_slug"] and env["kernel_version"] is not None


# ------------------------------------------------------------------------------------------
# 3. DRIFT
# ------------------------------------------------------------------------------------------
def test_the_envelope_file_has_no_drift():
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "core" / "receipt_envelope.py"),
                         "--check"], cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert rc.returncode == 0, rc.stdout[-600:]
