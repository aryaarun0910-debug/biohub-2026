"""Contract tests for the registry integrity gate.

These test the GATE, not the science. Each asserts that a specific class of mistake is
caught, because an unenforced registry is just one more place to be wrong -- which is the
failure mode the registry exists to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "research" / "00-system" / "registry"
GATE = REPO / "scripts" / "core" / "validate_registry.py"


def run_gate() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE)], capture_output=True, text=True, cwd=str(REPO)
    )


def load(name: str) -> dict:
    return yaml.safe_load((REG / f"{name}.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# the registry as committed must be clean
# --------------------------------------------------------------------------------------

def test_registry_passes_as_committed():
    result = run_gate()
    assert result.returncode == 0, f"registry gate failed:\n{result.stdout}\n{result.stderr}"


def test_all_four_registry_files_exist():
    for name in ("facts", "experiments", "levers"):
        assert (REG / f"{name}.yaml").is_file(), f"{name}.yaml missing"
    assert (REG / "packets").is_dir()
    assert (REG / "packets" / "PKT-TEMPLATE.yaml").is_file()


# --------------------------------------------------------------------------------------
# R3 -- strong provenance requires an instrument
# --------------------------------------------------------------------------------------

def test_strong_provenance_facts_all_name_an_instrument():
    """VERIFIED/MEASURED without an instrument is the shape of the unreproducible 3.44%."""
    offenders = [
        f["id"] for f in load("facts")["facts"]
        if f.get("provenance") in {"VERIFIED", "MEASURED"} and not f.get("instrument")
    ]
    assert not offenders, f"strong provenance with no instrument: {offenders}"


# --------------------------------------------------------------------------------------
# R4 -- a killed lever must name reproducible evidence
# --------------------------------------------------------------------------------------

def test_killed_levers_name_their_evidence():
    for lever in load("levers")["levers"]:
        if lever.get("status") == "killed":
            assert lever.get("closed_by"), (
                f"{lever['id']} is killed but names no closed_by evidence"
            )


def test_no_lever_is_killed_on_unverified_evidence():
    prov = {f["id"]: f.get("provenance") for f in load("facts")["facts"]}
    for lever in load("levers")["levers"]:
        for fid in lever.get("closed_by") or []:
            assert prov.get(fid) != "UNVERIFIED", (
                f"{lever['id']} killed on UNVERIFIED {fid}"
            )


# --------------------------------------------------------------------------------------
# R6 -- the anti-duplication lock actually fires
# --------------------------------------------------------------------------------------

def test_duplicate_lever_claim_is_rejected():
    """Two agents on one lever is duplicated work by construction; the gate must catch it."""
    held = [
        p for p in sorted((REG / "packets").glob("PKT-*.yaml"))
        if p.stem != "PKT-TEMPLATE"
        and (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("lock")
        in ("claimed", "running")
    ]
    if not held:
        pytest.skip("no held packet to collide with")

    victim = yaml.safe_load(held[0].read_text(encoding="utf-8"))
    probe = REG / "packets" / "PKT-9999.yaml"
    probe.write_text(
        yaml.safe_dump(
            {
                "id": "PKT-9999",
                "title": "duplicate-claim probe",
                "lever": victim["lever"],
                "owner": "probe",
                "lock": "claimed",
            }
        ),
        encoding="utf-8",
    )
    try:
        result = run_gate()
        assert result.returncode != 0, "gate accepted two packets holding the same lever"
        assert "R6" in result.stdout
    finally:
        probe.unlink()

    assert run_gate().returncode == 0, "gate did not recover after the probe was removed"


# --------------------------------------------------------------------------------------
# R5 -- superseded values may not be asserted as current in state docs
# --------------------------------------------------------------------------------------

def test_superseded_facts_point_at_a_successor():
    facts = load("facts")["facts"]
    ids = {f["id"] for f in facts}
    for f in facts:
        if f.get("provenance") == "SUPERSEDED":
            assert f.get("superseded_by") in ids, (
                f"{f['id']} is SUPERSEDED but names no valid successor"
            )


def test_state_docs_are_classified():
    """The entry points an agent reads first must be under the R5 guard."""
    import re

    must_be_state = [
        "research/README.md",
        "research/00-system/handoff.md",
        "research/07-outputs/leaderboard.md",
    ]
    for rel in must_be_state:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert re.search(r"^record_kind:\s*state", text, re.M), (
            f"{rel} is an orientation doc and must be record_kind: state"
        )
