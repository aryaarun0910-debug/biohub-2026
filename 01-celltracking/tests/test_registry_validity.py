"""Contract tests for the VALIDITY axis - registry rules R7, R8, R9, R10.

WHY THESE EXIST
---------------
EXP-0019 scored LOEO fold 1 (the 6bba embryo) with weights TRAINED on 6bba. It inflated the
fold score by +0.203 and manufactured 25 of 26 division true positives. Seven facts, eight
packets and a whole strategy were built on it before anyone noticed, and every gate in the
repo stayed green throughout - because the leaked facts were MEASURED, and they DESERVED it.
They were correctly computed from an invalid run, and the schema had no way to say so.

tests/test_loeo_weights_hygiene.py guards that SPECIFIC defect: a fold-1 spec that forgets
BIOHUB_LOEO_WEIGHTS_GLOB. These tests guard the CLASS: whatever the next invalid run turns
out to be, voiding it must reach every fact that came from it, and no lever may stay closed
on the wreckage.

Each test plants a violation, asserts the gate REJECTS it, removes it, and asserts the gate
RECOVERS - a rule that only ever passes has not been shown to fire.
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

FACTS = REG / "facts.yaml"
EXPERIMENTS = REG / "experiments.yaml"
LEVERS = REG / "levers.yaml"

# The known leak cohort, from the 2026-08-26 incident.
LEAK_COHORT = [
    "FACT-0130", "FACT-0131", "FACT-0140", "FACT-0141",
    "FACT-0150", "FACT-0151", "FACT-0152", "FACT-0153",
]


def run_gate() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE)], capture_output=True, text=True, cwd=str(REPO)
    )


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def facts_by_id() -> dict[str, dict]:
    return {f["id"]: f for f in load(FACTS)["facts"]}


class Planted:
    """Rewrite a registry file, run the gate, then restore the file byte-for-byte.

    Restoring the ORIGINAL BYTES rather than re-serialising matters: facts.yaml is
    hand-formatted with load-bearing comments, and a yaml round-trip would silently
    flatten them.
    """

    def __init__(self, path: Path):
        self.path = path
        self.original = path.read_bytes()

    def __enter__(self) -> "Planted":
        return self

    def write(self, doc: dict) -> None:
        self.path.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def __exit__(self, *exc) -> None:
        self.path.write_bytes(self.original)


@pytest.fixture(scope="module")
def clean_gate() -> subprocess.CompletedProcess:
    """The gate on the registry as committed.

    Module-scoped on purpose: the gate takes ~2 s (R5 walks every research/**.md) and a
    per-test before/after pair would spend two minutes proving the same thing 24 times.
    Each planting test asserts its own recovery, and this fixture bookends the module.
    """
    result = run_gate()
    assert result.returncode == 0, f"registry is not clean as committed:\n{result.stdout}"
    return result


@pytest.fixture(scope="module", autouse=True)
def _registry_is_clean_after_the_module(clean_gate):
    """No test may leave the registry dirty for the rest of the suite."""
    yield
    after = run_gate()
    assert after.returncode == 0, f"a test left the registry failing:\n{after.stdout}"


# ======================================================================================
# R7 -- the validity vocabulary, and a reason whenever it is not VALID
# ======================================================================================

def test_validity_vocabulary_is_closed():
    allowed = {"VALID", "SUSPECT", "INVALID", "UNKNOWN"}
    for f in load(FACTS)["facts"]:
        v = f.get("validity")
        assert v is None or v in allowed, f"{f['id']}: validity {v!r} outside the vocabulary"


def test_gate_rejects_an_unknown_validity_value():
    doc = load(FACTS)
    doc["facts"][0]["validity"] = "PROBABLY_FINE"
    with Planted(FACTS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "gate accepted a validity value outside the vocabulary"
        assert "R7" in result.stdout


def test_gate_rejects_a_retraction_with_no_reason():
    """INVALID with no reason is a dead end - the reader cannot tell what to do next."""
    doc = load(FACTS)
    target = next(f for f in doc["facts"] if f["id"] == "FACT-0130")
    target.pop("validity_reason", None)
    with Planted(FACTS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "gate accepted INVALID with no validity_reason"
        assert "R7" in result.stdout and "FACT-0130" in result.stdout


def test_every_non_valid_fact_states_a_reason():
    for f in load(FACTS)["facts"]:
        if f.get("validity") not in (None, "VALID"):
            assert str(f.get("validity_reason") or "").strip(), (
                f"{f['id']} is {f['validity']} but states no validity_reason"
            )


# ======================================================================================
# R8 -- retraction PROPAGATES from a voided experiment to its facts
# ======================================================================================

def test_the_exp0019_cohort_is_linked_and_marked():
    """The backfill: the eight facts the leak produced must be reachable AND graded."""
    by_id = facts_by_id()
    for fid in LEAK_COHORT:
        f = by_id[fid]
        assert f.get("experiment") == "EXP-0019", (
            f"{fid} came from EXP-0019 but names no experiment - a void cannot reach it"
        )
        assert f.get("validity") in {"INVALID", "SUSPECT"}, (
            f"{fid} derives from a VOID experiment but reads as {f.get('validity')}"
        )


def test_gate_rejects_a_clean_fact_hanging_off_a_voided_experiment():
    """THE RETRACTION MECHANISM. Plant the exact 2026-08-26 state: a MEASURED fact that
    came from EXP-0019 and was never marked. That is what the registry looked like for a
    day while eight packets were built on it, and it passed every gate."""
    doc = load(FACTS)
    target = next(f for f in doc["facts"] if f["id"] == "FACT-0131")
    target.pop("validity", None)
    target.pop("validity_reason", None)
    assert target["experiment"] == "EXP-0019"
    assert target["provenance"] == "MEASURED", "the leak facts were MEASURED, and deserved it"

    with Planted(FACTS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, (
            "gate accepted a VALID fact derived from a VOID experiment - "
            "retraction does not propagate"
        )
        assert "R8" in result.stdout and "FACT-0131" in result.stdout

    assert run_gate().returncode == 0, "gate did not recover after the probe was removed"


def test_voiding_an_experiment_retracts_its_facts_and_only_its_facts():
    """Void a live experiment and the gate must immediately name its derived facts.

    EXP-0009 is the deployed champion and FACT-0001 is its LB score. This also pins the
    LINK DIRECTION: EXP-0019 lists FACT-0111 in its `facts:` (an INPUT that motivated the
    run), and voiding must NOT retract that - it closed LEVER-0003 on unrelated evidence.
    """
    doc = load(EXPERIMENTS)
    exp = next(e for e in doc["experiments"] if e["id"] == "EXP-0009")
    exp["status"] = "void"

    with Planted(EXPERIMENTS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "voiding an experiment did not reach its facts"
        assert "R8" in result.stdout and "FACT-0001" in result.stdout
        assert "FACT-0111" not in result.stdout, (
            "a void retracted an INPUT fact - R8 must walk facts[].experiment, "
            "never experiments[].facts"
        )

    assert run_gate().returncode == 0, "gate did not recover after the probe was removed"


def test_a_cancelled_experiment_retracts_too():
    """`cancelled` is the quieter sibling of `void` - a partial run is not a result."""
    doc = load(EXPERIMENTS)
    exp = next(e for e in doc["experiments"] if e["id"] == "EXP-0009")
    exp["status"] = "cancelled"
    with Planted(EXPERIMENTS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "a cancelled experiment left a clean fact behind"
        assert "R8" in result.stdout


# ======================================================================================
# R9 -- no lever may be closed or supported by an INVALID fact
# ======================================================================================

def test_no_lever_rests_on_an_invalid_fact_as_committed():
    validity = {f["id"]: f.get("validity", "VALID") for f in load(FACTS)["facts"]}
    for lever in load(LEVERS)["levers"]:
        for fid in (lever.get("closed_by") or []) + (lever.get("supporting") or []):
            assert validity.get(fid) != "INVALID", (
                f"{lever['id']} rests on INVALID {fid}"
            )


def test_gate_rejects_a_lever_closed_on_an_invalid_fact():
    """The real 2026-08-26 state: LEVER-0012 killed by FACT-0131, which came from the leak.
    It read as settled science and shut a lane that may be open (FACT-0221)."""
    doc = load(LEVERS)
    lever = next(l for l in doc["levers"] if l["id"] == "LEVER-0012")
    lever["status"] = "killed"
    lever["closed_by"] = ["FACT-0131"]

    with Planted(LEVERS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "gate accepted a lever closed on an INVALID fact"
        assert "R9" in result.stdout and "LEVER-0012" in result.stdout

    assert run_gate().returncode == 0, "gate did not recover after the probe was removed"


def test_gate_rejects_a_lever_supported_by_an_invalid_fact():
    doc = load(LEVERS)
    lever = next(l for l in doc["levers"] if l["id"] == "LEVER-0011")
    lever["supporting"] = list(lever.get("supporting") or []) + ["FACT-0130"]

    with Planted(LEVERS) as p:
        p.write(doc)
        result = run_gate()
        assert result.returncode != 0, "gate accepted a lever supported by an INVALID fact"
        assert "R9" in result.stdout and "FACT-0130" in result.stdout

    assert run_gate().returncode == 0, "gate did not recover after the probe was removed"


def test_a_suspect_closure_is_noted_not_failed(clean_gate):
    """LEVER-0015 is killed by two SUSPECT facts whose GT-side half genuinely survives.
    Failing that would force us to either lie about the fact or fabricate a clean one, so
    it is a NOTE - visible on every run, blocking nothing."""
    assert "R9 LEVER-0015 is closed by SUSPECT" in clean_gate.stdout


def test_gate_notes_a_live_packet_taking_tainted_input():
    """A packet is where an invalid fact turns into GPU hours - eight of them did."""
    probe = REG / "packets" / "PKT-9998.yaml"
    probe.write_text(
        yaml.safe_dump({
            "id": "PKT-9998",
            "title": "tainted-input probe",
            "lever": None,
            "owner": "probe",
            "lock": "running",
            "inputs": {"facts": ["FACT-0130"]},
        }),
        encoding="utf-8",
    )
    try:
        result = run_gate()
        assert "PKT-9998 is live and takes INVALID FACT-0130" in result.stdout
        assert result.returncode == 0, "the packet-input check is a note, not a failure"
    finally:
        probe.unlink()


# ======================================================================================
# R10 -- protocol provenance: the leaky and honest facts must be distinguishable
# ======================================================================================

def test_the_leaky_and_honest_fold1_facts_differ_in_the_registry():
    """THE THIRD GAP. Both manifests were byte-identical in fold, arm, n_crops,
    det_threshold, secondary_enabled, deepcenter_enabled and experiment_tag. Only `weights`
    differed, and no fact recorded weights - so two measurements of the same quantity, one
    leaked and one honest, were indistinguishable to any reader."""
    by_id = facts_by_id()
    leaky = by_id["FACT-0130"]["protocol"]["weights"]
    honest = by_id["FACT-0161"]["protocol"]["weights"]
    assert leaky != honest, "protocol does not separate the leaked run from the honest one"
    assert "split_0" in leaky and "6bba" in leaky
    assert "split_1" in honest


def test_gate_reports_the_protocol_and_experiment_adoption_backlog(clean_gate):
    """R10 is adopted incrementally, so it must REPORT rather than fail - but it must
    report, on every run, or the backlog is invisible and adoption never happens."""
    assert "R10" in clean_gate.stdout
    assert "name no `experiment`" in clean_gate.stdout
    assert "name no `protocol.weights`" in clean_gate.stdout


def test_validity_is_reported_in_the_summary(clean_gate):
    """An axis nobody sees is an axis nobody maintains."""
    assert "validity:" in clean_gate.stdout
    assert "INVALID" in clean_gate.stdout and "SUSPECT" in clean_gate.stdout
