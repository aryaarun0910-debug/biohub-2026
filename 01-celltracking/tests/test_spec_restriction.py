"""FACT-0451 at the spec schema boundary: the four fail-closed rules, proved by mutation.

FACT-0451 requires MUTATION COVERAGE, "each proving a rejection rather than asserting one", for
four specific defects: omission of the field; a false-to-true override; a ROLE RENAME OVER THE
SAME HASH; and the harness DROPPING the field from its output. Each has a test below whose name
says which one it is, and each first proves the CLEAN case passes - a suite where everything
refuses is not evidence that the refusals are aimed.

These are software contracts only (CLAUDE.md rule 4). Nothing here promotes or demotes a result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import spec_restriction as sr  # noqa: E402

RESTRICTED_SHA = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"


def _clean_spec(**over):
    spec = {
        "name": "unit_clean",
        "table": "C:/temp/assoc/f0.parquet",
        "binding_restriction": {"fact": "FACT-0451", "offline_scoreable": True},
    }
    spec.update(over)
    return spec


def _fake_restricted_checkpoint(tmp_path: Path, name: str) -> Path:
    """A file whose sha256 IS the restricted digest.

    We cannot ship general_v1.pt into a test, and we must not depend on it being on disk, so the
    test monkeypatches the registry to the digest of bytes it controls. What is under test is the
    RESOLUTION MECHANISM - hash the file, look the hash up - not the literal constant.
    """
    p = tmp_path / name
    p.write_bytes(b"pretend these are the released tensors")
    return p


@pytest.fixture()
def registry_of(monkeypatch):
    """Point RESTRICTED at the digest of a file the test wrote."""
    def _register(path: Path, name: str = "general_v1.pt"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setattr(sr, "RESTRICTED", {digest: {
            "name_at_registration": name,
            "basis": "FACT-0451",
            "verdict": "training data CANNOT BE ESTABLISHED",
            "restriction": "SUBMISSION_ONLY_JUDGEMENT",
        }})
        monkeypatch.setattr(sr, "PENDING_PREFIX_SUFFIX", [])
        return digest
    return _register


# ------------------------------------------------------------------------------------------
# The clean case must PASS, or every refusal below is uninformative.
# ------------------------------------------------------------------------------------------
def test_clean_spec_passes_and_is_promotable():
    res = sr.enforce_spec(_clean_spec())
    assert res["offline_scoreable"] is True
    assert res["derived"] is True
    assert res["restricted_hits"] == []
    sr.assert_promotable(res, what="clean arm")          # must not raise


# ------------------------------------------------------------------------------------------
# MUTATION 1 - omission of the field (rule 2)
# ------------------------------------------------------------------------------------------
def test_mutation_omitted_field_refuses():
    spec = _clean_spec()
    del spec["binding_restriction"]
    with pytest.raises(sr.RestrictionRefusal, match="must DECLARE"):
        sr.enforce_spec(spec)


def test_mutation_block_present_but_field_absent_refuses():
    spec = _clean_spec(binding_restriction={"fact": "FACT-0451", "text": "restricted"})
    with pytest.raises(sr.RestrictionRefusal, match="must DECLARE"):
        sr.enforce_spec(spec)


def test_mutation_non_bool_declaration_refuses():
    """'false' as a STRING is truthy. That is how a false becomes a true without anyone lying."""
    spec = _clean_spec(binding_restriction={"offline_scoreable": "false"})
    with pytest.raises(sr.RestrictionRefusal, match="must be a bool"):
        sr.enforce_spec(spec)


# ------------------------------------------------------------------------------------------
# MUTATION 2 - a false-to-true override (rule 3)
# ------------------------------------------------------------------------------------------
def test_mutation_false_to_true_override_refuses(tmp_path, registry_of):
    ckpt = _fake_restricted_checkpoint(tmp_path, "general_v1.pt")
    registry_of(ckpt)
    spec = _clean_spec(cache={"trunk": str(ckpt)})
    spec["binding_restriction"]["offline_scoreable"] = True   # the override
    with pytest.raises(sr.RestrictionRefusal, match="Derived false stands"):
        sr.enforce_spec(spec)


def test_declaring_false_over_a_restricted_hash_is_accepted_but_not_promotable(
        tmp_path, registry_of):
    """The honest declaration is admissible. What it may not do is PROMOTE (rule 4)."""
    ckpt = _fake_restricted_checkpoint(tmp_path, "general_v1.pt")
    registry_of(ckpt)
    spec = _clean_spec(cache={"trunk": str(ckpt)})
    spec["binding_restriction"]["offline_scoreable"] = False
    res = sr.enforce_spec(spec)
    assert res["offline_scoreable"] is False
    assert res["restriction"] == "SUBMISSION_ONLY_JUDGEMENT"
    assert [h["name_at_registration"] for h in res["restricted_hits"]] == ["general_v1.pt"]
    with pytest.raises(sr.RestrictionRefusal, match="SUBMISSION-ONLY JUDGEMENT"):
        sr.assert_promotable(res, what="general_v1 arm")


# ------------------------------------------------------------------------------------------
# MUTATION 3 - a ROLE RENAME OVER THE SAME HASH (rule 1)
# ------------------------------------------------------------------------------------------
def test_mutation_role_rename_over_the_same_hash_still_refuses(tmp_path, registry_of):
    """FACT-0435 records one checkpoint at two paths under two role names. Bytes are identity."""
    original = _fake_restricted_checkpoint(tmp_path, "general_v1.pt")
    registry_of(original)
    renamed = tmp_path / "our_own_local_trunk_v3.pt"
    renamed.write_bytes(original.read_bytes())               # identical bytes, innocent name
    assert "general" not in renamed.name

    spec = _clean_spec(cache={"trunk": str(renamed)})
    spec["binding_restriction"]["offline_scoreable"] = True
    with pytest.raises(sr.RestrictionRefusal, match="renaming the file does not clear it"):
        sr.enforce_spec(spec)

    spec["binding_restriction"]["offline_scoreable"] = False
    res = sr.enforce_spec(spec)
    assert res["restricted_hits"][0]["name_at_registration"] == "general_v1.pt"
    assert res["restricted_hits"][0]["value"].endswith("our_own_local_trunk_v3.pt")


def test_declared_digest_alone_is_enough_even_with_the_file_absent(monkeypatch):
    """The digest travelling in the document is identity too - the file need not be on this box."""
    monkeypatch.setattr(sr, "PENDING_PREFIX_SUFFIX", [])
    spec = _clean_spec(provenance={"weights_sha256": RESTRICTED_SHA})
    with pytest.raises(sr.RestrictionRefusal, match="Derived false stands"):
        sr.enforce_spec(spec)


def test_elided_digest_matches_on_prefix_and_suffix():
    """general_v0's middle is not recorded anywhere, so the guard matches the recorded ends."""
    entry = sr.PENDING_PREFIX_SUFFIX[0]
    middle = "0" * (64 - len(entry["prefix"]) - len(entry["suffix"]))
    sha = entry["prefix"] + middle + entry["suffix"]
    spec = _clean_spec(provenance={"weights_sha256": sha})
    with pytest.raises(sr.RestrictionRefusal, match="Derived false stands"):
        sr.enforce_spec(spec)


# ------------------------------------------------------------------------------------------
# MUTATION 4 - the harness DROPS the field from its output (rule 4, the FACT-0417 shape)
# ------------------------------------------------------------------------------------------
def test_mutation_producer_drops_the_field_refuses_at_promotion():
    res = sr.enforce_spec(_clean_spec())
    del res[sr.FIELD]                                        # the producer stopped emitting it
    with pytest.raises(sr.RestrictionRefusal, match="carries no offline_scoreable"):
        sr.assert_promotable(res, what="arm with a dropped field")


def test_mutation_no_resolution_at_all_refuses_at_promotion():
    with pytest.raises(sr.RestrictionRefusal, match="no FACT-0451 resolution"):
        sr.assert_promotable(None, what="arm with no resolution")


def test_unresolved_checkpoint_reference_refuses_at_promotion_only(tmp_path):
    """A prepared-not-launched spec must LOAD; it must not PROMOTE."""
    spec = _clean_spec(cache={"trunk": str(tmp_path / "not_fetched_yet.pt")})
    res = sr.enforce_spec(spec)                              # declaration-time: no refusal
    assert res["unresolved_checkpoint_refs"] == ["$.cache.trunk"]
    with pytest.raises(sr.RestrictionRefusal, match="could not be resolved"):
        sr.assert_promotable(res, what="prepared arm")


# ------------------------------------------------------------------------------------------
# The committed specs must satisfy the schema the module now enforces.
# ------------------------------------------------------------------------------------------
def test_every_committed_assoc_spec_declares_the_field():
    spec_dir = ROOT / "scripts" / "win_bet" / "assoc_specs"
    paths = sorted(spec_dir.glob("*.json"))
    assert paths, "no association specs found - the guard would be vacuous"
    for p in paths:
        spec = json.loads(p.read_text(encoding="utf-8"))
        block = spec.get(sr.BLOCK)
        assert isinstance(block, dict) and sr.FIELD in block, (
            f"{p.name} does not declare {sr.BLOCK}.{sr.FIELD}; FACT-0451 rule 2 makes omission a "
            f"refusal, so this spec would refuse at load"
        )
        assert isinstance(block[sr.FIELD], bool), f"{p.name}: {sr.FIELD} must be a bool"


# ------------------------------------------------------------------------------------------
# THE VERDICT LAYER BLOCKS (rule 4), through the real harness module
# ------------------------------------------------------------------------------------------
def _payload():
    return {"models": [{"tag": "a", "verdict": {"promotable": True, "blockers": []}},
                       {"tag": "b", "verdict": {"promotable": True, "blockers": []}}]}


def test_a_restricted_arm_cannot_reach_a_promotable_verdict():
    """The rule FACT-0451 asked for: refuse PROMOTION, do not merely label the report."""
    import assoc_train_harness as H
    spec = _clean_spec(provenance={"weights_sha256": RESTRICTED_SHA})
    spec["binding_restriction"]["offline_scoreable"] = False
    res = sr.enforce_spec(spec)
    out = H.apply_restriction_to_payload(_payload(), res)
    assert out["binding_restriction"]["offline_scoreable"] is False
    for m in out["models"]:
        assert m["verdict"]["promotable"] is False
        assert any("SUBMISSION-ONLY JUDGEMENT" in b for b in m["verdict"]["blockers"])
        assert any("general_v1.pt" in b for b in m["verdict"]["blockers"])


def test_an_unrestricted_arm_keeps_its_verdict():
    """A gate that blocks everything proves nothing about aim."""
    import assoc_train_harness as H
    out = H.apply_restriction_to_payload(_payload(), sr.enforce_spec(_clean_spec()))
    assert all(m["verdict"]["promotable"] is True for m in out["models"])


def test_a_dropped_resolution_still_blocks_promotion():
    """The FACT-0417 shape: a consumer trusting a field its producer stopped emitting."""
    import assoc_train_harness as H
    out = H.apply_restriction_to_payload(_payload(), None)
    assert out["binding_restriction"] is None
    # the payload carries no mark, so `assert_promotable` - the reader's gate - must refuse it
    with pytest.raises(sr.RestrictionRefusal, match="no FACT-0451 resolution"):
        sr.assert_promotable(out["binding_restriction"], what="dropped-resolution arm")


def test_the_harness_imports_the_guard_at_module_scope():
    """A local import inside a function is skippable; a module-scope one is not."""
    import ast
    src = (ROOT / "scripts" / "win_bet" / "assoc_train_harness.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
    assert "spec_restriction" in top, (
        "assoc_train_harness must import spec_restriction at MODULE SCOPE so its absence is a "
        "startup error rather than a silently skipped guard")


# ------------------------------------------------------------------------------------------
# THE HALVES MUST NOT SPLIT AGAIN
# ------------------------------------------------------------------------------------------
def test_specs_declaring_the_field_cannot_exist_without_the_module():
    """A fresh clone must not keep the specs while losing the enforcer.

    At HEAD e5db190 this was exactly the state: 0 of 9 specs declared the field and the module was
    absent, so the guard was gone with no error anywhere. The halves are only safe committed
    together, and this test fails if a future change lands one without the other.
    """
    module = ROOT / "scripts" / "win_bet" / "spec_restriction.py"
    declaring = [p for p in sorted((ROOT / "scripts" / "win_bet" / "assoc_specs").glob("*.json"))
                 if sr.BLOCK in p.read_text(encoding="utf-8")]
    if declaring:
        assert module.is_file(), (
            f"{len(declaring)} spec(s) declare {sr.BLOCK} but {module.name} is absent - the "
            f"enforcement halves have been split and the guard is silently gone")


def test_load_spec_is_the_enforcing_entry_point(tmp_path):
    p = tmp_path / "unit.json"
    p.write_text(json.dumps({"name": "u", "binding_restriction": {}}), encoding="utf-8")
    with pytest.raises(sr.RestrictionRefusal):
        sr.load_spec(p)
    p.write_text(json.dumps(_clean_spec()), encoding="utf-8")
    spec, res = sr.load_spec(p)
    assert spec["name"] == "unit_clean" and res["offline_scoreable"] is True
