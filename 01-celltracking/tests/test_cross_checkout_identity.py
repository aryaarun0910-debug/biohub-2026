"""CROSS-CHECKOUT IDENTITY. Canonical hashes must agree where raw hashes must differ.

THE MEASUREMENT THAT FORCED THIS
--------------------------------
2026-09-01: **303 of 820 tracked files (37%) differ between this working tree and a fresh checkout
by LINE ENDINGS ALONE** - the worktree is LF, a checkout is CRLF - and **232 of the 816 distinct
sha256 constants recorded across the repository matched the WORKTREE bytes only**. A digest of a
tracked text file was a statement about a checkout, not about content, and six of the ten
clean-clone test failures traced to exactly that.

WHAT THESE TESTS PROVE, AND WHAT THEY DELIBERATELY DO NOT
---------------------------------------------------------
They prove the CONTRACT, not a happy path:

  * canonical text hashes are identical across LF, CRLF and mixed forms of the same content;
  * RAW hashes DIFFER across those forms - if they did not, canonicalisation would be pointless
    and the two kinds would not be distinguishable;
  * release artifacts and weights keep RAW identity and are never canonicalised;
  * canonical hashing REFUSES binary input rather than returning a number;
  * the consumers that used to compare raw digests of source now select the canonical kind.

NOTHING HERE IS SKIPPED TO OBTAIN AGREEMENT. The line-ending arms are constructed in memory from
real repository files, so no asset can be absent and no guard can quietly pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

import hashing as H  # noqa: E402

#: Real source files, chosen because each is a consumer the migration touched.
SOURCE_PROBES = [
    "scripts/kaggle_edits/kaggle_mount_ladder.py",
    "scripts/kaggle_edits/h1r_adabn_detection_only.py",
    "scripts/kaggle_edits/d1_response_audit.py",
    "scripts/core/hashing.py",
    "scripts/kaggle_specs/p35_dcveto_on_931.json",
]
NOTEBOOK_PROBE = "notebooks/kaggle_p35_dcveto_on_931/biohub-p35-dcveto-on-931.ipynb"


def _arms(data: bytes) -> dict[str, bytes]:
    """The same CONTENT in every line-ending form a checkout can produce."""
    lf = data.replace(b"\r\n", b"\n")
    return {
        "as_stored": data,
        "lf": lf,
        "crlf": lf.replace(b"\n", b"\r\n"),
        "mixed": lf.replace(b"\n", b"\r\n", 3),      # a partially-converted file
    }


# ------------------------------------------------------------------------------------------
# 1. CANONICAL AGREES, RAW DIFFERS
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rel", SOURCE_PROBES)
def test_canonical_hashes_agree_across_every_line_ending_arm(rel):
    p = ROOT / rel
    if not p.is_file():
        pytest.fail(f"{rel} is a committed probe and must exist; a missing probe is not a skip")
    arms = _arms(p.read_bytes())
    digests = {name: H.canonical_text_sha256_bytes(b) for name, b in arms.items()}
    assert len(set(digests.values())) == 1, (
        f"{rel}: canonical digests disagree across line-ending arms {digests}. The whole point of "
        f"CANONICAL_TEXT_SHA256 is that a checkout conversion cannot change it.")


@pytest.mark.parametrize("rel", SOURCE_PROBES)
def test_raw_hashes_differ_where_line_endings_differ(rel):
    """If raw did not differ, the two kinds would be indistinguishable and the split pointless."""
    arms = _arms((ROOT / rel).read_bytes())
    if arms["lf"] == arms["crlf"]:
        pytest.fail(f"{rel} has no line breaks, so it cannot exercise this contract")
    assert H.raw_sha256_bytes(arms["lf"]) != H.raw_sha256_bytes(arms["crlf"]), (
        f"{rel}: raw digests are equal across LF and CRLF, which cannot be true for a file with "
        f"line breaks - the raw path is not hashing bytes")


def test_the_champion_notebook_carries_both_identities_and_they_differ():
    p = ROOT / NOTEBOOK_PROBE
    rec = H.dual_record(p, repo_root=ROOT)
    assert rec["raw"]["hash_kind"] == H.RAW
    assert rec["canonical"]["hash_kind"] == H.CANONICAL
    assert rec["raw"]["measured_sha256"] != rec["canonical"]["measured_sha256"], (
        "the champion notebook is CRLF in this worktree, so its raw and canonical identities MUST "
        "differ; equality would mean one of the two is not being computed")


# ------------------------------------------------------------------------------------------
# 2. RAW IS NEVER REPLACED WHERE BYTES ARE THE IDENTITY
# ------------------------------------------------------------------------------------------
def test_canonical_hashing_refuses_binary_input():
    with pytest.raises(H.HashRefusal, match="binary suffix"):
        H.canonical_text_sha256(ROOT / "weights" / "anything.pth")
    with pytest.raises(H.HashRefusal, match="cannot decode"):
        H.canonical_text_sha256_bytes(b"\x00\x01\xff\xfe binary payload \x80")


def test_release_artifacts_and_weights_are_declared_raw():
    import digest_inventory as DI
    for path, expect in [("notebooks/kaggle_p35_dcveto_on_931/_out/audit_receipt.json", H.RAW),
                         ("weights/model.pth", H.RAW),
                         # CORRECTED in Phase 1.5: a build manifest is tracked repository
                         # metadata, not a shipped artifact. Declaring it RAW made every consumer
                         # of its digest checkout-dependent.
                         ("notebooks/kaggle_p35_dcveto_on_931/build_manifest.json", H.CANONICAL),
                         ("scripts/kaggle_specs/p35_dcveto_on_931.json", H.CANONICAL),
                         ("scripts/win_bet/kaggle_mounts.py", H.CANONICAL)]:
        kind, why = DI.declared_kind(path)
        assert kind == expect, f"{path} declared {kind}, expected {expect} ({why})"


def test_every_declared_kind_is_a_known_kind():
    import digest_inventory as DI
    for rule in DI.DECLARATIONS:
        assert rule["kind"] in (*H.KINDS, "DUAL"), rule
        assert rule["why"].strip(), f"rule {rule['match']} states no reason"


# ------------------------------------------------------------------------------------------
# 3. THE CONSUMERS SELECT THE RIGHT KIND
# ------------------------------------------------------------------------------------------
def test_the_catalog_declares_a_hash_kind_for_every_digest_it_records():
    cat = ROOT / "research" / "00-system" / "registry" / "generated" / "catalog"
    for name in ("scripts", "tests", "specs"):
        payload = json.loads((cat / f"{name}.json").read_text(encoding="utf-8"))[name]
        for key, entry in payload.items():
            d = entry.get("digest")
            assert d is not None, f"{name}/{key} records no digest block"
            if "refused" in d:
                continue
            assert d["hash_kind"] in H.KINDS, f"{name}/{key} has kind {d.get('hash_kind')}"
            if d["hash_kind"] == H.CANONICAL:
                assert d["canonicalization_version"] == H.CANONICALIZATION_VERSION


def test_the_notebook_catalog_carries_both_kinds():
    cat = ROOT / "research" / "00-system" / "registry" / "generated" / "catalog"
    nb = json.loads((cat / "notebooks.json").read_text(encoding="utf-8"))["notebooks"]
    entry = nb["kaggle_p35_dcveto_on_931"]["digest"]
    assert entry["raw"]["hash_kind"] == H.RAW
    assert entry["canonical"]["hash_kind"] == H.CANONICAL


def test_the_ladder_digest_is_canonical_and_survives_conversion():
    """`KM.LADDER_SHA256` was one of the measured clean-clone failures."""
    sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
    import kaggle_mounts as KM
    assert KM.LADDER_SHA256_KIND == H.CANONICAL
    p = ROOT / "scripts" / "kaggle_edits" / "kaggle_mount_ladder.py"
    arms = _arms(p.read_bytes())
    for name, b in arms.items():
        assert H.canonical_text_sha256_bytes(b) == KM.LADDER_SHA256, (
            f"the ladder digest changes under the {name!r} arm")


# ------------------------------------------------------------------------------------------
# 4. THE REAL CHECKOUT, NOT A SIMULATION
# ------------------------------------------------------------------------------------------
def test_canonical_digests_match_the_git_blob_for_lf_normalised_files():
    """The strongest available cross-checkout evidence without making a second clone.

    git stores blobs LF-normalised under `core.autocrlf=true`. So the canonical digest of a
    working-tree file must equal the canonical digest of its blob, whatever the worktree's line
    endings are. If that ever fails, canonicalisation and git's normalisation disagree and the
    contract is not what it claims.
    """
    checked = 0
    for rel in SOURCE_PROBES + [NOTEBOOK_PROBE]:
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(ROOT),
                              capture_output=True, timeout=120)
        if blob.returncode != 0:
            continue
        disk = H.canonical_text_sha256(ROOT / rel)
        assert H.canonical_text_sha256_bytes(blob.stdout) == disk, (
            f"{rel}: the canonical digest of the working tree differs from the canonical digest "
            f"of the git blob, so a checkout conversion WOULD change it")
        checked += 1
    assert checked >= 4, f"only {checked} probes resolved to a blob; the evidence is too thin"
