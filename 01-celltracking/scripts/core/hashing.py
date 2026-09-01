"""THE ONE HASHING MODULE. Two hash kinds, both explicit, neither replacing the other.

THE PROBLEM IT EXISTS FOR
-------------------------
Measured 2026-09-01: **303 of 820 tracked files (37%) differ between this working tree and a fresh
checkout by LINE ENDINGS ALONE** - the worktree is LF, a checkout is CRLF - and **232 of the 816
distinct sha256 constants recorded across the repository match the WORKTREE bytes only**. So a
digest of a tracked text file is a statement about a checkout, not about content. It resolves here
and nowhere else, which is the green-here-red-everywhere shape this campaign keeps paying for.

The champion notebook's pin is one instance: `FACT-0446` records the CRLF form, and the git blob
hashes differently.

WHY TWO KINDS AND NOT ONE
-------------------------
The instinct is to canonicalise everything. That would be wrong, and destructively so:

  * a RELEASE ARTIFACT's identity IS its bytes. `submission.csv` was uploaded byte for byte, and
    a canonicalised digest of it would describe a file nobody submitted.
  * DOWNLOADED WEIGHTS are binary. `FACT-0460` and `FACT-0461` bind FOCUS-3D and the HOCT
    checkpoint by raw sha256 against a publisher's declared digest; normalising bytes there is
    not merely wrong, it is unfalsifiable.

So: RAW where bytes are the identity, CANONICAL where content is the identity, and every record
says which it is. `canonical_text_sha256` NEVER replaces `raw_sha256`; it answers a different
question.

THE FOUR KINDS
--------------
    RAW_ARTIFACT_SHA256        exact bytes - built notebooks as shipped, weights, fetched
                               submissions, release artifacts. NOT build manifests: those are
                               tracked repository metadata and are CANONICAL (Phase 1.5
                               correction); the artifact a manifest describes stays bound by the
                               raw  recorded inside it
    CANONICAL_TEXT_SHA256      source identity, invariant to CRLF/LF checkout conversion
    STRUCTURED_CONTENT_SHA256  an EXISTING convention only. `kaggle_factory:575` and
                               `assemble_p3_d1_smoke_spec:130` already define it as
                               `json.dumps(obj, sort_keys=True)`. Nothing new is invented here -
                               casually asserting that two differently-serialised JSON documents
                               are "the same" is a semantic claim, not a hashing one.
    LEGACY_UNTYPED_SHA256      a recorded digest whose intended mode is NOT mechanically
                               established. It keeps its value and gains a status; it is never
                               silently reinterpreted.

CANONICALISATION, STATED EXACTLY
--------------------------------
`canon_text_v1` is: decode as the DECLARED encoding (default utf-8, strict); refuse if that
fails; strip ONE leading U+FEFF if present and RECORD that it was stripped; replace CRLF with LF,
then any remaining lone CR with LF; change nothing else - no trailing-newline insertion, no
whitespace trimming, no unicode normalisation; re-encode utf-8 and sha256 that.

BOM treatment is stripping rather than preserving because a BOM is a checkout/editor artifact of
the same class as a line ending. It is RECORDED so the decision is visible in the payload rather
than implied by the number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HASH_ALGORITHM = "sha256"
CANONICALIZATION_VERSION = "canon_text_v1"
STRUCTURED_VERSION = "json_sort_keys_v1"

RAW = "RAW_ARTIFACT_SHA256"
CANONICAL = "CANONICAL_TEXT_SHA256"
STRUCTURED = "STRUCTURED_CONTENT_SHA256"
LEGACY = "LEGACY_UNTYPED_SHA256"
KINDS = (RAW, CANONICAL, STRUCTURED, LEGACY)

#: Suffixes whose identity is their BYTES. Canonical hashing REFUSES these rather than producing
#: a number that looks like an answer.
BINARY_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".npz", ".npy", ".parquet",
                   ".png", ".jpg", ".tif", ".tiff", ".zip", ".gz", ".pkl", ".sqlite", ".pyd",
                   ".so", ".dll", ".lib", ".a", ".geff")

BOM = "﻿"


class HashRefusal(RuntimeError):
    """Raised rather than returning a plausible number for input the mode cannot describe."""


# ----------------------------------------------------------------------------------------------
# RAW - the bytes, unchanged. Never removed, never replaced.
# ----------------------------------------------------------------------------------------------
def raw_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# CANONICAL TEXT - content identity, invariant to checkout conversion
# ----------------------------------------------------------------------------------------------
def canonicalize_text(data: bytes, *, encoding: str = "utf-8") -> tuple[str, dict]:
    """Return ``(canonical_text, notes)``. Raises ``HashRefusal`` on undecodable input."""
    try:
        text = data.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise HashRefusal(
            f"cannot decode as {encoding!r}: {exc}. Canonical-text hashing describes SOURCE "
            f"content; binary input has no canonical text form and must use raw_sha256. "
            f"Returning a number here would be an answer to a question that was not asked."
        ) from exc
    notes = {"encoding": encoding, "bom_stripped": False,
             "crlf_count": data.count(b"\r\n"), "lone_cr_count": 0}
    if text.startswith(BOM):
        text = text[len(BOM):]
        notes["bom_stripped"] = True
    normalised = text.replace("\r\n", "\n")
    notes["lone_cr_count"] = normalised.count("\r")
    normalised = normalised.replace("\r", "\n")
    return normalised, notes


def canonical_text_sha256(path: str | Path, *, encoding: str = "utf-8") -> str:
    p = Path(path)
    if p.suffix.lower() in BINARY_SUFFIXES:
        raise HashRefusal(
            f"{p.name} has a binary suffix; its identity is its BYTES. Use raw_sha256. "
            f"FACT-0460 and FACT-0461 bind checkpoints against a publisher's raw digest, and a "
            f"normalised digest there could never be checked against anything."
        )
    text, _ = canonicalize_text(p.read_bytes(), encoding=encoding)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text_sha256_bytes(data: bytes, *, encoding: str = "utf-8") -> str:
    text, _ = canonicalize_text(data, encoding=encoding)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------
# STRUCTURED - the EXISTING convention, not a new one
# ----------------------------------------------------------------------------------------------
def structured_sha256(obj: Any) -> str:
    """``json.dumps(obj, sort_keys=True)`` - the convention already committed in this repository.

    Defined at `scripts/core/kaggle_factory.py:575` (`payload_sha256`) and
    `scripts/d1/assemble_p3_d1_smoke_spec.py:130` (`canonical_sha256`). It is reproduced here so
    both callers can share one implementation, NOT extended: this module does not claim that two
    JSON documents differing in anything but key order are equivalent.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------
# THE RECORD - every hash-bearing field declares what it is
# ----------------------------------------------------------------------------------------------
def hash_record(path: str | Path, kind: str, *, encoding: str = "utf-8",
                repo_root: Path | None = None) -> dict:
    """The shape every hash-bearing record must carry."""
    if kind not in KINDS:
        raise HashRefusal(f"unknown hash_kind {kind!r}; known: {list(KINDS)}")
    p = Path(path)
    src = p.as_posix()
    if repo_root is not None:
        try:
            src = p.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    rec = {"hash_kind": kind, "hash_algorithm": HASH_ALGORITHM, "source_path": src,
           "canonicalization_version": None, "measured_sha256": None}
    if kind == RAW:
        rec["measured_sha256"] = raw_sha256(p)
    elif kind == CANONICAL:
        rec["canonicalization_version"] = CANONICALIZATION_VERSION
        data = p.read_bytes()
        _, notes = canonicalize_text(data, encoding=encoding)
        rec["measured_sha256"] = canonical_text_sha256_bytes(data, encoding=encoding)
        # A STORED RECORD MUST NOT CARRY A CHECKOUT CENSUS. `crlf_count` and `lone_cr_count`
        # describe the CHECKOUT the record was generated in, so including them made a
        # canonical record checkout-dependent - the exact defect this module exists to remove,
        # reproduced inside it. They stay available from `canonicalize_text` for diagnostics.
        # `encoding` and `bom_stripped` are properties of the CONTENT and survive conversion.
        rec["canonicalization_notes"] = {"encoding": notes["encoding"],
                                         "bom_stripped": notes["bom_stripped"]}
    elif kind == STRUCTURED:
        rec["canonicalization_version"] = STRUCTURED_VERSION
        rec["measured_sha256"] = structured_sha256(json.loads(p.read_text(encoding=encoding)))
    else:
        raise HashRefusal(
            f"{LEGACY} cannot be MEASURED - it is a status for a digest already recorded whose "
            f"mode was never established. Use `legacy_record` to describe one."
        )
    return rec


def dual_record(path: str | Path, *, encoding: str = "utf-8",
                repo_root: Path | None = None) -> dict:
    """Both identities for one artifact. Built notebooks need this: the BYTES were shipped, and
    the CONTENT is what a source comparison is about."""
    raw = hash_record(path, RAW, repo_root=repo_root)
    out = {"source_path": raw["source_path"], "hash_algorithm": HASH_ALGORITHM,
           "raw": {"hash_kind": RAW, "measured_sha256": raw["measured_sha256"]}}
    try:
        can = hash_record(path, CANONICAL, encoding=encoding, repo_root=repo_root)
        out["canonical"] = {"hash_kind": CANONICAL,
                            "canonicalization_version": can["canonicalization_version"],
                            "measured_sha256": can["measured_sha256"],
                            "notes": can["canonicalization_notes"]}
    except HashRefusal as exc:
        out["canonical"] = {"hash_kind": None, "refused": str(exc)[:200]}
    return out


def legacy_record(value: str, *, where: str, why: str) -> dict:
    """Describe an existing digest whose intended mode is not mechanically established.

    It keeps its value. What it gains is a STATUS and an explanation, so nobody later reads it as
    canonical or as raw on the strength of where it happens to sit.
    """
    return {"hash_kind": LEGACY, "hash_algorithm": HASH_ALGORITHM, "measured_sha256": value,
            "canonicalization_version": None, "source_path": where,
            "migration_status": "PRESERVED_UNTYPED",
            "compatibility": why}


def agrees_across_checkouts(path: str | Path) -> bool:
    """Would this file's CANONICAL digest survive a line-ending conversion? Proof, not assertion."""
    data = Path(path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return (canonical_text_sha256_bytes(lf) == canonical_text_sha256_bytes(crlf)
            == canonical_text_sha256_bytes(data))
