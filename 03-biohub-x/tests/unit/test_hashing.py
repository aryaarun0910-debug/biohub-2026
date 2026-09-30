"""The identity contract. Everything the repository asserts rests on this."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from biohubx.hashing import (
    CANONICALIZATION_VERSION,
    Digest,
    DigestKind,
    NotTextError,
    canonical_text_digest,
    canonical_text_digest_file,
    canonicalize_text,
    digest_file,
    raw_digest,
    raw_digest_file,
)


def test_raw_digest_is_sha256_of_the_exact_bytes() -> None:
    data = b"cell 17 divides at t=42\n"
    assert raw_digest(data).hexdigest == hashlib.sha256(data).hexdigest()


def test_canonical_text_digest_is_sha256_of_the_canonical_bytes() -> None:
    data = b"a\r\nb\r\n"
    expected = hashlib.sha256(canonicalize_text(data)).hexdigest()
    assert canonical_text_digest(data).hexdigest == expected


# --- canonicalization v1 rules, one test per rule -------------------------


def test_crlf_and_lf_have_different_raw_identity_but_the_same_canonical_identity() -> None:
    crlf = b"alpha\r\nbeta\r\n"
    lf = b"alpha\nbeta\n"
    assert raw_digest(crlf) != raw_digest(lf)
    assert canonical_text_digest(crlf) == canonical_text_digest(lf)


def test_lone_carriage_return_is_normalised() -> None:
    assert canonicalize_text(b"alpha\rbeta") == b"alpha\nbeta\n"


def test_leading_byte_order_mark_is_stripped() -> None:
    assert canonicalize_text("\ufeffalpha\n".encode()) == b"alpha\n"
    assert canonical_text_digest("\ufeffalpha\n".encode()) == canonical_text_digest(b"alpha\n")


def test_missing_trailing_newline_is_appended() -> None:
    assert canonicalize_text(b"alpha") == b"alpha\n"


def test_empty_content_stays_empty() -> None:
    assert canonicalize_text(b"") == b""


def test_trailing_whitespace_and_blank_lines_are_preserved() -> None:
    # Canonicalization removes platform convention, not content. If it stripped
    # these, a real edit would be invisible to drift detection.
    assert canonicalize_text(b"alpha   \n\n\n") == b"alpha   \n\n\n"


def test_non_utf8_bytes_have_a_raw_identity_but_no_canonical_identity() -> None:
    data = b"\xff\xfe\x00weights"
    assert raw_digest(data).hexdigest == hashlib.sha256(data).hexdigest()
    with pytest.raises(NotTextError):
        canonical_text_digest(data)


# --- tokens declare what they are -----------------------------------------


def test_raw_token_declares_its_kind_and_carries_no_canonicalization() -> None:
    token = raw_digest(b"x").token
    assert token.startswith("raw_artifact_sha256:sha256:")
    assert "/v" not in token


def test_canonical_token_declares_its_canonicalization_version() -> None:
    token = canonical_text_digest(b"x").token
    assert token.startswith(f"canonical_text_sha256:sha256/{CANONICALIZATION_VERSION}:")


def test_tokens_round_trip() -> None:
    for digest in (raw_digest(b"x"), canonical_text_digest(b"x")):
        assert Digest.parse(digest.token) == digest


@pytest.mark.parametrize(
    "token",
    [
        "0" * 64,  # bare hex: says nothing about what it is a digest of
        "sha256:" + "0" * 64,  # algorithm only, no kind
        "raw_artifact_sha256:sha256:" + "0" * 63,  # wrong length
        "raw_artifact_sha256:sha256:" + "G" * 64,  # not hex
        "raw_artifact_sha256:sha256:" + "0" * 64 * 2,
        "unknown_kind:sha256:" + "0" * 64,
        "",
    ],
)
def test_untyped_or_malformed_digests_are_refused(token: str) -> None:
    with pytest.raises(ValueError):
        Digest.parse(token)


def test_raw_and_canonical_tokens_are_not_interchangeable() -> None:
    # Content already in canonical form, so the two hexdigests coincide. This is
    # the case that matters: the only thing separating the two identities here
    # is the declared kind, and it must be enough.
    already_canonical = b"x\n"
    raw = raw_digest(already_canonical)
    canonical = canonical_text_digest(already_canonical)
    assert raw.hexdigest == canonical.hexdigest
    assert raw != canonical
    assert raw.token != canonical.token
    assert Digest.parse(raw.token).kind is DigestKind.RAW_ARTIFACT
    assert Digest.parse(canonical.token).kind is DigestKind.CANONICAL_TEXT


def test_a_raw_digest_may_not_claim_a_canonicalization() -> None:
    with pytest.raises(ValueError):
        Digest(kind=DigestKind.RAW_ARTIFACT, hexdigest="0" * 64, canonicalization="v1")


def test_a_canonical_digest_must_declare_a_canonicalization() -> None:
    with pytest.raises(ValueError):
        Digest(kind=DigestKind.CANONICAL_TEXT, hexdigest="0" * 64)


def test_uppercase_hexdigest_is_refused() -> None:
    with pytest.raises(ValueError):
        Digest(kind=DigestKind.RAW_ARTIFACT, hexdigest="A" * 64)


# --- file-level identity ---------------------------------------------------


def test_streamed_file_digest_matches_in_memory_digest(tmp_path: Path) -> None:
    # The streaming path exists so weights never need to fit in memory; it must
    # agree with the in-memory path exactly.
    data = b"x" * ((1 << 20) + 7)
    target = tmp_path / "big.bin"
    target.write_bytes(data)
    assert raw_digest_file(target) == raw_digest(data)


def test_canonical_file_digest_matches_in_memory_digest(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_bytes(b"depth: 4\r\n")
    assert canonical_text_digest_file(target) == canonical_text_digest(b"depth: 4\r\n")


def test_digest_file_requires_an_explicit_kind(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"a\n")
    assert digest_file(target, DigestKind.RAW_ARTIFACT).kind is DigestKind.RAW_ARTIFACT
    assert digest_file(target, DigestKind.CANONICAL_TEXT).kind is DigestKind.CANONICAL_TEXT
