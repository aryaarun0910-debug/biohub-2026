from __future__ import annotations

import hashlib

import pytest

from scripts.win_bet.biohubx.hf_xet_digest import (
    DigestRefusal,
    authoritative_digest,
    find_api_entry,
    verify_download,
)


def test_xet_header_is_recorded_but_api_lfs_oid_is_authoritative(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"released tensor bytes")
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    xet = "1" * 64
    receipt = verify_download(
        artifact,
        {"path": "weights.bin", "lfs": {"oid": sha}},
        {"x-linked-etag": f'"{xet}"'},
    )
    assert receipt["digest_match"] is True
    assert receipt["expected_sha256"] == sha
    assert receipt["x_linked_etag"] == xet
    assert receipt["header_matches_file_sha256"] is False
    assert receipt["headers_are_authoritative"] is False


def test_loosened_or_missing_api_digest_refuses_even_with_sha_like_header(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"x")
    header = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with pytest.raises(DigestRefusal, match="no lfs object"):
        verify_download(artifact, {"path": "weights.bin"}, {"x-linked-etag": header})


def test_wrong_download_refuses_against_lfs_oid(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"wrong")
    with pytest.raises(DigestRefusal, match="pinned API lfs.oid"):
        verify_download(artifact, {"lfs": {"oid": "a" * 64}})


def test_malformed_oid_and_ambiguous_tree_refuse():
    with pytest.raises(DigestRefusal, match="not a SHA-256"):
        authoritative_digest({"lfs": {"oid": "abc"}})
    tree = [
        {"path": "w", "lfs": {"oid": "a" * 64}},
        {"path": "w", "lfs": {"oid": "b" * 64}},
    ]
    with pytest.raises(DigestRefusal, match="exactly one"):
        find_api_entry(tree, "w")
