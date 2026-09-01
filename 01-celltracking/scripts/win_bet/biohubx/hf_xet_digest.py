"""Fail-closed Hugging Face/Xet digest binding for large released artifacts.

On Xet-backed repositories, ``x-linked-etag`` identifies Xet content and is not necessarily the
SHA-256 of the downloaded bytes.  The pinned repository API tree's ``lfs.oid`` is the authoritative
SHA-256.  Treating the header as the expected file digest creates a false refusal; dropping digest
verification to make that refusal disappear deletes the real guard.  This module makes the source
choice mechanical and keeps the rejected header in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEARTBEAT = "HF_XET_DIGEST_OK"


class DigestRefusal(RuntimeError):
    pass


def _normalise(value: object) -> str:
    return str(value or "").strip().strip('"').lower()


def authoritative_digest(api_entry: Mapping[str, object],
                         response_headers: Mapping[str, object] | None = None) -> dict:
    """Return a source-labelled SHA-256; never promote an HTTP/Xet header to authority."""
    lfs = api_entry.get("lfs")
    if not isinstance(lfs, Mapping):
        raise DigestRefusal("pinned API tree entry carries no lfs object")
    oid = _normalise(lfs.get("oid"))
    if not SHA256.fullmatch(oid):
        raise DigestRefusal(f"pinned API tree lfs.oid is not a SHA-256: {oid!r}")
    headers = {str(k).lower(): _normalise(v) for k, v in (response_headers or {}).items()}
    linked = headers.get("x-linked-etag", "")
    etag = headers.get("etag", "")
    return {
        "expected_sha256": oid,
        "digest_source": "PINNED_HUGGINGFACE_API_TREE_LFS_OID",
        "x_linked_etag": linked or None,
        "etag": etag or None,
        "header_matches_file_sha256": linked == oid if linked else None,
        "headers_are_authoritative": False,
    }


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    p = Path(path)
    if not p.is_file():
        raise DigestRefusal(f"downloaded artifact is absent: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download(path: str | Path, api_entry: Mapping[str, object],
                    response_headers: Mapping[str, object] | None = None) -> dict:
    identity = authoritative_digest(api_entry, response_headers)
    measured = sha256_file(path)
    if measured != identity["expected_sha256"]:
        raise DigestRefusal(
            f"downloaded bytes hash to {measured}; pinned API lfs.oid is "
            f"{identity['expected_sha256']}. Keep the partial file and refuse")
    return {
        "file": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "measured_sha256": measured,
        **identity,
        "digest_match": True,
        "heartbeat": HEARTBEAT,
    }


def find_api_entry(tree: object, repo_path: str) -> Mapping[str, object]:
    entries = tree if isinstance(tree, list) else tree.get("siblings", []) \
        if isinstance(tree, Mapping) else []
    hits = [entry for entry in entries if isinstance(entry, Mapping)
            and str(entry.get("path", entry.get("rfilename", ""))) == repo_path]
    if len(hits) != 1:
        raise DigestRefusal(
            f"expected exactly one pinned API tree entry for {repo_path!r}, found {len(hits)}")
    return hits[0]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True)
    parser.add_argument("--api-tree-json", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--headers-json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tree = json.loads(Path(args.api_tree_json).read_text(encoding="utf-8"))
    headers = json.loads(Path(args.headers_json).read_text(encoding="utf-8")) \
        if args.headers_json else {}
    payload = verify_download(args.file, find_api_entry(tree, args.repo_path), headers)
    _atomic_json(Path(args.out), payload)
    print(HEARTBEAT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

