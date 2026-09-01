"""Inventory every digest in the repository and give each one a declared hash kind.

WHY THIS IS A GENERATED INVENTORY AND NOT A LIST IN A DOCUMENT
--------------------------------------------------------------
There are 816 distinct 64-hex constants across the tracked tree and 131 `hashlib.sha256` call
sites in 74 files. A hand-written table of those would be stale before it was reviewed. This
instrument re-derives the inventory, and its classification comes from a DECLARED table keyed by
path pattern and field name - so the judgement is reviewable in one place while the enumeration
stays mechanical.

THE MEASUREMENT THAT MADE IT NECESSARY
--------------------------------------
303 of 820 tracked files differ between the working tree and a fresh checkout by line endings
alone, and 232 of the 816 recorded constants match the WORKTREE bytes only. Those digests are
statements about a checkout rather than about content.

WHAT IT REFUSES TO DO
---------------------
It does not reclassify anything on its own. A constant whose intended mode cannot be established
from the declaration table is `LEGACY_UNTYPED_SHA256` with a migration status - never guessed
into RAW or CANONICAL because of where it happens to sit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

import hashing as H  # noqa: E402

HEARTBEAT_OK = "DIGEST_INVENTORY_OK"
HEARTBEAT_DRIFT = "DIGEST_INVENTORY_DRIFT"
OUT = REPO / "research" / "00-system" / "registry" / "generated" / "digest_inventory.json"

_HEX = re.compile(r"\b[0-9a-f]{64}\b")
SCAN_SUFFIX = (".py", ".json", ".yaml", ".yml", ".md")

#: The inventory must NOT scan its own output. The generated tree contains digests the inventory
#: itself produced, so including it made every regeneration discover more digests than the last -
#: the count grew 826 -> 966 in one cycle and the drift lock could never settle. An inventory of
#: SOURCE digests is what has meaning; a generated file's digests are derived.
SCAN_EXCLUDE = ("research/00-system/registry/generated/", "knowledge/")

#: Artifacts legitimately NOT tracked - release receipts, fetched submissions, weights. Their
#: digests must still be classifiable, so the resolver looks for them on disk too. Without this
#: they fell through to LEGACY for the wrong reason and RAW_ARTIFACT_SHA256 counted ZERO, which
#: is precisely the classification the raw kind exists to carry.
UNTRACKED_PROBE_GLOBS = ("notebooks/*/_out/*.json", "weights/*", "artifacts/kaggle/**/*.pth")

# ==============================================================================================
# THE DECLARATION TABLE. This is the reviewable judgement; everything else is derived.
# Order matters - the first matching rule wins.
# ==============================================================================================
DECLARATIONS: list[dict] = [
    {"match": r"^notebooks/[^/]+/_out/", "kind": H.RAW,
     "why": "release artifacts and fetched submissions: the bytes ARE the identity, because those "
            "exact bytes were uploaded and scored"},
    {"match": r"\.(pth|pt|ckpt|safetensors|bin)$", "kind": H.RAW,
     "why": "downloaded weights, bound against a publisher's declared digest (FACT-0460, "
            "FACT-0461). Normalising here would make the binding uncheckable"},
    {"match": r"^notebooks/[^/]+/build_manifest\.json$", "kind": H.CANONICAL,
     "why": "CORRECTED in Phase 1.5. A build manifest is TRACKED REPOSITORY METADATA, not a "
            "shipped artifact - the artifact it describes is bound separately by the "
            "`built_sha256` value RECORDED inside it, which stays raw. Classifying the manifest "
            "FILE as raw made every consumer of its digest checkout-dependent"},
    {"match": r"^notebooks/.*\.ipynb$", "kind": "DUAL",
     "why": "a built notebook needs BOTH: raw for what was pushed to Kaggle, canonical for source "
            "comparison across checkouts"},
    {"match": r"^scripts/kaggle_specs/.*\.json$", "kind": H.CANONICAL,
     "why": "a spec is SOURCE. Two checkouts of the same spec differ only in line endings and "
            "must compare equal"},
    {"match": r"^(scripts|tests|src)/.*\.py$", "kind": H.CANONICAL,
     "why": "source code; line endings are not semantically meaningful"},
    {"match": r"^data/d1_factorial/.*\.json$", "kind": H.CANONICAL,
     "why": "byte-reproducibility manifests, already pinned to eol=lf in .gitattributes - the one "
            "place the repository had already noticed this hazard"},
    {"match": r"^research/.*\.(md|yaml)$", "kind": H.CANONICAL,
     "why": "registry and research source text"},
]


def declared_kind(rel: str) -> tuple[str, str]:
    for rule in DECLARATIONS:
        if re.search(rule["match"], rel):
            return rule["kind"], rule["why"]
    return H.LEGACY, "no declaration rule matches this path; the intended mode is not established"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                         timeout=180)
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def producers() -> list[dict]:
    """Every call site that computes a digest, so a consumer can be traced to one."""
    rows = []
    for rel in tracked_files():
        if not rel.endswith(".py") or "__pycache__" in rel:
            continue
        p = REPO / rel
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "hashlib.sha256" in line or "hashlib.md5" in line:
                rows.append({"path": rel, "line": i, "code": line.strip()[:120]})
    return rows


def build() -> dict:
    tracked = tracked_files()
    # what each tracked file hashes to, under each identity
    raw_map: dict[str, str] = {}
    canon_map: dict[str, str] = {}
    for rel in tracked:
        p = REPO / rel
        if not p.is_file():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        raw_map.setdefault(hashlib.sha256(data).hexdigest(), rel)
        if p.suffix.lower() not in H.BINARY_SUFFIXES:
            try:
                canon_map.setdefault(H.canonical_text_sha256_bytes(data), rel)
            except H.HashRefusal:
                pass

    # UNTRACKED BUT PRESENT artifacts, so a gitignored release receipt or checkpoint still gets a
    # RAW classification instead of falling through to LEGACY for the wrong reason.
    for pattern in UNTRACKED_PROBE_GLOBS:
        for q in REPO.glob(pattern):
            if q.is_file():
                try:
                    raw_map.setdefault(hashlib.sha256(q.read_bytes()).hexdigest(),
                                       q.relative_to(REPO).as_posix())
                except OSError:
                    pass

    seen: dict[str, dict] = {}
    for rel in tracked:
        p = REPO / rel
        if (p.suffix.lower() not in SCAN_SUFFIX or "__pycache__" in rel or not p.is_file()
                or any(rel.startswith(x) for x in SCAN_EXCLUDE)):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for h in _HEX.findall(line):
                rec = seen.setdefault(h, {"digest": h, "cited_at": [], "resolves_to": None,
                                          "resolution": None})
                if len(rec["cited_at"]) < 6:
                    rec["cited_at"].append(f"{rel}:{i}")

    for h, rec in seen.items():
        if h in raw_map:
            rec["resolves_to"] = raw_map[h]
            rec["resolution"] = "raw bytes of a tracked file (as it stands in THIS worktree)"
        elif h in canon_map:
            rec["resolves_to"] = canon_map[h]
            rec["resolution"] = "canonical text of a tracked file"
        else:
            rec["resolution"] = ("no tracked file - external artifact, historical evidence, or a "
                                 "derived/structured digest")
        target = rec["resolves_to"]
        kind, why = declared_kind(target) if target else (
            H.LEGACY, "the digest resolves to no tracked file, so its mode cannot be derived")
        rec["declared_kind"] = kind
        rec["declaration_basis"] = why
        if target and kind in ("DUAL", H.CANONICAL):
            p = REPO / target
            try:
                rec["canonical_sha256"] = H.canonical_text_sha256(p)
                rec["checkout_sensitive"] = (rec["resolution"].startswith("raw bytes")
                                             and rec["canonical_sha256"] != h)
            except (H.HashRefusal, OSError):
                rec["canonical_sha256"] = None
        else:
            rec["checkout_sensitive"] = None
        if kind == H.LEGACY:
            rec["migration_status"] = "PRESERVED_UNTYPED"
            rec["compatibility"] = ("kept exactly as recorded. It is NOT reinterpreted as raw or "
                                    "canonical on the strength of where it sits")

    counts = {"distinct_digests": len(seen)}
    for k in ("DUAL", H.RAW, H.CANONICAL, H.STRUCTURED, H.LEGACY):
        counts[k] = sum(1 for r in seen.values() if r["declared_kind"] == k)
    counts["checkout_sensitive"] = sum(1 for r in seen.values() if r.get("checkout_sensitive"))

    return {
        "generated_by": "scripts/core/digest_inventory.py",
        "hash_algorithm": H.HASH_ALGORITHM,
        "canonicalization_version": H.CANONICALIZATION_VERSION,
        "structured_version": H.STRUCTURED_VERSION,
        "declaration_table": DECLARATIONS,
        "counts": counts,
        "n_producer_call_sites": len(producers()),
        "digests": dict(sorted(seen.items())),
    }


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    payload = build()
    text = canonical_json(payload)
    if args.check:
        if not args.out.is_file():
            print(f"{HEARTBEAT_DRIFT} {args.out} missing")
            return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(f"{HEARTBEAT_DRIFT} {args.out.name} differs from a fresh generation")
            return 1
        print(f"  digest inventory in sync: {payload['counts']}")
        print(HEARTBEAT_OK)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"  wrote {args.out.relative_to(REPO).as_posix()}")
    for k, v in payload["counts"].items():
        print(f"    {k:<28} {v}")
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
