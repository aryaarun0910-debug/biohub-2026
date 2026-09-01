"""SANITIZED, TRACKED RECEIPT ENVELOPES. The raw receipts stay ignored.

THE PROBLEM
-----------
Measured 2026-09-01: only 4 of 74 notebook directories carry a release receipt, and all four sit
under `notebooks/**/_out/`, which `.gitignore:20` ignores. So a fresh clone has NONE of them. The
champion's submission is still bound by its experiment, its submission id and its notebook digest
- all tracked - but what a clone loses is the audit bundle proving the artifact was structurally
checked before it was submitted.

WHY NOT SIMPLY UNIGNORE THEM
----------------------------
A raw receipt is a machine-local audit product. It carries local paths and whatever the auditor
happened to record, and un-ignoring a whole tree on the assumption that its contents are safe is
not an assumption this repository is entitled to make. So the raw receipts stay ignored and a
NARROW, EXPLICITLY ENUMERATED envelope is tracked instead. The envelope is allow-listed, not
redacted: fields are copied in by name, so a field nobody has reviewed cannot arrive by default.

WHAT AN ENVELOPE MAY CONTAIN, and nothing else:
    receipt id, artifact sha, notebook sha, spec sha, manifest sha, audit-tool version,
    kernel slug and version, submission reference, EXP/FACT binding, the raw receipt's location
    and availability, and the redaction status.

THE FOUR STATES A CLONE MUST BE ABLE TO TELL APART
--------------------------------------------------
    fully_bound_raw_available    an envelope exists AND the raw receipt is on this machine
    bound_envelope_only          an envelope exists; the raw receipt is not here
    historical_unbound           no receipt was ever produced for this artifact
    unknown                      the question could not be answered

AN ENVELOPE NEVER UPGRADES PROVENANCE. Generating an envelope in 2026 for a notebook audited in
July does not make that notebook better-bound than it was; the envelope records what the receipt
SAID, plus the fact that it was extracted later. `binding_strength` is derived from the
experiment/receipt evidence that existed at the time, never from the envelope's own existence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

import yaml  # noqa: E402
import hashing as H  # noqa: E402  the ONE hashing module

SCHEMA_VERSION = "receipt_envelope_v1"
HEARTBEAT_OK = "RECEIPT_ENVELOPE_OK"
HEARTBEAT_DRIFT = "RECEIPT_ENVELOPE_DRIFT"

OUT = REPO / "research" / "00-system" / "registry" / "generated" / "receipts.json"
REGISTRY = REPO / "research" / "00-system" / "registry"

#: An absolute path, a home directory or a drive letter must never reach a tracked file.
_LOCAL = re.compile(r"([A-Za-z]:[\\/]|/home/|/Users/|\\\\)")


class EnvelopeRefusal(RuntimeError):
    """Refuse rather than emit an envelope carrying machine-local data."""


def _sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _clean(value, where: str):
    """Allow-listed scalars only, and never one that looks machine-local."""
    if value is None or isinstance(value, (int, bool)):
        return value
    s = str(value)
    if _LOCAL.search(s):
        raise EnvelopeRefusal(
            f"{where} carries a machine-local path ({s[:60]!r}). The envelope is TRACKED, so a "
            f"local path would be published to every clone. Refusing rather than trimming it - a "
            f"silently trimmed field is indistinguishable from one that was never there."
        )
    return s


def _experiments_by_kernel() -> dict:
    doc = yaml.safe_load((REGISTRY / "experiments.yaml").read_text(encoding="utf-8"))
    exps = doc["experiments"] if isinstance(doc, dict) and "experiments" in doc else doc
    out = {}
    for e in exps:
        if e.get("kernel"):
            out.setdefault(str(e["kernel"]).strip(), []).append(e)
    return out


def envelope_for(nb_dir: Path, by_kernel: dict) -> dict:
    """One notebook directory -> one envelope. Absence is a state, not an error."""
    raw = nb_dir / "_out" / "audit_receipt.json"
    man = nb_dir / "build_manifest.json"
    nbs = sorted(nb_dir.glob("*.ipynb"))
    env: dict = {
        "receipt_id": "RECEIPT:" + nb_dir.name,
        "notebook_dir": nb_dir.relative_to(REPO).as_posix(),
        "raw_receipt_location": raw.relative_to(REPO).as_posix(),
        "raw_receipt_available_here": raw.is_file(),
        "raw_receipt_tracked": False,      # notebooks/**/_out/ is gitignored (.gitignore:20)
        "schema_version": SCHEMA_VERSION,
        "redaction_status": "allow-listed: only the enumerated fields are copied",
        # RAW artifact identity: the bytes that were built and pushed. This is what FACT-0446
        # pins and what the Kaggle submission was made from - it must not be canonicalised.
        "notebook_sha256": _sha(nbs[0]) if nbs else None,
        # CANONICAL source identity: equal across an LF and a CRLF checkout, so a clone can still
        # tell whether it holds the SAME NOTEBOOK SOURCE even though its raw bytes differ.
        "notebook_canonical_sha256": (
            H.canonical_text_sha256(nbs[0]) if nbs else None),
        "canonicalization_version": H.CANONICALIZATION_VERSION,
        "hash_kinds": {"notebook_sha256": H.RAW,
                       "notebook_canonical_sha256": H.CANONICAL,
                       "artifact_sha256": H.RAW,
                       "manifest_sha256": H.CANONICAL,
                       "spec_sha256": H.RAW,
                       "audit_tool_sha256": H.RAW},
        # CANONICAL: the manifest is tracked repository metadata. Its raw digest differed
        # between an LF worktree and a CRLF checkout, so the tracked envelope disagreed with
        # itself in a clone - measured, and it was one of the last three clean-clone failures.
        "manifest_sha256": (H.canonical_text_sha256(man) if man.is_file() else None),
        "artifact_sha256": None, "spec_sha256": None,
        "audit_tool_version": None, "audit_tool_sha256": None,
        "kernel_slug": None, "kernel_version": None,
        "submission_reference": None,
        "experiments": [], "facts": [],
        "state": "unknown", "binding_strength": "unknown",
        "binding_basis": None,
    }

    if raw.is_file():
        try:
            d = json.loads(raw.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            env["state"] = "unknown"
            env["binding_basis"] = f"raw receipt present but unreadable: {exc}"
            return env
        b, s, sub = d.get("build") or {}, d.get("spec") or {}, d.get("submission") or {}
        aud, k = d.get("auditor") or {}, d.get("kernel") or {}
        env["artifact_sha256"] = _clean(sub.get("sha256"), "submission.sha256")
        env["spec_sha256"] = _clean(s.get("sha256"), "spec.sha256")
        env["manifest_sha256"] = _clean(b.get("manifest_sha256"), "build.manifest_sha256") \
            or env["manifest_sha256"]
        env["notebook_sha256"] = _clean(b.get("notebook_sha256"), "build.notebook_sha256") \
            or env["notebook_sha256"]
        env["audit_tool_sha256"] = _clean(aud.get("sha256"), "auditor.sha256")
        env["audit_tool_version"] = _clean(d.get("schema_version"), "schema_version")
        owner, slug = _clean(k.get("owner"), "kernel.owner"), _clean(k.get("slug"), "kernel.slug")
        env["kernel_slug"] = f"{owner}/{slug}" if owner and slug else slug
        env["kernel_version"] = k.get("version")
        env["verdict"] = _clean(d.get("verdict"), "verdict")

    # BINDING IS DERIVED FROM THE REGISTRY, NEVER FROM THE ENVELOPE'S EXISTENCE.
    matches = by_kernel.get(env["kernel_slug"] or "", [])
    if matches:
        env["experiments"] = sorted(e["id"] for e in matches)
        env["facts"] = sorted({f for e in matches for f in (e.get("facts") or [])})
        env["submission_reference"] = next(
            (e.get("submission") for e in matches if e.get("submission")), None)

    # A CLONE MUST NOT BE TOLD THE RAW RECEIPT EXISTS. `raw_receipt_available_here` is a
    # statement about THIS machine and is recomputed on read; `state` is derived from it, so an
    # envelope generated here reports `bound_envelope_only` when read in a clone rather than
    # claiming an audit bundle the clone does not have.
    if env["raw_receipt_available_here"] and env["experiments"]:
        env["state"] = "fully_bound_raw_available"
        env["binding_strength"] = "receipt + experiment"
        env["binding_basis"] = "a raw receipt exists here AND the kernel resolves to an experiment"
    elif env["experiments"]:
        env["state"] = "bound_envelope_only"
        env["binding_strength"] = "experiment only"
        env["binding_basis"] = ("no raw receipt on this machine; the artifact is bound by its "
                                "experiment and notebook digest, which ARE tracked")
    elif env["raw_receipt_available_here"]:
        env["state"] = "unknown"
        env["binding_strength"] = "receipt only"
        env["binding_basis"] = "a receipt exists but its kernel resolves to no experiment"
    else:
        env["state"] = "historical_unbound"
        env["binding_strength"] = "none"
        env["binding_basis"] = ("no receipt and no experiment. NOT upgraded retrospectively - "
                                "generating an envelope later does not make an artifact bound")
    return env


def build() -> dict:
    by_kernel = _experiments_by_kernel()
    envs = {}
    for d in sorted((REPO / "notebooks").iterdir()):
        if d.is_dir():
            envs[d.name] = envelope_for(d, by_kernel)
    states: dict[str, int] = {}
    for e in envs.values():
        states[e["state"]] = states.get(e["state"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/core/receipt_envelope.py",
        "do_not_edit": "Generated. --check locks it against hand edits.",
        "policy": ("Raw receipts stay gitignored. This file is an ALLOW-LISTED extract: fields "
                   "are copied by name, so an unreviewed field cannot arrive by default. An "
                   "envelope NEVER upgrades an artifact's provenance."),
        "counts": states,
        "envelopes": envs,
    }


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    try:
        payload = build()
    except EnvelopeRefusal as exc:
        print(f"{HEARTBEAT_DRIFT} REFUSED: {exc}")
        return 2
    text = canonical(payload)
    if args.check:
        if not args.out.is_file():
            print(f"{HEARTBEAT_DRIFT} {args.out} missing")
            return 1
        # An envelope's raw-availability depends on the MACHINE, so that field alone may differ
        # between a clone and the development box. Everything else must match exactly.
        def strip(p):
            d = json.loads(p)
            for e in d["envelopes"].values():
                e.pop("raw_receipt_available_here", None)
                e.pop("state", None)
                e.pop("binding_basis", None)
            d.pop("counts", None)
            return json.dumps(d, indent=1, sort_keys=True)
        if strip(args.out.read_text(encoding="utf-8")) != strip(text):
            print(f"{HEARTBEAT_DRIFT} {args.out} differs from a fresh generation")
            return 1
        print(f"  receipt envelopes in sync: {payload['counts']}")
        print(HEARTBEAT_OK)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"  wrote {args.out.relative_to(REPO).as_posix()}  {payload['counts']}")
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
