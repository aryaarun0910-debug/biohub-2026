"""The FACT-0451 restriction, made mechanical AT THE SPEC SCHEMA BOUNDARY.

WHY THIS FILE EXISTS
--------------------
``general_v1.pt``'s training data CANNOT BE ESTABLISHED (FACT-0451) and overlap with the
competition movies cannot be excluded. The weights stay legitimate to USE; what dies is the
interpretability of any OFFLINE number computed with them, because no LOEO control we can build
can show the held-out fold was held out of THEIR training. Such an arm emits a figure
INDISTINGUISHABLE IN FORM from a valid one.

FACT-0451 records the enforcement status as HALF MECHANICAL: three instrument payloads carry
``binding_restriction.offline_scoreable`` and the spec schema carries nothing, so an arm routed
through ``assoc_train_harness --spec`` would still emit an unmarked number. This module is the
missing half, and it implements the four rules the host specified IN ADVANCE so they could not
drift while they waited.

THE FOUR RULES, each with the failure it exists to stop
------------------------------------------------------
(1) IDENTITY IS THE ARTIFACT HASH, NEVER THE ROLE NAME. A role can be renamed over identical
    bytes and FACT-0435 already records one checkpoint living at two paths under two role names.
    ``restricted_hits`` hashes what is on disk; ``general_v1.pt`` renamed to
    ``friendly_local_weights.pt`` is still restricted.
(2) DECLARATION IS MANDATORY. Omission is a REFUSAL, not a default. An absent field is exactly
    how a restriction silently stops applying.
(3) DERIVED false CANNOT BE OVERRIDDEN. A submitter must not be able to mark its own arm
    scoreable - the same principle that made ``gpu_contract_skip`` inert in ``submission_run_v1``.
(4) THE VERDICT LAYER REFUSES PROMOTION, it does not merely label the report. A label on a
    payload nobody blocks on is prose again, which is the state FACT-0451 found.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not forbid USING a restricted checkpoint. FACT-0451 and rule 2.6.b leave the weights
legitimate; the restriction is on reading an offline number off them. A submission-only arm is
admissible and this module marks it, so the marking travels with the payload.

An UNRESOLVED checkpoint reference - a spec naming a ``.pt`` that is not on this disk - cannot be
adjudicated by hash. It is recorded, it does NOT refuse at declaration time (a prepared-not-
launched spec routinely names artifacts that do not exist yet), and it DOES refuse at promotion
time. Silence about an unadjudicated artifact is the FACT-0417 shape: a consumer trusting a field
whose producer never established it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

FIELD = "offline_scoreable"
BLOCK = "binding_restriction"
CKPT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")

# --------------------------------------------------------------------------------------------
# Rule (1) - the registry is keyed by HASH. The name beside it is documentation, never identity.
# --------------------------------------------------------------------------------------------
RESTRICTED: dict[str, dict[str, str]] = {
    # HOCT general_v1.pt - FACT-0451 / FACT-0445. Organizer lab, released eight weeks into the
    # competition, no manifest anywhere in four releases, four PRs, 58 tree entries or the issues.
    "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33": {
        "name_at_registration": "general_v1.pt",
        "basis": "FACT-0451",
        "verdict": "training data CANNOT BE ESTABLISHED; overlap cannot be excluded",
        "restriction": "SUBMISSION_ONLY_JUDGEMENT",
    },
}

# Artifacts whose digest this repository only holds ELIDED. PKT-0047's inventory prints
# general_v0.pt's sha256 as "024c2e4606275c966679...ef8e4a" and the middle is genuinely not
# recorded anywhere on disk, so a full-hash entry above would be a fabricated number - which is
# exactly what the registry forbids. Matched on the recorded prefix AND suffix instead: that is
# weaker than a full digest and it is stated as weaker, but it fails CLOSED rather than leaving an
# organizer-lab checkpoint with no manifest silently scoreable. Replace with a measured digest the
# first time the file is on this disk.
PENDING_PREFIX_SUFFIX: list[dict[str, str]] = [
    {
        "name_at_registration": "general_v0.pt",
        "prefix": "024c2e4606275c966679",
        "suffix": "ef8e4a",
        "basis": "FACT-0445 asset inventory + the FACT-0451 argument, NOT separately adjudicated",
        "verdict": "no manifest; same publisher and same 'general' wording as general_v1",
        "restriction": "SUBMISSION_ONLY_JUDGEMENT",
        "caution": "ELIDED DIGEST - prefix/suffix match, not a full-hash identity",
    },
]

# Artifacts positively CLEARED, with the evidence that cleared them. Recorded so a reader can tell
# "adjudicated and clean" from "never looked at" - the distinction FACT-0451 turns on. Membership
# here grants nothing; it only documents that the question was asked.
NOT_RESTRICTED: dict[str, dict[str, str]] = {
    "d150fc228d231a470cccafd6be58851a379fa15657d4367e8cc8ee0098cae8b9": {
        "name_at_registration": "FOCUS-3D model_final_nuclei.pth",
        "basis": "FACT-0424",
        "verdict": "no overlap was FOUND between the published inventory and this competition's "
                   "sources - EXTERNAL and negative, never 'no overlap'",
    },
}


class RestrictionRefusal(RuntimeError):
    """Raised at the schema boundary. Every path out of this module is a refusal or a record."""


# --------------------------------------------------------------------------------------------
# Rule (1) - resolution
# --------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _walk_strings(obj: Any) -> Iterable[tuple[str, str]]:
    """Yield ``(json_path, value)`` for every string in a nested spec."""
    stack: list[tuple[str, Any]] = [("$", obj)]
    while stack:
        where, node = stack.pop()
        if isinstance(node, dict):
            stack.extend((where + "." + str(k), v) for k, v in node.items())
        elif isinstance(node, (list, tuple)):
            stack.extend((where + "[" + str(i) + "]", v) for i, v in enumerate(node))
        elif isinstance(node, str):
            yield where, node


def checkpoint_refs(spec: Any) -> list[dict[str, Any]]:
    """Every checkpoint-shaped string in the spec, resolved to a hash WHERE THE FILE EXISTS.

    The hash is the identity (rule 1). A reference we cannot resolve is reported as
    ``resolved: false`` and is fatal only at promotion time.
    """
    refs: list[dict[str, Any]] = []
    for where, value in _walk_strings(spec):
        if not value.lower().endswith(CKPT_SUFFIXES):
            continue
        rec: dict[str, Any] = {"at": where, "value": value, "resolved": False, "sha256": None}
        try:
            p = Path(value)
            if p.is_file():
                rec["sha256"] = _sha256(p)
                rec["resolved"] = True
                rec["bytes"] = p.stat().st_size
        except OSError as exc:                     # unreadable is not "absent" - say which it is
            rec["error"] = type(exc).__name__ + ": " + str(exc)
        refs.append(rec)
    return refs


def declared_hashes(spec: Any) -> list[dict[str, str]]:
    """Any 64-hex digest written anywhere in the spec, under any key.

    A spec that DECLARES a restricted digest is restricted even if the file is absent from this
    machine - the declaration is the artifact's identity travelling with the document.
    """
    out: list[dict[str, str]] = []
    for where, value in _walk_strings(spec):
        v = value.strip().lower()
        if len(v) == 64 and all(c in "0123456789abcdef" for c in v):
            out.append({"at": where, "sha256": v})
    return out


def _match(sha: str | None) -> dict[str, str] | None:
    """Adjudicate ONE digest. Full-hash identity first, elided prefix/suffix second."""
    if not sha:
        return None
    if sha in RESTRICTED:
        return RESTRICTED[sha]
    for entry in PENDING_PREFIX_SUFFIX:
        if sha.startswith(entry["prefix"]) and sha.endswith(entry["suffix"]):
            return entry
    return None


def restricted_hits(spec: Any) -> list[dict[str, Any]]:
    """Every restricted artifact this spec touches, by hash, however it is named."""
    hits: list[dict[str, Any]] = []
    for ref in checkpoint_refs(spec):
        hit = _match(ref["sha256"])
        if hit:
            hits.append(dict(ref, how="file on disk, hashed", **hit))
    for dec in declared_hashes(spec):
        hit = _match(dec["sha256"])
        if hit:
            hits.append(dict(dec, how="digest declared in the spec", **hit))
    return hits


# --------------------------------------------------------------------------------------------
# Rules (2) and (3) - declaration and non-overridable derivation
# --------------------------------------------------------------------------------------------
def enforce_spec(spec: dict, *, path: str | Path | None = None) -> dict:
    """Apply rules 1-3. Returns the RESOLUTION record; raises ``RestrictionRefusal`` otherwise.

    The record is what a payload must carry downstream. It is deliberately NOT merged into the
    spec: a caller that forgets to attach it is caught by ``assert_promotable``, whereas a caller
    that silently inherits a mutated spec is not.
    """
    src = str(path) if path is not None else str(spec.get("name", "<spec>"))

    # (2) omission is a refusal.
    block = spec.get(BLOCK)
    if not isinstance(block, dict) or FIELD not in block:
        raise RestrictionRefusal(
            src + ": every association spec must DECLARE " + BLOCK + "." + FIELD
            + " (FACT-0451 rule 2). Omission is a refusal, not a default - an absent field is how"
            " a restriction silently stops applying. Declare true only if the spec touches no"
            " restricted checkpoint; the harness derives false for you when it does."
        )
    declared = block[FIELD]
    if not isinstance(declared, bool):
        raise RestrictionRefusal(
            src + ": " + BLOCK + "." + FIELD + " must be a bool, got "
            + type(declared).__name__ + " " + repr(declared)
            + ". A truthy string is how a 'false' becomes a true."
        )

    hits = restricted_hits(spec)
    derived = not hits

    # (3) derived false cannot be overridden.
    if declared and not derived:
        names = ", ".join(sorted({str(h["name_at_registration"]) for h in hits}))
        wheres = ", ".join(sorted({str(h["at"]) for h in hits}))
        raise RestrictionRefusal(
            src + ": " + BLOCK + "." + FIELD + " is declared true but this spec touches a"
            " RESTRICTED artifact (" + names + ") at " + wheres + ". Derived false stands"
            " (FACT-0451 rule 3): a submitter must not be able to mark its own arm scoreable."
            " Identity is the sha256, not the role name - renaming the file does not clear it."
        )

    effective = bool(declared and derived)
    refs = checkpoint_refs(spec)
    return {
        "fact": "FACT-0451",
        "spec": src,
        "declared": declared,
        "derived": derived,
        FIELD: effective,
        "restricted_hits": hits,
        "checkpoint_refs": refs,
        "unresolved_checkpoint_refs": [r["at"] for r in refs if not r["resolved"]],
        "restriction": "SUBMISSION_ONLY_JUDGEMENT" if hits else None,
        "note": (
            "Restricted means the OFFLINE NUMBER is uninterpretable, not that the weights are "
            "illegitimate. A submission-only arm is admissible; a promoted offline score is not."
            if hits else
            "No restricted artifact reached this spec. This is NOT a claim that every referenced "
            "artifact was adjudicated - see unresolved_checkpoint_refs."
        ),
    }


def load_spec(path: str | Path) -> tuple[dict, dict]:
    """Read a spec JSON and enforce rules 1-3 in one place. The only sanctioned entry point."""
    p = Path(path)
    spec = json.loads(p.read_text(encoding="utf-8"))
    return spec, enforce_spec(spec, path=p)


# --------------------------------------------------------------------------------------------
# Rule (4) - the verdict layer BLOCKS
# --------------------------------------------------------------------------------------------
def assert_promotable(resolution: dict | None, *, what: str = "this arm") -> None:
    """Refuse promotion. Called by whatever turns a number into a decision.

    Four refusals, and the first two are the FACT-0417 shape - a consumer trusting a field its
    producer stopped emitting. A missing resolution is treated as a restricted one, never as a
    clean one, because the two are indistinguishable to a caller that does not check.
    """
    if resolution is None:
        raise RestrictionRefusal(
            what + ": no FACT-0451 resolution accompanies this result. A dropped field is"
            " indistinguishable from a clean one to everything downstream (the FACT-0417 shape),"
            " so absence refuses."
        )
    if FIELD not in resolution:
        raise RestrictionRefusal(
            what + ": the resolution carries no " + FIELD + ". Its producer stopped emitting the"
            " field; refusing rather than defaulting."
        )
    if not resolution[FIELD]:
        names = ", ".join(sorted({str(h.get("name_at_registration", "?"))
                                  for h in resolution.get("restricted_hits", [])})) or "unknown"
        raise RestrictionRefusal(
            what + ": offline_scoreable is FALSE (" + names + ", FACT-0451). This number cannot"
            " promote anything - no LOEO control we can build can show the held-out fold was held"
            " out of THEIR training, so it is interpretable as neither a pass nor a fail."
            " SUBMISSION-ONLY JUDGEMENT."
        )
    unresolved = resolution.get("unresolved_checkpoint_refs") or []
    if unresolved:
        raise RestrictionRefusal(
            what + ": " + str(len(unresolved)) + " checkpoint reference(s) could not be resolved"
            " to a hash (" + ", ".join(unresolved[:4]) + "). Identity is the hash (FACT-0451"
            " rule 1); an unadjudicated artifact refuses at promotion rather than passing"
            " silently."
        )
