r"""Registry integrity gate.

WHY
---
Measured 2026-08-25: the superseded score ``0.915`` appeared 244 times across 36 files
while the live score ``0.925`` appeared 31 times across 7. An agent opening a research doc
at random was 8x more likely to read a dead number than a live one. Numbers were COPIED
into prose instead of REFERENCED by id, so every restatement was an independent chance to
go stale, and nothing detected divergence.

A registry alone does not fix that - an unenforced registry is just a 4th place to be
wrong. This module is the enforcement.

CHECKS
------
R1  ids are unique and well-formed
R2  cross-references resolve (fact.experiment, lever.closed_by, packet.lever, ...)
R3  a VERIFIED or MEASURED fact must name an ``instrument``. Provenance is a claim about
    EVIDENCE, so a fact asserting the strongest provenance with no instrument is the exact
    shape of the 3.44% anchor that could not be reproduced.
R4  a lever with status ``killed`` must name ``closed_by``, and none of that evidence may
    be UNVERIFIED. You may not close a lever on evidence nobody can reproduce.
R5  no ``state`` document may assert a guarded superseded value as current
R6  no two packets may hold the same lever - the duplicate-work lock
R7  ``validity`` uses the fixed vocabulary, and anything not VALID states a reason
R8  a fact produced by a ``void`` or ``cancelled`` experiment may not stay VALID - this is
    the RETRACTION mechanism
R9  a lever may not be closed or supported by an INVALID fact
R10 adoption notes, not failures: an experiment-derived fact should name its experiment,
    and a held-out-fold fact should name its evaluation ``protocol``

THE SECOND AXIS - why R7-R10 exist
----------------------------------
EXP-0019 scored LOEO fold 1 (the 6bba embryo) with weights TRAINED on 6bba, because its
spec set no ``BIOHUB_LOEO_WEIGHTS_GLOB`` and fell back to the pack default. It inflated the
fold score by +0.203 and manufactured 25 of 26 division true positives. Seven facts, eight
packets and a whole strategy were built on it before anyone noticed - and the registry never
objected, because those facts were ``MEASURED``, the second-strongest provenance, and they
DESERVED it. They were correctly computed. They were correctly computed FROM AN INVALID RUN.

``provenance`` grades DERIVATION STRENGTH. It cannot express that, and stretching it to try
- filing a leaked number as UNVERIFIED - would be a lie about the evidence, which is the one
thing this registry exists to prevent. So validity is a SECOND, ORTHOGONAL axis:

    provenance   how strong is the derivation?            VERIFIED .. UNVERIFIED
    validity     was the run it derives from legitimate?  VALID .. INVALID

A fact may be MEASURED and INVALID at once. That pair is the exact shape of the leak, and
nothing in the old schema could write it down.

Run:  .\.venv\Scripts\python.exe scripts\core\validate_registry.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
REG = REPO / "research" / "00-system" / "registry"

ID_PATTERNS = {
    "facts": re.compile(r"^FACT-\d{4}$"),
    "experiments": re.compile(r"^EXP-\d{4}[A-Z]?$"),
    "levers": re.compile(r"^LEVER-\d{4}$"),
    "packets": re.compile(r"^PKT-\d{4}$"),
}

# A line may legitimately restate a dead number when it is explicitly historical.
HISTORICAL_MARKERS = (
    "supersed", "formerly", "historical", "archive", "provenance", "lineage",
    "was ", "previously", "no longer", "retired", "obsolete", "corrected",
    "not our score", "their number", "probe band",
)

# Any explicit registry citation on the line means the number was written deliberately
# against the registry rather than restated from memory, which is exactly the behaviour
# this check exists to encourage.
FACT_CITATION = re.compile(r"FACT-\d{4}")

STRONG_PROVENANCE = {"VERIFIED", "MEASURED"}
ALL_PROVENANCE = STRONG_PROVENANCE | {"EXTERNAL", "UNVERIFIED", "SUPERSEDED"}

# ---- the validity axis ----------------------------------------------------------------
# ABSENT means VALID - "no defect on record". That is deliberate. A schema demanding an
# explicit stamp on all 92 facts would be filled in by rote and would grade nothing; here
# the field being PRESENT is itself the signal, so only a fact with a known defect carries
# it. The cost is stated honestly: absence means "unexamined", not "audited clean".
VALIDITY_DEFAULT = "VALID"
#   VALID    no defect recorded in the run or protocol this came from
#   SUSPECT  a known defect affects PART of the claim - the structure may survive but every
#            magnitude is untrusted until re-derived on a clean run
#   INVALID  the run that produced it was not a legitimate measurement of what it claims;
#            no number in it may be used or cited
#   UNKNOWN  deliberately not assessed - an audit is owed
ALL_VALIDITY = {"VALID", "SUSPECT", "INVALID", "UNKNOWN"}
TAINTED_VALIDITY = {"SUSPECT", "INVALID"}

# An experiment in one of these states did not produce a result, so nothing derived from it
# may claim to be a clean measurement. This is what makes retraction PROPAGATE: void the
# experiment once, and every fact pointing at it is forced to declare itself.
RETRACTING_STATUS = {"void", "cancelled"}

# Facts scoped to a held-out fold are the class where train/test hygiene decides whether the
# number means anything at all. The leaky and honest fold-1 manifests were byte-identical in
# fold, arm, n_crops, det_threshold, secondary_enabled, deepcenter_enabled and
# experiment_tag - ONLY `weights` differed. Nothing in a fact recorded `weights`.
PROTOCOL_KEYS = ("weights",)

# An instrument that points at a run's output, rather than at a static artifact or a
# derivation, marks a fact as experiment-derived - so it should name the EXP id.
EXPERIMENT_DERIVED = re.compile(r"_evidence/|EXP-\d{4}|kaggle submission|submission \d{6}", re.I)


def _load(name: str) -> dict:
    path = REG / f"{name}.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_packets() -> list[dict]:
    out = []
    pdir = REG / "packets"
    if not pdir.is_dir():
        return out
    for p in sorted(pdir.glob("PKT-*.yaml")):
        if p.stem == "PKT-TEMPLATE":
            continue
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        d["_path"] = str(p.relative_to(REPO)).replace("\\", "/")
        out.append(d)
    return out


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    facts = _load("facts").get("facts", []) or []
    exps = _load("experiments").get("experiments", []) or []
    levers = _load("levers").get("levers", []) or []
    packets = _load_packets()

    fact_ids = {f["id"] for f in facts}
    exp_ids = {e["id"] for e in exps}
    lever_ids = {l["id"] for l in levers}

    # ---- R1 id format and uniqueness ------------------------------------------------
    for kind, rows in (("facts", facts), ("experiments", exps), ("levers", levers),
                       ("packets", packets)):
        seen = set()
        for r in rows:
            rid = r.get("id", "")
            if not ID_PATTERNS[kind].match(str(rid)):
                errors.append(f"R1 {kind}: malformed id {rid!r}")
            if rid in seen:
                errors.append(f"R1 {kind}: duplicate id {rid}")
            seen.add(rid)

    # ---- R2 cross-references ---------------------------------------------------------
    for f in facts:
        if f.get("experiment") and f["experiment"] not in exp_ids:
            errors.append(f"R2 {f['id']}: experiment {f['experiment']} does not exist")
        for dep in f.get("derived_from", []) or []:
            if dep not in fact_ids:
                errors.append(f"R2 {f['id']}: derived_from {dep} does not exist")
        if f.get("superseded_by") and f["superseded_by"] not in fact_ids:
            errors.append(f"R2 {f['id']}: superseded_by {f['superseded_by']} does not exist")
        prov = f.get("provenance")
        if prov not in ALL_PROVENANCE:
            errors.append(f"R2 {f['id']}: unknown provenance {prov!r}")

    for e in exps:
        for fid in e.get("facts", []) or []:
            if fid not in fact_ids:
                errors.append(f"R2 {e['id']}: facts references missing {fid}")
        spec = e.get("spec")
        if spec and not (REPO / spec).is_file():
            errors.append(f"R2 {e['id']}: spec not found: {spec}")

    for l in levers:
        # `retracted_closure.facts` records a closure that was WITHDRAWN because its
        # evidence turned out invalid. It is provenance, not a live claim, but a dangling
        # id there is still a broken audit trail.
        retracted = (l.get("retracted_closure") or {}).get("facts") or []
        for fid in (l.get("closed_by") or []) + (l.get("supporting") or []) + retracted:
            if fid not in fact_ids:
                errors.append(f"R2 {l['id']}: references missing fact {fid}")
        for eid in (l.get("experiments") or []) + (
                [l["reclose_pending"]] if l.get("reclose_pending") else []):
            if eid not in exp_ids:
                errors.append(f"R2 {l['id']}: references missing experiment {eid}")

    # ---- R3 strong provenance requires an instrument that EXISTS ---------------------
    # FACT-0080 named a scratchpad script that was never committed. The string was present
    # so the old check passed, but the file did not exist and the result was not
    # reproducible - and it turned out to disagree with the real scorer by 26x. A named
    # instrument must therefore be resolvable, not merely non-empty.
    # (?<![\w/]) so that a `src/...` segment INSIDE a longer path -- e.g.
    # vendor/kaggle-cell-tracking/src/tracking_cellmot/division_metrics.py -- is not
    # mistaken for a repo-root path and reported missing. That false positive fired twice
    # on 2026-08-26 against instruments that were perfectly real.
    repo_path = re.compile(
        r"(?<![\w/])(?:scripts|notebooks|src|tests|vendor|artifacts)/[\w./-]+\.(?:py|ipynb)\b"
    )
    for f in facts:
        prov = f.get("provenance")
        instrument = f.get("instrument")
        if prov in STRONG_PROVENANCE and not instrument:
            errors.append(
                f"R3 {f['id']}: provenance {prov} but no instrument named. "
                "An oracle that is not committed as code is not a result."
            )
            continue
        if prov in STRONG_PROVENANCE and instrument:
            for cited in repo_path.findall(str(instrument)):
                if not (REPO / cited).exists():
                    errors.append(
                        f"R3 {f['id']}: instrument cites {cited}, which does not exist. "
                        "A MEASURED fact must be reproducible from committed code."
                    )

    # ---- R4 a killed lever needs reproducible evidence --------------------------------
    prov_of = {f["id"]: f.get("provenance") for f in facts}
    for l in levers:
        if l.get("status") == "killed":
            closed_by = l.get("closed_by") or []
            if not closed_by:
                errors.append(f"R4 {l['id']}: status killed but names no closed_by evidence")
            for fid in closed_by:
                if prov_of.get(fid) == "UNVERIFIED":
                    errors.append(
                        f"R4 {l['id']}: closed on UNVERIFIED evidence {fid}. "
                        "A lever may not be killed by a number nobody can reproduce."
                    )

    # ---- R5 state docs may not assert a guarded superseded value ----------------------
    guarded = [f for f in facts if f.get("guard_state_docs") and f.get("value") is not None]
    state_docs = 0
    unclassified: list[str] = []
    for md in sorted((REPO / "research").rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        fm = _frontmatter(text)
        kind = fm.get("record_kind")
        rel = str(md.relative_to(REPO)).replace("\\", "/")
        if kind is None:
            if fm.get("status") == "active":
                unclassified.append(rel)
            continue
        if kind != "state":
            continue
        state_docs += 1
        # A document may carry an explicitly archived tail. Once a heading announces it,
        # everything below is provenance and its numbers are frozen on purpose.
        archived_from = None
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#") and re.search(
                r"superseded|archive|historical|provenance", line, re.I
            ):
                archived_from = i
                break

        for f in guarded:
            pat = re.compile(rf"(?<![\d.]){re.escape(str(f['value']))}(?![\d])")
            for i, line in enumerate(text.splitlines(), 1):
                if archived_from is not None and i >= archived_from:
                    break
                if not pat.search(line):
                    continue
                low = line.lower()
                if (f["id"].lower() in low or FACT_CITATION.search(line)
                        or any(m in low for m in HISTORICAL_MARKERS)):
                    continue
                errors.append(
                    f"R5 {rel}:{i}: asserts superseded value {f['value']} "
                    f"({f['id']}, superseded by {f.get('superseded_by')}) as current"
                )

    if unclassified:
        notes.append(
            f"{len(unclassified)} active docs have no record_kind "
            f"(state|ledger|archive) and are outside the R5 guard"
        )

    # ---- R6 the duplicate-work lock ---------------------------------------------------
    held: dict[str, str] = {}
    for p in packets:
        lv = p.get("lever")
        if lv and lv not in lever_ids:
            errors.append(f"R2 {p.get('id')}: lever {lv} does not exist")
        if p.get("lock") in ("claimed", "running") and lv:
            if lv in held:
                errors.append(
                    f"R6 {p.get('id')} and {held[lv]} both hold {lv}. "
                    "Two agents on one lever is duplicated work by construction."
                )
            else:
                held[lv] = p.get("id", "?")

    # ---- R7 the validity vocabulary, and a reason whenever it is not VALID -------------
    # "INVALID" with no reason is a dead end for whoever reads it next: they cannot tell
    # whether to re-derive the number, drop the claim, or go find the honest successor.
    validity_of: dict[str, str] = {}
    for f in facts:
        v = f.get("validity", VALIDITY_DEFAULT)
        if v not in ALL_VALIDITY:
            errors.append(
                f"R7 {f['id']}: unknown validity {v!r} "
                f"(expected one of {sorted(ALL_VALIDITY)})"
            )
            v = VALIDITY_DEFAULT
        validity_of[f["id"]] = v
        if v != "VALID" and not str(f.get("validity_reason") or "").strip():
            errors.append(
                f"R7 {f['id']}: validity {v} but no validity_reason. "
                "A retraction that does not say what broke cannot be acted on."
            )

    # ---- R8 retraction propagates from the experiment to its facts ---------------------
    # THE MECHANISM THE LEAK NEEDED. EXP-0019 was voided on 2026-08-26 and every number it
    # produced stayed MEASURED and unmarked, because nothing walked the link. Note this
    # keys on `fact.experiment` (the run that PRODUCED the fact), never on
    # `experiment.facts` - that list also holds the facts that MOTIVATED a run, and voiding
    # an experiment must not retract its own inputs. EXP-0019 cites FACT-0111, which closed
    # LEVER-0003 on unrelated evidence and is untouched by the leak.
    exp_status = {e["id"]: str(e.get("status", "")).lower() for e in exps}
    for f in facts:
        eid = f.get("experiment")
        if not eid or exp_status.get(eid) not in RETRACTING_STATUS:
            continue
        if validity_of.get(f["id"], VALIDITY_DEFAULT) not in TAINTED_VALIDITY:
            errors.append(
                f"R8 {f['id']}: derived from {eid} (status "
                f"{exp_status.get(eid)}) but validity is "
                f"{validity_of.get(f['id'], VALIDITY_DEFAULT)}. A voided run cannot leave a "
                "clean fact behind - mark it INVALID, or SUSPECT if a stated part survives."
            )

    # ---- R9 a lever may not rest on an invalid fact ------------------------------------
    # LEVER-0012 was killed by FACT-0131, which came from the voided EXP-0019. The kill read
    # as settled science for a day and closed a lane that may be open.
    for l in levers:
        for fid in l.get("closed_by") or []:
            v = validity_of.get(fid, VALIDITY_DEFAULT)
            if v == "INVALID":
                errors.append(
                    f"R9 {l['id']}: closed on INVALID evidence {fid}. "
                    "A lever closed by a retracted measurement is a lane shut for no reason."
                )
            elif v == "SUSPECT":
                notes.append(
                    f"R9 {l['id']} is closed by SUSPECT {fid} - the closure holds only on "
                    "the part of that fact which survives; re-close it on clean evidence"
                )
        for fid in l.get("supporting") or []:
            if validity_of.get(fid, VALIDITY_DEFAULT) == "INVALID":
                errors.append(
                    f"R9 {l['id']}: supported by INVALID evidence {fid}. "
                    "Cite its successor, or drop the support."
                )

    # A packet is where an invalid fact turns into GPU hours. Note-level: packets are
    # short-lived and this is meant to be read at claim time, not to block the gate.
    for p in packets:
        for fid in (p.get("inputs") or {}).get("facts") or []:
            v = validity_of.get(fid, VALIDITY_DEFAULT)
            if v in TAINTED_VALIDITY and p.get("lock") in ("claimed", "running"):
                notes.append(
                    f"R9 {p.get('id')} is live and takes {v} {fid} as an input"
                )

    # ---- R10 adoption notes - deliberately not failures --------------------------------
    # Only 3 of 92 facts name an experiment. Making that a hard error today would fail the
    # gate on 21 legacy facts and the rule would simply be deleted, so it reports and the
    # backlog is visible on every run.
    missing_exp = [
        f["id"] for f in facts
        if f.get("provenance") in STRONG_PROVENANCE
        and not f.get("experiment")
        and EXPERIMENT_DERIVED.search(str(f.get("instrument") or ""))
    ]
    if missing_exp:
        notes.append(
            f"R10 {len(missing_exp)} experiment-derived facts name no `experiment`, so a "
            f"void cannot reach them: {', '.join(missing_exp[:10])}"
            + (" ..." if len(missing_exp) > 10 else "")
        )

    missing_protocol = [
        f["id"] for f in facts
        if f.get("provenance") in STRONG_PROVENANCE
        and isinstance(f.get("scope"), dict) and "fold" in f["scope"]
        and not all(k in (f.get("protocol") or {}) for k in PROTOCOL_KEYS)
    ]
    if missing_protocol:
        notes.append(
            f"R10 {len(missing_protocol)} held-out-fold facts name no `protocol.weights`, "
            f"so leaky and honest measurements of the same quantity are indistinguishable: "
            f"{', '.join(missing_protocol[:10])}"
            + (" ..." if len(missing_protocol) > 10 else "")
        )

    # ---- report -----------------------------------------------------------------------
    print(f"registry: {len(facts)} facts, {len(exps)} experiments, "
          f"{len(levers)} levers, {len(packets)} packets")
    by_prov = {p: sum(1 for f in facts if f.get("provenance") == p) for p in sorted(ALL_PROVENANCE)}
    print("provenance: " + ", ".join(f"{k} {v}" for k, v in by_prov.items() if v))
    by_val = {v: sum(1 for f in facts if f.get("validity", VALIDITY_DEFAULT) == v)
              for v in sorted(ALL_VALIDITY)}
    print("validity:   " + ", ".join(f"{k} {v}" for k, v in by_val.items() if v)
          + "   (unmarked counts as VALID = no defect on record, NOT audited clean)")
    print(f"state docs guarded: {state_docs}; guarded values: "
          f"{[f['value'] for f in guarded]}")
    for nte in notes:
        print(f"  note: {nte}")

    if errors:
        print(f"\nFAILED - {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        return 1
    print("\nregistry OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
