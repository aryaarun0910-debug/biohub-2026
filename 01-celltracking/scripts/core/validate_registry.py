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
        for fid in (l.get("closed_by") or []) + (l.get("supporting") or []):
            if fid not in fact_ids:
                errors.append(f"R2 {l['id']}: references missing fact {fid}")
        for eid in l.get("experiments", []) or []:
            if eid not in exp_ids:
                errors.append(f"R2 {l['id']}: references missing experiment {eid}")

    # ---- R3 strong provenance requires an instrument ---------------------------------
    for f in facts:
        if f.get("provenance") in STRONG_PROVENANCE and not f.get("instrument"):
            errors.append(
                f"R3 {f['id']}: provenance {f['provenance']} but no instrument named. "
                "An oracle that is not committed as code is not a result."
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

    # ---- report -----------------------------------------------------------------------
    print(f"registry: {len(facts)} facts, {len(exps)} experiments, "
          f"{len(levers)} levers, {len(packets)} packets")
    by_prov = {p: sum(1 for f in facts if f.get("provenance") == p) for p in sorted(ALL_PROVENANCE)}
    print("provenance: " + ", ".join(f"{k} {v}" for k, v in by_prov.items() if v))
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
