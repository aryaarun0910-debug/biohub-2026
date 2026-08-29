r"""READ-ONLY field-level drift audit of the registry: working tree against git HEAD.

WHY A VALIDATOR IS NOT ENOUGH
------------------------------
`validate_registry.py` checks the rules it knows. A concurrent write that drops a `note`, a
`scope` or an `instrument` passes every rule and is invisible - the fact simply says less than it
did. So this instrument does not check rules at all: it compares EVERY key of EVERY entry against
`git show HEAD:<path>` and reports each difference, so a restore-from-HEAD decision can be made
knowing exactly what would be lost and what would be recovered.

THREE FAILURE MODES IT IS BUILT AROUND, ALL OBSERVED ON 2026-08-30
-------------------------------------------------------------------
  1. COMMENT DESTRUCTION. A YAML load-then-dump strips every comment. The registry's comments are
     load-bearing documentation - the validity-axis header, the note that R8 walks one hop - and
     no schema check can miss them because no schema check can see them. Counted per file.
  2. SILENT FIELD LOSS. Load-then-dump also loses whatever another agent wrote between the load
     and the dump, and can drop or alter individual keys. Reported key by key, old value against
     new, never summarised as a count.
  3. ORPHANED MAPPINGS - the one that makes a corrupted file look healthy. A dumper writes the
     top-level sequence at column 0; a later hand-append uses the original 2-space indent; and
     YAML then parses those appended entries as ITEMS OF THE PRECEDING ENTRY'S LAST LIST rather
     than as entries. The file parses, the validator passes, and the additions do not exist. Any
     mapping found inside a list of scalars is reported as a swallowed entry.

WRITES NOTHING. It opens the registry with `read_text` and never serialises YAML - a load-then-
dump on a shared registry file is the defect this exists to detect, so it must not be the tool's
own behaviour. Its only output is the JSON report at `--out`, outside the registry.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import yaml

HEARTBEAT = "REGISTRY_DRIFT_AUDIT_COMPLETE"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
LIST_KEY = {"facts.yaml": "facts", "experiments.yaml": "experiments", "levers.yaml": "levers"}


def _head_text(rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(ROOT), capture_output=True)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def _comments(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


def _entries(doc, key: str) -> dict:
    if not isinstance(doc, dict):
        return {}
    seq = doc.get(key) or []
    return {e["id"]: e for e in seq if isinstance(e, dict) and "id" in e}


def swallowed_mappings(doc, key: str) -> list[dict]:
    """Entries an indentation mismatch hid inside another entry's list."""
    found = []
    for entry in (doc.get(key) or []) if isinstance(doc, dict) else []:
        if not isinstance(entry, dict):
            continue
        for field, value in entry.items():
            if not isinstance(value, list):
                continue
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    found.append({
                        "hidden_id": item.get("id"), "swallowed_into": entry.get("id"),
                        "field": field, "index": i,
                        "effect": "invisible to every consumer; also corrupts the host field",
                    })
    return found


def diff_file(rel: str, key: str) -> dict:
    path = ROOT / rel
    out: dict = {"file": rel}
    head_text = _head_text(rel)
    if head_text is None:
        out["status"] = "NEW - not present at HEAD; a restore would delete it entirely"
        return out
    work_text = io.open(path, encoding="utf-8").read() if path.is_file() else None
    if work_text is None:
        out["status"] = "DELETED in the working tree"
        return out
    out["comment_lines"] = {"head": _comments(head_text), "worktree": _comments(work_text),
                            "lost": _comments(head_text) - _comments(work_text)}
    try:
        head_doc, work_doc = yaml.safe_load(head_text), yaml.safe_load(work_text)
    except yaml.YAMLError as exc:
        out["status"] = f"UNPARSEABLE: {exc}"
        return out
    a, b = _entries(head_doc, key), _entries(work_doc, key)
    out["entry_counts"] = {"head": len(a), "worktree": len(b)}
    out["entries_added_in_worktree"] = sorted(set(b) - set(a))
    out["entries_missing_from_worktree"] = sorted(set(a) - set(b))
    changes = []
    for eid in sorted(set(a) & set(b)):
        for k in sorted(set(a[eid]) | set(b[eid])):
            if a[eid].get(k) != b[eid].get(k):
                changes.append({
                    "id": eid, "field": k,
                    "head": (str(a[eid].get(k))[:160] if k in a[eid] else "<absent>"),
                    "worktree": (str(b[eid].get(k))[:160] if k in b[eid] else "<absent>"),
                    "kind": ("FIELD DROPPED" if k not in b[eid]
                             else "FIELD ADDED" if k not in a[eid] else "VALUE CHANGED"),
                })
    out["field_changes"] = changes
    out["swallowed_mappings"] = swallowed_mappings(work_doc, key)
    top = {}
    for k in sorted(set(head_doc or {}) | set(work_doc or {})):
        if k == key:
            continue
        if (head_doc or {}).get(k) != (work_doc or {}).get(k):
            top[k] = {"head": (head_doc or {}).get(k), "worktree": (work_doc or {}).get(k)}
    out["top_level_changes"] = top
    out["restore_from_head_would_lose"] = (
        out["entries_added_in_worktree"]
        + [f"{c['id']}.{c['field']}" for c in changes if c["kind"] == "FIELD ADDED"]
        + [f"HIDDEN {s['hidden_id']}" for s in out["swallowed_mappings"]]
    )
    out["restore_from_head_would_recover"] = (
        [f"{c['id']}.{c['field']}" for c in changes if c["kind"] == "FIELD DROPPED"]
        + [f"{c['id']}.{c['field']}" for c in changes if c["kind"] == "VALUE CHANGED"]
        + ([f"{out['comment_lines']['lost']} comment lines"]
           if out["comment_lines"]["lost"] > 0 else [])
    )
    return out


def diff_packets() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "research/00-system/registry/packets").glob("*.yaml")):
        rel = p.relative_to(ROOT).as_posix()
        head_text = _head_text(rel)
        work_text = io.open(p, encoding="utf-8").read()
        if head_text is None:
            rows.append({"file": rel, "status": "NEW at worktree - a restore would delete it"})
            continue
        if head_text.replace("\r\n", "\n") == work_text.replace("\r\n", "\n"):
            continue
        try:
            a, b = yaml.safe_load(head_text) or {}, yaml.safe_load(work_text) or {}
        except yaml.YAMLError as exc:
            rows.append({"file": rel, "status": f"UNPARSEABLE: {exc}"})
            continue
        rows.append({
            "file": rel,
            "comment_lines": {"head": _comments(head_text), "worktree": _comments(work_text)},
            "field_changes": [
                {"field": k, "kind": ("FIELD DROPPED" if k not in b else "FIELD ADDED"
                                      if k not in a else "VALUE CHANGED"),
                 "head_len": len(str(a.get(k, ""))), "worktree_len": len(str(b.get(k, "")))}
                for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if "/registry/" in str(args.out.resolve()).replace("\\", "/"):
        raise SystemExit("refusing to write inside the registry - this instrument is read-only")

    files = [diff_file(f"research/00-system/registry/{n}", k) for n, k in LIST_KEY.items()]
    packets = diff_packets()
    result = {
        "schema_version": 1,
        "head_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                      capture_output=True, text=True).stdout.strip(),
        "files": files,
        "packets": packets,
        "writes_nothing_to_the_registry": True,
        "passed": all(not f.get("field_changes") and not f.get("swallowed_mappings")
                      and f.get("comment_lines", {}).get("lost", 0) <= 0 for f in files),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for f in files:
        print(f"  {f['file']}")
        if f.get("status"):
            print(f"      {f['status']}")
            continue
        c = f["comment_lines"]
        print(f"      comments head={c['head']} worktree={c['worktree']} LOST={c['lost']}")
        print(f"      entries head={f['entry_counts']['head']} "
              f"worktree={f['entry_counts']['worktree']}  "
              f"added={f['entries_added_in_worktree']} missing={f['entries_missing_from_worktree']}")
        for ch in f["field_changes"]:
            print(f"      {ch['kind']}: {ch['id']}.{ch['field']}")
            print(f"          HEAD     = {ch['head']}")
            print(f"          WORKTREE = {ch['worktree']}")
        for s in f["swallowed_mappings"]:
            print(f"      SWALLOWED ENTRY: {s['hidden_id']} is parsed as "
                  f"{s['swallowed_into']}.{s['field']}[{s['index']}] - invisible to every consumer")
        if f["restore_from_head_would_lose"]:
            print(f"      RESTORE WOULD LOSE: {f['restore_from_head_would_lose']}")
    for p in packets:
        print(f"  {p['file']}: {p.get('status') or [c['field'] for c in p.get('field_changes', [])]}")
    print(f"{HEARTBEAT} clean={result['passed']} -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
