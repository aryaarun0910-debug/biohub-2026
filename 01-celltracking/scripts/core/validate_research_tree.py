r"""Validate the research/ machine against its manifest. Fail loudly on drift.

Same philosophy as claims_table.py: the tree is DECLARED in research/system.yaml and this
script asserts reality matches the declaration, so the "rigid scope" is enforceable rather
than aspirational. Checks:

  1. every area .md declared in system.yaml exists, and every area .md present is declared
     (no orphan files, no missing files);
  2. declared non-.md spine files (bets.yaml) exist;
  3. every area .md carries YAML frontmatter conforming to _schema/frontmatter.schema.json,
     with id == "<area>/<slug>" and area == its directory;
  4. bets.yaml records conform to _schema/bet.schema.json.

Generated files (claims-table.md) and machine/evidence subdirs (inventory/, internal-reports/,
research.sqlite) are exempt from the frontmatter check. README.md and system.yaml are meta.

Usage:
  .\.venv\Scripts\python.exe scripts\core\validate_research_tree.py          # print report, exit 1 on drift
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
RES = ROOT / "research"
SCHEMA = RES / "_schema"
GENERATED = {"claims-table.md"}  # generated, no frontmatter


def _load_frontmatter(path: Path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None, "no YAML frontmatter block"
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, "unterminated frontmatter block"
    try:
        return yaml.safe_load(text[4:end + 1]), None
    except yaml.YAMLError as e:  # pragma: no cover - defensive
        return None, f"frontmatter YAML error: {e}"


def validate() -> list[str]:
    errors: list[str] = []

    manifest = yaml.safe_load((RES / "system.yaml").read_text(encoding="utf-8"))
    fm_validator = Draft202012Validator(json.loads((SCHEMA / "frontmatter.schema.json").read_text("utf-8")))
    bet_validator = Draft202012Validator(json.loads((SCHEMA / "bet.schema.json").read_text("utf-8")))

    declared_md: set[str] = set()
    declared_other: set[str] = set()
    for area in manifest["areas"]:
        aid = area["id"]
        for f in area["files"]:
            name = f["file"]
            (declared_md if name.endswith(".md") else declared_other).add(f"{aid}/{name}")

    # 1. present vs declared (area docs = direct .md children of NN-* dirs)
    present_md = {
        f"{p.parent.name}/{p.name}"
        for p in RES.glob("[0-9][0-9]-*/*.md")
    }
    for missing in sorted(declared_md - present_md):
        errors.append(f"DECLARED-BUT-MISSING: {missing}")
    for orphan in sorted(present_md - declared_md):
        errors.append(f"PRESENT-BUT-UNDECLARED: {orphan} (add it to system.yaml)")

    # 2. declared spine files exist
    for other in sorted(declared_other):
        if not (RES / other).exists():
            errors.append(f"DECLARED-BUT-MISSING (spine): {other}")

    # 3. frontmatter on every present area doc (except generated)
    for rel in sorted(present_md):
        if Path(rel).name in GENERATED:
            continue
        path = RES / rel
        fm, err = _load_frontmatter(path)
        if err:
            errors.append(f"FRONTMATTER {rel}: {err}")
            continue
        schema_errs = sorted(fm_validator.iter_errors(fm), key=lambda e: e.path)
        for e in schema_errs:
            errors.append(f"FRONTMATTER {rel}: {e.message}")
        if not schema_errs:
            area_dir = Path(rel).parent.name
            slug = Path(rel).stem
            if fm.get("area") != area_dir:
                errors.append(f"FRONTMATTER {rel}: area '{fm.get('area')}' != dir '{area_dir}'")
            if fm.get("id") != f"{area_dir}/{slug}":
                errors.append(f"FRONTMATTER {rel}: id '{fm.get('id')}' != '{area_dir}/{slug}'")

    # 4. bets.yaml records
    bets_path = RES / "01-research-direction" / "bets.yaml"
    if bets_path.exists():
        bets = yaml.safe_load(bets_path.read_text(encoding="utf-8")) or {}
        for i, rec in enumerate(bets.get("bets", [])):
            for e in bet_validator.iter_errors(rec):
                errors.append(f"BETS bets[{i}] ({rec.get('id', '?')}): {e.message}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"research-tree validation FAILED ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    n_md = len(list(RES.glob("[0-9][0-9]-*/*.md")))
    print(f"research-tree OK: {n_md} area docs declared and valid; spine + bets conform.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
