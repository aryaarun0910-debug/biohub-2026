"""THE CANONICAL CATALOG'S OWN CONTRACTS.

The catalog is a generated VIEW. Its value depends entirely on three properties, and none of them
is self-evident from looking at the output:

  1. IT COVERS EVERYTHING. A catalog that silently omits a directory is worse than no catalog,
     because it answers "is this file known?" with a confident no.
  2. IT COPIES NO NUMBER. The registry is the only source of truth for scientific values
     (AGENTS.md section 1). A generated file is an especially dangerous place to duplicate one,
     since it looks authoritative and regenerates without review.
  3. IT INVENTS NO PROVENANCE. Where a binding cannot be established the entry says `unknown` or
     `unbound-historical`. Retrospectively attaching an EXP or a receipt to a historical artifact
     would manufacture evidence, which is the one thing the registry exists to prevent.

Software contracts only (CLAUDE.md rule 4).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

import catalog as C  # noqa: E402

CAT = ROOT / "research" / "00-system" / "registry" / "generated" / "catalog"


def load(name: str) -> dict:
    p = CAT / f"{name}.json"
    assert p.is_file(), f"{p} missing - run scripts/core/catalog.py"
    return json.loads(p.read_text(encoding="utf-8"))[name]


# ------------------------------------------------------------------------------------------
# 1. COVERAGE - every target file has an entry
# ------------------------------------------------------------------------------------------
def test_every_script_file_is_catalogued():
    on_disk = {p.relative_to(ROOT).as_posix()
               for p in (ROOT / "scripts").rglob("*")
               if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    missing = sorted(on_disk - set(load("scripts")))
    assert not missing, f"{len(missing)} script files absent from the catalog: {missing[:8]}"


def test_every_test_file_is_catalogued():
    on_disk = {p.relative_to(ROOT).as_posix()
               for p in (ROOT / "tests").rglob("*.py") if "__pycache__" not in p.parts}
    missing = sorted(on_disk - set(load("tests")))
    assert not missing, f"{len(missing)} test files absent: {missing[:8]}"


def test_every_notebook_directory_is_catalogued():
    on_disk = {p.name for p in (ROOT / "notebooks").iterdir() if p.is_dir()}
    missing = sorted(on_disk - set(load("notebooks")))
    assert not missing, f"{len(missing)} notebook directories absent: {missing[:8]}"


def test_every_research_markdown_is_catalogued():
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "research").rglob("*.md")}
    missing = sorted(on_disk - set(load("research_docs")))
    assert not missing, f"{len(missing)} research documents absent: {missing[:8]}"


def test_every_registry_entity_is_catalogued():
    reg = load("registry")
    import yaml
    src = ROOT / "research" / "00-system" / "registry"
    n_facts = len(yaml.safe_load((src / "facts.yaml").read_text("utf-8"))["facts"])
    n_levers = len(yaml.safe_load((src / "levers.yaml").read_text("utf-8"))["levers"])
    assert len(reg["facts"]) == n_facts
    assert len(reg["levers"]) == n_levers


# ------------------------------------------------------------------------------------------
# 2. EVERY ENTRY HAS A STATUS, AND `unknown` IS ALLOWED
# ------------------------------------------------------------------------------------------
def test_every_script_has_a_lifecycle_and_a_basis():
    for path, e in load("scripts").items():
        assert e.get("lifecycle"), f"{path} has no lifecycle"
        assert e.get("lifecycle_basis"), f"{path} states no basis for its lifecycle"


def test_every_notebook_has_a_status_and_a_basis():
    for name, e in load("notebooks").items():
        assert e.get("status"), f"{name} has no status"
        assert e.get("status_basis"), f"{name} states no basis for its status"


# ------------------------------------------------------------------------------------------
# 3. NO SCORE IS DUPLICATED
# ------------------------------------------------------------------------------------------
def _numbers(obj, path=""):
    """Every NUMERIC leaf with its json path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


@pytest.mark.parametrize("name", ["registry", "notebooks", "specs", "scripts", "tests",
                                  "research_docs"])
def test_no_catalog_file_stores_a_score_as_a_value(name):
    """The guard is on NUMERIC FIELDS, not on any 0.9xx in the text, and the distinction matters.

    A first version flagged every `0.9xx` anywhere and fired on `BIOHUB_DET_THRESHOLD` (0.965) and
    on lever prose quoting a historical score. A detection threshold is CONFIGURATION, not a
    score, and refusing it would have been a false positive of exactly the kind this project
    keeps paying for. What must never happen is the catalog STORING a score as its own field -
    that is the second maintained truth AGENTS.md forbids.

    Lever claim prose is not copied at all (see catalog.build_registry), so the remaining prose is
    each artifact quoting its OWN docstring, where the artifact is already the source.
    """
    # Match the LEAF KEY only. Matching anywhere in the json path fires on file NAMES that
    # contain the word - `scripts/core/score_oof.py.bytes` is a byte count, not a score.
    payload = json.loads((CAT / f"{name}.json").read_text(encoding="utf-8"))
    leaf = re.compile(r"^(score|lb|public_lb|leaderboard|lb_score|public_score)$", re.I)
    offenders = [(p, v) for p, v in _numbers(payload) if leaf.match(p.rsplit(".", 1)[-1])]
    assert not offenders, (
        f"{name}.json stores score-valued fields {offenders[:5]}. Link the FACT/EXP id instead.")


def test_the_registry_view_carries_no_values_only_pointers():
    for fid, f in load("registry")["facts"].items():
        assert "value" not in f, f"{fid} copied its value into the catalog"
        assert f.get("value_lives_in", "").endswith("facts.yaml")


# ------------------------------------------------------------------------------------------
# 4. NO INVENTED PROVENANCE
# ------------------------------------------------------------------------------------------
def test_unbound_historical_notebooks_have_no_fabricated_bindings():
    for name, e in load("notebooks").items():
        if e["status"] == "unbound-historical":
            assert not e["experiments"], f"{name} is unbound yet lists experiments"
            assert not e["submissions"], f"{name} is unbound yet lists submissions"
            assert not e["facts"], f"{name} is unbound yet lists facts"


def test_notebook_experiment_links_resolve():
    reg = load("registry")["experiments"]
    for name, e in load("notebooks").items():
        for eid in e["experiments"]:
            assert eid in reg, f"{name} links a non-existent experiment {eid}"


def test_script_fact_links_resolve():
    facts = load("registry")["facts"]
    for path, e in load("scripts").items():
        for fid in e["measures_facts"]:
            assert fid in facts, f"{path} links a non-existent fact {fid}"


def test_the_baseline_roles_appear_in_the_notebook_catalog():
    """A notebook can hold MORE THAN ONE role, and here one holds two."""
    roles = [r for e in load("notebooks").values() for r in (e.get("roles") or [])]
    assert "leaderboard_champion" in roles and "operational_base" in roles, (
        f"the catalog lost a baseline role: {sorted(set(roles))}")


# ------------------------------------------------------------------------------------------
# 5. DRIFT
# ------------------------------------------------------------------------------------------
def test_the_catalog_has_no_drift():
    """Regenerating must reproduce the committed files byte for byte."""
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "core" / "catalog.py"), "--check"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    assert rc.returncode == 0, (
        "the catalog disagrees with a fresh generation. Either the repository changed and the "
        f"catalog was not regenerated, or it was hand-edited.\n{rc.stdout[-800:]}")
