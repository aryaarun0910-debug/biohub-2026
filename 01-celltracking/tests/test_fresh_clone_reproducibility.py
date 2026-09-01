"""WHAT A FRESH CLONE CAN DO WITH TRACKED FILES ALONE.

Measured on a clean clone of `e5db190`, before this file existed:

  * `validate_registry.py`      exit 1, four R3 errors
  * `pytest --collect-only`     exit 2, INTERRUPTED, zero tests run

Both traced to `vendor/` - a PINNED EXTERNAL CHECKOUT, gitignored by design, with no bootstrap and
no diagnostic. The gates were green on the development machine because of files git does not
carry, which is precisely the condition R3 exists to forbid. Nothing detected the divergence,
because nothing ever asked the question from outside this working tree.

These tests ask it from inside, cheaply, so the answer cannot drift again:

  1. the production import surface resolves from tracked files;
  2. the FACT-0451 enforcement is COMMITTED, not just present locally - the failure that made a
     fresh clone lose the guard with no error at all;
  3. every external checkout is declared, pinned, and explains itself when absent;
  4. no tracked module imports something the repository deleted.

Software contracts only (CLAUDE.md rule 4). Nothing here runs a scientific gate.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import bootstrap_vendor as BV  # noqa: E402


def _tracked(pattern: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", pattern], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    return [line for line in out.stdout.splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------
# 1. THE FACT-0451 ENFORCEMENT MUST BE COMMITTED, NOT MERELY PRESENT
# ------------------------------------------------------------------------------------------
def test_the_fact_0451_enforcement_set_is_tracked_by_git():
    """The halves are only safe together; at e5db190 the module was untracked and the guard gone.

    `os.path.exists` cannot see this failure - the file was on disk, so every local check passed.
    Only `git ls-files` distinguishes "present" from "a fresh clone will have it".
    """
    required = [
        "scripts/win_bet/spec_restriction.py",
        "tests/test_spec_restriction.py",
    ]
    missing = [r for r in required if not _tracked(r)]
    assert not missing, (
        f"{missing} exist on disk but are NOT TRACKED. A fresh clone would keep the specs that "
        f"declare binding_restriction and lose the module that enforces it - silently, with no "
        f"import error anywhere. That is the exact state FACT-0451 was written to end."
    )


def test_every_spec_declaring_the_restriction_is_tracked_alongside_its_enforcer():
    declaring = [p for p in _tracked("scripts/win_bet/assoc_specs/*.json")
                 if "binding_restriction" in (ROOT / p).read_text(encoding="utf-8")]
    if declaring:
        assert _tracked("scripts/win_bet/spec_restriction.py"), (
            f"{len(declaring)} TRACKED specs declare binding_restriction but the enforcing module "
            f"is not tracked - the halves have been split across commits"
        )


# ------------------------------------------------------------------------------------------
# 2. EXTERNAL CHECKOUTS ARE DECLARED, PINNED, AND EXPLAIN THEMSELVES
# ------------------------------------------------------------------------------------------
def test_every_declared_external_checkout_has_an_unambiguous_pin():
    for name, spec in BV.CHECKOUTS.items():
        commit, source = BV.declared_pin(spec["pin_package"])
        assert len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit), (name, commit)
        assert source, f"{name}: the pin names no declaring file"


def test_an_absent_checkout_produces_a_repair_not_a_symptom():
    """The diagnostic must name the command, the URL and the commit - not 'file not found'."""
    text = BV.diagnostic("vendor/kaggle-cell-tracking")
    for token in ("bootstrap_vendor.py --apply", "PINNED EXTERNAL CHECKOUT",
                  "kaggle-cell-tracking-competition"):
        assert token in text, f"the diagnostic does not mention {token!r}: {text}"


def test_registry_validation_does_not_fail_merely_because_a_pinned_checkout_is_absent():
    """R3's promise is reproducibility, and a pinned checkout IS reproducible.

    Guarded so it still proves something when the checkout is present: in that case the paths
    resolve and R3 has nothing to say about them either way.
    """
    import validate_registry as VR
    for name in BV.CHECKOUTS:
        owner = VR._external_checkout_for(name + "/src/anything.py")
        assert owner == name, f"R3 does not recognise {name} as an external checkout"
    assert VR._external_checkout_for("scripts/win_bet/spec_restriction.py") is None, (
        "a repository path must never be excused as an external checkout"
    )


# ------------------------------------------------------------------------------------------
# 3. NO TRACKED MODULE IMPORTS SOMETHING THE REPOSITORY DELETED
# ------------------------------------------------------------------------------------------
def test_no_tracked_script_imports_a_module_this_repository_deleted():
    """`private_split_simulator` was deleted by 7143ef0 and its caller kept importing it.

    The caller is RETIRED rather than stubbed, so the import must be guarded and must raise a
    named retirement status - never resolve to a plausible substitute.
    """
    deleted = {"private_split_simulator"}
    offenders = []
    for rel in _tracked("scripts/*.py") + _tracked("scripts/**/*.py"):
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = set()
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            if names & deleted:
                # a guarded import that re-raises with a retirement status is the sanctioned form
                guarded = any(isinstance(a, ast.Try) and node in ast.walk(a)
                              for a in ast.walk(tree))
                if not guarded:
                    offenders.append(f"{rel}:{node.lineno} imports {sorted(names & deleted)}")
    assert not offenders, (
        "unguarded imports of deleted modules: " + "; ".join(offenders) +
        ". Guard them with a named retirement status; do NOT add a compatibility stub, which "
        "would return plausible output with nothing behind it."
    )


def test_the_retired_caller_states_its_status():
    src = (ROOT / "scripts" / "metric" / "pooled_bootstrap.py").read_text(encoding="utf-8")
    assert "RETIREMENT_STATUS" in src and "historical" in src
    assert "7143ef0" in src, "the retirement does not name the commit that caused it"


# ------------------------------------------------------------------------------------------
# 4. THE PRODUCTION IMPORT SURFACE RESOLVES
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("module", [
    "spec_restriction", "provenance_policy", "gpu_protection_contract",
    "assoc_train_harness", "assoc_tournament",
])
def test_core_production_modules_import(module):
    __import__(module)
