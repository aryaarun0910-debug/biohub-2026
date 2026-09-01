"""THE THREE PROPERTIES `test_provenance_policy.py` DOES NOT COVER.

That file proves the two guards AGREE and that mutating the policy moves both. Three things it
does not prove, each of which is a way the repair could rot back into FACT-0431:

  1. A LOCAL TABLE COULD BE REINTRODUCED. `hasattr` catches a module-level constant, but the
     defect would come back just as easily as a dict literal inside a clause body. This file
     reads the SOURCE, so a reintroduction anywhere in the module fails.
  2. THE POLICY IMPORT COULD BE MADE OPTIONAL. A `try: import provenance_policy / except:
     pass` would leave every DATA clause silently unchecked and every test in the sibling file
     still green, because they all import the policy themselves. A guard that quietly stops
     guarding is the exact failure class this contract exists to prevent (FACT-0432, FACT-0425,
     FACT-0454, FACT-0457, FACT-0459 - five instances, all green while wrong).
  3. THE TWO RECEIPTS COULD DISAGREE ABOUT WHICH POLICY RAN. Agreement today is not evidence
     that a later reader can tell WHICH policy produced a stored receipt pair.

Software contracts only (CLAUDE.md rule 4). Nothing here promotes or demotes a scientific result.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import audit_feature_cache as A        # noqa: E402
import gpu_protection_contract as G    # noqa: E402
import provenance_policy as PP         # noqa: E402

GUARD2 = ROOT / "scripts" / "win_bet" / "gpu_protection_contract.py"
GUARD1 = ROOT / "scripts" / "win_bet" / "audit_feature_cache.py"

#: names that were local truth tables, and the shape of any successor
RETIRED_NAMES = ("INVALID_PAIR", "LEGITIMATE_TRUNKS", "PKT0029_REQUIRED_TRUNK_ROLES")


# ------------------------------------------------------------------------------------------
# 1. REINTRODUCING A LOCAL TABLE FAILS - checked in the SOURCE, not only via hasattr
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [GUARD2, GUARD1], ids=["guard2", "guard1"])
def test_no_guard_assigns_a_retired_policy_table_anywhere(path):
    """A module-level `hasattr` check misses a dict literal built inside a clause body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id in RETIRED_NAMES:
                offenders.append(f"{path.name}:{node.lineno} assigns {t.id}")
    assert not offenders, (
        "a retired local legitimacy table has been reintroduced: " + "; ".join(offenders) +
        ". FACT-0431 is two guards answering one question differently; the answer lives in "
        "provenance_policy and nowhere else."
    )


def test_the_retired_names_are_absent_as_attributes_too():
    for name in RETIRED_NAMES:
        assert not hasattr(G, name), f"gpu_protection_contract.{name} is back"
        assert not hasattr(A, name), f"audit_feature_cache.{name} is back"


def test_both_guards_actually_import_the_policy_module():
    """Consuming the policy has to be visible in the source, not inferred from behaviour."""
    for path in (GUARD1, GUARD2):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)
        } | {
            n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
        }
        assert "provenance_policy" in imports, f"{path.name} does not import provenance_policy"


# ------------------------------------------------------------------------------------------
# 2. A MISSING POLICY CRASHES - it does not silently skip the check
# ------------------------------------------------------------------------------------------
def test_a_missing_policy_import_crashes_rather_than_skipping():
    """Run in a subprocess: block `provenance_policy`, then import the guard.

    An in-process monkeypatch cannot prove this - the module is already imported. The subprocess
    boundary is the only honest way to ask what a fresh interpreter does, and this repository has
    twice been bitten by state that did not cross exactly such a boundary (FACT-0060, FACT-0387).
    """
    # `find_module`/`load_module` were REMOVED in Python 3.12, so a finder written against the
    # legacy protocol is never consulted and this test would pass vacuously while proving
    # nothing. `find_spec` is the only hook that still fires. The first draft of this test made
    # exactly that mistake and reported the guard as unprotected; keeping the note because a
    # blocker that does not block is the same shape as a guard that does not guard.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'provenance_policy':\n"
        "            raise ImportError('provenance_policy is unavailable')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, r'{(ROOT / 'scripts' / 'win_bet').as_posix()}')\n"
        "try:\n"
        "    import provenance_policy\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    print('BLOCKER_DID_NOT_BLOCK')\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    import gpu_protection_contract\n"
        "except ImportError:\n"
        "    print('CRASHED_AS_REQUIRED')\n"
        "else:\n"
        "    print('IMPORTED_WITHOUT_THE_POLICY')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=180)
    assert "BLOCKER_DID_NOT_BLOCK" not in out.stdout, (
        "the import blocker itself failed, so this test proves nothing about the guard")
    assert "CRASHED_AS_REQUIRED" in out.stdout, (
        "gpu_protection_contract imported with NO policy available, so every DATA clause would "
        f"run unchecked. stdout={out.stdout!r} stderr={out.stderr[-400:]!r}"
    )


# ------------------------------------------------------------------------------------------
# 3. BOTH RECEIPTS CARRY THE SAME POLICY IDENTITY
# ------------------------------------------------------------------------------------------
def test_the_policy_stamp_is_well_formed():
    stamp = PP.policy_stamp()
    assert stamp["policy_version"] == "provenance_policy_v1"
    assert len(stamp["policy_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in stamp["policy_sha256"])


def test_guard1_receipt_carries_the_policy_stamp():
    """Guard 1's receipt is `PP.describe(fold)` - the object it stores after accepting a pair."""
    described = PP.describe("1")
    for key in ("policy_version", "policy_sha256", "policy_module"):
        assert key in described, f"guard 1's policy receipt has no {key}"


def test_guard2_receipt_carries_the_policy_stamp():
    spec = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "1"}}]}
    trunk = {"role": "oof_split1", "fold_legitimate": True,
             "dual_trunk_pair": ["oof_split1", "stabledet"], "contamination": "none",
             "checkpoint_sha256": "0" * 16}
    clauses = {c["id"]: c for c in G.section_data(spec, trunk)}
    for cid in ("DATA-2", "DATA-3", "DATA-4"):
        ev = clauses[cid]["evidence"]
        assert ev.get("policy_version") == PP.POLICY_VERSION, f"{cid} carries no policy version"
        assert ev.get("policy_sha256") == PP.policy_digest(), f"{cid} carries a stale policy hash"


def test_both_guards_report_the_SAME_policy_version_and_hash():
    """The property that makes a stored receipt pair auditable after the fact."""
    g1 = PP.describe("1")
    spec = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "1"}}]}
    trunk = {"role": "oof_split1", "fold_legitimate": True,
             "dual_trunk_pair": ["oof_split1", "stabledet"], "contamination": "none",
             "checkpoint_sha256": "0" * 16}
    g2 = {c["id"]: c for c in G.section_data(spec, trunk)}["DATA-3"]["evidence"]
    assert g1["policy_version"] == g2["policy_version"]
    assert g1["policy_sha256"] == g2["policy_sha256"]


def test_the_stamp_moves_when_the_policy_moves(monkeypatch):
    """A version that does not move on a policy change would be worse than no version at all."""
    before = PP.policy_digest()
    patched = {k: dict(v) for k, v in PP.ROLES.items()}
    patched["stabledet"]["claim_folds"] = {"1"}
    monkeypatch.setattr(PP, "ROLES", patched)
    after = PP.policy_digest()
    assert before != after, (
        "the policy digest is blind to a change in the decision table, so two receipts could "
        "report matching 'versions' while the guards acted on different policies"
    )
