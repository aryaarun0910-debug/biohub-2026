"""The claims table must stay resolvable against its artifacts.

This is the enforcement half of scripts/core/claims_table.py. Without a test, a renamed key or a
moved artifact would only be noticed the next time somebody happened to regenerate the table
-- which is exactly the delay that let four overstated headlines survive a whole cycle.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_claim_still_resolves():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "core" / "claims_table.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, (
        "A claim in scripts/core/claims_table.py no longer resolves against its artifact.\n"
        "Fix the path (or the artifact) -- do NOT hand-edit research/06-knowledge-system/claims-table.md.\n"
        f"{r.stdout}\n{r.stderr}"
    )


def test_every_claim_carries_a_legal_basis_tag():
    sys.path.insert(0, str(ROOT / "scripts"))
    import claims_table as ct

    for sec, label, _art, _path, basis, _fmt, _note in ct.CLAIMS:
        assert basis in ct.ALLOWED_BASES, (
            f"claim {label!r} in section {sec!r} carries basis {basis!r}, which is not one of "
            f"{ct.ALLOWED_BASES}. Every number must declare how it was measured."
        )


def test_generated_table_is_current():
    """research/06-knowledge-system/claims-table.md must match what the generator produces right now."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import claims_table as ct

    text, problems = ct.build()
    assert not problems, f"claims drifted: {problems}"
    out = ROOT / "research" / "06-knowledge-system" / "claims-table.md"
    assert out.exists(), "research/06-knowledge-system/claims-table.md missing -- run scripts/core/claims_table.py"
    on_disk = out.read_text(encoding="utf-8")
    # The generated header carries a date; compare everything after it so a stale date alone
    # does not fail the suite, while any changed VALUE does.
    def body(s: str) -> str:
        return s.split("## ", 1)[-1] if "## " in s else s
    assert body(on_disk) == body(text), (
        "research/06-knowledge-system/claims-table.md is stale relative to its artifacts. "
        "Regenerate with: .\\.venv\\Scripts\\python.exe scripts\\claims_table.py"
    )
