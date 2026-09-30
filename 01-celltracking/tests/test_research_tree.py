"""Structural contract: the research/ machine matches its manifest and every doc's
frontmatter is valid. This is a software invariant (tree integrity), not a scientific
promotion decision. See scripts/core/validate_research_tree.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_research_tree_matches_manifest_and_frontmatter_valid():
    import validate_research_tree as v
    errors = v.validate()
    assert not errors, "research-tree drift:\n" + "\n".join(errors)
