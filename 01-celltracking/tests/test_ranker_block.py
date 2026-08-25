"""Contract tests for the vendored local-association ranker loader.

Software contracts only. Whether the ranker is worth a submission slot is an experiment
result, not a unit test.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

MODULE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "kaggle_edits" / "ranker_block.py"


@pytest.fixture(scope="module")
def source() -> str:
    return MODULE.read_text(encoding="utf-8")


def test_module_parses(source):
    ast.parse(source)


def test_defines_the_loader_and_the_fallback_mlp(source):
    tree = ast.parse(source)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "_PublicLocalAssociationRanker" in classes
    # the fallback path matters: checkpoints that store a bare state_dict have no module
    assert "_InferredRankerMLP" in classes


def test_loader_exposes_predict_proba(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_PublicLocalAssociationRanker":
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            assert "predict_proba" in methods
            return
    pytest.fail("_PublicLocalAssociationRanker not found")


def test_kaggle_root_discovery_is_omitted(source):
    """This file must be definitions-only so it can be exercised off-Kaggle.

    The upstream notebook ends with a `_ranker_candidate_roots()` call that raises unless
    /kaggle/input is populated. Re-introducing it would make the module unimportable
    locally and silently break these tests.
    """
    assert "_ranker_roots = _ranker_candidate_roots()" not in source


def test_no_future_import_that_would_break_injection(source):
    """`from __future__` is only legal at the top of a file; injected mid-notebook it is a
    SyntaxError. It was stripped during extraction and must stay stripped."""
    assert "from __future__" not in source


def test_contamination_warning_is_documented(source):
    """The manifest records datasets_seen=199 with a dataset-level split, so this ranker has
    seen most of data/train. If that caveat is ever dropped from the docstring, someone will
    validate it locally and read an inflated number as a promotion signal."""
    head = source[:4000]
    assert "CONTAMINATED" in head.upper()
    assert "199" in head


def test_author_usage_constraint_is_recorded(source):
    assert "tie-breaker" in source[:4000]
