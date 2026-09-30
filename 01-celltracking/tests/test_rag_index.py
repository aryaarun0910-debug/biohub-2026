"""THE RAG INDEX'S CONTRACTS.

The index is a generated view under `.claude/rag/` (gitignored, per the project rule). What must
hold is not "it returns something" but that it cannot become a second, staler source of truth than
the registry:

  * a live VERIFIED/MEASURED registry entity outranks prose that merely mentions it;
  * SUPERSEDED and INVALID content is demoted and carries a REFUSAL, so it cannot be quoted as
    current state;
  * every result carries an entity ID and a path, so any answer can be checked at source;
  * no binary artifact or notebook body is indexed.

The index is rebuilt in a temp directory here rather than read from `.claude/rag/`, so these
contracts hold on a fresh clone that has never run the builder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

import rag_index as R  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    chunks = R.build_corpus()
    return chunks, R.build_index(chunks)


def test_the_corpus_indexes_every_entity_class(corpus):
    chunks, _ = corpus
    kinds = {c["kind"] for c in chunks}
    for required in ("ROLE", "FACT", "LEVER", "EXP", "PKT", "NOTEBOOK", "SCRIPT", "TEST",
                     "doc_section"):
        assert required in kinds, f"the corpus indexes no {required} chunks"


def test_no_binary_or_notebook_body_is_indexed(corpus):
    """The contract is about CONTENT, not about which path a chunk names.

    A NOTEBOOK chunk legitimately POINTS AT an .ipynb - that is where the artifact lives - while
    indexing only its metadata: status, roles, spec, parent, features, delta. The first version of
    this test asserted no chunk's path ends in .ipynb and failed on exactly that legitimate case.
    What must never happen is a notebook's CELLS entering the corpus, so the assertion is on the
    text.
    """
    chunks, _ = corpus
    for c in chunks:
        text = c["text"]
        assert '"cell_type"' not in text and '"outputs"' not in text, (
            f"{c['id']} appears to contain raw notebook JSON")
        if str(c["path"]).lower().endswith(R.SKIP_SUFFIX):
            assert c["kind"] in ("NOTEBOOK",), (
                f"{c['id']} indexes a binary artifact at {c['path']}")
            # A length bound would be arbitrary - a notebook that sets 60 environment variables
            # has a long METADATA line and that is fine. The structural property is that the
            # chunk is a single constructed summary line, never transcribed file content.
            assert "\n" not in text, (
                f"{c['id']} contains newlines, so it is transcribed content rather than the "
                f"single-line metadata summary the builder constructs")


def test_every_chunk_carries_provenance_metadata_and_an_id(corpus):
    chunks, _ = corpus
    for c in chunks:
        assert c.get("id") and c.get("path"), "a chunk with no id or path cannot be checked"
        for key in ("provenance", "validity", "superseded_by"):
            assert key in c, f"{c['id']} carries no {key} - ranking could not demote it"


def test_superseded_content_is_demoted_and_carries_a_refusal(corpus):
    """A superseded number returned without a banner is how stale state gets quoted as current."""
    chunks, idx = corpus
    sup = [c for c in chunks if c.get("superseded_by")]
    assert sup, "no superseded chunk in the corpus - this guard would be vacuous"
    clean = next(c for c in chunks if c["provenance"] == "VERIFIED" and not c.get("superseded_by"))
    assert R.rank_multiplier(sup[0]) < R.rank_multiplier(clean) * 0.5, (
        "superseded content is not meaningfully demoted")
    faked = dict(sup[0])
    res = R.search(idx, chunks + [faked], faked["text"][:80], 8)
    for r in res:
        if r["id"] == faked["id"]:
            assert "REFUSAL" in r, "a superseded hit was returned with no refusal banner"


def test_invalid_facts_rank_below_valid_ones(corpus):
    chunks, _ = corpus
    valid = next(c for c in chunks if c["kind"] == "FACT" and c["validity"] == "VALID")
    bad = [c for c in chunks if c["kind"] == "FACT" and c["validity"] == "INVALID"]
    if bad:
        assert R.rank_multiplier(bad[0]) < R.rank_multiplier(valid)


def test_identifier_tokens_are_split_so_prose_queries_can_reach_them():
    """`BIOHUB_MOTION_RELINK_LEARNED_BONUS` must be reachable from 'motion relink learned bonus'."""
    t = R.tok("BIOHUB_MOTION_RELINK_LEARNED_BONUS")
    for part in ("motion", "relink", "learned", "bonus"):
        assert part in t, f"{part!r} not emitted; a natural-language query could never match"
    assert "biohub_motion_relink_learned_bonus" in t, "the exact identifier form was lost"


def test_the_validation_queries_all_resolve(corpus):
    """The questions agents repeatedly need answered, and the answer must be registry-backed."""
    chunks, idx = corpus
    misses = []
    for q, expect in R.VALIDATION_QUERIES:
        ids = [r["id"] for r in R.search(idx, chunks, q, 5)]
        if expect and expect not in ids:
            misses.append((q, expect, ids[:3]))
        elif not expect and not ids:
            misses.append((q, "any result", []))
    assert not misses, f"validation queries unsatisfied: {misses}"


def test_the_operational_base_query_returns_the_manifest_not_prose_about_it(corpus):
    chunks, idx = corpus
    top = R.search(idx, chunks, "what is the operational base", 1)[0]
    assert top["id"] == "ROLE-operational_base", f"top hit was {top['id']}"
    assert top["path"].endswith("baseline_roles.yaml")
    assert top["provenance"] == "VERIFIED"
