---
id: 04-data/databases
title: Databases
area: 04-data
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- databases
- schema
---

# Databases

> Structured stores backing the machine.

## research.sqlite (gitignored, in-repo)

`../06-knowledge-system/research.sqlite` — tables `finding` (102 rows) and `agent_run` (7 rows).
Query recipes + schema in
[../06-knowledge-system/institutional-memory.md](../06-knowledge-system/institutional-memory.md).

## Machine-readable spine (tracked)

- [../system.yaml](../system.yaml) — the tree manifest.
- [../_schema/](../_schema/) — JSON Schemas (frontmatter, bet, experiment).
- [../01-research-direction/bets.yaml](../01-research-direction/bets.yaml) — the portfolio.
- `../06-knowledge-system/inventory/*.json` — machine artifacts the claims table reads.

Integrity is enforced by `../../scripts/core/validate_research_tree.py` (tree == manifest;
frontmatter valid) and its pytest.

## RAG store (gitignored)

Retrieval artifacts — embeddings, vector stores / indexes, chunk caches — live in
`../../.claude/rag/` (**gitignored**; large/binary, regenerable). This is the policy home for
anything RAG-related.

- **Corpus:** the tracked `research/**.md` knowledge machine (+ `system.yaml` as the manifest).
- **Rebuild, never commit:** indexes are regenerated from `research/`; only this policy is tracked.
- **Local pointer:** `.claude/rag/README.md` documents the folder for anyone browsing it.
