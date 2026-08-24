---
id: 00-system/readme
title: The Research Machine — Map
area: 00-system
status: active
updated: 2026-08-24
owner: biohub
links: []
tags: [meta, map]
---

# The Research Machine

This directory is the project's **scientific operating system**. It holds the knowledge —
narrative in Markdown, machine-readable spine in YAML/JSON — while the working code
(`../src`, `../scripts`, `../tests`, `../notebooks`) is the **instrument** and the raw
evidence lives in the gitignored `../_evidence/` store. One folder, one source of truth.

The scope is **rigid**: the tree is declared in [`system.yaml`](system.yaml) and enforced by
[`../scripts/core/validate_research_tree.py`](../scripts/core/validate_research_tree.py). New files must
be added to the manifest, or the validator (and its test) fail. This is deliberate — it is the
same "fail loudly on drift" discipline as the claims table.

## Repository at a glance

Deployed public score **0.915** (P3 harmonic); leader 0.962, top-3 boundary 0.953, gap **+0.038**
(2026-08-24 15:52 UTC).
Live direction: [`00-system/handoff.md`](00-system/handoff.md) →
[`01-research-direction/directional-updates.md`](01-research-direction/directional-updates.md).

| Path | What |
|---|---|
| `research/` | This knowledge machine — 8 ordered areas (00 + 01–07). |
| `../src/biotrack/` | Immutable scorer/graph core + a partial wrapper mirror; not the deployed program. |
| `../scripts/`, `../tests/`, `../notebooks/` | Tooling, software-contract tests, and Kaggle kernels; built notebooks are the deployed artifacts. |
| `../config/` | Dependency pins (`requirements.txt`, `requirements.lock.txt`). |
| `../_evidence/` | **Gitignored** durable-raw store (agent runs, snapshots, caches, `research.sqlite`). |
| `../.claude/rag/` | **Gitignored** RAG artifacts (embeddings, indexes) — policy in [`04-data/databases.md`](04-data/databases.md). |
| `../data/`, `../artifacts/`, `../weights/`, `../vendor/`, `../.venv/` | **Gitignored** large working locals. |
| Root: `../CLAUDE.md` · `../pyproject.toml` · `../.gitignore` · `../.gitattributes` · `../.github/` | The only load-bearing files left at the repo root. |

Recovery tags: `pre-lean-2026-07-30`, `pre-lean-2026-08-07`, `biohub-closed-2026-08-07`,
`pre-restructure-2026-08-16`.

## The areas (pipeline order)

| # | Area | What lives here |
|---|------|-----------------|
| 00 | **System** | The operating contract: **handoff** (live entry point) and **system-design** (architecture). *How the machine runs.* |
| 01 | **Research Direction** | Mission, questions, landscape, themes, **bets**, prioritisation, portfolio, evaluation, directional updates. *Where we decide what to chase.* |
| 02 | **Theory** | Existing knowledge, definitions + **metric semantics**, assumptions, **mechanisms**, models, derivations, predictions, competing explanations, **falsifiability**, theory revision. *Why a thing should work.* |
| 03 | **Experimentation** | Question → hypothesis → variables → design → protocol → materials → instrumentation → execution → QC → power → replication → failure analysis → iteration. *How we test it.* |
| 04 | **Data** | Acquisition, **storage (3-tier)**, cleaning, **metadata (corpus census)**, databases, versioning, **governance (external-data licensing)**. *What we test on.* |
| 05 | **Analysis** | Statistics, mathematical analysis, simulation, **ML/DL**, visualisation, error/uncertainty. *How we read the result.* |
| 06 | **Knowledge System** | Papers, lab notebooks, **experimental records (ledger)**, internal reports, **failed experiments**, results (+ generated **claims table**), **institutional memory (research.sqlite)**. *What we have learned.* |
| 07 | **Outputs** | Submissions, deployed artifacts, figures, manuscript, leaderboard. *What we ship.* |

## Storage tiers

- **Transient → `/temp`** (the session scratchpad): raw logs, intermediate dumps, anything
  regenerable. Never committed, never in the repo tree.
- **Durable-raw → `../_evidence/`** (in-repo, **gitignored**): agent runs, snapshots, caches,
  full experiment output, `research.sqlite`. One physical home; git stays lean.
- **Tracked-knowledge → `research/`** (git): narrative markdown + the machine-readable spine +
  the small `06-knowledge-system/inventory/*.json` artifacts the claims table reads.

## The frontmatter contract

Every `.md` opens with a YAML block conforming to
[`_schema/frontmatter.schema.json`](_schema/frontmatter.schema.json):

```yaml
---
id: 02-theory/mechanisms      # area/slug
title: Mechanisms
area: 02-theory
status: active                # active | scaffold | reference | generated | superseded
updated: 2026-08-16
owner: biohub
links: []                     # related ids or paths
tags: []
---
```

Cross-link related notes inline with their id or relative path. Numbers never get typed by
hand into a tracked doc — they come from an artifact via the claims table
([`06-knowledge-system/results.md`](06-knowledge-system/results.md) →
[`06-knowledge-system/claims-table.md`](06-knowledge-system/claims-table.md)).

## Machine-readable spine

- [`system.yaml`](system.yaml) — the manifest: every area + file, its purpose and status.
- [`_schema/`](_schema/) — JSON Schemas for frontmatter, bet records, experiment rows.
- [`01-research-direction/bets.yaml`](01-research-direction/bets.yaml) — the portfolio as
  structured records (mirrors `research-bets.md`).
- `06-knowledge-system/research.sqlite` — the 100+-finding DB (gitignored; see
  [`06-knowledge-system/institutional-memory.md`](06-knowledge-system/institutional-memory.md)).

## Relationship to the operating contract

[`00-system/handoff.md`](00-system/handoff.md) is the live entry point and points into
[`01-research-direction/directional-updates.md`](01-research-direction/directional-updates.md).
[`00-system/system-design.md`](00-system/system-design.md) documents this architecture.
[`../CLAUDE.md`](../CLAUDE.md) is the execution contract — the only load-bearing doc left at the
repo root. The working code surface is unchanged and is described in
[`03-experimentation/instrumentation.md`](03-experimentation/instrumentation.md).

## Validate

```powershell
.\.venv\Scripts\python.exe scripts\core\validate_research_tree.py   # tree == manifest, frontmatter valid
.\.venv\Scripts\python.exe -m pytest -q tests\test_research_tree.py
```
