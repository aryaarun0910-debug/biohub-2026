---
id: 00-system/system-design
title: System Design
area: 00-system
status: active
updated: '2026-08-24'
owner: biohub
links: []
tags: [architecture]
---

# System design

## Goal

Move the official score with the smallest reliable experiment, inside a knowledge system that
is **explainable** (documented, navigable) and **rigid in scope** (the tree is declared and
enforced). One physical folder. The repository is an execution product plus its research
memory, not a transcript of every conversation.

## Two things in one folder

- **The instrument** — the working code: `src/biotrack/` (immutable scorer/graph core + a
  partial wrapper mirror), `scripts/`, `tests/`, `notebooks/`. The built Kaggle notebook is the
  deployed program and must be audited directly; neither it nor `src/biotrack/wrapper.py` is a
  superset of the other. Unchanged by the knowledge layer.
- **The research machine** — `research/`, a scientific-method operating system in ordered
  areas. Narrative in Markdown; machine-readable spine in YAML/JSON. Mapped by
  [README.md](../README.md); declared by [system.yaml](../system.yaml).

## The areas (pipeline order)

`00-system` (contract: handoff, this file) → `01-research-direction` → `02-theory` →
`03-experimentation` → `04-data` → `05-analysis` → `06-knowledge-system` → `07-outputs`.
Direction decides what to chase; theory says why it should work; experimentation tests it;
data and analysis are what we test on and how we read it; the knowledge system is what we have
learned; outputs are what we ship.

## Root is minimal by necessity

Only load-bearing files stay at the repo root: `CLAUDE.md` (Claude Code auto-loads it),
`pyproject.toml` (build/test marker), `.gitignore` / `.gitattributes` (git reads from root),
and `.github/` (Actions). Everything movable was folded in: docs → `research/00-system/`,
dependency pins → `config/`, RAG artifacts → `.claude/rag/` (gitignored).

## Three-tier storage

1. **Transient → `/temp`** (session scratchpad): regenerable logs and intermediates. Never
   committed, never in the tree.
2. **Durable-raw → `_evidence/`** (in-repo, **gitignored**): agent runs, snapshots,
   environments, caches, full experiment output, `research.sqlite`. Folded in from the old
   `_RESEARCH` sibling — evidence lives here, one physical home.
3. **Tracked-knowledge → `research/`** (git): narrative + spine + the small
   `06-knowledge-system/inventory/*.json` artifacts the claims table reads.

Git also ignores the large working locals: `data/` (87GB), `artifacts/`, `.venv/`, `weights/`,
`vendor/`, and the RAG store `.claude/rag/`.

## The rigidity contract

- Every `research/**.md` carries YAML frontmatter conforming to
  `research/_schema/frontmatter.schema.json`.
- Every area file is declared in `research/system.yaml`; no orphans, no missing files.
- `scripts/core/validate_research_tree.py` (+ `tests/test_research_tree.py`) enforces both, and
  validates `bets.yaml` against `research/_schema/bet.schema.json`. Same "fail loudly on drift"
  discipline as the claims table.

## Experiment factory

Every experiment has one machine-readable spec, one immutable input manifest, one output
artifact directory outside Git (`_evidence/` or `/temp`), and one compact ledger row in
`research/06-knowledge-system/experimental-records.md` containing:

- hypothesis and parent;
- code/config/input hashes;
- validation basis (one of the six legal tags);
- pooled and per-family metrics;
- runtime and resource use;
- decision and falsification reason.

Smoke proves plumbing. A representative pilot estimates a mechanism. Full LOEO confirms it.
These stages must never be conflated. `scripts/core/kaggle_factory.py` builds/pushes kernels and
**never auto-submits** — `submitcmd` prints a command for a human.

## No hand-typed numbers

Every tracked number comes from an artifact. `scripts/core/claims_table.py` generates
`research/06-knowledge-system/claims-table.md` from `inventory/*.json` and fails loudly on a
missing path, a moved artifact, or an illegal basis tag. Narrative lives in `results.md`; the
generated table is authoritative for values.

## Test architecture

Tests may block execution only for software invariants:

- official scorer parity;
- graph degree and coordinate validity;
- deterministic manifests and joins;
- checkpoint/family provenance;
- serialization schemas;
- baseline byte parity;
- research-tree structural integrity (manifest match + frontmatter).

Tests must not encode a scientific conclusion such as "method X can never work", a score
promotion threshold, or a temporary research policy. Those belong in evidence and decision
records (`research/06-knowledge-system/`). A new hypothesis may run in an isolated experimental
path without weakening core invariants.

## Language policy

Python remains the control plane, model language, scorer language, and Kaggle packaging
language. Polars/NumPy/PyTorch already execute their expensive kernels in native code.

Rust or C++ is permitted only when all of the following hold:

1. profiling shows one stable CPU function consumes at least 30% of end-to-end wall time;
2. algorithmic and vectorised Python improvements are exhausted;
3. a pure Python reference and property/parity tests exist;
4. the compiled artifact can be reproduced inside the internet-off Kaggle environment;
5. expected saved compute exceeds integration and packaging cost.

Use Rust for safe parallel graph/candidate kernels; use C++/CUDA only for an unavoidable
PyTorch extension. TypeScript has no role in the scoring pipeline. Polyglot code is an
optimisation result, never an architectural starting point.

## Preservation / recovery

Git tags preserve pre-change trees: `pre-lean-2026-07-30`, `pre-lean-2026-08-07`,
`biohub-closed-2026-08-07`, and `pre-restructure-2026-08-16` (the tree just before the
research-machine restructure).
