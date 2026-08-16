---
id: 04-data/storage
title: Storage
area: 04-data
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- storage
- tiers
---

# Storage

> The three-tier storage model. One physical folder; git stays lean.

## Tiers

1. **Transient → `/temp`** (session scratchpad,
   `…\AppData\Local\Temp\claude\…\scratchpad`): raw logs, intermediate dumps, scratch
   scripts, anything regenerable. **Never committed, never in the repo tree.**
2. **Durable-raw → `../../_evidence/`** (in-repo, **gitignored**): agent runs, source
   snapshots, environments, kaggle runs, exports, derived features, caches, `research.sqlite`.
   Same-drive; folded in from the old `_RESEARCH` sibling.
3. **Tracked-knowledge → `../` (research/)** (git): narrative markdown + the machine-readable
   spine + the small `../06-knowledge-system/inventory/*.json` artifacts the claims table reads.

## Also gitignored (unchanged working locals)

`../../data/` (84G competition dataset, reproducible via kaggle download), `../../artifacts/`
(4G caches + OOF geffs), `../../.venv/`, `../../weights/`, `../../vendor/`.

## Rule

If an output is regenerable, it goes to `/temp`. If it is durable raw evidence, it goes to
`_evidence/`. Only curated knowledge and small code-read artifacts are tracked in `research/`.
