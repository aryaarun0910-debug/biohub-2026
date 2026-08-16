---
id: 06-knowledge-system/results
title: Results
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- results
---

# Results

> Curated results narrative. **No number here is typed by hand** — the artifact-backed table is
> generated into [claims-table.md](claims-table.md) by `../../scripts/core/claims_table.py` from
> `inventory/*.json`. Regenerate + verify:
> `.\.venv\Scripts\python.exe scripts\core\claims_table.py --check`.

## Headline

- **Deployment: P3 harmonic = 0.915 public** (lineage E0c 0.889 → v122 0.908 → P0-A 0.913 →
  P0-B 0.914 → P3 0.915).
- **LOEO substrate anchors** (the trustable base, reproduced exactly): 44b6 `0.759549`,
  6bba `0.648965`.
- **bet-motion-gate**: +0.0088 pooled (min-fold +0.0074) on E0c; +0.0080 P0-strict LOEO
  (P(d>0)=1.0). *Being re-measured on the deployment substrate together with harmonic.*

## How to read this area

The generated [claims-table.md](claims-table.md) is authoritative for every measured number,
each tagged with exactly one of the six legal bases (`public`, `exact-pooled-OOF`,
`cross-family-LOFO`, `in-family-CV`, `placeholder-proxy`, `GT-oracle`). Narrative context and
interpretation live here and in [../01-research-direction/](../01-research-direction/).
