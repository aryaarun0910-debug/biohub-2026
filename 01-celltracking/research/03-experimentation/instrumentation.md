---
id: 03-experimentation/instrumentation
title: Instrumentation
area: 03-experimentation
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- code
- surface
---

# Instrumentation

> The code instruments (the "apparatus"). Code is unchanged by the restructure; it lives at the
> repo root, referenced from here.

## Competition core — `../../src/biotrack/`

Immutable scorer/graph core + the deployed `wrapper.py` (harmonic fusion, `motion_relink`, arm-B
flow gate, DeepCenter veto). Changes only for a measured scoring mechanism or a correctness defect.

## Scripts — `../../scripts/`

- `kaggle_factory.py` — disciplined build/push/`submitcmd` (never auto-submits).
- `d1_postprocess.py` — scorer-exact M/C/T/L/D partition.
- `d1f_probe.py` — representation-vs-head diagnosis.
- `score_oof.py` — score exported `*.geff` vs `data/train`.
- `claims_table.py` — generates `../06-knowledge-system/claims-table.md` from
  `../06-knowledge-system/inventory/*.json`; `--check` fails on drift.
- `validate_research_tree.py` — tree == manifest + frontmatter valid (this machine's integrity).
- `win_bet/` — the learned-ranker / Zebrahub workstream (`h1t_conditional_ranker.py`,
  `h1t_zebrahub_events.py`, …).
- `build_d1_factorial_manifests.py`, `assemble_p3_d1_smoke_spec.py`, `kaggle_edits/`,
  `kaggle_specs/` — active kernel patches and specs.

## Tests — `../../tests/`

Enforce **software contracts only** (scorer parity, graph invariants, manifest determinism,
provenance, serialization, byte parity, and now research-tree integrity). Never a scientific
promotion threshold.
