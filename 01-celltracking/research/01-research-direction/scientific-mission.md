---
id: 01-research-direction/scientific-mission
title: Scientific Mission
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- mission
---

# Scientific Mission

> The single objective we optimise, the win condition, and the constraints that bound it.

## Mission

Climb the Biohub Cell Tracking public leaderboard to **top-3** (private-set-honest), from a
reproducible **0.915** system (P3 harmonic). Leader **0.950**; top-3 boundary **0.948**;
gap to close **+0.033**; rank at reopen ~143 (tied on the 0.915 public plateau).

## Win condition

A submission whose *private* score lands top-3 at the **2026-09-29** deadline, selected on
leave-one-embryo-out (LOEO) CV with the **patched scorer** — never on public-LB probing alone.
Final entry / team-merge cutoff **2026-09-22**.

## Objective

`weighted_avg(adjusted_edge_jaccard) + 0.1 * division_jaccard`. Both embryo directions are
reported separately; promotion needs **both folds non-regressive**. Exact semantics live in
[../02-theory/concepts-and-definitions.md](../02-theory/concepts-and-definitions.md).

## Constraints

- Notebook-only, internet-off rerun; 5 submissions/day; team ≤ 5.
- External data allowed (Zebrahub imaging + tracks host-confirmed 2026-08-13; synthetic CC0).
- **Never** infer hidden-set quality from the four visible placeholder movies (in-sample, biased).
- Guardrails: [../04-data/data-governance.md](../04-data/data-governance.md),
  [../03-experimentation/quality-control.md](../03-experimentation/quality-control.md).

## Why we can win from here

The 0.93–0.950 tier's edge is a **retrained / generalising edge model** enabled by external
same-domain data unlocked **2026-08-13 — six days after we closed on 2026-08-07**. The biggest
lever the top tier rides did not exist as a legal option when we quit. See
[research-landscape.md](research-landscape.md).
