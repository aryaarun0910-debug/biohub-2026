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
record_kind: state
---

# Scientific Mission

> The single objective we optimise, the win condition, and the constraints that bound it.

## Mission reframed, 2026-08-29 (host)

**The objective is no longer a rank or a boundary score. It is the machine's own attainable
ceiling.** The host's direction: pursue the maximum this system can reach, agnostically and for the
science, on the view that the work matters beyond the standing of any entry. A leaderboard position
is now a by-product of that pursuit, not the target it is measured against.

This is a better-posed objective because it is measurable offline rather than by inference from
other teams. The yardstick is the **ceiling ladder**: perfect one stage at a time and score each
rung through the official metric (`scripts/win_bet/ceiling_ladder.py`). As measured on the complete
fold 0, the remaining distance decomposes as association 87.0%, node selection 5.9%, detection 7.1%
(`FACT-0368`), and perfect association on today's node set is worth +0.180 of score once the
division term is counted (`FACT-0371`). Effort is allocated against those shares.

The top-3 framing below is **retained as history and as a calibration reference** - `FACT-0332` is
still the live boundary reading, and the deadline and submission constraints are unchanged and still
binding. What changed is which number the campaign is trying to move.

## Mission (original framing, retained for provenance)

Climb the Biohub Cell Tracking public leaderboard to **top-3** (private-set-honest), from a
reproducible **0.925** system (P9 coupled division, FACT-0001). Leader **0.962**
(FACT-0011); top-3 boundary **0.953** (FACT-0010); gap to close **+0.028** (FACT-0012);
rank **207 / 2,693** (FACT-0003). Superseded reading: 0.915 at leader 0.950, top-3 0.948,
gap +0.033, rank ~143 on the then-current plateau.

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
