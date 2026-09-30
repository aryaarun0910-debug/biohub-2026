---
id: 06-knowledge-system/failed-experiments
title: Failed Experiments
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- negative
- graveyard
record_kind: ledger
---

# Failed Experiments

> The graveyard: measured-negative levers and falsified findings, with why. Kept so we never
> re-pay for a closed lever. Source of truth for findings: `research.sqlite` (see
> [institutional-memory.md](institutional-memory.md); 5 findings marked *falsified*).

## Closed levers (measured negative / trivial)

| lever | result | basis | why it died |
|---|---|---|---|
| scalar-threshold re-acceptance | **−0.023** | LOEO | detection recovery is a ranker, not a threshold |
| GT-free component selector | **−0.008** | LOEO | selector removes more signal than junk |
| node-budget pruning | optimum = **no pruning** | pooled-OOF | any pruning regresses |
| split/merge arbitration | **+0.0006** | LOEO | below the noise floor |
| edge-TTA | **hurts (0.885)** | public + community | augmented views degrade the linker |

## Detours (mechanism does not transfer as a drop-in)

- **HOCT** as a drop-in linker — community-tested, **underperformed a tuned ILP** (#728551).
  Salvageable only as a division edge-head idea.
- **CoTracker** — loses morphology through divisions.

## Rule

Do not reopen a closed lever without (a) a new mechanism and (b) a stated falsification in
[../02-theory/falsifiability.md](../02-theory/falsifiability.md).
