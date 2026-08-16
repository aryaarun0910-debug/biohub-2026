---
id: 01-research-direction/research-bets
title: Research Bets
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- bets
- portfolio
---

# Research Bets

> The live portfolio. Machine-readable mirror: [bets.yaml](bets.yaml) (validated against
> [../_schema/bet.schema.json](../_schema/bet.schema.json)). Each bet carries a falsification.

| id | theme | status | expected | cost | falsification |
|---|---|---|---|---|---|
| `bet-motion-gate` | E | measuring | +0.0088 pooled (min-fold +0.0074) | low | not bilaterally positive vs P3-alone on LOEO |
| `bet-zebrahub-retrain` | A | proposed | past the ~0.91 adj_edge_J wall | high | no LOEO gain over public 50ep weights |
| `bet-learned-ranker` | B | proposed | restore precision on junk pool | high | ranker AUC no better than the scalar threshold it replaces |
| `bet-synthetic-division` | C | proposed | division supervision for A/B | medium | synthetic divisions don't transfer to real folds |
| `bet-division-selector` | D | parked | +0.06 ceiling (needs ~10% prec @ full recall) | medium | fork precision below ~10% break-even |

## Closed (do not reopen without a new mechanism + a stated falsification)

- `bet-scalar-reacceptance` — **lost** (−0.023). Detection recovery is not a threshold.
- `bet-component-selector` — **lost** (GT-free selector −0.008).
- `bet-node-budget` — **lost** (optimum = no pruning).
- `bet-splitmerge` — **lost** (+0.0006, trivial).
- `edge-TTA` — **lost** (hurts, 0.885; community-confirmed).

## Detail

- **bet-motion-gate** — flow-compensated eligibility predicate in the wrapper's `motion_relink`
  (gate on `|target-(source+flow(source))|`). Validated +0.0080 P0-strict LOEO (both folds,
  P(d>0)=1.0), +0.0088 E0c. **Interacts with harmonic** (both touch `prob`) → measured together
  via the two built LOEO-export kernels. Currently *measuring* (Kaggle GPU green-light parked).
- **bet-zebrahub-retrain** — H1. Detector recall lever: over-propose + learned re-scoring.
- **bet-learned-ranker** — H2/our own gap. "A ranker, not a threshold."
- **bet-synthetic-division** — H3. Pooling-matched to the evaluator's `vol[:, ::4, ::4]` stride.
- **bet-division-selector** — H4. Only after edges clear; watch CV/LB divergence.
