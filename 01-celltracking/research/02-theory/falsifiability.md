---
id: 02-theory/falsifiability
title: Falsifiability
area: 02-theory
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- falsification
---

# Falsifiability

> For each live theory/bet, the cheapest experiment that would kill it. A bet without a
> falsification does not get compute.

| bet / theory | cheapest kill experiment | kill threshold |
|---|---|---|
| `bet-motion-gate` | the two built P3+armB LOEO-export kernels, scored vs P3-alone | not bilaterally positive (either fold ≤ its P3-alone anchor) |
| `bet-zebrahub-retrain` | train on 6bba, eval 44b6 (and reverse) with patched scorer | no LOEO gain over public 50ep weights |
| `bet-learned-ranker` | fit ranker on one family, eval the other; ROC vs the scalar it replaces | AUC ≤ scalar-threshold baseline (cross-family) |
| `bet-synthetic-division` | add synthetic divisions to training, LOEO on real folds | real-fold division_J does not rise |
| `bet-division-selector` | fork-selector precision/recall vs the 10% break-even | precision < ~10% at full recall |

## Standing negative results (already falsified — see [../06-knowledge-system/failed-experiments.md](../06-knowledge-system/failed-experiments.md))

Scalar re-acceptance (−0.023), GT-free component selector (−0.008), node-budget pruning
(optimum = none), split/merge arbitration (+0.0006), edge-TTA (hurts, 0.885). Reopening any of
these requires a **new mechanism** and a fresh falsification.
