---
id: 01-research-direction/prioritisation
title: Prioritisation
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- prioritisation
- gating
record_kind: state
---

# Prioritisation

> How [bets](research-bets.md) are ranked and gated.

## Ranking (EV × novelty × cheap-to-falsify)

1. **bet-motion-gate** — free/low cost, banks a small real gain off the deployed score (FACT-0001) while the retrain
   spins up. *Measuring now.*
2. **bet-zebrahub-retrain + bet-learned-ranker** — the program; highest EV; the only measured
   path off the plateau. Run together (retrain feeds the ranker's candidate pool).
3. **bet-synthetic-division** — supports (2) with division supervision.
4. **bet-division-selector** — late, precision-gated, shakeup-prone.

## Gating rule (prize-critical)

No GPU submission until a candidate clears **≥ +0.005 min-fold** on **patched-scorer LOEO**,
both embryo directions non-regressive, no regime-slice regression. Sequence is always
**smoke → representative pilot → full LOEO**; a passing smoke is not scientific evidence.

## Current gate state

- bet-motion-gate: two LOEO-export kernels **built + audited locally**; awaiting explicit
  green-light for the Kaggle GPU run. Anchors to beat: 44b6 `0.759549`, 6bba `0.648965`.
