---
id: 01-research-direction/scientific-questions
title: Scientific Questions
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- questions
---

# Scientific Questions

> The concrete, answerable questions whose resolution moves the mission.

## Q1 — Does arm-B × harmonic clear the gate on the *deployment* substrate?

The motion-residual flow gate validated **+0.0080** on P0-strict LOEO (both folds, P(d>0)=1.0),
**+0.0088** on E0c. But arm-B and harmonic **both consume `prob`** and interact — they must be
measured together, never assumed additive. Measured via the two built LOEO-export kernels.
**Falsified if** not bilaterally positive vs P3-alone (44b6 `0.759549` / 6bba `0.648965`).
→ [bet-motion-gate](research-bets.md)

## Q2 — Does retraining the detector/associator on external **Zebrahub** beat the 50-epoch wall?

The public stack plateaus at `adj_edge_J ~0.90–0.91`. Does same-domain external training push
past it, LOEO-validated on the patched scorer? → [bet-zebrahub-retrain](research-bets.md)

## Q3 — Can a **learned FP-suppressing candidate ranker** fix precision (not recall)?

DoG already recovers 0.91–0.94 of nuclei; the junk candidate pool mis-links. *"A ranker, not a
threshold."* Marginal scalar precision ~0.003 where the ceiling needs ~0.55.
→ [bet-learned-ranker](research-bets.md)

## Q4 — Is there a high-precision fork selector that banks the **+0.06** division ceiling?

Division GT-oracle ceiling is +0.06 but needs ~10% precision at full recall; 304 events →
shakeup-prone. Late, precision-gated. → [bet-division-selector](research-bets.md)
