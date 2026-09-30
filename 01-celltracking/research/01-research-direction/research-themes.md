---
id: 01-research-direction/research-themes
title: The Research Themes
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- themes
---

# The Research Themes

> Durable lines of attack that group individual [bets](research-bets.md).

## Theme A — External same-domain training (the program)

Retrain the detector/associator on external Zebrahub imaging + dense Ultrack lineages. The only
measured path off the public plateau. Members: `bet-zebrahub-retrain`, `bet-learned-ranker`.

## Theme B — Learned FP-suppression / candidate ranking

The failure is **precision, not recall**: a learned ranker over an over-proposed candidate pool
restores precision on the rare/small/low-confidence candidates. Members: `bet-learned-ranker`.

## Theme C — Synthetic / dense supervision

Freitas 18.5GB CC0 (165k divisions) + dense pseudo-labels as division supervision feeding
Themes A/B. Members: `bet-synthetic-division`.

## Theme D — Division recovery

A high-precision fork selector to bank the +0.06 division ceiling — late, precision-gated,
shakeup-aware. Members: `bet-division-selector`.

## Theme E — Cheap post-processing harvest (near-exhausted)

Squeeze remaining un-shipped positive post-processing (the motion-residual gate) onto the
deployment. Bounded upside (~+0.008). Members: `bet-motion-gate`.
