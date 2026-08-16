---
id: 01-research-direction/directional-updates
title: Directional Updates
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- steering
- log
---

# Directional Updates

> Dated steering log. [`../00-system/handoff.md`](../00-system/handoff.md) points here for the current direction. Newest first.

## 2026-08-16 — Reopen + research-machine restructure

- **Reopened** the project (reverses the 2026-08-07 closure). Mission: top-3 (see
  [scientific-mission.md](scientific-mission.md)). Frontier cracked by the swarm: the edge is a
  retrained/generalising model on external Zebrahub, unlocked 2026-08-13 (six days after we quit).
- **Restructured** the repo into this research machine (one folder; `_RESEARCH` folded in;
  `reports/` absorbed). Storage is 3-tier (`/temp` → `_evidence/` → `research/`).
- **In flight:** `bet-motion-gate` — two P3+armB LOEO-export kernels built + audited locally
  (folds 0/1, 18/18 edits each). Kaggle GPU run **parked** for explicit green-light. On a
  bilaterally-positive result vs P3-alone, build the submission kernel + factory `submitcmd`.
- **Next program:** `bet-zebrahub-retrain` + `bet-learned-ranker` (GPU T4×2), building on the
  restored `scripts/win_bet/h1t_zebrahub_events.py` + `h1t_conditional_ranker.py`.

## Prior chronology

Deployment lineage E0c (0.889) → v122 (0.908) → P0-A (0.913) → P0-B (0.914) → **P3 harmonic
(0.915, deployed)**. Full history: [../06-knowledge-system/lab-notebooks.md](../06-knowledge-system/lab-notebooks.md).
