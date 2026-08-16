---
id: 00-system/handoff
title: Handoff
area: 00-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags: [handoff, entry-point]
---

# Current handoff

**Status:** REOPENED 2026-08-16, restructured into the `research/` machine. Branch `master`.

This file is the live entry point. The current direction, in full, is in
[directional-updates.md](../01-research-direction/directional-updates.md).
The system map is [README.md](../README.md); architecture is
[system-design.md](system-design.md); the execution contract is [CLAUDE.md](../../CLAUDE.md).

## Mission

Aggressive climb toward **top-3** (private-set-honest). We hold a reproducible **0.915** system
(P3 harmonic); leader **0.950**, top-3 boundary **0.948**, gap **+0.033**.
See [scientific-mission.md](../01-research-direction/scientific-mission.md).

## Immediate queue (LOEO-gated on the patched scorer before any GPU submission)

0. **`bet-motion-gate` (measuring):** two P3+armB LOEO-export kernels built + audited locally
   (folds 0/1, 18/18 edits). **Kaggle GPU run parked** for explicit green-light. On a
   bilaterally-positive result vs P3-alone (44b6 `0.759549` / 6bba `0.648965`), build the
   submission kernel + factory `submitcmd`.
1. **The program (`bet-zebrahub-retrain` + `bet-learned-ranker`, GPU T4×2):** retrain the
   detector + a learned FP-suppressing candidate ranker on external Zebrahub (+ synthetic),
   LOEO-validated. The only measured path off the plateau.
2. **`bet-division-selector` (parked):** +~0.02, shakeup-prone; only after edges clear.
3. Keep P3 harmonic 0.915 as the frozen hedge.

Portfolio + falsifications: [research-bets.md](../01-research-direction/research-bets.md)
(and `bets.yaml`). Prioritisation + gate: [prioritisation.md](../01-research-direction/prioritisation.md).

## Guardrails (prize-critical)

- Never infer hidden-set quality from the four visible placeholder movies.
- The unmatched-fork division-evaluator pathology is diagnostic ONLY — never in a submission.
- Exact public-trajectory transfer into an identified hidden crop needs written host clearance.
- Preserve `.claude/settings.json`. Stage explicit paths; never `git add -A`.
- Smoke → representative pilot → full LOEO; a passing smoke is not scientific evidence.
- Full guardrails: [data-governance.md](../04-data/data-governance.md).

## Verification

```powershell
git status
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\core\claims_table.py --check
.\.venv\Scripts\python.exe scripts\core\validate_research_tree.py
```
