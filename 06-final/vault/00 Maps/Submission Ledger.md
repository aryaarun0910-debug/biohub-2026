---
tags:
  - moc
---

# Submission Ledger

Recorded in `artifacts/ledger/runs.db` (SQLite: `runs`, `scores`,
`division_events`, `submissions`). **Predictions are written before results**, so
sign agreement stays honest.

## Ours, this lineage
| id | change | parent | local | board |
|---|---|---|---|---|
| [[s01]] | `SAFE_DIV_DIVERGE_UM` 2.25→0 | base | — | rejected by validator |
| s02/s03/s04 | safe-div symmetry variants | base | — | retired, never pushed |
| [[s05]] | `OUTPUT_MOTION_RELINK`=0 | base | +0.0224 | **PENDING** |
| [[s06]] | `OUTPUT_LINEFIT_WEIGHT` 0.8→0.4 | base | **−0.00135** | not pushed |
| [[s07]] | gap2 after safe_div | base | — | **superseded, wrong base** |
| [[s08]] | gap2 after safe_div | [[s05]] | +0.00748 | running |
| [[s09]] | `OUTPUT_LINEFIT_WEIGHT` 0.8→0.3 | [[s08]] | +0.00487 | running |

## Board history (earlier work)
0.947 — `MIN_TRACK_LEN` 6→9 · 0.947 EXP-19 · 0.947 verbatim repro ·
0.946 ppgrid · 0.946 EXP-33 · 0.932 learned-bonus 2.0 · 0.932 DeepCenter veto ·
0.931 public bridge · 0.928 p24 · 0.925 p9 · 0.925 p22 · 0.924 / 0.922 detection
threshold · 0.914 p27 · 0.912 P3 · **0.906 `SAFE_DIVISIONS=0`** · 0.496 untrained.

The 0.906 row is the useful one: switching divisions off costs 0.041, which sizes
the division axis.

Related: [[Axis Priors]], [[One Change Per Submission]]
