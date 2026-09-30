---
tags:
  - gotcha
  - finding
---

# Gotcha: the in-kernel sweep is inert once relink is off

The notebook runs an 8-candidate post-processing sweep on held-out train films
before producing its submission. **Five of the eight candidates are
[[Motion Relink]] knobs**, which cannot do anything with the stage switched off.

From `artifacts/s05_output/ppsweep_results.csv` — identical to the last digit:

| candidate | proxy |
|---|---|
| `base` | 0.9715059814960677 |
| `tight55` | 0.9715059814960677 |
| `relaxed9` | 0.9715059814960677 |
| `bonus125` | 0.9715059814960677 |
| `gap2step40` | 0.9715059814960677 |
| `reuse28` | 0.9715059814960677 |

`ppsweep_selected.json` picks `base` with `{}` overrides. The two that *do* move
(`dcgap035`, `gap45`) differ by ~1e-5.

**Cost: ~8 × 250 s ≈ 33 minutes** of every relink-off run, for nothing.

This upgrades a previously "untested but low risk" runtime cut to **evidenced** —
for relink-off variants only. `gap2step40` and `reuse28` tying base also
re-confirms [[Inert Variables]].

Related: [[Kaggle Mechanics]]
