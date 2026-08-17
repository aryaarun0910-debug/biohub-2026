---
id: 01-research-direction/research-bets
title: Research Bets
area: 01-research-direction
status: active
updated: '2026-08-17'
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
| `bet-motion-gate` | E | **ready** | +0.0088 pooled (min-fold +0.0074) | low | not bilaterally positive vs P3-alone on LOEO |
| `bet-subvoxel-refine` | E | proposed | several pts **if** currently integer-argmax | low | detector already sub-voxel-refines, or no LOEO gain |
| `bet-ot-linker` | B | proposed | edge + node + division jointly | medium | unbalanced-OT ≤ current linker cross-family (fit A / eval B) |
| `bet-meta-ranker` | B | proposed | precision on junk pool (cheap) | low | GBM/nnPU AUC ≤ scalar cross-family (fit A / eval B) |
| `bet-node-budget` | E | closed* | +0.001 bilateral (settled, sub-bar) | low | SETTLED: 199-crop sweep opt = keep_frac 0.975, +0.001, an order below bar |
| `bet-zebrahub-retrain` | A | active | past the ~0.91 adj_edge_J wall | high | no LOEO gain over public 50ep weights |
| `bet-learned-ranker` | B | proposed | restore precision on junk pool | high | ranker AUC no better than the scalar threshold it replaces |
| `bet-synthetic-division` | C | proposed | division supervision for A/B | medium | synthetic divisions don't transfer to real folds |
| `bet-division-selector` | D | parked | +0.06 ceiling (needs ~10% prec @ full recall) | medium | fork precision below ~10% break-even |

## Closed (do not reopen without a new mechanism + a stated falsification)

- `bet-scalar-reacceptance` — **lost** (−0.023). Detection recovery is not a threshold.
- `bet-component-selector` — **lost** (GT-free selector −0.008).
- `bet-splitmerge` — **lost** (+0.0006, trivial).
- `edge-TTA` — **lost** (hurts, 0.885; community-confirmed).
- (`bet-node-budget` was here; **reopened** 2026-08-17 — its kill used the forbidden
  placeholder-movie substrate. See table + detail.)

## Detail

- **bet-motion-gate** — flow-compensated eligibility predicate in the wrapper's `motion_relink`
  (gate on `|target-(source+flow(source))|`). Validated +0.0080 P0-strict LOEO (both folds,
  P(d>0)=1.0), +0.0088 E0c. **Swarm (2026-08-17) elevated this to the #1 ready win.** New
  insight: armB's gate is **pure geometry — it never reads `prob`**; harmonic only re-ranks
  cost within an unchanged eligible set (second-order). Cheap de-risk: re-run armB on local
  `p0strict` graphs with the `prob` cost term zeroed (~1 h CPU); if Δ stays bilateral-positive,
  harmonic can't break it → then the two built `p3_armb` LOEO kernels confirm on GPU.
- **bet-subvoxel-refine** — community EDA measured a localization cliff (3 µm → −41 %, 4 µm →
  −74 %). If our detector emits integer-voxel/coarse-argmax centroids, add parabolic/Gaussian
  sub-voxel refinement. **First step is an audit** (CPU minutes) of whether we already refine.
- **bet-ot-linker** — replace greedy/threshold linking with an entropic **unbalanced-Sinkhorn**
  transport plan (POT `ot.sinkhorn_unbalanced`); relaxed marginals give appearance/disappearance,
  mass-splitting = division, the mass knob calibrates N_pred vs N_est. CPU; τ calibrated on A,
  applied to B (cross-family test built in).
- **bet-meta-ranker** — GBM/shallow-MLP on existing candidate+edge features (MetaDetect/LMD) +
  **nnPU loss** for the 2.8 %-annotation PU regime + conformal/FDR-bounded acceptance. Seconds
  CPU; the cheap "ranker not threshold." Cross-family via fit-A/eval-B.
- **bet-node-budget** — the old "closed (optimum = no pruning)" verdict was **false** (it used
  four placeholder movies, a forbidden selection substrate). **Settled 2026-08-17** on the full
  199-crop P0-strict sweep: optimum at keep_frac 0.975 on all three folds, but only **+0.001
  bilateral** (44b6 +0.00110 / 6bba +0.00087) — an order below the +0.005 bar. Honestly reopened,
  then re-closed as a sub-bar footnote (correct evidence now).
- **bet-zebrahub-retrain** — H1. The real gap-closer (edge retrain); level-1 lane in progress,
  see [directional-updates.md](directional-updates.md).
- **bet-learned-ranker** / **bet-synthetic-division** / **bet-division-selector** — H2/H3/H4.
  Division levers are minority-of-gap and soundly dead on both families/factors; do not reopen a
  division head without a mother-gate at AUC ≥ 0.97.
