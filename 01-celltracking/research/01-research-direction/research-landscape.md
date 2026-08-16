---
id: 01-research-direction/research-landscape
title: The Research Landscape
area: 01-research-direction
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- landscape
- competitive
---

# The Research Landscape

> The competitive and scientific terrain. Full evidence:
> [../06-knowledge-system/internal-reports/competitive_frontier_2026-08-16.md](../06-knowledge-system/internal-reports/competitive_frontier_2026-08-16.md).

## Leaderboard (2426 teams; notebook-only; $60k)

Top-3 boundary **0.948** (leader Mark Cooper 0.950; TWEAK 0.949; Soheil Ayati 0.948).
Distribution: ≥0.950: 1 · ≥0.948: 3 · ≥0.945: 7 · ≥0.940: 10 · ≥0.935: 16 · ≥0.930: 27 (top 1.1%)
· ≥0.920: 73. Then a **cliff: 0.915 = 302 teams**, ≥0.915 = 447 teams (top 18.4%) = the
**public-notebook plateau**, where **we sit**. Gap to top-3 = **+0.033**.

## The plateau is real and exhausted at ~0.911–0.916

The whole field runs one shared engine: pilkwang 50-epoch weights (8,827 downloads) + 2-seed
detector blend + 22-feature UNET300 ranker + edge-TTA + ILP + motion relink + gap repair +
harmonic bidirectional fusion (= our **P3 harmonic**). Post-processing is exhausted (edge-TTA
*hurts*, 0.885; ILP probes 0.908; 2-seed 0.910–0.911). The metric hack was patched + re-scored
~2026-07-20; today's 0.93–0.95 is legitimate.

## The undisclosed 0.93–0.950 edge

`adj_edge_jaccard` plateaus ~0.90–0.91 on the public stack and `division_J ~0` for nearly
everyone → the separation is the **edge term = a genuinely better / retrained model**. Only 2
physical embryos, 199 chunks, ~2.8% nuclei annotated, ~304 division events → a
**data/generalisation** problem, not post-processing.

- **H1 (strongest)** retrain detector/associator on **external Zebrahub** (imaging + dense
  Ultrack `*_tracks.csv`), host-opened 2026-08-13 ("no overlap with the test set").
- **H2 (= our gap)** learned **FP-suppressing** candidate ranker (precision, not recall).
- **H3** dense pseudo-labels / synthetic tracks (José Freitas 18.5GB CC0, 165k divisions ~540×).
- **H4** division recovery +~0.02 (shakeup-prone).
- **H5** off-the-shelf pretrained linkers (Trackastra/CoTracker, even the host's HOCT) — a
  **detour**: HOCT *underperformed* a tuned ILP (discussion #728551).

## Named competitors

hengck23 (GM) dense pseudo-label + affinity flow; TWEAK (rank2, 164 subs) "universal bio-cell
plugin"; Felipe Kitamura (rank13) detector-transfer; Tang/mikelou1 candid: retrain + external
+ modest division. Hosts = Royer lab (Ultrack) → intended solution is a well-trained detector
feeding ILP.

## Timing

We closed **2026-08-07**; Zebrahub external data unlocked **2026-08-13** — six days later. The
reopen is well-timed.
