---
id: 02-theory/existing-knowledge
title: Existing Knowledge
area: 02-theory
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- methods
- frontier
---

# Existing Knowledge

> What we build on, internal and external. Full methods scan:
> [../06-knowledge-system/internal-reports/methods_frontier_2026-08-16.md](../06-knowledge-system/internal-reports/methods_frontier_2026-08-16.md).
> Per-paper mechanisms: [../06-knowledge-system/papers.md](../06-knowledge-system/papers.md).

## Internal (established on this dataset)

- **P3 harmonic = 0.915 public** — the deployment; the shared public engine (see
  [../01-research-direction/research-landscape.md](../01-research-direction/research-landscape.md)).
- **LOEO substrate anchors** reproduced exactly: 44b6 `0.759549`, 6bba `0.648965`.
- **Cross-family transfer is the trap**: a ranker fit on 44b6 fails on 6bba (the 2-embryo trap).
- **Detection recovery is a ranker, not a threshold** (marginal scalar precision ~0.003 vs a
  ceiling needing ~0.55).

## External methods (ranked by EV × feasibility)

- **LEAD — over-propose + learned candidate re-scoring** (D2D-Rescore / Learned-3D-NMS,
  arXiv:2606.03568, MIT-style). 6-layer/64-ch/4-head transformer over per-detection features;
  gains concentrate on sparse/small candidates = our failure mode. <1M params, minutes on a T4.
- **HOCT** (organizer lab, arXiv:2607.11754) — edge-centric transformer, 19 hand-crafted
  features, division edge-head. **Community-tested and underperformed a tuned ILP** → not a
  drop-in edge; useful as a division-head idea only.
- **Cellpose-SAM** (bioRxiv 2025.04.28.651001) — 2nd generalising nuclear proposer; robust to
  anisotropic blur; union into the candidate pool.
- **LSM foundation model** (arXiv:2605.26026) — frozen candidate-patch embeddings for the ranker.
- **Test-time SSL/TTA** (SELMA3D arXiv:2501.03880) — adapt to the hidden embryo in-notebook.

## Data assets (external)

Zebrahub imaging + `*_tracks.csv` (host-unlocked 2026-08-13); Freitas 18.5GB CC0 synthetic
(165k divisions, pooling-matched to `vol[:, ::4, ::4]`).
