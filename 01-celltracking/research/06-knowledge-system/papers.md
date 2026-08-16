---
id: 06-knowledge-system/papers
title: Papers
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- literature
---

# Papers

> External literature we depend on, with the extracted **mechanism** per paper. Detail:
> [internal-reports/methods_frontier_2026-08-16.md](internal-reports/methods_frontier_2026-08-16.md).

| ref | what | mechanism we take | status |
|---|---|---|---|
| arXiv:2606.03568 (Learned-3D-NMS / D2D) | over-propose + learned re-scoring | transformer re-scorer over per-detection features; gains on sparse/small = our failure mode | **lead** |
| arXiv:2607.11754 (HOCT, Royer lab) | edge-centric tracking transformer | division edge-head (focal γ=3.5) as a high-precision fork idea | reference (not a drop-in) |
| bioRxiv 2025.04.28.651001 (Cellpose-SAM) | generalising nuclear segmenter | 2nd proposer, union into candidate pool; anisotropy-robust | candidate |
| arXiv:2605.26026 (LSM FM) | 3D LSM foundation model | frozen candidate-patch embeddings for the ranker | candidate |
| arXiv:2501.03880 (SELMA3D) | light-sheet SSL | test-time adapt to the hidden embryo in-notebook | candidate |
| Zebrahub (host-unlocked 2026-08-13) | imaging + dense Ultrack tracks | external same-domain training data | **program** |
| Freitas 18.5GB CC0 | synthetic 3D microscopy | 165k division labels, pooling-matched supervision | program |

Licensing / governance notes: [../04-data/data-governance.md](../04-data/data-governance.md).
