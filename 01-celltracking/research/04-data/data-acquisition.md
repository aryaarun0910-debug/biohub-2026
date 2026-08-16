---
id: 04-data/data-acquisition
title: Data Acquisition
area: 04-data
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- acquisition
---

# Data Acquisition

> How raw data is obtained. Storage of what we acquire: [storage.md](storage.md). Licensing:
> [data-governance.md](data-governance.md).

## Competition data

- 87GB light-sheet zebrafish 3D+t nuclei; `../../data/` (gitignored). Reproducible via the
  Kaggle CLI download of `biohub-cell-tracking-during-development`.
- 2 physical embryos (`44b6`, `6bba`), 199 chunks, ~2.8% nuclei annotated, ~304 division events.

## External (allowed; see governance)

- **Zebrahub** imaging + dense Ultrack `*_tracks.csv` lineages — host-unlocked 2026-08-13
  ("no overlap with the test set", #734330). Same-domain.
- **Freitas synthetic** — 18.5GB CC0, 165k labeled divisions (~540× real), pooling-matched to
  the evaluator's `vol[:, ::4, ::4]` stride.

## Weights / models (reproducible)

pilkwang 50-epoch public weights; our fold weights (`edge_predictor_best_split_{0,1}.pth`,
`config_split_{0,1}.json`) in `../../artifacts/kaggle/weights_dataset/`.
