---
id: 04-data/data-acquisition
title: Data Acquisition
area: 04-data
status: active
updated: '2026-08-17'
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
- 2 physical embryos (`44b6`, `6bba`), 199 chunks, ~2.8% nuclei annotated, **151 division events** (302 daughter links; the older "~304" counted links, not events — remeasured 2026-08-17 over all 199 crops: 44b6 26, 6bba 125).

## External (allowed; see governance)

- **Zebrahub** imaging + dense Ultrack `*_tracks.csv` lineages — host-unlocked 2026-08-13
  ("no overlap with the test set", #734330). Same-domain.
- **Freitas synthetic** — 18.5GB CC0, 165k labeled divisions (~540× real), pooling-matched to
  the evaluator's `vol[:, ::4, ::4]` stride.
- **Packaged Zebrahub crops on Kaggle** (found 2026-08-17; lifts the H1 imaging gate for a pilot):
  - `kkunizaw/biohub-zh001r` (363 MB) — `zh001r_iso.npy` **(72, 20, 64, 64, 64) uint8** ZSNS001
    imaging (verified locally: min 0 / max 255 / mean 51.8), `zh001r_tgt.npy` same shape,
    `zh001r_nodes.npz` = 1440 arrays `f{crop*20+t}`, each `(N,4) float32 [t,z,y,x]`, ~900/frame.
  - `kkunizaw/biohub-zmnscrops` (3.66 GB) — `zmns00{1,2}_crops.npz`, "windowed crops derived from
    the public Zebrahub multi-view imaging dataset … No competition data included" (not yet opened).
  - **Unverified:** the third party's isotropic resampling scale / voxel size, and their node
    labels. Establish voxel scale before any transfer claim. Zebrahub is **CC BY-NC** (host-cleared
    #734330; record the license in any shipping notebook). Attachable to kernels — no local
    download and no kernel internet needed.

## Weights / models (reproducible)

pilkwang 50-epoch public weights; our fold weights (`edge_predictor_best_split_{0,1}.pth`,
`config_split_{0,1}.json`) in `../../artifacts/kaggle/weights_dataset/`.
