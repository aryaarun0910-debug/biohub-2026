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

## 2026-08-17 — H1 Zebrahub retrain scoped; level-1 acquisition decided

- **Committed to the H1 edge/detector retrain** as the real gap-closer (the ~0.029 to the
  leader is "not attributable to any recoverable public mechanism" — it needs external-data
  retraining). Parked the tested-negative division ranker (laneD: learned MLP ≈/< frozen
  geometry) and the motion-gate (still ready, committed `8fb5e83`).
- **Data gate found:** Zebrahub on disk is TRACKS-ONLY (no imaging). The detector/edge model
  is a UNet over raw volumes → retrain needs imaging. See [[biohub-zebrahub-imaging-gap]].
- **Disk-vs-resolution decision:** level-0 (exact deployed res, z-full/xy÷4) is ~1.85 TB and
  won't fit in 383 GB free. **Chose level-1** (~150–232 GB, z-half/xy-half) — retrain the
  whole pipeline at that resolution, re-deriving the competition features to match (this
  **voids the deployed 0.915 detector anchor**; the retrained model is a new lineage).
- **Feasibility validated (local, cheap):** remote OME-Zarr level-1 chunks stream + decode via
  urllib + numcodecs (zarr 3.3.0); one chunk 26.3 MB in 6.5 s. Training surface understood:
  `vendor/.../train_unet_transformer.py` reads zarr level "0" strided by `downsample`; the
  level-1 retrain needs the loader pointed at level "1" with adjusted downsample/scale.
- **Staged plan (smoke → pilot → full; GPU + full download held for green-light):**
  1. *smoke:* **DONE (2026-08-17).** `scripts/win_bet/h1r_fetch_imaging.py` streams+stores
     level-1 zarr crops; `scripts/win_bet/h1r_train_smoke.py` runs one full CPU step on a
     ZSNS003 t0→t1 crop: UNet(level-1) → node-feature index → edge transformer → loss 0.0121
     → backward, UNet receives grad. Plumbing proven (nuclei are shell-distributed → crops
     must be node-centred; level-0-voxel tracks map to level-1 via ÷2).
  2. *pilot:* retrain at level-1 on 1 embryo + re-derived competition data; LOEO-validate the
     delta vs a level-1 competition-only baseline. **Falsification:** no bilateral LOEO gain
     from the Zebrahub augmentation ⇒ external imaging doesn't transfer at level-1; kill.
  3. *full:* 3 embryos (ZSNS003/004/005) + full retrain only if the pilot clears.

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
