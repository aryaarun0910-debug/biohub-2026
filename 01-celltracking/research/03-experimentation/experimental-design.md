---
id: 03-experimentation/experimental-design
title: Experimental Design
area: 03-experimentation
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- loeo
- design
---

# Experimental Design

> Design choices for a valid measurement on this 2-embryo corpus.

## LOEO (leave-one-embryo-out) — the community-validated protocol

- **fold 0:** hold out **44b6** (71 crops), evaluate with split_0 weights → anchor `0.759549`.
- **fold 1:** hold out **6bba** (128 crops), evaluate with split_1 weights → anchor `0.648965`.
- Run the deployed wrapper exporting per-crop pred geffs, then score vs `data/train` with
  `scripts/core/score_oof.py`. Both folds reported separately; promotion needs both non-regressive.

## Controls

- Baseline = P3-alone OOF geffs at the **same** held-out crops
  (`artifacts/kaggle/oof_clean/pred_geffs_split_{0,1}`, 1:1 crop match verified).
- Interacting mechanisms (arm-B × harmonic both touch `prob`) are measured **together**, never
  assumed additive.

## Gate

≥ +0.005 min-fold on the **patched scorer**, no regime-slice regression, before any GPU
submission. Cross-family (fit-on-one-eval-the-other) is the honest generalisation test; in-family
CV inflates.
