---
tags:
  - data
---

# Frozen Weights

Nothing in this project trains a model. Three public datasets by `pilkwang`,
SHA256-verified against values pinned in the notebook:

- `biohub-tracking-support-pack-50ep-v1`
- `biohub-temporal-unet3d-seed314159-v1` → [[Temporal UNet3D Detector]]
- `biohub-deepcenter-unet3d-center-prior-v1` → [[DeepCenter]]

Already downloaded to `weights/`.

**This is the binding constraint on the whole project.** Every detection and
matching failure in [[Divisions Played Out]] is unfixable, and
[[Failed Division Classifier]] showed the frozen features do not carry the
division signal.

The base notebook is a fork of Reyhan Ksatria's 0.947 notebook with three input
paths changed and nothing else. See [[0.947 Plateau]].
