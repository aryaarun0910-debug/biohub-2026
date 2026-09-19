---
tags:
  - stage
---

# DeepCenter

A separate UNet3D centre-prior model (`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`)
used as a **veto** at two points: gap confirmation and safe division.

`DEEPCENTER_GAP_VETO` 1 / threshold 0.25; `DEEPCENTER_SAFE_DIV_VETO` 1 /
threshold 0.20 (the latter under the [[Drift Guard]]).

As a division *ranker* it is real — AUC 0.836–0.857 — but it is already **at its
oracle ceiling**, worth +0.0029, and a plain brightness threshold matches it. The
whole gain is one fork in one film. See [[Failed DeepCenter Veto]].

Not ported to the [[Local Harness]], which is why the harness carries one extra
division false positive.

Related: [[Harness Validated Against Kernel]]
