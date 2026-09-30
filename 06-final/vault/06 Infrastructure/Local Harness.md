---
tags:
  - infra
  - key
---

# Local Harness

**The single most valuable asset in the project.**

The Kaggle kernel keeps its prediction `.geff` files, and **8 of them are TRAIN
films** — so we hold *deployed-quality graphs with ground truth*. Post-processing
experiments run in **seconds** instead of ~1.75 h.

```
artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0/*.geff
```

**Validated three times against the board or the kernel:** it reproduced the
[[s01]] failure direction, it reproduces the deployed division ledger exactly
(3/1/9), and it agrees with Kaggle's own validator film-by-film on 7 of 8 — see
[[Harness Validated Against Kernel]].

Use, in this order: [[metric2]], [[Script 24 Keep ILP Edges]],
[[Script 91 Other Stages]], `scripts/25_relink_control.py`.

**Its hard limit:** it starts from the [[ILP Linker]] *output*, so it cannot test
[[Upstream Knobs]] at all.

⚠ Do not use the wrong modules — see [[Wrong Metric Module]].
