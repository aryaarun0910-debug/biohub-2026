---
tags:
  - stage
---

# Temporal UNet3D Detector

Produces candidate cell centres per frame. Runs from [[Frozen Weights]]
(`pilkwang/biohub-temporal-unet3d-seed314159-v1`).

Controlled by `BIOHUB_DET_THRESHOLD` (0.965), which is **upstream** — the
[[Local Harness]] cannot test it, because the harness starts from the
[[ILP Linker]] output. See [[Upstream Knobs]].

Raising the threshold 0.965→0.995 is ~33× cheaper per node removed than any
other route ([[Node Budget]]) — but that is the node-count axis, which
[[Operating Rules]] declares forbidden ground for *tuning*.

Related: [[Tracking Pipeline]], [[Drift Guard]]
