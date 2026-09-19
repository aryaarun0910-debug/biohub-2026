---
tags:
  - stage
---

# Node Transformer Edge Scorer

Scores candidate parent→child edges between consecutive frames. Frozen.

Its knobs — `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` (0.15) and
`BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT` (0.75) — are [[Upstream Knobs]]:
untestable locally and all under the [[Drift Guard]].

A quirk with consequences: every `solution` flag in the prediction `.geff` is
True, so the file stores **only ILP-selected edges**. That is what cripples
[[Learned Bonus Is Crippled]].

Related: [[ILP Linker]], [[Local Harness]]
