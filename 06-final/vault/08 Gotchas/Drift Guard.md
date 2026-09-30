---
tags:
  - gotcha
---

# The `_EXPECTED_NUMERIC` drift guard

A single dict in cell 2 asserting nine env values at import time. If a set value
disagrees with the guard, **the kernel aborts at cell 3** — roughly 20 minutes
into a 1.75 h run.

Guarded keys:
`DET_THRESHOLD` 0.965 · `ILP_APPEARANCE_WEIGHT` 0.0 ·
`ILP_DISAPPEARANCE_WEIGHT` 2 · `GAP_CLOSE_UM` 5.0 · `OUTPUT_MIN_TRACK_LEN` 6.0 ·
`SAFE_DIV_MAX_UM` 9.0 · `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.20 ·
`BIDIRECTIONAL_EDGE_WEIGHT` 0.15 · `SECONDARY_EDGE_FEATURE_TTA_WEIGHT` 0.75

Note the overlap with [[Upstream Knobs]] — **five of the nine are upstream**,
which is why the guard was the blocker on that whole axis.

[[make_env_variant]] now rewrites the guard entry in the same edit and proves it.

Related: [[One Change Per Submission]]
