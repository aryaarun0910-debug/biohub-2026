---
tags:
  - concept
---

# Adjusted Edge Jaccard

`adj = max(0, J_edge * multiplier)` where `J_edge` is the Jaccard index over
predicted vs ground-truth *edges* (parent→child links between detections across
frames), and `multiplier` is the [[Node Count Multiplier]].

Because `adj` is a product, a gain can come from either factor — and a gain that
arrives through the multiplier while `J_edge` is flat is the [[Node Count Exploit]],
not tracking quality. [[Operating Rules]] requires reporting them **separately**.

Related: [[The Metric]], [[Division Jaccard]]
