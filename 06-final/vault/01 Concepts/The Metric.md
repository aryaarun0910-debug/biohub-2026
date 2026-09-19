---
tags:
  - concept
---

# The Metric

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
adj   = max(0, J * (1 - 0.1 * (n_pred - n_est) / n_est))
```

Three parts, each with its own note:
- [[Adjusted Edge Jaccard]] — the bulk of the score
- [[Division Jaccard]] — worth 0.1×, but where the leverage is
- [[Node Count Multiplier]] — **uncapped above 1**, which is the trap

The multiplier being uncapped is the single most important property: deleting
nodes pays *without bound*. See [[Node Count Exploit]].

Locally computed by [[metric2]]. Do NOT use the other module — see [[Wrong Metric Module]].

Related: [[One Division Event Floor]], [[Proxy Score]]
