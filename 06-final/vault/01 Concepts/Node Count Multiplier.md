---
tags:
  - concept
---

# Node Count Multiplier

```
mult = 1 - 0.1 * (n_pred - n_est) / n_est
```

`n_est` is an *estimate* of the true node count, supplied per film. `n_pred` is
what we output.

**It is uncapped above 1.** If `n_pred < n_est`, `mult > 1` and keeps rising as
you delete more. There is no ceiling. This is the [[Node Count Exploit]].

Two traps around it:
1. Deleting nodes raises it forever — [[Short Track Filter]] rides this monotonically.
2. The *aggregate* `mult` is not a node-count readout — see [[Aggregate Mult Re-weighting]].

The honest invariant is **`ratio = n_pred / n_est`**, not `mult`.

Related: [[Node Budget]], [[The Metric]]
