---
tags:
  - infra
---

# `src/biohub/metric2.py`

**THE** metric implementation for the [[Local Harness]] graphs.

`SCALE = (1.625, 0.40625, 0.40625)` — **anisotropic**, original-voxel coordinates.

Two implementation details that have bitten:
- `aggregate()` averages with `weight = tp + fp + fn` (line 91), which depends on
  the *predictions*. See [[Aggregate Mult Re-weighting]].
- `score()` returns `J_edge`, `multiplier`, `adj`, `n_pred`, `n_est` and the
  division ledger separately, which is what makes [[Operating Rules]]'
  report-them-separately rule enforceable.

⚠ [[Wrong Metric Module]] — `metric.py` and `postprocess.py` are 4× wrong here.
