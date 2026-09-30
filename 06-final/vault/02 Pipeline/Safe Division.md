---
tags:
  - stage
---

# Safe Division

Proposes a second child for a parent that has exactly one, drawing the candidate
from the **orphan pool** (nodes with no incoming edge). This is what produces
[[Division Jaccard]] true positives.

Gates (all deployed values):
| gate | value |
|---|---|
| `SAFE_DIV_MAX_UM` (parent→daughter) | 9.0 |
| `SAFE_DIV_SISTER_MAX_UM` | 14.0 |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 10.0 |
| `SAFE_DIV_SISTER_SYMMETRY_TAU` | 0.6 |
| `SAFE_DIV_DIVERGE_UM` | 2.25 |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.0076 |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.00375 |

**The gates are at a sharp local optimum** — re-confirmed on the [[s09]] chain, see
[[Safe Div Gates At Local Optimum]]. `tau` loses a division in *both* directions.

One never-touched boolean is load-bearing: `SAFE_DIV_REQUIRE_DIVERGENCE=0` costs
**−0.025** with division FP 2→20.

Because it competes for orphans, everything that consumes orphans must run
*after* it. See [[Ordering Bug Class]].

Related: [[s01]], [[Divisions Played Out]]
