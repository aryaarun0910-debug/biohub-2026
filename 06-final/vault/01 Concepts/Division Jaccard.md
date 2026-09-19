---
tags:
  - concept
---

# Division Jaccard

`divJ = TP / (TP + FP + FN)` over *division events* — a cell splitting into two.

Small in weight (0.1×) but large in leverage, because the denominator is tiny:
**12 ground-truth divisions across the 8 [[Validator Films]]**. One event moves
divJ by a lot. See [[One Division Event Floor]].

Ledger history on the validator films:
| configuration | TP/FP/FN | divJ |
|---|---|---|
| unmodified 0.947 | 3/1/9 | 0.2308 |
| [[s01]] `DIVERGE_UM=0` | 3/15/9 | 0.1111 |
| [[s05]] no relink | 4/1/8 | 0.3077 |
| [[s08]] + reorder (harness) | 5/2/7 | 0.3571 |

Divisions are produced by [[Safe Division]]. See [[Divisions Played Out]].

**This is the good axis** — see [[Axis Priors]].
