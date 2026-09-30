---
tags:
  - finding
---

# Safe Div Gates Are At A Local Optimum

Confirmed twice: originally on clean graphs, and again on the [[s09]] chain by
[[Script 101 Safediv Gates]] — which was **not** redundant, because safe
division's *input* changed when [[s08]] moved [[Gap2 Recovery]] behind it.

No gate in any direction gains a division. The ledger holds at 5/2/7.

| gate | swept | result |
|---|---|---|
| `parent_max` | 8.5 → 12.0 | nothing, even at +33% |
| `sister_max` | 13 → 16 | inert |
| `child_max` | 9 → 12 | inert |
| `frame_cap`, `glob_cap` | ×2 | inert |
| `tau` | 0.5 → 1.0 | **sharp optimum at 0.6** — loses a division in *both* directions |
| `diverge` | 1.5 → 3.0 | 1.75 and 1.5 gain a TP but FP goes 2→5 and 2→8, divJ **falls** |

The `diverge` rows reproduce [[Failed Loosening Tau Diverge]] exactly: TP rises,
divJ falls, because FP rises faster.

Related: [[Parametric Search Closed]], [[Divisions Played Out]]
