---
tags:
  - finding
---

# Harness Validated Against The Kernel

On the **no-relink topology specifically** — which matters, because that is the
base [[s05]], [[s08]] and [[s09]] all sit on.

Source: `artifacts/s05_output/validator_results.csv`, Kaggle's own in-kernel
validator output, per film.

| film | kernel | harness |
|---|---|---|
| 44b6_12dfb391 | 0/0/1 | 0/0/1 |
| 44b6_267148e4 | 1/0/0 | 1/0/0 |
| 44b6_2a2eff9f | 1/1/0 | 1/**2**/0 |
| 44b6_341df25f | 0/0/1 | 0/0/1 |
| 6bba_062c8d37 | 1/0/0 | 1/0/0 |
| 6bba_07e24132 | 0/0/2 | 0/0/2 |
| 6bba_085bf656 | 0/0/1 | 0/0/1 |
| 6bba_09961292 | 1/0/3 | 1/0/3 |
| **total** | **4/1/8** | **4/2/8** |

**Agreement on 7 of 8**, one extra false positive — attributable to the
un-ported [[DeepCenter]] safe-division veto.

The kernel total 4/1/8 matches the figure recorded for [[s05]] independently.

Related: [[Local Harness]], [[Division Recovery in 44b6_341df25f]]
