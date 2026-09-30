---
tags:
  - failed
---

# Failed: loosening `tau` / `diverge` on recall evidence

The trap that [[s01]] fell into, in general form. Loosening [[Safe Division]]
gates raises true positives and raises false positives **faster**, so
[[Division Jaccard]] falls.

Original measurement: TP up, FP ×5, divJ **fell** 0.067 → 0.027.

Reproduced exactly on the [[s09]] chain by [[Script 101 Safediv Gates]]:
- `diverge=1.75` → 6 TP but FP 2→5, divJ −0.0042
- `diverge=1.50` → 6 TP but FP 2→8, divJ −0.0571

**The screening rule that comes from this:** a gate change only counts if
**divJ rises**. Never TP.

Related: [[Safe Div Gates At Local Optimum]]
