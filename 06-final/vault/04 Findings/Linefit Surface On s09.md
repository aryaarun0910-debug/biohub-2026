---
tags:
  - finding
---

# The Linefit (weight, window) Surface

Measured by [[Script 99 Linefit On s08]] on the full [[s08]] chain. Deployed
w=0.8/window=2 scores 0.97431; deltas against it:

| w \\ window | 2 | 3 | 4 |
|---|---|---|---|
| 0.2 | +0.00440 | +0.00278 | +0.00278 |
| **0.3** | **+0.00487** | +0.00505 | +0.00441 |
| 0.4 | +0.00372 | +0.00570 | **+0.00620** |
| 0.5 | +0.00291 | +0.00570 | +0.00166 |
| 0.6 | +0.00338 | +0.00261 | +0.00278 |
| 0.8 *(deployed)* | 0 | +0.00032 | −0.00241 |
| 1.0 | −0.00354 | −0.00821 | −0.00981 |

**w=0.3 is the robust pick**, not the argmax — its spread across windows is
**0.00064**, against 0.00248 for w=0.4 and 0.00404 for w=0.5.

Two invariants confirmed across all 24 cells:
- `ratio` **identical** at 0.8977 — linefit moves coordinates only.
- Division ledger **unchanged** at 5/2/7 — a pure edge-axis lever.

Aggregate `mult` drifts up to 8.85e-05, which is *not* a node-count change — see
[[Aggregate Mult Re-weighting]].

**Superseded as a shipping decision by [[Scored Films Measurement]]:** w=0.3 is
−0.00086 where it counts. Three film sets agree the plateau is **0.4–0.6** —
scored peaks at 0.6, validator at 0.4, the 199-film rebuild at 0.5 — and all
three put 0.3 outside it. s10 ships **0.6**.

⚠ The gain is concentrated: only 73/199 films move at all, median delta 0.00000.

Related: [[s09]], [[Linefit Smoothing]], [[Transfer Lesson]]
