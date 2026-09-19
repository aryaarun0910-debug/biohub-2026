---
tags:
  - infra
---

# `scripts/100_repice_on_s09.py`

Re-prices 21 settings across [[Gap Closing]], [[Gap2 Recovery]],
[[Prune Isolated]] and [[Short Track Filter]] on the [[s09]] chain.

**Result: nothing.** Every positive row is the [[Node Count Exploit]].

Two design choices worth copying:
- It computes the [[One Division Event Floor]] from the **live ledger**
  (`0.1/(tp+fp+fn)` = 0.00714) rather than using the 0.0083 figure, which assumed
  a different denominator.
- It prints `ratio` on every row and emits the verdict itself, so the exploit is
  *flagged* rather than left to be spotted.

Related: [[Parametric Search Closed]]
