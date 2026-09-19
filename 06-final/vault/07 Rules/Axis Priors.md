---
tags:
  - rule
  - key
---

# Axis Priors

The one published offline-vs-board ledger in this competition:

| axis | record |
|---|---|
| **division** | **3 for 3** |
| **edge** | **0 for 3** |

This is the single best prior available for deciding what to believe.

Applied to what is in flight:
- [[s08]]'s gain **is a division** ([[Division Recovery in 44b6_341df25f]]) →
  good axis.
- [[s09]]'s gain leaves the division ledger untouched at 5/2/7 in all 24 cells →
  **pure edge axis**, bad prior, despite better local evidence.
- [[s05]] is an edge-axis change with much better evidence than the three that
  failed — but the base rate is real.

Related: [[Operating Rules]], [[Proxy Score]]
