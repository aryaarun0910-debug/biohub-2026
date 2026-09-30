---
tags:
  - moc
---

# Scripts

**Harness core**
- [[Script 24 Keep ILP Edges]] — loaders and `safe_div`
- [[Script 91 Other Stages]] — ports of every post-processing stage
- [[metric2]] — the metric

**Builders** (each proves its own output)
- [[Script 96 Reorder Variant]] — the statement reorder, 7 proofs
- [[make_variant]] — substitute an existing env line
- [[make_env_variant]] — add or substitute on any parent, with [[Drift Guard]] rewrite

**Experiments on the current chain**
- [[Script 98 Reorder On Norelink]] — the [[s08]] measurement
- [[Script 99 Linefit On s08]] — the [[s09]] measurement
- [[Script 100 Reprice On s09]] — 21 settings
- [[Script 101 Safediv Gates]] — 7 gates

A convention worth keeping: every experiment script prints `J` and the
multiplier separately and flags the [[Node Count Exploit]] itself, so the guard
is executable rather than remembered.
