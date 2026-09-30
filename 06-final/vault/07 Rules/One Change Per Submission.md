---
tags:
  - rule
---

# One Change Per Submission

The discipline that keeps the [[Submission Ledger]] interpretable.

**The subtlety that matters:** "one change" means one change relative to **the
thing you are comparing against**, not relative to the 0.947 base. [[s09]] is
three changes from base but **one** from [[s08]], and that is what makes it a
legitimate probe.

This is *demonstrated*, not asserted: [[Script 96 Reorder Variant]] diffs the
built notebook against the shipped [[s05]] notebook, and [[make_env_variant]]
diffs against its stated parent.

A guard rewrite forced by a guarded key is bookkeeping, not a second change —
see [[Drift Guard]].
