---
tags:
  - infra
---

# Ledger — `artifacts/ledger/runs.db`

SQLite. Tables: `runs`, `scores`, `division_events`, `submissions`.

**Predictions are recorded before results**, so sign agreement between offline
and board stays honest rather than reconstructed afterwards.

Every row for [[s08]] and [[s09]] carries an explicit **falsifier** — for s08,
"if [[s05]] does not beat 0.947 this build is void, because its base is void".

Related: [[Submission Ledger]], [[Operating Rules]]
