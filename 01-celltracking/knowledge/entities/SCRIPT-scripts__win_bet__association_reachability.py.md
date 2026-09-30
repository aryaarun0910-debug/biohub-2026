---
id: SCRIPT-scripts__win_bet__association_reachability.py
kind: SCRIPT
tags: [script]
lifecycle: historical
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/association_reachability.py`
**Lifecycle:** historical — named by packet(s) ['PKT-0027']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0370]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0027]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_ilp_retention_instruments.py]]  <sub>test imports source</sub>
