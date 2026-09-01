---
id: SCRIPT-scripts__win_bet__assoc_lost_edge_ledger.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/assoc_lost_edge_ledger.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0042', 'PKT-0044']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0422]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0439]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0042]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0044]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_assoc_fold_pathology.py]]  <sub>test imports source</sub>
