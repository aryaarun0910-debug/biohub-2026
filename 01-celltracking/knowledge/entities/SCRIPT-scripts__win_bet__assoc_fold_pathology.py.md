---
id: SCRIPT-scripts__win_bet__assoc_fold_pathology.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/assoc_fold_pathology.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0042']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0419]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0420]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0421]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0042]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_assoc_fold_pathology.py]]  <sub>test imports source</sub>
