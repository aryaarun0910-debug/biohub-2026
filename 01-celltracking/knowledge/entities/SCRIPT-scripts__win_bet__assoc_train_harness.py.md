---
id: SCRIPT-scripts__win_bet__assoc_train_harness.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/assoc_train_harness.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0037', 'PKT-0038', 'PKT-0041', 'PKT-0043']
**Tests:** 4

## outgoing
- `measures` -> [[FACT-0398]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0405]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0406]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0413]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0414]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0432]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0037]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0038]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0041]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0043]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_assoc_fold_pathology.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_assoc_tournament.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_assoc_train_harness.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_spec_restriction.py]]  <sub>test imports source</sub>
