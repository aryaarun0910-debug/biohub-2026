---
id: SCRIPT-scripts__win_bet__assoc_parent_dataset.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/assoc_parent_dataset.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0030', 'PKT-0031', 'PKT-0032', 'PKT-0033', 'PKT-0034', 'PKT-0038', 'PKT-0041']
**Tests:** 3

## outgoing
- `measures` -> [[FACT-0381]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0382]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0386]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0388]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0030]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0031]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0032]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0033]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0034]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0038]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0041]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_assoc_fold_pathology.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_assoc_train_harness.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_hoct_compat.py]]  <sub>test imports source</sub>
