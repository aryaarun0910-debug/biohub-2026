---
id: SCRIPT-scripts__win_bet__divverify_dataset.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/divverify_dataset.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0035']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0383]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0385]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0035]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_divverify_contract.py]]  <sub>test imports source</sub>
