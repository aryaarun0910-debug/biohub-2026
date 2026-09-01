---
id: SCRIPT-scripts__win_bet__gpu_protection_contract.py
kind: SCRIPT
tags: [script]
lifecycle: historical
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/gpu_protection_contract.py`
**Lifecycle:** historical — named by packet(s) ['PKT-0043']
**Tests:** 3

## outgoing
- `measures` -> [[FACT-0431]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0043]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_gpu_protection_contract.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_provenance_policy.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_provenance_policy_single_source.py]]  <sub>test imports source</sub>
