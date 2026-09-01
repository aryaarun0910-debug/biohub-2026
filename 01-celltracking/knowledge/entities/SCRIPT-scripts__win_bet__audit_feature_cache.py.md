---
id: SCRIPT-scripts__win_bet__audit_feature_cache.py
kind: SCRIPT
tags: [script]
lifecycle: historical
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/audit_feature_cache.py`
**Lifecycle:** historical — named by packet(s) ['PKT-0043']
**Tests:** 5

## outgoing
- `measures` -> [[FACT-0410]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0416]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0043]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__assoc_v2_world.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_assoc_train_harness.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_audit_feature_cache.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_provenance_policy.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_provenance_policy_single_source.py]]  <sub>test imports source</sub>
