---
id: SCRIPT-scripts__win_bet__provenance_policy.py
kind: SCRIPT
tags: [script]
lifecycle: reusable-instrument
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/provenance_policy.py`
**Lifecycle:** reusable-instrument — named as a fact instrument
**Tests:** 2

## outgoing
- `measures` -> [[FACT-0434]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0435]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0436]]  <sub>fact instrument path</sub>

## incoming
- `protects` <- [[TEST-tests__test_provenance_policy.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_provenance_policy_single_source.py]]  <sub>test imports source</sub>
