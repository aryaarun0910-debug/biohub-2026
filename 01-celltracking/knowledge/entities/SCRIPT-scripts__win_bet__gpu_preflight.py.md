---
id: SCRIPT-scripts__win_bet__gpu_preflight.py
kind: SCRIPT
tags: [script]
lifecycle: reusable-instrument
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/gpu_preflight.py`
**Lifecycle:** reusable-instrument — named as a fact instrument
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0399]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0416]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0417]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0418]]  <sub>fact instrument path</sub>

## incoming
- `protects` <- [[TEST-tests__test_gpu_preflight.py]]  <sub>test imports source</sub>
