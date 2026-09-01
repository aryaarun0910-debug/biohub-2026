---
id: SCRIPT-scripts__win_bet__focus3d_probe.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/focus3d_probe.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0049']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0423]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0424]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0425]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0460]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0049]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_focus3d_probe_state_contract.py]]  <sub>test imports source</sub>
