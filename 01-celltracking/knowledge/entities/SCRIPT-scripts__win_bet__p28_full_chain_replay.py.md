---
id: SCRIPT-scripts__win_bet__p28_full_chain_replay.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/p28_full_chain_replay.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0027', 'PKT-0029', 'PKT-0031', 'PKT-0040', 'PKT-0042', 'PKT-0044']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0367]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0372]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0375]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0426]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0428]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0437]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0027]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0029]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0031]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0040]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0042]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0044]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_p28_full_chain_replay.py]]  <sub>test imports source</sub>
