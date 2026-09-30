---
id: SCRIPT-scripts__win_bet__hoct_scale_gate.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/hoct_scale_gate.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0049']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0455]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0456]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0049]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_pkt0048_maskassoc.py]]  <sub>test imports source</sub>
