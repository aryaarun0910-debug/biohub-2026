---
id: SCRIPT-scripts__win_bet__detpeak_curve.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/detpeak_curve.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0001']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0110]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0111]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0001]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_ilp_retention_instruments.py]]  <sub>test imports source</sub>
