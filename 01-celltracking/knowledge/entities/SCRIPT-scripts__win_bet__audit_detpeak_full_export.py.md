---
id: SCRIPT-scripts__win_bet__audit_detpeak_full_export.py
kind: SCRIPT
tags: [script]
lifecycle: historical
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/audit_detpeak_full_export.py`
**Lifecycle:** historical — named by packet(s) ['PKT-0025']
**Tests:** 1

## outgoing
- `measures` -> [[FACT-0352]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0025]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_audit_detpeak_full_export.py]]  <sub>test imports source</sub>
