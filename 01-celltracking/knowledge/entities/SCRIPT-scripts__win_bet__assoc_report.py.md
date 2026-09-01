---
id: SCRIPT-scripts__win_bet__assoc_report.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/win_bet/assoc_report.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0031', 'PKT-0032', 'PKT-0033', 'PKT-0034', 'PKT-0035', 'PKT-0038', 'PKT-0041', 'PKT-0042', 'PKT-0043', 'PKT-0044']
**Tests:** 4

## outgoing
- `measures` -> [[FACT-0386]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0398]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0432]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0031]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0032]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0033]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0034]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0035]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0038]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0041]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0042]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0043]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0044]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_assoc_report.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_assoc_train_harness.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_audit_receipt_harness.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_control_binding.py]]  <sub>test imports source</sub>
