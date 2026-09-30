---
id: SCRIPT-scripts__core__kaggle_factory.py
kind: SCRIPT
tags: [script]
lifecycle: active-experiment
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path / name:** `scripts/core/kaggle_factory.py`
**Lifecycle:** active-experiment — named by packet(s) ['PKT-0036', 'PKT-0039']
**Tests:** 4

## outgoing
- `measures` -> [[FACT-0063]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0387]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0397]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0401]]  <sub>fact instrument path</sub>
- `measures` -> [[FACT-0446]]  <sub>fact instrument path</sub>

## incoming
- `owns` <- [[PKT-0036]]  <sub>packet inputs.code</sub>
- `owns` <- [[PKT-0039]]  <sub>packet inputs.code</sub>
- `protects` <- [[TEST-tests__test_d1_factorial_smoke.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_experiment_defect_gate.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_kaggle_factory_release_receipt.py]]  <sub>test imports source</sub>
- `protects` <- [[TEST-tests__test_kaggle_queue.py]]  <sub>test imports source</sub>
