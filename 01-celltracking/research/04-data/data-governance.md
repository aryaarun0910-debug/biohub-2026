---
id: 04-data/data-governance
title: Data Governance
area: 04-data
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- governance
- licensing
- guardrails
---

# Data Governance

> External-data licensing, host clearances, and prize-critical use constraints.

## Allowed

- External data permitted, incl. **Zebrahub imaging + tracks** (host-confirmed 2026-08-13,
  #734330) and **generic public-data pretraining**.
- Pretrained / self-trained models allowed (attach as a dataset; reproducible if the entry wins).

## Constrained / forbidden

- **Exact public-source trajectory transfer** into an identified hidden crop needs **written
  host clearance**. Generic public-data pretraining is fine; copying a known lineage into a
  recognised test crop is not.
- **Never** infer hidden-set quality from the four visible placeholder movies (in-sample, biased).
- The unmatched-fork division-evaluator pathology is **diagnostic only** — never in a submission.

## Licenses (verify before shipping)

Freitas synthetic **CC0**; the current StableDet-HOCT checkpoint bundle's publisher licence and
component provenance are recorded in `FACT-0347` (superseding the stale blanket licence here);
Cellpose historically **BSD-3** (verify for Cellpose-SAM). Record the license
of every shipped external asset in the submission notebook.
