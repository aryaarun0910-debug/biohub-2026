---
title: Pending decisions
tags: [moc, pending]
generated: true
---

# Pending decisions

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


Decisions measured and routed, but NOT taken. Each names its evidence.

## Notebook line-ending contract
- Evidence: `research/00-system/registry/generated/lineending_experiment.json`
- Measured: `*.ipynb text eol=crlf` makes every platform resolve the recorded digests and rewrites zero tracked blobs. `-text` breaks them everywhere, including here.
- Blocked on: a migration decision. `.gitattributes` is unchanged.

## Stale packet locks
- 4 packets hold levers that are already `killed`/`closed`, so the anti-duplication lock reserves decided levers.
- Blocked on: a coordinator pass over the packet locks.

## Release receipts
- All raw receipts are gitignored; bindings are now durable via `generated/receipts.json`, but the raw audit bundles still do not survive a clone.

## Gate B
- Stopped at its recorded resume boundary. Resume state: `C:/temp/finaledge/gateB_v2/RESUME_STATE.md`.
