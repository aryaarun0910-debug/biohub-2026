---
id: 03-experimentation/protocol
title: Protocol
area: 03-experimentation
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- protocol
- factory
---

# Protocol

> The runnable protocol. Stages must never be conflated: **smoke → representative pilot → full**.

## Stage gate

1. **Smoke** — proves plumbing (e.g. `BIOHUB_LOEO_LIMIT=3`). *Not scientific evidence.*
2. **Representative pilot** — estimates a mechanism on a representative slice.
3. **Full LOEO** — confirms it on all held-out crops, both folds, patched scorer.

## Kaggle kernel flow (kaggle_factory.py)

1. `build --spec <spec.json>` — applies edits to a base notebook, **asserts exact match counts**
   (a skipped patch fails the build), records base/built sha256 + config hash in a manifest.
2. Human review of the manifest + built notebook.
3. `push` (explicit) — upload; `submitcmd` **prints** a command for a human. **No auto-submit.**

## Current runnable item

The two P3+armB LOEO-export kernels are built + audited
([../07-outputs/deployed-artifacts.md](../07-outputs/deployed-artifacts.md)). Next step is the
**smoke** (fold-0, `LIMIT=3`) on Kaggle T4×2 — pending green-light — then the two full folds,
then `score_oof.py` vs the anchors.
