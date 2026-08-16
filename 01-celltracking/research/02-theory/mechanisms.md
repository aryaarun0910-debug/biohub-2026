---
id: 02-theory/mechanisms
title: Mechanisms
area: 02-theory
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- mechanisms
---

# Mechanisms

> The causal mechanisms behind each candidate improvement — why it should move the score.
> Deployed mechanisms live in `../../src/biotrack/wrapper.py`.

## Harmonic bidirectional fusion (DEPLOYED — P3)

Fuse forward and reverse edge logits by their **harmonic mean in prob space** (`_bi_new`).
Rationale: a link is trustworthy only if *both* time directions agree; the harmonic mean
punishes one-sided confidence. Rewrites `prob`, which downstream `motion_relink` consumes.

## Motion-residual flow gate (arm B — UN-SHIPPED, measuring)

Change the relink eligibility gate from `raw = |target − source|` to
`gate_q = |target − (source + flow(source))|`, where `flow` is a **GT-free kNN-16 median flow**
(`_armb_flow_compensation`). Rationale: a link that looks long in absolute terms may be short
*relative to local tissue motion* — compensating removes false rejections in fast-moving regions.
Evidence: P0-strict LOEO **+0.0079822** (44b6 +0.0167567, 6bba +0.0067055, P(d>0)=1.0); E0c
**+0.0088059**. **Interacts with harmonic** (both touch `prob`) → measure together.

## Degree invariants (DEPLOYED — P3)

Post-filter graph invariants enforcing valid degree/division structure before scoring; the fixes
that unblocked the first arm-B run are already carried by P3.

## DeepCenter veto (DEPLOYED)

A UNet3D center-prior (epoch 500) vetoes gap-closure / safe-division candidates lacking center
support — a learned precision gate on the risky post-processing paths.

## The lever we still need — learned FP-suppression

Over-propose candidates (recall ~0.98) then a small transformer **re-scores** each by attending
to neighbours, restoring precision on the junk pool. This is the mechanism behind
`bet-learned-ranker` and the Zebrahub retrain (Theme A/B).
