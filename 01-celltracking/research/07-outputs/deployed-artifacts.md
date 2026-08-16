---
id: 07-outputs/deployed-artifacts
title: Deployed Artifacts
area: 07-outputs
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- deployment
- kernels
---

# Deployed Artifacts

> The live system and the built candidate kernels. Build/push tooling:
> `../../scripts/core/kaggle_factory.py` (audited builds; **never auto-submits**).

## Live deployment

- **P3 harmonic — 0.915 public.** Notebook: `../../notebooks/kaggle_p3_harmonic/`. Base P0-B
  (`base_sha256 01408a17…`). Mechanisms: harmonic fusion + degree invariants + DeepCenter veto
  ([../02-theory/mechanisms.md](../02-theory/mechanisms.md)).

## Built candidates (local, audited — not pushed)

| kernel | slug | built_sha256 | edits | purpose |
|---|---|---|---|---|
| P3 + arm-B | `biohub-p3-armb` | `a07fa92b…` | 15/15 | harmonic + motion-residual flow gate |
| P3 + arm-B LOEO f0 | `biohub-p3-armb-loeo-f0` | `c837869b…` | 18/18 | export held-out **44b6** (71 crops), split_0 weights |
| P3 + arm-B LOEO f1 | `biohub-p3-armb-loeo-f1` | `9036c6ae…` | 18/18 | export held-out **6bba** (128 crops), split_1 weights |

The two LOEO kernels are **export-only** (submission disabled) and await the Kaggle GPU
green-light. Post-run scoring: `scripts/core/score_oof.py --pred-dir <returned> --gt-dir data/train`,
compared to the P3-alone baseline in `../../artifacts/kaggle/oof_clean/pred_geffs_split_{0,1}`
(anchors 44b6 `0.759549`, 6bba `0.648965`).
