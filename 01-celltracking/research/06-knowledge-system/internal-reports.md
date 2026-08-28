---
id: 06-knowledge-system/internal-reports
title: Internal Reports
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- reports
- agents
---

# Internal Reports

> Index of internal + agent reports. The reports themselves are tracked under
> [internal-reports/](internal-reports/); raw agent transcripts are gitignored in
> `../../_evidence/agent_runs/`.

## Reports (tracked)

- [internal-reports/competitive_frontier_2026-08-16.md](internal-reports/competitive_frontier_2026-08-16.md)
  — leaderboard, plateau, the undisclosed-edge hypotheses, named competitors, timing insight.
- [internal-reports/methods_frontier_2026-08-16.md](internal-reports/methods_frontier_2026-08-16.md)
  — external methods ranked by EV × novelty × offline feasibility.

## Agent runs (gitignored evidence)

`../../_evidence/agent_runs/` holds the raw transcripts (agentA_kaggle, agentB_literature,
agentC_code, agentD_redteam) and `agentD_tests/`. Structured findings from these runs live in
`research.sqlite` (see [institutional-memory.md](institutional-memory.md)).

## 2026-08-28 architecture hunt map

The executable asset census is `FACT-0347`; it changes the order from architecture fashion to
contract risk:

- **wide-UNet3D / wide-ResUNet3D (`LEVER-0033`)** — public state dicts are real and safely
  inspectable, but their dataset has no model card or preprocessing contract. First recover the
  exact class, normalization, downsampling and peak extraction from the author's public notebook;
  then require baseline-output parity before reading quality.
- **StarDist3D (`LEVER-0032`)** — the public checkpoint is structurally a genuine probability-plus-
  radial-distance model. It is not a drop-in one-channel detector: integrate its NMS/centroid adapter
  and test star-convex reconstruction before any finetune.
- **StableDet/HOCT (`LEVER-0034`)** — best documented and already two-fold, but the bundle's own
  result separates association improvement from division failure (`FACT-0347`). Test the fixed-node
  association path first; do not call it a division solution.

Research in this order: the actual Kaggle model card/weights and public notebook; the method's
official paper/repository; then [BioImage.IO](https://bioimage.io/) for microscopy-ready model
contracts, [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for dataset-adaptive 3D baselines, and the
[Cell Tracking Challenge](https://celltrackingchallenge.net/) for detector/linker/division benchmark
separation. Architecture attributes to record before trying anything are input voxel scale, intensity
normalization, receptive field, output semantics, calibration, NMS/local-max rule, temporal context,
division representation, parameter/memory/runtime envelope, licence, training corpus, and honest-fold
weights. Missing attributes are blockers, not defaults to guess.

The Kaggle discussion is recorded as external evidence (`FACT-0348`) and reconciled with the patched
metric at source (`FACT-0349`): detection comes first, but aggregate recall cannot certify a correct
parent-plus-two-daughter topology.
