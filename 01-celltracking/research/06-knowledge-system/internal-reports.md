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

## 2026-08-28 addendum — association-first integration contracts (no training)

The hunt is re-ordered by what the measurements now say. Detection capability is bounded on the
honest fold (`FACT-0354`), division failure is topology-dominant (`FACT-0359`) and localisation-
sensitive (`FACT-0360`), so the order is **association → division representation → detector**.

**`LEVER-0034` HOCT — integrate first, and it is nearly free.** `FACT-0361`: their base edge
predictor is architecturally identical to the public pack's (136 tensors, 2,077,996 params, same
key set, same shapes), and the model card states the checkpoints score edges over detector-produced
nodes with the same fixed downstream ILP. So this is a head swap on our own contract, not a port.
What must be reconciled before their numbers are expected to transfer:

| attribute | ours | HOCT | status |
|---|---|---|---|
| downsample | `[1,4,4]` | `[1,4,4]` | matches |
| window_size | 2 | 2 | matches |
| unet layers / out channels | 32/`[32,64,128]` | 32/`[32,64,128]` | matches |
| **pool_kernel_um (local-max NMS)** | **3.0** | **5.0** | **blocker — different node set** |
| candidate gate | edge softmax > 0.5 | `gate_um` 15.0, `edge_neighbors` 64 | must be applied as theirs |
| baseline the gains sit on | P28 = `FACT-0345` | their official ref 0.7147 | **their deltas are not additive** |

**`LEVER-0027` RoPE-4D** — already built and gate-1 passed (`FACT-0340`); gate 2 is stuck because
Zebrahub saturates (`FACT-0342`). Re-gate it on a harder split, or straight onto the P28 fold-0
paired delta. No new integration work is needed.

**Explicit daughter-pair geometry** — `FACT-0362`: PQLT and TOQL both carry a learned
`child_slots (2, 64)` two-daughter decoder, so the representation itself is prior art. The finding
that matters is the negative one: HOCT's fork head was trained on 327,265 supervised fork pairs
containing **110 positives**, and its published division recall was zero. A pair head is therefore
unlikely to be won by architecture; the binding constraint is positive division examples, which
points at Zebrahub triples (`FACT-0296`) before any new head is designed.

**Detector swaps (`LEVER-0032` StarDist3D, `LEVER-0033` wide-ResUNet3D)** — demoted. `FACT-0354`
caps the peak-set gain on the honest fold at ~0.45% of annotated cells. They stay open only for
localisation quality, which `FACT-0354`'s note is explicit about not bounding and which
`FACT-0360` says divisions are sensitive to.

**DDPM / noise work** — bounded research branch, per host direction 2026-08-28: denoising or masked
pretraining for localisation robustness only, with synthetic-cell hallucination as the primary
falsifier. It does not displace retention or association work.
