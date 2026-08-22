---
id: 01-research-direction/directional-updates
title: Directional Updates
area: 01-research-direction
status: active
updated: '2026-08-17'
owner: biohub
links: []
tags:
- steering
- log
---

# Directional Updates

> Dated steering log. [`../00-system/handoff.md`](../00-system/handoff.md) points here for the current direction. Newest first.

## 2026-08-17 (newest) — H1 imaging gate LIFTED: packaged Zebrahub crops exist on Kaggle

The stated blocker on `bet-zebrahub-retrain` — "Zebrahub on disk is TRACKS-ONLY; retrain needs
imaging (150–232 GB at level-1)" — is **no longer binding for a pilot**. Another competitor
(`kkunizaw`) has published the imaging, already cropped, as public Kaggle datasets:

| dataset | size | contents (verified locally) |
|---|---|---|
| `kkunizaw/biohub-zh001r` | 363 MB | `zh001r_iso.npy` **(72, 20, 64, 64, 64) uint8** real ZSNS001 imaging; `zh001r_tgt.npy` same shape (target volume); `zh001r_nodes.npz` 1440 arrays `f{crop*20+t}` of `(N,4) float32 = [t,z,y,x]`, ~900 nuclei/frame, coords inside the 64³ box |
| `kkunizaw/biohub-zmnscrops` | 3.66 GB | `zmns001_crops.npz` 5.76 GB + `zmns002_crops.npz` 7.13 GB — "Windowed crops derived from the public Zebrahub multi-view imaging dataset (zebrahub.sf.czbiohub.org, CZ Biohub / Royer Lab)… No competition data included." (contents not yet opened) |

`zh001r_iso.npy` was downloaded and inspected: real intensity data (min 0, max 255, mean 51.8,
86% nonzero), 72 crops × 20 timepoints of 64³ **isotropic** volumes with matching per-frame
nuclei coordinates. That is a complete, ready-to-train detector set in **754 MB** instead of the
150–232 GB acquisition the level-1 plan budgeted for.

**Why this matters.** H1 is the only stated path to top-3 and it was gated on data acquisition,
not on method. The gate is now a *format* question rather than a *bandwidth* question, and Kaggle
datasets attach to kernels directly — so an H1 pilot can run on Kaggle GPU with zero local
download and no kernel internet.

**But it unblocks only half of H1 (measured same day, see the experimental record).** The node
arrays are `(N,4) = [t,z,y,x]` with **no track identity**, so `zh001r` supervises the **detector**
retrain and **cannot** supervise the **edge/association** retrain — which is the half our own
thesis says the plateau lives in. Our `h1r_fetch_imaging.py` level-1 stream does carry track ids.
Treat the two as complementary lanes, not as a replacement.

**Scaffold built and passing (CPU, no GPU spent):** `scripts/win_bet/h1r_zh001r_audit.py`
(integrity gate: structure, label/imaging alignment 2.29× contrast, geometry ruler) and
`scripts/win_bet/h1r_zh001r_smoke.py` (packaged crop → deployed `TemporalUNet3D` →
`compute_detection_loss` → backward → step; det_loss 0.7028→0.5787, UNet + detect_head both
receive grad; reproduced on a second crop).

**Caveats (do not skip before building on this).**
- It is a **third party's preprocessing**. Their "iso" scale was unstated — now **measured**
  (same day, see the experimental record): a nucleus-size radial-profile ruler puts it at
  **~1.6–1.8 µm/voxel isotropic** (median 1.762, 1.084× our grid), i.e. the *same geometry
  family* as our deployed 64³ @ 1.625 µm detector input, not a separate resolution lineage.
  This removes the level-1 premise that a retrain "voids the deployed 0.915 anchor". The ruler
  assumes comparable nucleus size across stages; their nuclei are denser (later stage), which
  would bias the estimate upward — true scale may be nearer 1.625 µm.
- Node labels are theirs (presumably Ultrack-derived); we have not audited them against Zebrahub.
- ZSNS001 only in `zh001r`; `zmnscrops` covers ZSNS001+002 as raw crops.
- Zebrahub is **CC BY-NC**. Host has cleared Zebrahub imaging + tracks (#734330), so use is
  permitted, but per [../04-data/data-governance.md](../04-data/data-governance.md) the license of
  every shipped external asset must be recorded in the submission notebook.

**Competitive read.** `zh001r` was uploaded **2026-08-17** (today) and `zmnscrops` 2026-08-16 —
a rival is actively executing the Zebrahub retrain right now. This confirms the H1 lane is the
live frontier and implies the plateau will move.

**Next (unauthorised until green-light):** voxel scale is now established (above), so the open
choice is (a) their crops as a pilot shortcut — geometry-compatible, ~750 MB, attachable to a
kernel — versus (b) our own `scripts/win_bet/h1r_fetch_imaging.py` level-1 stream, which we
control end-to-end and whose labels we can audit. Remaining pre-pilot checks: audit their node
labels, open `zmnscrops`, and settle intensity renormalisation (their uint8 vs our uint16). The
falsification for `bet-zebrahub-retrain` is unchanged.

## 2026-08-17 (earlier) — Motion-gate PROMOTED (first measured win off the plateau)

`bet-motion-gate` **WON** on the deployment substrate. Clean paired LOEO (four Kaggle T4×2
kernels; P0-B base, official `tracking_cellmot` scorer, identical crops, only
`BIOHUB_ARMB_FLOW_GATE` differs):

| fold | family | base (armB off) | armB | delta |
|---|---|---|---|---|
| 0 | 44b6 | 0.9037 | 0.9181 | +0.0144 |
| 1 | 6bba | 0.7051 | 0.7141 | +0.0090 |

Bilaterally positive, min-fold **+0.0090** > +0.005. (Correction: I first mis-anchored against
the E0c numbers 0.7595/0.6490 — wrong substrate; the paired baseline above is the honest read.)
**Next:** build the P3+armB *submission* kernel and hand the `submitcmd` (human submits, per the
factory discipline). Then the freshest cheap lever is `bet-subvoxel-refine` (scaffolded).

## 2026-08-17 (later) — Quick-wins swarm: shortlist before the heavy phase

Four-agent swarm (reports in `../06-knowledge-system/internal-reports/{quickwins_internal,
competitive_refresh,novel_crossdomain,redteam_blindspots}_2026-08-17.md`; +5 findings in
research.sqlite; portfolio updated in [research-bets.md](research-bets.md) + [bets.yaml](bets.yaml)).

**Ranked shortlist (EV × cheapness) — cheap wins to bank before the H1 retrain:**
1. **Ship Arm B / motion-gate** (`bet-motion-gate`) — unanimous #1 (quickwins + red-team).
   +0.0088 pooled / +0.0074 min-fold, P(d>0)=1.0; already built as the two committed `p3_armb`
   LOEO kernels. New insight: the gate is **pure geometry (never reads `prob`)**, so harmonic
   can't break it. Cheap de-risk: re-run armB on local `p0strict` graphs with the `prob` cost
   term zeroed (~1 h CPU) → then the two GPU kernels confirm. **The near-term bank.**
2. **Sub-voxel centroid refinement** (`bet-subvoxel-refine`) — measured scorer cliff at σ≈2 µm.
   **Audit first** (CPU minutes) whether the detector already refines; if integer-argmax, add
   parabolic refine → potentially several points. Highest-uncertainty / highest-upside cheap lever.
3. **Meta-ranker + OT linker** (`bet-meta-ranker`, `bet-ot-linker`) — the cheap "ranker not
   threshold" (GBM/nnPU/conformal on existing features) and an unbalanced-Sinkhorn linker, both
   CPU on OOF geffs, both cross-family-honest (fit A / eval B).

**Corrections & confounds (red-team):**
- **Node-budget "closed" was false** (used the forbidden placeholder-movie substrate) but settled
  at **+0.001 bilateral** on the full 199-crop sweep — sub-bar footnote, not a lever.
- **Every positive lever we hold is an E0c number**; the deployed P0-strict substrate differs
  materially → **persist P3 OOF graphs once** (fold into the motion-gate GPU session) = highest-
  value unspent compute; converts E0c replays to deployment-substrate replays.
- **Divisions soundly dead** (both families, both factors); node-count gap share is E0c-relative
  and already banked in P3 → both reinforce the **H1 edge-retrain** pivot.

**Infra picked up:** adopt sleepymegacat's 80-line numpy scorer for fast offline LOEO CV;
investigate `kkunizaw/biohub-zmnscrops` (3.66 GB packaged Zebrahub crops) — may shrink the H1
level-1 imaging acquisition. Leaderboard barely moved (0.945+ tier crowding; public ceiling now 0.918).

## 2026-08-17 — H1 Zebrahub retrain scoped; level-1 acquisition decided

- **Committed to the H1 edge/detector retrain** as the real gap-closer (the ~0.029 to the
  leader is "not attributable to any recoverable public mechanism" — it needs external-data
  retraining). Parked the tested-negative division ranker (laneD: learned MLP ≈/< frozen
  geometry) and the motion-gate (still ready, committed `8fb5e83`).
- **Data gate found:** Zebrahub on disk is TRACKS-ONLY (no imaging). The detector/edge model
  is a UNet over raw volumes → retrain needs imaging. See [[biohub-zebrahub-imaging-gap]].
- **Disk-vs-resolution decision:** level-0 (exact deployed res, z-full/xy÷4) is ~1.85 TB and
  won't fit in 383 GB free. **Chose level-1** (~150–232 GB, z-half/xy-half) — retrain the
  whole pipeline at that resolution, re-deriving the competition features to match (this
  **voids the deployed 0.915 detector anchor**; the retrained model is a new lineage).
- **Feasibility validated (local, cheap):** remote OME-Zarr level-1 chunks stream + decode via
  urllib + numcodecs (zarr 3.3.0); one chunk 26.3 MB in 6.5 s. Training surface understood:
  `vendor/.../train_unet_transformer.py` reads zarr level "0" strided by `downsample`; the
  level-1 retrain needs the loader pointed at level "1" with adjusted downsample/scale.
- **Staged plan (smoke → pilot → full; GPU + full download held for green-light):**
  1. *smoke:* **DONE (2026-08-17).** `scripts/win_bet/h1r_fetch_imaging.py` streams+stores
     level-1 zarr crops; `scripts/win_bet/h1r_train_smoke.py` runs one full CPU step on a
     ZSNS003 t0→t1 crop: UNet(level-1) → node-feature index → edge transformer → loss 0.0121
     → backward, UNet receives grad. Plumbing proven (nuclei are shell-distributed → crops
     must be node-centred; level-0-voxel tracks map to level-1 via ÷2).
  2. *pilot:* retrain at level-1 on 1 embryo + re-derived competition data; LOEO-validate the
     delta vs a level-1 competition-only baseline. **Falsification:** no bilateral LOEO gain
     from the Zebrahub augmentation ⇒ external imaging doesn't transfer at level-1; kill.
  3. *full:* 3 embryos (ZSNS003/004/005) + full retrain only if the pilot clears.

## 2026-08-16 — Reopen + research-machine restructure

- **Reopened** the project (reverses the 2026-08-07 closure). Mission: top-3 (see
  [scientific-mission.md](scientific-mission.md)). Frontier cracked by the swarm: the edge is a
  retrained/generalising model on external Zebrahub, unlocked 2026-08-13 (six days after we quit).
- **Restructured** the repo into this research machine (one folder; `_RESEARCH` folded in;
  `reports/` absorbed). Storage is 3-tier (`/temp` → `_evidence/` → `research/`).
- **In flight:** `bet-motion-gate` — two P3+armB LOEO-export kernels built + audited locally
  (folds 0/1, 18/18 edits each). Kaggle GPU run **parked** for explicit green-light. On a
  bilaterally-positive result vs P3-alone, build the submission kernel + factory `submitcmd`.
- **Next program:** `bet-zebrahub-retrain` + `bet-learned-ranker` (GPU T4×2), building on the
  restored `scripts/win_bet/h1t_zebrahub_events.py` + `h1t_conditional_ranker.py`.

## Prior chronology

Deployment lineage E0c (0.889) → v122 (0.908) → P0-A (0.913) → P0-B (0.914) → **P3 harmonic
(0.915, deployed)**. Full history: [../06-knowledge-system/lab-notebooks.md](../06-knowledge-system/lab-notebooks.md).
