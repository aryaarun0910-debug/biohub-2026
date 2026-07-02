# Competitive & Recent Methods — Deep Intel (2026-07-02)

Beyond prior reports. Most-actionable first. Cited; unverifiable flagged.

## 1. SPOTIFLOW — full recipe (Nature Methods 2025; github.com/weigertlab/spotiflow) — TOP detector candidate
- **Arch** (`model/config.py`): U-Net backbone, in=1/out=1, initial_fmaps=32, inc x2, levels=4,
  downsample 2, k=3, BN. mode `slim` default. Emits `levels` heatmap heads (per resolution) + 1 flow head
  (4 channels in 3D). `sigma=1.0`, `compute_flow=True`.
- **Heatmap target/loss** (`utils/peaks.py`, `trainer.py`): Gaussian blobs at points, `mode=max`, rendered
  at every pyramid level. Default loss = **Adaptive Wing** (theta=0.5, alpha=2.1, omega=14, eps=1). Per-level
  weight `/4^lv`. Positive re-weight `w = 1 + pos_weight*[max(tgt,pred)>=0.01]` — UP-weights fg, does NOT ignore unlabeled.
- **Stereographic flow** (3D, S^3): to nearest point d=(z,y,x), r=|d|:
  `w'=-(r^2-s^2)/(r^2+s^2)`, `z'=2sz/(r^2+s^2)` (y',x' likewise). At spot w'=+1; far w'=-1.
  Flow loss = L1 on 4-ch, x same pos-weight mask. Decode: `s=sigma/(1+w'+eps)`, offset=(z's,y's,x's) => SUB-VOXEL.
- **CRITICAL CAVEAT**: Spotiflow assumes DENSE annotation; NO ignore mask for unlabeled. Training naively on
  1-6% sparse Kaggle labels penalizes detecting the ~94-99% real-but-unlabeled nuclei. FIX: inject Linajea
  soft-mask (weight=1 within nucleus radius of each point, ~0.01 else) into BOTH heatmap+flow losses via
  `trainer._loss_weight`. This is THE adaptation to make Spotiflow work here.
- **3D first-class** (`is_3d=True`). **Offline pretrained 3D**: `synth_3d`, `smfish_3d` (zips at
  weigertlab/spotiflow-models releases 0.6.0) — synthetic/smFISH domain => fine-tune init, not turnkey.
  Norm: percentile pmin=1.0/pmax=99.8. Extract: threshold heatmap, 3D max-filter NMS, add flow offset.

## 2. Weakly-supervised / PU 3D nuclei from sparse points — the loss math
### 2a. Hirsch & Kainmueller arXiv:2002.02857 — auxiliary CPV loss (github.com/Kainmueller-Lab/aux_cpv_loss)
Add 3 channels; each fg voxel regresses ABSOLUTE vector to its nucleus center-of-mass, SSD/L2 loss,
added unweighted. Makes EVERY fg voxel penalize a merge/split (vs boundary losses that only fire at few
boundary voxels) -> robust instance separation, exactly the errors adjusted-edge-Jaccard punishes.
+CPV improved AP0.5 up to ~4%; 3-label+cpv (0.638) beat StarDist-3D (0.628). Sparse-point ("gauss" baseline:
fixed-radius Gaussian, SSD, NMS) is COMPETITIVE with dense -> point-only training viable.
### 2b/c. Linajea++ arXiv:2208.11467 (won CTC Fluo-N3DH-CE as JAN-US, DET .981/TRA .979)
4D U-Net -> Gaussian cell-indicator + 3D BACKWARD movement vectors (cells divide forward, never merge),
weighted MSE. Separate ResNet18 cell-state {parent,daughter,continuation,polar-body}.
**THE sparse-mask recipe (Supp T5, verbatim)**: pixels outside estimated nucleus radius trained with loss
factor **0.01 (confocal) / 1e-6 (light-sheet)** — soft down-weight, NOT hard ignore. Multiply into detection
loss. Tricks: division oversampling (>=25% iters contain a division); sample tiles centered on random
annotated cell; SWA every 1k iters after 50k; discard maxima <0.2; match radius 15.
### 2d. Division-aware ILP + sSVM weight tuning (verbatim constraints)
Node vars y_parent/y_daughter/y_continue; state `y_parent+y_daughter+y_continue-y_node=0`; division link
`y_parent,u + y_edge,e - y_daughter,v <= 1` (&sym). Objective min<Sw,y>. **Structured-SVM** tunes weights w
(minimize <Sw,y'> - min(<Sw,y> - Delta(y',y)) + lambda|w|^2, Hamming Delta, lambda=0.001) -> tune FP<->FN
toward our metric (division worth 0.1). motile ships this as `fit_weights`.

## 3. ITEC (bioRxiv 2026.03.12.711203) — iterative tracking + error correction
Unsupervised whole-embryo; zebrafish 18.5M cells, claimed >99.7% (abstract; PDF Cloudflare-locked, unverified).
**Min-cost CIRCULATION** modified for large displacement (low frame-rate). Loop: if cost of splitting/merging
a segmentation < original AND constraints hold, correct segmentation, re-track; repeat until cost stops
improving. NO public code found. Transfer: use linking cost as oracle to accept/reject split/merge of
DoG/Spotiflow detections, then re-link — cheap post-hoc booster.

## 4. Cell Tracking Challenge — 3D developmental nuclei winners
Closest analog = **Fluo-N3DL-*** (light-sheet, SPARSE GT over dense nuclei = our regime). Fluo-N3DL-DRO is
brutal (OP_CTB ~0.62). Top teams: **KIT-GE**, **CZB-US (Ultrack)**, **JAN-US (Linajea++)**, KTH (Viterbi),
MU-CZ, Elephant. Transferable: **EmbedTrack** (github.com/kaloeffler/EmbedTrack) = instance embedding +
offset-to-center regression + clustering, joint seg+2-frame link (offset idea ~ Spotiflow flow / Linajea
vectors). Ultrack = strongest turnkey for our exact data type.

## 5. Public solutions for THIS competition
Thin (days old). `inversion` NN starter; `xiaoleilian` classical DoG+Hungarian (~0.826-0.842, exact params
not fetchable — open notebook). No 3rd-party GitHub repos; top ~0.875 writeups unpublished. Actionable public
direction = better detector -> Ultrack/motile ILP.

## 6. motile — offline division-aware ILP on Kaggle (github.com/funkelab/motile)
Backend **ilpy -> SCIP (open-source, offline-OK)**; set preference SCIP/Any. Vars NodeSelected/EdgeSelected/
Appear/Disappear/NodeSplit/NodeMerge. Constraints **MaxChildren(2)** (allow div), **MaxParents(1)** (no merge),
ExclusiveNodes, Pin. Costs NodeSelection/EdgeSelection/EdgeDistance(gate at 7um anisotropic)/Appear/Disappear/
Split. Selected node w/ 2 children => NodeSplit => division. Built-in **fit_weights sSVM** to tune costs to our
metric offline. Solve block-wise over overlapping frame windows for scale.

**Recommended offline stack**: Spotiflow-3D (masked loss) + CPV aux head -> candidate graph (KNN gate) ->
motile ILP (MaxChildren=2/MaxParents=1, EdgeDistance 7um, Split via fit_weights) -> optional ITEC split/merge-
retrack. SCIP => fully offline.

## Caveats
Spotiflow does NOT ignore unlabeled (source-confirmed) — must add Linajea mask. ITEC from abstract only,
no code. CTC team attributions approximate. Kaggle classical-baseline DoG params not extracted. Spotiflow 3D
pretrained = synthetic/smFISH domain, fine-tune only.
