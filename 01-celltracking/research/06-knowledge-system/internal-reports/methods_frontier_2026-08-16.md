# Methods frontier scan (agent report) — 2026-08-16

Raw agent output; evidence store (outside Git). Ranked by EV x novelty x offline-Kaggle-feasibility.

## LEAD — over-propose + learned candidate re-scoring (detection-to-detection attention)
The literal "learned RANKER not a threshold" lever we measured we need. Over-generate candidates
(drop DoG threshold to recall ~0.98, accept 5-20x FP), then a small transformer re-scores each by
attending to neighbours -> restores precision on the rare/small/low-confidence candidates = our failure mode.
- **D2D-Rescore / Learned 3D NMS** (arXiv:2606.03568, IEEE IV 2026; code github.com/rst-tu-dortmund/learned-3d-nms, MIT-style).
  6-layer/64-ch/4-head transformer over per-detection features [pos, size, orient, conf, velocity],
  Fourier-encoded coords, all-pairs attention, BCE vs metric-aware greedy GT match. Gains concentrate on
  sparse/small classes (Bicycle +3.56, Motorcycle +2.93). <1M params, trains in minutes on 1 T4, no external weights.
  Evidence is automotive LiDAR -> mechanism transfers, numbers do NOT; re-measure on our 2 embryos.
- Requires over-proposal first (a re-scorer can't rank what wasn't proposed) -> pair with lower-threshold
  multiscale LoG + a second proposer (Cellpose-SAM / LSM FM).
- First experiment: DoG threshold -> candidate recall ~0.98 on both embryos; per-candidate features
  (physical-um pos, DoG response @3 scales, local intensity, kNN spacing, #neighbours <7um); train the
  re-scorer with focal/BCE vs 7um bipartite GT; sweep operating point cross-embryo.

## 1. HOCT — Higher-Order Cell Tracking Transformer (FROM THE ORGANIZER LAB) *** likely the undisclosed edge ***
arXiv:2607.11754, Bragantini/Theodoro/**Royer, CZ Biohub SF** (the organizers), Jul 2026. CC-BY-NC-ND 4.0.
Code not public yet (likely -> royerlab/tracksdata); REIMPLEMENT from the paper (all eqns/hparams given).
- EDGE-centric transformer. Detections given; candidate graph (links within tau=300px across frames);
  edge token per link = MLP([z_i||z_j]); edges attend under a 3D LINE-TO-LINE geometric prior as a per-head
  attention bias (alternating attractive/repulsive heads). Divisions learned as edge probs (focal gamma=3.5,
  division weight 3.5x, parental softmax per target/gap) + ILP (<=1 parent, <=2 children). Only **19 hand-
  crafted features** (t,z,y,x, eq. diameter, intensity min/max/mean/std, 3x3 inertia tensor, FOV-border dist),
  NO image encoder.
- Why it breaks our plateau: (a) a learned ASSOCIATION ranker built around candidate-graph pathologies
  (divisions entangle node embeddings; sibling edges non-homophilic H_adj~0.01, so node-GNNs can't aggregate);
  (b) its division edge-head = the high-precision fork selector for the +0.06 division ceiling; (c) human-in-loop:
  logistic head on FROZEN HOCT features refit with L-BFGS (AOGM -59% w/ 400 annotations) = drop-in learned
  re-acceptance head, fittable on 2 embryos without overfitting an encoder.
- Offline: 19-dim -> tiny; single-organism (2-embryo) model fits T4x2; anisotropy native via physical-um coords
  + line-to-line distance. Reported Fluo-N3DH-SIM+ CLB 0.997/LNK 1.000/BIO 0.988.
- First experiment: reimplement edge-token + line-to-line bias + parental-softmax head; feed existing
  detections; train embryo A eval B and reverse (official patched scorer); ablate HOCT edges vs current linker
  on adj-edge-J, and HOCT division-prob precision/recall vs the 10% break-even.

## 2. Cellpose-SAM as a 2nd generalizing nuclear proposer (recall + domain generalization)
bioRxiv 2025.04.28.651001 (MouseLand). SAM ViT backbone + Cellpose head; robust to channel shuffle/size/
noise/downsample/**anisotropic blur** (matches our z-anisotropy + 2->1 embryo shift); nuclear (DAPI/Hoechst)
variant. Weights public (verify BSD-3), fits T4, 3D via slice+stitch. Union into candidate pool; fine-tune head only.
First exp: zero-shot on train embryos; measure added recall on the 77-80% DoG misses within 7um.

## 3. Multimodal 3D LSM foundation model — frozen features for the ranker (orthogonal)
arXiv:2605.26026 (May 2026); code github.com/AdinaScheinfeld/lsm_fm_public_repo. SwinUNETR SSL-pretrained on
unlabeled LSM. Use: (a) frozen candidate-patch embeddings -> feed D2D/HOCT re-scorer (learned "is-this-a-nucleus"
signal DoG lacks); (b) few-shot deblur to normalize hidden embryo. Domain-match to zebrafish nuclei UNVERIFIED
-> feature prior, not turnkey.

## 4. Test-time SSL/TTA on the hidden embryo's own volumes (domain generalization; cheapest)
Hidden test volumes are LOCAL at inference (internet-off doesn't stop using data you must process). Self-
supervisedly adapt at submission (masked-volume/denoise SSL, or collapse-safe entropy-min: COME/BEM ICLR'25,
arXiv:2606.02339). SELMA3D (arXiv:2501.03880) = light-sheet-specific proof SSL closes the unseen-domain gap.
Runs in-notebook (<12h); guard collapse. First exp: train A/test B; compare no-adapt vs BN-recalib vs short
masked-SSL fine-tune per direction.

## 5. Cell-TRACTR joint detection+association via track queries (research bet, low feasibility)
PMC12101859 (2025). DETR-style persistent queries -> joint seg+track+division, removes DoG->link handoff.
Mostly 2D/dense-culture; 3D+t light-sheet memory-heavy, unproven on T4x2. Log as research bet.

## Division specifically (bank the +0.06)
Best NEW high-precision fork selector = HOCT's division edge-head (#1). Orthogonal cheap option: frame-order-
flip mitosis detection from partial annotations (arXiv:2307.04113). Detect-then-classify pattern: MIDOG 2025
two-stage localize->classify (arXiv:2509.02597; transfer pattern not weights).

## Flags
- HOCT: no released checkpoint (reimplement); CC-BY-NC-ND permits USE not redistribution of a modified model.
- Cellpose-SAM / "Revisiting foundation models" PDFs paywalled -> license/params inferred (Cellpose historically BSD-3); verify.
- D2D-Rescore evidence is LiDAR; mechanism transfers, numbers don't.
- LSM FM zebrafish-nuclei domain match unverified.
