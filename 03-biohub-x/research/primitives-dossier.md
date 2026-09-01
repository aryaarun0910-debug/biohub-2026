# Biohub Cell Tracking — Hypercompetitive Research-Primitives Dossier

**Version:** 2026-09-01  
**Purpose:** a research and experiment registry for pushing a 3D+t cell-tracking system beyond the public ~0.935 architecture family and toward a robust top-leaderboard system.  
**Competition focus:** Biohub — Cell Tracking During Development.  

> **Core rule:** this is a *primitive tournament*, not a primitive pile. A list of 100+ methods does not create a better system by itself. Every primitive must earn promotion through end-to-end validation with the official Biohub metric, hard-case analysis, stability across held-out embryos/directions, runtime, and provenance/license checks.

---

## 0. Read this first

The public Biohub baseline repository is already a strong systems baseline: a `TemporalUNet3D` detects cell centers, a `SimpleNodeTransformer` scores links between adjacent frames, and sparse labels supervise annotated edges. The official repository also states that its released baseline was trained for only three epochs and was **not trained to convergence**, which means training quality itself is still a meaningful axis before replacing the architecture.

The official metric makes this a graph-reconstruction problem, not a mask-beauty contest: predicted nodes are matched to ground truth by centroid distance (up to 7 µm), edge Jaccard dominates the score, and division Jaccard receives a 0.1 weight. The dataset is sparsely labeled. Therefore, the highest-value research directions are likely to be:

1. **hard-case detection recall** — crowded and division-heavy frames;
2. **identity-oriented representation learning** — make nearby visually similar cells separable;
3. **longer/higher-order temporal reasoning** — not only `t -> t+1`;
4. **multi-hypothesis inference** — do not commit too early to one detector/segmentation;
5. **global graph consistency** — ILP/min-cost-flow/factor-graph constraints;
6. **metric-aware calibration** — node count, edge threshold, division threshold, birth/death costs;
7. **efficient cascades** — cheap model for obvious links, expensive specialist for ambiguous ones;
8. **offline training maximization** — the Kaggle notebook should be a lean, frozen inference program.

### Primary competition references

- [Biohub public baseline / evaluation code — GitHub](https://github.com/royerlab/kaggle-cell-tracking-competition)
- [Official metric description](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [Official metric implementation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/metrics.py)

---

# 1. The architecture to compete with — not merely copy

```text
4D microscopy volume
        │
        ▼
┌──────────────────────────────┐
│  SPECIALIST A: PROPOSALS     │
│  multi-detector hypothesis   │
│  bank                        │
└─────────────┬────────────────┘
              │ candidate cells + uncertainty
              ▼
┌──────────────────────────────┐
│  SPECIALIST B: IDENTITY      │
│  3D crop SSL / metric        │
│  learning                    │
└─────────────┬────────────────┘
              │ identity embeddings
              ▼
┌──────────────────────────────┐
│  SPECIALIST C: GEOMETRY      │
│  physical coords, local      │
│  tissue graph, equivariance  │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SPECIALIST D: MOTION        │
│  velocity / scene flow /     │
│  local deformation prior     │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SPECIALIST E: ASSOCIATION   │
│  pairwise + temporal window  │
│  + higher-order edge model   │
└─────────────┬────────────────┘
              │ edge logits
              ▼
┌──────────────────────────────┐
│  SPECIALIST F: DIVISION      │
│  parent-daughter hyperedge   │
│  / fork temporal model       │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SPECIALIST G: UNCERTAINTY   │
│  ensemble disagreement /     │
│  margin / entropy routing    │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SPECIALIST H: GLOBAL SOLVER │
│  ILP / min-cost flow /       │
│  lineage constraints         │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  SPECIALIST I: CALIBRATION   │
│  threshold + node-count +    │
│  graph repair                │
└─────────────┬────────────────┘
              ▼
        submission.csv
```

## Specialist interface contracts

| Specialist | Input | Output | Diagnostic ceiling metric |
|---|---|---|---|
| Proposal/detection | raw `(T,Z,Y,X)` | candidate nodes, confidence, optional masks/features | GT-node recall within 7 µm; candidate count ratio |
| Identity | 3D crop / local volume | 128–512d embedding | same-track retrieval@k; hard-negative margin |
| Geometry | physical point cloud | context embedding / pair features | neighbor consistency; equivariance tests |
| Motion | trajectories / frame pairs | displacement prior + uncertainty | successor-in-gate recall; displacement error |
| Association | candidate graph | edge logits | candidate-edge AUROC/AP; GT-edge rank |
| Division | parent + candidate daughters + context | fork/hyperedge score | division PR/Jaccard proxy |
| Uncertainty | model logits/ensembles | uncertainty scalar / route flag | error-detection AUROC |
| Solver | candidate graph + costs | legal lineage graph | official edge/division metric |
| Calibration | graph + confidence | final graph | **official score** |

---

# 2. Promotion system: champion–challenger, not architecture soup

Every slot gets **one champion** and several challengers. A challenger is promoted only when it passes all of the following gates.

### Gate A — end-to-end score

Use the official Biohub metric locally. Module F1/AP is only diagnostic. A detector with slightly lower segmentation F1 can still be superior if it yields a better lineage graph.

### Gate B — hard-case slices

Log separately:

- crowded frames;
- frames ±2 around divisions;
- high-displacement edges;
- low-SNR frames;
- boundary-of-volume tracks;
- high local cell-density quartile;
- ambiguous pairwise margin (`p1 - p2` small).

### Gate C — generalization

Prefer group splits by embryo/dataset/direction. Do not randomly split adjacent frames across train/validation. If a gain exists only on one embryo direction, it is not a robust promotion.

### Gate D — variance

Tiny gains need repeated seeds or bootstrap confidence intervals. Treat a `+0.001` public-LB movement as suspect until reproduced offline.

### Gate E — runtime

The final Kaggle notebook must finish within 12 hours. Heavy specialists should be routed only to ambiguous cases or distilled.

### Gate F — provenance

No checkpoint enters the final artifact until code, weight, training-data and redistribution/use terms are recorded.

## Suggested promotion score

```text
40%  end-to-end official-score uplift
15%  hard-case uplift
15%  cross-split / cross-seed stability
10%  candidate-recall / error-ceiling improvement
10%  inference cost
10%  provenance + integration risk
```

---

# 3. The highest-EV experiments first

These are the experiments I would run before trying hundreds of lower-priority ideas.

## S1 — multi-hypothesis proposal bank

Do **not** force the tracker to inherit one detector's mistakes. Generate candidate centers from several independent families, merge near-duplicates in physical coordinates, retain source-specific confidence/features, and let the downstream graph choose.

Candidate champion set:

- `TemporalUNet3D` center heatmap (competition-native);
- **FOCUS-3D** volumetric segmentation;
- **StarDist3D** center/instance proposals;
- optionally nnU-Net/MedNeXt center head or a lightweight LoG/DoG detector for high-recall rescue.

The conceptual inspiration is strongly aligned with **Ultrack**, which explicitly delays commitment by evaluating multiple segmentation hypotheses and using temporal consistency/global optimization to choose among them.

## S2 — ASCENT-style identity model, retrained for zebrafish

Use a 3D crop encoder with position + frame context. Train with deformation-aware positives and **nearby cells as hard negatives**. Do not rely on random negatives: the dangerous negative is the cell 4–15 µm away that the tracker could realistically confuse.

Suggested loss family:

```text
L = L_InfoNCE_hard
  + λ1 L_temporal_cycle
  + λ2 L_neighbor_context
  + λ3 L_variance/covariance_regularization
```

ASCENT is directly relevant because it learns annotation-free 3D neuronal identity embeddings invariant to tissue deformation.

## S3 — HOCT as the hard-edge specialist

The 0.935-family adjacent-frame pair model is strongest where association is obvious. Route only low-margin/high-disagreement candidate subgraphs into a **HOCT-style edge-centric model**. HOCT is unusually relevant because it explicitly models candidate **edges** and their higher-order relations, addressing division entanglement.

## S4 — Trackastra-style temporal windows

Pairwise `t -> t+1` association throws away history. Benchmark windows such as `t±2` and `t±4`, but keep the candidate graph sparse. Trackastra is a direct proof that detection-level tokens over a temporal window can learn division-aware cell association.

## S5 — physical geometry + local tissue deformation specialist

Operate in microns, not raw voxels:

```text
p_phys = (1.625*z, 0.40625*y, 0.40625*x)
```

Add:

- relative physical position;
- local neighbor graph;
- estimated local affine/Kabsch displacement;
- EGNN/equivariant context or a lightweight point transformer;
- predicted velocity/acceleration.

This is likely much cheaper than making the image encoder larger.

## S6 — division as a hyperedge, not two independent links

Score the set `(parent, daughter_1, daughter_2)` jointly. Include daughter symmetry, midpoint displacement, appearance change, intensity/volume behavior, neighbor reorganization and multi-frame context. Compare ForkHead, DeepSets, Set Transformer, hypergraph networks and HOCT-style edge interactions.

## S7 — uncertainty cascade

Use cheap pairwise inference everywhere. Invoke expensive temporal/higher-order reasoning only when:

```text
margin = p_best - p_second < threshold
OR ensemble disagreement > threshold
OR local density > threshold
OR proposed division exists
```

This is the natural way to spend a 12-hour inference budget aggressively without wasting compute on easy links.

## S8 — metric-aware final calibration

Search detection threshold, NMS radius, candidate radius, birth/death costs, edge threshold, division prior, fork vetoes and node-count calibration using the **official metric**, not a generic tracking score.

---

# 4. Curated research source anchors

These are especially high-priority source-of-record links.

- **FOCUS-3D** — [GitHub](https://github.com/yu-lab-vt/FOCUS-3D) · [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.25.746907v1)
- **HOCT** — [GitHub](https://github.com/royerlab/hoct) · [arXiv](https://arxiv.org/abs/2607.11754)
- **Trackastra** — [GitHub](https://github.com/weigertlab/trackastra) · [arXiv](https://arxiv.org/abs/2405.15700)
- **Ultrack** — [GitHub](https://github.com/royerlab/ultrack) · [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0) · [ImageJ overview](https://imagej.github.io/plugins/ultrack)
- **ASCENT** — [GitHub](https://github.com/lu-lab/ascent) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1)
- **StarDist** — [GitHub](https://github.com/stardist/stardist)
- **EmbedSeg** — [GitHub](https://github.com/juglab/EmbedSeg) · [arXiv](https://arxiv.org/abs/2101.10033)
- **Cellpose** — [GitHub](https://github.com/MouseLand/cellpose)
- **CellSAM** — [GitHub](https://github.com/vanvalenlab/cellSAM) · [Nature Methods](https://www.nature.com/articles/s41592-025-02879-w)
- **Cell-DINO** — [GitHub documentation](https://github.com/facebookresearch/dinov2/blob/main/docs/README_CELL_DINO.md) · [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828)
- **DINOv2** — [GitHub](https://github.com/facebookresearch/dinov2) · [arXiv](https://arxiv.org/abs/2304.07193)
- **DINOv3** — [GitHub](https://github.com/facebookresearch/dinov3) · [arXiv](https://arxiv.org/abs/2508.10104)
- **I-JEPA** — [GitHub](https://github.com/facebookresearch/ijepa) · [arXiv](https://arxiv.org/abs/2301.08243)
- **V-JEPA 2** — [arXiv](https://arxiv.org/abs/2506.09985) · [Meta research page](https://ai.meta.com/research/vjepa/)
- **t-SNE** — [JMLR](https://jmlr.org/papers/v9/vandermaaten08a.html) · [ResearchGate mirror](https://www.researchgate.net/publication/228339739_Viualizing_data_using_t-SNE)
- **openTSNE** — [Docs](https://opentsne.readthedocs.io/en/stable/)
- **PHATE** — [GitHub](https://github.com/KrishnaswamyLab/PHATE)
- **Point Transformer V3** — [GitHub](https://github.com/Pointcept/PointTransformerV3) · [arXiv](https://arxiv.org/abs/2312.10035)
- **SuperGlue** — [GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork) · [arXiv](https://arxiv.org/abs/1911.11763)
- **LightGlue** — [GitHub](https://github.com/cvg/LightGlue) · [arXiv](https://arxiv.org/abs/2306.13643)
- **motile** — [GitHub](https://github.com/funkelab/motile) · [Docs](https://funkelab.github.io/motile/)

---

# 5. Primitive registry

### Legend

- **S** — benchmark immediately; unusually aligned with Biohub.
- **A** — strong challenger.
- **B** — useful exploration / ablation.
- **C** — diagnostic or low-priority adaptation.
- **Direct** — can plausibly sit in the inference/training stack.
- **Adapt** — concept/code needs meaningful 3D+t adaptation.
- **Diagnostic** — use to understand representations/errors, not usually as production geometry.
- **Infra** — research/training/deployment tool.
- **License: REVIEW** — do not ship weights/code until terms and training-data provenance are checked.
- **License: RED** — known non-commercial/restricted provenance issue for a cash-prize setting; method may still inspire a clean-room reimplementation trained only on permitted data.


## A. Detection / segmentation / proposal generation

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| A01 | TemporalUNet3D | Direct | Competition-native 3D center heatmap + temporal attention. Train to convergence before assuming architecture is saturated. | S | [Repo](https://github.com/royerlab/kaggle-cell-tracking-competition) | BSD-3 code |
| A02 | FOCUS-3D | Direct | Generalizable volumetric cell segmentation; use instance centers/features as a second proposal family and specifically audit crowded/division frames. | S | [GitHub](https://github.com/yu-lab-vt/FOCUS-3D) · [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.25.746907v1) | REVIEW weights/data |
| A03 | StarDist3D | Direct | Star-convex polyhedra yield sharp center/instance proposals for blob-like nuclei. Strong independent detector family. | S | [GitHub](https://github.com/stardist/stardist) | BSD-3 code; verify weights/data |
| A04 | Ultrack-style multi-hypothesis segmentation bank | Direct | Union proposals from multiple algorithms/thresholds and defer selection until temporal/global optimization. | S | [GitHub](https://github.com/royerlab/ultrack) · [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0) | BSD-3 code |
| A05 | nnU-Net v2 | Direct | Self-configuring supervised 3D segmentation backbone; use as a robust center/boundary baseline and exploit sparse-ignore labels where appropriate. | A | [GitHub](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0 code; verify pretrained provenance |
| A06 | MedNeXt | Direct | Modern ConvNeXt-like 3D backbone; challenger for center heatmaps when U-Net capacity/receptive-field is limiting. | A | [GitHub](https://github.com/MIC-DKFZ/MedNeXt) · [arXiv](https://arxiv.org/abs/2303.09975) | Review repo/weights |
| A07 | SwinUNETR | Adapt | Transformer-UNet hybrid for 3D volumes; test whether broader context helps dense embryos. | A | [arXiv](https://arxiv.org/abs/2201.01266) · [MONAI](https://github.com/Project-MONAI/MONAI) | Apache-2.0 MONAI code |
| A08 | UNETR | Adapt | ViT encoder + 3D decoder; alternative long-range spatial context. | B | [arXiv](https://arxiv.org/abs/2103.10504) | Review implementation/weights |
| A09 | SegResNet | Direct | Efficient MONAI 3D residual segmentation model; good speed/accuracy control experiment. | B | [MONAI](https://github.com/Project-MONAI/MONAI) | Apache-2.0 |
| A10 | DynUNet | Direct | Dynamic 3D U-Net family in MONAI; useful architecture-search baseline. | B | [MONAI](https://github.com/Project-MONAI/MONAI) | Apache-2.0 |
| A11 | U-Mamba | Adapt | State-space/conv hybrid for biomedical segmentation; test long-range context with lower attention cost. | B | [arXiv](https://arxiv.org/abs/2401.04722) | REVIEW implementation/weights |
| A12 | 3D U-Net | Direct | Canonical volumetric baseline; invaluable as a low-complexity control. | B | [Paper](https://arxiv.org/abs/1606.06650) | Method |
| A13 | V-Net | Direct | Volumetric FCN with Dice-style learning; another control for dense 3D masks. | C | [arXiv](https://arxiv.org/abs/1606.04797) | Method |
| A14 | Attention U-Net 3D adaptation | Adapt | Attention gates can suppress background and emphasize cell centers; compare against temporal attention. | C | [arXiv](https://arxiv.org/abs/1804.03999) | Method |
| A15 | EmbedSeg | Direct | Spatial-embedding instance segmentation for 2D/3D microscopy; useful when touching cells break watershed. | A | [GitHub](https://github.com/juglab/EmbedSeg) · [arXiv](https://arxiv.org/abs/2101.10033) | Review repo/data |
| A16 | PanSeg | Direct | Dense 3D instance segmentation with neural boundary prediction + graph partitioning; interesting for crowded tissues. | A | [GitHub](https://github.com/kreshuklab/panseg) | Review weights/data |
| A17 | Omnipose | Adapt | Morphology-independent segmentation family; use mainly as diversity in proposal ensemble. | B | [GitHub](https://github.com/kevinjohncutler/omnipose) · [Nature Methods](https://www.nature.com/articles/s41592-022-01639-4) | REVIEW weights/data |
| A18 | Cellpose 4 / cpdino | Adapt | Strong general segmentation proposals and new DINOv3 backbones; useful for research comparison. | B | [GitHub](https://github.com/MouseLand/cellpose) | RED: repository states models trained on CC-BY-NC data |
| A19 | Cellpose-SAM / cpsam_v2 | Adapt | Independent segmentation family; potentially useful as proposal diversity. | B | [GitHub](https://github.com/MouseLand/cellpose) | RED: CC-BY-NC training-data provenance |
| A20 | CellSAM + CellFinder | Adapt | Foundation-model segmentation plus learned detector; benchmark zero/few-shot proposal recall. | B | [GitHub](https://github.com/vanvalenlab/cellSAM) · [Nature Methods](https://www.nature.com/articles/s41592-025-02879-w) | REVIEW: repo notes Cellpose-derived data in evaluation dataset |
| A21 | micro-sam | Adapt | SAM-based microscopy tooling, useful for annotation acceleration and potential proposal generation. | B | [GitHub](https://github.com/computational-cell-analytics/micro-sam) · [Nature Methods](https://www.nature.com/articles/s41592-024-02580-4) | Review weights/data |
| A22 | Mask2Former 3D adaptation | Adapt | Query-based mask prediction can inspire proposal-set formulation rather than dense per-voxel commitment. | B | [GitHub](https://github.com/facebookresearch/Mask2Former) · [arXiv](https://arxiv.org/abs/2112.01527) | Review |
| A23 | SAM-Med3D | Adapt | 3D promptable segmentation research primitive; useful as architecture inspiration or auxiliary annotation tool. | C | [GitHub](https://github.com/uni-medical/SAM-Med3D) · [arXiv](https://arxiv.org/abs/2310.15161) | REVIEW |
| A24 | CenterNet-style 3D heatmap | Adapt | Predict centers directly and avoid mask complexity; naturally aligned with centroid metric. | A | [GitHub](https://github.com/xingyizhou/CenterNet) · [arXiv](https://arxiv.org/abs/1904.07850) | Method/code review |
| A25 | LoG / DoG blob detector | Direct | Cheap, diverse high-recall rescue proposals; valuable ensemble member because its errors differ from deep nets. | B | [scikit-image blobs](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) | Permissive library |
| A26 | Seeded watershed + learned boundaries | Direct | Separate touching cells using learned center/boundary maps; pair with multi-hypothesis thresholds. | A | [scikit-image watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) | Permissive library |

## B. Identity / self-supervised / metric learning

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| B01 | ASCENT / NETr | Direct | Annotation-free 3D identity embedding using appearance, position and frame context; retrain on zebrafish crops with deformation augmentations. | S | [GitHub](https://github.com/lu-lab/ascent) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1) | MIT code; verify checkpoints/data |
| B02 | Hard-negative InfoNCE | Direct | Positive = same/pseudo-same cell; negatives = nearby plausible competitors. Optimizes exactly the confusion set the linker sees. | S | [SimCLR paper](https://arxiv.org/abs/2002.05709) | Method |
| B03 | DINOv2 | Adapt | Dense/general visual features; compare frozen 2D slice/projection features, 2.5D fusion, or retrained microscopy encoder. | A | [GitHub](https://github.com/facebookresearch/dinov2) · [arXiv](https://arxiv.org/abs/2304.07193) | Apache-2.0 model card; verify exact weights |
| B04 | DINOv3 | Adapt | Very strong dense features; test as 2D/2.5D teacher or feature source, not assumed 3D-native. | A | [GitHub](https://github.com/facebookresearch/dinov3) · [arXiv](https://arxiv.org/abs/2508.10104) | REVIEW custom DINOv3 license |
| B05 | Cell-DINO | Adapt | Direct evidence that DINO-style SSL learns useful cellular morphology embeddings; copy the research idea, not restricted weights. | A | [GitHub docs](https://github.com/facebookresearch/dinov2/blob/main/docs/README_CELL_DINO.md) · [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828) | RED: code CC-BY-NC; weights FAIR Non-Commercial Research |
| B06 | I-JEPA | Adapt | Predict masked latent regions rather than pixels; useful template for learning cell-state representations without hand-crafted invariances. | A | [GitHub](https://github.com/facebookresearch/ijepa) · [arXiv](https://arxiv.org/abs/2301.08243) | REVIEW license |
| B07 | V-JEPA 2 objective | Adapt | Adapt latent temporal prediction to 3D+t microscopy: predict future local/cell embedding from context. | S | [arXiv](https://arxiv.org/abs/2506.09985) · [Meta](https://ai.meta.com/research/vjepa/) | Method; verify code/weights terms |
| B08 | Masked Autoencoder (MAE) | Adapt | Pretrain 3D volume encoder by masking most patches; scalable unlabeled pretraining baseline. | A | [GitHub](https://github.com/facebookresearch/mae) · [arXiv](https://arxiv.org/abs/2111.06377) | Review code/weights |
| B09 | DINO self-distillation | Adapt | Teacher-student self-distillation with multi-crop views; strong baseline for unlabeled cell crops. | A | [GitHub](https://github.com/facebookresearch/dino) · [arXiv](https://arxiv.org/abs/2104.14294) | Review |
| B10 | SimCLR | Direct | Simple contrastive baseline; ideal control before more complex SSL. | B | [arXiv](https://arxiv.org/abs/2002.05709) | Method |
| B11 | MoCo v3 | Adapt | Momentum-encoder contrastive learning; good when batch size is constrained. | B | [GitHub](https://github.com/facebookresearch/moco) · [arXiv](https://arxiv.org/abs/2104.02057) | Review |
| B12 | BYOL | Adapt | Non-contrastive teacher/student SSL; compare collapse resistance and morphology preservation. | B | [arXiv](https://arxiv.org/abs/2006.07733) | Method |
| B13 | VICReg | Direct | Variance/invariance/covariance regularization; attractive for small, visually similar cell crops. | A | [GitHub](https://github.com/facebookresearch/vicreg) · [arXiv](https://arxiv.org/abs/2105.04906) | Review |
| B14 | Barlow Twins | Direct | Redundancy reduction SSL; another low-complexity identity representation control. | B | [GitHub](https://github.com/facebookresearch/barlowtwins) · [arXiv](https://arxiv.org/abs/2103.03230) | Review |
| B15 | iBOT | Adapt | Masked-image modeling + online tokenizer; useful if patch-level morphology matters. | B | [arXiv](https://arxiv.org/abs/2111.07832) | Review |
| B16 | Supervised Contrastive Learning | Direct | Use sparse true edges/tracklets as positives while leveraging many negatives. | A | [arXiv](https://arxiv.org/abs/2004.11362) | Method |
| B17 | Triplet loss | Direct | Anchor/true-successor/hard-neighbor triplets; easy to audit and mine. | B | [Paper](https://arxiv.org/abs/1503.03832) | Method |
| B18 | Circle Loss | Direct | Unified pair-similarity optimization; candidate for more stable hard-positive/hard-negative weighting. | B | [arXiv](https://arxiv.org/abs/2002.10857) | Method |
| B19 | ProxyNCA++ / proxy metric learning | Adapt | Compact identity-space learning with proxies; useful if tracklet labels become reliable. | C | [arXiv](https://arxiv.org/abs/1911.13063) | Method |
| B20 | Temporal cycle-consistency embedding | Direct | Enforce `t -> t+1 -> t` identity consistency and multi-step consistency on pseudo-tracks. | S | [TimeCycle](https://arxiv.org/abs/1904.07846) | Method inspiration |

## C. Manifold learning / embedding diagnostics

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| C01 | t-SNE | Diagnostic | Visualize whether embeddings cluster by identity, embryo, brightness, location, or cell-cycle state. Do not treat 2D t-SNE distance as production tracking geometry. | S | [JMLR](https://jmlr.org/papers/v9/vandermaaten08a.html) · [ResearchGate](https://www.researchgate.net/publication/228339739_Viualizing_data_using_t-SNE) | Diagnostic |
| C02 | openTSNE | Diagnostic | Scalable/controllable t-SNE with affinity APIs and insertion of new points; useful for million-cell audits. | A | [Docs](https://opentsne.readthedocs.io/en/stable/) | Review package license |
| C03 | FIt-SNE | Diagnostic | Fast interpolation-based t-SNE for large embedding banks. | B | [GitHub](https://github.com/KlugerLab/FIt-SNE) · [Nature Methods](https://www.nature.com/articles/s41592-018-0308-4) | Review |
| C04 | Parametric t-SNE | Adapt | Neural mapping that learns a reusable low-dimensional embedding; test only as auxiliary regularizer/diagnostic, not default linker geometry. | B | [GitHub](https://github.com/jsilter/parametric_tsne) | Review |
| C05 | UMAP | Diagnostic | Fast local-manifold audit; compare trajectory continuity and hard-neighbor separation. | A | [Docs](https://umap-learn.readthedocs.io/en/latest/) · [arXiv](https://arxiv.org/abs/1802.03426) | BSD-style package; verify |
| C06 | Parametric UMAP | Adapt | Trainable mapping can become an auxiliary latent regularizer or reusable diagnostic projection. | B | [Docs](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html) | Review |
| C07 | PHATE | Diagnostic | Designed to preserve transitions and has strong biological trajectory use; potentially excellent for visualizing developmental/mitotic state manifolds. | A | [GitHub](https://github.com/KrishnaswamyLab/PHATE) | Review |
| C08 | PaCMAP | Diagnostic | Balances local and global structure; useful sanity check when t-SNE clusters are misleading. | B | [GitHub](https://github.com/YingfanWang/PaCMAP) | Review |
| C09 | TriMAP | Diagnostic | Triplet-based dimensionality reduction with greater emphasis on global structure. | B | [GitHub](https://github.com/eamid/trimap) | Review |
| C10 | PCA | Diagnostic | First-line audit for embryo/location/brightness leakage; cheap and deterministic. | S | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | BSD library |
| C11 | Incremental PCA | Diagnostic | Large embedding-bank audit without loading every cell simultaneously. | C | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) | BSD library |
| C12 | Kernel PCA | Diagnostic | Nonlinear global audit without stochastic t-SNE behavior. | C | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) | BSD library |
| C13 | Isomap | Diagnostic | Geodesic manifold audit; can reveal curved developmental trajectories. | C | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html) | BSD library |
| C14 | Locally Linear Embedding | Diagnostic | Local neighborhood reconstruction diagnostic. | C | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html) | BSD library |
| C15 | Spectral Embedding / Laplacian Eigenmaps | Diagnostic | Audit graph-neighborhood structure independently of appearance-space Euclidean distance. | B | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html) | BSD library |
| C16 | Diffusion Maps | Diagnostic | Potentially useful for slow developmental state trajectories and transition structure. | B | [pydiffmap](https://github.com/DiffusionMapsAcademics/pyDiffMap) | Review |
| C17 | Autoencoder / β-VAE latent audit | Diagnostic | Test whether a compact generative latent preserves identity, morphology and mitotic state. | C | [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) | Method |
| C18 | Trustworthiness + kNN retention | Diagnostic | Quantify whether a 2D/low-d projection preserves local neighbors instead of trusting visual clusters. | S | [scikit-learn trustworthiness](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html) | Metric |

## D. Geometry / point clouds / equivariance

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| D01 | Physical-coordinate encoding | Direct | Always encode `(1.625z, 0.40625y, 0.40625x)` in µm; compare raw, normalized, Fourier and learned relative encodings. | S | [Metric source](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/README.md) | Internal |
| D02 | EGNN | Adapt | E(n)-equivariant message passing over cell-centroid graphs; cheap geometry-aware context. | S | [GitHub](https://github.com/vgsatorras/egnn) · [arXiv](https://arxiv.org/abs/2102.09844) | MIT code |
| D03 | e3nn | Adapt | E(3)-equivariant building blocks; useful if rotation-aware local morphology/geometry adds value. | A | [GitHub](https://github.com/e3nn/e3nn) · [arXiv](https://arxiv.org/abs/2207.09453) | Review |
| D04 | SE(3)-Transformer | Adapt | Equivariant attention over 3D point sets; potentially stronger but heavier than EGNN. | B | [GitHub](https://github.com/FabianFuchsML/se3-transformer-public) · [arXiv](https://arxiv.org/abs/2006.10503) | Review |
| D05 | Point Transformer V3 | Adapt | Large-context point-cloud reasoning with efficient serialized neighborhoods; test on centroid clouds. | A | [GitHub](https://github.com/Pointcept/PointTransformerV3) · [arXiv](https://arxiv.org/abs/2312.10035) | MIT repo |
| D06 | PointNet++ | Adapt | Hierarchical local point-set baseline; useful lower-compute control. | B | [arXiv](https://arxiv.org/abs/1706.02413) | Method |
| D07 | DGCNN / EdgeConv | Adapt | Dynamic local graphs can model neighborhood deformation and crowding. | A | [arXiv](https://arxiv.org/abs/1801.07829) | Method |
| D08 | KPConv | Adapt | Kernel point convolutions for irregular cell-centroid clouds. | B | [arXiv](https://arxiv.org/abs/1904.08889) | Method |
| D09 | PointNeXt | Adapt | Modernized PointNet++ family; useful efficiency/accuracy challenger. | B | [arXiv](https://arxiv.org/abs/2206.04670) | Review |
| D10 | MinkowskiEngine sparse 3D CNN | Adapt | Sparse voxel features around detected cells; potentially cheaper than dense 3D volumes. | B | [GitHub](https://github.com/NVIDIA/MinkowskiEngine) | Review |
| D11 | Radius graph | Direct | Candidate/context graph in physical distance; robust simple baseline and essential ablation. | S | [PyG radius_graph](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.radius_graph.html) | Library |
| D12 | KNN graph | Direct | Fixed-degree local context; compare to radius gating under varying density. | A | [PyG knn_graph](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.knn_graph.html) | Library |
| D13 | Fourier / sinusoidal relative position features | Direct | Represent fine and coarse displacement scales without forcing MLP to discover them. | A | [Fourier Features](https://arxiv.org/abs/2006.10739) | Method |
| D14 | Local Procrustes / Kabsch tissue alignment | Direct | Estimate local rigid/affine tissue motion from neighbors, then score residual displacement after alignment. | S | [Kabsch overview](https://en.wikipedia.org/wiki/Kabsch_algorithm) | Classical method |

## E. Motion / temporal dynamics

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| E01 | Constant-velocity features | Direct | Add velocity and acceleration residuals over tracklets; cheap and likely high-EV. | S | — | Internal |
| E02 | Kalman filter | Direct | Probabilistic motion prior and covariance gating; strong cheap specialist. | A | [FilterPy](https://github.com/rlabbe/filterpy) | MIT library |
| E03 | Unscented / extended Kalman filter | Adapt | Use when tissue/cell motion is mildly nonlinear. | B | [FilterPy](https://github.com/rlabbe/filterpy) | MIT library |
| E04 | Particle filter | Adapt | Multi-modal motion hypotheses when association is highly ambiguous; likely too expensive globally. | C | [FilterPy](https://github.com/rlabbe/filterpy) | MIT library |
| E05 | Local affine tissue-flow model | Direct | Fit neighborhood deformation between frames and score cells by residual-to-flow instead of absolute displacement. | S | — | Internal |
| E06 | FlowNet3D | Adapt | Learn scene flow directly between point clouds; potential displacement prior for cell-centroid sets. | B | [GitHub](https://github.com/xingyul/flownet3d) · [arXiv](https://arxiv.org/abs/1806.01411) | Review |
| E07 | PointPWC-Net | Adapt | Coarse-to-fine point-cloud scene flow, including self-supervised ideas. | B | [GitHub](https://github.com/DylanWusee/PointPWC) · [arXiv](https://arxiv.org/abs/1911.12408) | Review |
| E08 | RAFT optical-flow prior | Adapt | 2D/2.5D image-flow prior; use projections/slices or inspire iterative correlation-volume matching. | B | [GitHub](https://github.com/princeton-vl/RAFT) · [arXiv](https://arxiv.org/abs/2003.12039) | BSD-style code; verify weights |
| E09 | Neighbor-consensus motion | Direct | A link is stronger if nearby cells support a similar local displacement field. | S | — | Internal |
| E10 | Temporal convolution network | Adapt | Cheap multi-frame history encoder over tracklet features. | B | [TCN paper](https://arxiv.org/abs/1803.01271) | Method |
| E11 | Bidirectional temporal transformer | Direct | Encode tracklets over `t-k:t+k`; compare to Trackastra-style window. | S | [Trackastra](https://arxiv.org/abs/2405.15700) | Method |
| E12 | Mamba / state-space temporal model | Adapt | Longer temporal horizon with linear-ish sequence scaling. | B | [Mamba](https://arxiv.org/abs/2312.00752) | Method |
| E13 | V-JEPA-style latent future prediction | Adapt | Predict future cell/local-volume latent; use prediction error as association evidence. | S | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Method |
| E14 | Kalman / RTS smoother for post-hoc tracklets | Direct | Smooth trajectories after graph construction and flag impossible kinks for local repair. | B | [FilterPy](https://github.com/rlabbe/filterpy) | MIT library |

## F. Association / matching / linking

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| F01 | SimpleNodeTransformer | Direct | Competition-native adjacent-frame cross-attention baseline; essential control. | S | [Biohub repo](https://github.com/royerlab/kaggle-cell-tracking-competition) | BSD-3 |
| F02 | Trackastra | Direct | Temporal-window transformer over detections, explicitly division-aware. | S | [GitHub](https://github.com/weigertlab/trackastra) · [arXiv](https://arxiv.org/abs/2405.15700) | REVIEW license/weights |
| F03 | HOCT | Direct | Edge-centric higher-order attention under 3D geometric prior; extremely aligned with candidate-edge Biohub graph. | S | [GitHub](https://github.com/royerlab/hoct) · [arXiv](https://arxiv.org/abs/2607.11754) | MIT code; verify weights |
| F04 | SuperGlue-style graph matching | Adapt | Contextual feature matching + differentiable optimal transport; natural primitive for frame-to-frame cell matching. | A | [GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork) · [arXiv](https://arxiv.org/abs/1911.11763) | Review |
| F05 | LightGlue-style adaptive matcher | Adapt | Adaptive depth/width based on matching difficulty; excellent conceptual template for a 12h uncertainty cascade. | S | [GitHub](https://github.com/cvg/LightGlue) · [arXiv](https://arxiv.org/abs/2306.13643) | Apache-2.0 repo; verify |
| F06 | LoFTR-style cross-attention correspondence | Adapt | Dense/coarse-to-fine correspondence principles can inspire candidate refinement. | B | [GitHub](https://github.com/zju3dv/LoFTR) · [arXiv](https://arxiv.org/abs/2104.00680) | Review |
| F07 | Sinkhorn optimal transport | Direct | Soft doubly-stochastic assignment prior; use before hard graph optimization or as differentiable training surrogate. | A | [POT](https://github.com/PythonOT/POT) | BSD library |
| F08 | Gumbel-Sinkhorn | Adapt | Differentiable approximate permutations for end-to-end assignment training. | B | [arXiv](https://arxiv.org/abs/1802.08665) | Method |
| F09 | Hungarian / linear assignment | Direct | Classical high-quality baseline; useful for isolating value of learned context. | S | [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | BSD library |
| F10 | LapTrack | Direct | Linear-assignment cell tracking with custom costs; excellent classical challenger and parameter-search baseline. | A | [GitHub](https://github.com/yfukai/laptrack) · [Paper](https://academic.oup.com/bioinformatics/article/39/1/btac799/6887138) | Review |
| F11 | DeepCell tracking | Adapt | Learned assignment + Hungarian lineage assembly; useful architecture comparison. | B | [GitHub](https://github.com/vanvalenlab/deepcell-tracking) | Review |
| F12 | btrack | Direct | Bayesian motion + multiple hypothesis/global optimization; strong non-transformer comparison. | A | [GitHub](https://github.com/quantumjot/btrack) | MIT/BSD-style; verify |
| F13 | Bi-encoder retrieval + cross-encoder reranker | Direct | Fast ANN candidate retrieval from identity embeddings, then expensive pair/higher-order scoring only on top-k. | S | [FAISS](https://github.com/facebookresearch/faiss) | Architecture pattern |
| F14 | DeepSORT-style appearance + motion fusion | Adapt | Fuse learned identity embedding with motion covariance; natural tracking primitive despite natural-image origin. | B | [arXiv](https://arxiv.org/abs/1703.07402) | Method |
| F15 | ByteTrack two-threshold association | Adapt | Use high-confidence detections first then recover lower-confidence candidates through existing track context. | A | [GitHub](https://github.com/ifzhang/ByteTrack) · [arXiv](https://arxiv.org/abs/2110.06864) | MIT repo |
| F16 | OC-SORT observation-centric association | Adapt | Robust association after occlusion/missed detections; ideas useful for rescue links and short gaps. | B | [GitHub](https://github.com/noahcao/OC_SORT) · [arXiv](https://arxiv.org/abs/2203.14360) | Review |

## G. Division / higher-order lineage structure

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| G01 | ForkHead triplet scorer | Direct | Baseline idea: score `(parent,d1,d2)` jointly instead of multiplying two pair probabilities. | S | [Biohub repo](https://github.com/royerlab/kaggle-cell-tracking-competition) | Competition-native |
| G02 | HOCT edge-centric division reasoning | Direct | Model interactions among candidate links so division branches are represented explicitly. | S | [HOCT](https://arxiv.org/abs/2607.11754) | MIT code |
| G03 | DeepSets | Adapt | Permutation-invariant scoring for unordered daughter sets. | A | [arXiv](https://arxiv.org/abs/1703.06114) | Method |
| G04 | Set Transformer | Adapt | Attention over parent + candidate daughter set; captures interactions beyond pair scores. | A | [arXiv](https://arxiv.org/abs/1810.00825) | Method |
| G05 | Hypergraph neural network | Adapt | Represent a division as a hyperedge connecting one parent to two daughters. | A | [arXiv](https://arxiv.org/abs/1809.09401) | Method |
| G06 | Higher-order factor graph | Adapt | Encode pair edges, forks, births/deaths and local topology as separate factors. | A | [pgmpy](https://github.com/pgmpy/pgmpy) | Library/inspiration |
| G07 | Division temporal-window classifier | Direct | Classify a local `t-2:t+2` volume/tracklet rather than exact single-frame mitosis. | S | — | Internal |
| G08 | Daughter symmetry / midpoint geometry | Direct | Feature: daughter midpoint vs predicted parent trajectory; daughter separation/orientation; local density. | S | — | Internal |
| G09 | Intensity / volume conservation prior | Adapt | Test whether parent signal approximately redistributes into two daughters; use only if empirically stable. | B | — | Internal |
| G10 | Explicit lineage-state machine | Direct | States normal / pre-division / post-division / birth / death; constrain improbable rapid repeated splits. | B | — | Internal |

## H. Global graph optimization

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| H01 | motile ILP | Direct | Purpose-built multi-object tracking global optimization; natural replacement/benchmark for custom ILP. | S | [GitHub](https://github.com/funkelab/motile) · [Tracker](https://github.com/funkelab/motile_tracker) | MIT/BSD tools |
| H02 | Ultrack global optimization | Direct | Jointly choose segmentation hypotheses and temporal associations under biological constraints. | S | [GitHub](https://github.com/royerlab/ultrack) · [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0) | BSD-3 |
| H03 | SCIP / PySCIPOpt | Direct | Open-source MIP solver option for ILP lineage formulation. | A | [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) | Apache-2.0 SCIP ecosystem; verify packaging |
| H04 | OR-Tools min-cost flow | Direct | Fast graph-flow formulation for mostly one-to-one tracking; divisions need augmentation. | A | [GitHub](https://github.com/google/or-tools) | Apache-2.0 |
| H05 | OR-Tools CP-SAT | Direct | Flexible discrete constraints for divisions, births/deaths and mutual exclusion. | A | [GitHub](https://github.com/google/or-tools) | Apache-2.0 |
| H06 | Gurobi ILP | Direct | Potentially very fast solver in research; final Kaggle availability/license must be guaranteed. | B | [Gurobi](https://www.gurobi.com/) | REVIEW runtime/license |
| H07 | Min-cost circulation | Adapt | Classical global tracking objective; useful if lineage constraints can be encoded efficiently. | B | [OR-Tools](https://github.com/google/or-tools) | Apache-2.0 |
| H08 | Viterbi / dynamic programming | Adapt | Efficient tracklet-level decoding when candidate branching is bounded. | C | — | Classical |
| H09 | Belief propagation / factor graph | Adapt | Approximate global inference with higher-order factors; useful when exact ILP becomes too expensive. | B | [pgmpy](https://github.com/pgmpy/pgmpy) | Review |
| H10 | Lagrangian relaxation / dual decomposition | Adapt | Decompose large graph optimization into easier local problems and coordinate constraints. | C | — | Research primitive |
| H11 | Local graph repair search | Direct | After global solve, inspect low-margin neighborhoods and attempt score-improving swaps/fork repairs. | S | — | Internal |
| H12 | Tracklet stitching | Direct | First solve confident short tracklets, then globally stitch tracklets using richer long-context features. | S | — | Internal |

## I. Training / augmentation / data engineering

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| I01 | TorchIO | Infra | Physically aware 3D transforms, patch sampling and medical-volume augmentation. | S | [GitHub](https://github.com/TorchIO-project/torchio) · [arXiv](https://arxiv.org/abs/2003.04696) | Apache-2.0 |
| I02 | MONAI | Infra | 3D medical imaging models/losses/transforms; useful training backbone. | S | [GitHub](https://github.com/Project-MONAI/MONAI) | Apache-2.0 |
| I03 | Albumentations volumetric transforms | Infra | Fast augmentation library; verify true 3D behavior for each transform rather than assuming 2D transforms generalize. | A | [GitHub](https://github.com/albumentations-team/albumentations) · [Docs](https://albumentations.ai/docs/3-basic-usage/volumetric-augmentation/) | Review current license/version |
| I04 | Elastic 3D deformation | Direct | Simulate tissue deformation; critical for ASCENT-style invariance. | S | [TorchIO](https://github.com/TorchIO-project/torchio) | Transform |
| I05 | Anisotropic affine augmentation | Direct | Rotate/scale/translate in physical coordinates while respecting z/xy anisotropy. | S | [TorchIO](https://github.com/TorchIO-project/torchio) | Transform |
| I06 | Intensity / gamma / contrast jitter | Direct | Prevent embeddings/detector from using fragile absolute brightness cues. | S | [TorchIO](https://github.com/TorchIO-project/torchio) | Transform |
| I07 | Gaussian + Poisson noise | Direct | Approximate microscopy noise families; tune from empirical train statistics. | A | [MONAI](https://github.com/Project-MONAI/MONAI) | Transform |
| I08 | PSF / blur augmentation | Direct | Simulate defocus and resolution variation; particularly useful along z. | A | — | Internal/physics |
| I09 | z-slice dropout / missing-slice augmentation | Direct | Robustness to slice artifacts and anisotropic sampling. | B | — | Internal |
| I10 | Hard-negative mining | Direct | Continuously mine nearest wrong neighbors / competing links from current model. | S | — | Training strategy |
| I11 | Mean Teacher | Adapt | EMA teacher for semi-supervised pseudo-label stability. | A | [arXiv](https://arxiv.org/abs/1703.01780) | Method |
| I12 | FixMatch-style consistency | Adapt | Confidence-filtered pseudo labels + strong augmentation for unlabeled cells/edges. | A | [arXiv](https://arxiv.org/abs/2001.07685) | Method |
| I13 | Pseudo-label consensus across trackers | Direct | Only promote pseudo edges agreed by geometry + identity + temporal model; reduce confirmation bias. | S | — | Internal |
| I14 | Curriculum: easy -> hard association | Direct | Train pair scorer on clear links first, then aggressively mine crowded/division negatives. | A | — | Internal |
| I15 | Focal loss | Direct | Focus detector/linker on difficult positives/negatives; tune carefully under sparse labels. | B | [arXiv](https://arxiv.org/abs/1708.02002) | Method |
| I16 | Tversky / Dice family | Direct | Useful for center/foreground imbalance in segmentation auxiliary heads. | B | [arXiv](https://arxiv.org/abs/1706.05721) | Method |

## J. Uncertainty / calibration / ensembling

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| J01 | Deep ensembles | Direct | Independent seeds/backbones improve robustness and provide disagreement signal. | S | [arXiv](https://arxiv.org/abs/1612.01474) | Method |
| J02 | Ensemble disagreement routing | Direct | Route only links/divisions with model disagreement to expensive higher-order specialist. | S | — | Internal |
| J03 | Probability margin routing | Direct | Use `p_best - p_second` as a simple ambiguity trigger. | S | — | Internal |
| J04 | Entropy routing | Direct | High entropy over candidate successors triggers extra context. | A | — | Internal |
| J05 | Temperature scaling | Direct | Calibrate edge/fork probabilities before global solver and cost fusion. | S | [arXiv](https://arxiv.org/abs/1706.04599) | Method |
| J06 | Isotonic regression | Direct | Non-parametric calibration when logit-to-probability relationship is nonlinear; fit out-of-fold only. | B | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) | BSD library |
| J07 | Out-of-fold stacking | Direct | Train a small meta-calibrator on predictions from models that never saw that validation embryo. | S | — | Internal |
| J08 | Monte-Carlo dropout | Adapt | Approximate epistemic uncertainty; probably slower than ensembles for final system but useful diagnostic. | C | [Paper](https://arxiv.org/abs/1506.02142) | Method |
| J09 | Test-time augmentation | Direct | Average physically valid flips/rotations where biology/imaging makes them invariant; audit runtime. | A | — | Internal |
| J10 | Regime-specific calibration | Direct | Separate calibration for crowded, division, high-motion and normal neighborhoods. | S | — | Internal |

## K. Search / runtime / deployment engineering

| ID | Primitive | Mode | Biohub hypothesis / experiment | Tier | Links | Provenance |
|---|---|---|---|:---:|---|---|
| K01 | Optuna | Infra | Bayesian/ASHA-style hyperparameter search for thresholds, losses, window sizes and solver costs. | S | [GitHub](https://github.com/optuna/optuna) | MIT |
| K02 | Ray Tune | Infra | Distributed experiment scheduling if many GPUs/nodes are available. | B | [GitHub](https://github.com/ray-project/ray) | Apache-2.0 |
| K03 | FAISS | Direct | Fast nearest-neighbor retrieval for identity embeddings and candidate pruning. | S | [GitHub](https://github.com/facebookresearch/faiss) | MIT/BSD-style; verify |
| K04 | HNSW | Direct | Alternative ANN index for embedding retrieval. | B | [hnswlib](https://github.com/nmslib/hnswlib) | Apache-2.0 |
| K05 | torch.compile | Infra | Compile stable inference/training sections after correctness is locked. | A | [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.compile.html) | PyTorch license |
| K06 | AMP / FP16 / BF16 | Infra | Reduce memory and increase throughput; numerical audit required for solver costs and tiny logits. | S | [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) | PyTorch |
| K07 | FlashAttention / SDPA | Infra | Speed long-window attention if architecture/runtime support it. | A | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | BSD-style; verify |
| K08 | Chunked / tiled graph inference | Direct | Bound memory by processing spatial/temporal subgraphs with overlap and reconciliation. | S | — | Internal |
| K09 | Knowledge distillation | Direct | Train heavy ensemble/teacher offline; distill to final Kaggle student if runtime is limiting. | S | [Distillation paper](https://arxiv.org/abs/1503.02531) | Method |
| K10 | Checkpoint/model artifact manifest | Infra | Store SHA256, source URL, commit, license, data provenance and exact preprocessing with every weight. | S | — | Internal governance |


---

# 6. A practical champion–challenger board

This is the **starting** board I would use. It is intentionally smaller than the registry.

| Slot | Champion to establish | First challengers | What earns promotion |
|---|---|---|---|
| Proposals | converged TemporalUNet3D | FOCUS-3D; StarDist3D; nnU-Net/MedNeXt; LoG rescue | +official score, especially division-frame node recall |
| Hypothesis selection | merged center bank + simple NMS | Ultrack-style multi-hypothesis selection | fewer missed/duplicate nodes without node-count penalty |
| Identity | 3D hard-negative contrastive encoder | ASCENT retrain; VICReg; DINO/JEPA variants | higher true-successor rank among local competitors |
| Geometry | physical relative positions | EGNN; DGCNN; PTv3 | hard-crowding edge uplift at low cost |
| Motion | velocity + local affine field | Kalman; PointPWC/FlowNet3D-inspired scene flow | successor-in-gate recall and edge metric |
| Pair association | SimpleNodeTransformer | SuperGlue/Sinkhorn; LightGlue-style adaptive matcher | pairwise edge rank + end-to-end score |
| Higher order | ForkHead | HOCT; Trackastra temporal window | division + ambiguous-edge uplift |
| Solver | current ILP-style constraints | motile; OR-Tools; Ultrack-style selection | global edge Jaccard at acceptable runtime |
| Uncertainty | logit margin | ensemble disagreement; entropy; regime calibrator | error-detection AUROC and compute savings |
| Final calibration | manual sweep | Optuna/OOF meta-calibration | robust official-score uplift |

---

# 7. How to maximize offline training

## Phase 0 — dataset audit and deterministic evaluation

1. Freeze the exact official metric implementation in the research repo.
2. Build group CV by dataset/embryo/direction.
3. Cache per-frame statistics: cell density, intensity distribution, estimated motion, division proximity.
4. Create immutable train/validation manifests and hash them.
5. Reproduce a simple baseline before adding specialists.

## Phase 1 — detector/proposal training

- Train the competition-native temporal detector **to convergence**; the public repo explicitly says its example was not trained to convergence.
- Run at least 3 seeds for the final detector shortlist, not for every early experiment.
- Sample hard frames disproportionately: crowded regions and windows around known divisions.
- Train center heatmap plus optional boundary/distance auxiliary heads.
- Calibrate proposal threshold against **end-to-end** official score, not only detector F1.
- Generate multiple thresholds/NMS radii as proposal hypotheses; do not immediately throw low-confidence cells away.

## Phase 2 — unlabeled identity pretraining

Construct millions of cell-centered 3D crops from candidate detections. The training curriculum should contain:

- two augmented views of the same crop;
- pseudo-temporal positives from very high-confidence links;
- deformation-simulated positives;
- spatially nearby cells as hard negatives;
- same-embryo/different-time negatives that look extremely similar;
- parent/daughter-aware samples so mitosis does not look like arbitrary identity destruction.

A useful objective family:

```text
L_identity = L_hard_InfoNCE
           + 0.1..1.0 * L_temporal_cycle
           + λ * L_VICReg_or_covariance
           + μ * L_JEPA_prediction
```

Do not blindly combine all four. Run a factorial shortlist, then promote only clear winners.

## Phase 3 — pair linker

For every source node, construct a high-recall physical-radius candidate set. Train with:

- edge BCE / focal loss;
- listwise or pairwise ranking loss so the true successor outranks nearby alternatives;
- in-batch + mined hard negatives;
- detector confidence and identity embedding as inputs;
- physical relative displacement;
- local density and tissue-flow residual;
- separate birth/death/no-link token.

Track **candidate edge recall**. If the true edge is absent from the candidate set, no transformer can recover it.

## Phase 4 — higher-order / temporal specialist

Train only on hard neighborhoods at first:

- low pairwise margin;
- candidate forks;
- high local density;
- high motion;
- disagreement among seed models.

Compare:

1. Trackastra-style temporal-window transformer;
2. HOCT edge-centric model;
3. EGNN/PTv3 local point-cloud context;
4. hybrid: identity features + physical motion + HOCT.

## Phase 5 — division specialist

Construct explicit `(parent, d1, d2)` examples and near-miss hard negatives. Include windows around the split so the model is not forced to identify one exact mitotic frame. Record whether the final official division Jaccard improves **without sacrificing edge Jaccard**.

## Phase 6 — global solver + calibration

Fit solver costs only using out-of-fold predictions. Search:

- node threshold / node count target;
- NMS/merge radius;
- link radius;
- edge-cost scaling;
- birth/death costs;
- division prior;
- fork veto/confirmation thresholds;
- gap-recovery thresholds;
- uncertainty-routing threshold.

Use Optuna/Bayesian search for the coarse region, then a dense local grid around the best robust region.

## Phase 7 — ensemble and distillation

A strong final ensemble could include **diverse error families**, not merely more seeds:

```text
Detector A: TemporalUNet3D
Detector B: FOCUS-3D-derived proposals
Detector C: StarDist3D / independent center model

Linker A: pairwise cross-attention
Linker B: HOCT
Linker C: Trackastra-window / geometry-heavy model
```

Then either:

- fuse logits/graphs out-of-fold; or
- distill the expensive ensemble into a fast student for Kaggle inference.

---

# 8. t-SNE and manifold methods: how to use them correctly

`t-SNE` is valuable here, but mostly as a **microscope for the embedding space**.

### Good uses

Color t-SNE/UMAP/PHATE points by:

- true/pseudo track ID;
- frame;
- embryo/dataset;
- physical xyz region;
- cell brightness;
- local density;
- pre/post division state;
- model error type.

You want to detect bad shortcuts such as:

- embeddings clustering primarily by **position** rather than identity;
- embeddings clustering by **brightness/SNR**;
- same-track points jumping across the manifold at division;
- neighboring confusing cells collapsing together.

### Quantify the visualization

Do not judge a pretty plot. Report:

```text
same-track retrieval@1 / @5
true-successor rank
hard-negative margin
kNN label purity
embedding temporal smoothness
trustworthiness / continuity of projected manifold
linear probe for embryo/location/brightness leakage
```

### Production warning

A 2D t-SNE embedding intentionally distorts global geometry and is stochastic. Do not use ordinary 2D t-SNE distance directly as the final tracking cost unless a controlled experiment proves it. If neighborhood-preservation is useful, transfer the **objective idea** into a learnable metric space or try parametric t-SNE/UMAP as a carefully validated auxiliary loss.

---

# 9. Multi-hypothesis proposal bank — concrete design

For each frame:

```text
TemporalUNet centers  ─┐
FOCUS-3D centers      ─┼─> physical-space clustering / proposal union
StarDist3D centers    ─┤
classical rescue      ─┘
```

For each proposal retain:

```text
(z,y,x) in voxel coordinates
(z,y,x) in µm
source detector bitmask
per-source confidence
local intensity statistics
mask volume / shape if available
TemporalUNet feature
SSL identity embedding
local density
proposal disagreement
```

Instead of deleting duplicates immediately, form **mutual-exclusion groups** of nearby hypotheses and let a global selector choose one. This is the primitive most directly inspired by Ultrack and could attack the detector bottleneck without requiring one detector to be perfect.

---

# 10. Runtime architecture for a 12-hour Kaggle notebook

The final notebook should be boring and deterministic:

```text
load frozen weights
    ↓
read test zarr
    ↓
proposal ensemble / candidate merge
    ↓
identity embeddings (batched)
    ↓
cheap candidate link scoring
    ↓
uncertainty routing
    ├─ confident -> keep cheap result
    └─ ambiguous -> HOCT / temporal specialist
    ↓
division scoring
    ↓
global solver
    ↓
metric-aware graph repair
    ↓
submission.csv
```

### Runtime levers

- mixed precision for neural inference;
- prefetch/cached chunks from zarr;
- sparse candidate graphs;
- ANN retrieval instead of all-pairs when node counts are large;
- hard-case routing;
- chunked temporal/spatial attention;
- compile only after shapes/control flow are stable;
- distill heavy models if they are too expensive;
- keep solver candidate graph aggressively sparse **after** proving high recall.

---

# 11. Experiment ledger schema

Keep every experiment machine-readable. Example:

```yaml
experiment_id: B01_hardnce_t2_seed3
slot: identity
primitive: ASCENT-inspired-NETr
hypothesis: nearby-cell hard negatives improve ambiguous link ranking
source:
  paper: https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1
  code: https://github.com/lu-lab/ascent
license_review: passed|blocked|pending
data:
  train_manifest_sha256: ...
  external_datasets: []
split:
  scheme: leave-one-embryo-out
  fold: 2
training:
  epochs: 120
  seed: 3
  crop_um: [..]
  augmentations: [elastic3d, gamma, psf_blur]
  loss: hard_infonce+temporal_cycle
checkpoint:
  sha256: ...
metrics:
  node_recall_7um: ...
  candidate_edge_recall: ...
  edge_jaccard: ...
  adjusted_edge_jaccard: ...
  division_jaccard: ...
  official_score: ...
  hard_crowded_score: ...
  hard_division_score: ...
  runtime_minutes: ...
  peak_vram_gb: ...
comparison:
  champion_experiment_id: ...
  delta_official_score: ...
  seeds: [...]
promotion: rejected|challenger|champion
notes: ...
```

---

# 12. Provenance and licensing gate

**This is not legal advice.** For a cash-prize competition, “publicly downloadable” does not automatically mean “safe to use.” Record four separate things:

1. **code license**;
2. **checkpoint/weight license**;
3. **training-data license/provenance**;
4. **competition rules on external data/models**.

### Known flags worth treating seriously

- **Cellpose:** the repository explicitly states that all Cellpose models are trained on data licensed **CC-BY-NC**. Treat pretrained deployment in a prize competition as **blocked/pending review**, even if the research primitive is useful.
- **Cell-DINO:** the repository states that Cell-DINO code is **CC-BY-NC** and model weights use the **FAIR Non-Commercial Research License**. Treat the pretrained implementation/weights as **red**; a clean reimplementation of the general method on permitted data is a separate question.
- **DINOv3:** uses a custom DINOv3 license rather than the old Apache-2.0 DINOv2 model card; review the exact use/redistribution terms before packaging.
- **CellSAM:** code is Apache-2.0, but the repository notes Cellpose data in its evaluation dataset; audit the exact checkpoint training provenance rather than inferring safety from code license.
- **ASCENT / FOCUS-3D / HOCT / Trackastra / Ultrack:** code and papers may be permissive/open, but still record the exact **weight** and **training-data** terms independently.

Recommended artifact manifest:

```csv
artifact_name,sha256,source_url,git_commit,code_license,weight_license,training_data,competition_ok,reviewer,date
```

---

# 13. What I would *not* do

1. **Do not implement 150 primitives simultaneously.** That destroys causal attribution and creates an impossible debugging surface.
2. **Do not tune primarily against the public leaderboard.** It is a sparse and noisy optimization oracle and invites overfitting.
3. **Do not use t-SNE plots as proof of tracking quality.** Quantify retrieval/ranking and end-to-end score.
4. **Do not use segmentation IoU as the main detector objective.** The metric cares about centroids/graph edges.
5. **Do not make the final Kaggle notebook a training notebook.** Train and search offline; package frozen inference.
6. **Do not assume bigger transformer = better.** Geometry, motion and candidate recall may be higher-EV than capacity.
7. **Do not let a low-confidence detector delete a plausible cell too early.** Preserve hypotheses until evidence accumulates.
8. **Do not promote `+0.001` without robustness checks.** Require cross-fold/seed evidence.

---

# 14. Recommended 4-week research order

## Wave 1 — establish ceilings

- train TemporalUNet3D properly;
- exact local metric harness;
- node recall / candidate edge recall / hard-case reports;
- FOCUS-3D and StarDist proposal audit;
- simple multi-hypothesis union.

## Wave 2 — identity and geometry

- ASCENT-style 3D crop encoder;
- hard-negative InfoNCE;
- t-SNE + UMAP + PHATE diagnostics;
- physical coordinate features;
- local affine tissue motion;
- EGNN challenger.

## Wave 3 — association and division

- converged SimpleNodeTransformer control;
- HOCT challenger;
- Trackastra-style longer window;
- fork/hyperedge division specialist;
- OOF calibrated ensemble.

## Wave 4 — global score and runtime

- motile/ILP calibration;
- uncertainty cascade;
- multi-hypothesis node selection;
- solver hyperparameter search;
- TTA/seed/model ensemble;
- distillation/runtime optimization;
- frozen inference notebook rehearsal under a strict 12-hour wall clock.

---

# 15. Stop conditions / decision discipline

A primitive should be killed quickly when:

- candidate recall does not improve;
- end-to-end score is neutral across two or more robust splits;
- benefit comes entirely from one sequence;
- it adds substantial inference cost for <~0.001 robust gain;
- its best effect is already subsumed by another specialist;
- provenance cannot be cleared;
- it makes the solver graph combinatorially explode.

Conversely, a primitive deserves deep investment when it:

- improves the **ceiling** (node/candidate recall);
- disproportionately improves crowded/division frames;
- changes error type rather than merely duplicating another model;
- provides useful uncertainty/disagreement for an ensemble;
- is cheap enough to run everywhere or valuable enough to route selectively.

---

# 16. Final target stack to investigate

The research hypothesis I would currently place the highest probability on is:

```text
Converged TemporalUNet3D
       + FOCUS-3D / StarDist proposal diversity
                   ↓
        multi-hypothesis node bank
                   ↓
   ASCENT-inspired 3D identity encoder
     + hard-negative / temporal SSL
                   ↓
 physical geometry + local tissue-flow features
                   ↓
 cheap pairwise cross-attention / retrieval scorer
                   ↓
        uncertainty-based routing
                   ↓
 HOCT / Trackastra-window specialist on hard subgraphs
                   ↓
      explicit division hyperedge scorer
                   ↓
         motile/ILP global optimizer
                   ↓
 out-of-fold probability + metric-aware calibration
                   ↓
           frozen Kaggle inference
```

The key idea is **diversity of inductive biases**. A detector ensemble should not be five copies of the same U-Net. A linker ensemble should not be five seeds of the same transformer. The strongest ensemble is likely to mix:

- appearance;
- identity;
- physical geometry;
- motion;
- higher-order topology;
- global constraints.

That is much more likely to create genuinely complementary errors and therefore a meaningful leaderboard jump.

---

# 17. Source-quality policy

For every new primitive added to this dossier, prefer sources in this order:

1. original GitHub/project repository;
2. arXiv/bioRxiv/source-of-record paper;
3. peer-reviewed publisher page;
4. project documentation/blog by the authors;
5. ResearchGate mirror only as a supplementary discovery/mirror source;
6. third-party blogs only for intuition — never for license, benchmark or implementation-critical claims.

### Useful ecosystem portals

- [Cell Tracking Challenge](https://celltrackingchallenge.net/)
- [Latest Cell Linking Benchmark results](https://celltrackingchallenge.net/latest-clb-results/)
- [Latest Cell Tracking Benchmark results](https://celltrackingchallenge.net/latest-ctb-results/)
- [napari hub](https://www.napari-hub.org/)
- [MONAI](https://project-monai.github.io/)
- [BioImage.IO](https://bioimage.io/)
- [Papers With Code](https://paperswithcode.com/)

---

## Bottom line

The competitive advantage is not “we know more model names.” It is building a **reproducible research market** in which every specialist has multiple candidates, every candidate is scored against the real bottleneck it claims to solve, and only complementary winners survive into the final inference graph.

If this process is run rigorously, FOCUS-3D and t-SNE become two entries in a much larger system:

- FOCUS-3D = one proposal specialist;
- t-SNE = one representation-audit primitive;
- ASCENT = one identity-learning family;
- HOCT = one higher-order association family;
- Ultrack/motile = one global-selection family;
- EGNN/PTv3 = geometry families;
- JEPA = temporal/latent-prediction family;
- SuperGlue/LightGlue/Sinkhorn = matching families;
- ensembles/calibration = uncertainty families.

The leaderboard system should be the **intersection of the best validated primitives**, not the union of every clever paper.
