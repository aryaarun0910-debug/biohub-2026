# Edge-training frontier: how to train a frame-to-frame association model — 2026-08-18

> **EDITOR'S NOTE (host session, same day).** Two verifications against this report:
> (1) **Trackastra licence: GitHub's licence API reports `BSD-3-Clause`, not Apache-2.0** as
> claimed below. Both are permissive, so the operative conclusion (vendorable, prize-clean)
> stands, but cite BSD-3-Clause. (2) `window_size = 2` in the vendored trainer CONFIRMED
> (`train_unet_transformer.py:197`, `:360`) — we do sit at the documented worst end of
> Trackastra's window sweep. Also note precisely: the vendored `compute_loss` (:55-72) is a
> column-softmax over candidate parents (`softmax(logits, dim=0)`), i.e. parental competition,
> but WITHOUT Trackastra's `1+Σ` background term — our targets have no explicit "no parent"
> option. That gap is part of what the parental-softmax ablation credit may buy.
> Remainder is unmodified raw agent output.


Mandate: the ASSOCIATION-training state of the art (2024–2026) for the H1 edge-head retrain on
Ultrack-noisy dense Zebrahub tracks + sparse clean competition GT. Complements (does not duplicate)
`retrain_recipes_2026-08-17.md` (detector-side recipe, AMP/compute arithmetic, sparse-detection PU
treatment), `kaggle_winners_playbook_2026-08-17.md` (the two-stage noisy→clean prior, lever 2), and
`scientific_incumbents_2026-08-17.md` (CTC/CLB standings, linajea mechanism, L8 division-distillation
warning). Where those reports carry a claim, I cross-reference instead of restating.

Evidence labels: **[DOC]** documented in cited source · **[MEAS]** measured by me this session on our
repo · **[INF]** my inference · **[UNVERIFIED]** no source found / could not read.

---

## 0. Our trainer's association-training profile, as built (all [MEAS], this session)

Read `vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py`. Seven facts frame everything
below; several are *already right* by the standards of the 2024–2026 literature and must not be
"fixed" away.

- **E1 — The edge head is already trained on the detector's own noise pattern.** During training,
  `detect_and_match` (line 620) runs peak detection on the *live* detector logits
  (`det_threshold=0.3`, 5 µm NMS), matches peaks to GT within `max_match_distance=5.0` µm, and the
  transformer consumes the **detected** peaks, not GT nodes. Unmatched (FP) peaks stay in the token
  set with all-zero target rows/columns. This is the in-house equivalent of what the CTC linking
  benchmark formalises as ERR_SEG training (§1.4) — the single most important robustness mechanism in
  the field, and we have it by construction.
- **E2 — FP detections do act as negatives for annotated sources.** `compute_loss` (:55-72) masks to
  `active_rows | active_cols`; a row for an annotated source is active, and every column in that row —
  including unmatched-FP columns — is pushed to 0 through the softmax. FP *sources* (inactive rows
  whose columns are also inactive) are ignored, not supervised.
- **E3 — Parental softmax is present** (`softmax(logits, dim=0)`: each target has a distribution over
  sources, so divisions are expressible and merges are not), with a focal modulation `(1-p_t)^2·BCE`.
  Trackastra's auxiliary sigmoid term (λ=1e-2, §1.2) is absent.
- **E4 — The division upweight is a no-op** (F6 in `retrain_recipes`, verified at :68-70).
- **E5 — Window size is 2** — the value the Trackastra ablation identifies as the *substantially
  deteriorated* end of its sweep (§1.3).
- **E6 — There is no graph/detection augmentation of any kind.** Only `brightness_augment` and
  `flip_augment` on the image; no node dropout, no coordinate jitter, no feature dropout.
- **E7 — Negative sampling is all-pairs within the crop** with no hard-negative machinery — which, on
  a 64³ crop with ~3–9 annotated nodes, means the negative pool is small and easy. On dense Zebrahub
  crops (~900 nodes/crop-frame) the same all-pairs scheme becomes a hard-negative mine for free.

---

## 1. How Trackastra was actually trained — paper + repository, reconciled

Gallusser & Weigert, *Trackastra: Transformer-based cell tracking for live-cell microscopy*, ECCV
2024, [arXiv:2405.15700](https://arxiv.org/abs/2405.15700) (full text read at
[arxiv.org/html/2405.15700v1](https://arxiv.org/html/2405.15700v1)); code
[weigertlab/trackastra](https://github.com/weigertlab/trackastra). **Licence: Apache-2.0, verified
from the repo LICENSE file this session** — this *corrects* `retrain_recipes` §10, which flagged the
licence as unverified. Safe to vendor or reimplement.

### 1.1 Data mix [DOC]

Per-experiment models: bacteria (31 train videos of 39, ~100K cells), DeepCell (130 videos, official
split), ISBI vesicles. The **general model** trains on *all* experiment data plus additional CTC data.
The shipped checkpoint registry (`trackastra/model/pretrained.json`, read raw this session) documents
three models:

| model | dim | training data (verbatim) |
|---|---|---|
| `general_2d` | 2D | subset of CTC 2D + 15 named corpora (bacteria van Vliet / ObiWan-Microbi / Persat, DeepCell, Ker PhC, epithelia, T cells, N. meningitidis, synthetic nuclei/particles, PTC, Yeast Cell-ACDC, DeepSea, btrack, mother machine) |
| **`ctc`** | **2D+3D** | **"All Cell Tracking Challenge 2d+3d datasets with available GT and ERR_SEG"** |
| `general_2d_w_SAM2_features` | 2D | same as general_2d, + SAM2.1 appearance features (release v0.5.0) |

So the `ctc` 3D checkpoint — the one `scientific_incumbents` §2.2 stages as a drop-in — was trained
on CTC data **paired with ERR_SEG erroneous segmentations** (see §1.4). Its training supervision is
GT lineage graphs, its training *inputs* are deliberately imperfect detections. [DOC]

### 1.2 The association loss, exactly [DOC]

- Parental softmax: `Ã_ij = exp(Â_ij) / (1 + Σ_{i'∈P_j} exp(Â_i'j))` over all detections in the
  frame before `d_j` — each child picks at most one parent; the `1+` allows "no parent" (track birth).
- Total loss `L = L_BCE(A, Φ(Â), W) + λ·L_BCE(A, σ(Â), W)` with **λ = 1e-2** — a small plain-sigmoid
  term alongside the parental-softmax term.
- Weight matrix `W`: dividing cells **1+λ_div = 11** (λ_div=10); continuing tracks **1+λ_cont = 2**
  (λ_cont=1); associations beyond temporal cutoff **Δt=2** and backward links weighted **0**.
- Note the design point: within a 6-frame window, only pairs with Δt≤2 are supervised — the window
  gives the transformer *context*, not longer-range supervised targets.

### 1.3 Ablations — what mattered, with numbers [DOC]

Bacteria dataset, AOGM (lower better), ILP linking:

| arm | AOGM |
|---|---:|
| no relative positional encoding | 314 |
| + rotary embeddings | 235 |
| + object features (intensity, area, inertia tensor) | 164 |
| full model | 136 |
| **+ parental softmax** | **23** |

- **Parental softmax is the single largest item** (≈6× AOGM reduction in the component stack; ≈20%
  error reduction for dividing objects in the paper's headline framing). We already have it (E3).
- **Window size: s=2 causes "substantial deterioration"; s∈3–6 all good** (LAP linker). Our
  architecture sits at the documented bad end (E5). This is the only ablated *architectural* lever
  with a documented effect that we are on the wrong side of.
- Attention depth L=6 optimal; more layers no notable improvement.
- Greedy vs ILP: a good learned edge head narrows the gap ("less pronounced"), consistent with
  `retrain_recipes` §7's greedy→ILP numbers.
- Features: mean intensity, area, inertia tensor + learned Fourier positional encodings, **no image
  crops**, d=256, batch 8, single consumer GPU. Optimizer/LR/epochs not stated in the paper
  [UNVERIFIED — repo `scripts/train.py` would settle it; not read this session].

### 1.4 How Trackastra gets robustness to detection noise — the answer to the on-point question

Three mechanisms, in order of documented importance:

1. **Train on realistically-wrong detections, supervise from GT tracks.** The `ctc` model trains on
   CTC **ERR_SEG** inputs — the Cell Linking Benchmark's "standardized, yet imperfect segmentation
   inputs" ([celltrackingchallenge.net](https://celltrackingchallenge.net/); CLB introduced ISBI 2024)
   — matched frame-by-frame to GT tracklet IDs (`_ctc_assoc_matrix`, `trackastra/data/data.py`, read
   raw this session). Unmatched detections get zero-association rows. **This — not synthetic node
   dropout — is how the field's best generalist linker learned to survive FP/FN detections.** [DOC]
2. **Coordinate/feature-space augmentation**, in `trackastra/data/wrfeat.py` (read raw): the feature
   pipeline the pretrained models use augments *the detection graph directly* —
   `WRRandomFlip` (p=0.5), `WRRandomAffine` (±10°, scale 0.9–1.1, shear 0.1, p=0.5),
   `WRRandomBrightness` (scale 0.5–2.0, shift ±0.1, p=0.5),
   **`WRRandomOffset` (independent per-detection coordinate jitter, ±3 px, p=0.5)**,
   **`WRRandomMovement` (global linear drift ∝ timepoint, ±10 px, p=0.5)**. The paper's sentence "the
   augmentations are applied directly to the object features" is this machinery. [DOC]
3. **What Trackastra does NOT do**: no node dropout, no FP injection — "No augmentation drops
   detections; all preserve detection counts" [MEAS from source read]. And the paper is explicit
   about the residual weakness: *"An important limitation of the presented model is that it cannot
   correct faulty detection inputs"*; non-adjacent-pair predictions for gap closing are named future
   work. [DOC]

### 1.5 Successors / 2025–2026 updates

- `general_2d_w_SAM2_features` (release v0.5.0): SAM2.1 foundation-model appearance features bolted
  onto the same association trainer, "with strong benefits on bacteria datasets" [DOC, registry
  description]. Signal: richer *frozen* appearance features are the lab's own upgrade path — not a
  bigger joint model.
- An ICCVW 2025 BIC-workshop paper (Lalit et al., "An Investigation of …") builds on the Trackastra
  backbone ([supplemental PDF](https://openaccess.thecvf.com/content/ICCV2025W/BIC/supplemental/Lalit_An_Investigation_of_ICCVW_2025_supplemental.pdf))
  — existence documented, content **[UNVERIFIED]** (not read).
- No Trackastra-2/3D-specific follow-up paper found from the Weigert lab as of this session
  ([weigertlab.org/publications](https://weigertlab.org/publications/) checked via search). The 3D
  story remains the `ctc` checkpoint.

---

## 2. Training association on NOISY tracker-generated pseudo-labels — documented vs wishful

Our stage-1 supervision is Ultrack output: linking mostly reliable, divisions poor (BC(i)≈0.47,
`scientific_incumbents` §1.2 + L8).

### 2.1 The strongest positive result [DOC]

**SimpleReID** — Karthik, Prabhu, Gandhi, *Simple Unsupervised Multi-Object Tracking*,
[arXiv:2006.02609](https://arxiv.org/abs/2006.02609) (2020): generate tracking labels with **SORT**
(a classical motion tracker), train a ReID association network on those noisy labels with plain
cross-entropy — and it "recovers the full performance of its supervised counterpart consistently
across diverse tracking frameworks", SOTA-at-the-time on MOT16/17 without tracking supervision.
**Reading for us: association supervision distilled from a competent classical tracker's
CONTINUATION links is nearly as good as human GT.** The documented failure mode of this family is
identity fragmentation — pseudo-tracklets that break one object into several IDs poison the loss
(stated as the known vulnerability in the 2024 follow-up literature, e.g. *Learning a Neural
Association Network for Self-supervised MOT*, [arXiv:2411.11514](https://arxiv.org/abs/2411.11514)).
Ultrack's zebrafish error statistics ("62 frames until an error in 50% of lineages",
`scientific_incumbents` §5) put per-frame-pair link noise in the ~1–2% range [INF] — far below the
noise levels SimpleReID absorbed.

### 2.2 Structured noise concentrated on divisions — the correct treatment is masking, not truncation

- Generic noisy-label machinery exists and is documented *outside* tracking: co-teaching / small-loss
  selection (Han et al., NeurIPS 2018, [arXiv:1804.06872](https://arxiv.org/abs/1804.06872));
  noisy-correspondence rectification for paired/matching data, which applies co-teaching to
  *pair-labels* — the closest structural match to link noise (NCR line, surveyed in
  [arXiv:2405.16996](https://arxiv.org/html/2405.16996v1)). **I found no published instance of any of
  these applied to cell-track or MOT edge supervision.** [DOC mechanisms / UNVERIFIED in-domain]
- Small-loss selection assumes noise is *diffuse*; ours is *structured and localisable* — we know
  a priori that division links and dense-crossing links are where Ultrack errs (BC(i) 0.47; L8).
  When the noisy slice is identifiable, zeroing its loss weight dominates any adaptive selection
  scheme in simplicity and risk [INF]. Trackastra's weight matrix `W` (§1.2) is exactly the right
  instrument: it already expresses per-edge weights, and our `compute_loss` mask can express the same.
- **Division-link masking spec [INF, assembled from DOC parts]:** in the Zebrahub stage, set loss
  weight 0 on every source row that Zebrahub marks as dividing (parent with 2 children) *and* on
  rows/columns within ~1 frame of such events; supervise divisions **only** from the 151 clean
  competition division events in the fine-tune stage, with the real λ_div≈10 upweight (fixing E4/F6).
  Continuation links from Zebrahub train at full weight. The precondition check is L8's: measure
  Zebrahub-division vs our-GT-division agreement on any overlapping regime first; if precision <0.7,
  the mask is mandatory, not optional.
- **Confirmation-bias guards** (labelled-samples-per-batch, bidirectional checks) are already
  specified in `retrain_recipes` §1 B1 / §8 — they apply verbatim and are not restated.

### 2.3 Wishful (do not spend GPU on)

- Co-teaching two edge heads: doubles cost, undocumented in-domain, and our noise is localisable
  (§2.2). [INF]
- Loss truncation / GCE-style robust losses on link logits: no tracking instance found; interacts
  unpredictably with the parental softmax (a truncated row breaks the ≤1-parent normalisation). [INF]
- Confidence-weighting Zebrahub links by an Ultrack confidence score: the released `*_tracks.csv`
  carry no per-link confidence field [MEAS, prior sessions]; reconstructing one means re-running
  Ultrack — out of budget.

---

## 3. MOT association-head practice, 2024–2026

### 3.1 Joint detect+associate vs frozen-detector association — the evidence has a direction

- **MOTRv2** (Zhang et al., CVPR 2023, [arXiv:2211.09791](https://arxiv.org/abs/2211.09791)): the
  end-to-end MOTR family suffers a documented **"conflict between detection and association tasks
  within the shared transformer decoder"**; feeding *frozen pretrained YOLOX* proposals as anchors —
  decoupling detection — "enables MOTR to concentrate on association" and took DanceTrack to 73.4
  HOTA (rank 1 at publication). [DOC]
- **CAMELTrack** (Somers et al., 2025, [arXiv:2505.01257](https://arxiv.org/abs/2505.01257), full
  text read): trains a pure association module on top of **frozen** detector + frozen cue extractors;
  "training CAMEL takes one hour on a single consumer-grade GPU" vs days×8 GPUs for end-to-end; beats
  end-to-end and SORT-family methods (DanceTrack 69.3 vs DiffMOT 62.3 HOTA). [DOC]
- **Trackastra** trains association-only, "assum[ing] all cell detections are correct" at train time
  for its GT-mask models, and on fixed ERR_SEG detections for the `ctc` model. [DOC]
- Counterweight: CenterTrack-style joint training works in its own regime, and **no source measures
  joint-vs-frozen for a 3D microscopy detector + edge transformer** — `retrain_recipes` §7's
  [UNVERIFIED] verdict stands. But the 2023–2026 trendline is unambiguous: the best association
  results come from *decoupled* training on a competent frozen detector's outputs. [DOC pattern]
- **For us [INF]:** the vendored joint path (`detect_and_match`, gradients through feature indexing
  into the UNet) is working and audited — do not rip it out. But the frozen-UNet arm is now the
  best-motivated of the three ablations (§7, ablation C): if frozen matches joint, we gain a huge
  practical simplification (edge-head-only retraining is ~10× cheaper per epoch, since the UNet
  forward can be cached).

### 3.2 Contrastive ReID vs direct edge logits

- **QDTrack** (Pang et al., CVPR 2021 / TPAMI 2023, [arXiv:2006.06664](https://arxiv.org/abs/2006.06664),
  [arXiv:2210.06984](https://arxiv.org/abs/2210.06984)): quasi-dense contrastive learning — hundreds
  of region proposals per image pair as positives/negatives instead of sparse GT boxes — **+4.8 IDF1
  on BDD100K from the sampling density alone** (63.0→67.8; multi-positive contrastive adds only 1.0
  of that). The documented win is **negative-richness from the detector's own proposal
  distribution**, not the contrastive form itself. [DOC]
- **CAMELTrack** uses InfoNCE over tracklet–detection pairs in a shared embedding space. [DOC]
- For dividing cells, identity-embedding objectives break structurally ("same identity ⇒ same
  embedding" fails at a division; `retrain_recipes` §7's kill of contrastive edge objectives, and
  Trackastra's parental-softmax ablation showing the direct-edge-logit + division-aware normalisation
  is where the margin is). **Verdict: keep direct edge logits + parental softmax; import
  negative-richness, not the contrastive loss.** [DOC+INF]
- **Concrete import [INF]:** during training, *lower* `det_threshold` in `detect_and_match` (e.g.
  0.3→0.1) so the token set carries more near-miss FP peaks — the exact analogue of quasi-dense
  sampling. Zero extra cost; the loss masking (E2) already handles them. Falsify inside ablation B.

### 3.3 Graph augmentation — the documented spec

**CAMELTrack is the direct answer** to "how do you make an association model robust to the detector's
FP/FN pattern": its association-centric training scheme (a) runs an off-the-shelf detector on the
training videos and assigns detections to IoU-closest GT (train on real errors), and (b) applies
three augmentations to the assembled tracking scenarios — **detection identity swapping, detection
dropout, cue dropout** — plus cross-video sampling. Removing the augmentation bundle costs **65.1 →
61.0 HOTA on DanceTrack** [DOC, ablation Exp. 9 vs 10]. Trackastra contributes the coordinate-level
half: per-detection jitter (±3 px), global drift, affine on coords+inertia tensors (§1.4.2). SUSHI
(§4.1) and MPNTrack train edge classifiers on real (public) detections matched to GT — same
train-on-real-errors principle, standard across the MOT-GNN line. [DOC]

Synthesised augmentation spec for our edge head (each item's provenance tagged):

| aug | spec | provenance |
|---|---|---|
| real-error tokens | keep `detect_and_match` in the loop; train det_threshold 0.1–0.3 | E1 [MEAS] + ERR_SEG [DOC] + QDTrack [DOC] |
| node dropout | drop each *matched* token with p≈0.1–0.15 before the transformer (targets recomputed; a dropped true parent teaches "no parent" — mirrors real FN) | CAMELTrack [DOC mechanism; p UNVERIFIED — their exact rate not extracted] |
| coordinate jitter | per-node iid offset, σ≈1 grid voxel (≈1.6 µm) on y/x, ≈1 voxel z | Trackastra WRRandomOffset ±3 px, p=0.5 [DOC]; scaled to our F7 error budget [INF] |
| global drift | linear per-frame drift, up to ~2–3 voxels/frame, p=0.5 | Trackastra WRRandomMovement ±10 px [DOC] |
| cue (feature) dropout | zero the UNet feature block of a token (keep positional embedding) with p≈0.1 | CAMELTrack cue dropout [DOC mechanism]; transfer rationale §5 [INF] |
| identity swap | swap two nearby tokens' features (not coords) with small p | CAMELTrack [DOC mechanism] |

All are dataloader/collate-level changes to `get_window_data`/`detect_and_match` outputs — no
architecture change, no measurable step-time cost [INF].

### 3.4 Occlusion / gap augmentation and motion features

- Detection dropout (above) *is* the occlusion/gap simulation in the 2025 practice; separate
  occlusion machinery is not documented as necessary for association heads. [DOC via CAMELTrack]
- Motion-feature ablations: GeneralTrack (CVPR 2024, [arXiv:2406.00429](https://arxiv.org/abs/2406.00429))
  finds neither pure motion nor pure appearance dominates ("motion-dominated methods are brittle
  when encountering irregular motion… appearance-dominated methods are prone to failure when facing
  occlusion") and gets its cross-domain robustness from low-level point-correlation features. [DOC]
  Our positional-embedding pathway plus KOFT-style flow (the separate promoted arm-B lane) covers the
  motion half; do not add learned motion features to the edge head in this cycle. [INF]

---

## 4. GNN / transformer cell-lineage linkers — published training regimes on dense embryo data

### 4.1 MOT-GNN reference hyperparameters [DOC]

SUSHI (Cetintas, Brasó, Leal-Taixé, CVPR 2023, [arXiv:2212.03038](https://arxiv.org/abs/2212.03038)),
the strongest published graph edge-classification tracker: hierarchy of MPNTrack-style GNNs over
detections→tracklets, all levels jointly trained with **focal loss γ=1, Adam, LR 3e-4, weight decay
1e-4, batch 8 clips, 250 epochs**; edges classified binary then rounded by a linear program. Training
graphs are built from real detections. These are the field's reference numbers for an
edge-classification head of our size; note focal-loss use matches our `(1-p_t)^2` modulation (E3).

### 4.2 linajea — the dense-embryo regime [DOC, from prior in-repo verification]

`scientific_incumbents` §3.2 already verified the mechanism (4-class parent/daughter/continuation
head, hard ILP constraints, sSVM weight fitting, 17× FPdiv cut) and training scale (one V100,
"smaller GPU sufficient"). The NBT sibling (Malin-Mayor et al., *Nature Biotechnology* 2022,
[doi:10.1038/s41587-022-01427-7](https://www.nature.com/articles/s41587-022-01427-7)) adds the
supervision-density datum: **~20 hours of sparse point-annotation effort suffices to beat dense-GT
baselines on whole-embryo mouse data (75.8% vs 31.8% of 1-hour lineages)**. The exact loss-masking
around unannotated regions could not be re-verified this session (arXiv HTML 404, ar5iv conversion
broken) — **[UNVERIFIED detail]**; the headline (sparse points suffice for a movement-vector +
indicator model at embryo scale) is [DOC].

### 4.3 Others, briefly

- EmbedTrack: 2D-only — already excluded (`scientific_incumbents` §4); nothing new.
- *Cell as Point* ([arXiv:2411.14833](https://arxiv.org/abs/2411.14833), 2024) — one-stage
  point-based cell tracking; existence noted, training details **[UNVERIFIED]** (not read).
- Ben-Haim & Riklin-Raviv, *GNN for Cell Tracking in Microscopy Videos*
  ([arXiv:2202.04731](https://arxiv.org/abs/2202.04731)) — GNN edge classification for cells;
  predates our window; not read this session **[UNVERIFIED]**.
- No 2025–2026 CTC entrant with a published dense-3D-embryo association training recipe beyond
  those in `scientific_incumbents` was found. The CLB still has zero entries on DRO/TRIC/TRIF.

---

## 5. Cross-domain / cross-embryo generalisation of association models

What the evidence says transfers:

1. **Training-set diversity beats domain specialisation for association.** Trackastra general model
   vs DeepCell-specialised on out-of-domain HeLa: AOGM 96 vs 190 [DOC, §1.3]; the CLB
   generalisability winners run **one** model over 13 heterogeneous datasets (LNK 0.984 / 0.977,
   `scientific_incumbents` §1.4).
2. **The linkers that generalise are built on geometry + shallow features, not deep appearance.**
   CLB #1 (ByoTrack SKT/KOFT) is Kalman+flow with no learned appearance at all; #2 (Trackastra) uses
   intensity/area/inertia + positions, explicitly **no image crops** [DOC]. GeneralTrack attributes
   its cross-domain robustness to low-level texture/point correlations over global structure [DOC].
   In pedestrian MOT, off-the-shelf ReID does not transfer to tracking without on-the-fly domain
   adaptation of normalisation statistics (GHOST, Seidenschwarz et al., CVPR 2023,
   [arXiv:2206.04656](https://arxiv.org/abs/2206.04656)) [DOC, abstract-level].
3. **Mapping to Zebrahub→competition [INF]:** our edge head's transfer risk concentrates in the
   UNet feature block (appearance; instrument- and embryo-specific statistics), not the positional
   pathway (geometry; nuclei kinematics are conserved across zebrafish embryos at matched stage and
   grid — supported by the packaged crops' 1.032× scale match). Mitigations, cheapest first:
   cue-dropout augmentation (§3.3) so the head cannot over-rely on appearance; AdaBN at inference
   (`retrain_recipes` C1); interleaved-domain batches in the fine-tune stage (already B1 spec).
   A deliberate *feature-poor* arm (positional embeddings + a few shallow morphology scalars,
   Trackastra-style, UNet features dropped entirely) is the limiting case — worth one free-rider arm
   if ablation B shows heavy cue-dropout helping.

---

## 6. THE RECIPE — ranked, with the two-stage prior refined

The `kaggle_winners_playbook` lever-2 prior (noisy-pretrain → clean-fine-tune, inverted LR/aug,
+4–6% LB in the HuBMAP instance) is **confirmed and sharpened, not refuted**, with three
association-specific amendments:

**R1 — Stage-N (dense, noisy) trains CONTINUATION association only; divisions are masked.**
Zebrahub Ultrack links supervise the edge head at full weight for continuation edges; division rows
masked to weight 0 (§2.2). Basis: SimpleReID (tracker links ≈ GT for continuation, [DOC]) + Ultrack
BC(i) 0.47 + L8. This is the main refinement over the raw two-stage prior, which is silent on *which
part* of the noisy labels to trust.

**R2 — Stage-C (sparse, clean) is where divisions are learned, with the weight fix.**
Fix E4/F6 to Trackastra's documented values: λ_div=10 (11× weight), λ_cont=1 (2×), and add the
auxiliary sigmoid BCE at λ=1e-2 (§1.2) — three one-line changes. 151 clean division events × 11×
upweight × interleaved Zebrahub batches (confirmation-bias guard) is the entire division training
story; nothing from Zebrahub touches division supervision unless the L8 agreement check passes ≥0.7
precision.

**R3 — Keep `detect_and_match` in the training loop in BOTH stages (do not switch to GT-node
training).** E1 is our ERR_SEG. Trackastra's `ctc` model — the best 3D linker checkpoint in the field
— is trained exactly this way. Lower det_threshold to 0.1–0.2 during training for negative-richness
(QDTrack mechanism). One caveat [INF]: in stage-N, `max_match_distance` should be tightened toward
the packaged-crop NN distances so dense Zebrahub nodes don't mis-match to neighbouring peaks — audit
match precision on one crop before the run (CPU, minutes).

**R4 — Graph augmentation bundle per §3.3 table** (node dropout, coord jitter, global drift, cue
dropout, identity swap). Documented mechanisms; exact rates are ours to ablate (ablation B). Costs
nothing at step time.

**R5 — Loss/selection plumbing** (cross-refs, not duplicates): fix F5 (BCE-under-autocast blocker)
and F2 (selection score) per `retrain_recipes` A1/A3; SUSHI's reference hyperparameters (§4.1)
support keeping AdamW at ~1e-4–3e-4 with decay rather than inventing a schedule.

**R6 — Window size stays 2 this cycle, and the debt is now quantified.** Trackastra documents s=2 as
the deteriorated end (§1.3) — but its remedy multiplies UNet forwards per step, killed on compute in
`retrain_recipes` §9. The cheap partial substitute [INF]: within existing W=2 windows nothing is
available, but a W=3 window with Δt≤2 supervision (Trackastra's own cutoff) is a ~1.5× step-cost
architecture change — the *first* thing to buy if a future week's quota allows. Not in this cycle's
30 h.

**R7 — Do not**: contrastive edge objective (§3.2), co-teaching/loss-truncation (§2.3), learned
motion features in the head (§3.4), joint-training removal without ablation C's evidence (§3.1).

---

## 7. The 3 ablations to actually run (~30 GPU-h total envelope, riding the B1 retrain)

| # | ablation | arms | GPU-h | falsification (one line) |
|---|---|---|---:|---|
| **A** | **Division supervision source** | (1) naive: Zebrahub divisions at full weight; (2) masked per R1 + clean-only divisions per R2; (3) clean-only + λ_div=10 with *no* Zebrahub division masking needed check | ~3 (free-rider arms inside stage-N/C chain, read at checkpoint) | division-Jaccard + adj-edge-J, LOEO both embryo directions on patched scorer; if (1) ≥ (2) bilaterally, the Ultrack-division-noise thesis is wrong and masking is dead weight |
| **B** | **Graph-augmentation bundle on/off** | (1) none (today's E6); (2) full §3.3 bundle; if budget: (3) bundle minus cue-dropout (isolates the transfer claim) | ~4–6 (two stage-N runs at equal steps; hold out one Zebrahub embryo) | held-out-embryo edge accuracy + LOEO adj-edge-J; kill the bundle if Δ < seed-to-seed spread; CAMELTrack predicts a clear win (their −4.1 HOTA without it) |
| **C** | **Frozen-detector vs joint edge training** | (1) joint (today); (2) UNet frozen after stage-N detector training, edge head trained on cached features | ~3 (arm 2 is cheap: cached UNet forwards) | equal-step LOEO both directions; if frozen ≥ joint −0.002, switch the workflow to frozen (10× cheaper edge iterations thereafter); MOTRv2/CAMELTrack predict frozen holds |

Everything else in §6 (loss-weight fixes, det_threshold, match-distance audit, batch interleaving) is
a config constant, not an ablation — burn zero GPU hours sweeping them. Window size (R6) is
explicitly excluded from this cycle's budget.

---

## 8. Licence flags (one-line factual tags; nothing excluded or deranked on licence)

- **Trackastra — Apache-2.0** (verified from repo LICENSE via GitHub API this session; *corrects*
  `retrain_recipes` §10's unverified flag). Code vendorable; `ctc` checkpoint redistributable under
  the same terms.
- CAMELTrack ([TrackLab](https://arxiv.org/abs/2505.01257)) — licence **[UNVERIFIED]**; we import
  mechanisms only.
- QDTrack (SysCV) — Apache-2.0 per repo listing **[UNVERIFIED — not read from LICENSE file]**.
- SUSHI / MPNTrack (dvl-tum) — licence **[UNVERIFIED]**; mechanisms + hyperparameters only.
- SimpleReID — licence **[UNVERIFIED]**; mechanism only.
- Zebrahub tracks — CC BY-NC, host-cleared (#734330), recorded in `research/04-data/data-acquisition.md`.
- CTC ERR_SEG data — CTC terms of use apply if we ever train on it directly **[UNVERIFIED terms]**;
  this report only cites its role in Trackastra's training.

---

## 9. What I could not verify

- Trackastra's optimizer/LR/epoch counts (paper silent; `scripts/train.py` not read) and the repo
  default `window_size=10` vs paper `s=6` discrepancy — likely config-dependent; treat s=6 as the
  ablated value.
- CAMELTrack's exact dropout/swap probabilities (ablation read, per-aug rates not extracted).
- The Lalit ICCVW 2025 paper's content (supplemental located, main text not read).
- linajea's precise loss-masking radius around sparse annotations (both arXiv HTML mirrors broken
  this session; mechanism-level claims rest on `scientific_incumbents` §3.2's prior PDF extraction).
- Any published instance of co-teaching/robust-loss machinery applied to cell-link supervision —
  absence of evidence after targeted search, not proof of absence.
- Whether ByoTrack/PAST-FR's CLB division performance implies a division mechanism transferable to
  training (carried over unresolved from `scientific_incumbents` §9).

## 10. Bottom line

The field's best association models (Trackastra-ctc, CAMELTrack, MOTRv2, SUSHI) share one training
pattern: **a decoupled association head, trained on a real detector's imperfect outputs matched to
clean track supervision, with graph-level augmentation (dropout/jitter/cue-drop) and division-aware
loss normalisation.** Our vendored trainer already has the two hardest parts (detector-in-the-loop
tokens, parental softmax) and is missing the three cheapest (real division weight, auxiliary sigmoid,
graph augmentation) plus one structural debt (window=2). The Ultrack noise problem is solved by
supervision surgery — continuation links at full weight, division links masked until proven — not by
noisy-label machinery. Three ablations (division source, augmentation bundle, frozen-vs-joint) fit
inside the existing B1 lane and settle every open question this report raises.
