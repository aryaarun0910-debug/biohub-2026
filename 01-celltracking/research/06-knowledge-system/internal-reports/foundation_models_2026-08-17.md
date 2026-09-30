# Foundation models & self-supervision for microscopy — frontier scan (agent report) — 2026-08-17

Mandate: is there any 2025–2026 foundation / self-supervised model, with public licence-clean
weights, that we could attach to a Kaggle kernel and that would plausibly beat the shared public
50-epoch UNet3D detector on **this** data (zebrafish 3D+t light-sheet nuclei, 2 embryos, ~2.8%
annotated, `tracking_cellmot`, T4x2 / 12 h / internet-off)?

**ANSWER: NO.** Not one. Every candidate fails on at least one of: (a) it is 2D and its "3D" is
slice-and-stitch; (b) its weights are licence-encumbered for a competition that requires a
redistributable winning solution; (c) it is 2–3 orders of magnitude too slow for ~1,000 nuclei
x 100 frames x N crops on a T4; (d) nobody has measured it beating an in-domain-trained model,
and the head-to-head evidence that *does* exist points the other way.

**Retraining our existing architecture on external Zebrahub (H1) is strictly better value**, and
this report adds three independent 2026 head-to-head results that say so (§6). Two narrow
foundation-model levers survive as *auxiliaries* — offline pseudo-label generation and a
self-supervised pretraining recipe — never as drop-in replacements (§7).

Does **not** re-derive material already in `methods_frontier_2026-08-16.md` (HOCT, D2D-Rescore,
LSM-FM, SELMA3D), `novel_crossdomain_2026-08-17.md` (OT linker, MetaDetect, nnPU, conformal,
BiologicalNeeds, MoTT), `competitive_*` (Zebrahub legality, plateau structure, CoTracker tested
and failed) or `redteam_blindspots_2026-08-17.md`. Where it touches those, it *adds a measurement
or a licence fact* and says so.

---

## 1. Ranked summary

EV ranked by (expected value x cheapness) under T4x2 / 12 h / internet-off / redistributable-if-we-win.

| # | Model (date) | 3D? | Weights + licence | T4-feasible in-kernel? | Documented evidence it beats a task-specific UNet | EV for us |
|---|---|---|---|---|---|---|
| 1 | **CellSeg3D / WNet3D** (eLife 2025-06-24) | **Yes — native 3D** | Public, **MIT** (code + mesoSPIM weights) | Yes (small 3D CNN) | Claims self-supervised ≈ supervised **on sparse cleared-brain light-sheet nuclei**; *not* vs a task-trained UNet on dense embryo nuclei | **LOW-MODERATE — as a pretraining *recipe*, not a detector** |
| 2 | **Cellpose-SAM (`cpsam`)** (bioRxiv 2025-04-28; v4.2 2026-06) | Slice+stitch only; authors say 3D is worse than 2D | HuggingFace; code **BSD-3**; SAM-1 backbone (Apache-2.0) | Marginal (xy-slice), **no** for `do_3D` | mSA 0.544/0.363/0.483/0.418 on 2D generalist eval (MIDL 2026) — **not measured against an in-domain-trained model** | **LOW — offline pseudo-label proposer only** |
| 3 | **Spotiflow** (Nat Methods 2025) | **Yes — native 2D+3D** | Public, permissive (verify) | Yes (small) | Subpixel spot detection, threshold-agnostic; **architecture** lever not a foundation model | **MODERATE — architecture idea for our own head** |
| 4 | **Trackastra `ctc` model** (ECCV 2024; repo live 2026) | Yes (2D+3D `ctc` model only) | Public, **BSD-3**; already mirrored on Kaggle | Yes | Documented to **degrade sharply on non-native masks**; beaten by HOCT | **LOW — already effectively tested by the field, no score posted** |
| 5 | **CELLECT** (Nat Methods 2025-10-20) | **Yes — 3D+t, detects + divides** | Public weights, **GPL-2.0 (HARD FLAG)** | Probably (small UNet) | 91.5% vs Imaris 70.7% on C. elegans embryo light-sheet; **no comparison to a domain-trained UNet** | **LOW — licence-toxic; C. elegans domain** |
| 6 | **micro-SAM (μSAM)** (Nat Methods 2025-03) | Per-slice + multicut merge | Public, BioImage.IO | Marginal | mSA 0.518 nuclei-fluor (best in MIDL 2026) — still a generalist number | **LOW — same class as #2, no 3D advantage** |
| 7 | **MedSAM2** (arXiv 2025-04-03) | Yes (memory attention over slices) | Public, **Apache-2.0** | Yes, but prompt-driven | Radiology (CT/MRI/PET). Zero microscopy evidence. Needs a bbox prompt per object | **NEAR-ZERO — wrong domain, wrong interface** |
| 8 | **SAM 2 / 2.1** (2024-07; on Kaggle Models) | Video, not volumetric | **Apache-2.0**, `metaresearch/segment-anything-2.1` | Yes as an encoder; **NO** as a tracker (§4) | Trackastra's SAM2-feature variant is **beaten by HOCT's hand-crafted features** | **NEAR-ZERO as a tracker; marginal as a frozen 2D encoder** |
| 9 | **Cellpose `cpdino`** (2026-06) | Same as #2 | HuggingFace; **DINOv3 Licence — gated, non-Apache, redistribution clauses** | Marginal | DINOv3 frozen features **lose to nnU-Net 71.0 vs 81.4 avg Dice in 3D** (§6) | **NEGATIVE — licence hazard AND the wrong direction** |
| 10 | **SAM 3 / 3.1** (2025-11-19) | Concept segmentation in video | **Custom SAM Licence** (not Apache) | Yes | **Worst microscopy FM measured**: 0.143 mSA fluor-cell vs Cellpose-SAM 0.363 | **NEGATIVE — kill** |
| 11 | **SAM2 cell-tracking** (arXiv 2025-09-12) | Yes (patch-per-slice + SAM-Med3D) | Code CC-BY-4.0 | **NO — ~20–30 h/crop on T4 (§4)** | CTC top-3 LNK on 13 sets, TRA 0.936 3D | **ZERO — 2–3 OOM too slow** |
| 12 | **CoTracker3 / TAPIR / LocoTrack / TAPNext / TAPIP3D** | 2D RGB video; "3D" = monocular RGB-D world space | Mixed (CoTracker non-commercial-ish; verify) | No at our density | Published statement that general point trackers **fail on mitosis** | **ZERO — kill (already tested by our field)** |
| 13 | **LSM foundation model** (arXiv 2026-05-25) | Yes (SwinUNETR/UNet) | Public repo, **licence not stated** | Fine-tune-only | Few-shot plaque Dice 0.68 vs 0.50 scratch; **mouse/human brain, no zebrafish, no embryo, no nnU-Net baseline** | **LOW — prior only (already logged in methods_frontier #3)** |
| 14 | **3D MAE for microscopy** (MICCAI 2026) | Yes | Code public | n/a | Downstream tasks are **protein localisation / PPI classification**, not detection | **ZERO — wrong downstream task** |
| 15 | **StarDist-3D / Omnipose / InstanSeg** | StarDist yes; InstanSeg 2D only | BSD-3 / Apache-2.0 | Yes | Not foundation models — need in-domain training anyway | **ZERO as FMs; StarDist-3D is a rival architecture, not a free lunch** |
| 16 | **Denoising / restoration FMs** (UniFMIR, FM2S, Poisson2Gaussian, SAVeD) | Mixed | Mixed | Yes | **No measurement on light-sheet zebrafish nuclei detection** | **LOW — unquantified, and our crops are from a published atlas** |

---

## 2. Segmentation / detection foundation models for microscopy

### Cellpose-SAM (the strongest microscopy segmentation FM)
- bioRxiv 2025.04.28.651001, MouseLand, **28 Apr 2025**. SAM ViT-L backbone + Cellpose flow head.
  Trained on **22,826 images / 3,341,254 ROIs** (Cellpose, Cellpose-Nuclei, Omnipose, TissueNet,
  LiveCell and others), made explicitly robust to channel shuffling, cell size, shot noise,
  downsampling, and **isotropic and anisotropic blur** — a genuine domain match to our 4:1 z-anisotropy.
- **3D is the weak point, by the authors' own account.** It "runs on 3D volumes by tiling along each
  axis and stitching the results with a 3D flow field, though quality on 3D is somewhat below the 2D
  case because the underlying flow network is trained mostly on 2D data." Docs add:
  `do_3D=True` computes flows on YX, ZY and ZX slices and averages them; 3D segmentation *ignores*
  `flow_threshold` "because we did not find that it helped to filter out false positives in our test
  3D cell volume"; and it "may fail on highly anisotropic volumes".
  <https://cellpose.readthedocs.io/en/latest/do3d.html>
- Trained diameters 7.5–120 px (mean 30 px). Our native xy is 0.40625 µm; nuclei ~5–7 µm → ~12–17 px
  at native resolution. **In range, but at the small end of the training distribution** (MY INFERENCE).
- **Licence**: repo **BSD-3-Clause** (<https://github.com/MouseLand/cellpose>); `cpsam`/`cpsam_v2`
  weights auto-download from HuggingFace. SAM-1 backbone was Apache-2.0. **Clean.**
- **v4.2 (June 2026)** adds `cpsam_v2` (SAM-ViT-L, "predicts fewer spurious masks in low-contrast
  regions") and `cpdino` / `cpdino-vitb` (**DINOv3** backbones). <https://cellpose.readthedocs.io/en/latest/models.html>
- **HARD LICENCE FLAG on `cpdino`**: it requires `pip install git+https://github.com/facebookresearch/dinov3`.
  DINOv3 is **not** Apache-2.0 (unlike DINOv2): it is a custom Meta licence with gated access
  (date-of-birth + approval), redistribution obligations and a "Built with DINOv3" attribution
  requirement. Attaching weights as a Kaggle dataset **is** redistribution. Avoid `cpdino`.
  <https://ai.meta.com/resources/models-and-libraries/dinov3-license/>

### micro-SAM (μSAM)
- Nature Methods 22, 579–591 (**March 2025**), doi 10.1038/s41592-024-02580-4. SAM fine-tuned for LM
  and EM; napari plugin; models on BioImage.IO.
  <https://www.nature.com/articles/s41592-024-02580-4> / <https://github.com/computational-cell-analytics/micro-sam>
- Volumetric = **segment each slice independently, then Multicut-merge across slices by object
  overlap**. Same slice-and-stitch limitation as Cellpose-SAM; no volumetric backbone.

### The decisive 2026 benchmark of microscopy FMs
**"Revisiting foundation models for cell instance segmentation", arXiv:2603.17845, MIDL 2026, 18 Mar 2026.**
<https://arxiv.org/html/2603.17845v1>

Mean segmentation accuracy (mSA):

| Method | Label-free (cell) | Fluor. (cell) | **Fluor. (nucleus)** | Histopath. (nucleus) |
|---|---|---|---|---|
| Cellpose-SAM | **0.544** | **0.363** | 0.483 | **0.418** |
| APG (μSAM) | 0.541 | 0.344 | **0.518** | 0.398 |
| AIS (μSAM) | 0.480 | 0.347 | 0.513 | 0.390 |
| Cellpose3 | 0.424 | 0.218 | 0.438 | 0.155 |
| **SAM3** | 0.269 | 0.143 | 0.255 | 0.331 |

Three things matter for us:
1. **This is a 2D-slice evaluation, including on the 3D datasets** — the authors flag it as "a
   limitation of our study": "CellPoseSAM, μSAM, and SAM3 support 3D segmentation" but they
   "evaluate over individual slices / frames". **Nobody has published a proper 3D instance
   benchmark of these models.** Any 3D claim you read is extrapolation.
2. **SAM3 is the worst model in the table on microscopy** — it "lacks knowledge of biological terms"
   and is prompt-sensitive. The newest, biggest, most-hyped general model is the least useful here.
3. The authors' own conclusion: *"The performance of models seems to correlate with the size of
   their domain-specific training data."* That is an argument for **more in-domain data**
   (= Zebrahub, = H1), not for a bigger general model.

### CellSeg3D — the only genuinely 3D, genuinely light-sheet, genuinely nuclei asset
- eLife (reviewed preprint → version of record **24 Jun 2025**), doi 10.7554/eLife.99848; PMC12187128.
  <https://elifesciences.org/articles/99848> / <https://github.com/AdaptiveMotorControlLab/CellSeg3D>
- **WNet3D** (self-supervised, no labels) + SwinUNetR (supervised); **direct 3D**, not slice-and-stitch.
- Benchmarked on 4 datasets including a new hand-annotated **mesoSPIM light-sheet** volume set, a
  **Platynereis-nuclei light-sheet** set, a Platynereis-ISH confocal set and Mouse-Skull-nuclei confocal.
  Documented claim: matches or outperforms Cellpose and StarDist in **3D semantic** segmentation on
  mesoSPIM volumes, and **self-supervised WNet3D is as good as or better than supervised models**.
- **Licence: MIT.** Pretrained mesoSPIM WNet3D weights public. Cleanest licence in this report.
- **Where it does NOT apply:** the reported wins are *semantic* segmentation on comparatively sparse
  nuclei in cleared brain, not *instance* detection of densely-packed embryo nuclei with ~9–10 µm
  nearest-neighbour spacing. Do not credit it with performance on our task.

### The rest
- **StarDist**: BSD-3, repo active to 2026-02-14, star-convex polyhedra, native 3D, anisotropy
  supported ("a 5x larger axial vs lateral pixel size should not be a problem", stardist.net/faq).
  **Not a foundation model** — it needs in-domain training exactly like our UNet. A rival
  architecture, not a free asset.
- **Omnipose**: no 2025–2026 3D-foundation development found; its data is folded into the
  Cellpose-SAM training corpus.
- **InstanSeg**: arXiv:2408.15954, **Apache-2.0**, ≥60% faster than alternatives, but **2D only** —
  no documented volumetric support. Not applicable.
- **Spotiflow**: Nature Methods 22, 1495–1504 (**2025**), doi 10.1038/s41592-025-02662-x,
  <https://github.com/weigertlab/spotiflow>. Not a foundation model, but the most *relevant*
  specialist architecture in this scan: multiscale heatmap + **stereographic flow regression** giving
  **subpixel-accurate**, **threshold-agnostic** spot detection, native **2D and 3D**. See §7.2 — this
  connects directly to our measured sub-voxel cliff at σ≈2 µm.

---

## 3. SAM 2 / SAM 3 and the licence picture

| Model | Released | Licence | On Kaggle Models? |
|---|---|---|---|
| SAM 1 | 2023 | Apache-2.0 | `metaresearch/segment-anything` |
| SAM 2 / 2.1 | 2024-07 | **Apache-2.0** | `metaresearch/segment-anything-2`, `...-2.1` |
| SAM 3 / 3.1 | **2025-11-19** (arXiv:2511.16719) | **Custom "SAM Licence"** — commercial use allowed but with use restrictions (no military/ITAR), copyleft-style propagation (derivatives must stay under the SAM Licence and ship a copy of it) | `keras/sam3`, third-party mirrors |
| DINOv3 | 2025-08 | **Custom Meta DINOv3 Licence** — gated access, attribution + redistribution clauses | via `cpdino` |

Verified live: `kaggle models list -s "segment anything"` returns official `metaresearch/segment-anything`,
`-2`, `-2.1` and `keras/sam3`. So SAM-family weights **are** trivially attachable to an offline kernel.
Availability is not the blocker; usefulness is.

**Competition-legal reading (MY INFERENCE, not legal advice):** SAM 2.1 (Apache-2.0), Cellpose
(BSD-3 + SAM-1 Apache-2.0), CellSeg3D (MIT), Trackastra (BSD-3), MedSAM2 (Apache-2.0), StarDist
(BSD-3), InstanSeg (Apache-2.0) are all safe for "attach as dataset + open-source the solution".
**SAM 3 (custom licence)**, **`cpdino`/DINOv3 (gated, custom)** and **CELLECT (GPL-2.0)** are the
three landmines. GPL-2.0 in particular is copyleft: vendoring CELLECT code into our kernel would
push our whole solution to GPL-2.0. Do not touch it without an explicit host ruling.

---

## 4. Video / temporal foundation models — the blunt answer is NO

**SAM2 as a cell tracker.** "Segment Anything for Cell Tracking", arXiv:2509.09943, **12 Sep 2025**,
Chen/Edgü/Jin/Stegmaier; code <https://github.com/zhuchen96/sam4celltracking>, paper CC-BY-4.0.
Zero-shot: SAM2 memory features + cosine similarity for forward linking; mitosis fires when two
candidates score within 0.1. Real results on the CTC blind test (13 datasets): **top-3 LNK on all 13**,
**2nd in TRA (0.936 avg)** and 3rd in SEG (0.695) on the large-scale 3D sets. It is a legitimately
good method.

**And it is unusable for us on cost.** Documented: *"a sequence with 200 frames and about 100 cells
per frame takes roughly one hour"* on an **RTX 4090 (24 GB)**. That is ~20,000 cell-frames/hour.

> MY INFERENCE (arithmetic on their stated throughput, not a measurement):
> one of our crops = 100 frames x ~1,000 nuclei = **100,000 cell-frames ≈ 5 h on a 4090**.
> A T4 is roughly 4–6x slower in realised fp16 throughput → **~20–30 h per crop on one T4**.
> The scored set is many crops. Under 12 h on T4x2 we could process on the order of **one crop**.
> **This is 2–3 orders of magnitude short. Kill.**

**Point-tracking models.** CoTracker3 / TAPIR / LocoTrack / TAPNext are 2D RGB video point trackers;
TAPIP3D (NeurIPS 2025, arXiv:2504.14717), SpatialTracker V2 and DELTA do "3D" tracking in
**camera-stabilised monocular RGB-D world space** — a data model (camera extrinsics, depth maps,
occlusion) that has no counterpart in a volumetric fluorescence stack. There is no published
application of any of them to volumetric microscopy.

Two documented failure statements, not vibes:
- *Cell as Point* (Pattern Recognition 2026, arXiv:2411.14833v4): general-purpose trackers
  "are ineffective for cell tracking because the unique properties of cell behaviors, i.e., cell
  mitosis and apoptosis, have not been considered", and they cannot simultaneously represent cell
  position **and** division state with lineage relationships.
- Our own field already measured this (competitive_frontier: CoTracker loses morphology through
  divisions). **Consistent. Stays killed.**

The only 2026 microscopy point-tracking work found, RIPPLE (arXiv:2605.29220, **28 May 2026**),
is explicitly **human-in-the-loop** — "users click starting points… manual intervention occurs only
where drift appears", 3–25x fewer clicks than exhaustive manual annotation, 2D jellyfish/sperm data.
Not automatable, not 3D, not applicable.

---

## 5. Learned trackers in biology (2025–2026)

- **Trackastra** (ECCV 2024, arXiv:2405.15700). **BSD-3-Clause.** Pretrained models are
  `general_2d`, `general_2d_w_SAM2_features` (**both 2D**) and **`ctc` (2D *and* 3D)** — the latter
  described as "the successor of the winning model of the ISBI 2024 CTC generalizable linking
  challenge". So the **only** 3D-capable Trackastra checkpoint is the CTC one.
  Documented weakness that predicts failure on our regime: highest overall Cell-HOTA **when coupled
  with high-quality segmentation masks**, but "when provided segmentation masks from other sources,
  the score dropped significantly, demonstrating its reliance on high quality inputs." Our candidate
  pool is a deliberately over-proposed junk-heavy set — the worst case for it.
  **Field evidence:** `subinium/biohub-trackastra-public-weights-mirror` (250 MB, uploaded
  **2026-07-02**, 39 downloads) has been attachable for six weeks; **no team has posted a score from
  it.** Verified live via Kaggle CLI. Treat 39 silent downloads as 39 negative results.
- **HOCT** (arXiv:2607.11754, Jul 2026, Bragantini/Theodoro/Royer — **the organisers**). Already the
  #1 entry in `methods_frontier_2026-08-16.md`. **New fact relevant to this mandate:** on the CTC
  leaderboard as of 2026-05-01 it ranks 1st in CLB/LNK/BIO by overall generalisability and top-3 on
  14–16 of 16 datasets, **"without deep pre-trained image encoders"** — using only 19 hand-crafted
  features — and it **beats the best Trackastra variant (`general, SAM2.1`: 6.53±1.68)** on the
  bacteria benchmark. Competing CTC entries include *Medical SAM2 and SAM 3D*, zTrack, MAGIK,
  TrackTour, ByoTrack, MAMHT, Baxter. **The current SOTA cell tracker deliberately declines to use a
  foundation-model encoder and wins.** This is the single most on-point piece of evidence in this
  report. (Licence CC-BY-NC-ND; we already tested HOCT as a drop-in and it underperformed a tuned ILP.)
- **CELLECT** (Nature Methods 22, 2411–2422, **20 Oct 2025**, doi 10.1038/s41592-025-02886-x;
  <https://github.com/zzz333za/CELLECT>). Contrastive embedding learning; a primary UNet takes
  consecutive frames and emits **segmentations, centre points, features, size estimates and division
  estimates** — i.e. it detects, links and predicts divisions itself, in **3D+t**, with a z-ratio
  anisotropy parameter. Benchmarked on **C. elegans embryo confocal and light-sheet**; 91.5% accuracy
  vs Imaris 70.7%; real-time 3D tracking of dividing B cells in a lymph node.
  On paper this is the most on-task learned tracker released in the window.
  **HARD KILL on licence: GPL-2.0.** Also: the Imaris comparison is against commercial classical
  software, **not** against a domain-trained UNet, and the domain is C. elegans, not zebrafish.
- **Cell as Point (CAP)** (Pattern Recognition 2026, arXiv:2411.14833v4, CC-BY-4.0,
  <https://github.com/YXSong000/CAP>). 194.8M params, 8–32x faster than multi-stage methods,
  beats KIT-GE and Trackastra on 2D CTC/DeepCell (e.g. PC-3 TRA 0.952 vs 0.937 in 1.1 s vs 9.1 s).
  **2D only — no 3D extension documented.** Mechanism (joint point + division-state representation)
  is interesting; the model is not usable.
- **Ultrack** (Nature Methods 2025, doi 10.1038/s41592-025-02778-0, royerlab) — the organisers' own
  tool and the thing that generated the Zebrahub `*_tracks.csv` lineages we are allowed to train on.
  Already known; listed here so the lineage of the external labels is explicit.

---

## 6. Cross-domain transfer: does a general vision FM beat a task-specific 3D UNet? — head-to-head

This is the mandate's core question and there **is** direct evidence. All three point the same way.

**(a) DINOv3 vs nnU-Net in 3D — arXiv:2509.06467 ("Does DINOv3 Set a New Medical Vision Standard?"),
v3 dated 21 Jan 2026.** Frozen DINOv3 backbones, slice-wise, on Medical Segmentation Decathlon
(10 tasks), CREMI/AC3-4 electron microscopy, AutoPET-II, HECKTOR.

| Task | nnU-Net | DINOv3-L |
|---|---|---|
| MSD Brain (Task 01) | **78.9** | 65.9 |
| MSD Heart (Task 02) | **89.4** | 78.2 |
| **MSD average, 10 tasks** | **81.4** | **71.0** |
| EM + PET | classical methods | *"fails catastrophically"*, error an order of magnitude worse |

Authors' conclusion, quoted: *"The simple frozen-backbone, slice-by-slice approach is insufficient
to compete with fully optimized 3D segmentation architectures."*
**A 10.4-point Dice gap, in favour of the task-specific 3D CNN, on the strongest general vision
foundation model of 2025.** This is medical rather than microscopy — the *mechanism* (2D general
features applied slice-wise vs an optimised 3D architecture) is identical to what we would be doing.

**(b) Microscopy-specific FMs, MIDL 2026 (§2):** the ranking is driven by **domain-specific training
data volume**, not by backbone scale — and the biggest general model (SAM3) is last.

**(c) The tracking side, HOCT (§5):** SOTA achieved *without* pretrained image encoders, beating the
SAM2-feature variant of the leading learned tracker.

**(d) Our own prior measurement:** `redteam_blindspots` RT-09 — foreign detectors falsified at
0.36–0.51 on this data.

There is **no** published head-to-head in which a general vision foundation model's features beat a
task-specific 3D UNet for nucleus detection in 3D light-sheet. Not one. Anyone claiming otherwise is
extrapolating from 2D natural-image benchmarks.

---

## 7. What actually survives — two auxiliary levers and one architecture idea

Nothing here is a drop-in detector replacement. All three are cheap and all three fold into the
existing H1 retrain rather than competing with it.

### 7.1 CellSeg3D WNet3D as a **self-supervised pretraining recipe** for our own UNet3D
Not "attach their weights" — their weights are cleared mouse brain. **Run their method** (MIT,
native 3D, no labels needed) on the 97.2% unannotated nuclei plus the Zebrahub corpus, then use the
resulting encoder as the initialisation for our detector before supervised fine-tuning on the 2.8%.
This is the one 2026 idea that directly attacks our binding constraint (label scarcity) with a
3D-native, light-sheet-validated, licence-clean method.
- **Falsification test:** LOEO, patched scorer — does a WNet3D-pretrained detector beat an
  identically-trained scratch detector by ≥+0.005 on **both** embryo directions? If pooled delta is
  ≤0 on either direction, kill.
- **GPU estimate:** ~6–10 T4-hours for SSL pretraining on the crops + ~4–6 T4-hours fine-tune per
  fold. Call it **~20 T4-hours for a two-fold LOEO answer**, i.e. two 12 h sessions.
- **Honest prior:** the closest published analogue (LSM-FM few-shot: 0.68 pretrained vs 0.50 scratch)
  is a *few-shot* gain. At our label count the gap usually shrinks. I would not bet above 30% that
  this clears +0.005 bilaterally.

### 7.2 Spotiflow-style stereographic-flow / subpixel head on **our** detector — HIGHEST EV in this report
Not a foundation model at all, which is exactly why it survives. `competitive_refresh` CW1 measured
a **cliff at σ≈2 µm centroid error** on the official scorer (σ=2.5 µm → −16%, 3 µm → −41%,
4 µm → −74%) because matching is one-to-one at 7 µm with ~9–10 µm neighbour spacing. Spotiflow
(Nature Methods 2025) is a published, native-3D, **subpixel-accurate, threshold-agnostic** detection
head built for precisely this failure mode, and it is a small architecture we can graft onto the
existing UNet3D rather than a 300M-parameter import.
- **Falsification test:** measure our current peak centroid residual against GT on both embryos. If
  the residual is already <1.5 µm, the lever is closed and Spotiflow is irrelevant. If it is >2 µm,
  add a stereographic-flow/offset regression head and re-score LOEO.
- **GPU estimate:** **0 GPU-hours to falsify** (the residual measurement is CPU on cached graphs);
  ~8–12 T4-hours if the head is worth training.
- This is already partially live as the sub-voxel refine lane (commit `09e1c23`). **This report's
  contribution: the published architecture that formalises it, with a Nature Methods citation.**

### 7.3 Cellpose-SAM as an **offline** pseudo-label / second-proposer generator (zero kernel cost)
`methods_frontier` #2 proposed Cellpose-SAM as an in-kernel second proposer. **Correcting that on
cost:** at native 256x256 xy, one crop is 100 t x 64 z = **6,400 ViT-L slice passes**; `do_3D`
(YX+ZY+ZX) is ~9x that. At ~20–50 slices/s on a T4 (MY ESTIMATE, unmeasured) the slice-and-stitch
path is ~2–5 min/crop — borderline for a whole test set inside 12 h, and `do_3D` is out entirely.
- **Correct use: run it once, offline, on the *training* crops and on Zebrahub**, to mine
  pseudo-positive nuclei among the 97.2% unannotated, feeding the learned re-scorer / PU-loss
  pipeline in `novel_crossdomain` #2/#3. **Zero submission-time cost, zero licence risk (BSD-3 +
  Apache-2.0 SAM-1), no dependence on it being good in 3D** — we only need it to be *complementary*,
  and its verified robustness to anisotropic blur and downsampling is the reason to expect that.
- **Falsification test:** on embryo A, what fraction of GT nuclei that our DoG/UNet pool **misses**
  are recovered within 7 µm by a Cellpose-SAM slice-stitch pass? If <10% added recall, kill.
- **GPU estimate:** ~2–4 T4-hours offline, one-off. **Do NOT use `cpdino` — licence.**

---

## 8. Explicit kills (do not reopen without a new mechanism)

| Killed | Reason (with the fact that kills it) |
|---|---|
| **SAM 3 / 3.1 in any role** | Worst microscopy FM measured (0.143 mSA fluor-cell vs Cellpose-SAM 0.363, MIDL 2026) **and** a restrictive custom licence with derivative-propagation clauses |
| **SAM2 as a video tracker on our data** | Authors' own throughput: 20,000 cell-frames/h on a 4090 → ~20–30 T4-hours **per crop** (MY ARITHMETIC). 2–3 OOM short |
| **All point-tracking models** (CoTracker3, TAPIR, LocoTrack, TAPNext, TAPIP3D, SpatialTracker, DELTA) | Wrong data model (RGB / monocular RGB-D + camera geometry, not volumetric intensity); published statements that they fail on mitosis; already measured failing in our field |
| **`cpdino` / any DINOv3-backed model** | Gated non-Apache licence with redistribution clauses **and** DINOv3 loses to nnU-Net by 10.4 Dice points in 3D |
| **CELLECT** | **GPL-2.0** copyleft would infect the winning solution; C. elegans domain; baseline is Imaris, not a trained UNet |
| **MedSAM2** | Radiology domain (CT/MRI/PET/US/endoscopy); prompt-driven (bbox per object) — no automatic dense instance detection; zero microscopy evidence |
| **3D MAE for microscopy (MICCAI 2026)** | Downstream tasks are protein localisation and PPI classification, not detection or segmentation |
| **Denoising / restoration as a scored lever** | No documented measurement that any 2025–2026 restoration model improves nuclei *detection* on light-sheet embryo data. Unquantified = not a lever. Revisit only if the H1 retrain shows a depth-dependent recall gradient |
| **Trackastra as a drop-in** | Documented sharp degradation on non-native masks; beaten by HOCT; weights mirrored on Kaggle since 2026-07-02 with 39 downloads and **zero posted scores** |

---

## 9. Bottom line

Retraining the existing UNet3D + edge-head architecture on external Zebrahub, with our own
self-supervision on the 97.2% unannotated nuclei, is **strictly better value** than importing any
2025–2026 foundation model. Three independent 2026 results support that:

1. **DINOv3 frozen features lose to nnU-Net by 10.4 Dice points averaged over 10 3D tasks**, and the
   authors state plainly that slice-wise frozen backbones cannot compete with optimised 3D
   architectures (arXiv:2509.06467, Jan 2026).
2. **Microscopy FM performance tracks domain-specific training-data volume, not backbone scale**, and
   nobody has yet published a true 3D instance benchmark of any of them (MIDL 2026, arXiv:2603.17845).
3. **The current SOTA cell tracker — from the competition organisers' own lab — achieves it
   deliberately *without* a pretrained image encoder**, and beats the SAM2-feature variant of the
   leading learned tracker (HOCT, arXiv:2607.11754).

The competition's binding constraint is **labelled in-domain data**, not model capacity. Every
foundation model in this report is an answer to a capacity problem we do not have. The two things
worth taking from the 2025–2026 literature are a **self-supervised 3D pretraining recipe**
(CellSeg3D/WNet3D, MIT) and a **subpixel detection-head architecture** (Spotiflow, Nature Methods
2025) — both of which make *our* model better rather than replacing it, and the second of which
attacks an already-measured +σ≈2 µm cliff for zero GPU-hours to falsify.

---

## 10. Sources

Segmentation FMs — Cellpose-SAM: <https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1> (2025-04-28);
repo/licence <https://github.com/MouseLand/cellpose> (BSD-3); models <https://cellpose.readthedocs.io/en/latest/models.html>;
3D docs <https://cellpose.readthedocs.io/en/latest/do3d.html>.
μSAM: <https://www.nature.com/articles/s41592-024-02580-4> (Nat Methods 22:579-591, 2025-03);
<https://github.com/computational-cell-analytics/micro-sam>.
FM benchmark: <https://arxiv.org/html/2603.17845v1> (MIDL 2026, 2026-03-18).
CellSeg3D: <https://elifesciences.org/articles/99848> (eLife, 2025-06-24, PMC12187128);
<https://github.com/AdaptiveMotorControlLab/CellSeg3D> (MIT).
StarDist: <https://github.com/stardist/stardist> (BSD-3); <https://stardist.net/faq/>.
InstanSeg: <https://arxiv.org/abs/2408.15954> (Apache-2.0).
Spotiflow: <https://www.nature.com/articles/s41592-025-02662-x> (Nat Methods 22:1495-1504, 2025);
<https://github.com/weigertlab/spotiflow>.

SAM family — SAM 3: <https://arxiv.org/abs/2511.16719> (2025-11-19); licence
<https://github.com/facebookresearch/sam3/blob/main/LICENSE>; SAM 3.1 <https://ai.meta.com/blog/segment-anything-model-3/>.
SAM 2 licence: <https://github.com/facebookresearch/sam2/blob/main/LICENSE> (Apache-2.0).
DINOv3 licence: <https://ai.meta.com/resources/models-and-libraries/dinov3-license/>.
MedSAM2: <https://arxiv.org/abs/2504.03600> (2025-04-03), Apache-2.0, <https://github.com/bowang-lab/MedSAM2>.
Kaggle availability verified live 2026-08-17 via `kaggle models list -s "segment anything"`.

Video/temporal — SAM2 cell tracking: <https://arxiv.org/abs/2509.09943> (2025-09-12),
<https://github.com/zhuchen96/sam4celltracking>.
CoTracker3: <https://github.com/facebookresearch/co-tracker>. TAPIP3D: <https://arxiv.org/abs/2504.14717>
(NeurIPS 2025). RIPPLE: <https://arxiv.org/abs/2605.29220> (2026-05-28).

Trackers — Trackastra: <https://arxiv.org/abs/2405.15700> (ECCV 2024), <https://github.com/weigertlab/trackastra>
(BSD-3), model list `trackastra/model/pretrained.json`. HOCT: <https://arxiv.org/html/2607.11754v1> (2026-07).
CELLECT: <https://www.nature.com/articles/s41592-025-02886-x> (Nat Methods 22:2411-2422, 2025-10-20),
<https://github.com/zzz333za/CELLECT> (**GPL-2.0**). CAP: <https://arxiv.org/html/2411.14833v4>
(Pattern Recognition 2026, CC-BY-4.0). Ultrack: <https://www.nature.com/articles/s41592-025-02778-0> (Nat Methods 2025).

Cross-domain — DINOv3 medical benchmark: <https://arxiv.org/html/2509.06467v3> (v3 2026-01-21).
LSM foundation model: <https://arxiv.org/html/2605.26026v1> (2026-05-25), <https://github.com/AdinaScheinfeld/lsm_fm_public_repo>
(licence unstated). 3D MAE: <https://arxiv.org/abs/2606.23964> (MICCAI 2026, 2026-06-22).
STED-FM: <https://www.biorxiv.org/content/10.1101/2025.06.06.656993v1.full> (2025-06).
UniFMIR: <https://www.nature.com/articles/s41592-024-02244-3>.

## 11. UNVERIFIED / open

- **T4 throughput numbers in §4 and §7.3 are my arithmetic on published 4090/A100 figures, not
  measurements.** If someone wants to reopen SAM2-tracking they should measure one crop on a T4
  first; I predict >10 h.
- **LSM-FM licence is not stated in the paper** — must be read off the repo before any use.
- **CoTracker / TAPIR licence terms not verified** in this scan (they are killed on capability, so
  it did not matter).
- **No published true-3D instance benchmark exists for Cellpose-SAM, μSAM or SAM3.** Every 3D claim
  about them in this report is either the authors' own caveat or my inference, and is labelled.
- **CellSeg3D's wins are semantic segmentation on sparse cleared-brain nuclei**, not instance
  detection on dense embryo nuclei. §7.1's EV is a prior, not a measurement.
- I did not find any 2025–2026 paper measuring a restoration/denoising model's effect on **nuclei
  detection recall in light-sheet embryos**. If that paper exists I missed it.
