# Scientific incumbents: who actually does this best in the literature — 2026-08-17

Mandate: the SCIENTIFIC (not Kaggle) frontier. Cell Tracking Challenge, named tool/lab incumbents,
zebrafish/Zebrahub specifically, and the documented hard failure modes. Deliberately does NOT
re-cover: HOCT (arXiv:2607.11754), D2D-Rescore, Cellpose-SAM, LSM-FM, SELMA3D, unbalanced-OT,
MetaDetect/LMD, nnPU, conformal/MOT-CUP, MoTT, heterophily-LP, model soups — see
`methods_frontier_2026-08-16.md` and `novel_crossdomain_2026-08-17.md`.

**Primary-source note.** The CTC leaderboard is published only as a PNG. I downloaded and parsed
the challenge's own spreadsheets — `public.celltrackingchallenge.net/documents/CellTrackingBenchmark.xlsx`
(vintage **2025-08-15**) and `.../CellLinkingBenchmark.xlsx` (vintage **2025-01-31**) — and resolved
team codes against `celltrackingchallenge.net/participants/`. Every CTC number below is extracted
from those files by me, not quoted from a paper. **Vintage caveat: the CTC results are ~12 months
stale relative to today; I could not verify any submission after 2025-08-15.**

---

## Ranked summary — by (expected value × cheapness) FOR US

| # | Method (team) | Core mechanism | Code / licence | Portable to us? | EV |
|---|---|---|---|---|---|
| 1 | **ByoTrack — SKT/KOFT** (PAST-FR, Institut Pasteur) | Optical-flow **measures** velocity, fed into a Kalman filter; LAP linking + adaptive gating + EMC2 stitching + FB interpolation | `github.com/raphaelreme/byotrack` **MIT**, pip, pure-Python, 2D+3D, has GEFF I/O | **YES — highest.** CPU, no Java/Gurobi. Deps already in Kaggle base image | **HIGH.** #1 on the CTC linking-only benchmark; principled version of our best un-shipped lever (arm-B flow gate, +0.008 measured) |
| 2 | **Trackastra `ctc` model** (EPFL-CH, Gallusser & Weigert) | Transformer over full spatio-temporal context of detections in a window; division-aware; greedy or ILP linking | `github.com/weigertlab/trackastra` **BSD-3**; `ctc` checkpoint is **2D+3D**, single zip at a fixed GitHub release URL | **YES.** Download zip → attach as Kaggle dataset → offline. Fine-tunable on CTC-format data | **HIGH.** #2 generalizable linker (LNK 0.977 / BIO 0.794 over 13 datasets). New fact vs our prior "detour" verdict: a **3D CTC-trained checkpoint exists** |
| 3 | **UQ for LAP trackers** (Paul et al., Jülich/FZJ) | Frame-to-frame linking recast as Bayesian inference *and* as classification → calibrated per-link uncertainty; framework-agnostic wrapper | arXiv:2503.09244; repo UNVERIFIED | **YES.** Pure post-hoc, CPU | **MED-HIGH.** Directly feeds FP-suppression + node-count calibration; cheapest of all |
| 4 | **linajea + cell-state classifier** (JAN-US, Funke/Kainmueller) | 4D U-Net → cell-indicator + **backward movement vectors**; 3D ResNet18 classifies each candidate **parent / daughter / continuation / polar-body**; classes enter ILP as costs **and hard consistency constraints** | `github.com/funkelab/linajea` **MIT** | **PARTIAL.** Mechanism yes; the repo needs `pylp` + a MIP solver — reimplement, don't vendor | **MED-HIGH for divisions.** Only documented mechanism with a large measured **FPdiv cut on 3D+t nuclei (17×)** |
| 5 | **ELEPHANT** (IGFL-FR, Sugawara/Averof) | U-Net → {background, nucleus **centre**, nucleus **periphery**}; 2nd U-Net → 3D displacement field; link by nearest-neighbour **after flow warping** | `elephant-track/*` **BSD-2** (client & server) | **MECHANISM ONLY.** Software is a Fiji/Mastodon client + Docker server — unusable in a kernel | **MED.** Published proof that **~2 % annotation suffices** (483 of 23,829 nuclei) — our exact regime |
| 6 | **OrganoidTracker 2.0** (AMOLF-NL, Betjes/van Zon) | NN + statistical physics → **calibrated error probability per linking step**; keep only high-confidence segments | `github.com/jvzonlab/OrganoidTracker` **GPL-2.0** ⚠ (NN files MIT) | **MECHANISM ONLY.** GPL-2.0 is a prize-competition risk | **MED.** Best TRA/BC(i)/CT/CCA on the closest CTC dataset, by calibration not by a bigger model |
| 7 | **CELLECT** (THU-CN (3), Tsinghua) | Contrastive embedding per cell centre + 2 MLPs (intra-frame, inter-frame) for association; U-Net also emits division estimate | `github.com/zzz333za/CELLECT` **GPL-2.0** ⚠; weights in-repo; `--zratio` for anisotropy | **MECHANISM + weights, but licence-blocked** | **MED.** Strong generalisation claim; GPL-2.0 makes shipping it in a prize kernel unwise |
| 8 | **KTH-SE / Baxter Algorithms** (Magnusson/Jaldén/Blau) | Global track linking: greedily add one track at a time, each optimised over **all frames** by Viterbi; no ML | Software available via CTC; MATLAB | **MECHANISM ONLY** (MATLAB) | **MED, narrow.** **The only method that does divisions well on dense light-sheet embryos** (BC 0.67 vs field ≤0.49) |
| 9 | **Ultrack** (CZB-US = **the organisers**) | Hierarchy of candidate segmentations (ultrametric contour map) → ILP selects a disjoint, temporally-consistent subset | `github.com/royerlab/ultrack` **BSD-3**; Gurobi optional, **CBC** free fallback | **RUNNABLE but wrong shape.** Wants foreground+contour maps; CBC is slow/memory-hungry | **LOW for us.** Wins CTC via **SEG**, which our metric does not score; division-weak (BC 0.47 / 0.18 / 0.44) |
| 10 | **PAC-MAP** (DeVosLab) | 3D U-Net predicts **proximity-adjusted** centroid probability map; weak pretrain on baseline labels + finetune on manual | `github.com/DeVosLab/PAC-MAP` **CC BY-NC-SA 4.0** 🚫 | **NO — licence blocker** | **ZERO.** Non-commercial licence vs a $60 k prize. Mechanism (proximity-weighted centroid map) is reimplementable from the paper |
| — | TGMM, TrackMate/Mastodon/MaMuT, Elephant server, btrack, LapTrack, EmbedTrack, BiologicalNeeds, CellTracksColab, DaXi | see §4 | — | **NO** | **ZERO** — reasons in §4 |

Cheapest-first falsification order: **3 → 1 → 2 → 4**. Items 1 and 2 are the two independent
load-bearing bets; 3 de-risks both; 4 is the only live division mechanism left.

---

## 1. Cell Tracking Challenge — what the benchmark actually says

### 1.1 Team identities (DOCUMENTED, from `celltrackingchallenge.net/participants/`)

| Code | Who | What |
|---|---|---|
| **CZB-US** | Jordão Bragantini, Loïc Royer, CZ Biohub SF | **Ultrack** — and **the organisers of our Kaggle competition** |
| **JAN-US** | Malin-Mayor, Hirsch, Guignard, McDole (HHMI Janelia + MDC Berlin) | **linajea + cell-state classifier** |
| **IGFL-FR** | Ko Sugawara, IGFL Lyon | **ELEPHANT** |
| **EPFL-CH (\*)** | Benjamin Gallusser, Martin Weigert, EPFL | **Trackastra** (linking-only, generalizable, 04/2024) |
| **PAST-FR (\*)** | Rémé, Manneville, Newson, Angelini, Olivo-Marin (Institut Pasteur + Télécom Paris) | **ByoTrack SKT/KOFT** (linking-only, generalizable, 12/2024) |
| **LUH-GE** | Kaiser, Schier, Rosenhahn, Leibniz Hannover | **BiologicalNeeds** (linking-only, 04/2024) |
| **KTH-SE (1),(2)** | Magnusson, Jaldén, Blau | **Baxter Algorithms** (Viterbi global linking) |
| **KIT-GE** | Karlsruhe Institute of Technology | graph-based coupled min-cost-flow on CNN distance predictions |
| **MPI-GE (CBG)** | MPI-CBG Dresden | (multiple submissions; (3) is the strong 3D one) |
| **AMOLF-NL** | AMOLF, Amsterdam | page is empty (added 2025-08-15). **INFERENCE:** = OrganoidTracker 2.0 (Betjes et al., *Nat. Methods* 22:2400–2410, 2025) — AMOLF's own press release states an AMOLF algorithm ranked 1st in the CTC for *C. elegans*, and that is their only 2025 *Nat. Methods* tracking paper |
| **THU-CN (3)** | Tsinghua | **INFERENCE:** = CELLECT (*Nat. Methods* 22:2411–2422, 2025), which itself claims top CTC ranking on Fluo-N3DH-CE |

### 1.2 Fluo-N3DH-CE (*C. elegans* embryo, 3D+t confocal nuclei — the closest CTC analogue to us)

Extracted from `CellTrackingBenchmark.xlsx`, mean over videos 01/02:

| Team | = | SEG | **TRA** | **BC(i)** | CT | CCA |
|---|---|---|---|---|---|---|
| AMOLF-NL | OrganoidTracker 2.0 | 0.573 | **0.994** | **0.930** | **0.777** | **0.967** |
| MPI-GE (CBG) (3) | — | 0.465 | 0.987 | 0.757 | 0.583 | 0.770 |
| JAN-US | linajea+csc | 0.599 | 0.979 | 0.887 | 0.678 | 0.932 |
| IGFL-FR | ELEPHANT | 0.631 | 0.975 | 0.854 | 0.414 | 0.590 |
| THU-CN (3) | CELLECT | **0.725** | 0.974 | 0.672 | 0.416 | 0.819 |
| CZB-US | **Ultrack (organisers)** | 0.722 | 0.967 | 0.469 | 0.257 | 0.588 |
| KTH-SE (1) | Baxter | 0.662 | 0.945 | 0.630 | 0.295 | 0.762 |
| KIT-GE (2) | KIT min-cost-flow | **0.729** | 0.886 | 0.088 | 0.008 | 0.597 |

**The single most load-bearing finding in this report: SEG rank and TRA/BC(i) rank are
anti-correlated.** The best segmenter on this dataset (KIT-GE (2), SEG 0.729) has the **worst**
branching correctness in the table (0.088) and reconstructs 0.8 % of complete tracks. The best
tracker (AMOLF-NL) has the **worst** SEG. Ultrack, which wins the challenge's headline OP_CTB
(= 0.5·(SEG+TRA)) more often than anyone, sits mid-table on divisions.

Why this matters operationally: **our Kaggle metric has no segmentation term.** Adjusted edge
Jaccard + division Jaccard + node-count multiplier is a DETECTION-and-LINKING metric. The correct
external incumbents to copy are therefore the point/marker-based, learned-linking, division-aware
methods (**AMOLF-NL, JAN-US, IGFL-FR, PAST-FR, EPFL-CH**) — *not* the segmentation-selection
methods that top OP_CTB. **We have been reading the wrong leaderboard column by default.**

### 1.3 Dense light-sheet embryo datasets — divisions collapse for almost everyone

`Fluo-N3DL-DRO` (*Drosophila*, SIMView light-sheet), `Fluo-N3DL-TRIC/TRIF` (*Tribolium*, Zeiss
LightSheet). All three are **partially annotated** — DRO annotates only nervous-system cells,
TRIC/TRIF only blastoderm lineages (`celltrackingchallenge.net/3d-datasets/`). Same sparse-GT
regime as our 2.8 %.

BC(i), branching correctness (extracted):

| Dataset | Best | Runner-up | Rest of field |
|---|---|---|---|
| Fluo-N3DL-TRIC | **KTH-SE (2) 0.668** | MPI-GE (CBG) (3) 0.206 | CZB-US 0.176, KIT-GE (2) 0.150, RWTH-GE (2) 0.026, DREX-US 0.000 |
| Fluo-N3DL-TRIF | **KTH-SE (2) 0.666** | RWTH-GE (3) 0.490 | CZB-US 0.442, MPI-GE (CBG) (3) 0.307, KIT-GE (2) 0.016 |
| Fluo-N3DL-DRO | *no BC(i) reported at all* (n=0 submissions scored) | | |

Only 5–8 teams have ever submitted to these datasets at all. **DOCUMENTED external corroboration
that division-Jaccard ≈ 0 on dense embryo light-sheet is the field norm, not our failure.** It also
says the ceiling is not zero: a classical global-Viterbi method reaches 0.67 where deep methods
reach 0.02–0.49.

INFERENCE (mine, flagged): KTH-SE's advantage is that Viterbi optimises each track over **all**
frames jointly, so a division is accepted on long-horizon evidence rather than a per-frame score.
That is the same structural point our own D-01/D-07 analysis reached from the other direction
("within-mother rank is informative; a global threshold is not").

### 1.4 The Cell Linking Benchmark (CLB) — the benchmark that is actually shaped like our task

Introduced at ISBI 2024 (`celltrackingchallenge.net/news/`, 2024-05-27). **LNK = linking accuracy
computed against synchronised (i.e. ground-truth) vertex sets, with zero penalty for detection
errors** (`celltrackingchallenge.net/evaluation-methodology/`). OP_CLB = 0.5·(LNK + BIO), BIO =
mean(CT, TF, BC(i), CCA).

Fluo-N3DH-CE, extracted from `CellLinkingBenchmark.xlsx`:

| Team | = | LNK | BC(i) | CT | TF |
|---|---|---|---|---|---|
| PAST-FR (\*) | **ByoTrack SKT/KOFT** | **0.982** | **0.838** | **0.729** | **0.971** |
| EPFL-CH (\*) | **Trackastra** | 0.971 | 0.763 | 0.620 | 0.941 |
| RWTH-GE | — | 0.962 | 0.641 | 0.436 | 0.933 |
| MON-AU (\*) | — | 0.902 | 0.000 | 0.192 | 0.803 |
| SIAT-CN (\*) | — | 0.899 | 0.189 | 0.177 | 0.792 |
| KTH-SE (\*) | Baxter | 0.851 | 0.203 | 0.135 | 0.710 |

Generalizability ranking over all 13 datasets (same file): **LNK — PAST-FR 0.984 (#1), EPFL-CH
0.977 (#2)**, RWTH-GE(\*) 0.972, SIAT-CN 0.965, KTH-SE 0.958, MON-AU 0.914. **BIO — PAST-FR 0.862
(#1), EPFL-CH 0.794 (#2)**, KTH-SE 0.752.

Three operationally important facts:
1. **Only 9 linkers have ever entered the CLB.** Linking-only is the *least* crowded benchmark in
   the field — the opposite of the Kaggle plateau.
2. **Zero CLB entries on DRO/TRIC/TRIF.** Nobody has published a linking-only result on dense
   light-sheet embryo data. There is no external incumbent for exactly our problem.
3. The two winners are both **MIT / BSD-3 and pip-installable**.

### 1.5 What the CTC organisers themselves conclude

*The Cell Tracking Challenge: 10 years of objective benchmarking*, Maška, Ulman et al.,
**Nature Methods 20(7):1010–1020 (2023)**, doi:10.1038/s41592-023-01879-y, PMID 37202537.
Full text is paywalled and I could not retrieve it (Nature IDP redirect, PMC reCAPTCHA, Europe PMC
stub, Caltech 403). The following are **the only statements I could verify**, from the abstract and
from secondary sources; treat the rest of the paper as UNVERIFIED:

- "even if the cell detection task seems nearly solved for most datasets, the segmentation task
  still requires further attention" — DOCUMENTED (abstract).
- The 7th edition (ISBI 2024) introduced the **linking-only benchmark** to allow "objective
  evaluation of object-linking methods over standardized segmentation inputs" — DOCUMENTED
  (`celltrackingchallenge.net/history/`).
- The paper contains dedicated **generalizability** and **reusability** studies of top-performing
  methods — DOCUMENTED that they exist; their conclusions are **UNVERIFIED** by me.
- "Large three-dimensional datasets, such as those of developing embryos, were identified as
  extremely challenging due to the high number and density of cells" — attributed to this paper by
  a secondary source; **UNVERIFIED** verbatim.

**Do not cite this paper's conclusions to me second-hand.** If a decision hinges on it, the cheapest
fix is a library/institutional PDF.

---

## 2. The two methods worth actually running

### 2.1 ByoTrack — SKT / KOFT (rank 1)

DOCUMENTED. `github.com/raphaelreme/byotrack`, **MIT**, `pip install byotrack`, v2.0.4.
README states verbatim: "**ByoTrack (PAST-FR)** won the Cell Linking Benchmark of the Cell Tracking
Challenge with its **SKT/KOFT** implementation." Paper: Rémé et al., *Particle tracking in
biological images with optical-flow enhanced Kalman filtering*, **IEEE ISBI 2024**,
doi:10.1109/ISBI56570.2024.10635656.

Mechanism (from the KOFT README, verbatim abstract): most trackers "assume near-constant position,
velocity or acceleration… such assumptions are not robust to the large and sudden changes in
velocity that typically occur in in vivo imaging. In this paper, we exploit optical flow to
**directly measure the velocity** of particles in a Kalman filtering context." Reported to "divide
tracking errors by two" vs other trackers under high density + fast elastic motion.

**Why this is our #1.** Our strongest un-shipped internal lever is arm B, the flow-compensated
motion-relink gate — measured **+0.00798 pooled / +0.01676 (44b6) / +0.00671 (6bba), P(Δ>0)=1.000**
on the deployment substrate (`quickwins_internal_2026-08-17.md` Candidate 1). Arm B is a *one-step,
gate-only* version of KOFT: it uses kNN16 tissue flow to shift the eligibility test, but keeps a
memoryless cost. KOFT is the same idea done properly — flow as a **velocity measurement** inside a
Kalman state that carries across frames, plus adaptive gating. The literature therefore says our
best measured lever is the right *shape* and we are running the crude version of it.

Runtime fit: dependencies are `networkx, numba, numpy, opencv-python, scipy, scikit-image,
tifffile, torch, torch-kf, pylapy` — all present or trivially vendored in a Kaggle image. **No
Java, no Icy, no Fiji, no Gurobi** for the native components (SKT, KOFT, RTSSmoother, EMC2Stitcher,
ForwardBackwardInterpolater, WaveletDetector). It ships **`byotrack.geff`** I/O — the same graph
exchange format our submission uses. 2D **and 3D**.

Honest caveats:
- The public linker list (nearest-neighbour euclidean / optical-flow / Kalman / KOFT, EMHT wrapper,
  TrackMate wrapper) advertises **no division model**. Yet PAST-FR scores BC(i)=0.838 on
  Fluo-N3DH-CE. **How their CTC submission produces divisions is UNVERIFIED** — read their
  submission description before assuming divisions come free.
- README warns: "Some components assume that individual frames fit in memory, which may limit
  scalability to very large 3D volumes." Our crops are (100,64,256,256) — fine.
- `ForwardBackwardInterpolater` "replace[s] miss-detection by an interpolated **position**" — this
  is the published implementation of our CW3 audit item (interpolate a node; a dt≠1 skip edge scores
  nothing).

**Falsification test (CPU, no GPU).** Replay `artifacts/kaggle/p0strict_cache/graphs` through
ByoTrack's KOFT linker in place of our motion-relink, LOEO both directions, patched scorer; promote
only on bilateral min-fold ≥ +0.005 against the P3 anchor. **GPU: No.**

### 2.2 Trackastra `ctc` checkpoint (rank 2)

DOCUMENTED. `github.com/weigertlab/trackastra`, **BSD-3-Clause**. Gallusser & Weigert,
*Trackastra: Transformer-based cell tracking for live-cell microscopy*, **ECCV 2024**,
arXiv:2405.15700, doi:10.1007/978-3-031-73116-7_27. Accepts `time,(z),y,x`.

**New fact vs our prior verdict.** `competitive_frontier_2026-08-16.md` filed Trackastra under "H5
pretrained-plugin — likely a DETOUR unless retrained." That judgement was about a 2D-general
drop-in. The model registry (`trackastra/model/pretrained.json`) documents a **third checkpoint**:

```
"ctc": { "dimensionality": [2, 3],
         "description": "For tracking Cell Tracking Challenge datasets. This is the successor of
                         the winning model of the ISBI 2024 CTC generalizable linking challenge.",
         "url": ".../trackastra-models/releases/download/v0.3.0/ctc.zip",
         "datasets": { "All Cell Tracking Challenge 2d+3d datasets with available GT and ERR_SEG" } }
```

So: a **3D-capable, CTC-2D+3D-trained, BSD-3 linking transformer available as one static zip at a
fixed URL**. Download once → attach as a Kaggle dataset → fully offline. Training on custom data is
supported (`python train.py --config …`, CTC format), so this is also a **fine-tune target** for our
2 embryos, not only a drop-in.

Honest caveats:
- Trackastra normally consumes **masks**, not points. Our pipeline emits centroids. Mapping our
  detections to the mask/label input is real work and may be where a naive drop-in previously failed.
- Weights auto-download at runtime → **must be pre-staged** for an internet-off rerun.
- EPFL-CH is #2, not #1, on the CLB — and its BIO (0.794) trails PAST-FR (0.862) by more than its
  LNK does, i.e. its relative weakness is exactly the biological/division half.

**Falsification test.** Stage `ctc.zip` as a dataset; run the 3D model on one LOEO fold's detections
converted to labels; require adj_edge_J ≥ our P3 linker on both embryo directions before any
fine-tuning spend. **GPU: light T4 for inference; T4 for fine-tune.**

### 2.3 Calibrated uncertainty for LAP trackers (rank 3, cheapest of all)

Paul, Seiffarth, Rügamer, Scharr, Nöh, *How To Make Your Cell Tracker Say "I dunno!"*,
arXiv:2503.09244. Abstract (DOCUMENTED): methods "take inspiration from statistics and machine
learning, leveraging two perspectives… as a Bayesian inference problem and as a classification
problem… Our methods admit a **framework-like character** in that they **equip any frame-to-frame
tracking method with uncertainty quantification**… we demonstrate empirically that our methods yield
useful and **well-calibrated** tracking uncertainties." Demonstrated on existing trackers including
transformer-based ones. Repo: UNVERIFIED.

This is the cell-tracking-native sibling of the conformal/MOT-CUP entry in
`novel_crossdomain_2026-08-17.md` #4, and it is more directly applicable because our linker *is* a
linear-assignment-style method. It converges with the AMOLF-NL result in §3.1: **on the closest CTC
dataset, the method that wins does so by being calibrated, not by being bigger.**

**Falsification test.** Fit the classification-view calibrator on embryo-A edges from cached OOF
graphs; on B, check (a) reliability-diagram calibration error and (b) whether thresholding by
calibrated probability beats our current score threshold on adj_edge_J at matched N_pred.
**GPU: No.** ~hours CPU.

---

## 3. Division detection — the only live mechanism left

### 3.1 What the CTC says about divisions

BC(i) = "division event detection efficiency with tolerance of *i* frames"
(`celltrackingchallenge.net/evaluation-methodology/`) — the CTC's analogue of our division-Jaccard.
Two documented regularities from §1.2/§1.3:

- On sparse-cell 3D nuclei (Fluo-N3DH-CE), BC(i) ranges **0.088 → 0.930** across teams whose TRA all
  sit in 0.886–0.994. **Divisions are where methods separate; linking is nearly saturated.**
- On dense light-sheet embryos (TRIC/TRIF), the whole field is **0.00–0.67**, with everyone except
  KTH-SE below 0.50.

Kaiser, Schier, Rosenhahn, *Cell Tracking according to Biological Needs*, **IEEE TMI 2025**,
arXiv:2403.15011, state the field-level diagnosis directly: despite "near-perfect technical benchmark
scores (~100 %), mitosis detection requires further improvement to enable deeper lineage tree
analysis", and "local errors, such as missing segmentations, are penalized more heavily than rare
association errors or missed mitosis detections" — i.e. **the standard metrics under-reward exactly
the thing our metric rewards.** (Their own results are **9 datasets, all 2D** — no 3D — so
BiologicalNeeds is a mechanism source only. Code `github.com/TimoK93/BiologicalNeeds`, **MIT**.)

### 3.2 linajea's cell-state classifier — the mechanism, and its measured effect

Hirsch, Malin-Mayor, Santella, Preibisch, Kainmueller, Funke, *Tracking by weakly-supervised
learning and graph optimization for whole-embryo C. elegans lineages*, **MICCAI 2022**,
arXiv:2208.11467, doi:10.1007/978-3-031-16440-8_3. Code `github.com/funkelab/linajea` (**MIT**).
Journal sibling: Malin-Mayor et al., *Automated reconstruction of whole-embryo cell lineages by
learning from sparse annotations*, **Nature Biotechnology** (2022), doi:10.1038/s41587-022-01427-7.

Mechanism, quoted/paraphrased from the PDF I extracted:
- 4D U-Net predicts **cell candidates + movement vectors**; "the **backwards** direction of the
  movement vectors simplifies tracking as cells can only divide going forward but cannot merge."
- A **3D ResNet18** assigns each candidate one of four classes: **parent** (about to divide),
  **daughter** (just divided), **continuation**, **polar body**.
- The classes enter the ILP **twice**: as weighted costs, and as **hard feasibility constraints** —
  `y_parent,u + y_daughter,u + y_continue,u − y_node,u = 0` (exactly one state per selected node),
  plus `y_parent,u + y_edge,e − y_daughter,v ≤ 1` and `y_daughter,v + y_edge,e − y_parent,u ≤ 1`
  (a selected edge out of a parent *must* land on a daughter, and vice-versa).
- A **structured SVM** (loss-augmented objective, Hamming loss, λ=0.001) auto-tunes the four ILP
  weights instead of a grid search.

Measured effect (Table 1, errors per 1000 GT edges):

| dataset | arm | FPdiv | FNdiv | div sum | DET | TRA |
|---|---|---|---|---|---|---|
| mskcc-confocal | linajea | 0.89 | 0.26 | 1.2 | 0.99514 | 0.99418 |
| mskcc-confocal | **+csc+sSVM** | **0.053** | 0.40 | **0.46** | 0.99570 | 0.99480 |
| nih-ls | linajea | 1.5 | 0.40 | 1.86 | 0.99367 | 0.99279 |
| nih-ls | **+csc+sSVM** | **0.20** | 0.49 | **0.69** | 0.99511 | 0.99433 |

**17× and 7.5× reduction in division false positives**, at the cost of ~+50 % FNdiv — i.e. it buys
division *precision*, which is exactly the currency our break-even analysis is denominated in. On
Fluo-N3DH-CE it took JAN-US to DET 0.981 / TRA 0.979, beating the then-leader ELEPHANT (0.979/0.975).
Compute: "Trained on one machine with one V100 (for anisotropic data a **smaller GPU is sufficient**,
too)"; ILP solve ~15 min, parallelisable; weight search is CPU-parallel.

**Why this is not a re-opening of a closed lever.** `redteam_blindspots_2026-08-17.md` Claim 1 kills
(a) a *conditional pair-ranker* (structurally: MLP's correct set ⊂ geometry's, b=0 on 44b6) and (b) a
*global mother-gate threshold* (0/100 true dividers in the top-100 mothers). linajea's csc is neither:
it is a **two-sided classifier** (the daughter-at-t+1 head is a separate signal from the parent-at-t
head — our own DB-05/06 measured appearance-at-t−1 at LOEO AUC 0.719 *in isolation*) that is
**coupled by hard ILP constraints**, so a division is only accepted when both sides agree *and* the
edge is selected. That is a new mechanism with a stated falsification test, which is the bar CLAUDE.md
sets for reopening.

**Falsification test.** Train a small 3D CNN on cached OOF candidate patches for the 4-class
parent/daughter/continuation/other target on embryo A; on B, measure **precision at 50 % recall
among metric-visible fork candidates** and require it to clear the **4.07 % (44b6) / 6.38 % (6bba)**
break-even from D-07. Predicted by our own arithmetic (base rate ~0.1 %, required AUC 0.97–0.999,
best measured anywhere 0.86–0.92): **still fails.** This is a one-experiment door-closer, not a
programme. **GPU: yes, one small T4 session.**

### 3.3 What I would NOT do on divisions

Re-tuning safe-div caps, global midpoint-residual ranking, suppress-all — all already closed with
mechanisms in our ledger, and nothing in the external literature contradicts those closures. The CTC
BC(i) table (§1.3) is external evidence that the ceiling is genuinely low on this data type.

---

## 4. Named incumbents: verdicts, bluntly

Licences below were verified by me via the GitHub licence API on 2026-08-17.

| Tool | Licence (verified) | Verdict for us |
|---|---|---|
| **Ultrack** (royerlab) | **BSD-3-Clause** | **Runnable, wrong shape.** Nature Methods 22:2423–2436 (2025), doi:10.1038/s41592-025-02778-0. Its CTC wins are OP_CTB = 0.5·(SEG+TRA); my extraction reproduces its reported numbers exactly (worm OP 0.8444, fly 0.7075 vs next 0.6172, beetle-TRIF 0.8406 vs 0.8043) — but **all of its margin is SEG, which our metric does not score.** BC(i) 0.469/0.176/0.442. Needs foreground+contour maps; Gurobi optional, CBC free but docs say "slower, uses more memory and harder to install on Window[s]". **Low EV.** Note `royerlab/tracksdata` (BSD-3, pushed 2026-08-10) is the successor data layer and the likely landing spot for HOCT. |
| **Trackastra** | BSD-3-Clause | **Rank 2.** See §2.2. |
| **ByoTrack** | MIT | **Rank 1.** See §2.1. |
| **linajea** | MIT | **Mechanism, rank 4.** See §3.2. Repo needs `pylp` + MIP solver → reimplement the classifier + constraints against our existing ILP rather than vendoring. |
| **ELEPHANT** | BSD-2-Clause (client & server) | **Mechanism only.** Sugawara, Çevrim, Averof, *eLife* 11:e69380 (2022). Two U-Nets: detection predicts {background, nucleus **centre**, nucleus **periphery**}; linking predicts a **3D displacement field**, then nearest-neighbour after warping — the published ancestor of our arm-B flow gate. **Sparse-annotation evidence is the most relevant thing in this report after §1:** CE1 needed **~483 manually annotated nuclei = ~2 % of 23,829**, and linking trained on **1,162 validated links from 10 timepoints, including only 18 division links**. Trained on a single GTX 1080 Ti. The *software* is a Fiji/Mastodon client + Docker server — unusable inside a kernel. |
| **OrganoidTracker 2.0** | **GPL-2.0** ⚠ (NN files MIT per file headers) | **Mechanism only.** Betjes, Kok, Tans, van Zon, *Cell tracking with accurate error prediction*, *Nat. Methods* 22:2400–2410 (2025), doi:10.1038/s41592-025-02845-6. NN + statistical physics → **per-step error probability** that behaves "like P values"; enables fully automated analysis by retaining only high-confidence segments. **Best TRA (0.994), BC(i) (0.930), CT (0.777), CCA (0.967) on Fluo-N3DH-CE — with the worst SEG in the table.** GPL-2.0 in a $60 k prize kernel is a licence risk; port the idea. |
| **CELLECT** | **GPL-2.0** ⚠ | Nature Methods 22:2411–2422 (2025), doi:10.1038/s41592-025-02886-x, `github.com/zzz333za/CELLECT`. U-Net (2 consecutive frames) → segmentation + centre points + features + size + **division estimate**; 2 MLPs for intra-/inter-frame association; **contrastive** latent embeddings give cross-modality/species generalisation. Weights ship in `./model/`; `--zratio` handles anisotropy (default 5; ours is 1.625/0.40625 = 4.0 — a near match); `--cpu` mode exists. **Technically the most drop-in-able 3D method here, and licence-blocked.** GPL-2.0 obliges source disclosure of derivative works; do not ship it. |
| **KTH-SE / Baxter Algorithms** | CTC "software download: available"; MATLAB | Magnusson et al., *Global Linking of Cell Tracks Using the Viterbi Algorithm*, **IEEE TMI 2015**, doi:10.1109/TMI.2014.2370951. Greedily adds one track at a time, each optimised over **all** frames, so it can revise past decisions using future frames. **The only method above 0.5 BC(i) on dense light-sheet embryos.** MATLAB → do not port wholesale; the reimplementable idea is *iterative global track addition with re-optimisation* rather than per-frame greedy. |
| **TrackMate** | **GPL-3.0** 🚫 | Ershov/Tinevez et al., *Nat. Methods* 19:829–832 (2022). Java/Fiji. GPL-3 + JVM in a Kaggle kernel = no. Ultrack reports beating it 0.9951 vs 0.9895 F1 on identical Cellpose segmentations. |
| **Mastodon / MaMuT** | BSD-2-Clause | Java/Fiji GUI for large lineage curation. It is the CLB *baseline*, not a competitor. Not runnable in a kernel. |
| **btrack** | MIT | Ulicna et al., *Front. Comput. Sci.* 3:734559 (2021). Bayesian MOT, C++ core + Python. 2D-centric cell-culture lineages; no 3D+t embryo CTC standing. **No EV.** |
| **LapTrack** | BSD-3-Clause | Fukai & Kawaguchi, *Bioinformatics* 39(1):btac799 (2023). Tunable-metric LAP tracking. This is the *baseline family* we already implement. **No EV.** |
| **EmbedTrack (KIT-Loe-GE)** | git.scc.kit.edu (licence UNVERIFIED) | Löffler & Mikut, *IEEE Access* (2022), arXiv:2204.10713. **2D only** ("nine 2D datasets"). **No EV for 3D.** |
| **BiologicalNeeds (LUH-GE)** | MIT | TMI 2025, arXiv:2403.15011. **9 datasets, all 2D.** Mechanism source for the Erlang mitosis cost (already logged in `novel_crossdomain_2026-08-17.md` #5); no 3D evidence. |
| **PAC-MAP** | **CC BY-NC-SA 4.0** 🚫 | Verified from the repo LICENSE file (opens "Attribution-NonCommercial-ShareAlike 4.0 International"). *Comput. Biol. Med.* 185 (2025); bioRxiv 2024.07.18.602066; weights on Zenodo 14138806. Mechanism — proximity-adjusted centroid probability map, weak pretrain on baseline labels then finetune on manual, "boosting recall, especially in conditions of high cell density" — is a very good fit for our dense-nuclei detection problem. **The licence, not the science, is the blocker.** Reimplement the proximity-weighted target from the paper if we want it. |
| **CellTracksColab** | MIT | *PLOS Biology* (2024), doi:10.1371/journal.pbio.3002740. Post-hoc analysis/exploration of tracking data. **Not a tracker. No EV.** |
| **TGMM** (Keller lab) | SourceForge, legacy | Amat et al., *Nat. Methods* 11:951–958 (2014), doi:10.1038/nmeth.3036. Gaussian-mixture nuclei + sequential Bayesian propagation; 20k cells/timepoint at 26k cells/min. Historic and important, but a 2014 C++/SourceForge codebase. **No EV as code**; its descendants (linajea, ELEPHANT) supersede it. |
| **DaXi** | repo NOASSERTION | Yang, Lange, Millett-Sikking, …, Royer, *Nat. Methods* 19:461–469 (2022), doi:10.1038/s41592-022-01417-2. **This is a MICROSCOPE, not an algorithm** — 450 nm lateral / 2 µm axial over 3000×800×300 µm. Any "DAXI weights" in our notes are a model *trained on DaXi data*, not DaXi itself. Correct the framing in our ledger. |
| **ARGUS** | arXiv:2607.08297 (2026-07) | Adaptive detection + dense Farnebäck optical flow + LAP + tracklet refinement. DET 0.905–0.971 / TRA 0.897–0.964 on public CTC data — i.e. **below** the incumbents in §1.2. Unsupervised and fast. **No EV** except as further evidence that flow-driven association is the field's converged prior. |

---

## 5. Zebrafish / Zebrahub specifically

- **Zebrahub.** Lange, Granados, VijayKumar, …, Royer, *A multimodal zebrafish developmental atlas
  reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors*, **Cell**
  187 (2024), doi:10.1016/j.cell.2024.09.047, PMID 39454574. Light-sheet lineage reconstructions
  over the first ~24 hpf plus scRNA-seq at 10 timepoints. **The lineages were produced by Ultrack**
  (documented in the Zebrahub companion-paper listing and CZ Biohub's own release). **Implication we
  should internalise: the Zebrahub `*_tracks.csv` we are treating as external supervision are
  ULTRACK OUTPUT, not human ground truth** — and §1.2 shows Ultrack's BC(i) on the closest CTC
  dataset is 0.469. Training a division head on Zebrahub tracks risks distilling Ultrack's division
  errors. Zebrahub data licence: **UNVERIFIED** (not stated on the site pages I could fetch); host
  permission for our use is separately documented (#734330).
- **Ultrack on zebrafish, quantified** (Nature Methods 2025 / PMC12615266): DaXi recordings of
  1.7–3.7 TB over 8.6–13.2 h; "on average, it took **62 frames** for a tracking error to appear in
  50 % of the lineages"; validation by stratified sampling of ~140 lineages per embryo (n=3). Sparse-
  labelling validation on a zebrafish embryo used **152 annotated tracklets** spanning 85–521 frames,
  sum error rate 0.049 (DL) / 0.070 (classical) at 150 frames. Neuromast (71 nuclei, 500 frames):
  TRA 0.9989, F1 0.9951 vs TrackMate 0.9895. A 3.7 TB dataset took ~8.2 h on HPC.
  **Read: even the organisers' own zebrafish pipeline is validated against ~150 sampled lineages,
  not dense truth.** Sparse validation is the norm in this exact system.
- **No zebrafish-specific *tracking algorithm* exists** beyond "Ultrack tuned for DaXi/Zebrahub."
  Searches surfaced only application papers (3D nnU-Net for regional deformation in zebrafish,
  bioRxiv 2024.11.04.621759) and imaging-restoration work. **This is a real gap and it is ours to
  exploit: there is no published zebrafish nuclear-lineage method that beats the organisers' own
  general-purpose tool.**

---

## 6. Documented hard failure modes in this regime

1. **Dense nuclei / crowding.** Ultrack's own limitations section: it "may fail when presented with
   an overwhelming number of incorrect segmentations or systematic errors that persist over time",
   and performance degrades as "cell density and imaging artifacts" increase over long development.
   PAC-MAP's entire premise is that recall collapses under high density. Our own scorer-side fact
   (nearest neighbours ~9–10 µm apart vs a 7 µm one-to-one match radius, `sleepymegacat` fact 7) is
   the same phenomenon expressed in the metric.
2. **Divisions.** §3.1. The field's BC(i) on dense light-sheet embryos is 0.00–0.67. Kaiser et al.
   state plainly that standard metrics under-penalise missed mitoses, which is why the field's
   division performance has been allowed to stay this bad. **Our metric does penalise them, so this
   is a place where the literature's incumbents are weakly optimised for our objective.**
3. **Sparse annotation.** Not a pathology in this field — a norm, and a solved-enough one.
   CTC's own DRO/TRIC/TRIF are partially annotated (nervous-system cells only; blastoderm lineages
   only). ELEPHANT reaches DET 0.979 / TRA 0.975 from ~2 % annotated nuclei. linajea trains from
   "a small set of nuclei **center point** annotations". **Documented: 2–3 % point annotation is
   sufficient to train a competitive 3D+t nuclear tracker.** Our 2.8 % is not the binding constraint;
   *2 embryos* is.
4. **Low SNR deep in the sample.** Documented as an imaging problem attacked with imaging solutions:
   scattering and refractive-index variation reduce contrast and SNR with depth in LSFM; deep-learning
   restoration (e.g. CNN-transformer UI-Trans, *Light Sci. Appl.* 2024, doi:10.1038/s41377-024-01710-z)
   is the standard remedy. I found **no** tracking paper that quantifies a depth-vs-accuracy curve on
   zebrafish nuclei. **UNVERIFIED whether depth is a first-order term in our error budget — this is
   measurable on our own data in an hour and nobody has done it.**
5. **Cross-embryo / cross-domain generalisation.** The CTC's answer is the generalizability track:
   PAST-FR 0.984 and EPFL-CH 0.977 LNK across 13 heterogeneous datasets with **one** model. That is
   the strongest external evidence that a single linker can hold up across domains — but note both
   are *linking-only over standardised inputs*, so it says nothing about detector transfer.
   Ultrack's failure mode "large cell movements between adjacent frames violate our assumption of
   segmentation consistency… unless registration is used" is the concrete cross-family risk for us.

---

## 7. Actionable levers — each with a one-line falsification test and GPU flag

| # | Lever | Falsification test (one line) | GPU? |
|---|---|---|---|
| L1 | **KOFT: flow-as-velocity-measurement inside a Kalman filter**, replacing our memoryless motion-relink (generalises arm B) | Replay `p0strict_cache/graphs` through ByoTrack KOFT vs our relink, LOEO both directions, patched scorer; kill unless bilateral min-fold ≥ +0.005 | **No** |
| L2 | **Trackastra `ctc` 3D checkpoint** as an alternative linker / second opinion | Stage `ctc.zip` offline, convert one fold's detections to labels, score adj_edge_J vs P3 on both directions; kill if either direction loses | Light T4 |
| L3 | **Calibrated per-link uncertainty** (Paul et al. classification view) as the acceptance score | Fit on A, evaluate on B: kill unless calibration error < 0.05 **and** thresholding on calibrated p beats our current threshold at matched N_pred | **No** |
| L4 | **linajea 4-class cell-state head + hard ILP parent/daughter constraints** | Train 4-class head on A's cached candidate patches; on B require precision ≥ 4.07 %/6.38 % at 50 % recall among metric-visible forks; kill otherwise | Yes, 1 small session |
| L5 | **KTH-SE global-Viterbi track addition** (iterative, all-frames re-optimisation) instead of per-frame greedy | On cached graphs, add tracks one at a time with full-sequence re-optimisation; kill unless adj_edge_J gains ≥ +0.003 bilaterally over the greedy baseline | **No** |
| L6 | **ELEPHANT-style centre+periphery detection target** (3-class, not binary) to sharpen centroids | Retrain the detector head with a centre/periphery/background target; kill unless sub-voxel centroid σ drops below the 1.5–2 µm cliff **and** LOEO improves | Yes |
| L7 | **Depth-conditioned error audit** (untested anywhere in the literature for this system) | Bin our LOEO edge errors by z-depth and local density; kill the whole idea if the top and bottom depth quintiles differ by < 10 % relative error | **No**, ~1 h |
| L8 | **Do NOT distil Zebrahub division labels naively** — they are Ultrack output whose BC(i) is ~0.47 on the closest benchmark | Before any Zebrahub division training: measure agreement between Zebrahub-track divisions and our own annotated divisions on any overlapping regime; abort if precision < 0.7 | **No** |

---

## 8. Licence flags (all verified by me on 2026-08-17 via the GitHub licence API or the LICENSE file)

| 🚫 Blocked | ⚠ Risky | ✅ Clear |
|---|---|---|
| **PAC-MAP — CC BY-NC-SA 4.0** (non-commercial; incompatible with a $60 k prize) | **CELLECT — GPL-2.0** (copyleft on derivatives) | ByoTrack — MIT |
| **TrackMate — GPL-3.0** (also JVM-bound) | **OrganoidTracker — GPL-2.0** (NN files MIT per-file; mixed) | Trackastra — BSD-3 |
| | **HOCT — CC-BY-NC-ND 4.0** (already flagged in `methods_frontier_2026-08-16.md`) | linajea — MIT; BiologicalNeeds — MIT; btrack — MIT; CellTracksColab — MIT |
| | **EmbedTrack — licence UNVERIFIED** (GitLab, not GitHub) | Ultrack, LapTrack, tracksdata, `royerlab/kaggle-cell-tracking-competition` — BSD-3 |
| | **DaXi repo — NOASSERTION** (no licence declared) | ELEPHANT client & server, Mastodon — BSD-2 |

---

## 9. What I could not verify (stated, not guessed)

- **The CTC 10-years paper's actual conclusions** on generalizability/reusability and on what
  separates top from mid methods. Paywalled behind four independent walls. Only the abstract-level
  claims in §1.5 are documented.
- **CTC results newer than 2025-08-15** (CTB) / **2025-01-31** (CLB). Anything submitted in the last
  ~12 months is invisible to this report.
- **How PAST-FR/ByoTrack produces divisions** despite advertising no division model (BC(i)=0.838).
- **AMOLF-NL = OrganoidTracker 2.0** and **THU-CN (3) = CELLECT** — both are high-confidence
  INFERENCES; the CTC participant pages carry no description for AMOLF-NL and do not list THU-CN (3).
- **Zebrahub data licence** and the exact provenance/curation level of `*_tracks.csv`.
- **Repo for arXiv:2503.09244** (UQ for LAP trackers).
- **Depth-vs-accuracy behaviour** for nuclear tracking in zebrafish light-sheet — no paper found.

---

## 10. Bottom line

1. **We have been reading the wrong column of the CTC leaderboard.** OP_CTB is half segmentation;
   our metric has no segmentation term. On TRA/BC(i)/LNK — the columns that match our objective —
   the incumbents are AMOLF-NL, JAN-US, IGFL-FR, PAST-FR and EPFL-CH, **not** Ultrack and not the
   min-cost-flow segmentation methods.
2. **The organisers' own tool is division-weak** (Ultrack BC(i) 0.469 / 0.176 / 0.442). Our
   division-Jaccard ≈ 0 is the field norm on this data type, and the external ceiling is ~0.67,
   set by a classical global-Viterbi method, not by a transformer.
3. **The two methods worth actually running are MIT/BSD-3, pip-installable, CPU-capable and 3D:**
   ByoTrack SKT/KOFT (CLB #1) and Trackastra's `ctc` 2D+3D checkpoint (CLB #2). Neither has ever
   been applied to dense light-sheet embryo linking — **the CLB has zero entries on DRO/TRIC/TRIF.**
4. **KOFT is the principled version of our own best un-shipped lever.** Arm B's measured +0.008
   bilateral is a one-step gate; KOFT is flow-as-velocity inside a Kalman state. Cheapest high-EV
   action on the board, CPU-only.
5. **Sparse annotation is not our binding constraint.** ELEPHANT hit DET 0.979 / TRA 0.975 from ~2 %
   annotated nuclei; linajea trains from centre points alone. **Two embryos is the constraint** — and
   the external literature's answer to that is calibration and uncertainty (AMOLF-NL, Paul et al.),
   not more capacity.
6. **Two licence traps to avoid:** PAC-MAP is CC BY-NC-SA (hard block); CELLECT and OrganoidTracker
   are GPL-2.0 (do not ship in a prize kernel — port mechanisms only).
7. **One correction to our ledger:** the Zebrahub `*_tracks.csv` are **Ultrack output**, not human
   ground truth. Treat them as a noisy teacher, especially for divisions.
