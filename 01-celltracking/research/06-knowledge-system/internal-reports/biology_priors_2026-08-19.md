# Developmental biology of the specimen as a source of tracking priors — 2026-08-19

Scope: nine levers have died and all nine were computational. This report asks the question
nobody has asked: what does the developmental biology of *this* specimen imply about *this*
tracking problem, and which of those implications is a usable prior or constraint.

Labels: **[DOC]** = documented biology, cited. **[MEAS]** = measured this session against
`data/train/*.geff` + `*.zarr` (199 crops, 133,318 GT nodes, 128,883 GT edges, 151 divisions).
**[INF]** = inference, assumptions stated. **[NULL]** = tested and found absent — do not pursue.

Everything under **[MEAS]** was computed from ground truth only; no GPU, no submission, no commit.

---

## 1. Ranked exploitable priors

Ranked by (expected score impact) x (confidence) x (cheapness of test).

| # | Prior | Biological basis | Measured status | How to test | How it enters the pipeline |
|---|---|---|---|---|---|
| **1** | **Per-frame global rigid drift.** A single translation vector per (crop, frame) absorbs most nuclear motion. | Coherent tissue flow: epiboly margin 3.3 um/min [Hernández-Vega 2017]; deep cells ~1 um/min [Kobitski 2015]; PSM 0.75 um/min [Yin 2008]. Velocity correlation length in gastrula is unmeasured, but tailbud is 20–150 um — **at or beyond our 104 um field**. | **[MEAS]** Removing the per-frame mean displacement explains **55% (44b6) / 65% (6bba)** of displacement variance. 6bba median 1.82→1.14 um, p90 4.16→2.42 um. **Global beats kNN local flow at every k** (kNN k=12: 40%/59%). | Done — see §4.1. Re-run per fold before deploying. | Estimate drift **correspondence-free** from detections (mean-shift mode of all candidate offsets < 8 um), subtract before computing linker edge costs. **Validated GT-free**: under 50% detection recall + 100% decoys the estimate still has cos +0.90 with truth and removes 37–43% of variance (§4.2). |
| **2** | **Division refractory period.** A cell that just divided cannot divide again for K frames. | Post-MBT cycle floor: cycle 13 = 54 min, cycle 14 = 78 min, cycle 15 = 151 min, cycle 16 = 240 min [Kane 1999]. Only pre-MBT cleavages are ~15 min [Kane & Kimmel 1993]. | **[MEAS]** **0 of 302 GT daughters re-divide**, over **5,648 daughter-frames at risk**. Poisson expectation at the observed base rate = 6.40. **p = 0.0017**. Longest daughter followed: 95 frames. | Already done (pooled p=0.0017). Falsification: any single GT lineage with two divisions kills it. | **Hard constraint in the ILP/linker**: forbid a division hypothesis within K frames downstream of an accepted division. Recommend **K = 40** (60 min at 90 s/frame — safe from gastrulation on [DOC]); GT supports K up to 95. Zero measured cost, removes a whole class of FP. |
| **3** | **Division-count operating point.** Expected divisions per crop = 1.13e-3 x `estimated_number_of_nodes`. | Cell cycle at cycle 15–16 is 151–240 min and first postmitotic cells appear at 90% epiboly [Kimmel 1995; Kane 1999] — division is **rare**, not frequent. | **[MEAS]** Rate = **1.29e-3 /cell/frame (44b6)**, **1.11e-3 (6bba)** — statistically indistinguishable. Implies **~42 divisions/crop (44b6)**, **~11 divisions/crop (6bba)**; **0.42 vs 0.11 divisions per frame**. | Compare the current division lane's predicted count per crop against these. `division_lane_2026-08-18.md` reports **5 TP / 613 FP / 146 FN** — the FP count alone is ~10x the biologically expected total. | Calibrate the division threshold **per crop** so predicted division count ≈ 1.13e-3 x N_est, rather than by a global probability cutoff. This is a per-crop normaliser the pipeline does not currently have. |
| **4** | **Inter-daughter distance window + monotone separation.** | Daughters are ~1 nuclear diameter apart at cytokinesis and separate at ~7–9 um/min during anaphase/telophase — an order of magnitude above interphase migration (D = 0.09 um^2/min) [Azizi 2020; Krzic 2012]. | **[MEAS]** Inter-daughter median **8.98 um (44b6) / 11.47 um (6bba)**, p05 4.53/5.72, p95 12.13/15.79. Separation then grows **monotonically**: 8.98→11.89→13.03→…→19.66 um over 10 frames (44b6); 11.47→14.10→15.30→…→19.19 (6bba). | Done. | Two gates on every division hypothesis: (a) inter-daughter distance in **[4, 16] um**; (b) **require separation to increase over the following 2–3 frames**. Gate (b) is free — the linker already has those frames — and no other object class produces monotone separation from a common origin. |
| **5** | **Post-division nuclear volume dip and recovery.** | Chromatin condenses 2–3x at mitotic entry [Martin & Cardoso 2010]; a newborn nucleus then re-expands (long semi-axis 3.9 um at 10 min → 4.8 um at 20 min) [Azizi 2020]. | **[MEAS]** Daughter half-max nuclear volume is **~52–58% of the interphase local control** at k=0–2 and recovers over **~7–10 frames** (6bba: 294→231→…→342 um^3 vs control 567; 44b6: 348→283→…→369 vs 484). Novel — unpublished quantity (§3.3). | Done. | A nucleus at <70% of the local median volume is **recently born**: (a) suppress new division hypotheses on it (reinforces prior #2 with an image-side signal that needs no lineage history); (b) do not raise the detection threshold on it — small dim nuclei near a recent division are real, and this is a named FN mode. |
| **6** | **Division symmetry and H2B conservation.** | Anaphase segregates sister chromatids equally about the spindle equator, so the two chromatin masses are equal-mass by construction; mitotic chromosome volume tracks DNA content [Puah 2017]. | **[MEAS]** daughter/daughter integrated-intensity ratio **0.79 (44b6) / 0.76 (6bba)** median; sum(daughters)/parent **1.63 / 1.30**. Positional symmetry min/max parent-daughter distance = **0.62** median. Novel — the sum-ratio is explicitly unpublished for embryonic light-sheet. | Done. | Feature on the division-hypothesis scorer: penalise daughter pairs whose intensity ratio < 0.5 or whose sum-ratio is outside ~[0.6, 2.2]. Weaker than #4; use as a tiebreaker, not a gate. |
| **7** | **Mitotic elongation prefilter.** | Prophase condensation → metaphase plate (disc) → anaphase (two-lobed): an anisotropy signature [Martin & Cardoso 2010; Gilad 2019; Turley 2024]. | **[MEAS] — weak.** Best single feature is elongation (l1/l3), **AUC 0.676 (44b6) / 0.687 (6bba)** for pre-division (t-1, t-0) vs interphase. At 10% FPR: recall 0.27–0.33, **~3x enrichment**. Elevated over roughly **t-4 … t+8**, not just at t. | Done, §3.4. | Only as a **candidate prefilter / input feature**, never a detector. A 3x enrichment against a 1.1e-3 base rate is a real search-space reduction but nowhere near sufficient alone. See the honest explanation in §3.4 — at 90 s/frame most of the classic signature is sub-frame. |
| **8** | **Boundary-flux prior (embryo-specific).** | 6bba crops sample a thin, motile blastoderm sheet; 44b6 samples denser interior tissue (§5). | **[MEAS]** Interior track starts/ends: **2.88%/2.90% of nodes in 6bba vs 0.95%/1.22% in 44b6 (3x)**. 6bba GT nuclei sit at median **12.2 um** from a crop face vs 22.8 um; **67–68% of 6bba termini are within 10 um of a face.** | Done. | Set the appearance/disappearance cost **per embryo, not globally**. 6bba needs ~3x the track-initiation budget of 44b6. A single global birth/death prior is mis-specified for one of the two embryos by a factor of three. |

### Negative results — do not spend cycles here

| **[NULL]** | Finding |
|---|---|
| **Global division synchrony** | Divisions are **uniform over the 100-frame window** (44b6 deciles `[2,4,3,1,2,3,7,3,0,1]`, 6bba `[9,14,17,14,14,18,14,8,12,5]`). Within-crop temporal clustering is at best marginal (6bba obs mean \|dt\| 26.8 vs null 32.8, **p=0.0155**; 44b6 p=0.27). **[DOC]** explains why: embryo-scale division waves **cease at cycle 11 (~3.0–3.3 hpf)** [Kimmel 1995], far before our stage. **The famous zebrafish cleavage synchrony is not available to us.** |
| **Mitotic domains (spatial clustering)** | Co-timed division pairs (\|dt\| <= 3, same crop) have median separation **53.7 um vs a null of 53.5 um**. n=8 pairs — underpowered, but no hint of an effect. |
| **Pre-division kinematic slowdown** | Parent centroid speed over the 6 frames before division is flat (44b6 1.72/1.82/1.37/1.70/1.82/1.75 um/frame; 6bba 1.82/2.07/1.99/1.86/2.03/1.86). **There is no motion signature of an about-to-divide nucleus** — any pre-division signal must come from appearance. |
| **kNN neighbourhood-flow prior** | Strictly **worse** than a single global translation at every k tested (3, 5, 8, 12). See §4.1 — the flow has no resolvable local structure inside 104 um. |

---

## 2. What developmental stage is this?

### 2.1 Staging

**[DOC]** Kimmel et al. 1995 contains **no table of blastomere or nuclear diameters** — it stages by
tier counts and morphology. Diameter cannot be inverted to stage from it. The usable published
anchor is nuclei-based and stage-matched:

> **[DOC]** "We used the median value **d = 16.1095 um** as an estimate for the reference cell
> diameter", from nearest-neighbour distances over 12 embryos, sphere → 50% epiboly (4.0–5.25 hpf);
> nucleus tracking diameter **7 um**. — Bensch et al. 2013, *Biology Open* 2(8):845–854,
> doi:10.1242/bio.20134614.

**[INF]** Under reductive cleavage (blastoderm volume ~conserved, cell number doubling ⇒ spacing
∝ 2^(-n/3)), our 9.9 um nearest-neighbour spacing sits `(16.11/9.9)^3 = 4.31` ⇒ `log2(4.31) = 2.11`
cycles beyond sphere/dome. Kimmel places sphere/dome at cycle 13→14, so this puts us at
**cycle 15–16**, i.e. shield (6 hpf, "many DEL cells are beginning cycle 15") through 90% epiboly
(9 hpf, "many DEL cells are in cycle 16, and the earliest postmitotic cells are present").

**[MEAS]** The division rate pushes later. Observed **1.13e-3 divisions/cell/frame** ⇒ only
**11–13% of cells divide across the whole 100-frame window**. At 90 s/frame that window is 150 min.
If every cell were cycling at cycle 15 (151 min), essentially *all* of them would divide. The
observed 11–13% implies an effective cycle **~8–20x longer** than cycle 15 — i.e. a largely
**postmitotic or very long-cycle population**, which is the segmentation period, not gastrulation.
(Some of this gap is annotation incompleteness — GT is 2.8% sampled and heavily fragmented — so
treat 1.13e-3 as a lower bound on the true rate.)

**[INF] Best joint estimate: late gastrula through segmentation, ~8–20 hpf.** Spacing gives a
lower bound (>= cycle 15–16) because cell-size reduction stalls once cells exit the cycle; the low
division rate gives an upper-bound-shifting push toward segmentation. This is consistent with
**[DOC]** Ultrack Methods, which states that the DaXi zebrafish sessions "usually cover a
developmental window from 1 or 2 somites to 27 somites" (≈10.3–22.5 hpf).

**Do not stage from density.** **[MEAS]** (104 um)^3 = 1.125e6 um^3. At the measured 9.9 um NN
spacing, a space-filling field would hold ~1,160 nuclei. We observe **327/frame (44b6)** and
**97/frame (6bba)** — **28% and 8% tissue occupancy**. The cube is mostly yolk and exterior; the
volumetric density measures the yolk fraction, not cell size.

### 2.2 Frame interval — three independent estimates converge on 60–90 s

| Method | Estimate | Basis |
|---|---|---|
| **Instrument metadata** **[DOC]** | **90 s** | Ultrack Methods, "Simultaneous sparse and ubiquitous labels imaging": "The imaging volume comprised 493 z-planes with **1.625-um z-steps**. Images were captured at **90-s intervals**, starting at the **shield stage (6 hpf)**, and ending after **24 hpf**." **Our deployed z-step is exactly 1.625 um.** The DaXi/ZSNS Zebrahub lineage is 1.24 um z at 60 s and does **not** match. |
| **Displacement** **[INF]** | 60–120 s, best 90 s | Drift-corrected residual **1.14 um/frame** ⇒ 0.76 um/min at 90 s, which equals the published medial-PSM total speed of **44.8 ± 8.5 um/h = 0.75 um/min** [Yin 2008]. Raw 1.8 um/frame ⇒ 1.2 um/min = the V0 used for zebrafish PSM cells [Uriu & Morelli 2014]. |
| **Post-division nuclear regrowth** **[MEAS]+[INF]** | **53–70 s** | Daughter nuclear volume grows 231→332 um^3 over 5 frames (6bba) and 283→393 over 6 frames (44b6). Anchoring on Azizi 2020 (long semi-axis 3.9 um at 10 min → 4.8 um at 20 min ⇒ volume x1.86 per 10 min, rate 0.0623/min) gives **70 s (6bba)** and **53 s (44b6)** per frame. Azizi is retina at a different temperature, so this is a soft anchor. |

**[INF] Adopt 90 s/frame** (documented, exact geometry match), with 60 s as the plausible
alternative. **The 100-frame window is 100–150 min.** This is the number every time-based prior
below is converted through, and it is the single most load-bearing inference in this report.

### 2.3 Cell cycle and synchrony at this stage

**[DOC]** Cycle lengths (Kane 1999, *Methods Cell Biol* 59:11–26, as quoted in Zhang et al. 2008):
cycle 13 = **54 min**, cycle 14 = **78 min**, cycle 15 = **151 min**, cycle 16 = **240 min**. The
15-minute oscillator governs only cycles 1–9 [Kane & Kimmel 1993].

**[DOC]** Synchrony — the sharpest published statements, all from Kimmel et al. 1995:
- Cycle 10 (512-cell, 2.75 hpf): "one can still find **a minute or so** of time when all cells
  … are in mitosis … The 512-cell stage is the last cycle when this is possible."
- Cycle 11 (1k-cell, 3.0 hpf): "the last one to [pass as a discernable wave] … for the first time,
  many of the cells can be seen to be out of phase with their neighbors."
- High stage (3.3 hpf): "In any region of the blastodisc throughout this whole stage there are some
  cells in interphase and others in mitosis."

**[DOC]** No published sigma (minutes) for a post-MBT division wave exists, and no published
autocorrelation of division timing exists. This is a real gap in the literature, not a search
failure.

**[MEAS]** **We tested for synchrony and it is absent** (see the [NULL] table). This is the correct
prediction from [DOC]: embryo-scale waves end at cycle 11 and we are at cycle 15+.

**[DOC]** One residual hope, not tested here: "Cells in single clones divide very synchronously"
through cycles 15 and 16 [Kimmel, Warga & Kane 1994, *Development* 120:265–276], and post-MBT
division timing carries a deterministic **radial** structure driven by asymmetric division and the
N/C ratio [Mishra et al. 2026, *Nat Phys*, doi:10.1038/s41567-025-03122-1]. **[INF]** So a
*lineage-conditioned* or *radially-conditioned* division prior may survive even though a global one
does not — but with 151 divisions total and no clone labels, we cannot test it on this data. Low
priority.

---

## 3. Division mechanics

### 3.1 Is 8.98 / 11.47 um "just divided"?

**Yes.** **[DOC]** Freshly born daughters are ~1 nuclear diameter apart; nuclear long semi-axis is
3.9 ± 0.5 um at 10 min post-division [Azizi 2020, *eLife* 9:e58635] ⇒ ~8 um diameter. Interphase
nuclear motion is far too slow to manufacture the distance any other way: deep-cell migration
~1 um/min [Kobitski 2015] and the diffusive component is D = 0.09 ± 0.05 um^2/min [Azizi 2020].
Cell-cell spacing at blastula is 16.1 um [Bensch 2013]; relaxed daughters approach *that*, and we
are well below it.

**[MEAS]** Our separation curve settles the question directly: separation rises **fastest in the
first frame** (8.98→11.89, +2.9 um; 11.47→14.10, +2.6 um) and then decelerates sharply
(+1.1, +1.2 um at k=2), asymptoting near **19–20 um ≈ 2x the NN spacing** by k=10. Catching the
fast phase means the first two-daughter frame is **within ~1 frame of cytokinesis**.

**[INF]** The 44b6/6bba difference (8.98 vs 11.47 um) is ~1 frame of separation velocity, or the
6bba cell-size difference — not a difference in elapsed time worth modelling.

### 3.2 Mitotic duration vs our frame interval — why the appearance signature is weak

**[DOC]** Whole mitosis in the zebrafish embryo: **~15 min** at cleavage/blastula [Adar-Levor 2021,
*PNAS* 118(15):e2021210118]; **~25 min** in 24 hpf neural tissue [Percival & Parant 2016, *JoVE* 113].
But the *early* sub-phases are sub-minute: at the 1000-cell stage, **NEBD → metaphase plate takes
~37.5 s** [Oda et al. 2023, *Biology Open* 12(5):bio059783].

**[INF]** At 90 s/frame, prophase condensation and metaphase are each **≤1 frame**. The classic
"bright compact prophase → metaphase disc → anaphase dumbbell" progression is largely **sub-frame**.
The realistic observation is: interphase nucleus → one ambiguous frame → two daughters. This is
the mechanistic explanation for §3.4's weak result, and it is a reason to stop expecting a strong
single-frame mitotic classifier from this data.

**[DOC]** Also worth knowing: cytokinesis is not the end of physical connection. Abscission begins
only at cycle 10, and intercellular bridges persist **~20 min for cycles 10–13** [Adar-Levor 2021].
Two resolved nuclei appear well before the cells are independent.

### 3.3 Conservation through division — novel measurements

**[DOC]** The parent enters mitosis at 4C DNA and each daughter receives 2C, so integrated H2B
should be conserved; but the free histone pool grows ~29.5 ± 5.7% per nuclear cycle [Puah 2017,
*Biology Open* 6:390–401] and disperses at NEBD, so exact conservation is not expected. **[DOC]** No
embryonic light-sheet quantification of parent-vs-daughter integrated H2B intensity has been
published.

**[MEAS]** Measured here, paired per event, half-max segmentation in physical units:

| Quantity | 44b6 (n=18) | 6bba (n=88) |
|---|---|---|
| sum(daughters) / parent, integrated intensity | **1.63** (IQR 1.08–1.84) | **1.30** (IQR 0.88–1.66) |
| min/max daughter integrated intensity | **0.79** (IQR 0.64–0.85) | **0.76** (IQR 0.61–0.89) |
| sum(daughters) / parent, volume | 1.53 | 1.33 |
| min/max daughter volume | 0.66 | 0.74 |

**[INF]** The sum-ratio exceeding 1.0 is expected: the parent's last frame is a condensed
anaphase/telophase object whose half-max mask captures less signal than two re-expanding daughter
nuclei. The **daughter/daughter ratio (~0.77) is the better-conditioned feature** — both daughters
share the same segmentation regime and the same free-pool bias — which matches the design choice in
[Gilad et al. 2019, *Bioinformatics* 35(15):2644–2653].

### 3.4 Appearance signature — measured, and honestly weak

**[MEAS]** Half-max nuclear morphometry, pre-division parent (t-1, t-0) vs same-frame interphase
controls in the same crop:

| Feature | 44b6 AUC | 6bba AUC | Direction | Ratio to control (t-1) |
|---|---|---|---|---|
| **elongation (l1/l3)** | **0.676** | **0.687** | higher | 1.34 / 1.33 |
| smallest axis l3 | 0.661 | 0.669 | lower | 0.73 / 0.78 |
| volume | 0.563 | 0.647 | lower | 0.75 / 0.67 |
| peak intensity | 0.647 | 0.584 | inconsistent across embryos | 0.82 / 1.21 |
| integrated intensity | 0.596 | 0.525 | weak | 0.64 / 0.96 |

Operating points on elongation (threshold at the control percentile):

| FPR | 44b6 recall | 6bba recall | enrichment |
|---|---|---|---|
| 0.50 | 0.80 | 0.75 | 1.5–1.6x |
| 0.25 | 0.52 | 0.49 | 2.0x |
| **0.10** | **0.27** | **0.33** | **2.6–3.3x** |
| 0.05 | 0.23 | 0.18 | 3.7–4.3x |

**[MEAS]** Elongation is elevated across roughly **t-4 … t+8**, not sharply at t: parent elong
2.0–2.2 at t-4…t-1, and daughters decay 2.35→1.8 over 10 frames toward the control value of 1.6.
**[INF]** So this is a broad "peri-mitotic" state, ~10 frames wide on each side, consistent with the
§5 volume-recovery timescale — not a crisp single-frame metaphase detector, exactly as §3.2 predicts.

**[DOC]** For reference, no published intensity ratio or elongation distribution for mitotic vs
interphase nuclei in an embryonic H2B light-sheet dataset exists; the only hard published number is
the **2–3x chromatin volume drop at mitotic entry** [Martin & Cardoso 2010, *FASEB J*]. We measure
only **1.3–1.5x** at half-max, which is expected — half-max under-detects condensation because the
peak rises as the volume falls.

**[INF] Verdict: use as a feature, not a lever.** A 3x enrichment against a 1.1e-3 base rate is
worth having in a candidate generator, but hand-crafted morphometry will not carry a division lane.
The temporal encoding used by [Turley et al. 2024, *eLife* 12:RP87949] — stacking 5 timepoints as
channels, Dice 0.964 vs 0.752 for 3 frames — is the design that actually works, and our §3.1 result
(monotone separation over 2–3 frames) says the discriminating information is genuinely temporal.

---

## 4. Tissue-scale motion

### 4.1 The flow is a bulk translation, and it is huge

**[MEAS]** Velocity coherence vs pair separation (mean cosine between unit displacement vectors,
same crop and frame):

| Separation | 44b6 | 6bba |
|---|---|---|
| 5–10 um | +0.386 | +0.434 |
| 15–20 um | +0.371 | +0.490 |
| 30–40 um | +0.321 | +0.459 |
| **60–200 um** | **+0.287** | **+0.388** |

Coherence **barely decays across the entire 104 um field**. **[INF]** The velocity correlation
length exceeds our field of view, so within a crop the flow has no resolvable local structure —
it is a rigid translation plus noise. This is confirmed by the decomposition:

| Model | 44b6 var explained | 6bba var explained |
|---|---|---|
| **Global per-frame translation** | **55.0%** | **64.8%** |
| kNN local flow, k=3 | 36.3% | 52.1% |
| kNN local flow, k=8 | 44.6% | 57.9% |
| kNN local flow, k=12 | 40.3% | 58.7% |

**A single global vector beats every local-neighbourhood flow model.** Residual after global
removal: 44b6 median 1.72→1.17 um; 6bba median 1.82→1.14 um, p90 4.16→2.42 um.

**[MEAS]** The drift is temporally persistent, especially in 6bba (autocorrelation of the per-frame
drift direction): 6bba cos = +0.635 / +0.652 / +0.618 / **+0.600** at lag 1 / 2 / 5 / 10;
44b6 = +0.348 / +0.347 / +0.287 / +0.261. **[INF]** 6bba's motion is a **sustained directional
tissue flow** (epiboly / convergence-extension); 44b6's is closer to jitter. A persistent drift is
also *predictable* — it can be extrapolated from previous frames, not just estimated per frame.

**[DOC]** Note the literature gap this fills: **there is no published measurement of the spatial
velocity correlation length in the zebrafish gastrula or blastula.** The nearest values are the
tailbud (~15 cell diameters [Lawton et al. 2013, *Development* 140:573–582], reported as 20–100 um
by [Uriu & Morelli 2014, *Biophys J* 107:514–526] — the two are not reconcilable) and chick gastrula
mesoderm (polar-order decay length 57 um). Our +0.29 to +0.39 at 60–200 um is consistent with a
correlation length at the upper end of that range or beyond.

### 4.2 The drift is recoverable without ground truth

**[MEAS]** Correspondence-free estimator: take candidate positions at t and t+1, form all pairwise
offsets under 8 um, mean-shift to the mode. Compared against the true per-frame drift, with the
detection set deliberately corrupted:

| Scenario | embryo | \|err\| median | cos vs truth | var explained by the *estimated* drift |
|---|---|---|---|---|
| GT points, uncorrupted | 44b6 / 6bba | 0.36 / 0.36 um | +0.96 / +0.97 | 45.1% / 56.7% |
| **70% kept + 40% decoys** | 44b6 / 6bba | 0.47 / 0.49 um | **+0.90 / +0.94** | 37.4% / **51.3%** |
| **50% kept + 100% decoys** | 44b6 / 6bba | 0.59 / 0.59 um | **+0.89 / +0.91** | 36.6% / **42.9%** |

Under the harsh setting (half the detections missing, as many decoys as real nuclei) 6bba still goes
from median 1.82 → **1.23 um** and p90 3.75 → **2.79 um**. **[INF] This is deployable today** and
does not depend on detector quality being good.

**[MEAS]** One thing that does *not* work: integer-voxel phase correlation on the raw volumes
returned zero shift almost everywhere (the drift is ~0.6 z-voxels / ~2.5 xy-voxels, below its
resolution). Sub-voxel refinement would be needed; the point-based vote above is simpler and
already validated, so prefer it.

### 4.3 Where the biology predicts coherence breaks

**[DOC]**, none of these tested here — flagged as where to look when the global-drift residual is
anomalous:
- **Prechordal plate / neurectoderm interface** — the best-quantified break, and it is *anti*-correlated:
  local velocity correlation **0.37 ± 0.03 (WT)** falling to **-0.24 ± 0.04** in *slb*/wnt11 morphants,
  with interface friction ~100x the pure-fluid expectation [Smutny et al. 2017, *Nat Cell Biol* 19:306–317].
- **Germ ring margin / involution** — a marginal cell inverts its radial position relative to its
  neighbours within **15–30 min** [Moriyama et al. 2025, *Development*, doi:10.1242/dev.204261].
  No published velocity-jump magnitude.
- **EVL vs deep cell layer** — an explicit slip interface [Behrndt et al. 2012, *Science* 338:257–260;
  Morita et al. 2017, *Dev Cell* 40:354–366]. **No published velocity difference in um/min** — a gap.
- **Radial intercalation during epiboly** — **28% upward / 41% lateral / 31% downward**, i.e. balanced
  and non-directional: pure neighbour shuffling with no net drift [Bensch et al. 2013].

**[DOC]+[INF] How long a neighbourhood prior stays valid.** Neighbour exchange in the gastrula
paraxial mesoderm: **0.16–0.7 exchanges per cell per hour** (derived from Yin et al. 2008, *JCB*
180:221–232: 34% of WT cells moved medially with no exchange in 90 min); neighbour half-life ~1 h.
At 90 s/frame that is **one exchange per ~100–450 frames per cell** — a neighbourhood *identity*
prior is stable across our whole 100-frame window. **[INF]** So neighbourhood structure is stable;
what fails is using it to predict *velocity*, because §4.1 shows there is no local velocity structure
to exploit inside 104 um.

---

## 5. Why 6bba is harder than 44b6

**First, a correction to the premise.** **[MEAS]** 6bba does **not** have a higher division rate.
Per annotated cell-frame: **44b6 1.29e-3, 6bba 1.11e-3** — 44b6 is marginally *higher*. 6bba has 5x
the raw division count purely because it has **5.6x the annotated cell-frames** (113,121 vs 20,197)
across 1.8x the crops. **There is no biological division-rate difference between the embryos.**

**[MEAS]** The real differences:

| | 44b6 | 6bba | Ratio |
|---|---|---|---|
| Nuclei per frame (N_est/100) | 327 | 97 | 0.30 |
| Tissue occupancy of the 104 um cube | 28% | **8%** | 0.29 |
| Interphase nuclear equivalent diameter | 9.74 um | **10.30 um** | 1.06 |
| Nucleus peak intensity | 2150 | **728** | 0.34 |
| Relative contrast (peak-bg)/bg | 1.2 | **2.4** | 2.0 |
| Raw displacement median / p90 | 1.72 / 2.93 um | 1.82 / **4.16** um | 1.06 / **1.42** |
| Global drift magnitude, median | 1.01 um | **1.35 um** | 1.34 |
| Drift persistence (cos at lag 10) | 0.26 | **0.60** | 2.3 |
| Interior track starts / ends (% of nodes) | 0.95 / 1.22 | **2.88 / 2.90** | **3.0 / 2.4** |
| Median GT distance to nearest crop face | 22.8 um | **12.2 um** | 0.54 |
| Median GT track segment length | 50 frames | **20 frames** | 0.40 |

**[INF] The picture that explains all of it at once: 6bba crops sample a thin, motile blastoderm
sheet hugging the crop faces; 44b6 crops sample denser interior tissue.**

Evidence chain: 6bba occupancy is 8% (a thin curved shell cutting through a mostly-yolk cube) →
its nuclei sit at half the distance from a crop face → cells enter and leave the field constantly →
3x the interior track initiations/terminations and 2.5x shorter GT track segments. Independently,
the sheet is *moving* — 6bba's drift is 34% larger and 2.3x more temporally persistent, which is what
sustained epiboly/convergence-extension flow looks like [DOC §4.3], not jitter.

**[INF]** Larger nuclei (+6% by equivalent diameter here; +25% by half-max radius per the detector
measurement) point the same way: 6bba is **slightly earlier in development** than 44b6, i.e. ~1
cleavage cycle less reductive division, consistent with sparser and larger.

**[INF] Predicted dominant error modes in 6bba** — each falsifiable against the existing
`d1_postprocess.py` M/C/T/L/D partition:

1. **Track initiation/termination (birth/death) errors dominate, not detection.** 6bba needs ~3x the
   appearance/disappearance budget of 44b6. A single global birth/death cost is mis-specified by 3x
   for one embryo. **Test:** partition 6bba's edge losses by whether the GT terminus is within 10 um
   of a crop face; the prior predicts ~2/3 of them are.
2. **Association ambiguity from bulk flow, not from crowding.** 6bba is 3.4x *sparser*, so crowding
   cannot be the cause; its p90 displacement is 1.42x higher and drift-driven. **Test:** re-score
   6bba after applying the §4.2 correspondence-free drift correction; the prior predicts the linker
   gap closes disproportionately on 6bba (its var-explained is 65% vs 44b6's 55%).
3. **Detection is *not* predicted to be the bottleneck.** 6bba has 2x better *relative* contrast
   despite 3x lower absolute intensity, and larger nuclei. **Test:** if detection recall is
   comparable between embryos, this confirms the error budget is linking, not detection — and the
   0.90 vs 0.70 gap should be attacked at the linker, not the detector.

---

## 6. What to do next

In order, cheapest and highest-confidence first:

1. **Implement the correspondence-free drift correction (§4.2)** and re-score both embryos. It is
   validated GT-free, robust to 50% detection loss, and removes 43–65% of displacement variance.
   Predicted to help 6bba more than 44b6. This is the single largest measured effect in this report.
2. **Add the division refractory hard constraint (§1 #2, K=40 frames)**. Zero measured cost against
   GT (p=0.0017 that it is wrong), removes a class of FP outright.
3. **Recalibrate the division threshold per crop against 1.13e-3 x N_est (§1 #3)**, and check that
   against the 613 FP reported in `division_lane_2026-08-18.md`.
4. **Add the monotone-separation gate (§1 #4b)**. Free, uses frames the linker already holds.
5. **Set birth/death costs per embryo (§1 #8)**, 3x higher for 6bba.

Items 2–5 are constraints and calibrations, not new models — they cost no training and no GPU.

**Do not** build a division-synchrony prior, a mitotic-domain prior, a pre-division kinematic
feature, or a kNN neighbourhood-flow prior. All four were tested this session and are absent (§1,
[NULL] table).

---

## 7. Reproduction

All **[MEAS]** numbers come from scripts run this session against `data/train/*.geff` and
`data/train/*.zarr` only. Method notes that matter for re-running:

- GEFF node coordinates are **integer voxel indices**, not microns; scale by (z, y, x) =
  (1.625, 0.40625, 0.40625) um. All distances in this report are physical microns.
- Node ids encode `label * 1e9 + t`. All GT edges have **dt = 1** (verified: 128,883/128,883) — there
  are no frame-skipping edges, so any gap-closing in the pipeline is unsupported by GT.
- `attrs['geff']['extra']['estimated_number_of_nodes']` is the **total nucleus count over 100 frames**
  (median 32,681 for 44b6, 9,691 for 6bba); divide by 100 for the per-frame count. This is the
  denominator for every rate prior above and the per-crop normaliser for prior #3.
- Nuclear morphometry used a half-max threshold (bg = 10th percentile of a
  ±6 z / ±18 y / ±18 x voxel window, capped to a 6.5 um ball), connected component through the
  centre, intensity-weighted covariance in physical units.
- Divisions are nodes with out-degree 2 (151 total: 26 in 44b6, 125 in 6bba). Interphase controls
  are annotated non-division nodes drawn from **the same crop and the same frame** as a division —
  same-frame matching is essential, since absolute intensity differs 3x between embryos.

---

## Sources

**Staging, cell cycle, synchrony**
- Kimmel CB, Ballard WW, Kimmel SR, Ullmann B, Schilling TF (1995). Stages of embryonic development of the zebrafish. *Dev Dyn* 203(3):253–310. doi:[10.1002/aja.1002030302](https://doi.org/10.1002/aja.1002030302)
- Kane DA, Kimmel CB (1993). The zebrafish midblastula transition. *Development* 119(2):447–456. doi:[10.1242/dev.119.2.447](https://doi.org/10.1242/dev.119.2.447)
- Kane DA (1999). Cell cycles and development in the embryonic zebrafish. *Methods Cell Biol* 59:11–26 — source of the 54/78/151/240 min cycle lengths
- Kane DA, Warga RM, Kimmel CB (1992). Mitotic domains in the early embryo of the zebrafish. *Nature* 360:735–737. doi:[10.1038/360735a0](https://doi.org/10.1038/360735a0)
- Kimmel CB, Warga RM, Kane DA (1994). Cell cycles and clonal strings during formation of the zebrafish central nervous system. *Development* 120(2):265–276. doi:[10.1242/dev.120.2.265](https://doi.org/10.1242/dev.120.2.265)
- Mishra N, Li YI, Hannezo E, Heisenberg C-P (2026). Geometry-driven asymmetric cell divisions pattern cell cycles and zygotic genome activation in the zebrafish embryo. *Nature Physics*. doi:[10.1038/s41567-025-03122-1](https://doi.org/10.1038/s41567-025-03122-1)
- Bensch R, Song S, Ronneberger O, Driever W (2013). Non-directional radial intercalation dominates deep cell behavior during zebrafish epiboly. *Biol Open* 2(8):845–854. doi:[10.1242/bio.20134614](https://doi.org/10.1242/bio.20134614) — **the 16.11 um NN-spacing staging anchor**
- Kobitski AY et al. (2015). An ensemble-averaged, cell density-based digital model of zebrafish embryo development. *Sci Rep* 5:8601. doi:[10.1038/srep08601](https://doi.org/10.1038/srep08601)
- Reisser M et al. (2018). Single-molecule imaging correlates decreasing nuclear volume with increasing TF-chromatin associations during zebrafish development. *Nat Commun* 9:5218. doi:[10.1038/s41467-018-07731-8](https://doi.org/10.1038/s41467-018-07731-8)

**Division mechanics and appearance**
- Adar-Levor S et al. (2021). Cytokinetic abscission is part of the midblastula transition in early zebrafish embryogenesis. *PNAS* 118(15):e2021210118. doi:[10.1073/pnas.2021210118](https://doi.org/10.1073/pnas.2021210118)
- Percival SM, Parant JM (2016). Observing mitotic division and dynamics in a live zebrafish embryo. *JoVE* 113:e54218. doi:[10.3791/54218](https://doi.org/10.3791/54218)
- Oda H et al. (2023). Actin filaments accumulated in the nucleus remain in the vicinity of condensing chromosomes in the zebrafish early embryo. *Biol Open* 12(5):bio059783. doi:[10.1242/bio.059783](https://doi.org/10.1242/bio.059783)
- Azizi A, Herrmann A, Wan Y et al. (2020). Nuclear crowding and nonlinear diffusion during interkinetic nuclear migration in the zebrafish retina. *eLife* 9:e58635. doi:[10.7554/eLife.58635](https://doi.org/10.7554/eLife.58635)
- Martin RM, Cardoso MC (2010). Chromatin condensation modulates access and binding of nuclear proteins. *FASEB J*. PMID [19897663](https://pubmed.ncbi.nlm.nih.gov/19897663/) — the 2–3x chromatin volume drop
- Puah WC, Chinta R, Wasser M (2017). Quantitative microscopy uncovers ploidy changes during mitosis in live Drosophila embryos and their effect on nuclear size. *Biol Open* 6(3):390–401. doi:[10.1242/bio.022079](https://doi.org/10.1242/bio.022079)
- Gilad T, Reyes J, Chen J-Y, Lahav G, Riklin Raviv T (2019). Fully unsupervised symmetry-based mitosis detection in time-lapse cell microscopy. *Bioinformatics* 35(15):2644–2653. doi:[10.1093/bioinformatics/bty1034](https://doi.org/10.1093/bioinformatics/bty1034)
- Turley J, Chenchiah IV, Martin P, Liverpool TB, Weavers H (2024). Deep learning for rapid analysis of cell divisions in vivo during epithelial morphogenesis and repair. *eLife* 12:RP87949. doi:[10.7554/eLife.87949](https://doi.org/10.7554/eLife.87949)
- Krzic U, Gunther S, Saunders TE, Streichan SJ, Hufnagel L (2012). Multiview light-sheet microscope for rapid in toto imaging. *Nat Methods* 9(7):730–733. doi:[10.1038/nmeth.2064](https://doi.org/10.1038/nmeth.2064) — daughter-separation peak timing (supplementary)

**Tissue motion**
- Hernández-Vega A et al. (2017). Polarized cortical tension drives zebrafish epiboly movements. *EMBO J* 36(1):25–41. doi:[10.15252/embj.201694264](https://doi.org/10.15252/embj.201694264)
- Yin C, Kiskowski M, Pouille P-A, Farge E, Solnica-Krezel L (2008). Cooperation of polarized cell intercalations drives convergence and extension of presomitic mesoderm during zebrafish gastrulation. *J Cell Biol* 180(1):221–232. doi:[10.1083/jcb.200704150](https://doi.org/10.1083/jcb.200704150)
- Smutny M et al. (2017). Friction forces position the neural anlage. *Nat Cell Biol* 19(4):306–317. doi:[10.1038/ncb3492](https://doi.org/10.1038/ncb3492)
- Lawton AK et al. (2013). Regulated tissue fluidity steers zebrafish body elongation. *Development* 140(3):573–582. doi:[10.1242/dev.090381](https://doi.org/10.1242/dev.090381)
- Uriu K, Morelli LG (2014). Collective cell movement promotes synchronization of coupled genetic oscillators. *Biophys J* 107(2):514–526. doi:[10.1016/j.bpj.2014.06.011](https://doi.org/10.1016/j.bpj.2014.06.011)
- Mongera A et al. (2018). A fluid-to-solid jamming transition underlies vertebrate body axis elongation. *Nature* 561:401–405. doi:[10.1038/s41586-018-0479-2](https://doi.org/10.1038/s41586-018-0479-2)
- Moriyama Y et al. (2025). Hoxb genes determine the timing of cell ingression by regulating cell surface fluctuations during zebrafish gastrulation. *Development*. doi:[10.1242/dev.204261](https://doi.org/10.1242/dev.204261)
- Behrndt M et al. (2012). Forces driving epithelial spreading in zebrafish gastrulation. *Science* 338(6104):257–260. doi:[10.1126/science.1224143](https://doi.org/10.1126/science.1224143)
- Morita H et al. (2017). The physical basis of coordinated tissue spreading in zebrafish gastrulation. *Dev Cell* 40(4):354–366.e4. doi:[10.1016/j.devcel.2017.01.010](https://doi.org/10.1016/j.devcel.2017.01.010)
- Shah G et al. (2019). Multi-scale imaging and analysis identify pan-embryo cell dynamics of germlayer formation in zebrafish. *Nat Commun* 10:5753. doi:[10.1038/s41467-019-13625-0](https://doi.org/10.1038/s41467-019-13625-0)

**Dataset and instrument**
- Lange M et al. (2024). A multimodal zebrafish developmental atlas reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors (Zebrahub). *Cell* 187(23):6742–6759.e17. doi:[10.1016/j.cell.2024.09.047](https://doi.org/10.1016/j.cell.2024.09.047)
- Bragantini J et al. (2025). Ultrack: pushing the limits of cell tracking across biological scales. *Nat Methods*. doi:[10.1038/s41592-025-02778-0](https://doi.org/10.1038/s41592-025-02778-0) — **Methods carries the 1.625 um / 90 s / shield(6 hpf)→24 hpf acquisition that matches our geometry**
- Yang B, Lange M, Millett-Sikking A et al. (2022). DaXi — high-resolution, large imaging volume and multi-view single-objective light-sheet microscopy. *Nat Methods* 19(4):461–469. doi:[10.1038/s41592-022-01417-2](https://doi.org/10.1038/s41592-022-01417-2)

## Documented gaps — numbers that do not exist in the literature

These were searched for and are genuinely unpublished. Where we measured them, the measurement is
novel and worth keeping.

1. **Sigma (minutes) of a post-MBT division wave** — unpublished. Mishra et al. 2026 source data is
   the best place to look.
2. **Autocorrelation of division timing** — unpublished in any form.
3. **Velocity correlation length in the zebrafish gastrula or blastula** — unpublished. **We measured
   it (§4.1): coherence +0.29 to +0.39 at 60–200 um, i.e. correlation length >= our 104 um field.**
4. **Nuclear density in nuclei per um^3 at any zebrafish stage** — unpublished in absolute units.
5. **Mitotic vs interphase intensity ratio / elongation distribution in an embryonic H2B light-sheet
   dataset** — unpublished. **We measured it (§3.4): elongation AUC 0.68–0.69, ratio 1.33–1.34.**
6. **Parent-vs-daughter integrated H2B intensity across division in an embryo** — unpublished.
   **We measured it (§3.3): sum-ratio 1.30–1.63, daughter/daughter 0.76–0.79.**
7. **EVL-vs-deep-cell velocity difference (the epiboly shear magnitude)** — unpublished.
8. **Absolute T1 / neighbour-exchange rate per cell per hour in zebrafish** — stated only as ratios;
   the 0.16–0.7 h^-1 bracket in §4.3 is derived, not quoted.
9. **Per-embryo hpf start/end for the Zebrahub timelapses** — unpublished.
