# Detection-Ceiling-Breaking Research — Recovering Weak/Missing Nuclei + Better Localization

Lane owner: DETECTION-CEILING lane. Date: 2026-07-06.
Competition: Biohub "Cell Tracking During Development" (CZ Biohub / Royer Lab). 3D+t light-sheet zebrafish nuclei.
Metric: weighted adjacent-edge Jaccard + 0.1·division-Jaccard on a DISJOINT hidden embryo. Match gate ~7 µm. Voxel (z,y,x)=(1.625,0.40625,0.40625) µm (z ~4× coarser → strongly anisotropic).
Platform: Kaggle T4×2, internet-off at submit, ≤12 h.

## Problem framing (verified from lane brief + oracle)
- Detector = TemporalUNet3D, node recall ~95% / ~90% on the two public embryos. **(verified — given)**
- Candidate-edge oracle: max edge-Jaccard is capped at **~0.885 (min-fold)** because **~11.5% of GT edges have an endpoint that was never detected** → unrecoverable by *any* linker on the current candidate set. **(verified — measured by oracle)**
- Therefore the score ceiling is a **detection-recall** ceiling, not a linking ceiling. To lift it we must recover the missing/weak nuclei **and** tighten endpoint localization at the 7 µm gate, **without hallucinating false cells** (FPs create wrong edges → hurt the Jaccard denominator).

### The two failure sub-populations (this determines which method wins)
1. **Interior gaps** — cell detected at t−1 and t+1 but missing at t (freeze/bleach/z-overlap dropout). A track *brackets* the miss → track-conditioned redetection recovers it almost for free. **(inference)**
2. **Endpoint/edge misses** — the missing node sits at a track start/end, a division daughter, or an isolated weak cell with no bracketing track. No linker context → needs threshold-lowering with multi-frame temporal evidence, i.e. genuine track-before-detect (TBD). **(inference)**

The oracle's 11.5% is a *mix* of (1) and (2). Split it empirically before committing compute (see falsification #0).

---

## Ranked table

EV = expected edge-Jaccard gain toward the 0.885 min-fold ceiling, conditional on the sub-population it addresses. Effort/compute in Kaggle-days and T4×2-hours.

| # | Method | Discipline | Mechanism (microscopy translation) | EV | Effort | Compute | Generalization risk | Evidence | Repo | First falsification experiment |
|---|--------|-----------|------------------------------------|----|--------|---------|--------------------|----------|------|-------------------------------|
| **1** | **Track-conditioned redetection (graph-guided gap filling + image verification)** | Cell tracking (ELEPHANT), MOT (ByteTrack 2nd-round low-score assoc.) | For every linked gap or predicted-but-missing endpoint, search that voxel neighborhood in raw image; accept a node ONLY if a local blob/peak passes a *low* threshold AND is claimed by the track. Recovers interior gaps (sub-pop 1) exactly. | **High** (recovers most of sub-pop 1; est. +0.02–0.05 Jaccard) | Low (1–2 d) | Low (<1 h) | **Low** — only accepts image-supported, track-predicted nodes; can't invent free-floating cells | ByteTrack ECCV'22 (associate every box, recover low-score via tracklet motion); ELEPHANT eLife'21 predict→proofread loop | ByteTrack `github.com/ifzhang/ByteTrack`; ELEPHANT `github.com/elephant-track` | Take current OOF tracks; at each gap, sample raw intensity at linear/GP-interpolated position; check recall of recovered nodes vs GT. If <40% of gaps have a real local max within 7 µm, gaps are not image-supported → drop method. |
| **2** | **Multi-hypothesis / multi-threshold candidate detection (hierarchical), linker picks** | Cell tracking (Ultrack UCM/ILP) | Emit candidate nuclei at *several* detector thresholds (or nested UCM segments); pass ALL as candidates; let the temporal-consistency linker/ILP select the subset that forms coherent tracks. Weak nuclei survive if temporally supported; spurious ones are pruned by the linker. Addresses both sub-pops. | **High** (raises candidate recall globally; est. +0.02–0.06) | Med (2–4 d) | Med (2–5 h; ILP scales) | Med — extra candidates raise FP pressure; needs the linker to be strong enough to prune; "systematic errors persisting over time" break it | Ultrack Nat. Methods'25 (multiple seg. hypotheses, ultrametric contour hierarchy, ILP selects temporally consistent segments) | `github.com/royerlab/ultrack` | Lower detector threshold to double candidate count; measure oracle max edge-Jaccard on the enlarged candidate set. If ceiling doesn't rise above 0.885, the missing endpoints are truly sub-threshold (need TBD, not just a lower cut). |
| **3** | **Sub-voxel 3D anisotropic Gaussian / PSF centroid refinement** | Fluorescence localization microscopy | Fit a 3D anisotropic Gaussian (separate σz vs σxy) or PSF model to each detected + candidate peak; replace argmax centroid with fitted center + uncertainty. Tightens endpoint precision at the 7 µm gate → converts near-miss matches into hits. Pure precision play, orthogonal to recall. | Med (+0.005–0.02; helps matches already near the gate) | Low (1–2 d) | Low (<1 h) | **Low** — refines existing detections only, adds no new cells | Anisotropic-Gaussian spot model beats isotropic (Measurement'23); gradient/Gaussian-mask sub-pixel fits (Sci.Rep. srep02462; PMC4585720) | scikit-image `blob_log`; custom `scipy.optimize` fit | Perturb GT-matched detections by refit vs raw argmax; measure fraction of edges that flip from >7 µm to <7 µm. If <2% flip, localization is not the binding constraint. |
| **4** | **Track-before-detect proper: shift-and-stack / synthetic tracking / DP-TBD** | Astronomy (synthetic tracking), radar (DP-TBD) | Integrate *subthreshold* nuclear response over 3–10 frames along motion-compensated linear paths; promote a cell only when coadded temporal SNR beats a matched background path. Recovers sub-pop 2 (never-detected weak cells with no bracketing track) — the part nothing else reaches. | Med–High on sub-pop 2 only (its unique reach; est. +0.01–0.04) | High (4–7 d) | High (velocity-swept path search; GPU) | **Med–High** — motion prior + FP control must transfer to unseen embryo; over-integration hallucinates | Synthetic tracking detects >1000 asteroids below single-frame limit (arXiv 2509.26279, 1309.3248); YOSO motion-filter cuts FPs (arXiv 2605.06913); DP-TBD range-Doppler (MDPI 16/14/2639); unified DP-TBD analysis (arXiv 2512.11170) | KBMOD `github.com/dirac-institute/kbmod` (GPU shift-and-stack); no cell-specific repo | Motion-compensate a 5-frame window around a KNOWN missing GT nucleus; coadd; check whether the true cell's SNR crosses threshold while a random-path control does not. If SNR gain <2×, TBD won't separate signal from background here. |
| 5 | Matched-filter / multiscale LoG re-scoring at candidate voxels | Signal processing / classical CV | Convolve raw image with a bank of anisotropic LoG/matched nuclear templates; use response as a cheap weak-nucleus score to seed candidates for #1/#2 in low-confidence regions. | Low–Med (feeder, not standalone) | Low (1 d) | Low | Low–Med (scale/template tuned to embryo) | Generalized LoG blob detection (Kong et al.); LoG-MIP 3D nuclei localization | scikit-image `blob_log`, `blob_dog` | Compare LoG-peak recall at missing-GT locations vs learned detector. If LoG finds no extra true peaks the detector missed, it adds nothing. |
| 6 | Richardson–Lucy deconvolution as preprocessing | Deconvolution microscopy | Deblur the anisotropic z-PSF before detection to separate z-overlapping nuclei and sharpen weak ones, raising recall + localization jointly. | Low–Med (uncertain; risk of ringing FPs) | Med (2 d) | Med (iterative 3D) | Med–High — needs PSF estimate; artifacts create FPs; may not transfer | Standard RL deconvolution; localization reviews (arXiv 2011.03296) | `RedLionfish`, `flowdec` (GPU RL) | Deconvolve a crop with known z-overlapping GT pair; measure whether two peaks resolve. If they don't split, RL won't fix z-overlap dropout. |
| 7 | RFS / labelled multi-Bernoulli TBD (joint detect+track) | Radar / target tracking | Principled joint Bayesian detection+tracking over subthreshold measurements; multi-Bernoulli set posterior promotes tracks from raw pixels. | Low (theoretically strongest, practically overkill) | High (5–8 d) | High | **High** — heavy tuning, dense-nuclei state explosion, poor transfer, hard to fit in 12 h | LMB-TBD particle filter (arXiv 1604.00082); Stone Soup GM-PHD tutorials; PHD-TBD (arXiv 2302.11356) | Stone Soup `github.com/dstl/Stone-Soup` | Prototype LMB-TBD on a 2D slice with 10 cells; if it can't beat #1 on recovered-node recall within a day of tuning, abandon (won't scale to dense 3D). |

---

## The microscopy translation of Track-Before-Detect (TBD)

**Radar/astronomy TBD** postpones the thresholding decision: instead of thresholding each frame then linking survivors ("detect-before-track"), it hypothesizes a target *trajectory*, coadds the raw signal along it, and only then thresholds the *integrated* evidence. Weak-per-frame targets that never survive a single-frame cut become detectable once energy is accumulated coherently along the correct motion path. Canonical implementations: **shift-and-stack / synthetic tracking** (astronomy — coadd pixels along a candidate linear path; >1000 sub-threshold asteroids recovered, arXiv 2509.26279, 1309.3248), **DP-TBD** (radar — dynamic-programming path with max value-function over a kinematic transition graph, MDPI 16/14/2639), **particle-filter TBD** and **RFS/LMB-TBD** (arXiv 1604.00082).

**Translation to zebrafish nuclei:**
- *"Target amplitude below single-frame threshold"* → a nucleus whose TemporalUNet3D response is below the detection cut on a given frame (bleaching dip, z-overlap, low expression).
- *"Motion model / velocity trial"* → local cell motion between frames (small, smooth over 3–10 frames), *not* the fast linear sweep of asteroids. This is the crucial advantage: cell displacement per frame is small and locally predictable, so the motion-hypothesis search space is tiny compared to astronomy → coadding is cheap.
- *"Coadd along the path"* → motion-compensate a short temporal window (align frames to the hypothesized cell position via optical-flow/local-registration) and sum the raw or logit response.
- *"Threshold the integrated statistic"* → promote a nucleus only when the coadded response along the true path beats a matched-filter background path (YOSO's key FP-control idea: require a point-source-consistent trail, arXiv 2605.06913).
- *"Matched background path"* → the anti-hallucination guarantee: a candidate is accepted only if its temporal evidence exceeds what a random motion-compensated path through background would accumulate. **This is what stops TBD from inventing cells** and is the direct analog to the metric's need to avoid FP edges.

Net: TBD is the only family that reaches sub-population 2 (never-detected cells with no bracketing track). But for the bulk of the oracle's 11.5% that are *interior gaps* (sub-pop 1), full TBD is overkill — track-conditioned redetection (#1) already gets them at ~1% of the cost.

---

## THE ONE BET

**A track-conditioned redetection pass with a two-tier ("ByteTrack-style") threshold gate, plus 3D anisotropic-Gaussian localization refinement on every accepted node.**

Concretely:
1. Run the existing detector at its normal threshold → high-precision nodes → link into OOF tracks (current pipeline).
2. **Second, low-threshold candidate pass** (or nested UCM levels à la Ultrack) generates weak candidate peaks everywhere — high recall, low precision.
3. **Gate the weak candidates by track context**: accept a weak candidate ONLY if (a) it fills a bracketed gap or extends a predicted track endpoint into a location the interpolated motion predicts, AND (b) the raw image has a local maximum / matched-filter response there. (ByteTrack's second-round association of low-score boxes via tracklet motion, translated to 3D nuclei; ELEPHANT's predict→verify loop.)
4. **Refine every accepted node** with a 3D anisotropic Gaussian fit (σz ≠ σxy) to sharpen the endpoint under the 7 µm gate.

Why this bet: it targets exactly the oracle-identified missing *endpoints*, recovers them **only where both a track predicts a cell and the image supports one** (dual gate → recall up, FPs controlled → protects the Jaccard denominator), is cheap (<1 h on T4×2, fully offline), needs no new training, and generalizes to the unseen embryo because it rides the *existing* detector + linker rather than a new learned prior. Keep #4 (full TBD, KBMOD-style shift-and-stack) staged as the **second wave** to mop up sub-population 2 *only if* falsification #0 shows a large never-bracketed remainder.

**Falsification #0 (run first, before any build):** classify the oracle's missing-endpoint edges into (1) bracketed-by-a-track vs (2) not. If ≥~60% are bracketed → the ONE bet captures most of the ceiling and TBD is deferred. If the miss is dominated by unbracketed endpoints → escalate to method #4 immediately.

---

## Sources
- DP-TBD range–Doppler: https://www.mdpi.com/2072-4292/16/14/2639
- Unified DP-TBD analysis: https://arxiv.org/pdf/2512.11170
- LMB-TBD particle filter: https://arxiv.org/pdf/1604.00082
- PHD-TBD (Poisson conjugate prior): https://arxiv.org/pdf/2302.11356
- Synthetic tracking / shift-and-stack (asteroids below single-frame limit): https://arxiv.org/abs/2509.26279 , https://arxiv.org/pdf/1309.3248
- YOSO motion-filtered deep detection of faint moving sources: https://arxiv.org/pdf/2605.06913
- ELEPHANT incremental 3D cell tracking (predict→proofread): https://elifesciences.org/articles/69380
- Ultrack (multi-hypothesis, UCM hierarchy, ILP): https://www.nature.com/articles/s41592-025-02778-0 , https://github.com/royerlab/ultrack
- ByteTrack (associate every detection box, recover low-score via tracklet motion): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf
- Anisotropic Gaussian spot localization: https://www.sciencedirect.com/science/article/abs/pii/S0263224123003202
- 3D single-particle localization precision: https://www.nature.com/articles/srep02462 , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4585720/
- Generalized LoG blob/nuclei detection: https://www.researchgate.net/publication/260586629
- Localization microscopy review: https://www.arxiv.org/pdf/2011.03296v1
- Stone Soup (RFS/PHD toolbox): https://github.com/dstl/Stone-Soup
- KBMOD (GPU shift-and-stack): https://github.com/dirac-institute/kbmod

## Verified / claim / inference tags
- **Verified (given):** detector recall figures; oracle 0.885 min-fold ceiling and 11.5% never-detected-endpoint fraction; voxel anisotropy; metric/gate.
- **Claim (from cited literature):** TBD/synthetic-tracking recovers sub-threshold targets; Ultrack multi-hypothesis + ILP prunes FPs; ByteTrack low-score recovery; anisotropic-Gaussian localization gains.
- **Inference (mine):** the split into interior-gap vs unbracketed-endpoint sub-populations; the EV ranges; that track-conditioned redetection covers most of the oracle gap cheaply. All EV numbers are estimates pending falsification #0.
