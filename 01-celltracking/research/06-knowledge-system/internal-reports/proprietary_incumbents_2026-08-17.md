# Proprietary / commercial incumbents and the non-academic knowledge pool — 2026-08-17

Mandate: mechanisms from **commercial cell-tracking platforms, vendor manuals, patents, HCS practice,
and adjacent proprietary tracking (radar/MOT/PTV)** that are portable to a candidate-graph linker +
division head on existing detections. Deliberately non-overlapping with
`competitive_frontier_2026-08-16`, `competitive_refresh_2026-08-17`, `methods_frontier_2026-08-16`,
`novel_crossdomain_2026-08-17`, `quickwins_internal_2026-08-17`, `redteam_blindspots_2026-08-17`
(all read first; overlaps are called out explicitly in §3).

**Evidence labels used throughout.**
`DOCUMENTED` = the vendor's own manual / help system / API reference states the mechanism.
`PATENT` = disclosed in a granted or published application (mechanism is real, but it is *claimed* prose).
`MARKETING` = product-page copy with no algorithmic content.
`INFERENCE` = my reading, not stated by the source.
`UNVERIFIED` = I could not confirm it from a primary source; treated as a lead, not a fact.

Headline: **the commercial world converges on four things we do not do** — (a) *ordered/tiered*
association instead of one-shot association, (b) a **likelihood-ratio track score** as the
confirm/delete ranking function instead of track length, (c) an **appearance/mass-continuity term**
inside the association cost, and (d) an **anisotropy-aware (PSF-elongated) detection kernel**. It also
independently validates two things we already believe (prediction-gated relinking = Arm B;
gap-fill must materialise a node) and re-labels a lot of what we already ship.

---

## 1. Ranked summary — expected value x cheapness for us

| # | Mechanism | Source class | Attacks | GPU? | Cost | Status vs our corpus |
|---|---|---|---|---|---|---|
| 1 | **Tiered (two-round) association** — match high-confidence candidates first, then let *only leftover unmatched tracks* absorb low-confidence candidates | commercial MOT standard (BYTE/ByteTrack; NVIDIA DeepStream target mgmt) | EDGE + node recall (6bba 0.855) | No | ~1 day CPU | **NEW.** Directly unblocks the over-proposal that branch-A (−0.16) proved is structurally blocked |
| 2 | **Track score = accumulated log-likelihood ratio**, confirm/delete on score, replacing `MIN_TRACK_LEN=6` | MATLAB Sensor Fusion & Tracking Toolbox (`trackerGNN`/`trackerTOMHT`) `DOCUMENTED` | NODE + EDGE | No | ~4 h CPU on `p0strict_cache/graphs` | **NEW ranking function** for a channel redteam §6 already measured (+0.001 with *length* ranking) |
| 3 | **Intensity / mass-continuity term in the association cost** (`TotalCost = ΣDistanceCost + IntensityWeight·IntensityCost`) | Imaris manual `DOCUMENTED`; Aivia `Motion vs Intensity` `DOCUMENTED`; NIS-Elements mass-conservation `DOCUMENTED` | EDGE (crossing/adjacent paths) | No | ~1 day CPU | **NEW for our hand-built costs** (Arm B cost carries no intensity term) |
| 4 | **Anisotropic / PSF-elongated detection kernel + anisotropic centroid fit** (separate `Estimated Z Diameter`) | Imaris `DOCUMENTED` | centroid σ (CW1 cliff at 2 µm), node recall | No | ~1 day CPU | **NEW specifics** for the running sub-voxel refine lane |
| 5 | **Gap fill must emit an *optimized expected position* node** — vendor-verbatim confirmation of CW3, upgraded from linear interpolation to motion-model/image-optimized placement | Imaris `DOCUMENTED`; NVIDIA shadow tracking `DOCUMENTED` | EDGE (dt≠1 edges are silently dropped) | No | ~2 h audit + ~4 h CPU | **Confirms CW3**; the "optimized" part is new |
| 6 | **Per-frame cell-count consistency prior** (PGM enforcing non-decreasing / smooth count) as a missed-detection *locator* | US9542591B2 (Ares Trading / ex-Progyny) `PATENT` | node recall + NODE | No | ~2 h CPU probe | **NEW** |
| 7 | **Shake-The-Box loop: predicted position as a detection prior + residual-image re-detection** | LaVision DaVis 4D-PTV `DOCUMENTED` (product) / Schanz 2016 (mechanism) | node recall (6bba) | No for DoG re-detect; **Yes** if UNet re-run | 2–4 days | **NEW**; heaviest item on this list |
| 8 | **Two-phase global LAP**: frame-to-frame, then a *single* global track-segment assignment scoring {link, split, merge, independent} jointly | NIS-Elements `DOCUMENTED` (ships Jaqaman/u-track) | EDGE + gap + DIV | No | 3–5 days | **Partly closed** (split/merge alone measured +0.0006); only the *joint global* framing is new |
| 9 | **Cardinality-explicit RFS tracking** (PHD/CPHD `BirthRate`, weight-thresholded extraction) + the **spawn** term as the native division primitive | MATLAB `trackerPHD` `DOCUMENTED`; US10922820B2 (Sandia) `PATENT` | NODE + DIV | No | 3–5 days | **Elegant, low EV** — the channels it attacks are the two redteam closed |
| 10 | **Disagreement-triggered cascade arbiter** for mitosis (CNN vs handcrafted; third classifier only on disagreement) | US9430829B2 (Case Western) `PATENT` | DIV | No | — | **CLOSED BY OUR OWN DATA** (b=0 strict-subset result) — do not build |

Immediate order: **1, 2, 3** are all CPU-only, all run on `artifacts/kaggle/p0strict_cache/graphs`, and
all are independent of the parked Arm B GPU green-light. **4** folds into the sub-voxel lane already
in flight. Everything below 5 is a research bet, not a quick win.

---

## 2. Detail

### 1. Tiered (two-round) association — the commercial MOT default we do not do

**Mechanism.** Do not run one assignment over all candidates. Round 1: associate only
*high-confidence* candidates to tracks. Round 2: take the tracks left unmatched after round 1 and
associate them against the *low-confidence* candidates — and only to **extend** an existing track,
never to seed a new one. Low-confidence candidates that match nothing are discarded.

**Why this matters more for us than for MOT.** Our metric matches one-to-one within 7 µm. Our own
branch-A experiment measured **−0.16** from simply widening the candidate pool: extra candidates
*steal* true matches. That is not a scoring failure, it is an **ordering** failure — in a single-round
assignment a junk candidate competes on equal footing with a true one. Tiered association makes
that structurally impossible, which is exactly the block that has kept "over-propose then re-score"
(`methods_frontier` LEAD, `novel_crossdomain` #2) from being testable. It is also the cheapest
attack on the **6bba node_recall = 0.855** deficit (vs 44b6 0.985) that redteam Claim 4 identified as
where the remaining edge mass actually is.

**Source / status.** `DOCUMENTED` as the industry-standard two-stage association: "a primary stage
dedicated to linking high-confidence tracks with detections and a subsequent stage focusing on
pairing residual tracks with low-confidence detections"
(https://trackers.roboflow.com/latest/trackers/bytetrack/; original method ByteTrack, arXiv:2110.06864).
NVIDIA's shipping tracker documents the same philosophy from the other side — an unmatched track is
kept alive in **Shadow Tracking** and produces its own localization rather than dying
(NvMultiObjectTracker Parameter Tuning Guide,
https://docs.nvidia.com/metropolis/deepstream/6.4/dev-guide/text/DS_plugin_NvMultiObjectTracker_parameter_tuning_guide.html).

**Falsification (one line).** On `p0strict_cache/graphs`, split candidates at the detector-score median,
run round-1 assignment on the top half only, then a round-2 assignment of *unmatched tracks only*
against the bottom half; LOEO both folds; ship iff adj_edge_J is up bilaterally and N_pred/N_est does
not degrade. **GPU: No.**

**IP.** ByteTrack is MIT-licensed academic work in wide commercial use; no infringement exposure.

---

### 2. Track score (log-likelihood ratio) as the confirm/delete ranking function

**Mechanism `DOCUMENTED`.** Every commercial radar/MOT tracker ranks tracks by a **sequential
log-likelihood ratio**, not by length. MathWorks states it plainly: "Track confirmation and deletion
based on 'Score' are based on a log-likelihood computation… the ratio of the probability that the
track is from a real target to the probability that the track is false", with three interchangeable
policies — **History (M-of-N)**, **Score**, **Integrated (probability of existence)** — and their
duals for deletion (`trackerGNN` reference:
https://www.mathworks.com/help/fusion/ref/trackergnn-system-object.html;
overview: https://www.mathworks.com/help/fusion/ug/introduction-to-multiple-target-tracking.html).
Verbatim parameterisation: `TrackLogic` ∈ {`'History'`,`'Score'`}; History
`ConfirmationThreshold = [M N]` (default `[2,3]`), `DeletionThreshold = [P R]`; Score
`ConfirmationThreshold` default **20**, `DeletionThreshold` default **−7** — "a track is deleted if
its score decreases by at least the threshold from the maximum track score". The score is built from
`DetectionProbability` (default 0.9) and `FalseAlarmRate` (default 1e-6).

**Why portable.** Our `BIOHUB_OUTPUT_MIN_TRACK_LEN=6` is the crudest member of that family (M-of-N with
M=N=6, applied once at the end). Redteam §6 ran the full 199-crop deployed-substrate sweep and got a
**bilateral +0.001** from trimming the weakest 2.5% *ranked by length*. That result says the channel is
open but the **ranking function** is the binding constraint. LLR is the industry's answer, it consumes
the per-edge probabilities we already carry in the geffs, and — critically — the "delete when score
falls Δ below its own maximum" rule is *relative*, so it prunes a track that degrades without punishing
a short-but-clean one, which pure length cannot distinguish.

**Falsification.** On `p0strict_cache/graphs`, replace the length rank inside
`filter_short_track_components` with accumulated per-edge LLR (using our measured P_D and the
false-candidate rate), hold **node count identical**, re-score LOEO both folds; promote iff bilateral
Δ ≥ +0.002 (i.e. beats the length-ranked +0.001 baseline by 2x). **GPU: No.**

**IP.** The scoring rule is textbook (Blackman/Bar-Shalom lineage) and predates the toolbox; MathWorks
documents but does not own it. Do not vendor toolbox code — reimplement from the documented formulae.

---

### 3. Intensity / mass-continuity term inside the association cost

**Convergent across three independent vendors — all `DOCUMENTED`:**

- **Imaris** (Bitplane / Oxford Instruments), Reference Manual 7.6, Tracking section, verbatim:
  "All of the algorithms except for Connected Components use a Linear Assignment algorithm and Total
  Cost function… Autoregressive Expert mode combines changes in position and intensity to compute
  total cost", formalised as `TotalCost = Σ(all connections) DistanceCost + IntensityWeight *
  IntensityCost`. The manual's stated indication for turning it on is *our exact regime*: "suitable for
  tracing of multiple objects with adjacent or crossing paths, especially if the neighboring objects
  have a stable (but different) intensity."
  (https://www.ijm.fr/wp-content/uploads/2022/01/Imaris-Reference-Manual-7_6_0.pdf)
- **Leica Aivia** 3D Object Tracking recipe exposes exactly one blending knob: **`Motion vs Intensity`
  (0–10)** — "Adjusts the relative weighting between motion and object intensity for track-point
  matchmaking" (https://aivia-software.atlassian.net/wiki/spaces/AW/pages/96469083/3D+Object+Tracking).
- **Nikon NIS-Elements** uses mass as the *division* criterion: "Splitting (when enabled) is done first.
  Objects that split in two are determined based on selected mass conservation model: either the number
  of pixels or the sum intensity should be constant during the division."
  (NIS-Elements help, "Algorithm Overview",
  https://www.nisoftware.net/NikonSaleApplication/Help/Docs-AR/eng_ar/p2c24s9.html — **caveat: this page
  returned 404 on direct fetch; the wording above is from the indexed snippet, corroborated by a second
  independent query that also surfaced the Jaqaman/Kalman description in §8. Treat as `DOCUMENTED`
  (indexed), not re-verified.**)

**What is actually new for us.** Our learned edge head does carry intensity statistics, but our
*hand-built post-processing costs do not*: the Arm B relink cost is
`motion + 0.05*raw − LEARNED_BONUS*prob` (`scripts/kaggle_edits/armb_flow_gate.py`) — pure geometry plus
a learned scalar, with no appearance-continuity term. Gap-close and safe-div are likewise geometric.
Adding |ΔI|/Ī and |ΔV|/V̄ continuity is a few features, and it is the one axis three commercial
vendors independently decided was worth a first-class knob.

**Bonus: mass conservation as a division feature.** `sum(daughter mass) ≈ mother mass` is a *physical*
constraint that none of our division features encode (ours are all geometric: midpoint residual,
distance caps). Honest expectation: redteam Claim 1 established the mother-gate needs **AUC ≥ 0.97**
against a ~0.1% base rate and the best measured signal anywhere is 0.86–0.92, so this is very unlikely
to clear the 4.07%/6.38% break-even on its own. It is worth ~30 min as a *feature-AUC measurement*,
not as a lever.

**Falsification.** Add intensity- and volume-continuity features to (a) the motion-relink / gap-close
cost and (b) the candidate-edge meta-classifier; LOEO both folds; promote iff bilateral ≥ +0.002.
Separately, one 30-min AUC read of the mass-conservation residual on the mother-gate — predicted <0.97,
which closes it. **GPU: No.**

**IP.** Cost-combination of distance and intensity is decades-old prior art; no exposure.

---

### 4. Anisotropic / PSF-elongated detection kernel and centroid fit

**Mechanism `DOCUMENTED`.** Imaris's spot detector is an explicit DoG with a documented σ-ratio:
background subtraction is "the intensity … of a Gaussian filtered channel (Gaussian filtered by 3/4)
minus the intensity of the original channel Gaussian filtered by 8/9 of sphere radius", and its
`Quality` statistic is "the intensity at the center of the spot" in that filtered channel — i.e.
Quality *is* the DoG response (Imaris Reference Manual 7.6, Spots / Classify Spots). Crucially, Imaris
ships an explicit anisotropy switch: **"Model PSF-elongation along Z-axis"** with a separate
**`Estimated Z Diameter`**, for the case where "you might encounter z-axis elongation of cells, so
cells won't appear as spherical"
(https://www.allevi3d.com/livedead-assay-quantification-imaris/; feature listed for current Imaris,
https://imaris.oxinst.com/products/imaris-for-tracking).

**Why it matters here, now.** Our voxels are (1.625, 0.40625, 0.40625) µm — 4x anisotropic — on a
light-sheet whose PSF is *additionally* elongated axially. `sleepymegacat`'s measured centroid cliff is
**σ ≈ 2 µm** (CW1). Pure z-voxel quantisation is only σ ≈ 1.625/√12 ≈ 0.47 µm, i.e. safely under the
cliff — so the risk is **not** quantisation, it is **z-localisation bias from fitting an isotropic
kernel to an axially-elongated blob**, which is unbounded by the voxel size and is exactly the failure
mode the commercial default exists to prevent. This is a concrete specification for the sub-voxel
refine lane already in flight (`Sub-voxel refine lane: detector kaggle_edit + LOEO kernels`): fit
separate σ_xy and σ_z, do not fit one isotropic parabola.

**Falsification.** Refine centroids with an anisotropic 3D Gaussian (free σ_xy, σ_z) vs the current
refinement; measure the **per-axis** residual to GT among 7 µm-matched nodes; if the z-residual exceeds
~1.5 µm under isotropic fitting and drops under anisotropic, re-score LOEO. **GPU: No** (post-hoc
refinement on existing peaks; only a detector retrain with an elongated target would need GPU).

**IP.** LoG/DoG spot detection is Marr–Hildreth-era prior art. Note for completeness that at least one
patent claims a "three-dimensional linear Laplacian of Gaussian filter" for spot detection in a
fluorescence stack (US9896720, single-molecule FISH context) — the generic technique is not enclosable,
but do not copy a claimed *combination* wholesale.

---

### 5. Gap fill must materialise an optimized node (vendor confirmation of CW3, plus an upgrade)

**Mechanism `DOCUMENTED`, verbatim from the Imaris manual** ("Fill the gaps with all detected
objects"): "This algorithm allows tracks to be continued, even if the object was not detected in
periods of up to two consecutive time points… If this option is selected the algorithm performs the
**optimization of the object expected position** for the particular time point. The track is generated
by connecting objects assuming their optimized expected positions." NVIDIA does the same thing under a
different name: during Shadow Tracking, when the detector misses, "each object tracker makes its own
localization using the learned correlation filter", and that localization is fed to the Kalman filter
as a measurement with its own noise term (`measurementNoiseVar4Tracker`).

**Read.** Two independent commercial stacks *always* materialise a state at a missed frame. That is an
independent confirmation of CW3 (the scorer silently drops every dt≠1 edge, so a skip edge is worth
nothing). The **new** part is "optimized", not "interpolated": Imaris solves for the expected position
rather than taking the straight-line midpoint. With our median step 1.8 µm against a 7 µm radius,
linear interpolation almost certainly lands inside the radius already, so the upside here is small —
this is primarily an **audit** item with a bounded upgrade.

**Falsification.** (a) Audit: confirm our gap repair emits an interpolated *node* + two dt=1 edges, not
a dt=2 edge — free, ~2 h. (b) If it does, replace linear interpolation with a motion-model-optimized
position (constant-velocity fit over the flanking window) and re-score LOEO; expect ≤ +0.001.
**GPU: No.**

---

### 6. Per-frame cell-count consistency as a missed-detection locator

**Mechanism `PATENT`.** US9542591B2, *Apparatus, method, and system for automated, non-invasive cell
activity tracking*, assignee **Ares Trading SA** (originally Progyny/Auxogyn — the Eeva embryo
time-lapse product), filed 2014-02-28, granted 2017-01-10
(https://patents.google.com/patent/US9542591B2/en). The disclosed pipeline generates 50–200 competing
per-frame hypotheses by **perturbing the previous frame's fitted parameters**, refines each by EM, and
then selects the best *sequence* with a probabilistic graphical model + belief propagation that
enforces constraints including **non-decreasing cell count** and temporal coherence.

**Portable idea (not the claimed embodiment).** The count trajectory is a global, label-free consistency
signal. In a developing embryo the true cell population is monotone-increasing and smooth; a *dip* in
our per-frame predicted count is a frame where the detector missed. That gives a **targeted** gap-repair
/ re-detection trigger instead of a global one. Caveat that cuts against it: our units are
256x256x64 spatial **crops**, so cells legitimately enter and leave — count is not monotone per crop,
only *smooth*. So the usable form is a smoothness/anomaly test, not a hard monotone constraint.

**Falsification (cheap, do first).** On cached OOF graphs, compute GT and predicted per-frame node
counts per crop; test whether frames with a negative predicted-count anomaly are enriched for GT nodes
we missed. If the enrichment is not strong, drop it. ~2 h CPU. **GPU: No.**

**IP flag.** Claim 1 is an *apparatus* claim on "a hypothesis selection module configured to select a
hypothesis from a plurality of hypotheses characterizing one or more cells shown in an image" — that is
broad, and it is live. We would be taking the *idea* of a temporal count-consistency prior, which is
generic, not the claimed multi-hypothesis-ellipse-EM-PGM apparatus. Risk is low for a research
submission and non-trivial for a shipped product; **do not reproduce the hypothesis-perturbation +
EM + PGM pipeline as a whole.**

---

### 7. Shake-The-Box: prediction as a detection prior, and residual-image re-detection

**Mechanism.** LaVision's DaVis FlowMaster 4D-PTV is a shipping commercial product built on
Shake-The-Box: `DOCUMENTED` on the vendor page as "an Iterative Particle Reconstruction (IPR) technique
in combination with an advanced 4D-PTV algorithm using the time-information for track reconstruction",
claiming "higher reconstruction accuracy at much faster processing speed" than tomographic PIV
(https://www.lavision.de/en/products/flowmaster/3d-ptv-shake-the-box/). The vendor page does **not**
detail the shake step — the mechanism detail is from the method paper (Schanz et al., *Shake-The-Box*,
Exp. Fluids 2016, https://link.springer.com/article/10.1007/s00348-016-2157-1): short tracks are
extracted "by searching for **low-acceleration combinations**", extended to the next timestep by
prediction, and "an image matching scheme applied to correct prediction errors prior to reconstruction"
— then newly-entering particles are triangulated from the **residual** image.

**Two portable sub-mechanisms.**
1. **Prediction-as-detection-prior (closed detect↔link loop).** Instead of detecting independently per
   frame and then linking, use each track's predicted position at t+1 to *lower the detection
   threshold locally*. This is the single most credible attack on 6bba's node_recall 0.855 that does
   not require a retrain, because it adds recall exactly where a track already asserts a cell must be.
2. **Residual re-detection.** Subtract a fitted intensity model of every already-explained nucleus and
   re-run detection on the residual to surface the ones the first pass merged or masked.

**Falsification.** (1) is testable cheaply: for every track with a 1-frame gap, re-query the detector
heatmap in a small window around the predicted position with a lowered threshold; measure recovered GT
nodes vs added FPs, then LOEO. (2) needs the raw volumes and is a 2–4 day build. **GPU: No** for
heatmap re-query and DoG-on-residual; **Yes** if the residual pass re-runs the UNet3D.

**Overlap flag.** Sub-mechanism (1) is adjacent to — but distinct from — CW4/`liyansen`'s acceleration
lookahead: CW4 *scores* an existing candidate pair, this *creates* a candidate that does not exist.

---

### 8. Two-phase global LAP with link/split/merge scored jointly

**Mechanism `DOCUMENTED`.** Nikon NIS-Elements ships the Jaqaman/Danuser LAP tracker and documents it:
"inspired by the paper by Jaqaman, K. et al.: *Robust single-particle tracking in live-cell time-lapse
sequences*, Nature Methods 2008"; "as each track is built a Kalman filter is applied and its prediction
and error estimate are used to calculate a multidimensional Gaussian probability distribution
function"; and then the key architectural statement — "In the track processing phase, every pair of two
tracks can link (Gap Closing), branch (Splitting), merge to each other or remain independent… every of
these possibilities for all possible track pairs is given a probability and a **globally best
combination is found**" (NIS-Elements help, "Algorithm Overview",
https://www.nisoftware.net/NikonSaleApplication/Help/Docs-AR/eng_ar/p2c24s9.html; same 404 caveat as §3).

**Read for us.** Our stack runs gap repair, motion relink and safe-div as **sequential, independently
tuned heuristics**. The commercial default is one *global* second-level assignment in which gap-closing,
splitting and merging compete on a common probability scale. That is a genuinely different architecture
and it would remove the interaction hazard we already flagged internally (`p3_armb.json`: arm B's cost
consumes `prob`, which harmonic rewrites — "MUST be measured together and never assumed additive").

**Why it is only rank 8.** Our split/merge lever was already measured at **+0.0006** and closed, and
the gap-closing half is what our gap repair already does. Only the *joint global* framing is untested,
and it is a 3–5 day rebuild of the post-processing spine.

**Falsification.** Build the second-level cost matrix over track segments (end-to-start link, end-to-mid
split, mid-to-start merge, plus the null diagonal), solve one LAP, LOEO both folds; promote iff bilateral
≥ +0.005. **GPU: No.** **IP:** Jaqaman 2008 is academic with public u-track code — no exposure.

---

### 9. Cardinality-explicit tracking (PHD/CPHD) and the RFS "spawn" primitive

**Mechanism `DOCUMENTED`.** MathWorks' `trackerPHD` documents the random-finite-set formulation in
which the tracked object *is* a density whose integral is the target count: "its value at a state is
defined as the expected number of targets per unit state-space volume… Integrating D(x) over the whole
state space results in the total expected number of targets (Σwᵢ)". Birth is an explicit rate —
`BirthRate`, "the expected number of targets added in the density per unit time" — and extraction is by
weight thresholds (`ExtractionThreshold`, `ConfirmationThreshold`), with components merged when their
KL distance falls under `MergingThreshold`
(https://www.mathworks.com/help/fusion/ref/trackerphd-system-object.html).

**Why it is theoretically the right frame and practically low EV for us.** The metric's
`(1 − 0.1·(N_pred−N_est)/N_est)` term is a **cardinality** penalty, and RFS is the only tracking
framework that treats cardinality as a first-class estimand rather than a by-product; the PHD
recursion additionally carries a **spawn** term (targets born *from* existing targets) which is
literally cell division rather than a bolt-on. But redteam has already closed both channels this
attacks: the multiplier is banked on the deployed substrate (residual ≈ +0.001), and the division
mother-gate is arithmetically dead. So this buys elegance, not points.

**Overlap flag.** `novel_crossdomain` #1 (unbalanced OT) already provides a soft-cardinality knob via
marginal relaxation — PHD's `BirthRate` and OT's τ are the same knob wearing different hats. Prefer the
OT version: it is cheaper and already scoped.

**IP flag — the real one on this list.** **US10922820B2**, *Data-driven delta-generalized labeled
multi-Bernoulli tracker*, assignee **National Technology and Engineering Solutions of Sandia LLC**,
filed 2017-07-31, granted 2021-02-16, **active**
(https://patents.google.com/patent/US10922820B2/en). Claim 1 covers receiving unlabeled measurements,
generating a multi-target likelihood from persistent/birth/clutter densities, and using it to associate
persistent targets and initiate new tracks — with the specific data-driven birth rule "no target may
exist without a seed measurement". A data-driven δ-GLMB implementation could read on this. **Classical
PHD/GM-PHD (Mahler, early 2000s) is older prior art and is the safer route; avoid δ-GLMB-with-
detection-driven-birth.**

---

### 10. Disagreement-triggered cascade for mitosis — closed by our own data

**Mechanism `PATENT`.** US9430829B2 (published as US20150213302A1), *Automatic Detection Of Mitosis
Using Handcrafted And Convolutional Neural Network Features*, assignee **Case Western Reserve
University**, filed 2014-12-08, granted 2016-08-30, active
(https://patents.google.com/patent/US20150213302A1/en). Candidates come from LoG response +
thresholding; a 3-layer CNN and a 253-feature handcrafted classifier each emit a probability; **a third
Random Forest trained on stacked features is invoked only when the two disagree**.

**Why we must not build it.** The "route disagreements to an arbiter" trick only pays if the two
experts have *complementary* correct sets. Our laneD measurement is the exact opposite: on 44b6,
**b = 0** — there is not one mother where the learned ranker is right and frozen geometry is wrong, so
the MLP's correct set is a **strict subset** of geometry's and the union oracle over any
blend/gate/cascade equals geometry alone (redteam Claim 1). This mechanism is **pre-falsified for our
division head.** Logging it here so it is not re-proposed from the patent literature.

---

## 3. Explicitly re-labelled — commercial names for things we already ship

Listing these so nobody re-opens a closed lever because a vendor gave it a new name.

| Commercial name | Source | What it is for us |
|---|---|---|
| Imaris **Autoregressive Motion**: "a maximum distance disallows the connection… if the distance between the **predicted future position** of the spot and the candidate position does exceed the maximum distance" | Imaris manual `DOCUMENTED` | **Exactly Arm B's flow-compensated gate.** Every commercial tracker gates on the *predicted* position; raw-displacement gating is the outlier. This is a strong external prior that Arm B (+0.0080 pooled, P(Δ>0)=1.0 on the deployment substrate) ships — it is the industry default, not an exotic trick |
| Imaris **Connected Components** — "the only algorithm that automatically handles lineage (spots that split or merge)", by sphere overlap | Imaris manual `DOCUMENTED` | Our safe-div geometric proximity caps (`SAFE_DIV_MAX_UM=4.66`, `EXISTING_CHILD_MAX_UM=7.65`). Overlap-based division = a distance cap in disguise |
| Imaris **Max Gap Size**; Aivia **Minimum Track Length**; NVIDIA **`probationAge`** | three vendors `DOCUMENTED` | `BIOHUB_GAP_CLOSE_MAX_GAP=2`, `BIOHUB_OUTPUT_MIN_TRACK_LEN=6` |
| MATLAB **GNN + Munkres** ("Munkres is the only assignment algorithm that guarantees an optimal solution, but it is also the slowest") | `DOCUMENTED` | our LAP/ILP linker |
| PTV **four-frame low-acceleration matching** (STB "searching for low-acceleration combinations") | Schanz 2016 / LaVision | **CW4 / `liyansen`'s 3-frame forward-acceleration lookahead is a re-labelled, one-frame-shorter version of a 1990s PTV standard.** Worth knowing the canonical form uses **four** frames — a free variant to sweep |
| NVIDIA **Shadow Tracking** (`maxShadowTrackingAge`), **Late Activation**, `minTrackerConfidence` | `DOCUMENTED` | gap closing + short-track filter + score threshold. The one non-redundant piece is `earlyTerminationAge` — killing a *tentative* track fast on early misses, which is a different rule from deleting short tracks at the end. Low EV |
| MHT "deferred decision" / long-term conflict resolution | radar; MATLAB `trackerTOMHT` | already `novel_crossdomain` #5 (BiologicalNeeds mitosis-aware MHT) |
| PHD `BirthRate` / marginal mass | MATLAB | already `novel_crossdomain` #1 (unbalanced-OT marginal relaxation τ) |

---

## 4. One vendor fact worth treating as a constraint, not a lever

ZEISS/arivis states the sampling condition for reliable geometric tracking outright: **"The typical
movement of objects from one time-point to the next is no more than 20% of the typical distance between
neighboring objects."**
(https://knowledge.zeiss.com/rms/en/arivis-pro/time-lapse-analysis/tracking-in-arivis-pro,
`DOCUMENTED`.)

Our numbers: median inter-frame step **1.8 µm** (CW3) against nearest-neighbour spacing **~9–10 µm**
(CW1) → **≈ 18–19%**. We are sitting *on* the vendor-stated boundary of what nearest-neighbour geometry
can resolve. `INFERENCE`, but a well-supported one: this is quantitative external corroboration that
squeezing the geometric association further is near its information limit, and that the remaining
headroom is in **appearance/learned features** (§3 above) and in **detection quality**, exactly where
redteam Claim 3 and the H1 retrain thesis already point. It also offers a mechanism for why the whole
public field plateaus at adj_edge_J ≈ 0.90–0.91 with geometry-only post-processing.

---

## 5. Patents examined — mechanism and IP posture

| Patent | Assignee | Filed / granted | Mechanism | Our exposure |
|---|---|---|---|---|
| **US9542591B2** | Ares Trading SA (orig. Progyny / Auxogyn, "Eeva") | 2014-02-28 / 2017-01-10 | Multi-hypothesis cell configurations from Hessian boundary segments; hypotheses by perturbing prior-frame ellipse params; EM refinement; PGM + belief propagation with a **non-decreasing cell count** constraint | Taking only the generic *count-consistency prior* (§6) — low risk. **Do not** reproduce the hypothesis-perturbation + EM + PGM apparatus |
| **US10922820B2** | National Technology & Engineering Solutions of Sandia LLC | 2017-07-31 / 2021-02-16 | Data-driven δ-GLMB; labeled-Poisson birth with "no target may exist without a seed measurement"; full cardinality distribution via marginalisation | **Highest flagged risk on this list** if we implement a data-driven δ-GLMB. Use classical GM-PHD or the OT formulation instead |
| **US9430829B2** (pub. US20150213302A1) | Case Western Reserve University | 2014-12-08 / 2016-08-30 | LoG candidate generation; 3-layer CNN + 253 handcrafted features; **Random-Forest arbiter invoked only on classifier disagreement** | Not building it (pre-falsified, §10). If ever revisited, the disagreement-cascade is the claimed element |
| **US9896720** | (single-molecule FISH context) | — | "three-dimensional linear Laplacian of Gaussian filter designed to enhance spot-like signals" in a 3D stack | Generic 3D LoG is Marr–Hildreth prior art; low risk. Noted only to show 3D-LoG appears in claim language |

**General IP posture.** Reading patents is unrestricted; *practising* a claim is what infringes.
Everything ranked 1–5 above is either textbook prior art (LLR track scoring, DoG detection, LAP
assignment, distance+intensity cost) or permissively-licensed academic work (BYTE). The two places to
be careful are **δ-GLMB with detection-driven birth** (Sandia) and the **multi-hypothesis + PGM
cell-count apparatus** (Ares Trading). A Kaggle notebook is research use, not a product, so practical
risk is low — but the mechanism choices above avoid the claimed combinations anyway, at no cost to us.

**Not found:** no cell-tracking or spot-detection patent assigned to **Bitplane / Oxford Instruments**
surfaced in searching. Imaris appears in the patent corpus only as a *tool used by* third-party
applicants (e.g. WO2019002621A1, US20180306780A1). `UNVERIFIED` whether such a family exists — absence
of a search hit is not proof of absence.

---

## 6. What I could not verify — say so plainly

- **Revvity Harmony / Columbus** "Track Objects" building-block internals: **not publicly documented.**
  Product pages describe building blocks and PhenoLOGIC/Phenologic.AI at a `MARKETING` level only
  (https://www.revvity.com/category/cellular-imaging-software). No algorithm, cost function or
  division rule is disclosed. Industrial HCS practice therefore could **not** be characterised from
  vendor primary sources beyond the general observation that the building blocks are per-frame
  segmentation plus simple linking.
- **Molecular Devices MetaXpress**: only a "proprietary adaptive background correction algorithm" for
  segmentation is described in the public datasheets; **no tracking-module algorithm documentation
  found.**
- **Sartorius Incucyte**: Cell-by-Cell is segmentation + classification; **no lineage/division tracking
  algorithm documented** publicly.
- **ZEISS arivis Vision4D / Pro**: cost function, max-gap and division handling are **not documented**.
  The help pages give qualitative guidance only ("Reduce the search radius…", "Don't allow fusions or
  divisions unless this is necessary"). Only the 20%-spacing rule (§4) is a hard, quotable statement.
- **Leica Aivia**: parameter names are documented (`Maximum Search Distance`, `Motion vs Intensity`,
  `Minimum Track Length`, `Average Object Radius`, `Min Edge Intensity`); the **linking algorithm itself
  and the mitosis/lineage logic are not.** "Seamlessly blends the advantages of motion and intensity
  based object tracking" (Aivia 7 page) is `MARKETING`, not an algorithm description — do not cite it
  as mechanism.
- **LaVision DaVis**: the vendor page confirms the product implements IPR + 4D-PTV but **does not
  document the shake/predictor/residual steps**; those come from the academic paper.
- **Dantec Dynamics DynamicStudio / TSI**: brochures confirm volumetric-PTV and Tomo-PTV products exist
  but **document no association algorithm**. The 3-frame/4-frame PTV matching literature is academic,
  not vendor-disclosed.
- **Nikon NIS-Elements**: the two most valuable statements (mass-conservation splitting; Jaqaman/Kalman
  + globally-best track-pair processing) come from the help system's "Algorithm Overview" page, which
  **404'd on direct fetch**; both were surfaced independently by two separate queries against the
  indexed page. Treated as `DOCUMENTED (indexed)`, flagged as not re-verified.
- **syGlass, MetaCell, Elucidata, Media Cybernetics**: searched once, jointly; **no cell-tracking
  algorithm documentation surfaced.** Not exhaustively searched — do not read this as a finding.
- **TWEAK's "universal bio-cell plugin"** (rank 2, claiming instant +0.030/0.040/0.050) remains
  `MARKETING` from an AI-agent company, unreproducible — unchanged from `competitive_refresh` §2.

---

## 7. Bottom line

The proprietary pool does **not** contain a hidden algorithm that closes our 0.035 gap — the commercial
platforms are, algorithmically, LAP + Kalman/AR1 + overlap-lineage, i.e. our stack with different
parameter names. Its value is in three specific places:

1. **Three cheap, CPU-only, deployment-substrate-testable levers we genuinely do not do** — tiered
   two-round association (#1), LLR track scoring in place of length filtering (#2), and an
   appearance/mass-continuity term in the hand-built costs (#3). All three run on
   `artifacts/kaggle/p0strict_cache/graphs` without touching the parked Arm B GPU green-light.
2. **A concrete specification for the sub-voxel lane already in flight** — fit anisotropic σ_xy/σ_z,
   because the commercial default explicitly models axial PSF elongation and our data is 4x
   anisotropic (#4).
3. **External corroboration of two internal positions**: Arm B's prediction-gated relinking is the
   universal commercial default (so shipping it is the conservative choice, not the adventurous one),
   and at ~19% step-to-spacing we are at the vendor-stated limit of geometric association — which is
   independent, non-Kaggle evidence for the H1 retrain thesis over more post-processing.

Nothing here displaces the retrain bet. Items 1–3 are worth roughly a day each and are the only
proprietary-derived mechanisms that clear the bar of being both new to us and cheap.

---

## Sources

Vendor documentation — [Imaris V7.6 Reference Manual (Bitplane / Oxford Instruments)](https://www.ijm.fr/wp-content/uploads/2022/01/Imaris-Reference-Manual-7_6_0.pdf) · [Imaris for Tracking](https://imaris.oxinst.com/products/imaris-for-tracking) · [Imaris 9.2 release notes (linear-assignment solver)](https://imaris.oxinst.com/support/imaris-release-notes/9-2-0) · [Imaris PSF-elongation / Estimated Z Diameter usage](https://www.allevi3d.com/livedead-assay-quantification-imaris/) · [Aivia 3D Object Tracking recipe](https://aivia-software.atlassian.net/wiki/spaces/AW/pages/96469083/3D+Object+Tracking) · [ZEISS arivis Pro — Tracking](https://knowledge.zeiss.com/rms/en/arivis-pro/time-lapse-analysis/tracking-in-arivis-pro) · [NIS-Elements Algorithm Overview](https://www.nisoftware.net/NikonSaleApplication/Help/Docs-AR/eng_ar/p2c24s9.html) · [NIS-Elements Binary Tracking](https://www.nisoftware.net/NikonSaleApplication/Help/Docs-AR/eng_ar/track.binary.html) · [Revvity Cellular Imaging Software](https://www.revvity.com/category/cellular-imaging-software) · [LaVision FlowMaster 4D-PTV / Shake-the-Box](https://www.lavision.de/en/products/flowmaster/3d-ptv-shake-the-box/) · [Dantec DynamicStudio brochure](https://pdf.directindustry.com/pdf/dantec-dynamics-s/dynamicstudio/15753-405793.html) · [Sartorius Incucyte Cell-by-Cell](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-software/incucyte-cell-by-cell-analysis-software)

Adjacent proprietary tracking — [MATLAB Introduction to Multiple Target Tracking](https://www.mathworks.com/help/fusion/ug/introduction-to-multiple-target-tracking.html) · [trackerGNN](https://www.mathworks.com/help/fusion/ref/trackergnn-system-object.html) · [trackerTOMHT](https://in.mathworks.com/help/fusion/ref/trackertomht-system-object.html) · [trackerPHD](https://www.mathworks.com/help/fusion/ref/trackerphd-system-object.html) · [NVIDIA DeepStream NvMultiObjectTracker Parameter Tuning Guide](https://docs.nvidia.com/metropolis/deepstream/6.4/dev-guide/text/DS_plugin_NvMultiObjectTracker_parameter_tuning_guide.html) · [ByteTrack / BYTE two-stage association](https://trackers.roboflow.com/latest/trackers/bytetrack/) · [Schanz et al., Shake-The-Box, Exp. Fluids 2016](https://link.springer.com/article/10.1007/s00348-016-2157-1) · [Jaqaman et al., Nat. Methods 2008](https://pubmed.ncbi.nlm.nih.gov/18641657/)

Patents — [US9542591B2 (Ares Trading SA)](https://patents.google.com/patent/US9542591B2/en) · [US10922820B2 (Sandia / NTESS)](https://patents.google.com/patent/US10922820B2/en) · [US20150213302A1 → US9430829B2 (Case Western Reserve University)](https://patents.google.com/patent/US20150213302A1/en) · [US9896720 (3D LoG spot detection)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9896720)
