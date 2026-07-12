# Lane 02 — Track-before-detect & motion-compensated temporal integration

**Date:** 2026-07-12
**Scope:** cross-disciplinary survey (radar/sonar, astronomy, video/biomedical vision,
motion estimation) for methods that recover missed nuclear endpoints and raise the
detection ceiling in the Biohub Cell Tracking 2026 problem, feeding the
"motion-compensated 4D track-before-detect lineage field" bet in
[../../THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md](../../THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md)
(Hedge B / §2.3, and the Main Bet / §2.1).

---

## Executive summary: 5 highest-value transferable methods

1. **Dynamic-programming track-before-detect (DP-TBD)** (radar). Instead of thresholding each
   frame independently, accumulate a state-transition-weighted score along all admissible
   trajectories through a stack of *raw, sub-threshold* response frames, and only threshold the
   final merit function. This is the most direct formal analogue of our "integrate weak evidence
   along candidate trajectory tubes, declare a node only after temporal integration" bet — it is a
   textbook implementation of exactly that idea, with 40+ years of false-alarm theory behind it.
   [Buzzi et al., IEEE TSP 2013](https://ieeexplore.ieee.org/document/6475194/); survey in
   [MDPI Remote Sensing 2024](https://www.mdpi.com/2072-4292/16/14/2639).

2. **Shift-and-stack / digital tracking (KBMOD)** (astronomy). Search over a discretized
   velocity/acceleration phase space; for each candidate trajectory, warp (shift) neighboring
   frames into the object's co-moving frame and stack (sum/median) the pixel likelihoods; keep
   only trajectories whose stacked likelihood clears a threshold *no single frame could reach
   alone*. This is a GPU-parallel, embarrassingly-parallelizable version of our "warp frames into
   material coordinates, integrate along the tube" step, and it comes with a documented ~10x
   sensitivity gain over single-frame detection. [Digital tracking, Heinze et al., arXiv:1508.01599](https://arxiv.org/pdf/1508.01599);
   [Whidden et al., "Fast algorithms for slow moving asteroids," arXiv:1901.02492](https://arxiv.org/pdf/1901.02492);
   [KBMOD code, DiRAC Institute](https://github.com/dirac-institute/kbmod).

3. **ByteTrack's low-confidence second-tier association**. A cheap, already-proven, near-zero-cost
   pattern: keep every detection below the operating threshold (don't discard it), then in a
   second matching pass link low-score boxes to *already-confirmed* tracks by motion/IoU
   consistency only, never to start new tracks. This is directly implementable this week on our
   existing detector's response volumes as the first rung of the track-before-detect ladder before
   any new model is trained. [Zhang et al., ECCV 2022, arXiv:2110.06864](https://arxiv.org/abs/2110.06864);
   adaptive-threshold variant [arXiv:2312.01650](https://arxiv.org/html/2312.01650v2).

4. **ELEPHANT's incremental predict→verify→retrain loop** (3D nucleus tracking). Rather than one
   static detector, ELEPHANT closes the loop: predict nuclei + flow, propagate track hypotheses,
   let a human (or here, an automated agreement/consistency check) verify/reject, then feed
   corrections back as new supervision. The transferable idea for us is the *architecture* of
   coupling flow-based propagation to detection confidence, and treating "detected only when
   temporally supported" as a first-class state, not a post-hoc trick. [Sugawara, Čapek et al.,
   eLife 2022, "Tracking cell lineages in 3D by incremental deep learning," DOI
   10.7554/eLife.69380](https://elifesciences.org/articles/69380); code/platform: ELEPHANT (Mastodon/Fiji
   plugin), summarized alongside Trackastra/Ultrack in recent tool comparisons.

5. **Matched-filter / null-path likelihood-ratio thresholding** (radar CFAR + astronomy weak-source
   detection). The common thread across every mature TBD literature is: never threshold raw
   integrated score in isolation — always compare the candidate-tube score against a
   *matched background/null estimate* (CFAR local noise estimate; astronomy's local sigma-clipped
   background; radar's likelihood-ratio vs. no-target hypothesis) so the false-alarm rate stays
   calibrated as you lower the single-frame threshold. This is the mechanism that keeps the
   competition's **count penalty** (over-prediction penalized via `(N_pred - N_est)/N_est`) from
   exploding when we start acting on weak evidence. [CFAR overview, Wikipedia](https://en.wikipedia.org/wiki/Constant_false_alarm_rate);
   [multi-frame-integration CFAR under heavy-tailed clutter, IET Signal Processing 2023](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sil2.12145);
   [matched-filter false-detection-probability theory, A&A 2016](https://www.aanda.org/articles/aa/full_html/2016/05/aa27463-15/aa27463-15.html).

---

## Findings table

| Method | Source | Core idea | Why it fits our metric/constraints | Maps to bet | First experiment + kill-gate | Cost |
|---|---|---|---|---|---|---|
| DP-TBD (dynamic programming track-before-detect) | [Buzzi et al. 2013](https://ieeexplore.ieee.org/document/6475194/), [survey 2024](https://www.mdpi.com/2072-4292/16/14/2639) | Recursive merit function `M_t(x) = max over x_{t-1} [M_{t-1}(x_{t-1}) + transition(x_{t-1}->x) ] + response(x,t)`, accumulated over T frames on *raw* (unthresholded) response grids before any detection decision | Directly operationalizes "declare a node only after temporal integration"; state-transition term = our motion-compensation prior; complexity is linear in grid size x T, cheap on CPU | Hedge B (§2.3) and Main Bet (§2.1) decoder step 4-5 | Run DP-TBD over 5-7 frame windows of raw detector heatmaps restricted to bracketed-miss regions (frames flanking a known missed endpoint); kill if recovered-endpoint recall <20% or marginal precision <70% (matches architecture doc's stated gate) | Low — pure CPU, uses existing response volumes |
| PF-TBD (particle-filter track-before-detect) | [comparison of PF for TBD, IEEE](https://www.researchgate.net/publication/4221007_A_comparison_of_particle_filters_for_recursive_track-before-detect); [Bayesian TBD for passive radar, EURASIP 2013](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2013-45) | Sample-based recursive Bayesian filter over continuous state (position+velocity) directly from pixel intensities, propagating a posterior "target present" probability instead of a hard detection | Handles continuous, non-grid-aligned 3D anisotropic motion better than DP's discretized states; gives calibrated existence probability per track, which is exactly the "existence head" in our Lineage Field design | Main Bet §2.1 "existence" head | Prototype only if DP-TBD kill-gate passes and grid discretization artifacts appear at our anisotropic voxel scale (1.625 x 0.406 x 0.406 um); compare recall/precision vs DP-TBD on the same bracketed-miss set | Medium — particle degeneracy tuning, more engineering than DP |
| Shift-and-stack / digital tracking (KBMOD-style) | [Digital tracking, arXiv:1508.01599](https://arxiv.org/pdf/1508.01599); [Whidden et al. arXiv:1901.02492](https://arxiv.org/pdf/1901.02492); [KBMOD repo](https://github.com/dirac-institute/kbmod) | Discretize candidate velocity (here: local tissue-flow-corrected displacement) space; for each candidate, shift+stack raw frames into the candidate's co-moving frame; score stacked likelihood; ~10x sensitivity gain documented over single-frame limits | Same operation as "warp neighboring frames into a track's material coordinates + integrate," just phrased for point sources; GPU-native (their CUDA kernel design generalizes to our T4 budget) and gives a concrete recipe for the trajectory-tube search grid | Main Bet §2.1 steps 1-4 (motion-compensated tube search) | On one hard crop, replace bespoke motion-compensation code with a KBMOD-style discretized-displacement stacking search seeded from local tissue flow (+/- residual grid); measure SNR gain of stacked vs. single-frame response at true missed-endpoint locations | Medium — needs GPU kernel or vectorized numpy/cupy; reuse KBMOD's open pipeline structure as reference, don't need their exact code |
| YOSO Gaussian Motion Filter (continuous alternative to discretized shift-and-stack) | [arXiv:2605.06913](https://arxiv.org/abs/2605.06913) | Replaces discrete velocity trials with a continuous motion-matched Gaussian filter that amplifies along-trajectory signal while suppressing static background, avoiding combinatorial blow-up of trial velocities | Removes the main cost objection to shift-and-stack (grid size grows with search-space dimension); relevant once we need per-cell local (not just global) motion search | Main Bet §2.1, efficiency concern for step 4 | Compare compute cost and recall of GMoF-style continuous filter vs. discretized DP-TBD on same bracketed-miss benchmark; kill if no wall-clock win under Kaggle T4 12h budget | Medium-high — newer, less battle-tested method (2026 preprint) |
| ByteTrack low-confidence second-tier matching | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864); [adaptive threshold arXiv:2312.01650](https://arxiv.org/html/2312.01650v2) | Never discard sub-threshold detections outright; in a second association pass, match them only to *already-confirmed* tracklets via IoU/motion similarity; never spawn new tracks from low-confidence boxes alone | Cheapest possible first rung of TBD: reuses existing single-frame detector output, adds one extra matching step, and structurally cannot inflate node count (`N_pred`) because it only extends existing tracks, directly respecting the count penalty | Hedge B §2.3 ("sample low-threshold evidence... promote missing nodes only with both temporal and image support") | Re-run current linker with a two-tier threshold (e.g., 0.5 / 0.15) on the two labelled embryos; measure recovered endpoints and marginal precision; this is the cheapest possible falsification of the whole TBD thesis and should run *before* DP-TBD | Very low — reuses existing detector, ~1 day |
| Two-threshold radar TBD (low provisional threshold + high confirm threshold) | [likelihood-ratio TBD patents/background](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8026842) | Set a very permissive first threshold to admit many weak candidates, then confirm only those whose accumulated likelihood ratio over multiple frames clears a second, high threshold | Formalizes the ByteTrack pattern with an explicit Bayesian confirmation criterion; gives a principled way to set the "how weak is weak enough" hyperparameter via target false-alarm rate rather than by hand-tuning | Hedge B §2.3 | Fit a two-threshold scheme where the low threshold is chosen from ROC on the labelled embryos and the high (confirm) threshold is chosen to hold the metric's count-penalty term near zero at held-out fold; kill if no monotonic recall/precision improvement over single threshold | Low |
| ELEPHANT incremental predict-detect-verify-retrain loop | [eLife 2022, DOI 10.7554/eLife.69380](https://elifesciences.org/articles/69380) | Couples per-frame nucleus detection with a flow/motion model, propagates hypotheses forward, and treats prediction-vs-verification disagreement as the training signal for the next round | Architecture template for closing the loop between our lineage-field's flow head and its existence head; also directly relevant to Layer-1's "teacher-student pseudo-trajectories gated by temporal persistence" | Data layer §1.2/1.3 and Main Bet flow+existence heads | Implement one predict→propagate→flag-disagreement cycle on the March-22 dense public trajectories (already in the trajectory lake) using our current detector + a simple flow estimate; check whether disagreement regions correlate with known missed-endpoint regions | Medium |
| MPM (Motion and Position Map) | [Hayashida, Nishimura, Bise, CVPR 2020, arXiv:2002.10749](https://arxiv.org/pdf/2002.10749) | Jointly regresses a single dense map encoding both cell position and per-cell displacement vector to the next frame, so detection and association share one representation instead of being solved independently, including for division events | Directly matches our Main Bet's "existence/motion jointly decoded" design; published +5.2% over prior best on cell-tracking benchmarks, i.e. empirical evidence that joint detect+motion beats separate detect-then-link on cell data specifically | Main Bet §2.1 (nuclear evidence + forward motion heads) | Reproduce MPM-style joint position+motion regression as a lightweight auxiliary head on our existing detector backbone using March-22 dense tracks as supervision; compare endpoint recall against detect-then-link baseline on held-out embryo | Medium |
| Viterbi global track linking (Magnusson et al.) | [Magnusson & Jaldén, ISBI 2012/2015](https://pubmed.ncbi.nlm.nih.gov/25415983/); [batch iterative Viterbi](https://www.researchgate.net/publication/261206085_A_batch_algorithm_using_iterative_application_of_the_Viterbi_algorithm_to_track_cells_and_construct_cell_lineages) | Iteratively finds the single highest-scoring complete track (via Viterbi/shortest-path in a space-time graph) across the *whole* sequence, adds it, removes its nodes, repeats — winner of multiple ISBI Cell Tracking Challenge years | This is essentially DP-TBD already adapted to cell tracking and already validated on ISBI benchmarks — strong prior that the DP-TBD family transfers to this exact application domain | Main Bet §2.1 decoder / structured solver step 7 | If DP-TBD prototype (row 1) works, compare directly against a from-scratch Viterbi-linker reimplementation on the same bracketed-miss set — this tells us whether our gain comes from "temporal integration before detection" (novel) or just "better global linking" (already known and possibly already captured by baseline+Trackastra) | Medium |
| Mitosis-aware multi-hypothesis tracker (MHT) | [arXiv:2403.15011](https://arxiv.org/pdf/2403.15011) | Extends classical MHT (existing/false-alarm/new-target hypotheses) with an explicit mitosis-aware assignment and aleatoric uncertainty, resolving long-term association conflicts around divisions | Relevant to the Division posterior (§2.5) more than pure TBD, but the "existing/false-alarm/new" MHT hypothesis space is a clean formalization of exactly the ambiguity our count-penalty-aware component selection (§2.6) needs to reason about | §2.5 Division posterior, §2.6 metric-aware selection | Out of scope for lane 02's primary TBD focus; hand off to lane 04 (lineage/division) for deeper follow-up | Medium |
| Optical flow / 3D flow field estimation for microscopy | [super-voxel optical flow, Bioinformatics 2013](https://academic.oup.com/bioinformatics/article/29/3/373/257856); [3D flow field estimation for live-cell fluorescence, Bioinformatics 2019](https://doi.org/10.1093/bioinformatics/btz780) | Dense motion field estimation specialized for low-texture, noisy fluorescence volumes using super-voxel MRF regularization or variational 3D flow, rather than generic natural-image optical flow (RAFT etc.) | This is the concrete method to build the "local tissue-flow" warp field our Main Bet needs at step 2-3 (estimate local tissue flow, warp evidence into material coordinates) — purpose-built for exactly our imaging modality and noise regime | Main Bet §2.1 decoder steps 2-3 (motion field, warping) | Benchmark super-voxel optical flow vs. a lightweight learned 3D flow head (trained on public dense tracks with known ground-truth displacement) for warping accuracy (median residual in um) on March-22 data; kill flow candidates whose warp residual exceeds the 3 um Layer-1 gate | Medium |
| Coherent Point Drift (CPD) / non-rigid point-set registration | [Myronenko & Song, NeurIPS 2006](https://papers.nips.cc/paper/2962-non-rigid-point-set-registration-coherent-point-drift) | Probabilistic (GMM/EM) non-rigid registration between two point sets with a motion-coherence prior, without needing pre-established correspondences | Alternative/complementary to dense optical flow for warping sparse point-cloud-like nucleus centroids (rather than dense image volumes) into a track's coordinates — cheaper than dense flow when we only need centroid-level correspondence | Main Bet §2.1 (alternative warp mechanism); overlaps Hedge C §2.4 (local CPD for OT) | Compare CPD-based point warping vs. dense flow warping for accuracy and runtime on a dense-point subset; useful mainly if dense flow proves too slow for the Kaggle T4 budget | Low-medium |
| Test-time augmentation (TTA) for 3D detection | [nnU-Net TTA](https://www.emergentmind.com/topics/nnunet); [cell-segmentation TTA, Sci Reports 2020](https://www.nature.com/articles/s41598-020-61808-3) | Average predictions over flips/mirrors (and here, small sub-voxel shifts) at inference time to reduce detector variance and recover borderline detections without retraining | Cheap, zero-training way to squeeze extra recall out of the existing detector before committing to any TBD architecture — a sanity floor to compare TBD gains against | Baseline hardening, prerequisite sanity check before Hedge B/Main Bet | Apply standard nnU-Net-style flip TTA + small anisotropic sub-voxel jitter to current detector; measure how much of the "missed endpoint" gap TTA alone closes — any TBD method must beat this for free-lunch comparison | Very low |
| Tubelet Proposal Networks / T-CNN | [T-CNN, arXiv:1604.02532](https://arxiv.org/abs/1604.02532); [Tubelet Proposal Networks](https://www.researchgate.net/publication/320971846_Object_Detection_in_Videos_with_Tubelet_Proposal_Networks) | Propose spatio-temporal tubes (not per-frame boxes) directly, then classify/regress the whole tube, aggregating appearance evidence across the tube's frames before final detection decision | Conceptually the "trajectory tube" unit of prediction our architecture doc calls for, validated on ImageNet-VID; mainly useful as an architectural precedent, less directly portable since it targets 2D video object categories, not sparse 3D point sources | Main Bet §2.1 conceptual framing | Low priority: read as design reference only; not worth reimplementing given our sources are point-like nuclei, not extended objects with rich per-frame appearance | Reference only, no build cost |
| Matched filter for sub-threshold point-source detection | [matched filter false-detection theory, A&A 2016/2017](https://www.aanda.org/articles/aa/full_html/2017/08/aa29330-16/aa29330-16.html) | Convolve raw response with the known/estimated point-spread function (PSF) shape and derive an analytically correct false-detection probability, rather than an empirical threshold | Our nuclei have a known approximate 3D anisotropic-Gaussian PSF shape (per project's known voxel/PSF anisotropy) — this gives a principled, closed-form way to convert "match to expected nucleus shape" into a calibrated p-value, directly usable as the response channel DP-TBD/shift-stack integrate over | Main Bet §2.1 "nuclear evidence" head; feeds §2.6 calibration | Fit a 3D anisotropic-Gaussian matched filter to the current response volumes and check whether its calibrated p-value at true missed-endpoint locations is measurably elevated vs. background — a cheap pre-check before building any TBD integration on top | Low |
| CFAR (constant false alarm rate) local background estimation | [CFAR overview](https://en.wikipedia.org/wiki/Constant_false_alarm_rate); [multi-frame CFAR, IET 2023](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sil2.12145) | Adapt the detection threshold locally based on estimated local noise/clutter power, so that false-alarm rate stays constant even as background density/intensity varies across the volume (e.g. dense vs. sparse embryo regions) | Zebrafish nucleus density and intensity vary hugely across developmental stage/depth; a fixed global threshold for "promote a weak-evidence node" will over-trigger in dense/bright regions and under-trigger in sparse/dim ones — CFAR-style local normalization is the standard fix and maps directly onto our count-penalty risk | §2.6 metric-aware selection; guards Hedge B/Main Bet against the count penalty | Compute a local background/noise estimate per spatial region (density/depth/intensity bucket, per Layer-1's regime index) and use it to locally normalize the promotion threshold for weak-evidence nodes; verify the global count-penalty term stays flat across regimes | Low |

---

## The single most promising concrete algorithm to prototype first

**Two-tier confidence + DP-TBD hybrid on existing raw response volumes**, i.e. do the ByteTrack
cheap check first, then escalate to DP-TBD only on the crops where it doesn't already resolve the
miss. This ordering is chosen because it is the cheapest possible path to falsifying or confirming
the whole track-before-detect thesis (matches the architecture doc's "decisive first experiment,"
§ "The decisive first experiment") before any GPU training is committed.

### Step-by-step

1. **Identify the test set.** Take the bracketed-miss analysis already planned in Phase 0 (§ Phase 0
   item 1 of the architecture doc): frames where a track has a confirmed node at `t-k` and `t+k`
   but no detection at some `t` in between, on the two labelled embryos (44b6, 6bba). This is the
   ground-truth "genuinely missed endpoint" set.
2. **Cheap tier (ByteTrack-style).** Re-run the existing single-frame detector at a much lower
   confidence threshold (e.g. 0.15 instead of 0.5) on just these bracketed windows. Attempt to link
   the low-confidence detections to the existing confirmed tracklets on either side using
   motion-consistency (predicted position from neighboring confirmed nodes via local tissue flow)
   instead of raw IoU. Do **not** allow a low-confidence detection to start a new, unlinked track.
   Record recovered-endpoint count and false-positive count.
3. **Escalate remaining misses to DP-TBD.** For windows the cheap tier still misses, pull the raw
   (pre-NMS, pre-threshold) response volume for a 5-7 frame window centered on the gap. Build a
   discretized local-displacement state space seeded from the local tissue-flow estimate (super-voxel
   optical flow or simple frame-to-frame registration) restricted to a small physically plausible
   radius (use known nucleus speed priors from the public dense tracks). Run the DP-TBD recursion:
   `M_t(x) = response(x, t) + max_{x' in neighborhood(x)} [M_{t-1}(x') + log P(x | x')]`
   where `log P(x|x')` penalizes deviation from the flow-predicted displacement. Take the
   maximum-merit path through the window as the candidate trajectory.
4. **Null-path calibration (CFAR/matched-filter step).** For every candidate recovered node, compute
   the same accumulated merit score along a small number of *matched null paths* — same window,
   same local motion statistics, but centered on random nearby locations with no bracketing track.
   Compute a local z-score / likelihood ratio of the true candidate vs. this null distribution.
   Only promote nodes whose z-score clears a threshold chosen from the labelled-embryo ROC. This
   step is what prevents the count penalty from firing.
5. **Score.** Compute recovered-endpoint recall (fraction of the bracketed-miss set now correctly
   filled) and marginal precision (of all newly promoted nodes, what fraction correspond to true
   missed endpoints vs. spurious insertions), plus the actual competition metric delta on both
   embryos with 5-fold cross-validation as specified in the architecture doc's gates.
6. **Freeze and reverse-direction check.** Freeze all thresholds/parameters chosen on one embryo,
   apply unchanged to the other, and confirm the result holds in both directions (architecture doc
   Layer-2 gate: "freeze configuration on one embryo; transfer unchanged; reverse direction").

### Falsification gate (must pass both, matching the architecture doc's stated numbers)

- **Recall of missed endpoints:** >= 20% of the bracketed-miss set recovered.
- **Marginal precision:** >= 70% of newly promoted nodes are true positives (i.e. do not blow the
  count-penalty term).
- **Or, directly:** >= +0.005 exact competition score on both embryos.

If none of these hold after the cheap tier *and* the DP-TBD escalation, kill the whole
track-before-detect thesis (per portfolio kill-gate) and reallocate that 20% compute budget to the
Lagrangian Lineage Field or selective baseline repair instead — do not proceed to training a new
backbone (Hedge B exists precisely to test this cheaply before Main Bet §2.1's expensive model).

---

## Pitfalls / what will raise false positives or violate the count penalty

- **Discretization aliasing in DP-TBD / shift-and-stack.** If the candidate-displacement grid is
  too coarse relative to our anisotropic voxel size (z step 1.625 um vs. xy 0.406 um), true tracks
  will fall between grid points and either get missed or get a systematically biased position,
  inflating both false negatives and false positives simultaneously. Astronomy TBD literature spends
  substantial effort on grid resolution vs. compute tradeoffs for exactly this reason — budget for
  it explicitly rather than picking a grid size arbitrarily.
- **Lowering the single-frame threshold without a matched-null comparison.** Every literature
  (radar CFAR, astronomy matched-filter, ByteTrack) is unanimous: never simply lower a per-frame
  score threshold to gain recall. Doing so without a background/null-path comparison will directly
  increase `N_pred` and can flip `adj_edge_J` negative through the `(1 - 0.1*(N_pred-N_est)/N_est)`
  term. Any recovered node must clear a *local*, motion-aware null comparison, not a static global
  cutoff.
- **Flow/warp error compounding over multi-frame integration.** DP-TBD and shift-and-stack both
  assume the motion model used to warp/predict displacement is accurate. In regions of high local
  strain (rapid morphogenesis, divisions, sudden global jumps already flagged in the architecture
  doc's regime index) a wrong flow estimate will smear evidence across the wrong path and manufacture
  spurious high-merit trajectories — these are exactly the "severe global jumps" case the
  architecture doc already routes to registration + re-association (§3.3) rather than TBD.
  Track-before-detect should be explicitly *disabled or down-weighted* in flagged high-strain/frozen-frame
  regimes rather than applied uniformly.
- **MHT/DP state-space explosion near divisions.** Classical single-target TBD assumes one trajectory
  per tube; near a division the state space branches. Treat division windows as out-of-scope for
  the TBD prototype (hand to the dedicated division posterior, §2.5) rather than trying to make DP-TBD
  division-aware in the first pass — conflating the two will make both harder to falsify cleanly.
- **Confusing "recovered a missed endpoint" with "created a new, real cell."** Because our metric
  penalizes over-prediction relative to an *estimated* true count, a TBD method that is good at
  finding real signal but bad at rejecting look-alike debris/autofluorescence will still hurt score
  even at high raw sensitivity. The null-path/z-score calibration step is not optional polish — it
  is the mechanism that keeps this method compatible with the metric's asymmetric penalty.
- **Viterbi/global-linking overlap risk.** Because the Cell Tracking Challenge's own winning method
  (Magnusson's Viterbi linker) is already essentially a cell-tracking-flavored DP-TBD, there is a
  real risk that any gain we measure is just "better linking of already-visible-but-discarded
  detections" (already partially captured by the existing baseline + Trackastra) rather than genuine
  "recovering evidence no single frame could see." The step in the findings table comparing DP-TBD
  against a from-scratch Viterbi-linker reimplementation exists specifically to separate these two
  effects — without it we risk claiming credit for the wrong mechanism.

---

## Open questions

1. **What is the actual PSF/point-response shape of a zebrafish nucleus detection in our current
   detector's raw output** (pre-NMS heatmap or logit volume)? The matched-filter and DP-TBD
   approaches both need this to build the "response(x,t)" term; if the current detector's raw
   output is already a well-calibrated probability rather than a raw matched-filter response, TBD's
   marginal gain over simple threshold-lowering may be smaller than in radar/astronomy where the
   raw sensor signal genuinely contains sub-visual information invisible after single-frame
   detection.
2. **How much of the current detector's near-ceiling recall (~0.88 min-fold) is truly a
   single-frame information limit vs. an NMS/threshold artifact?** If a large chunk of the
   remaining misses are recoverable purely by TTA (test-time flip/jitter averaging) or a lower
   static threshold with ordinary linking, that would mean the missed endpoints are not "TBD-shaped"
   problems at all, and the more expensive DP-TBD/shift-and-stack machinery would be solving a
   smaller residual than hoped. The TTA row in the findings table is explicitly there to measure
   this free-lunch floor first.
3. **Is local tissue flow accurate enough, at our label sparsity, to seed the state-transition /
   displacement prior DP-TBD needs?** Public dense March-22 trajectories give ground truth to
   validate this, but the flow model must generalize to the two labelled and eventually hidden
   embryos; if flow error exceeds the ~1-2 voxel scale needed for tube-based integration to help,
   TBD gains will be swamped by warp error before precision/recall analysis even matters.
4. **Division windows**: should TBD attempt to be division-aware from the start (branching DP/MHT),
   or is it strictly cleaner to route all division-adjacent ambiguity to the dedicated division
   posterior (§2.5, lane 04) and keep the TBD prototype single-trajectory only? The literature
   (mitosis-aware MHT) suggests a principled joint answer exists eventually, but building it first
   risks conflating two unproven mechanisms in one falsification test.
5. **Compute budget under the 12h Kaggle T4/T4x2, internet-off constraint**: KBMOD's GPU kernel and
   YOSO's continuous filter both assume dedicated compute research infrastructure; we need a concrete
   measurement of wall-clock cost for DP-TBD / shift-and-stack over the full hidden-embryo volume
   before deciding whether Main Bet's full lineage field (dense per-voxel integration) is even
   feasible in the submission time budget, or whether it must be restricted to a cascade of
   "uncertain regions only" as already specified in Layer-3 §3.3.
6. **Are there existing zebrafish/embryo-specific TBD or shift-and-stack precedents we haven't
   found?** This survey found strong precedent in radar/astronomy/generic video/generic cell
   tracking (Viterbi, MPM, ELEPHANT) but no paper doing TBD-style integration specifically for
   light-sheet/anisotropic embryo imaging with known tissue deformation — worth a follow-up
   targeted search (e.g. "light-sheet," "SPIM," "zebrafish," "sub-threshold," "temporal
   accumulation") before committing full engineering effort, in case a directly-transferable
   implementation already exists in the microscopy community.

---

## Sources consulted (full list)

- [Buzzi et al., "A Novel Dynamic Programming Algorithm for Track-Before-Detect in Radar Systems," IEEE TSP 2013](https://ieeexplore.ieee.org/document/6475194/)
- [Dynamic programming TBD via polynomial time-series prediction, IET RSN 2016](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-rsn.2015.0332)
- [DP-TBD for weak maneuvering targets, MDPI Remote Sensing 2024](https://www.mdpi.com/2072-4292/16/14/2639)
- [Cost-reference particle filter bank TBD, arXiv:2309.13922](https://arxiv.org/pdf/2309.13922)
- [Comparison of particle filters for recursive TBD](https://www.researchgate.net/publication/4221007_A_comparison_of_particle_filters_for_recursive_track-before-detect)
- [Bayesian TBD procedure for passive radars, EURASIP JASP 2013](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2013-45)
- [Digital tracking observations for faint asteroids, arXiv:1508.01599](https://arxiv.org/pdf/1508.01599)
- [Fast algorithms for slow moving asteroids (KBO shift-and-stack), arXiv:1901.02492](https://arxiv.org/pdf/1901.02492)
- [KBMOD GitHub repository, DiRAC Institute](https://github.com/dirac-institute/kbmod)
- [You Only Stack Once (YOSO), arXiv:2605.06913](https://arxiv.org/abs/2605.06913)
- [Sifting through the Static: moving object detection in difference images, AJ 2021](https://iopscience.iop.org/article/10.3847/1538-3881/ac22ff)
- [ELEPHANT: tracking cell lineages in 3D by incremental deep learning, eLife 2022](https://elifesciences.org/articles/69380)
- [ByteTrack: multi-object tracking by associating every detection box, arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- [Adaptive confidence threshold for ByteTrack, arXiv:2312.01650](https://arxiv.org/html/2312.01650v2)
- [Nuclei detection for 3D microscopy with a fully convolutional regression network, IEEE 2021](https://ieeexplore.ieee.org/document/9406585/)
- [T-CNN: tubelets with CNNs for object detection from videos, arXiv:1604.02532](https://arxiv.org/abs/1604.02532)
- [Object detection in videos with tubelet proposal networks](https://www.researchgate.net/publication/320971846_Object_Detection_in_Videos_with_Tubelet_Proposal_Networks)
- [Test-time augmentation for deep learning-based cell segmentation, Sci Reports 2020](https://www.nature.com/articles/s41598-020-61808-3)
- [Fast and robust optical flow for time-lapse microscopy using super-voxels, Bioinformatics 2013](https://academic.oup.com/bioinformatics/article/29/3/373/257856)
- [3D flow field estimation and assessment for live-cell fluorescence microscopy, Bioinformatics 2019](https://doi.org/10.1093/bioinformatics/btz780)
- [Myronenko & Song, Coherent Point Drift, NeurIPS 2006](https://papers.nips.cc/paper/2962-non-rigid-point-set-registration-coherent-point-drift)
- [Structured Analytic Coherent Point Drift, arXiv:2605.00934](https://arxiv.org/pdf/2605.00934)
- [MPM: joint representation of motion and position map for cell tracking, CVPR 2020, arXiv:2002.10749](https://arxiv.org/pdf/2002.10749)
- [Global linking of cell tracks using the Viterbi algorithm, Magnusson & Jaldén](https://pubmed.ncbi.nlm.nih.gov/25415983/)
- [Batch iterative Viterbi algorithm for cell tracking and lineage construction](https://www.researchgate.net/publication/261206085_A_batch_algorithm_using_iterative_application_of_the_Viterbi_algorithm_to_track_cells_and_construct_cell_lineages)
- [Cell Tracking according to Biological Needs: mitosis-aware multi-hypothesis tracker, arXiv:2403.15011](https://arxiv.org/pdf/2403.15011)
- [Constant false alarm rate, Wikipedia](https://en.wikipedia.org/wiki/Constant_false_alarm_rate)
- [Improved CFAR detector with multi-frame integration under heavy-tailed clutter, IET Signal Processing 2023](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sil2.12145)
- [The correct estimate of the probability of false detection of the matched filter, A&A 2016](https://www.aanda.org/articles/aa/full_html/2016/05/aa27463-15/aa27463-15.html)
- [Further results on matched-filter false-detection probability, A&A 2017](https://www.aanda.org/articles/aa/full_html/2017/08/aa29330-16/aa29330-16.html)
- [Trackastra: transformer-based cell tracking for live-cell microscopy, ECCV 2024, arXiv:2405.15700](https://arxiv.org/abs/2405.15700)
