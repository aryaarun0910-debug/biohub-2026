# Codex PLAN REVIEW — second set of eyes on the post-swarm strategic pivot (2026-07-03, evening)

Mission: **red-team and improve the plan below.** A 5-agent research swarm + a repo red-team just
forced a strategic reframe. I want you to (a) challenge whether the reframe/plan is right, (b) find
what it's missing or gets wrong, (c) add or change concrete steps, and (d) flag any EV-negative or
DQ-risky move. Be adversarial. Assume I am biased toward my own conclusion. Verify against the repo
and the cited reports rather than trusting my summary.

## What changed today (the reframe)
Prior belief (now falsified): "classical ceiling ~0.854; our DoG anchor 0.807 is competitive; break
past 0.854 only with a from-scratch learned detector." **All three premises are wrong.**

Verified 2026-07-03 (kaggle CLI + repo inspection; see reports/research/gap_competition_intel_2026-07-03.md):
- **Live public LB top = 0.896**; dense cluster 0.86–0.896; ~62 teams above 0.854. Our 0.807 ≈ median (~p52).
- The upper LB runs the **organizer's own learned baseline** = the `tracking_cellmot` package, ALREADY
  vendored here at `vendor/kaggle-cell-tracking/src/tracking_cellmot/` (`models/temporal_unet.py`
  TemporalUNet3D detector + `models/simple_node_transformer.py` SimpleNodeTransformer edge model;
  real metric in `vendor/kaggle-cell-tracking/metrics.md`, BSD-3).
- Shipped weights are ~3-epoch (~LB 0.81). **Public community 50-epoch weights are attachable OFFLINE**
  (e.g. `pilkwang/biohub-tracking-support-pack-50ep-v1`, ~349MB) → reported ~0.85–0.856. 0.89+ adds
  ILP linking + gap recovery + safe division on top.

## Today's other verified facts (use these)
- **Kaggle op_bright coord bug FIXED + committed (2fa4e64):** the pending submission that returned 0.727
  was a broken artifact — candidates were clustered on the ISOTROPIC grid but scaled by the anisotropic
  SCALE (XY compressed 4x, ~22% of nuclei merged). Fix = scale by ISO. Notebook now faithfully deploys
  op_bright_smooth (4 diagnostic crops match local on node count AND adjJ). COM-refine was removed to get
  exact match (see red-team note #6 / metric quirk below — it may actually help, unvalidated).
- **op_bright_smooth full-199:** min-fold adjJ 0.6996 (44b6 0.700, 6bba 0.762, weighted 0.752). Real
  both-fold gain over op_bright (weighted 0.741). Still median-tier vs the learned baseline.
- **est_n IS hidden at inference** (test ships only .zarr, no .geff, no estimated_number_of_nodes) —
  count calibration must self-estimate the count, cannot use the oracle value.
- **Metric quirks decoded from metrics.md** (verify these yourself): over-prediction taxed
  ×(1−0.1·(T_pred−T_true)/T_true) → count-calibration is the lever, NOT "continuity"; edges between two
  UNMATCHED nodes are FREE (not FP); sub-voxel centroid refine is free recall at the 7µm gate.

## The proposed plan (RANKED) — tear this apart
1. **Deploy the organizer's learned baseline + public 50ep weights as an offline notebook** (~0.85,
   ~p52→~p88). No training. Highest EV. First: verify weights load + run <12h on Kaggle T4x2.
2. **Stand up the REAL metric locally** (metrics.md + tracksdata, INCLUDING the division term); retire
   the edge-only numpy proxy + the 0.854 "ceiling" as the selection gate.
3. **Push to 0.89:** replace greedy linker with **motile** (ILP, division-native, MIT, SCIP fallback —
   no Gurobi) + 1-frame gap recovery + safe capped division. (Cheapest division win first: post-hoc 1→2
   split test — we currently capture 0 of the 0.1 division bucket.)
4. **Only if beating public weights:** retrain the detector (STAR-3D anisotropy-aware, or longer training)
   pretrained on **NIS3D** (dense, CC-BY, includes zebrafish). PAC-MAP is NonCommercial (license gate);
   Zebrahub is exact-domain but same-lab → train/test LEAKAGE risk.
- **Hard gate before any of this:** confirm the JS-gated competition `/rules` permit external community
  weights/backbones (admins publish them → de-facto yes, but must be read verbatim).

## Red-team findings on our OWN evidence (returned in-session, not yet filed — validate & prioritize)
1. HIGH: `phase1_v3.csv` at HEAD is only 20 crops — the 199-crop V3 baseline of record was clobbered by a
   20-crop run (`9e7a552`). 0.632 survives in `reports/phase1_v3_run.log` + git. (Being regenerated now.)
2. HIGH: HANDOFF scores table stale/optimistic (lists op_bright_smooth 0.734/0.791 from 20 crops; real 199
   = 0.700/0.762; lists op_bright LB "~0.812" for what returned 0.727).
3. HIGH: the 20-crop screening subset is systematically EASIER (44b6 V3 0.677@20 vs 0.632@71) and is the
   first-10-alphabetical, not stratified → every "first look" go-decision biased up.
4. HIGH: divisions = 10% of metric, 0% captured, scheduled LAST; even div_J=0.3 → +0.030 > entire
   op_bright+smoothing edge gain. Internal plan valuations contradict each other.
5. HIGH: op_bright's marginal value over FREE v3_smooth is never measured at 199 (v3_smooth only 20 crops;
   regenerating now).
6. MED: adjusted-J count term is ONE-SIDED (no upper clip; N_pred<N_est gives a >1 bonus) → optimal
   N_pred may be BELOW est, so the "0.95–1.05" count-cal target is likely mis-specified.
7. MED: 0.854 "ceiling" and the "smooth_sample w=0.7 == public V11" identity were both unreproduced
   assumptions (the LB reframe now shows 0.854 was never a ceiling at all).
8. MED: local→LB +0.07 offset is n=1 (the only 2nd point, op_bright, was the corrupted 0.727).
9. MED: hyperparams (nms=1.0, r_same=3.0, w=0.7) tuned on the same 2 embryos they're scored on; the
   "nested inner split" the methodology claims does not exist in the harness.
10. LOW: dense `linear_sum_assignment` in metric_numpy vs host's sparse matcher could disagree on crowded
    6bba frames under ties — validated only on 7 hand-built cases, not a real dense crop.

## What I want from you (Codex)
- Is the pivot correct, or am I over-rotating? Is step 1 (ship the public-weights baseline) actually the
  highest-EV first move, or is there a higher one I'm missing? Argue the strongest case AGAINST it.
- What's the fastest CORRECT route to standing up the real metric locally (step 2), and is there a trap in
  reusing `tracking_cellmot` code for both training and scoring (leakage / config drift)?
- Divisions: cheapest reliable way to bank part of the 0.1 term — post-hoc heuristic vs motile split-cost?
  Quantify expected div_J and the FP risk under the metric.
- The 0.89→0.896 gap: what do the top teams almost certainly do that the swarm did NOT surface? (test-time
  aug, ensembling, det-threshold tuning, gap/division post-processing, better linking window, etc.)
- Rank the red-team findings by whether they should INTERRUPT the pivot vs. be handled inline vs. dropped.
- Any DQ / rules landmine in building on community weights or same-lab (Zebrahub) external data. Quote the
  rules if you can retrieve them.
- Add anything we're structurally blind to. End with: the ONE change to this plan you'd fight for.

## Where to look (repo, self-contained)
reports/research/gap_competition_intel_2026-07-03.md (leaderboard, weights, metric quirks),
gap_learned_detector_2026-07-03.md, gap_tracking_division_2026-07-03.md, gap_external_datasets_2026-07-03.md;
vendor/kaggle-cell-tracking/ (tracking_cellmot + metrics.md); notebooks/kaggle_op_bright/kaggle_op_bright_infer.py
(the fixed notebook); reports/inventory/*.csv; reports/journal/JOURNAL.md; HANDOFF.md.
