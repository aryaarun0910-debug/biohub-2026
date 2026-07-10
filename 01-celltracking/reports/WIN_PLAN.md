# WIN_PLAN — win the private shuffle

**Commander reset:** 2026-07-10
**Deadline:** 2026-09-29 23:59 UTC
**Objective:** finish #1 on the private leaderboard. Public rank is a serve check, not the target.

## 1. Live field truth

Authenticated Kaggle snapshot on 2026-07-10:

- 1,014 ranked teams in the downloaded public leaderboard snapshot.
- Leader: **0.910**; #10: **0.901**; #20: **0.898**.
- Top-20 spread: **0.012**.
- **385 teams ≥0.880**; public median **0.839**.
- Our completed submissions: **0.807** valid V3 anchor and **0.727** broken-coordinate artifact.

The public board is saturated and unusually tied. It contains four visible movies whose IDs also occur with labels in train, so it is vulnerable to seen-movie tuning. The 71% private board is a disjoint hidden embryo. Our competitive opening is not another public-only tweak; it is reducing degradation on the unseen embryo.

## 2. North-star doctrine

1. **Private-shuffle generalization is the product.** Every model decision is gated on embryo-held-out OOF in both directions.
2. **Public LB is deployment telemetry.** Use it to catch coordinate/schema/runtime failures and to measure broad calibration—not to select hyperparameters.
3. **Attack both edge ceilings.** Missing endpoints and wrong associations are coupled. Measure their oracles separately, then spend compute where the oracle says headroom exists.
4. **Use the unlabeled target movie legally.** Test-time association adaptation is the main asymmetric bet. The current rules permit Competition Data use for any purpose and prohibit hand labeling/human prediction—not automated self-supervision. Preserve that distinction and document it.
5. **Optimize the real metric.** Candidate probabilities feed a metric-aligned lineage solver; generic tracking likelihood is not the final objective.
6. **Preserve prize eligibility.** No private-label reconstruction, public-source label transfer into an identified test crop, submission probing, or quarantined division-evaluator exploit.

## 3. Current evidence—provisional until artifacts are re-scored

- Classical fallback: V3 `0.632/0.756` adjusted edge-J by embryo; op-bright-smooth `0.700/0.762`.
- Learned detector is reported at roughly 90–95% held-out node recall, with the weaker learned fold around 0.885 in prior notes. This is **not solved**: track-conditioned recovery may lift the edge oracle materially.
- Learned greedy OOF was reported around `0.656` on held-out 44b6 and `0.559` on held-out 6bba.
- ILP produced large local gains and suppressed division-FP catastrophes, but a complete full-fold, both-direction, exact artifact score is not yet banked locally.
- More training appeared to overfit, but the negative 45-epoch read was only three held-out crops.
- Trackastra, endpoint/candidate oracles, metric-aligned fractional ILP, and target-time adaptation have not yet been tested.

The first commander action is therefore artifact recovery and exact rescoring. No inherited `~0.70` number is treated as truth without its GEFFs and TP/FP/FN decomposition.

## 4. Parallel attack fronts

### Front A — regain experimental control

- Download completed fold weights and prediction GEFFs from Kaggle.
- Score every available crop locally with `biotrack.metric`.
- Emit per-crop and aggregate edge TP/FP/FN, adjusted-J, node recall/count ratio, division TP/FP/FN/J, runtime, commit, and artifact hashes.
- Repair the fold-1 ablation mount bug; never launch a GPU job whose weights cannot be located in preflight.

**Gate:** no new training or leaderboard submission without a reproducible baseline bundle.

### Front B — Trackastra zero-shot, then ensemble

Use the official Trackastra `ctc` checkpoint—the 2D/3D successor to the ISBI generalizable-linking winner—on frozen detections.

1. Convert detections into small anisotropic ellipsoid instance masks or direct Trackastra features.
2. Produce association probabilities without tuning on the evaluated embryo.
3. Run Trackastra linking alone and Trackastra scores through the existing SCIP lineage solver.
4. Fuse organizer + Trackastra edge logits only after each is independently scored.

**Go gate:** `≥+0.010` edge-J on both embryo directions, or `≥+0.015` min-fold with no fold worse than `-0.003`.

### Front C — break the endpoint ceiling

Measure before claiming 99%:

1. Endpoint oracle: GT edges whose two endpoints have a detection within 7 µm.
2. Candidate-edge oracle: recoverable GT edges already present in the graph.
3. Lineage-constrained oracle: best valid edge-J from those candidates.
4. Miss taxonomy by intensity, density, depth, time, track context, and division proximity.

Then attack only genuine missing endpoints:

- track-conditioned redetection at predicted gaps/births/deaths;
- raw-logit search around motion-extrapolated coordinates;
- multiscale detector union with track-aware arbitration;
- two-pass graph repair with a hard node-count budget;
- detector ensemble only where complementary recall is measured.

**Go gate:** endpoint recall `≥+0.020` on both folds and adjusted edge-J non-negative before association retuning. Stretch target: 97–99% node recall, never assumed in advance.

### Front D — capture association headroom

1. Calibrate organizer and Trackastra candidate probabilities out of fold.
2. Add physical-coordinate, density-invariant and three-frame motion features.
3. Add a local parent mask and explicit birth/no-parent dustbin.
4. Replace generic MAP edge selection with a Dinkelbach-style expected-J objective:

```text
maximize Σ [pTP(e) - λ·pFP(e)] x_e
```

under lineage constraints. Sparse annotations require separate TP, metric-counted-FP and ignored-edge calibration; `1-pTP` is not automatically `pFP`.

**Go gate:** `≥+0.005` exact combined OOF on both directions. If lineage oracle `<0.88`, stop calling 0.88 an ILP problem and return to candidates/features.

### Front E — private-shuffle main event: target-time adaptation

The authenticated rules text supports automated optimization on provided unlabeled test images: Competition Data may be used for any purpose, while hand labeling/human prediction of test records is prohibited. Proceed with automated self-supervision; retain the exact rules snapshot and optionally seek organizer confirmation before the final submission.

- build pseudo-positive edges only where organizer, Trackastra, physical matching and forward/backward consistency agree;
- enforce direct `t→t+2` association = composed `t→t+1→t+2` association;
- adapt only a small motion residual, calibration/normalization parameters, or final edge head;
- reset per embryo, cap steps, preserve the unadapted model, and ensemble adapted/unadapted probabilities;
- optionally fit a per-embryo continuous scene-flow prior and use its residual as an ILP feature.

Simulate test time honestly: hide all labels for embryo A, adapt on A's movie, score A; reverse for B.

**Go gate:** `≥+0.010` edge-J in both directions with no count/division regression. A one-fold win is not banked.

### Front F — finishers

- explicit calibrated fork posterior with mother/daughter appearance, geometry and tissue-flow residuals;
- division probability inside the joint solver;
- isolated/edge-neutral node pruning and count-cost calibration;
- subvoxel centroid refinement;
- K-best solver/association ensemble on ambiguous crops only.

**First division gate:** division-J `≥0.25` with edge-J loss `≤0.005` on both folds. Stretch: `0.50–0.60`.

## 5. Score pathway—targets, not promises

The public 0.910 frontier does not define the private winning score. We optimize relative private robustness.

| Layer | Target evidence |
|---|---|
| Recovered learned baseline | complete both-fold exact artifact score |
| Endpoint attack | node recall 97–99% if oracle/raw signal permits |
| Association stack | edge-J 0.80–0.88, conditional on lineage oracle |
| Target-time adaptation | positive in both reverse-fold simulations |
| Divisions | J 0.25 first; 0.50+ stretch |
| Final | strongest both-fold OOF ensemble that fits offline ≤12h |

We do not cap ambition at 0.89, and we do not plan from 0.94 mythology. The oracle determines the reachable edge band. The mission is to maximize private rank, not defend a forecast.

## 6. Compute and sequencing

- Local CPU: exact scorer, oracles, redetection diagnostics, calibration, small ILPs, artifact registry.
- Kaggle T4: learned inference and Trackastra screens.
- Kaggle T4×2 / cloud: only gated training or adaptation jobs.
- Current Kaggle quota snapshot: 16.85 GPU-hours remaining before the 2026-07-11 00:00 UTC reset; no Biohub kernel running.
- Submission notebook: internet off, ≤12h, all wheels/weights shipped as versioned datasets.

Run concurrently:

1. artifact recovery + exact rescoring;
2. Trackastra adapter and zero-shot kernel;
3. endpoint/candidate oracle and redetection diagnostic;
4. rules wording retrieval for target-time adaptation.

Do not block CPU diagnostics on GPU work.

## 7. Submission discipline

A candidate reaches Kaggle submission only after:

1. exact both-fold OOF report and min-fold gate;
2. runtime/memory projection under 9h, leaving safety margin;
3. internet-off dry run and schema/coordinate parity;
4. immutable commit, weight checksum, config, and provenance manifest;
5. clear description distinguishing model hypothesis from deployment test.

Public score may reject a broken deployment. It may not promote a model that fails OOF. No score probing.

## 8. Immediate 72-hour orders

1. Recover all learned OOF GEFFs/weights and publish the actual score decomposition.
2. Run endpoint/candidate/lineage oracles on recovered folds.
3. Land and unit-test the Trackastra point→mask→association adapter.
4. Package Trackastra `ctc` code/weights for an offline T4 screen.
5. Repair and preflight the frozen-detection ablation kernel.
6. Prototype track-conditioned redetection on the worst no-candidate crops.
7. Preserve the authenticated rules rationale for target-time optimization; optionally ask the organizer for confirmation before final submission.
8. Launch only the experiments whose inputs and output paths pass preflight.

## 9. Legal red lines

- No identification of hidden crops followed by public-source label/trajectory transfer.
- No reconstruction of private labels through submissions or scoring.
- No unmatched-fork division evaluator defect.
- No non-commercial weights/data in a prize solution without permission.
- Public external data/models are allowed, but every artifact needs URL, license, checksum and transformation record.

## 10. The decisive bet

The highest-upside private-shuffle stack is:

```text
high-recall track-conditioned detections
    → organizer + Trackastra association ensemble
    → path/cycle-consistent target-embryo adaptation
    → metric-aligned lineage solver
    → calibrated divisions/count polish
```

The next move is not choosing Trackastra *or* endpoint recovery. They run in parallel; the oracle tells us which becomes the dominant spend.
