# Experiment plan (canonical)

## CRYSTALLIZED DIRECTION (measured data + Codex edge-audit CONVERGE, 2026-07-02)
Two independent analyses point to the SAME move:
- **My V3 taxonomy (measured on real crops):** 73-93% of missed edges are DETECTION-miss
  (GT node has no pred within 7um); node recall only 0.79-0.87; count ratio 0.74-0.96 (we UNDER-detect).
- **Codex edge-audit (metric reverse-engineered):** the evaluator does one-to-one bipartite
  assignment FIRST, so a spatially-closer duplicate can STEAL a GT match from the proposal carrying
  the correct trajectory. Everyone optimizes detection & linking separately = shared blind spot.

=> **THE bet (Codex #1, confirmed by our data): matching-aware, track-conditioned proposal
arbitration + gap redetection + local tissue-flow linking.**
Concretely: over-propose INTERNALLY (fixes our recall/det-miss deficit + pushes count ratio->1.0,
which also helps the count multiplier), then pick ONE representative per cell by joint image +
temporal-continuity + one-to-one-matching-stability scoring (avoids assignment stealing), then
recover missing nodes along strong tracks (track-conditioned redetection), emit only deduplicated
track-supported nodes. This is metric-native, embryo-general, and attacks a correlated field weakness.

### Codex verified metric facts to exploit (permitted)
- **Count band 0.95-1.05** (blind undercount is catastrophic: raw-J may fall <=0.5% at r=.95 to break
  even). We're at r<1 -> detect MORE, win on both recall and multiplier.
- **FP-free zone is NARROWER than we thought**: an edge is FP if EITHER endpoint matches an annotated
  node w/ the relevant degree (not both). So don't flood submitted space; over-propose internally,
  emit deduplicated. (Corrects earlier assumption.)
- **Assignment stealing**: sub-voxel precision matters most in crowded regions / near 7um gate; pick
  proposals by the EDGES they carry, not peak intensity. (Our XY/4 localization discards this info.)
- **Division arbitrage**: at J=0.82 edge-only fork break-even = 45% precision; with the 0.1 term and
  small G, first true division is very valuable (break-even ~0.3-3% for small G). Sparse VERIFIED
  forks = MED-HIGH EV; blanket heuristics = damaging.

### Public-data / provenance (Codex rules audit)
- Public external data + pretrained models = **PERMITTED** (rules verified logged-in).
- Exact public-embryo dense-LABEL transfer after fingerprinting a test crop = **NEEDS-RULES-CHECK**
  (post one narrow public rules question; do NOT disclose a suspected match; preserve written answer).
- Submission/score probing to reconstruct private labels = **FORBIDDEN-DQ**. Zero cycles.
- Strongest provenance lead = the 522-frame EXACT-SCALE Ultrack embryo (not ZSNS001-005, whose voxel
  scale does NOT match). Identity unproven; no public dense CSV for it yet.
- ZSNS003 priors (4.06M detections, 15.9k divisions): step displacement median 1.3 um/frame (p99 6.4);
  daughter separation median 5.85 um (~ the 7um gate!); parent-daughter median 3.07 um; near-symmetric
  splits. Use as SOFT feature priors (not hard thresholds; only 2-embryo CV).

### Reconciliation note
"Detection-miss dominant" (ours) does NOT mean "just scale the detector" (Codex warns a bigger
detector can LOWER score via assignment stealing). It means: raise RECALL via over-proposal +
track-conditioned redetection, but SELECT the emitted node by matching-aware arbitration. Same move.

Codex reports: maximal audit + asymmetric-edge addendum (in Codex/.../ outputs).
Ranked asymmetric edges (Codex): 1 matching-aware arbitration+redetection (HIGH), 2 tissue-flow
kNN linking (HIGH), 3 sparse verified division head (MED-HIGH), 4 density/depth/time-adaptive
detection + full-res centroid refine (MED-HIGH), 5 count calibration 0.95-1.05 (MED), 6 public
lineages for edge/division scorer pretraining (MED-HIGH), 7 Ultrack registration/normalization (MED),
8 exact provenance transfer (VERY HIGH if permitted; NEEDS-RULES-CHECK).

---

# (prior) Experiment plan — from Codex maximal audit 2026-07-02

Full audit: `Codex/.../biohub_maximal_competitive_intelligence_2026-07-02.md`.
Selection policy: rank by **min(foldA,foldB)** first, then mean fold, then worst
density/intensity slice, then runtime. Public LB is a sanity check, NOT the optimizer.

## Strategic corrections (act on these)
1. **0.842 edge over 0.826 = temporal logic, not DoG params.** V3 diff (verified from source):
   NMS 3.2->4.0 um; XY +1.5 voxel offset before refine; refine window (3,9,9); TWO-PASS
   velocity-aware Hungarian (pass1 6um gate w/ 0.5x inherited-velocity cost, pass2 8um gate);
   gap-close 1 frame @6um; remove connected components < 4 nodes; divisions OFF.
2. **The 0.839 "UNet+ILP" notebook runs NEITHER UNet nor ILP** — titles/upvotes are not evidence.
3. **No public evidence on 0.86-0.875 methods** — all claims are speculation.
4. **Highest-EV = track-conditioned error correction, NOT a bigger detector.** Instrument edge
   errors first; fund learned detection only if endpoint-miss FNs dominate after V3.
5. **Spotiflow/CPV = residual proposal generator, gated on evidence** — not the primary bet.
6. **Divisions post-hoc, precision-gated** — never let the linker fork freely.
7. **"Soft negative is still negative"**: 0.01/1e-6 weight over millions of unlabeled voxels can
   dominate; compare zero-weight/ignore vs trusted-negatives vs teacher-pseudo-positives.

## Ranked queue (Expected delta = prior, not promise; gate on exact metric + BOTH embryo folds)
| # | Experiment | Exp Δ | Gate | Our status |
|--:|---|--:|---|---|
| 1 | Port V3 exactly; factorial ablate NMS=4, XY+1.5, (3,9,9), 2-pass velocity linker, min-len=4 | recover +0.016 | keep only changes +ve on BOTH folds | notebook upgrade IN PROGRESS |
| 2 | Track-conditioned 1-frame redetection at predicted midpoint (stable track ends/starts) | +0.005-0.015 | min-fold +0.003; added-node edge precision >70% | TODO (highest EV) |
| 3 | Metadata-aware threshold/count calibration -> 0.9-1.1x Nest, temporal smoothing | +0.003-0.010 | min-fold +0.002; no count-decile loses >0.01 | TODO |
| 4 | Multi-hypothesis detector union (DoG param bank + dup-merge + node-quality rank) | +0.004-0.012 | min-fold +0.003; <=2x infer time | TODO |
| 5 | Registration/background ablation: phase-corr; q .005/.99999; sigma=20 white-tophat | +0.002-0.012 | min-fold +0.002; reject embryo-specific | TODO |
| 6 | Learned pairwise edge score on fixed DoG nodes (displacement/velocity/intensity/density) | +0.005-0.020 | min-fold +0.005 over V3 linker | TODO |
| 7 | DAXI U-Net foreground/contour as proposal/refinement signal (not replacement) | +0.003-0.015 | zero-shot must improve recall/edge on untouched embryo first | notebook exists (kaggle_daxi_infer.py) |
| 8 | Compact ITEC-style split/merge correction (size/intensity continuity + stable neighbors) | +0.003-0.012 | min-fold +0.003; correction precision >75% | TODO |
| 9 | Spotiflow-3D residual detector (sparse mask + CPV) union w/ DoG | +0.005-0.020 (hi var) | min-fold +0.005; DoG recall must not fall; <8h | TODO (gated on #1-2 taxonomy) |
| 10 | Compare Hungarian vs learned min-cost-flow vs motile/organizer ILP, tuned to exact metric | +0.002-0.012 | min-fold +0.004; solver p95 <8h | TODO |
| 11 | Precision-first division classifier AFTER final tracking | 0-0.030 (hi var) | div precision LCI >0.5; overall min-fold +0.003 else abstain | TODO (last) |
| 12 | Test-time ensemble of 2-3 graphs (consensus edges + union nodes) | +0.002-0.010 | min-fold +0.003; <10h | TODO |
| 13 | Cellpose-SAM/uSAM/BiomedParse quick crop benchmark | unknown | no full run unless zero-shot beats DoG at <=3x cost | TODO (low) |

## Immediate next builds (local)
- [x] **Numpy metric harness** (src/biotrack/metric_numpy.py) — edge term reimplemented from host algo,
  VALIDATED 20/20 exact match vs tracksdata harness (identity/drop-edges/jitter/spurious/drop-nodes).
  Runs WITHOUT tracksdata -> Kaggle OOF scoring + fast local sweeps. Division term still TODO.
- [x] **V3 DoG notebook** (kaggle_dog_infer.py) upgraded to exact V3 config, machine-validated.
- [ ] Run V3 on all 129 crops (data unzipping) + score both embryo-held-out folds via numpy metric.
- [ ] **Edge-error taxonomy** on V3: FN = {missing endpoint | endpoint outside 7um | wrong association};
  FP; count ratio; per-crop + per-embryo. Decides whether to fund learned detection (#9). HIGHEST-VALUE.
- [ ] Add division term to numpy metric (per documented TP rule) once edge work is stable.

## 72-hour focus (Codex)
V3 exact reproduction + error taxonomy -> track-conditioned 1-frame redetection -> count
calibration -> association upgrade -> DAXI zero-shot audit -> 3 controlled submissions
(robust V3, best edge-only CV, one orthogonal). Single highest-value action:
**edge-error taxonomy on V3 + track-conditioned redetection.**
