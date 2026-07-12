# MASTER ROADMAP — the complete map to winning

**Date:** 2026-07-12. Consolidates `SYNTHESIS.md`, `WIN_BET.md`,
`METRIC_SEMANTICS_VERIFIED.md`, the six lanes, and measured results into one battle
plan. This is the operating map; the others are the evidence beneath it.

---

## 0. North star + score targets

Win the **private** board = degrade least on the hidden disjoint embryo. Public score
is deployment telemetry. Targets in the currencies we can actually measure:

| Horizon | Target | Basis |
|---|---|---|
| Floor (never breach) | public ≥ 0.889; OOF ≥ current | do not regress the wrapper |
| Base (this round) | min-fold OOF adj-J **0.559 → 0.60+**; public clear **0.90** | selective repair on ambiguous links |
| Target | min-fold OOF **→ 0.65**; public **0.905–0.915** | repair + division posterior |
| Domination | public **0.92+**; smallest public→private gap = **private #1** | breadth generalization + divisions + endpoint recovery |

Hard ceilings (do not plan past without new capability): selected-edge oracle caps
pure-linking OOF at **0.71 / 0.73**; endpoint (detection) oracle caps at **0.88 / 0.94**.
Beyond linking needs redetection (endpoint recovery) and the division term.

---

## 1. Architecture in one picture

```
raw learned detections + greedy OOF graph   (exists: oof_clean/pred_geffs_split_{0,1})
   → [E0b] full 0.889 wrapper post-processing  = the honest production baseline
       → [selective repair]  breadth association scorer overrides ONLY uncertain edges
           → [division posterior]  add-only forks, trained on external breadth
               → [target-time gating]  gradient-free f/b consistency on hidden movie
                   → [count/N_est operating point]  image-derived count target
                       → metric-aware final selection → submission
```

Every stage is **add-only / override-under-gate**: worst case falls back to the
wrapper, so the system can only climb. This is the core defensive primitive.

---

## 2. What is proven vs open (2026-07-12)

- Raw greedy OOF: 0.6562 / 0.5593. Public wrapper: 0.889.
- **External transfer scorer works:** on ambiguous links (where nearest≠truth) it hits
  **top-1 0.2935 / MRR 0.5807 vs NN 0.0 / 0.4297**; candidate recall@6 0.96. This is
  the whole thesis — it recovers ~29% of links pure-distance linking gets wrong.
  Pooled AUC is noise (easy-negative dominated); per-source top-1/MRR is the metric.
- **Blocker:** E0b — the full 0.889 wrapper reproduced on fold-specific OOF. Until it
  exists, no repair delta is honestly measurable.
- Verified metric facts drive everything: ignored edges free; count penalty is the only
  off-annotation restraint; division worth up to **+0.1**, FP bar soft; `N_est` hidden
  at inference.

---

## 3. Phased execution map

### Phase A — unblock + build in parallel (NOW, CPU-local, no GPU)

**A1 — E0b: reproduce the 0.889 wrapper (BLOCKER).**
The wrapper (`notebooks/kaggle_lb897_trackastra/biohub_lb897_trackastra.py`, 2274 lines)
is a parameterized graph post-processor. Extract its output-stage functions
(motion-relink, gap-close, division-geometry filter, single-parent repair,
isolated-node prune, short-track filter) into `src/biotrack/wrapper.py`. Run the
**pure LB897 config** (NOT the trackastra_direct_fusion_prune preset, which is the
0.865 reject) over `oof_clean/pred_geffs_split_{0,1}`. Export per crop: final GEFF,
candidate edges + `edge_prob`/`edge_dist`, node count, and regime metadata.
Gate: reproduced OOF matches the known wrapper behavior; this becomes THE baseline.

**A2 — Finalize + freeze the breadth association scorer.** Larger time windows across
all 4 Zebrahub embryos; swap logreg → gradient-boosted trees (add `lightgbm`/`sklearn`
to a side env, not the metric venv); reserve an `edge_prob` feature slot for A1's output.
Report per-source top-1/MRR, not pooled AUC.

**A3 — Division fork-posterior on breadth (NEW, high EV).** The 4 Zebrahub embryos hold
~90k division events. Train an add-only fork classifier on scale-free mother/daughter
features (displacement symmetry, mass ratio, flow divergence) across embryos; calibrate
per-embryo prior. Division term is worth **up to +0.1** and the pack suppresses it — this
is the biggest underpriced point source.

**A4 — Bracketed-miss oracle.** From GT + wrapper detections, measure what fraction of
missed endpoints are bracketed by an existing track. Sizes the endpoint-recovery prize
before any GPU spend (Phase D go/no-go).

**A5 — Acquire March-14 DAXI benchmark** (`tracks_benchmark/2024_03_14_daxi_tracks.zarr`)
— likely closer to the competition source domain than ZSNS → better transfer. Confirm
Zebrahub + DAXI license text for prize eligibility.

### Phase B — the decisive gate (E0b-gated)

**B1 — Competition-transfer gate.** Train breadth scorer on {4 Zebrahub + one competition
family}; evaluate per-source top-1/MRR on held-out family's candidate edges (from A1) vs
the wrapper's own `edge_prob`. **KILL if no lift on either fold.**

**B2 — Selective repair reverse-fold.** Override wrapper edges only where: baseline
confidence low AND breadth score high AND forward/backward consistent AND no lineage
constraint violated AND node-count neutral-or-better. Score full wrapper+repair OOF.
**Promote only if both folds ↑ and regime slices stable.** ≤2 confirmatory tests/round.

### Phase C — stack the safe add-ons (each independently gated)

- **C1** Integrate the A3 division posterior on the frozen repaired graph; gate div-J ≥ 0.25.
- **C2** Gradient-free target-time consistency gating on the hidden movie (legal self-supervision).
- **C3** Image-derived `N_est` estimator → set the count-penalty operating point.

### Phase D — endpoint recovery (only if A4 oracle says a prize exists)

Track-conditioned redetection at bracketed gaps: accept a low-threshold candidate only
where a track predicts it AND the raw image supports it (dual gate). Temporal-flip
detection TTA. Raises the 0.88/0.94 detection cap. GPU, gated by A4.

### Phase E — ensemble + submit

- **E1** Trackastra `ctc` (weights local) as an independent second linker feeding the
  same repair frame — uncorrelated errors vs the monoculture.
- **E2** K-best / multi-hypothesis solver on ambiguous crops only.
- **E3** Submission gates: exact both-fold OOF, runtime <9h, internet-off dependency
  smoke, schema/graph integrity, immutable hashes, provenance manifest, and a stated
  reason the submission differs from the anchor. Submit only after a positive
  wrapper-relative OOF.

---

## 4. Additional levers (new ideas beyond the plan)

1. **Consistency-distilled per-embryo threshold.** On the hidden movie, treat
   high-forward/backward-consistency links as pseudo-GT to calibrate the repair
   acceptance threshold per embryo — legal self-supervision that adapts to the target.
2. **Hard-negative mining from the wrapper's own errors.** Train the repair scorer
   specifically on edges where the breadth model and the wrapper disagree — spend all
   capacity on the decision boundary, not the easy 95%.
3. **Multi-linker cost blending.** Feed wrapper `edge_prob` + breadth score + Trackastra
   score + OT/Sinkhorn cost into one calibrated blend (GNN or ILP over the costs) —
   ensemble at the edge level, promoted only if complementary error is measured.
4. **Lineage-length / division-interval priors from breadth.** Dense external data gives
   realistic per-developmental-time division intervals → a soft prior that improves both
   division recall and false-fork suppression without embryo-specific tuning.
5. **Regime router / OOD gate.** Classify each crop's acquisition regime; apply repair
   and adaptation only where the model is confident/in-distribution, skip easy crops —
   protects against regression (the false-positive-gate lesson made structural).
6. **Subvoxel centroid refinement** on kept nodes — cheap precision on the 7µm matching.
7. **Temporal detection TTA** (time-flip + multi-window) for the endpoint cap.

---

## 5. Compute & sequencing

| Resource | Work |
|---|---|
| local CPU | E0b, all repair/division/oracle/calibration, breadth scorer — the whole gate stack |
| Kaggle T4 | endpoint redetection (Phase D), Trackastra inference (E1) — only after CPU gates pass |
| Kaggle T4×2 / cloud | gated training only (learned feature branch, if GBDT saturates) |
| storage datasets | immutable weights, wheels, cached candidates, Zebrahub/DAXI shards + provenance |

Phases A–C are 100% CPU-local. Do not spend a GPU-hour until A4 sizes the endpoint prize
or B/C pass. Cache expensive boundaries (detections, candidate edges, features) so most
experiments become CPU reselection.

---

## 6. Gate doctrine & risk register

- **Right baseline:** every delta measured vs the E0b wrapper, never the greedy.
- **Both-fold + regime-slice:** min-fold ≥ +0.005, no slice regresses beyond ε.
- **≤2 confirmatory tests/round** (≈2 effective samples; more hypotheses = luck passes).
- **Augment-never-replace:** worst case = wrapper fallback.
- **Eligibility:** Zebrahub/DAXI license confirmed before a submission model uses them;
  gradient-free target adaptation is safe, weight-update TTA is BLOCKED pending host
  clearance; the unmatched-fork evaluator defect stays quarantined.
- **Known failure mode:** OOF gains that don't transfer (fusion +0.035 OOF → −0.024
  hidden). Mitigated by breadth-in-training + reverse-fold + regime slices.

---

## 7. Primitives of domination

1. **Data moat** — nobody public trains on external dense lineages; more + closer-domain
   breadth than anyone is the structural moat.
2. **Repair, never replace** — you can only climb.
3. **Own the hard tail** — 29% vs 0% on ambiguous links; spend capacity only where the
   baseline is uncertain.
4. **Take the metric's free gifts legally** — off-annotation edges free (recall-first
   where safe), divisions 10× underpriced by the pack, count→N_est a free multiplier.
5. **Adapt to the hidden embryo** — gradient-free self-supervision the static forks can't do.

---

## 8. Immediate queue (do in this order)

1. **A1 / E0b** — extract wrapper → `src/biotrack/wrapper.py`, replay 199 OOF, export
   candidates + confidence + regime. (BLOCKER)
2. In parallel: **A2** finalize breadth scorer, **A3** division posterior on breadth,
   **A4** bracketed-miss oracle, **A5** acquire March DAXI + license check.
3. **B1** competition-transfer gate → **B2** selective-repair reverse-fold.
4. **C1–C3** division integration, target-time gating, count estimator.
5. Gate → first wrapper-relative submission.
