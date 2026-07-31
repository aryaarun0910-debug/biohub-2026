# Next decision

**Updated:** 2026-07-31
**Status:** active
**Best public:** 0.914 (P0-B) · **Slot 3 HELD**
**Objective:** exact **pooled** composite (min-fold is a robustness constraint only)

Superseded plans are preserved below and in `reports/journal/JOURNAL.md`.

## 1. The one measurement that changes the picture

**Re-run LOEO fold-0, sharded.**

`aryaarun07/biohub-loeo-f0-strict` had a **correct** manifest — `fold 0`, `arm strict`,
`secondary_enabled: false`, `deepcenter_enabled: false`, pack `split_0` primary, 71 crops — so
there is no leakage path and the science is sound. It died after **16 of 71 crops**. Not a
timeout (~26 predict-minutes against a 9h session); the output tree holds a full zarr geff per
crop plus a copied `secondary_seed_weights` dir, so the `/kaggle/working` size limit is the
prime suspect.

**Fix:** shard the fold across kernels, or convert each geff to rows and delete it before the
next crop. Use `scripts/kaggle_factory.py` (spec-driven, trap-hardened, never submits).

**Why it matters more than anything else queued:** the H0c division cascade is worth
**+0.06012 pooled** on E0c's substrate, and the ceiling is substrate-dependent (E0c 20/26 + 93/125
vs v122 15/26 + 68/125). We have never measured P0-A/P0-B's substrate. If its reach matches
clean903's 20/26 with node recall holding, the 0.914 platform and the division track **multiply**
instead of competing. Early hint: on the only placeholder crop with divisions P0-A reaches 3/3,
and its own division layer converts them to TP 0 / FP 6 / FN 3.

## 2. Ready to run (CPU; exact commands in the journal)

| item | status |
|---|---|
| H2a hybrid full scored oracle (4 arms × 199 crops, ~45–90 min wall) | census done, scoring not run |
| `agent5_utility.py --max-admit 6000` (utility vs probability ordering) | ledger done, comparison not run |
| `h4_ssl_gate_replay.py` (exact replay of the ssl×geom gate) | built, smoked, not run |
| `h1n_exact_replay.py` (H1-M admissions, ~82 min) | built, smoked, not run |
| node budget on **arm D** (v122) | only arm A measured |

The last one matters: node budget gave **+0.00157** on E0c, which over-predicts nodes (+0.0832
ratio). v122 already sits at −0.1598 and may have **no headroom at all**. Until arm D is measured,
node budget is not known to be a deployable gain on our actual base.

## 3. Slot 3 policy

Priority: (1) a portable H1 composition with positive expected pooled utility; (2) repaired P0-C
(`biohub-p0cr-v122-revtime-volguard` v1, COMPLETE, audit PASS, output byte-identical to the
offline repair) **if** it answers a still-open causal question; (3) a genuinely diverse candidate.

**Do not spend it on another global reverse-time weight or a threshold sweep.**

Nothing currently qualifies: the only corpus-verified deployable mechanisms are +0.00157 and
+0.00141, against +0.006 needed just to reach 0.920.

## 4. Open research, ranked by (expected pooled Δ × transfer plausibility) ÷ cost

1. **Association recovery** — 43.3% of missed GT edges were *detected then discarded* by our own
   pipeline (bipartite competition, wrapper filters, linefit displacement). Zero GPU, no precision
   bar to clear, upstream of both the division track and the portfolio problem.
2. **A within-track frame-selection signal** — the only measured route to +0.005 on the division
   layer (temporal-snap oracle reaches +0.00599). 32% of H1-M's FPs are duplicate admissions of the
   same track within ±2 frames. Nothing tested can pick the frame: H1-I, reverse-time and geometry
   all discriminate *across* cells, not *within* a track.
   **The +0.00369 GT-free snap figure was computed on the in-family basis that inflated H1-M ~30×.
   Re-derive cross-family before quoting it.**
3. **A second independent veto** — geometry-as-broad-veto is the only fusion that survived
   cross-family transfer (~2× FP reduction) and it is spent. Forward-association support and
   daughter persistence are label-free and plausibly FP-disjoint from appearance. A second 0.44×
   veto is worth more than any further appearance modelling.
4. **Hybrid deployable version** — the oracle recovers 33 divisions on 6bba for ~50 aux nodes, but
   the FP cost of running the proposer over the *enlarged* surface is unmeasured and will decide it.
5. **E0c volume-guard decision** — thousands of out-of-volume coordinates sit in the published
   baseline. Enabling the guard shifts the baseline the promotion gate is defined against. Needs a
   human call, and the 7,349-vs-14,319 count dispute needs reconciling first.

## 5. Do not reopen without new evidence

Exploit structures (score-negative, −0.0027/−0.0007) · detector diversity (nested; 0 new nodes) ·
appearance × appearance stacking (lift 0.00, mechanism known) · Zebrahub for divisions (lineage
fragmentation, 5.7–11.7 terminations per division) · CTC (licence-blocked) · H1-T (structurally a
subset of geometry) · a global reverse-time blend weight (measured at one LSB).

---

# Superseded — preregistered plans of 2026-07-30 (kept for audit)

# Next decision — attack the family boundary, not the same recipe

**Preregistered:** 2026-07-30
**Status:** superseded 2026-07-30 by the division track (D0' GREEN). Branch A CLOSED;
branch B demoted behind divisions.
**Compute state:** idle

## Primary track — Jaccard-optimal joint fork selection (D0' GREEN)

D0' measured the composed operation Oracle C never tested: remove existing false forks, then
reconstruct reachable true ones. Five parity-controlled arms, 199 crops, exact patched scorer,
edges-only edits on a FIXED node set (see 2026-07-31 correction: node invariance was assumed by construction, and the live pipeline DOES delete division-exempt components under suppression).

| arm | 44b6 | Δ | 6bba | Δ |
|---|---:|---:|---:|---:|
| baseline (parity OK) | 0.7595 | +0.0000 | 0.6490 | −0.0000 |
| suppress_all (GT-informed child retention) | 0.7630 | +0.0035 | 0.6517 | +0.0027 |
| suppress_all (GT-FREE control) | 0.7595 | -0.00004 | 0.6470 | **-0.00204** |
| suppress_all_then_add_replace | 0.8413 | +0.0818 | 0.7273 | +0.0783 |
| selective_suppress_then_add_replace | 0.8413 | +0.0818 | 0.7273 | +0.0783 |
| add_replace_then_selective_suppress | 0.8413 | +0.0818 | 0.7273 | +0.0783 |

GT-free child-retention control (`--fallback-only`): **+0.0783 / +0.0737**. Order does not
matter and the operations do not interfere. Division goes `TP0/FP93/FN26 → TP20/FP0/FN6` and
`TP4/FP582/FN121 → TP93/FP0/FN32`.

CORRECTION 2026-07-31: the decomposition below used the GT-INFORMED suppress_all row.
The GT-FREE control is -0.00004 / -0.00204, so suppression alone contributes NO free gain.

GREEN on both conditions: composed min-fold `+0.0737` (bar `+0.03`); suppression contributes
`+0.0601 / +0.0599` over Oracle-C-alone (bar `+0.01` bilateral). Strongly super-additive —
`0.0035 + 0.0182 = 0.0217` apart versus `0.0783` composed.

**Everything above is a GT-informed oracle ceiling and is not submittable.** Fork selection is
oracle in every arm.

### D0P — GT-free proposer audit: RED on frozen surfaces, but the cap is the limiter

Generation never consults GT; the oracle below is "proposable-only" (`suppress_all` then
add-replace restricted to forks the proposer generated).

| surface | 44b6 Δ | reachable covered | 6bba Δ | reachable covered | candidates |
|---|---:|---|---:|---|---|
| geometric_core 10.5/8.5 µm | +0.0195 | 5/20 | +0.0142 | 20/93 | 4.20M / 3.21M |
| native (candidate-edge cache) | +0.0116 | 3/20 | +0.0036 | 7/93 | 0.30M / 0.27M |
| outer_diag 15/15 µm (fixed) | **+0.0783** | **20/20** | **+0.0646** | 82/93 | 64.7M / 44.1M |

**RED by the preregistered rule** — 6bba `geometric_core` is `+0.0142`, below the `+0.015`
floor. No frozen deployable surface passes.

**But the diagnostic did its job.** `outer_diag` recovers 20/20 reachable divisions on 44b6 and
reproduces the GT-free D0′ ceiling exactly (`+0.0783`). The prize *is* present in a GT-free
surface; the `10.5 µm` cap discards it. That cap is E0c's motion-relink gate — calibrated on
**migration**. At division the daughters separate, so parent→daughter displacement is
systematically larger than ordinary motion, which makes a migration-calibrated cap the wrong
prior for a division proposer. Branch A corroborates: ordinary recoverable continuations sit at
median 8.29 / 7.62 µm, just under the cap.

Effective cost is far below the raw count: division FP accrues only at annotated mothers, so
`outer_diag`'s metric-visible denominator is 405,212 / 965,478 reliable negatives, not 64.7M /
44.1M. With `J = k/(26+m)`, holding `m ≤ ~50` at `k = 20` keeps `J ≈ 0.29`.

Caps were frozen before execution and have **not** been re-selected afterwards. Any successor
surface must be preregistered on principle, not fitted to these numbers.

**Open decision — RED says close reconstruction, but the diagnostic that was built to
distinguish "cap-limited" from "approach-limited" says cap-limited.** Resolving that tension is
a command decision, not a threshold to quietly retune.

Next gates, in order (D0R remains conditional and is NOT started):

- **D0** — Jaccard-optimal operating-point reanalysis of the existing division posterior. No
  saved v4 predictions exist under `artifacts/` (`divevents/*.npz` are balanced training events
  only), so this needs the authorised inference-only re-run. Threshold on exact composite after
  fork edits, cross-fitted between embryos — never precision `0.9`. Gate: `≥+0.005` exact
  composite on both families, or `≥30%` of the D0' bilateral upside.
- **D1** — covariance fork audit if D0 falls short. Candidate population is the union of E0c's
  existing forks, the reachable proposals, and their competing parent assignments; the scorer
  jointly decides suppress/retain/reconstruct/steal. Note the retain decision is near-vacuous:
  only `0` and `2` of ~20k existing forks sit on a true divider.
- **D2** — joint fork/parent re-optimiser, only after D1 passes.

Missed-node displacement forensics run in parallel; the broad atlas stays deferred.

## Decision

Do not accept `0.889` as the ambition, but stop spending GPU on generic training. Seven
methods now show the same failure: apparent progress within one embryo family evaporates or
reverses across the family boundary.

The next round has two cheap falsification branches. At most one may graduate to full OOF
compute. Branch A ran on 2026-07-30 and failed; branch B is the only one still open.

## Branch A — CLOSED 2026-07-30 (failed Gate A1.2 on CPU)

Executed in full without a Kaggle session: `scripts/branchA_gate.py --stage all`.

- **A0.2** — the cache does not contain the mechanism. Both OOF GEFFs are hard-pruned
  upstream at `p >= 0.5` (observed minimum `0.500008 / 0.500015`), with in-degree exactly
  `1` for every target and zero targets carrying two parents. Top-two-to-`0.25` cannot be
  replayed from cache.
- **A1.1 PASS** — of E0c's matched-edge FN (`60 / 215`): recoverable within the cap
  `73.3% / 49.8%`; not already a transformer edge `60.0% / 29.8%`; never enumerated by E0c
  at all `36.7% / 18.1%`. Every level clears the 10% bar.
- **Mechanism** — 100% of those never-enumerated candidates are beyond E0c's 6 µm tight gate
  (median `8.29 / 7.62 µm`). Enumerating them needs no transformer, so the breadth half
  was testable on CPU.
- **A1.2 FAIL** — one global 10 µm enumeration, exact patched scorer:
  44b6 adjJ `0.8821 -> 0.7225` (`-0.1596`), 6bba `0.8068 -> 0.6572` (`-0.1496`).
  Edge TP *falls* (`1268 -> 1177`, `1664 -> 1532`) while FP roughly triples/doubles.
  Re-running at the E0c gate reproduced the cached graphs exactly, so the harness is sound.
- **Ceiling under a perfect probability** — granting the widened surface an oracle
  `p = 1` on true pairs still gives `-0.1319 / -0.1281`. No attainable edge scoring
  rescues it. Not a downstream artifact: with the short-track filter off, `-0.1281 / -0.1125`.
- **Cause** — the per-frame assignment is one-to-one; ~1550 / ~940 extra relink edges per
  crop mean each false assignment can displace a true one.

Gate A1 requires all bullets, so branch A is closed with zero GPU spent. Residual: this
falsifies expanded candidates inside E0c's Hungarian assignment, not inside the public
notebooks' global ILP. That combination rests on two independently negative components
(this result and closed lever 5, v122 coupled ILP at `-0.0633` on 44b6). Reopening needs an
argument for why the interaction beats both parts, not just that it is untested.

Evidence: `inventory/branchA_surface_audit.json`, `branchA_oracle_ceiling.json`,
`branchA_widen10um_twocrop.json`, `branchA_oracle_prob_twocrop.json`,
`branchA_control_nofilter.json`.

<details>
<summary>Original branch A preregistration (kept verbatim for audit)</summary>

### Branch A — clean extraction of the current public candidate-breadth idea

Two newly popular public notebooks report scores near `0.95`, but their scored submissions
append negative-time, out-of-volume hub/fork structures. That output is an evaluator exploit
and is permanently quarantined.

Sources inspected 2026-07-30:

- `https://www.kaggle.com/code/boristown/dark-agi-biohub-cell-tracking-solution`
  — downloaded notebook SHA256
  `3072d128e03e7ae4a874570bb38aea33eff1d7bb6ed00e58522de2a5be66b1d4`.
- `https://www.kaggle.com/code/yoikoarmor/biohub-modular-last-call-turned`
  — downloaded notebook SHA256
  `d14fa960c7e7d532185690e1476831982b746c918fb00db21ec61f08154ac6da`.

The clean pre-exploit pipeline contains one potentially unmeasured mechanism:

- retain each target's top two transformer parents down to probability `0.25`;
- apply a `10 µm` motion cap;
- solve globally with ILP afterward.

Threshold changes, D4 TTA, survival-cost ILP, and gap closing have already been measured.
Only the expanded pre-ILP edge topology may be new.

### Gate A0 — zero/low compute audit

1. Diff the clean pre-exploit inference path against the parity-proven E0c/coupled cache.
2. Prove whether the existing cache contains the top-two/`0.25` candidates.
3. If it does, replay locally. If it does not, build only a two-crop T4 cache:
   `44b6_d29c9ab2` and `6bba_bb9f20c3` (the highest-edge-mass crops in each family).
4. No exploit cell, negative time, out-of-volume coordinate, artificial hub, or synthetic
   division is retained.

### Gate A1 — candidate/oracle

Proceed to full 199-crop OOF only if both preregistered crops satisfy all of:

- expanded candidates recover at least 10% of baseline matched-edge false negatives;
- exact clean graph score is positive versus E0c on both crops;
- node recall does not fall by more than `0.005`;
- candidate growth and runtime remain compatible with a 9-hour Kaggle session.

Failure closes the public candidate-breadth idea. Do not tune thresholds on the two crops.

### Full gate

One global configuration, both LOEO folds, exact patched scorer. Promotion remains:
both folds positive, min-fold at least `+0.005`, no major regime collapse.

</details>

## Branch B — CPU-only family-boundary decomposition

Use existing E0c, coupled, selector, and M1 artifacts. Do not train a model.

Decompose each per-crop delta into:

1. node population/recall;
2. raw edge matching;
3. wrapper changes;
4. division term;
5. count multiplier.

Then test whether any deployment-observable descriptor has the same directional relationship
to error within both families. Family/crop identity and hidden `N_est` are forbidden.

Branch B graduates only if it identifies:

- a bilateral oracle ceiling of at least `+0.01`; and
- a feature/mechanism whose sign is stable within both families under leave-family-out
  evaluation.

Otherwise it is documentation, not another selector search.

## Stop rule and final hedge

Branch A has already failed. Branch B is the last open gate: if it does not clear a
bilateral `+0.01` oracle ceiling with a leave-family-out sign-stable mechanism, the stop
rule fires.

If neither branch passes, stop research compute. Preserve two final candidates:

- E0c for private robustness;
- clean v122 for public strength.

Do not spend on full-data training, additional M1 seeds, logit ensembles, or external
pretraining until a family-boundary mechanism clears its cheap gate.

## division_flow_pair (H0) — RED; the SISTER cap is the sole limiter

Frozen before execution (hash `35b6abef7f13`): parent <=15.0 um, sister <=8.5 um, pair-midpoint
<=6.0 um from `mother + local_flow`, knn_k=16 / knn_min=4.

| surface | parent | sister | 44b6 delta / reach | 6bba delta / reach | candidates |
|---|---|---|---|---|---|
| geometric_core | 10.5 | 8.5 | +0.0195 - 5/20 | +0.0142 - 20/93 | 4.20M / 3.21M |
| division_flow_pair | 15.0 | 8.5 | +0.0195 - 5/20 | +0.0142 - 20/93 | 2.94M / 2.21M |
| outer_diag | 15.0 | 15.0 | +0.0783 - 20/20 | +0.0646 - 82/93 | 64.7M / 44.1M |

RED: 6bba `+0.0142` below the `+0.015` floor and reachable recall 25.0% / 21.5% below the 40%
floor. Raising the parent cap changed coverage by exactly zero; raising the sister cap recovered
everything. **The sister-separation cap was the sole binding constraint.** The frame-median
control scored identically, so the flow estimator is irrelevant here.

## H0b — flow-gated wide-sister surface: PASSES scientific viability

Preregistration amendment (hash `34f91ada4626`). Every division_flow_pair constant identical;
only the falsified 8.5 um sister prior relaxed: parent 15.0, sister 15.0, midpoint 6.0, same kNN
estimator, same resolver, no family routing, no sweep.

| | 44b6 | 6bba |
|---|---:|---:|
| composite | 0.8299 | 0.7103 |
| oracle delta | **+0.0704** | **+0.0613** |
| reachable recall | **18/20 = 90.0%** | **78/93 = 83.9%** |
| candidates | 14.37M | 10.15M |
| metric-visible | 91,560 | 253,526 |

Both viability gates PASS (>=+0.03 and >=60% bilaterally), retaining 90% / 83% of the
GT-informed D0' ceiling on a surface that never consults ground truth. Candidate total
**24.52M exceeds the 15M raw budget** -> engineering classification, not a kill.

## H0b rank-compression audit — COMPLETE, PASS

Superseding the earlier "unfinished" note: the run completed. Rank of the true pair among its
own mother's candidates on the full 15/15 surface (label-free single features, no threshold
selected from any candidate's label):

| ranking | 44b6 med / K1 / K3 | 6bba med / K1 / K3 |
|---|---|---|
| **midpoint_residual** | **0.0 / 0.70 / 0.80** | **0.0 / 0.71 / 0.93** |
| parent_midpoint | 0.0 / 0.55 / 0.85 | 0.0 / 0.55 / 0.87 |
| fwd_support | 5.0 / 0.15 / 0.45 | 2.0 / 0.34 / 0.63 |
| sister_separation | 8.0 / 0.05 / 0.30 | 2.0 / 0.33 / 0.57 |
| persistence | 10.5 / 0.10 / 0.20 | 4.0 / 0.21 / 0.45 |

Shortlist sizes (both families): K1 4.96M, K3 14.37M, K10 42.74M.

**Gate PASSES bilaterally with margin.** End-to-end reachable retention:
- **K=3**: 80.0% (16/20) and 81.7% (76/93), shortlist **14.37M** — inside the 15M budget.
- K=1: 70.0% (14/20) and 62.4% (58/93), shortlist 4.96M.

The flow-midpoint residual puts the true pair at **median rank zero in both families**, while
sister separation ranks it 8.0 / 2.0 — a weak discriminator, which is precisely why using it as
a hard 8.5 um veto destroyed 75-79% of reachable divisions in H0. The quantity that works is the
flow-predicted centre-of-mass residual: the pair midpoint, not either daughter individually.

## NEXT ACTION

H0c cascade compression is viable and is the recommended architecture: broad 15/15 geometry
generation -> cheap flow-midpoint rank pruning to top-K per mother -> expensive multimodal critic
on the shortlist only -> graph conflict resolution last. Recommended operating point K=3.

Not yet tried: a learned cross-fitted combination of these features (H1), which should beat any
single fixed ranker. Reverse-time association, secondary detection and DeepCenter remain
unavailable locally; reverse-time needs the V18 source still blocked at retrieval (403 on
version-pinned Kaggle pulls). H1/H2/H3 remain unstarted and no GPU has been spent.

## H0c exact replay — PASSES all four gates (2026-07-31)

Frozen cascade, config hash `04eeac97500d`, K and ranking inherited from H0b and not retuned:
15/15 generation -> flow-midpoint top-3 per mother -> suppress-all -> oracle selection (ceiling
only) -> add-replace -> exact patched scorer. Per-crop baseline scored alongside.

| | 44b6 | 6bba |
|---|---:|---:|
| baseline composite | 0.7595 | 0.6490 |
| H0c composite | **0.8221** | **0.7086** |
| **delta** | **+0.0625** | **+0.0597** |
| division | TP0/FP93/FN26 -> **TP16/FP0/FN10** | TP4/FP582/FN121 -> **TP76/FP0/FN49** |
| divJ | 0.0000 -> 0.6154 | 0.0057 -> 0.6080 |
| adjEdgeJ effect | +0.0010 | **-0.0006** |
| retention | 16/20 = 80.0% | 76/93 = 81.7% |
| shortlist | 8,284,112 | 6,086,890 |
| node invariance | N_pred and node_recall identical every crop | same |

Gates: delta >=+0.03 bilaterally PASS; retention >=60% bilaterally PASS; <=15M candidates PASS
(14,371,002); no unexplained node damage PASS (invariance verified per crop, not assumed).

**Edge-side cost is real and must not be glossed.** adjEdgeJ falls `-0.0006` on 6bba, and
22/71 and 79/128 individual crops regress (worst `-0.0117` / `-0.0113`). Suppress-all deletes
fork children and add-replace steals parents, trading a little edge quality for division credit.
At the oracle point the trade is overwhelmingly favourable, but **the edge cost is unconditional
while the division gain is conditional on correct fork selection** — a weak classifier keeps the
cost and loses the gain. H1 must report exact composite, never the division term alone.

## H1 — OPEN (next action)

Not started. Requirements, in order:

1. Persist the canonical candidate table over the frozen 14.37M top-3 shortlist with full
   provenance (crop, frame, mother/daughter ids, config hash, proposal-source flags).
2. Emit the **metric-visible (annotated-mother) subset size** of that shortlist — not yet
   measured; H0b's 91,560 / 253,526 are for the un-ranked wide surface and are upper bounds.
   This is the true precision denominator.
3. Cheap deployment-observable features first: midpoint and parent-midpoint residuals, forward
   association support, parent/daughter persistence, local density and track history,
   physically scaled covariance/eigenstructure (voxel spacing 1.625/0.40625/0.40625),
   fluorescence mass conservation, peak splitting.
4. One cross-fitted L2 logistic or compact MLP, leave-family-out both directions, before any
   3D CNN. Unlabeled candidates excluded from supervised loss, retained during exact replay.
5. Encode expensive image features once per node/event and fuse cached embeddings per
   candidate. Never run a 3D encoder over 14M pairs.

Promotion: exact composite >=+0.005 bilaterally, or >=30% of the H0c upside. Only then is the
five-frame multimodal critic or large pretraining authorised. Reverse-time, secondary-model and
DeepCenter evidence are later ablations; V18 remains blocked at retrieval and H1 does not wait.
