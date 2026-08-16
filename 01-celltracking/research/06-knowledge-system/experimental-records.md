---
id: 06-knowledge-system/experimental-records
title: Experimental Records
area: 06-knowledge-system
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- ledger
- experiments
---

> **Provenance:** migrated verbatim from `reports/EXPERIMENT_LEDGER.md` on 2026-08-16 during the research-machine restructure. Body preserved unchanged below.

# Experiment ledger

**Frozen evidence through:** 2026-07-30
**Full historical tree:** Git tag `pre-lean-2026-07-30` (`7897511`)

## Authoritative floor

E0c is the private-safe baseline: public `0.889`, exact patched LOEO OOF
`0.7595 / 0.6490`. The full wrapper is parity-proven and scored on all 199 crops.

Canonical result: `inventory/e0c_score_full.txt`.

## Closed methods

| Method | Exact decisive result | Verdict | Recovery point |
|---|---|---|---|
| Breadth candidate reranking | beat the wrapper ranking on neither held-out family | saturated | journal 2026-07-13 |
| Learned division posterior | mean recall at precision 0.9 ≈ `0.045`, near zero on 3/4 embryos | closed | journal 2026-07-13 |
| Isolated DAXI redetection | oracle recovery `0% / 8.8%`, below 20% gate | closed | journal 2026-07-13 |
| Temporal accumulation v3 | 44b6 null; 6bba negative with bootstrap lower/upper evidence below zero | closed | journal 2026-07-14 |
| v122 coupled ILP | 44b6 `-0.0633`, 6bba `+0.0507` | bilateral fail | `inventory/coupled_score_2026-07-29.txt` |
| A/D selector | perfect oracle min-fold only `+0.0056`; learned leave-family-out rules harmful | closed | `inventory/selector_audit_2026-07-29.json` |
| M1 domain-randomised model | same-family `+0.0074`; cross-family `+0.0010`, CI `[-0.0102,+0.0132]` | stopped after one seed | `inventory/m1_selection.json`, `inventory/m1_heldout_result.json` |
| Pre-ILP candidate breadth (10 µm) | 44b6 `-0.1596`, 6bba `-0.1496`; still `-0.1319 / -0.1281` under a perfect oracle edge probability | closed on CPU, no GPU spent | `inventory/branchA_*.json`, journal 2026-07-30 |

## Reopened method

| Method | Exact decisive result | Verdict | Evidence |
|---|---|---|---|
| Jaccard-optimal joint fork suppression + reconstruction | composed oracle ceiling `+0.0783 / +0.0737` (GT-free child retention); suppression contributes `+0.0601 / +0.0599` over Oracle-C-alone | GREEN, primary track | `inventory/phaseb_oracle_d0prime.json`, journal 2026-07-30 |

Distinct from the killed "high-precision trajectory posterior": that was gated at precision
`0.9`, whereas the metric rewards maximising exact composite. After suppression the division
count starts at `TP0/FP0/FN26`, so `J = k/(26+m)` for `k` true and `m` false forks added — a
detector at 30–50% precision clears the `+0.005` gate. Realizability is unproven; the oracle
selects forks with ground truth in every arm.

## Mechanistic conclusions

- The v122 improvement is survival pruning: it improves the count multiplier but collapses
  node recall on sparse 44b6.
- No deployment-observable selector transferred the sign of that pruning benefit.
- M1 made raw held-out linking worse (`0.6499 -> 0.6421`) and emitted 20% fewer nodes. Its
  `+0.0010` composite change was count credit, not improved tracking.
- The recurrent obstacle is embryo-family conditional shift, not insufficient compute on
  the same training recipe.
- E0c's fork layer is essentially pure noise: 11,441 forks on 44b6 and 9,012 on 6bba, of which
  `0` and `2` sit on a true GT divider. Only 93 / 584 are metric-evaluable; the rest fall in
  unannotated regions. Suppressing all of them is edge-neutral (`-0.0000 / -0.0020`), which is
  why suppression and reconstruction are worthless apart and strongly super-additive together.
- Candidate breadth fails for a structural reason, not a scoring one: the per-frame
  assignment is one-to-one, so widening the gate to 10 µm adds ~1550 / ~940 extra relink
  edges per crop and each false assignment can displace a true one. Edge TP falls below
  baseline even when every true pair is given probability 1.
- Exact graph-level checkpoint selection is retained as good infrastructure: M1 epoch 10
  scored `0.7963` internally while later epochs fell as low as `0.6820`, despite monotonically
  improving training loss.

## Public deployment evidence

- E0c: public `0.889`, private-safe OOF floor.
- Clean v122: public `0.908`, but fails bilateral OOF and is a hedge rather than a promoted
  private model.
- Public notebooks advertising roughly `0.95` on 2026-07-30 were inspected and contain a
  scored negative-time hub/fork augmentation stage. Their leaderboard score is not evidence
  of a clean tracking advance and that stage is quarantined.

## Recovery

Use Git history rather than keeping dead code in the active tree:

```powershell
git show pre-lean-2026-07-30:<path>
git worktree add ..\Biohub-CellTracking-2026-historical pre-lean-2026-07-30
```

Do not restore an entire historical plan into the active tree. Recover only the file needed
to reproduce or audit a specific result.

## Cycle 2026-07-31 — deployment programme

**Best public moved 0.908 -> 0.914.** P0-A reproduced the public 0.913 exactly; P0-B (clean base +
source-locked reverse-time w=0.20) reached 0.914. The `+0.001` delta is exactly one unit of LB
resolution, so reverse-time is not harmful but not established as beneficial.

### Objective corrected

The leaderboard POOLS with edge-volume weighting; 44b6 is only **14.94%** of edge mass. The old
bilateral-delta gate rejected every better pooled arm (v122 +0.0337, C +0.0315, Bp +0.0262 pooled
over E0c). Primary metric is now the exact pooled composite; min-fold is a robustness constraint.
Proof `scripts/verify_pooled_objective.py`, locked by `tests/test_pooled_objective.py`.
The objective is closed-form, verified to 2.27e-13 at corpus scale.

### Newly closed methods

| method | decisive result | verdict |
|---|---|---|
| Hub/fork exploit | -0.0027 / -0.0007 under the patched scorer | score-NEGATIVE, not merely illegitimate |
| Detector diversity (cheap) | threshold variants strictly nested, 0 new nodes; union of 5 gains +0/+1 GT nodes | empty |
| Appearance x appearance stacking | FPs concentrate on the SAME mothers (7-232x independence), lift 0.00 | closed with mechanism |
| Zebrahub as a division corpus | 5.7-11.7 terminations per division; oracle over 30 anchor x stride pairs still wrong-signed | dead on lineage |
| CTC replacement corpus | "cloning of datasets or their parts, including reference annotations, is strictly forbidden" | licence-blocked |
| H1-T conditional pair ranker | b = 0 in 44b6 -- correct set is a strict SUBSET of geometry's | closed structurally |
| Synthetic split patches | real-vs-synth CV AUC 0.9888; synth-trained -> real AUC 0.664 vs 0.867 | dead |

### Deployable and corpus-verified

| mechanism | pooled delta |
|---|---:|
| node budget (keep_frac 0.975, arm A) | +0.00157 |
| ssl x geometry veto | +0.00141 |

### Oracles (real ceilings, not deployable)

- H0c cascade **+0.06012** pooled; live-filter variant +0.0641/+0.0646 per-family.
- Hybrid substrate: 6bba reachable divisions **68 -> 101** for ~50 aux nodes; reachability sets are
  NOT nested, so the prior table understated the ceiling.

### Reporting corrections (all mine)

H1-M pooled ~+0.0023 -> **+0.00007 / -0.00131** (an in-family CV ceiling probe quoted as
cross-family, ~30x); node budget +0.00822 -> **+0.00157** (5.2x); FN association share 63.5% ->
**43.3%** (1.5x). Standing rule: corpus numbers only, basis named explicitly.

### Method findings

- SMD audits drastically understate multivariate domain separability (passed at mean |SMD| 0.302
  while 98.9% separable).
- FN attribution: 56.7% never detected, 43.3% detected then discarded.
- Break-even: detection needs 40.6% precision, a division action 10.15% -- and the 10.15% is at
  full recall.
- E0c's published baseline contains thousands of out-of-volume coordinates, hidden by
  `max(0, int(round(v)))`. Count disputed (7,349 vs 14,319); volume guard defaults OFF.

## Cycle 2026-07-31 (late) — the deployable list went to zero, the substrate went green

### Substrate: RESOLVED, and it is the best we have

LOEO fold-0 strict on the P0-A/P0-B substrate, recovered from a FAILED kernel with **zero GPU**
(all 71 crops had been predicted; the run died at the export assertion on 2 nodes out of 1,900,633
with out-degree 3). Manifest audit PASS, no leakage path.

| quantity (71 crops, fold 0 / 44b6) | value |
|---|---:|
| adj_edge_jaccard | 0.89859 |
| node_recall | 0.98457 |
| division_jaccard | 0.01587 (TP 2 / FP 100 / FN 24) |
| **reachable GT divisions** | **22 / 26** |

Beats E0c 20/26, clean903 20/26, v122 15/26. H0c oracle on this substrate is **+0.0830** on 44b6;
a selector recovering 10 of 26 while admitting 60 false forks still returns **+0.010**.
**Bounded:** 44b6 is 14.94% of edge mass and 6bba is unmeasured on this substrate.

### Newly closed methods

| method | decisive result | verdict |
|---|---|---|
| Node budget (any under-predicting substrate) | arm D corpus **−0.00088**, P(Δ>0)=0.109, pooled optimum keep_frac 1.00; P0-B direct −0.00001 | **CLOSED** — gain requires over-prediction |
| ssl × geometry veto as a bolt-on fork filter | P0-B divisions TP0/FP8/FN3 ⇒ divJ already 0, ceiling **exactly 0** | **CLOSED** — mis-scoped; it is the H0c admission gate |
| Association repair via bipartite competition | LOFO **+0.00099 / +0.00002**, 50× disagreement on an in-sample net of +4 targets; inside `target_taken` the transformer picks the true parent **9.36%** vs ~50% break-even | **CLOSED** — sign-unstable |
| Orphan-target swap | blind swap −0.01667; best in-sample rule +0.00059; LOFO −0.00019 / +0.00006 | **CLOSED** |
| Blanket short-component retention | branch A exact control: edge TP *falls*, adjJ 0.8821→0.8766 / 0.8068→0.7924 | **CLOSED** — re-added nodes steal bipartite matches |
| Edge-level selection among short-component deletions | deleted true edges statistically identical to selected (prob 0.785 vs 0.786; raw_um 2.30 vs 2.30) | **CLOSED** |

### Mechanistic conclusion that generalises

**Node budget's arm-A gain was a count-multiplier effect, not a tracking improvement** — ~116% of
+0.00157 was the multiplier and ~−16% was edge quality. Any future mechanism whose gain decomposes
mostly into the multiplier should be treated as a metric artifact and audited against the
pipeline's own `metric_hack_used: false` declaration before deployment.

**CORRECTION 2026-08-01 — the node-ratio explanation was WRONG.** This section previously said the
sign "tracks the node ratio" (E0c +0.0832 → +0.00157; v122 −0.1598 → −0.00088; P0-B −0.1028 →
−0.00001). **The empirical verdicts are unchanged and node budget remains CLOSED, but the mechanism
was misattributed.** The per-node count cost is `0.1·tp_i/N_est_i`, and `N_est_i` is GT metadata, so
it is **exactly invariant** to whether the substrate over- or under-predicts. The node ratio enters
only through `w_i = 1 − 0.1·r_i`, which moves the decision threshold by **1.1%** across the entire
±0.16 range. Per-crop node ratios are in fact **mixed-sign on both substrates**.

What actually flipped the sign is the **d_tp/d_fp composition of the deleted components**
`[4 movies, exact scorer]`: E0c deletions carried `d_tp = −5, d_fp = +8` (`q_net = −1.67`, so
deleting was correct → +0.006339), while P0-B deletions carried `d_tp = +5, d_fp = 0`
(`q_net = +1.00`, so deleting was wrong → −0.0000103). P0-B already runs
`filter_short_track_components`, so its weakest surviving components are *correct tracks*.

**Generalised rule (closed form, verified):** retain a component iff
`q = d_tp/(d_tp+d_fp) > (Jbar + 0.1·n·ρ_i/a)/(w_i + Jbar)`, floor **0.3994** — which independently
reproduces this project's separately-derived 40.6% detection-precision bar. Judge an edit by the
edge quality of what it touches, **not** by the substrate's aggregate node ratio.

### Association FN attribution — complete

All 27,705 corpus FN assigned to a first-loss stage, parity-exact against the agent5 ledger (zero
mismatches) and against the patched scorer (dNUM error 0.000e+00 on 10 crops). 43.80% never
detected; recoverable pool 43.32% → **+0.13288 GT ORACLE**.
**Unit economics: one net-correct repair = 1.095e-05 pooled ⇒ +0.002 needs 183 net-correct repairs.**
Largest single pipeline-caused loss: `filter_short_track_components` deletes **5,311 GT edges the
relink had already linked correctly** (19.2% of all FN). Component-level selection is the only
untested handle on it.

### Defect fixed

`scripts/win_bet/phaseb_node_budget.py` accepted `--arm` and ignored it — `--arm D` scored arm A
while labelling the output "D". Fixed and recorded as **trap 14**. Caught only by per-crop parity
against cached anchors.

### Deployment state

Best public **0.914** (P0-B) unchanged. **P0-CR submitted** (ref 55147215) as a base-dependence
probe: v122 + reverse-time + volume guard, audit PASS 10/10, expected 0.908–0.912 — not a climb.
**There is currently no deployable mechanism between 0.914 and 0.920.** The division track on the
22/26 substrate is the only live route and its selector is unbuilt.


## Licence-blocked external assets (running list)

| asset | block | date |
|---|---|---|
| CTC (Cell Tracking Challenge) data | "cloning of datasets or their parts, including reference annotations, is strictly forbidden" | 2026-07-31 |
| OrganoidTracker marginalisation code | GPL-2, no MIT header — reimplement, never vendor | 2026-07-31 |
| **CAP (`YXSong000/CAP`)** | **NO LICENCE AT ALL** (`license: null`, `/license` 404, zero LICENSE blobs) ⇒ all rights reserved. Stricter than GPL-2: GPL-2 grants use, no-licence grants nothing. Compounded by three more: it imports **CoTracker (CC BY-NC 4.0)** at runtime, its only obtainable weights are CC BY-NC 4.0, and it trains on **CTC** (already blocked) while redistributing CTC eval binaries. | 2026-08-01 |

**Standing rule:** audit the licence BEFORE reading the code for reuse, not after. Two of the three
blocks above were found only after substantial reading. Also verify advertised checkpoints exist —
CAP's abstract claims "code and model checkpoints are available" and **none exist**: no weights in
the tree, zero releases, no HuggingFace repo.

## Cycle 2 (2026-08-01) — the division route CLOSES; acquisition state is the only live mechanism

### Closed

| method | decisive result | verdict |
|---|---|---|
| Shape-aware localisation | oracle +0.009125, deployed **−0.008778**, shrunk arms sign-opposite; shape does not separate misses (SMD ≤ 0.122) | CLOSED |
| CAP / track-as-point | licence-blocked four ways (no licence at all; CC BY-NC upstream; CTC training data); no weights exist; not detection-free | CLOSED |
| Flat mother classification | 92 true among 4,957,806 ⇒ needs AUC 0.983–0.9992, measured 0.86–0.92; capacity makes it worse | CLOSED |
| **Branch-emergence proposer (D1)** | **G2 11/92 vs 50% · G5 29.8× vs 100×. A GT-ORACLE in-sample logit is still 19.5× over budget.** 50% retention at K=20,000 needs AUC **0.9695**; measured 0.856/0.777 | **CLOSED — division route closes** |
| **Counterfactual image critic** | **foreclosed by arithmetic, never launched**: a perfectly independent channel needs AUC **0.9386**; appearance measures 0.657/0.496 and is 7–232× dependent | **CLOSED without GPU spend** |
| Dense registration as a motion vector | block deformable 2.167 µm, phase correlation 2.801 — both worse than assuming ZERO motion (1.817) vs kNN flow's 1.329 | DO NOT FUND (keep as state flags only) |

### Live

| mechanism | pooled Δ | basis | status |
|---|---:|---|---|
| **suppress-all, complete wrapper, P0-strict OOF** | **+0.0016970** | exact-pooled-OOF, 199 crops, CI [+0.000716,+0.002545] | **PROMOTED** — not bilateral (44b6 −0.000619) |
| acquisition-state relink (WS-A) | not yet measured | oracle ceiling +0.020212; bilateral core +0.003389 | RUNNING |

### Corrections to earlier ledger entries

- **suppress-all sign**: recorded −0.001728 (edges-only artifact) → **+0.002706 on E0c** through the
  complete wrapper. Trap 14 in reverse.
- **suppress-all does not port**: E0c +0.002706 *bilateral* → P0-strict **+0.001697** with 44b6
  **negative**. Mechanism: suppression is free only where divJ is already 0; E0c 44b6 had TP0,
  P0-strict 44b6 has TP2.
- **"P0-B has 8 forks" was a UNITS ERROR** — 8 is its metric division-FP count. P0-B has **305**
  graph forks (2.52e-3/node) vs P0-strict 2.86e-3 and E0c 3.98e-3. Substrates are comparable in
  fork density; the reason to re-measure was always divJ.
- **Temporal NMS is NOT free.** Recorded as losing zero true forks; that was a property of one
  score. Under other rankers it destroys **38–48 of 92**. Re-measure per lane.
- **Annotation-coverage inflation is FALSE** for division features — full-denominator AUC matches
  the annotated subpopulation. They are not inflated, just not strong enough.
- **`cos_daughter_axis` is wrong-signed** (axis vs migration, AUC 0.3234/0.5528); use
  `daughter_angle` (0.6551/0.6024).

### The programme-level pattern, now five instances

Node budget · H0d live-filter cross-term · reverse-time · the 22/26 substrate · suppress-all.
**Every one transferred badly across substrate or family.** Combined with three lanes where the
oracle cleared the bar and the deployable selector did not (localisation +0.009→−0.009, component
retention +0.0087→−0.0078, division +0.0646→+0.0001), the conclusion is that **+0.036 will not come
from better selection over existing candidate populations.** It requires a mechanism that changes
the base rate or the detection surface itself.

## External notebook audit 2026-08-01 (two user-supplied leads)

**A — `rauffauzanrambe/tracking-development-biotech`** (sha256 `33bcdd3a3c05150e…`, licence absent
from metadata). **NOISE — off-topic.** It predicts *biotech drug-pipeline* success from
`np.random.uniform` synthetic tabular data (molecule weight, binding affinity, clinical trial
phase). It is attached to the competition but contains no cell-tracking content. Exploit scan clean
(0 hits on all 8 signatures). No action.

**B — `josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking`**
(sha256 `f038937e995f7d00…`, dataset `josefreitasalvesneto/biohub-synthetic-dataset`, **CC0** claimed).
Exploit scan clean. Offers 18.5 GB of fully-labelled synthetic 3D volumes with **165,267 division
events** against our ~151, generated by a forward physical model of the microscope. **The licence
clears** — unlike CAP — and the craft is real. **REJECT anyway, on two measured grounds.**

**1. Its division geometry is mis-calibrated in the one parameter that governs reachability.**
Independently re-measured by me over all 199 crops and all 151 GT divisions:

| quantity | our measurement | notebook B |
|---|---:|---:|
| GT division events (199 crops) | **151** | ~304 (**2.01× over** — it is counting daughter *links*, not division *events*) |
| sister separation, median | **10.57 µm** (p10 6.36 · p90 14.36 · max 20.30) | **7.24 µm** |

7.24 µm is **32% below** the true median and sits *below the 8.5 µm sister cap that we measured
retains only 29.1% of true pairs* (against 93.4% at 15 µm). A model trained on that data would learn
a daughter-separation prior wrong in exactly the direction that already destroyed one branch of this
project.

**2. Synthetic→real transfer is already falsified here.** Synthetic split patches: real-vs-synth CV
AUC **0.9888** (i.e. trivially separable), synth-trained→real AUC **0.664** against 0.867
real-trained. The red team additionally showed per-feature SMD audits *understate* multivariate
separability — synthetic patches passed at mean |SMD| 0.302 while being 98.9% separable.

**Reopening condition:** a corrected generator whose sister-separation distribution matches
10.57 µm median / 14.36 µm p90, **plus** a demonstration that real-vs-synth discriminability is near
chance. Craft alone is not the bar; distributional fidelity in the load-bearing parameter is.
