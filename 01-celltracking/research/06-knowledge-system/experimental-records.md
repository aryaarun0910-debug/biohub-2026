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

---

## 2026-08-17 — Motion-gate (arm B) PROMOTED: bilaterally positive on the deployment substrate

Clean **paired** LOEO — P0-B base, official `tracking_cellmot` scorer, identical crops, only
`BIOHUB_ARMB_FLOW_GATE` differs — via four Kaggle T4×2 kernels
(`biohub-p3-armb-loeo-f{0,1}` treatment, `biohub-p3-base-armb-off-loeo-f{0,1}` baseline):

| fold | family | baseline (armB off) | armB | delta |
|---|---|---|---|---|
| 0 | 44b6 | 0.9037 | 0.9181 | **+0.0144** |
| 1 | 6bba | 0.7051 | 0.7141 | **+0.0090** |

Bilaterally positive, min-fold **+0.0090** (> +0.005 bar). Division FP dropped 103→98 (f0) while
edge Jaccard rose — the "admit valid relinks" mechanism, prob-independent (harmonic cannot break it).
**Decision: PROMOTE.** Next: build the P3+armB *submission* kernel; a human submits it per the
factory `submitcmd` discipline. Note: I first mis-anchored against the E0c numbers (0.7595/0.6490,
wrong substrate); the delta above is the corrected paired result.

---

## 2026-08-17 — H1 imaging gate lifted: packaged Zebrahub crops are public on Kaggle

**Question.** `bet-zebrahub-retrain` (the only stated top-3 path) was blocked on acquiring
Zebrahub imaging — 150–232 GB at level-1, against 383 GB free. Is there a cheaper source?

**Method.** Followed up the quick-wins swarm's lead on `kkunizaw/biohub-zmnscrops`. Listed both
`kkunizaw` datasets and their metadata via the Kaggle CLI; downloaded `zh001r_nodes.npz` (9 MB)
and `zh001r_iso.npy` (360 MB) locally and inspected them with numpy.

**Result (measured, not inferred).**
- `zh001r_iso.npy` → `(72, 20, 64, 64, 64) uint8`; crop 0 min 0 / max 255 / mean 51.81 /
  86.2% nonzero → real intensity imaging, 72 crops × 20 timepoints of 64³ isotropic volumes.
- `zh001r_tgt.npy` → identical byte size (377,487,488) → same shape; a paired target volume.
- `zh001r_nodes.npz` → 1440 arrays `f0..f1439` (= 72×20), each `(N,4) float32`, uniformly
  `[t, z, y, x]` with coords inside the 64³ box; 1,357,051 node rows total, ~900/frame.
- `kkunizaw/biohub-zmnscrops` (3.66 GB) declares `zmns001_crops.npz` (5.76 GB) +
  `zmns002_crops.npz` (7.13 GB), "windowed crops derived from the public Zebrahub multi-view
  imaging dataset … No competition data included". **Not opened yet.**

**Interpretation.** A complete, ready-to-train H1 detector substrate exists in 754 MB, and
Kaggle datasets attach to kernels directly — so the pilot needs no local download and no kernel
internet. The H1 blocker changes from *bandwidth* to *format*.

**Confounds / what this does NOT establish.**
- The third party's isotropic resampling scale and voxel size are unstated; no evidence yet that
  they match our level-1 (z-half/xy-half) convention or the competition feature derivation.
- Their node labels are unaudited against Zebrahub.
- `zh001r` is ZSNS001 only.
- Nothing here is a scored result. No LOEO number moved; `bet-zebrahub-retrain` stays `proposed`
  and its falsification is unchanged.

**Decision.** Record only. Establishing the voxel scale is the next cheap step; the choice between
their crops and our own `h1r_fetch_imaging.py` level-1 stream is deferred to green-light.

**Competitive note.** `zh001r` was uploaded 2026-08-17 and `zmnscrops` 2026-08-16 — a rival is
actively executing this retrain.

### Follow-up (same day): their voxel scale MEASURED — near-drop-in with our deployed grid

**Question.** The record above left "their isotropic resampling scale is unstated" as the blocking
unknown. What is it, relative to our deployed detector input?

**Our reference geometry (established).** `data/train/*.zarr` is `(100, 64, 256, 256) uint16` at the
competition `DEFAULT_SCALE = (1.625, 0.40625, 0.40625)` µm → a **104 µm cube**. The deployed detector
reads `zarr[t, ::dz, ::dy, ::dx]` with xy÷4, i.e. **64³ isotropic at 1.625 µm** — the *same shape* as
their crops. GEFF node coords are level-0 zarr voxel indices (verified: ranges z 1–62, y 4–252, x 0–240).

**Method 1 — nuclei density (weak).** Median nearest-neighbour spacing: ours 9.88 µm (dense P3
detector nodes, `DEFAULT_SCALE`); theirs 4.39 box-units. Implies ~2.25 µm/voxel. **Discounted** —
spacing is strongly stage-dependent (our own frames drift 11.7 → 8.9 µm within 5 frames).

**Method 2 — nucleus size (the ruler used).** Mean radial intensity profile around **real-label**
nuclei in both volumes (ours: 884 GT nodes over 18 crops, competition zarr subsampled to the
deployed 64³ grid; theirs: ~1500 nodes over 24 crop/frame combinations). Normalised profiles are
nearly identical in shape:

| r (vox) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ours | 1.000 | 0.750 | 0.431 | 0.208 | 0.121 | 0.088 | 0.067 | 0.049 |
| theirs | 1.000 | 0.711 | 0.391 | 0.168 | 0.092 | 0.077 | 0.065 | 0.047 |

Scale implied at seven profile thresholds (80% → 20% drop): **1.745, 1.746, 1.756, 1.762, 1.772,
1.814, 1.874 µm** — median **1.762 µm**, i.e. **1.084×** our 1.625 µm grid; crop extent ~113 µm vs
our 104 µm.

**Result.** Their crops are isotropic at **~1.6–1.8 µm/voxel** — the *same geometry family* as our
deployed detector input, roughly 8% coarser, not a different resolution lineage.

**Consequence for the plan.** The level-1 decision assumed retraining at z-half/xy-half, which
"voids the deployed 0.915 detector anchor". That premise does not hold for this substrate: a
retrain on `zh001r` stays close to the deployed input geometry, so the deployed anchor may survive.

**Confounds (stated, not resolved).**
- The size ruler assumes comparable physical nucleus size across embryos/stages. Their nuclei are
  independently **denser** (NN ~7.1 µm at 1.76 µm/vox vs our 9.88 µm) → likely a later stage with
  smaller nuclei, which would bias the estimate **upward**; true scale may sit nearer 1.625 µm.
  Either way the conclusion (same geometry family) is unchanged.
- Their intensity is rescaled to uint8; ours is uint16. Any retrain must renormalise.
- Their node labels remain unaudited against Zebrahub; `zmnscrops` remains unopened.
- Still not a scored result. `bet-zebrahub-retrain` stays `proposed`.

### Follow-up 2: H1 pilot scaffold built on `zh001r` — and a limit that narrows the claim

**Built.** Two scripts, both run and passing:
- `scripts/win_bet/h1r_zh001r_audit.py` — integrity gate over the packaged dataset (structure,
  label/imaging alignment, geometry ruler). Makes the measurements below reproducible.
- `scripts/win_bet/h1r_zh001r_smoke.py` — one CPU training step through the **deployed** surface
  (`vendor/.../train_unet_transformer.py`).

**Audit result (measured).**
- `iso` and `tgt` are both `(72, 20, 64, 64, 64) uint8`; `tgt` mean 27.8, 18.3% nonzero — a
  dilated/soft target volume, NOT the single-voxel convention `compute_detection_loss` uses, so
  we build our own targets from the node arrays.
- 1440 node arrays = 72×20 exactly; per-frame counts min/median/max = 551/926/1502; 1,357,051 rows.
- **Alignment PASS:** mean intensity at node positions 161.5 vs 70.6 at random voxels (**2.29×**).
  Their labels really do sit on nuclei.
- **Geometry, refined:** with the audit's broader sampling the ruler tightens to **median 1.677 µm
  (range 1.649–1.804), 1.032× our 1.625 µm grid**, crop extent 107.3 µm vs our 104.0 µm.
  (Supersedes the 1.762 µm figure in the record above, which sampled fewer crops. Same verdict,
  tighter: same geometry family as the deployed input.)

**Smoke result.** Packaged crop → `TemporalUNet3D.encode` → `compute_detection_loss` → backward
→ step. crop 0 / t 0→1: det_loss 0.7028 → 0.5787, grad-norm 4.55 → 2.83, UNet grad True,
detect_head grad True. Reproduced on crop 7 / t 5→6: 0.7138 → 0.5907. `det_logits` (1,1,64,64,64).

**THE LIMIT — this narrows the earlier claim.** The node arrays are `(N, 4) = [t, z, y, x]` with
**no track identity** (verified: every one of the 1440 arrays has exactly 4 columns). Without
`track_id`/`parent_track_id` no GT transition matrix can be built, so:

| H1 half | supported by `zh001r`? |
|---|---|
| **Detector** retrain (point supervision) | **YES** |
| **Edge / association** retrain | **NO** |

The earlier record called this "a complete, ready-to-train H1 detector substrate" — accurate as
written, but it should not be read as unblocking H1 in full. Our strategic thesis is that the
plateau is the shared public *edge* model, and the edge half is precisely the half this dataset
cannot supervise. Recovering identity would require registering these 64³ crops back onto the
ZSNS001 Zebrahub tracks we already hold (an unsolved point-set registration: their crop origins
and timepoints are not stated). Our own `h1r_fetch_imaging.py` stream **does** carry track ids and
`h1r_train_smoke.py` already proves the edge path on it.

**Consequence for the plan.** `zh001r` is a cheap detector-retrain substrate, not a full H1
unblock. The two lanes are complementary: packaged crops for the detector half, our own level-1
stream for the edge half.

**Still not scientific evidence.** A passing smoke is plumbing (CLAUDE.md rule 5). No LOEO number
moved; `bet-zebrahub-retrain` stays `proposed`; falsification unchanged.

---

## 2026-08-18 — Two negatives: motion-gate does NOT transfer to LB; sub-voxel refine KILLED

### A. Motion-gate (arm B): LOEO +0.0144/+0.0090 → **LB +0.000**

Submission `55585140` (P3 + arm-B) returned **0.915**, byte-identical in score to P3 alone
(`55274582`, 0.915). The lever was promoted on a clean **paired deployment-substrate LOEO**
(official `tracking_cellmot`, identical crops, only `BIOHUB_ARMB_FLOW_GATE` differing):
44b6 0.9037→0.9181, 6bba 0.7051→0.7141 — bilaterally positive, min-fold +0.0090 > the +0.005 bar.

**It delivered nothing.** This is the fourth consecutive optimistic central estimate.

**The important part is the instrument, not the lever.** The paired deployment-substrate LOEO was
adopted precisely to cure the substrate mismatch that discredited the earlier E0c-anchored
numbers. It is our best instrument and it still failed to predict the LB. Therefore:

- **The "+0.005 bilateral LOEO" promotion bar is now falsified as a sufficient gate.** It passed a
  lever worth 0.000. Discount every LOEO-only projection in `bets.yaml` accordingly.
- **Open diagnostic (highest-value methodology question we hold):** LOEO scores held-out-embryo
  crops from `data/train` with per-fold weights; the LB scores `data/test` crops. Whether the
  signal dies on the *crop population* or on the *per-fold weights* is not established. A cheap
  discriminator: score the deployed full-weight pipeline on the same LOEO crops and compare its
  LOEO-vs-LB relationship to the per-fold one.

### B. Sub-voxel centroid refinement: bilaterally negative → CLOSED

Kernels `biohub-p3-subvoxel-loeo-f{0,1}` COMPLETE; outputs fetched and scored with
`scripts/core/score_loeo_submission.py` against the **paired** `p3_base` baseline:

| fold | family | baseline | sub-voxel | delta |
|---|---|---|---|---|
| 0 | 44b6 | 0.9037 | **0.9033** | **−0.0004** |
| 1 | 6bba | 0.7051 | **0.7042** | **−0.0009** |

Bilaterally negative, an order below the promotion bar. **Decision: KILL `bet-subvoxel-refine`.**

**Mechanism, and it was predicted in advance from code.** The `retrain_recipes_2026-08-17` agent
derived from `predict_unet_transformer.py:495` (`coords[:, 1:] *= ds_arr`, no grid-centre term)
that z carries *zero* quantisation error and y/x a systematic 0.609 µm bias, RMS radial
**1.075 µm** — already inside the metric's free zone, since the measured scorer cliff starts at
~1.5–2 µm. A refinement that moves an error which is already free cannot pay, and
`OUTPUT_LINEFIT_SMOOTH` dilutes what little remains. The measurement confirms the prediction.

**Residual (not pursued here):** the same analysis implies a *constant* +0.5-voxel grid-centre
shift in y/x would cut RMS radial 1.075 → 0.642 µm at zero inference cost. That is a different
intervention from parabolic refinement and remains untested.

### C. Division confirmation (free observation from the same two scoring runs)

Full-fold division counts on the deployment substrate, both folds:
- 44b6 (71 crops): divJ **0.0152**, TP=2 FP=106 FN=24, reachable GT div 21/26
- 6bba (128 crops): divJ **0.0047**, TP=3 FP=507 FN=122, reachable GT div 68/125

613 division FPs against 5 TPs across 199 crops. This independently corroborates the
`metric_forensics_2026-08-17` finding on the full substrate rather than an 8-crop sample:
division **precision**, not recall, is the failure — and our deployed pipeline recovers
essentially zero divisions.

---

## 2026-08-18 (later) — Why arm-B moved nothing: it RESHUFFLES edges, it does not add them

Host-verified directly from the two TEST submissions pulled off Kaggle
(`c:/temp/armb_diag/p3_armb/submission.csv` vs `p3_harmonic/submission.csv`, different sha/md5,
diff in `churn_diff.json`). Aggregated over the 4 diffed test crops:

| quantity | P3 base | P3+armB | net |
|---|---|---|---|
| edges | 117,803 | 118,126 | **+323 (+0.274%)** |
| nodes | 122,083 | 122,214 | +131 (+0.107%) |
| divisions | 307 | 328 | **+21** |
| edge churn per crop | — | — | min 3.02% / **median 5.98%** / max 9.89% |

**The mechanism.** Arm B swapped roughly **6% of edges** for different ones and netted only
**+0.27%** more edges. It is a reshuffle, not an addition — and the metric was indifferent
(0.915 → 0.915). It also added 21 divisions, which given our 613-FP / 5-TP division regime is
neutral-to-harmful, not a gain.

**What this kills.** The hypothesis that the gate simply never fired on the hidden test
population is **REFUTED**: it fired hard (6% median churn) and produced no score movement.
So the LOEO→LB failure is NOT "the lever is inert at deployment". Remaining live mechanisms:
per-fold vs deployed weights (the swapped-in edges may already be correct under the stronger
deployed weights, so the swap is lateral), and/or the crop population. The full 2×2 is with the
`loeo_lb_gap_2026-08-18.md` agent.

**Generalised lesson (load-bearing for lever triage).** A lever that *reshuffles graph structure*
can move a large fraction of edges and still score identically. Corroborated twice now: arm-B solo
churned 7.148% for +0.000 (2026-08-02 ledger), and P3+armB churned ~6% for +0.000. **Prefer levers
that change the DETECTION SURFACE or NODE RECALL over levers that re-wire edges among fixed
nodes** — the latter class has a two-for-two record of scoring nothing.

## 2026-08-18 — A public retrained-UNet3D stack exists and is attachable (H1 stage-0 A/B)

Verified via the Kaggle API: `xiaoleilian/biohub-unet3d-weights-v2models` (created 2026-08-14)
contains `unet3d_v2_tophat_b32.pt`, `unet3d_v4_bright40.pt`, `unet3d_v4_bright40s1.pt`
(~22.4 MB each) **and `edge_prune_hgb.npz`** (a gradient-boosted edge pruner — i.e. a shipped
instance of the `bet-meta-ranker` idea, from a team at LB 0.918).

**Why this matters more than its size.** It is a **zero-training** answer to the central H1
question. Swapping a publicly retrained detector into our pipeline costs inference only and tests
"does a retrained detector help *our* stack?" before we spend any Colab hours on H1. Make it
stage 0 of the H1 programme. It also gives an independent read on whether retraining alone
reaches only ~0.918 (their public score) or more when composed with our post-processing.

**Caveat:** their weights were trained to their own preprocessing conventions; a naive swap may
mismatch normalisation/grid. That is itself cheap to establish and is the first thing to check.

---

## 2026-08-18 — DIVISIONS: the gate is set below the biology. Host-verified.

The `division_lane_2026-08-18.md` agent found that the 613 division FPs and the 146 FNs are
**disjoint populations** — `fork_at_matched_mother = 0` in BOTH folds. No re-ranking or filtering
of existing forks can recover a single division. **Host-verified the mechanism independently:**

**Deployed gate constants** (`src/biotrack/wrapper.py:123-124`):
`SAFE_DIV_MAX_UM = 4.7` (parent→daughter), `SAFE_DIV_SISTER_MAX_UM = 7.2` (inter-daughter).

**Measured GT division geometry** (all 151 events, all 199 crops, physical µm via DEFAULT_SCALE):

| family | n | inter-daughter median | p25 / p75 | frac ≤ 7.2 µm | max parent-daughter median | frac ≤ 4.7 µm |
|---|---|---|---|---|---|---|
| 44b6 | 26 | **8.98 µm** | 7.19 / 10.46 | 0.308 | 6.17 | 0.269 |
| 6bba | 125 | **11.47 µm** | 8.33 / 13.01 | 0.144 | 7.37 | 0.104 |

**The sister gate (7.2 µm) sits BELOW the GT median inter-daughter distance in both embryos.**
Passing BOTH current gates: **16 / 151 GT divisions (10.6%)**. Relaxing to sister ≤ 15 µm /
parent ≤ 10 µm: **129 / 151 (85%)** — an ~8× increase in what is even proposable.
(The agent reports 22/151 admissible and 81/89 under relaxation; its denominator is the
metric-reachable subset, mine is all GT events. Same direction, same magnitude.)

Worse, the proposal ranking is `score = parent_dist + 0.15 * sister_dist`
(`wrapper.py:831`), selected ascending — **tightest-first, i.e. exactly backwards** relative to
the measured GT geometry. The 11,582 emitted forks are hard-truncated at the gate constants
(99.2% ≤ 7.2 µm, p99 = 7.13), confirming the gate binds rather than the ranking.

**So the division failure is not precision and not ranking — it is that we cannot PROPOSE real
divisions.** We emit 613 tight FP forks and 0 forks at the mothers of the 146 real divisions.

**This lever is the right CLASS.** Per the churn finding recorded above, edge-rewiring levers are
0-for-2 on the LB; this one changes what is *proposed* (detection surface), and the division term
`0.1·dTP/(dFP+151)` (`metrics.py:34,519-522`) has **no offsetting mode** — every recovered TP is
strictly monotone, unlike edge churn which swapped TPs for TPs.

**Economics (agent-derived, not host-verified):** a pure FP filter caps at **+0.0027** perfect-play
and **+0.0007** honestly-fitted — below LB resolution, do NOT spend a slot on it. Turning divisions
off entirely is −0.0007. Break-even for +0.001 LB is **+8 net divisions**; post-processing ceiling
is 89/151 ≈ **+0.058**. First lever whose break-even sits comfortably below its measured substrate.

**Next action (CPU-only, no GPU, no slot):** relax the gates and measure
`fork_at_matched_mother` as a **COUNT**, not a score delta — deliberately immune to the
LOEO→LB transfer problem. Only if the count moves does the lane earn a scoring run.
Sequencing correction from the agent: linajea-style D3 is a *precision* mechanism and must come
AFTER the proposal fix, since today it would filter a fork population containing zero recoverable
divisions.

---

## 2026-08-18 — LOEO→LB: the deployed pipeline CANNOT be validly measured. Host-verified.

`loeo_lb_gap_2026-08-18.md` refutes all three pre-specified mechanisms and finds a fourth.
Host-verified the structural fact the whole argument rests on:

**All four `data/test` crops are BYTE-IDENTICAL to `data/train` crops** — SHA-256 over 102 files
each, content-only (paths excluded): `44b6_0113de3b` 587112aa…, `44b6_0b24845f` da73dcc2…,
`6bba_05b6850b` 361aa8ce…, `6bba_05db0fb1` 93b7f230…, all matching. Our local `data/test` is
**only the four placeholder movies**, and they are train crops. The LB is therefore scored by a
**hidden rerun on data we do not hold**.
*(Method note: a first host check using `md5sum` output compared FALSE because that output embeds
differing pathnames — the corrected content-only hash confirms identity. Recorded so the error
isn't repeated.)*

**Mechanism refutations (agent-measured):**
- **(b) "gate didn't fire on test": REFUTED.** `GATE_STATS` from the submission kernel: 8,150
  newly admitted / 9,373 newly excluded, 13.2% re-aim; churn 7.19% test vs 6.42%/7.69% folds —
  same intervention magnitude. (Independently corroborated by the host churn analysis above.)
- **(a′) chain redundancy: REFUTED, and in the opposite direction** — on identical crops the
  deployed chain *amplifies* arm B 3.8× (Δ +0.0365) vs the LOEO-strict chain (Δ +0.0097).
- **(c) pooling/rounding: REFUTED by arithmetic** — `summarise` pools convexly, so Δ_pooled
  ≥ +0.0090 for any composition; +0.009 on 0.9150 would print 0.924.
- **(d) division term: REFUTED** — both deployed arms score divJ = 0.0000 on the placeholders.

**The actual finding.** The deployed pipeline includes `pilkwang/biohub-temporal-unet3d-seed314159-v1`
(host-confirmed present in `scripts/kaggle_specs/p3_armb.json` and the LOEO specs), reported
`"train_datasets": 199` — i.e. **trained on every labelled crop**. LOEO must ablate it to avoid
leakage, so **LOEO measures a materially different pipeline from the one we submit**; and the
placeholder measurement is contaminated by memorisation (hence the implausible +0.0365).
The brief's mechanism (a) is also factually wrong for fold 0: the pack ships only `split_0`, so
LOEO f0 and deployment share the same primary weights (`loeo_retarget.py:124`).

**Consequence — the central methodological fact of the project:** *there is no configuration in
which the deployed system can be validly measured against ground truth.* Not a tuning problem;
LOEO cannot be fixed by adding crops.

**Not proven (agent's own caveat, endorsed):** the hidden-rerun mechanism is inferred, not proven;
a different GT annotation on the same movies survives as an alternative.

**Cheapest next experiment — zero GPU, zero slots:** score the five existing LB anchors
(0.906/0.913/0.914/0.915/0.915) locally on the placeholders and regress. Zero rank correlation
⇒ no instrument we hold is validated; high correlation with slope ≪ 1 ⇒ instruments usable with
a measured discount. **Do this before spending any Colab hours.**

**This inverts the H1 argument.** If H1 *replaces* the contaminated `seed314159` secondary rather
than sitting beside it, the pipeline would contain nothing trained on the 199 labelled crops —
making the system honestly measurable for the first time. That is a new argument for H1 not
previously in the portfolio (constrained by the known limit that `zh001r` supervises the detector
half only).

**Infrastructure bug found:** `GATE_STATS` is structurally unrecoverable from every LOEO run —
`loeo_export.py:117-124` deletes `submission.csv` before the provenance cell reads it, so all four
kernels died with `FileNotFoundError`. Fix is an edit-ordering swap in the spec.

## 2026-08-18 — H1: our OWN recorded Zebrahub scale is wrong by ~4.5× laterally

`h1_execution_spec_2026-08-18.md` turned the scale ruler on our own fetch, which had never been
checked (the host had verified only what the attrs *say*, not whether they are correct).
`scale_zyx = [0.62, 0.2195, 0.2195]` µm in `data/external/zebrahub/imaging/ZSNS003_L1.zarr` is
**wrong**; three independent instruments agree — nucleus half-width vs competition (4.48×
disagreement), NN spacing (1.89 µm, biologically impossible, vs 6.45 µm under the ruler), and
embryo extent (77×160×173 µm cannot hold 9,638 nuclei; 176×717×774 µm can). Likely an off-by-two
in pyramid indexing at `h1r_fetch_imaging.py:50-59`.

**Consequences:** the recipe's (2.62, 7.40, 7.40) resample factors are wrong twice over; the true
resample is a mild **z ×0.87, xy ×0.61**, making the **edge half much cheaper** than budgeted; and
the claim that a level-1 retrain "voids the deployed 0.915 anchor" is unsupported.
**Blocks the edge half; fix is a zero-cost metadata re-audit (needs internet, no GPU).**

---

## 2026-08-18 — Operating point: shift lever KILLED, N_est lever INVERTED, and fold 0 is our worst instrument

### A. Grid-centre shift — KILLED bilaterally, and its premise refuted at code level

Paired re-scores through an identical code path (baselines reproduced to <4e-5):

| arm | fold 0 (44b6) | fold 1 (6bba) |
|---|---|---|
| +1 | +0.00129 | **−0.00080** |
| **+2** (the predicted optimum) | **−0.00163** | −0.00012 |
| +3 | −0.00652 | — |

The +1 arm looks live on fold 0 and **flips sign on fold 1**; a 4,000-sample crop bootstrap puts
every arm's CI across zero except +3, which is significantly *negative*.

**Premise refuted (host-verified):** the claimed +0.5-voxel deficit requires block-averaged
downsampling, but the loader is **pure striding** — `raw = zarr_arr[t, ::dz, ::dy, ::dx]`
(`predict_unet_transformer.py:215`, `downsample=[1,4,4]`). Under striding, downsampled index `i`
*is* level-0 index `4i`, so `coords *= ds_arr` already lands correctly: the estimator is coarse
but **unbiased**. The `F.interpolate` fallback never fires (`target_shape = ceil(s/d)` = the
strided shape).

**CORRECTION TO OUR OWN LEDGER.** The 2026-08-18 record above credits
`retrain_recipes_2026-08-17`'s F7 with "a systematic 0.609 µm y/x bias, RMS radial 1.075 → 0.642
µm from a constant shift". The host verification behind that entry confirmed **only that the code
line `coords[:, 1:] *= ds_arr` exists** — not the bias interpretation, which is now refuted by
code AND by measurement. **F7's bias claim is withdrawn; the residual it proposed is closed.**
(F1/F2/F5/F6 are unaffected — those were verified as behaviour, not interpretation.)

### B. THE METHODOLOGICAL FINDING — fold 0 cannot resolve the effects we chase

Fold 0's entire scoring weight is **W = 21,210** edge events; fold 1's is **123,413**. Individual
crops carry `w_i` of 82–411, so a 4-edge change swings `44b6_2f31fc2f` by **+0.066** in its own
Jaccard. Measured noise floors: **fold 0 ±0.003, fold 1 ±0.004**.

**Fold 0 is our weakest instrument and the source of our most exciting deltas.** This is
*independent* of the `seed314159` substrate mismatch — arm B's +0.0144 cleared this floor and
still returned 0.000. **Levers must now clear BOTH bars: the noise floor and the substrate
problem.** Any future single-fold delta under ~0.005 should be treated as unmeasured.

### C. N_est lever is INVERTED — kill it

`N_est` is **GT metadata** (`metric.py:37-47`), unreadable at test time, so the
`kaggle_winners_playbook` "quantile operating point keyed to N_est" is not implementable as
written. Worse, the direction is backwards: we **under**-produce (pooled 0.70 / 0.93; 55/71 and
96/128 crops below 1.0), so we already collect a node-budget *bonus*. Forcing `N_pred = N_est`
costs **−0.018 (fold 0) / −0.005 (fold 1)** before any edge gain.

### D. Duplicates — confirmed free, but not worth spending on

20% of nodes sit within 7 µm of another in-frame node, but **91% are parallel-disjoint
components** (free); only 1.9%/5.8% of pairs contest the same GT node. Ceiling ≈ **+0.003**.
The merge half is already closed: **0 in-degree≥2 nodes across 3.8M**. It reshuffles edge
ownership without touching node recall — arm-B's exact profile, which is 0-for-2 on the LB.

### E. Break-even 0.50 confirmed — and the one surviving lever

`p* = adjJ/(m+adjJ)`; the node-budget term contributes only ~0.04, so the deployed **0.96875 is
not defended by the budget**. Surviving recommendation is the threshold-superset export: specs
written and validated but **NOT pushed** — `scripts/kaggle_specs/p4_detsweep_export_f{0,1}.json`
+ `scripts/kaggle_edits/detpeak_export.py`. Design note: `BIOHUB_DET_THRESHOLD` is deliberately
left at the deployed value so the graph stays bit-identical to `p3_base`; the patch records the
superset but returns only the pipeline subset, keeping edge prediction from going quadratic.
It is the only lever in this report whose mechanism changes **node recall** rather than edge
assignment, and one GPU export turns the whole threshold curve into a CPU replay.

---

## 2026-08-18 — IDENTITY RECOVERED: zh001r now supervises the EDGE half too. Host-verified.

**This overturns the 2026-08-17 record's central limit** ("EDGE/ASSOC retrain: NOT SUPPORTED —
no track identity"). Registration of the 72 packaged crops back onto the ZSNS001 Ultrack tracks
**succeeded exactly**: 72/72 crops, all 1,357,051 nodes matched, median residual **4e-5 µm**
(float32 round-off, not a fit).

**Host-verified the produced sidecar** (`data/external/zebrahub/zh001r_identity.npz`, 8.7 MB):

| check | result |
|---|---|
| frames with id-count mismatch vs the packaged node arrays | **0 / 1440** |
| labelled nodes | **1,357,051** (100%; zero unlabelled sentinels) |
| unique track ids | **116,320** |
| recoverable GT association edges | **1,258,182** (1,192,441 continuation + 65,741 division-daughter) |

**Why the first pass failed, and a correction to host guidance.** `ZSNS001_tracks.csv`
coordinates are **already in microns**, not level-0 voxel units. The transform is therefore
**isotropic**: `global_um = origin + 1.625 * crop_coord`, identity axis order, no flips. The
host's resume-message guidance — that an anisotropic ~4.1× y/x scale was needed — **was wrong**,
and so was the brief's "coordinates in level-0 voxels". Verified against the published pyramid:
Zebrahub level-0 is (1.24, 0.439, 0.439) µm; the (1.625, 0.40625, 0.40625) tuple belongs to the
**competition** zarr, a different acquisition.

**Geometry, now exact.** zh001r voxel size is **exactly 1.625 µm** — *identical* to the deployed
detector input. This supersedes the host's nucleus-size ruler estimate of 1.677 µm (the ruler was
right to ~3%, but the registration is exact). Crop origins sit on a 78 µm (48-voxel) lattice;
`t0` takes four values (158/316/474/632), 18 crops each.

**Method:** geometric-hash (Hough) vote over timepoint and origin jointly — 16/16 votes on every
crop. Node counts were too weak to filter on (~900 of ~27,000 nodes/frame); FFT histogram
correlation worked but was slower with far worse peak contrast.

**Consequence for H1.** The whole retrain — detector AND edge — is now supervisable from a
**363 MB + 8.7 MB attachable Kaggle dataset**, at the deployed 1.625 µm geometry, needing no
download, no resample, and voiding no anchor on geometry grounds. Recommendation: build
**Dataset A only** (`biohub-zh001r-identity`) until an experiment shows its four temporal blocks
are limiting.

**Caveat that bounds the whole lane.** The recovered labels are Ultrack's *automated* output, so
an edge model trained on them learns to **imitate Ultrack** — a ceiling as well as a floor.
Nothing here is scored; `bet-zebrahub-retrain` stays `proposed`.

### zmnscrops: characterised, and it supervises nothing

12.9 GB uncompressed raw uint8 imaging; keys only `ts` + `w0..w5`. Byte-level verified as
1.625 µm isotropic, full-z, 200×200 xy windows of Zebrahub **multi-view ZMNS001/ZMNS002**
(NCC 0.855 against the fetched level-3 pyramid, unique peak). **No public lineage table exists
for ZMNS embryos**, so it carries no labels. zh001r + the sidecar dominates it.

### Two infrastructure blockers found

- **`h1r_fetch_imaging.py` cannot fetch ZSNS001 as written** — it assumes 1 y/x chunk per frame;
  ZSNS001 level-1 has 3×4, so it raises on shape mismatch. Blocks any ZSNS001 stream lane.
- **`.venv\Scripts\kaggle.exe` is blocked by Windows Application Control** — use
  `.venv\Scripts\python.exe -m kaggle` (auth confirmed, 13 datasets under `aryaarun07`).

### Resample costs (measured, for the stream lane if ever needed)

Anti-aliasing is **mandatory**: skipping the Gaussian prefilter costs 28 RMS units against a
125-unit signal at 7.4× decimation. Linear + prefilter chosen (within 1.1% of cubic).
ZSNS001 frame: 16.5 s resample; 23.7 MB download in 63.3 s (latency-bound). 100 timepoints ≈ 2.2 h.

---

## 2026-08-18 — DIVISION THESIS TESTED AND REFUTED: the linker eats the second daughter

**Hypothesis under test (host, 2026-08-18):** the whole leaderboard gap (0.915→0.951 = 0.036) fits
inside the unclaimed division term (worth 0.1); the blocker is a mis-set geometric gate; relaxing
it recovers ~54 divisions and reaches ~0.951.

**Result: REFUTED.** Instrument: `scripts/win_bet/div_proposal_funnel.py` — walks every GT division
through the exact admission chain of `add_safe_divisions_postlink` (`src/biotrack/wrapper.py:770-845`).
Measured as COUNTS, so immune to the LOEO→LB transfer problem.

| stage | 44b6 (26) | 6bba (125) | **both (151)** |
|---|---|---|---|
| GT divisions | 26 | 125 | **151** |
| (a) mother detected | 26 (100%) | 105 (84%) | 131 |
| (b) both daughters detected | 25 | 77 | **102 (68%)** |
| (c) mother out-degree == 1 | 24 | 76 | 100 |
| (d) linked child IS a true daughter | 23 | 73 | 96 |
| **(e) other daughter is ORPHAN** | **4** | **31** | **35 (23%) — 61 LOST** |
| (f) child_dist ≤ 7.8 | 4 | 31 | 35 |
| **(g) parent_dist ≤ 4.7** | **0** | **1** | **1 — 34 LOST** |
| (h) sister_dist ≤ 7.2 [ADMITTED] | 0 | 1 | **1 of 151** |

**Blocker 1 — the linker consumes the second daughter (61 of 96, 64%).** `candidate_ids` admits
only nodes with **no incoming edge** (`wrapper.py:806`). In 64% of divisions whose daughters we
detect, the linker has already attached the second daughter to some other parent. Those divisions
are **unproposable at ANY gate setting**. This is a LINKER defect, not a division-gate defect, and
no post-processing can reach it.

**Blocker 2 — the binding gate is the PARENT gate, not the sister gate (my earlier framing was
wrong).** Measured true second-daughter geometry among survivors: parent_dist median **7.42 µm
(44b6) / 8.87 µm (6bba)** against `SAFE_DIV_MAX_UM = 4.7`; sister_dist median 9.01/10.35 against
`SAFE_DIV_SISTER_MAX_UM = 7.2`. The parent gate kills 34 of the 35 survivors; the sister gate then
removes **zero** more. The earlier record blamed the sister constant — **correction: 4.7 µm parent
is the binding one.**

**Blocker 3 — and this is what kills the thesis outright: relaxation drowns the signal.**
Gate sweep, true divisions admitted vs the false-candidate pool it opens (6bba):

| parent | sister | TRUE admitted | false-candidate pool | **pool per TP** |
|---|---|---|---|---|
| 4.7 | 7.2 | 1 | 1,640 | 1,640 |
| 8.0 | 10.0 | 9 | 25,042 | 2,782 |
| 10.0 | 12.0 | 24 | 57,170 | 2,382 |
| 12.0 | 15.0 | 31 | 103,008 | 3,323 |

Every setting that admits real divisions opens **~2,400–3,300 false candidates per true one**, and
the ranker `parent_dist + 0.15*sister_dist` sorts **tightest-first** — so true divisions, being
*wide* (7.4–8.9 µm), rank near LAST of thousands. With `divJ = TP/(TP+FP+FN)`, precision collapses
and the term goes to ~0. **Geometric proposal cannot separate true sisters from the pool.**

**Correction to the host's own ceiling estimate.** The earlier "89 of 151 metric-reachable" figure
counted divisions whose daughters were *detected*; it did not account for (d)/(e). The genuinely
proposable pool is **35, not 89** — so the post-processing division ceiling is roughly
0.1×(35/151) ≈ **+0.023 at perfect precision**, and perfect precision is exactly what is
unavailable.

### What survives, and where it points

1. The 0.1 division headroom is real and still unclaimed by the field. **It is not reachable by
   post-processing geometry.**
2. Reaching it needs two things, both of which are the **edge model**: a linker that does not steal
   the second daughter (blocker 1, 64% of the loss), and a **learned** discriminator able to pick a
   true sister out of ~2,400 geometric candidates using appearance/morphology, not distance
   (blocker 3).
3. This **collapses "divisions vs H1" into "divisions REQUIRE H1"** — and it is a direct argument
   for the Zebrahub sidecar's **65,741 division-daughter links** as the training signal we lack
   (we own 151 events; the sidecar has ~436× more).
4. Consistent with `scientific_incumbents_2026-08-17`: every method that does divisions well uses a
   *dedicated supervised division signal* (linajea's 4-class head, Trackastra's parental softmax,
   OrganoidTracker's division-likelihood CNN). **None uses geometric gates over a linker's output —
   which is exactly what we deploy.**

**Decision:** close the post-processing division lane. Do NOT relax the gates. Divisions become an
acceptance criterion for the H1 edge retrain rather than a lane of their own.

---

## 2026-08-18 — Agent contradiction RESOLVED: do NOT turn safe divisions off

`tier_reverse_engineering_2026-08-18.md` recommends, as a zero-cost win, deleting the
safe-division code path: "holding TP=2 and dropping FP to 0 gives 2/26 = 0.077, ~+0.006 by
deleting a code path". `division_lane_2026-08-18.md` had already **measured** the same action at
**−0.0007**. Host-resolved in favour of the measurement.

**The deciding fact** (`c:/temp/armb_diag/run_stats.csv`): `division_like_sources` equals
`safe_divisions_added` on **every** crop — 44=44, 63=63, 10=10, 211=211. **Every fork in the output
is produced by the safe-division patch**; no other code path emits divisions (consistent with the
pre-wrapper graph being a strict one-to-one matching). Therefore deleting the patch removes the
TPs along with the FPs, and the premise "hold TP=2" is false.

**Arithmetic, fold 0** (scorer: TP=2, FP=106, FN=24; current divJ 0.0152 → +0.00152):

| surviving TP after removal | divJ | contribution | delta |
|---|---|---|---|
| 2 (the report's assumption) | 0.0769 | +0.00769 | **+0.00618** |
| 1 | 0.0385 | +0.00385 | +0.00233 |
| **0 (what actually happens)** | **0.0000** | **+0.00000** | **−0.00152** |

Pooled by edge weight (f0 W=21,210; f1 W=123,413; f1 contribution +0.000475):
**−0.00063** — which reproduces `division_lane`'s independently measured **−0.0007** to three
decimals. Two independent routes agree.

**Verdict: keep `OUTPUT_SAFE_DIVISIONS = 1`.** The patch is net-positive by ~+0.0007 despite a
122:1 FP:TP ratio, because division FPs are only charged on annotated cells (~2.8%) while the
few TPs are charged in full. Removing it is a small but real loss.

**Class-C lesson (premise false in code), logged against the screening rules:** the recommendation
was arithmetically sound and rested on an unverified premise about provenance. One column of
`run_stats.csv` settled it. This is the second time in two days that an agent's *arithmetic* was
right and its *premise* was wrong — cf. the F7 "0.609 µm bias" withdrawal.

---

## 2026-08-18 — INSTRUMENT VERDICT: we hold none. The LB is the instrument. Host-verified.

### Contamination proven from the artifact, not inferred

Downloaded `pilkwang/biohub-temporal-unet3d-seed314159-v1 :: weights/unet_transformer/split_0/split_manifest.json`:

| field | value |
|---|---|
| `method` | `unet_transformer_`**`alltrain`**`_seed314159_v1` |
| `train` | **199 entries** — every labelled crop |
| `test` | 40 entries, **a strict subset of `train`** (199 unique stems across both lists) |
| all four placeholder crops present in `train` | **True** |

The component's own name says `alltrain`, its validation set is inside its training set, and it
memorised the four crops we score locally. **The deployed pipeline cannot be honestly evaluated
against any labelled data we hold.**

Two further facts the earlier diagnosis lacked:
- **DeepCenter was trained on `44b6` only** (71 train / 128 `6bba` val). It is CLEAN on `6bba`, so
  `loeo_retarget.py` **over-ablates it on fold 1** — contamination is layered by family in
  *opposite* directions, so no labelled subset is clean for the deployed chain.
- **The primary 50ep model's training set is undocumented** — the pack ships no
  `training_config.json` and no `split_manifest.json`. "split_0 held out 44b6" rests entirely on a
  directory name.

### Calibration: n = 6, and the local substrate is ANTI-informative

All four missing LB anchors were fetched and locally rescored (all four submission hashes
reproduce the ledger; `p2_armb_baseline` returned byte-identical to P0-B, independently confirming
the ledger's GATE 1 claim).

**LB vs local placeholder-4 score: Spearman +0.500, exact permutation p = 0.333, OLS slope 0.066.**
Indistinguishable from zero. The structure is worse than the summary statistic: the four non-armB
configs span the entire known LB range (0.906→0.915, nine quanta) while the local score moves
**0.0015**, with Pearson **−0.199**. All apparent correlation comes from the two arm-B configs
sitting high on both axes by coincidence. On paired contrasts — the form an instrument is actually
used in — of five contrasts: **one right sign (CI spans zero), one wrong sign, two confidently
signed FALSE POSITIVES, one vacuous.**

### Power: the +0.005 bar could never have gated anything

Minimum detectable effect at 80% power (20k-resample crop bootstrap, measured):

| substrate | MDE |
|---|---|
| fold 0 (44b6) | **±0.0043** |
| fold 1 (6bba) | **±0.0057** |
| pooled | **±0.0049** (pooling is WORSE than fold 0 alone) |
| placeholder-4 | **±0.0354** |

**The standing "+0.005 bilateral" promotion bar sits BELOW fold 1's own detection limit of
+0.0057** — it was never a valid gate, independently of the pipeline mismatch. Fold 0's weakness is
annotation sparsity, not crop count, so it cannot be fixed by adding crops.

### Verdict: the leaderboard IS the instrument — and it is also the cheapest

| | LB | bilateral LOEO |
|---|---|---|
| resolution | **0.001** | ±0.0057 |
| measures | **the shipped pipeline** | an ablated one |
| throughput | **~35 tests/week** (5/day) | 3–4/week |

**5.7× more sensitive, unbiased, ~10× the throughput.** This **strikes the standing rule** at
`research/07-outputs/submissions.md:275-276` ("never spend a submission to resolve an effect
smaller than ~0.005; OOF is the instrument") — that rule was written on the assumption that OOF
*was* an instrument. It is not.

**Protocol. Step 0 = a 1-slot determinism probe: resubmit the reigning best UNCHANGED.** If it
returns exactly 0.915, the LB is deterministic and every subsequent ΔLB is exact to ±1 quantum —
the instrument is characterised for one slot. Then one lever per slot, churn census before each,
promotion ladder ≥+0.002 adopted / +0.001 provisional / 0.000 killed, 25 slots reserved for the
final week. ~120 lever tests fit the remaining budget.

Alternatives rejected with reasons: dropping the contaminated component still leaves an
undocumented primary and caps at ±0.0043 — worse than the LB — for real score paid permanently
(kept as fallback if step 0 fails); leave-crop-out is provably invalid against a model whose
training set is the universe; variance-reduction has no unexploited variance left.

**Blocking infrastructure bug (now load-bearing for the protocol's churn census, not a nicety):**
`GATE_STATS` is unrecoverable from every LOEO run — `loeo_export.py:117-124` deletes
`submission.csv` before the provenance cell reads it.

---

## 2026-08-18 — Detection-threshold lever REFUTED (the last cheap lever), with a caveat I had to add

`detection_threshold_2026-08-18.md` recommends NOT spending the GPU export or a submission slot.

**Headline (agent):** T-class recoverability — the fraction of missed GT nodes that a lower
threshold could return — is **0.00% on 44b6** (0 of 263 misses) and **45.2% on 6bba**, of which
most is artifact: a crop-matched null (300 reps) puts the genuine excess at **84 of 9,604 (0.88%)**,
and **810 of the 1,098 T rows are ONE crop** (`6bba_6feb10f0`, recall 0.113) whose T-maxima sit at
median p = 6e-4 — functionally class D.

**Independent corroboration (strong, and label-free):** on 44b6, **0 of 28,800** sampled
subthreshold local maxima lie within 7 µm of any annotated GT, against a uniform-null expectation
of **97**. That is evidence about the detector's response field itself, independent of the
`d1_class` labelling.

**Economics:** break-even is **2,853 nodes per recovery (fold 0)** and **493 (fold 1)**; measured
at t=0.5 on 6bba it is **811** after correcting the pilot's 1.71× miss-rate bias. Theft is
negligible (15 events vs 84 recoveries; **zero** on fold 0), so the failure is on the GAIN side.
Reconciles with the `p*≈0.50` break-even: conditional precision in annotated territory is
**19.6%** against a required **41%**. Predicted curve: fold 0 monotone decreasing (−0.0064 to
−0.0196); fold 1 optimistic peak **+0.0014 at t≈0.60** — a quarter of its ±0.004 noise floor;
honest is negative everywhere. The mirror arm (raising t) gives 44b6 +0.0016 (half its floor).

**Verified in passing:** the agent's closed-form `pooled adjJ = Σ TP_i·m_i / Σ w_i` reproduces the
official scorer to the last digit (0.901802 / 0.703756), matching the host's independently
measured 0.9018 / 0.7038. Also corrects the sibling's export sizing: superset ratio is
**1.15–1.17×** (≈12–16 MB/fold gzipped), not the 3–6× previously assumed.

### HOST CAVEAT — the pilot artifacts come in two CONFLICTING sets

The `d1_class` pilots exist in two variants per family, same crops, same `representation=detector`,
same `match_authority=pregraph`, but produced under different fold weights:

| artifact | family | n | T | T/miss |
|---|---|---|---|---|
| `p3_d1_pilot_f0_v1` | 44b6 | 2,366 | **0** | **0.0%** |
| `p3_d1_pilot_f1_src_v1` | 44b6 | 2,366 | **203** | **72.8%** |
| `p3_d1_pilot_f1_v1` | 6bba | 9,604 | 1,098 | 45.2% |
| `p3_d1_pilot_f0_src_v1` | 6bba | 9,604 | 10 | 0.8% |

The report used `f0_v1` (44b6) and `f1_v1` (6bba) and did **not** flag the other pair, which tells
the opposite story on 44b6 (72.8% T-recoverable). The most likely reading is in-sample vs
out-of-sample: a detector evaluated on an embryo it trained on shows sub-threshold response at GT
sites it memorised, inflating T. If so the report used the honest (out-of-sample) artifacts and
its conclusion holds — and the **28,800-maxima null check is independent of this labelling
entirely**, which is why the conclusion survives the caveat. **But the fold→holdout mapping was
not verified in the report, so treat the exact T percentages as provisional.**

**Verdict: do not spend the export or a slot on thresholding.** Kill criterion already met on
artifacts in hand. Falsified iff some t < 0.96875 gives fold-1 Δ ≥ +0.005 with fold 0 no worse
than −0.002.

### The genuinely new finding: the L-class

**L-class (displaced/localisation deficit) is 95 on 44b6 = 36% of its misses**, and 597 on 6bba.
This is unreachable by ANY operating point — no threshold recovers a node whose peak is in the
wrong place — and on 44b6 it is **larger than anything thresholding touches**. Note this is NOT
the sub-voxel lever (killed: localisation already inside the metric's free zone at RMS 1.075 µm
vs a ~1.5–2 µm cliff); L is a coarser displacement. Together with `6feb10f0`-class crops
(recall 0.113) it points at the retrain lane, not the pipeline.

---

## 2026-08-18 — EDGE LOSS: hypothesis half right, and the important half FALSIFIED

### Confirmed (host-verified numerically): the loss cannot express "no parent"

`compute_loss` (`train_unet_transformer.py:55-72`) applies a **bare column softmax**
(`softmax(logits, dim=0)`), so every column sums to exactly 1 and the loss is **provably
shift-invariant down a column**. Host check: shifting one whole column by −5, −1, +1, +5, **+50**
changes the loss by **exactly 0.000e+00** in every case. Therefore `edge_prob` is a **SHARE, not a
confidence**, and "this node has no parent" is inexpressible. Trackastra's `1 + Σexp` background
term is what makes it expressible.

### Falsified: fixing the loss alone buys nothing, because the head is barely used

The agent ported `biotrack.wrapper.motion_relink_edges` (verified edge-for-edge identical,
`sym_diff = 0`) and replayed the deployed linker on real pre-wrapper graphs:

| `MOTION_RELINK_LEARNED_BONUS` | reachable GT edges recovered (of 9,020) |
|---|---|
| **0.00 (learned prob OFF)** | **8,263** |
| 0.75 (deployed) | 8,264 |
| 3.00 | 8,269 |

**Deleting the learned edge probability entirely costs ONE GT edge in 9,020.** At bonus 0 the
geometry linker already agrees with the head's nomination on 95.3% of edges — and in fact the
**geometry linker recovers MORE reachable GT edges (92.6%) than the head nominated (89.1%)**.

**Consequence for H1: the edge retrain is NOT a drop-in improvement.** A better edge model
changes nothing while the pipeline consumes its output this weakly. Loss structure and inference
assignment are **not separable levers** — abstention must be CREATED in the loss and CONSUMED by
the linker; either alone returns ~0.

### The one live route: ABSTENTION, with a single number gating it

The gap is mostly **false positives, not missing links** — 6bba: FP 14,355 vs reachable-but-unlinked
6,860 → **59% FP / 41% missing** (44b6: 76/24). So the lever is deleting bad edges, not adding good
ones. **Break-even deletion precision = 59.0% (6bba).** Measured operating points:

| signal | AUC | best usable precision |
|---|---|---|
| learned `edge_prob` | **0.701** (best) | **49.6%** (worst) |
| motion | 0.641 | — |
| raw distance | 0.630 | 57.3% (geometry best, ΔJ = −0.0002) |

**Good AUC, bad operating point — the exact signature of an uncalibrated share**, which is what the
shift-invariance predicts. Nothing currently clears 59.0%. Realistic value if the bar is cleared:
**+0.010 to +0.040** — the largest remaining number in the portfolio.

### Corroborations and traps

- **The head emits zero divisions, ever**: out-degree histogram `{1: N}` across three independent
  pregraph exports (2,044,806 edges) — host independently confirmed this earlier on
  `p3_d1_pilot_f0_v1` (all 246,287 sources out-degree 1, all targets in-degree 1). So λ_div returns
  **exactly 0** today, and divisions are only 0.117% of GT edges (≤0.0012 of edge Jaccard).
  Corroborates the linker-capacity lane from the opposite end.
- **Trap in the sibling's masking recipe:** `weight[div_rows] = 0` does NOT withhold division
  supervision under a column softmax (measured gradient 2.1e-02 on daughter columns). Removing the
  daughter **columns** from the mask is what actually zeroes it.
- **`test_acc` is degenerate:** on random 200×200 logits with 10 GT edges the vendored accuracy is
  **0.9973** while top-1 parent accuracy is **0.0**. This compounds defect F2 (checkpoint selection
  `score = test_acc * test_recall` has no precision term) — the selection metric is not merely
  precision-blind, it is near-constant.
- Reproduced the host's ceilings independently (0.9816 vs 0.9830; 0.8533 vs 0.8537).

### Files created (inert; nothing pushed or committed)

`scripts/kaggle_edits/h1r_edge_loss_patch.py` — E1 background term, E2 auxiliary sigmoid
(λ=1e-2), E3 real per-row weights + `H1R_DIV_MASK_COLS`, E4 focal-γ knob, E5 association-aware
checkpoint selection, E6 matching inference normalisation. Layers on `h1r_trainer_patch.py`;
13+9 patches apply cleanly to scratch copies, compile-checked, `--self-test` 12/12.

**Experiment 0 is CPU-only and can kill or confirm the whole lane before any GPU is booked.**

---

## 2026-08-18 — CROSS-EMBRYO GAP: hypothesis FALSIFIED, and it corrects the host's own framing

### The 44b6/6bba gap is largely a WEIGHTS artefact, not domain shift

Host-verified from `scripts/kaggle_edits/loeo_retarget.py:16-24`, which states the leakage
structure up front:
- **fold 0 = held-out 44b6 (71 crops); fold 1 = held-out 6bba (128 crops)**
- the support pack ships **ONLY `split_0`, trained on 6bba, holding out 44b6** → LOEO-**clean on
  fold 0**, **leaky on fold 1**
- confirmed in the run JSONs: both folds ran `arm=strict`, `secondary_enabled=False`,
  `deepcenter_enabled=False` — **but different weight files** (fold 0 = pack
  `split_0/edge_predictor_best.pth`; fold 1 = `/kaggle/working/loeo_weights/edge_predictor.pth`).

**Therefore the host's headline 0.9871 / 0.8695 node-recall gap (11.8 pp) compares TWO DIFFERENT
CHECKPOINTS and is confounded.** Apples-to-apples — one checkpoint (`split_0`) applied to both
families — the agent measures **0.8888 on 44b6 (never seen) vs 0.8637 on 6bba (in its training
set): a 2.5 pp gap, and in the "wrong" direction.**

**The detector scores WORSE on the family it trained on.** Memorisation does not rescue 6bba.
That falsifies the domain-shift hypothesis AND removes this gap as a target for H1: 6bba is
intrinsically harder, not distributionally shifted.

The 11.8 pp figure is further inflated when the `alltrain_seed314159` secondary is enabled — it
lifts 44b6 by ~9.8 pp but 6bba by only ~0.6 pp, because 44b6 has ~20k annotated nodes to 6bba's
~113k and a model that memorised all 199 crops recovers far more of the smaller set.

**Host correction:** the "n=2 embryos / cross-specimen generalisation is the real task" reframe I
offered earlier rested on this gap. **It is substantially weakened.** The *score* gap
(0.9033 vs 0.7042) is real, but its attribution to domain shift is not supported.

### Where 6bba's residual loss actually is — and it is NOT detection

Of the 13.5% residual on 6bba:
- **C = 7.3% (700 nodes)**: the detector **did** fire an accepted peak within 7 µm — min
  `best7_prob` **0.9698** — but the node never reached the matched graph. **A linking/pruning loss
  misfiled as detection.** Converges with the abstention finding (59% FP / 41% missing) and with
  the host's own split (6bba loses 0.150 to linking).
- **L = 6.2%**: no local maximum within 7 µm under either checkpoint, and no image peak within
  3.5 µm at *native* resolution in 83.6% of cases — **structurally unreachable**.

### Threshold lever: independently re-killed on the DEPLOYED weights

Under the deployed `split_0`, T = **10 of 9,604 GT nodes (0.10%)** on 6bba. A full offline τ sweep
gives **+0.10 pp of GT reach for a 2.67× node budget** at τ=0.01; on 44b6 the reach curve is
**perfectly flat from 0.99 down to 0.01**. **This resolves the host's earlier caveat about the two
conflicting `d1_class` pilot variants:** the 11.4% T class exists only under `split_1`, a
checkpoint that never saw 6bba — **not the deployed configuration**. The threshold kill stands, on
the right artifacts.

### Normalisation fixes measured dead

Per-crop standardisation is **already the deployed baseline**
(`predict_unet_transformer.py:319-324, 368-369` — per-crop quantiles from zarr attrs, train and
inference identical). Histogram matching would move background, not nuclei (bright tail matches to
8%, values at GT nuclei to 9%). The decisive control: both checkpoints consumed **byte-identical
input**, yet on the same 1,098 nodes `best7_prob` is **2.4e-3 under one and 0.999 under the
other** — the input is not the bottleneck, the weights are.

### Two genuine family differences that did survive

6bba nuclei are **~25% larger in radius** (half-max 4.47 vs 3.57 µm), and 6bba's image is
**sparser, not denser** (97 vs 244 nucleus-scale peaks/frame; NN 8.7 vs 7.6 µm) while carrying
**4× the annotation**. Note this contradicts the host's earlier inference from nuclei spacing that
6bba is a later/denser stage.

### Top-ranked next experiment (agent's, endorsed)

**One GPU kernel: `strict` arm over all 199 crops, no retrain, no submission** — separates the
leakage artefact from the real family gap and bears directly on the broken LOEO→LB instrument.

---

## 2026-08-18 — ROOT CAUSE FOUND: our own ILP setting makes divisions mathematically impossible

`linker_division_capacity_2026-08-18.md`. Hypothesis (linker structurally cannot divide)
**CONFIRMED**; mechanism different from the guess — it is not a bipartite assignment but a
**min-cost-flow ILP whose cost table makes division a strictly dominated action.**

**Host-verified smoking gun:**

| | value | source |
|---|---|---|
| our deployed override | **`BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"`** | `notebooks/kaggle_p0b_clean913_revtime/*.ipynb` (P3 derives from P0-B) |
| **vendor default** | **0.1** | `vendor/.../predict_unet_transformer.py:78`, argparse `:629` |
| ILP active | `--use-ilp` | live fold-0 run log |
| live weights | `--ilp-edge-weight -1.0 --ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5 --ilp-division-weight 1.0` | `p3_d1_pilot_f0_v1/*.log:163` |

Under the minimised flow objective, turning one link into a division changes the cost by
`division_weight − appearance_weight − p = 1.0 − 0.0 − p ≥ 0` **for every p ≤ 1**. Division never
pays. Making track-start FREE means the solver always prefers "daughter appears from nothing" over
"mother divides".

**Proved locally on CPU** (`tracksdata`/`ilpy` are in `.venv`), toy dividing lineage: deployed
weights emit **0 divisions even at p = 0.99**; vendor `appearance_weight = 0.1` emits the division
at p ≥ 0.90 and not at 0.88 — break-even `division_weight < appearance_weight + p` confirmed to
the boundary.

**Three layers, fixing any one alone changes nothing:**
1. a hard-coded `threshold = 0.5` on a **column**-softmax (`predict_unet_transformer.py:73,455-468`)
   — no CLI flag exists for it. In-degree ≤ 1 is then a *theorem*, so `max_parents_per_node` is
   dead code.
2. the ILP economics above.
3. `motion_relink_edges` (`wrapper.py:309-343`) **replaces the ILP's edges wholesale**
   (`:1158-1161`) with a strict `linear_sum_assignment` — active on 100% of crops,
   `motion_relink_fallback_raw = 0`.

The greedy caps are NOT the constraint: under `use_ilp=True` both caps are `None` (`:85-93`) and
`max_children_per_node = 2` is the vendor default anyway.

**Measured:** 1,976,653 pre-wrapper edges (fold 0 9 crops + fold 1 **all 128**) — out-degree
`{1: 1976653}`, in-degree `{1: 1976653}`; min `edge_prob` 0.5000001 → a hard threshold, not a
softmax artefact. (Independently corroborates the host's early pregraph check and the
edge-loss agent's 2,044,806-edge histogram.)

**The prize (fold 1, 125 GT divisions, pre-wrapper graph):** assignment-only ceiling
**61–65 of 125 (49–52%)** with NO new detection. Decomposed: **31 orphan** second daughters —
exactly what an ILP fix targets — and **30 stolen** by another parent, where the thief's prob
(median 0.748) caps the mother's at `1 − p_thief < 0.5`, so those need the **threshold** lowered,
not the ILP. This refines the host's own funnel (61 of 96 lost to the orphan constraint) into two
separately actionable halves.

**Negative result that partially refutes a sibling.** A full 199-crop CPU replay of the safe-division
proposer at widened gates: every setting raises TP and FP by the **same factor**, precision pinned
near **0.03%**, best Δscore **≤ +0.0004** — inside the ±0.003 noise floor. So
`division_lane_2026-08-18.md`'s D1 ("relax the gates") is dead: **proposability was never the
binding constraint, precision is.** Corroborates the host's own funnel conclusion (relaxation
drowns the signal at ~2,400:1). The probability signal that COULD separate them is discarded by
the ILP and never reaches the proposer (`"edge_prob": None`, `wrapper.py:854`).

**Cheapest next step (falsification-first):** export the **pre-ILP candidate graph** — it already
exists in memory at `predict_unet_transformer.py:554`, one line before the solver. ~+2% kernel
wall-clock, no extra GPU. **Kill criterion:** if fewer than ~15 of the 31 fold-1 orphan cases had
a declined candidate, the ILP lane is dead.

**Two corrections carried:**
- `artifacts/kaggle/p0strict_cache/graphs` is **NOT** usable for this replay — post-wrapper, no
  `edge_prob`, and its degree histogram `{1: 3576550, 2: 10724, 3: 2}` reveals an older, more
  permissive wrapper config than the deployed one.
- The DEPLOYED safe-division constants differ from the `wrapper.py` defaults the host's funnel
  used: notebook sets `SAFE_DIV_MAX_UM=4.66`, `SISTER=8.5`, `EXISTING_CHILD=7.65` (vs 4.7 / 7.2 /
  7.8). The funnel's conclusion is unaffected (the parent gate binds either way) but the exact
  stage-(h) count is for the 7.2 configuration.

---

## 2026-08-18 — LAUNCHED: pre-ILP candidate export (falsification of linker lane L1)

**Host-authorised GPU launch.** Kernel `aryaarun07/biohub-p4-preilp-loeo-f1`, version 1, pushed
and QUEUED. Spec `scripts/kaggle_specs/p4_preilp_loeo_f1.json` (20 edits, derived from
`p3_base_loeo_f1`).

**Question.** The ILP never OUTPUTS a division (post-ILP out-degree `{1: N}` over ~2M edges;
`BIOHUB_ILP_APPEARANCE_WEIGHT=0.0` makes division strictly dominated). But was it ever OFFERED
one? Two worlds: (a) the candidate graph contains second-daughter edges above the hard 0.5
threshold and the solver declined them on cost → lane L1 recovers real divisions; (b) it never
contained them → L1 is dead and the lane collapses to the threshold or the training fix.

**Design — a pure observation, verified.** Two additive edits:
`scripts/kaggle_edits/pre_ilp_export.py` dumps the candidate graph one line before
`solver.solve` (`predict_unet_transformer.py:554`), and `pre_ilp_rollup.py` rolls the geffs into
one parquet so the measurement survives kernel cleanup (the failure mode that lost the fold-1
geffs once before). **Verified the built notebook against `p3_base_loeo_f1`: cells 5 and 6 differ
by +48 and +98 lines with ZERO deletions** — nothing in the pipeline changed, so the emitted
submission graph is bit-identical to the base arm.

**Trap avoided:** the notebook globs `predictions/*/{METHOD}/split_0/*.geff` and hard-fails unless
the count equals `len(test_stems)`. Writing `*.preilp.geff` beside the real outputs would have
doubled that count and killed the run at the assertion; the export goes to
`/kaggle/working/preilp` instead, outside the glob.

**Fold 1 chosen** (128 crops, 125 GT divisions) because the 31 orphan cases and the stated kill
criterion live there. Note fold 1 is the LEAKY arm for scoring — irrelevant here, since this is a
structural count, not a score.

**KILL CRITERION (stated before the result):** of the 31 fold-1 divisions whose second daughter
was detected but left orphan, **fewer than 15 having a declined candidate edge ⇒ lane L1 DEAD.**

**Audit instrument written and ready:** `scripts/win_bet/preilp_div_audit.py` consumes the parquet,
matches >=2-candidate sources to GT dividing mothers within 7 µm, and reports the count against the
bar. Deliberately a COUNT, not a score — no score delta at this scale would be interpretable given
the unvalidated instrument and ±0.003–±0.004 fold noise floors.

---

## 2026-08-18 — PRE-ILP EXPORT v1: the solver is offered 176,835 divisions and outputs ZERO

Kernel `biohub-p4-preilp-loeo-f1` v1 **COMPLETE**. The export and roll-up both ran correctly:

```
pre-ILP candidate-graph export patch applied (-> /kaggle/working/preilp)
pre-ILP roll-up: found 128 candidate graphs in /kaggle/working/preilp
pre-ILP roll-up: wrote /kaggle/working/preilp_split1.parquet rows=4,685,519 sha256=b94bb1fb...
pre-ILP roll-up: TOTAL sources with candidate out-degree >= 2 = 176,835
```

**Measured over all 128 fold-1 crops** (per-crop rows recovered from the kernel log; log archived
at `_evidence/kaggle_runs/p4_preilp_loeo_f1_v1/`):

| quantity | value |
|---|---|
| candidate nodes | 2,523,479 |
| candidate edges | 2,162,040 (0.857 per node) |
| **sources with candidate out-degree ≥ 2** | **176,835** (7.0% of nodes) |
| per-crop: median / min / max | 922 / 247 / 7,884 |
| **post-ILP sources with out-degree ≥ 2** | **0** |

**This is the ILP-economics finding confirmed end to end on real data at full fold scale.** The
solver is handed 176,835 opportunities to emit a division and takes **none of them** — exactly as
the cost algebra predicts when `appearance_weight = 0.0` makes
`division_weight − appearance_weight − p = 1.0 − p ≥ 0` for every `p ≤ 1`. Divisions are not
absent from the candidate graph; they are **priced out of the solution**.

**What this does NOT yet settle.** 176,835 is the raw supply and is close to the loose upper bound
(≤179,037 fold-1 orphan targets), so it does not discriminate: it says nothing about whether the
*true* second daughters are among those candidates. The stated kill criterion — how many of the 31
fold-1 GT orphan cases had BOTH true daughters offered — needs the parquet's node/edge rows matched
against GT, which `scripts/win_bet/preilp_div_audit.py` does.

**HOST ERROR, and the fix.** The parquet was written but **deleted before output**:
`loeo_export.py:116-122` keeps an allowlist
(`_LOEO_KEEP = {gz, out, "loeo_manifest.json", "run_stats.csv"}`) and `rmtree/unlink`s everything
else in `/kaggle/working`. I placed the roll-up correctly (before the export cell, so it could read
the geffs) but did not add its output to that allowlist. The kernel log confirms:
`kept: ['loeo_manifest.json', 'loeo_split1_strict.csv.gz', 'loeo_split1_strict.json', 'run_stats.csv']`.
**Fixed in v2** by a surgical `replace` edit appending `f"preilp_split{LOEO_FOLD}.parquet"` to
`_LOEO_KEEP` — the shared `loeo_export.py` is left untouched so other specs are unaffected.
Verified in the built notebook, pushed as **version 2**.

**Lesson for the machine:** the roll-up printed its headline per crop to stdout, which is the only
reason v1 is not a total loss — 128 crops of counts survived in the log after the artifact was
deleted. Keep that habit: **print the headline number, never only write it.**

---

## 2026-08-18 — **L1 IS LIVE.** The ILP declines real divisions it is 92% confident in

Kernel `biohub-p4-preilp-loeo-f1` **v2 COMPLETE**; parquet fetched (31,137,093 bytes, sha256
`b94bb1fbf609e1ae...` — **matches v1's logged hash**, so the run is deterministic).
Audited with `scripts/win_bet/preilp_div_audit.py` against `data/train` GT.

```
pre-ILP candidate graph: 2,523,479 nodes, 2,162,040 candidate edges, 128 crops
  edge_prob: min 0.500000  median 0.8924  max 1.0000
  sources with candidate out-degree >= 2 : 176,835
  GT division events in these crops      : 125
  ... whose mother matched a >=2 source  : 34
  ... with BOTH true daughters offered   : 29     <-- the number
  declined candidate edge_prob: median 0.9188  min 0.5506
```

**VERDICT: 29 ≥ 15 → L1 LIVE**, at nearly double the pre-stated kill bar.

**The decisive detail is the confidence.** The declined second-daughter edges carry a **median
`edge_prob` of 0.9188** (min 0.5506). The model is ~92% sure of these links. They are not marginal
candidates the solver sensibly rejected — they are high-confidence divisions **priced out by our own
`BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"`**, under which converting a link to a division costs
`1.0 − p ≥ 0` for every `p ≤ 1`. World (a) confirmed; world (b) refuted.

### Score arithmetic (fold 1; current divJ 0.0047 from TP=3/FP=507/FN=122)

| scenario | divJ | contribution | delta |
|---|---|---|---|
| current | 0.0047 | +0.00047 | — |
| L1 alone, safe-division patch still ON (FP≈507) | 0.0506 | +0.00506 | **+0.0046** |
| **L1 + safe-division patch OFF** | **0.2320** | **+0.02320** | **+0.0227** |
| L1 at the 61-case ceiling, patch OFF | 0.4880 | +0.04880 | +0.0483 |

**This REVERSES the earlier "keep `OUTPUT_SAFE_DIVISIONS = 1`" conclusion — conditionally.** That
call was correct *given* that every output fork (and therefore every division TP) came from the
patch. Once the ILP supplies genuine divisions at ~0.92 confidence, the patch's 507 FPs become
pure denominator noise and turning it OFF is worth **~5× more than L1 alone**. The two changes are
**coupled**: L1 without disabling the patch buys +0.0046; together, +0.0227.

### What must still be true

- **L3 remains mandatory.** `motion_relink_edges` replaces the ILP's edges wholesale
  (`wrapper.py:1158-1161`) with a strict Hungarian matching, so an ILP division dies there
  regardless of L1. L1 + patch-off + L3 is the minimum viable set.
- Only **34 of 125** GT mothers matched a ≥2-candidate source, so 29 is a floor, not the ceiling;
  the linker-capacity analysis puts the assignment-only ceiling at 61–65.
- Fold 1 is the LEAKY arm for scoring — irrelevant here (this is a structural count), but the
  eventual score delta must be measured on the LB, not on this fold
  (see `instrument_repair_2026-08-18.md`).
- These are candidate-graph confidences, not calibrated probabilities: the loss is a bare column
  softmax and is provably shift-invariant, so `edge_prob` is a **share**
  (`edge_loss_structure_2026-08-18.md`). 0.9188 means "best available parent by a wide margin",
  not "92% likely to be real".

**This is the first live lane found since the plateau, and its trigger is a one-line env change.**

---

## 2026-08-19 — EDA the host should have run in week one. Four findings that reframe the metric.

Prompted by the host question "have you looked at the data properly?" — the honest answer was no:
every prior measurement was targeted at a specific hypothesis; no exploratory pass had been done.
This is that pass, over all 199 `data/train/*.geff`.

### 1. HALF THE CORPUS HAS NO DIVISIONS AT ALL

**112 of 199 crops contain ZERO GT division events**, and those crops carry **48.7% of all GT edge
mass**. Among the 87 crops that do have divisions, the median is **1** and the max is **5**.
Division counts: 26 in 44b6 (71 crops), 125 in 6bba (128 crops).

**Consequence:** the `0.1 x division_jaccard` term is structurally unavailable on half the
objective. Any division lever is competing for points that do not exist in most crops.

### 2. ANNOTATION DENSITY IS WILDLY UNEVEN — and the metric weights by it

`adj_edge_jaccard = sum(TP_i * m_i) / sum(w_i)` with `w_i` the crop's GT edge count, so crops are
weighted by annotation, not by size or difficulty.

| family | crops | GT nodes/crop min / p25 / median / p75 / max | GT edges total |
|---|---|---|---|
| 44b6 | 71 | 50 / 124 / **214** / 377 / 1,353 | 19,826 |
| 6bba | 128 | 209 / 655 / **826** / 1,079 / 1,950 | 109,057 |

**The top 50 crops carry 46.3% of ALL GT edge mass; the bottom 100 carry 24.4%.** Roughly a
quarter of the corpus determines nothing.

### 3. THE FOUR PLACEHOLDER CROPS ARE DEEPLY UNREPRESENTATIVE

| crop | GT nodes | percentile | GT edges | GT divisions |
|---|---|---|---|---|
| 44b6_0113de3b | 52 | **2nd** | 50 | **0** |
| 44b6_0b24845f | 51 | **1st** | 49 | **0** |
| 6bba_05b6850b | 861 | 68th | 845 | **0** |
| 6bba_05db0fb1 | 1,229 | 89th | 1,183 | **3** |

Two of the four sit in the **1st-2nd percentile of annotation density** in the entire dataset, and
the four together contain **3 GT divisions**. Any division experiment evaluated here has a
denominator of three.

### 4. DIVISION SYNCHRONY IS NOT A USABLE PRIOR (pre-empts a hypothesis in flight)

KS test of division timepoints against uniform over t in [0,100]:
- **44b6: D=0.203, p=0.204** — consistent with uniform.
- **6bba: D=0.130, p=0.027** — weakly clustered, and the signal is a thin final decile (5 vs 9-18),
  consistent with an edge effect (a division needs a following frame) rather than biological
  synchrony.

So the "zebrafish early cleavages are synchronous" prior does **not** show up in this data at this
stage. Recorded here so the biology agent's ranking can be corrected against measurement.

### Track structure (context)

4,435 GT tracks; length median **21** frames, mean 29.4, max 100; **zero** length-1 tracks;
median crop has all 100 timepoints annotated.

---

## 2026-08-19 — THE DIVISION FIX WORKS LOCALLY (+0.0071) AND SCORED 0.000 ON THE LB

Both submitted artifacts were scored locally with the official scorer against the four
placeholder crops' GT (they are byte-identical to train crops, so GT is available):

| artifact | adj_edge_jaccard | division_jaccard | SCORE |
|---|---|---|---|
| `p3_harmonic/submission.csv` (LB 0.915) | — | — | **0.8907** |
| `p5_divfix/submission.csv` (LB 0.915) | 0.8911 | **0.0667** (TP=1 FP=12 FN=2) | **0.8978** |

**Local delta +0.0071. Leaderboard delta +0.000.**

The fix behaved exactly as designed: it converted **1 of the 3** available GT divisions, at a cost
of only 12 charged FPs (the 703 emitted divisions are damped by the ~1.6% annotation density of
that crop). On this substrate it is a real, measurable gain.

**This is now the THIRD instance of local-positive to LB-zero:**
1. arm-B motion gate: paired LOEO **+0.0144 / +0.0090** -> LB **+0.000**
2. P3+divfix: placeholder **+0.0071** -> LB **+0.000**
3. (P3 harmonic: predicted modal +0.001..+0.006 -> LB +0.001)

**Caveat, stated plainly:** the placeholder crops are memorisation-contaminated (`seed314159`
trained on all 199 crops including these four), so the ABSOLUTE 0.89 is inflated. But the two
artifacts were scored on the identical substrate with the identical scorer, so the **delta** is
the honest quantity, and it is +0.0071 against an LB movement of zero.

**This sharpens the control probe from useful to decisive.** A +0.0071 local gain should print as
0.922 if the chain is sound. It printed 0.915. Either the hidden test differs enough from these
four crops to erase a 0.7-point gain, or the measurement chain is not doing what we think.
The degraded control (`det_threshold` 0.999) settles it.

---

## 2026-08-19 — A CLEAN LOCAL INSTRUMENT EXISTS, and it is a drop-in. HOST-VERIFIED.

`leevvin/biohub-movie-heldout-edge-predictor-v1` (7.3 MB, CC0), downloaded to
`data/external/public_weights/leevvin/`. **Host-verified by loading it:**

```
checkpoint tensors: 136
model.load_state_dict(sd, strict=True)  ->  OK
```

136/136 tensors into our exact `UNetNodeTransformer(unet=TemporalUNet3D(32,[32,64,128]), 32, 4*_POS_EMBED_DIM)`.
Contrast `xiaoleilian/biohub-unet3d-weights-v2models`, which was measured at 0 intersecting keys.

**Its README independently documents the contamination we discovered on 2026-08-18** — that the
public support-pack checkpoint used `train=199` with a `test` list that is a **subset** of those
199, so "anyone validating locally on movies from `train/` is scoring a model that memorised
them, which makes offline post-processing sweeps point the wrong way." Two independent parties,
same defect, same week.

`splits_movie_heldout.json`: `train` = 195 movies; `test` = exactly our four placeholder crops
(`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`).

**Why this matters more than any lever.** We hold GT for those four crops. We now hold an edge
model that has never seen them. **That is an uncontaminated local instrument** — the thing
`instrument_repair_2026-08-18.md` concluded we did not possess and could not build. It does not
fix the primary detector's provenance (still undocumented) but it removes the largest known
contamination from the association stage.

**Caveat (stated by the source, endorsed):** the held-out set is the four public twins, not an
embryo-disjoint split, so it is not a substitute for LOEO across embryos — it is a clean
substrate for the four crops we can actually score.

**Standing correction to the ledger:** `instrument_repair_2026-08-18.md` concluded "we hold no
validated instrument" and recommended the LB as the only trustworthy one. That conclusion was
correct **for the artifacts we had**. It should now be re-tested with this checkpoint before the
LB is treated as the sole instrument.

---

## 2026-08-19 — BIOLOGY: two claims HOST-VERIFIED, one of them indicts arm B directly

`biology_priors_2026-08-19.md`. The two load-bearing, actionable claims were re-measured
independently by the host over all 199 crops.

### VERIFIED 1 — a free hard constraint: daughters never re-divide

| quantity | value |
|---|---|
| GT daughters | **302** |
| daughter-frames at risk | **5,648** |
| daughters whose lineage divides again | **0** |

Zero, exactly as reported. At the measured base rate the Poisson expectation is ~6.4
(p ≈ 0.0017), and post-MBT cycle floors of 78–240 min are documented biology. **A refractory
constraint — a cell that just divided cannot divide again for ~K frames — is free, safe, and
currently absent from the pipeline.** It cannot add TPs, but it deletes FPs in the one place we
over-fire by ~10×, and deletion is the action class we have never shipped.

### VERIFIED 2 — the motion field is a BULK TRANSLATION, and arm B modelled it locally

Displacement variance over all GT edges (physical µm, 199 crops):

| model | residual variance | variance explained |
|---|---|---|
| raw displacement (median \|d\| = 1.82 µm) | 7.119 µm² | — |
| − **global translation** per (crop, frame) | **2.706** | **62.0%** |
| − kNN-16 median local flow | 2.959 | 58.4% |

**A single global translation beats kNN-16 local flow**, and is simpler, cheaper, and has no
neighbourhood-size parameter. **Arm B — our promoted-then-killed motion gate — used exactly the
kNN-16 median flow that loses this comparison.** Its LOEO gain was real; its motion model was the
weaker of the two available. The agent additionally reports the global offset is recoverable
**without GT** by a correspondence-free mean-shift vote that survives 50% detection deletion plus
100% decoys at cos +0.90 with truth.

**Consequence:** before any retrain, the linker's gate quantity should be
`|target − (source + global_offset(crop, frame))|`, not raw displacement and not kNN flow. This is
a pure post-processing change on a substrate we already hold.

### Corrections to the host's own framing (accepted)

- **6bba does NOT divide more.** Per annotated cell-frame: 6bba 1.11e-3 vs 44b6 1.29e-3. The 5× raw
  division count is **5.6× more annotation**, nothing biological. The host's earlier "6bba holds
  125 of 151 divisions" framing was a counting artifact.
- **6bba is harder because of boundary flux, not crowding.** It is 3.4× *sparser* (8% vs 28% tissue
  occupancy), its nuclei sit at half the distance to a crop face (12.2 vs 22.8 µm), and it has 3×
  the interior track starts/ends — a thin motile sheet vs interior tissue. Predicted error budget:
  **linking and birth/death, not detection** — which matches the measured 59% FP / 41% missing.
- **Division rate is 1.13e-3 per cell-frame, identical in both embryos** → a per-crop expected
  division count the pipeline has never used as a calibration target. Against 613 measured FP, we
  over-fire by roughly an order of magnitude.

### Four priors tested and KILLED (recorded so they are not re-proposed)

Global division synchrony (host independently measured this too: 44b6 KS p=0.204, 6bba p=0.027
driven by a thin final decile — an edge effect); mitotic domains; pre-division kinematic slowdown;
and the kNN neighbourhood-flow prior (superseded by global translation, above).

### Stage / frame interval, and the caveat that carries it

Late gastrula through segmentation (~8–20 hpf); **frame interval 90 s** from three independent
estimates, the strongest being Ultrack's Methods documenting a 1.625 µm z-step / 90 s
shield-to-24 hpf acquisition that matches our deployed geometry exactly (the DaXi ZSNS lineage is
1.24 µm / 60 s and does NOT match). **Flagged by the agent as the most load-bearing inference in
its report** — everything time-based converts through it. The appearance signature of mitosis is
honestly weak (~3× enrichment at 10% FPR); at 90 s/frame prophase and metaphase are sub-frame, so
it belongs in a candidate generator, not as a standalone lever.

---

## 2026-08-19 — The 90 s frame interval CROSS-VALIDATES against published cell speeds

`biology_priors_2026-08-19.md` flagged the **90 s frame interval as its most load-bearing
inference** — every time-based quantity in that report converts through it. A literature sub-search
(`zebrafish gastrula cell speeds`, primary sources read directly) supplies an independent test.

Our measured **median GT displacement = 1.82 µm/frame**. Implied cell speed by candidate interval:

| interval | implied speed | verdict |
|---|---|---|
| 30 s | 3.64 µm/min | **above the published single-cell range (0–3 µm/min)** — implausible |
| 60 s | 1.82 µm/min | plausible |
| **90 s** | **1.21 µm/min** | **matches lateral mesoderm 1.17 µm/min almost exactly** |
| 120 s | 0.91 µm/min | plausible |
| 300 s | 0.36 µm/min | slow, below all tissue anchors |

Published anchors, read from primary sources: prechordal plate **2.89 ± 0.79 µm/min**
(Dumortier 2012, PNAS); lateral mesoderm total **1.17 µm/min** (von der Hardt 2007, quoted
verbatim in Lin 2010, JCB); blastoderm pre-shield **~1.0 µm/min** (Kobitski 2015, Sci Rep); dorsal
PSM **0.75 µm/min** (Yin 2008, JCB); whole-gastrula single-cell dynamic range **0–3 µm/min**
(Faure 2016, Nat Commun).

**Verdict: the 90 s interval is CORROBORATED but not proven** — it lands our displacement squarely
on the lateral-mesoderm anchor, and 30 s is excluded outright. 60–120 s all remain admissible, so
any downstream claim should be robust across that band rather than assuming 90 s exactly. The
independent structural argument still stands: Ultrack's Methods documents a 1.625 µm z-step / 90 s
acquisition matching our deployed geometry, while the DaXi ZSNS lineage (1.24 µm / 60 s) does not.

**Why this matters beyond bookkeeping:** it confirms our nuclei move at ordinary gastrula speeds,
so the tracking problem is not unusually hard kinematically — consistent with the finding that a
single global translation explains 62% of displacement variance. It also independently supports
the "late gastrula through segmentation" staging, and therefore the post-MBT cycle floor
(78–240 min) that licenses the **refractory division constraint** (0 of 302 daughters re-divide,
host-verified).

**Literature access caveat, recorded honestly:** four of the six most relevant primary papers
(Myers 2002; Sepich 2005; Sepich 2000; Behrndt 2012) are **paywalled and were not read** — the
D/V-resolved convergence tables that would refine this are not obtainable on the open web.
Ventral convergence speed has **no published numeric value** at all; the literature states it is
absent rather than measuring it.

---

## 2026-08-19 — THE STRATEGIC PIVOT: abstention is CLOSED; the CANDIDATE GENERATOR is the bottleneck

`error_atlas_2026-08-19.md`, the most decisive report of the campaign. Its harness reproduces the
official SCORE exactly (0.90332 / 0.70423), the loss identity checks to 2.2e-16, and its FN counts
match `edge_loss_structure_2026-08-18` to within one edge — an independent cross-check of both.

### 1. ABSTENTION IS CLOSED BY MEASUREMENT, not left unproven

Best deletion precision on 6bba: **58.3% ± 2.5% (n=400) against a 58.88% bar** → simulated
**ΔSCORE = −0.00001**. The winner is a NEW feature, `rev_margin_um` (source-side geometric margin),
which beats everything previously tried **including `edge_prob`** — and still does not clear.
On 44b6 (bar 53.1%) it reaches 61.4% for **+0.0012**.

**The +0.010…+0.040 abstention lane — the main line since 2026-08-18 — is dead.** It was the
corpus's top-ranked survivor; it is now measured and closed.

**Why nothing clears it:** a 34-term crop-held-out logistic combination achieves the best
discrimination in the whole study (**AUC 0.7752**) and the **worst** useful operating point (49.5%,
below one of its own inputs). **AUC is the wrong selector for this metric** — the bar is a
statement about the extreme tail; AUC integrates the whole ranking. Recorded as a general lesson.

Also confirmed (host had verified this independently): `logitdiff`-style probability margins are
**structurally unavailable** — in-degree is exactly 1 for all 2,162,040 candidate edges, so there
is no runner-up to take a margin against. AUC 0.52.

### 2. THE LOSS IS NOT CONCENTRATED — there is nothing to target

Worst decile (20 crops) carries **31.6% of loss** while already holding **19.5% of edge weight** —
only **1.62× over-represented**. Making all 20 perfect is worth +0.0845. Within crops it is flatter
still: 6bba miss rate sits between **0.19 and 0.24 across every** z-band, time-band and
density-band. All 20 worst crops are 6bba and their distinguishing property is **detection rate
(median 0.785 vs 0.955)** — not anything the linker does.

### 3. THE CANDIDATE GENERATOR IS THE BOTTLENECK — HOST-VERIFIED

Independent host measurement over fold 1 (109,057 GT edges), pre-ILP candidate graph vs GT:

| stage | count | share of GT edges |
|---|---|---|
| GT edges | 109,057 | 100% |
| both endpoints **detected** (≤7 µm) | 98,070 | **89.9%** |
| … **and nominated as a candidate** | 81,003 | **74.3%** |
| **detectable but NEVER NOMINATED** | **17,067** | **15.65%** |

(Agent reported 17,001; host measured 17,067 — agreement within a match-rule rounding.)

**15.65% of all GT edges are thrown away before the solver ever sees them.** In recall space, a
perfect solver over today's candidates tops out at 0.743 while a perfect candidate generator
reaches 0.899. The agent's complementary score-space figure: a perfect linker over the pre-ILP
candidates is worth only **+0.0012 SCORE** — i.e. **the solver is already near-optimal for the
candidates it is given.** (Note the two are different quantities: recall-space fraction vs
score-space adj_edge_jaccard with the node-count multiplier. They agree in direction.)

### 4. FP AND FN ARE THE SAME EVENTS — relinking dominates deletion

**96.6% of 6bba FPs are substitutions** (neither endpoint already carries a TP) and **80.7% of
reachable FNs are mis-links**. So the same edge is counted twice. Oracle values:
**relink +0.1007 vs delete-all-FP +0.0926**, and relinking's break-even is far softer — a right
relink gains 3 units, a wrong one costs 1, against deletion's knife-edge 59% bar.

**This reframes the metric asymmetry the corpus had been reading backwards.** The blind spot was
never "we only add, never delete" — it is that **wrong edges should be MOVED, not removed.**

### 5. One flagged lead, deliberately not acted on

The output node filter appears to discard **~5,400 GT-matched nodes** on fold 1 (detection
0.912 → 0.869, asymmetry 5,381 vs 548). **Cross-run confounded** — the agent recommends a
same-run A/B before anyone touches it, and notes that adding nodes back erodes the measured
+0.005/+0.018 node-count bonus.

### Where this leaves the campaign

Three lanes are now closed by measurement (division post-processing, detection threshold,
abstention). The surviving direction is **candidate generation and relinking** — supported
independently by the biology finding that a global translation explains 62% of displacement
variance (a better gate quantity for proposing candidates) and by the public 0.917 stack's
loose-propose/strict-filter division design.

**Reproducibility:** five scripts under `scripts/win_bet/` (`ea_atlas`, `ea_analyze`, `ea_features`,
`ea_abstain`, `ea_solver_gap`); no new dependency (logistic fit is numpy Newton-IRLS, since
`sklearn` is absent and the agent declined to mutate the pinned venv — correct call).
**Caveat carried:** the pre-ILP dump is from a different run than the scored export (§6/§6.1 only);
§§2–5 and §7 are confound-free.

---

## 2026-08-22 — CORRECTION: the ILP weights are INHERITED, not self-inflicted. Host-verified.

`audit_vendordrift_2026-08-22.md` refutes the framing the host used on 2026-08-18. **Verified
directly:**

| setting | vendor default | ours (0.915) | **kimi-v17 (0.923)** |
|---|---|---|---|
| `ilp_appearance_weight` | 0.1 | **0.0** | **0.0** |
| `ilp_disappearance_weight` | 0.1 | **1.5** | **1.5** |

(vendor: `predict_unet_transformer.py:78-79`; kimi values read from the pulled notebook.)

**The 2026-08-18 record framed `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"` as "our own ILP setting"
and "our deployed override". That is WRONG.** It is verbatim in the public lineage — kimi-v17,
pilkwang, and independently in `kaiwalyaatulraut`'s separate lineage. The division-dominance proof
still holds mechanically (`delta = division_weight - appearance_weight - p`, re-derived from
`tracksdata/_ilp_solver.py:296-319`), but its consequences are:

1. **It is not self-inflicted damage** — it is a lineage-wide inherited value.
2. **It does not explain our gap to 0.923** — the 0.923 notebook carries the identical value.
3. It remains a real mechanism that suppresses divisions **for the entire public field**.

**The audit's larger correction:** our deployed pipeline is **not a divergent fork of the public
baseline — it IS the public baseline, one generation stale.** The env block, the 8-view TTA source
patch, the dual-seed secondary block and the whole `filter_output_graph` chain are byte-shared with
public notebooks; even the harmonic edit that earned our +0.001 is character-identical to the
public one (same `Biohub 145:` comment), because both were transcribed from the same CC0 source.
**Our position is a LAG, not a wound.**

### The genuinely unjustified item, and why it is interesting

**`ilp_disappearance_weight = 1.5` against a vendor default of 0.1** — a 15× divergence never named
once in `experimental-records.md`, `bets.yaml`, `submissions.md` or `failed-experiments.md`.

**Mechanism:** 1.5 exceeds `edge_prob`'s entire dynamic range (max 1.0), so the keep/drop decision
is made almost entirely by the flow term and `p` survives only as a tie-break. **The ILP degenerates
toward maximum-cardinality bipartite matching.**

**This retroactively explains two findings the corpus could not account for:** that the geometric
relink can replace every ILP edge at no loss, and that deleting the learned probability from the
relink costs **1 GT edge in 9,020**. If the solver is effectively ignoring `p`, the learned edge
head *should* look useless — and it does.

**But note it is ALSO lineage-wide** (kimi-v17 has 1.5 too, host-verified above), so it is not a
repair that closes our gap — it is an **unexplored lever for the whole field**, in the
candidate-set class (3.5× transfer). That makes it a candidate EDGE rather than a fix, which is a
more valuable thing to hold.

### Two further recommendations from the audit (endorsed, not yet acted on)

- **8-view D4 TTA** is unmeasured on our side — but `pilkwang` uses 4-view and `kimi` uses 8-view on
  otherwise-identical notebooks, so **the public leaderboard already prices this knob for free.**
- **`kaiwalyaatulraut`**, a high-vote public notebook on the same vendor architecture and the same
  pilkwang weights, has **zero post-ILP track editing**. Our nine-stage `wrapper.py` stack has never
  been measured in aggregate against not having it, and `loeo_pregraph_export.py` already exports
  the pre-wrapper graph that comparison needs — **no submission slot required.**

### On LOEO, a fair correction to the corpus

Nothing in the deployed 0.915 artifact rests on a LOEO number alone. LOEO acted purely as a
**rejection** instrument (sub-voxel, grid-shift, component selector all correctly killed), and the
one lever it accepted — arm B — was flattened to +0.000 and is correctly not carried. The
instrument was misused as a promoter, not miscalibrated as a rejector.

---

## 2026-08-22 — SELF-HARM AUDIT: three findings that correct the host's own work

### 1. `p8_loosefilter` is an INCOMPLETE port — host error, found before the score landed

Host-verified by knob-name enumeration of the pulled 0.923 notebook:

| knob | kimi-v17 (0.923) | ours |
|---|---|---|
| `SAFE_DIV_REQUIRE_DIVERGENCE` | **present** | **absent** |
| `SAFE_DIV_DIVERGE_UM` | **present** | **absent** |
| `SAFE_DIV_REQUIRE_MUTUAL_NN` | **present** | **absent** |

Concept counts: kimi-v17 has *diverge* 15, *mutual* 18, *sister* 22. `src/biotrack/wrapper.py` has
**diverge 0, mutual 0, orphan 0**.

**p8 copied their loose gates (12/15/10) WITHOUT their three precision filters.** The host's diff
compared the VALUES of knobs both notebooks share and never checked for knobs present in theirs
and absent in ours. Against our own measurement that 12.0/15.0 opens ~103,008 false candidates per
31 true ones, **p8 ships a strictly weaker design than the one that scored 0.923** and should be
expected to be flat or negative. (A first host grep appeared to refute this; it used an
assignment-pattern regex and missed the knobs. The knob-name sweep above is authoritative.)

**What the veto actually did** (`c:/temp/p8/run_stats.csv`): checked **14,188**, accepted 8,288,
**rejected 5,900 (41.6%)** — so it is NOT a no-op; it discriminated. But `DEEPCENTER_SAFE_DIV_VETO`
takes the max of the DeepCenter heat-map in a ±(1,2,2) window — it asks *"is there a real cell
here?"*, not *"are these two cells sisters?"*. At 2,400:1 most false candidates **are** real cells.
**A cell-ness test was substituted for a sisterhood test.** Also: `DEEPCENTER_SAFE_DIV_THRESHOLD`
is never set anywhere and runs at its **0.12** default, while the *gap* veto's threshold was
deliberately tuned 0.10 → **0.25**. Same model, same accept function, two call sites — one tuned,
one untouched.

### 2. The division funnel that closed `bet-division-proposal` was STRUCTURALLY BLIND

`scripts/kaggle_edits/loeo_retarget.py:131-141`, verified verbatim: in `LOEO_ARM == "strict"` it
sets `DEEPCENTER_SAFE_DIV_VETO = False` (along with the three other vetoes and the secondary
model). **`div_proposal_funnel.py` was run on `loeo_split1_strict.csv.gz`** — so the instrument
that concluded "relaxation drowns at 2,400:1 and needs a discriminator" was measuring a pipeline
in which **the discriminator could not fire**. The conclusion may still hold, but it was never
tested against the mechanism it named.

### 3. The ILP is structurally irrelevant — which explains three dead lanes at once

`motion_relink_edges` discards **99.87%** of the solver's output (measured 164,470 of 164,677) and
rebuilds edges from `nodes_by_id` alone via `linear_sum_assignment` — a 1:1 matching, so every ILP
division is destroyed. Measured: `division_like_sources` **=** `safe_divisions_added` **exactly**
(557 = 557; the host independently measured 703 = 703 on divfix). **100% of output divisions come
from the post-processor; zero from the model or the ILP.**

This makes `ILP_APPEARANCE_WEIGHT`, `ILP_DISAPPEARANCE_WEIGHT`, `ILP_DIVISION_WEIGHT` and
`USE_ILP` **structurally inert at deployment**, and explains in one line why three separate
division lanes all measured 0.000 on the LB.

### Two free, transferring-class levers surfaced

- **`ADAPTIVE_SHORT_TRACK_RESCUE = 0`** while the short-track filter deletes **11,028 nodes (5.9%
  of raw)** per fold-1 run — 61× more than isolated pruning. The rescue built to give some back is
  disabled, **and its trigger fires at `removed_frac >= 0.10` when the measured rate is 0.029–0.085
  on all ten crops — it could not fire even if enabled**, and is budget-capped at ~1.6% of what was
  removed. Detection-surface class.
- **`GAP_CLOSE_MAX_GAP = "2"` is silently clamped** by `wrapper.py:482` `min(GAP_CLOSE_MAX_GAP, 1)`
  in every deployed kernel. The 2-frame bridge never runs; the threshold evaluates at 11.6 µm
  instead of 17.4 µm. Gap-close inserts synthetic nodes → candidate-set class.

### The meta-answer (coordinator, endorsed)

**103 knobs, 55 levers, 13 bets, 33 reports, a claims table with drift detection — and the finding
that mattered was six env-var comparisons against a public notebook, by hand, in an afternoon.
The machine produces new measurements and has no routine that re-reads its own configuration.**

Four process failures, all of **inventory**, none of analysis: an instrument that disables the
mechanism under test without declaring it (F1 above); a bet closed as "needs a discriminator" while
the discriminator sat in our own wrapper switched off; an inherited default recorded as
"already shipped" (i.e. as our win) six days before it was measured to emit 0 divisions from
176,835 opportunities; and `CLAUDE.md` naming `src/biotrack/` as the deployed wrapper when
`grep -i deepcenter src/biotrack/` returns nothing.

**Also corrected:** three of the four headline "self-inflicted" defects are **inherited** — the
safe-div constants and the veto default sit in the public 0.914 base preset, and the trainer no-op
is the vendor's. `p3_harmonic.json` and `p7_cleanedge.json` carry **zero env edits**. Across
fourteen submissions we changed the mechanism repeatedly and the configuration essentially never.

---

## 2026-08-22 — CORRECTION: the "free TTA natural experiment" does not exist

The 2026-08-22 vendor-drift entry above reported that `pilkwang` uses 4-view TTA while kimi-v17
uses 8-view on otherwise-identical notebooks, and concluded that **"the public leaderboard already
prices this knob for free."** The host relayed that to the operator as fact. **It is false.**

Host-verified by pulling both notebooks and counting directly:

| notebook | `rot90` | `flip` | `tta` |
|---|---|---|---|
| `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | 8 | 26 | 14 |
| `yunusgmsoy/kimi-notebook-v17` (0.923) | 8 | 26 | 14 |
| ours (0.915) | 8 | 26 | 14 |

**Identical. Extended TTA is unanimous across the lineage; there is no natural experiment and no
free reading.** Measuring the TTA knob would require a real kernel run. The agent self-reported
this error before the host asked, having initially written the claim from an unverified subagent
result while that subagent was still running — the same premise-vs-arithmetic failure mode logged
twice before in this ledger.

**Two things this strengthens rather than weakens:**
1. **"Our pipeline IS the public baseline, one generation stale"** — now supported by
   byte-level TTA identity across three notebooks, not just shared config values.
2. **A third independent lineage converges on the same constants.** `kaiwalyaatulraut` is a genuine
   independent implementation (own model wrapper, plain constants instead of `BIOHUB_*` env vars,
   two-tier edge thresholding, **no post-solver stack at all**) and still lands on appearance
   **0.0**, disappearance **1.4**, detection **0.9500**, broad TTA. Two independent lineages
   arriving at these values is weak but real evidence that somebody tuned them.

**Consequence for the UNJUSTIFIED list:** the reverts stay on it — we still hold no measurement of
our own — but their expected value drops. **They are worth running for the knowledge, not as free
points.** `ilp_disappearance_weight = 1.5` (15× vendor, never named in any record) remains the
largest wholly-unexamined divergence, now with the caveat that a third lineage independently sits
at 1.4.

---

## 2026-08-22 — CONSTANT AUDIT: 26 of 40 constants have no record; one new gate at the 0.8th percentile

Full report: [internal-reports/audit_constants_2026-08-22.md](internal-reports/audit_constants_2026-08-22.md).
Every numeric constant in the deployed P3-harmonic config, measured against the distribution of
the quantity it gates. GT (199 crops / 151 divisions), both LOEO strict exports, and the fold-1
pre-ILP parquet. Counts and distances only — no GPU, no submission.

**The new wound.** `GAP_DENSITY_REFERENCE_UM = 6.5` sits at the **0.8th percentile** of the
detector's local neighbour spacing (median **10.36 µm**, n = 3.8 M). No record of it exists
anywhere. The mechanism is therefore degenerate, not merely mis-set: **49.1%** of nodes are pinned
at the `+0.125` clip, only **0.8%** ever contract, mean applied delta **+0.101 µm/step** = +1.75%
of the 11.6 µm gate. `GAP_DENSITY_ADAPTIVE=1` is a near-constant widening, not an adaptive gate.
Headroom is honest-small: only **109** residual unclosed gap opportunities across 199 crops.

**Six constants are structurally inert.** `GAP_CLOSE_MAX_GAP=2` is hard-clamped to 1
(`min(GAP_CLOSE_MAX_GAP, 1)`); `DUAL_SEED_EDGE_THRESHOLD=0.48` sits **below** the softmax>0.5
candidate floor — measured: no exported candidate edge has prob < 0.500, so any value ≤ 0.5 is
identical. Plus `DIV_*_MAX_UM`, `GAP2_*`, `DEEPCENTER_SAFE_DIV_THRESHOLD`, `SHORT_TRACK_RESCUE_*`.
`MOTION_RELINK_MAX_FRAME_NODES=2600` is a latent cliff (max frame observed **872**): if it ever
binds, motion relink returns `[]` for the WHOLE crop, silently.

**The division budget binds and was never chosen.** `SAFE_DIV_GLOBAL_FRAC_CAP=0.00375`: **26 of
199** crops land on **exactly** the cap, **48.2%** at or above it, fold-1 median forks/cap
**1.000**. For half the crops the division count is set by an inherited budget, not by geometry.

**Emitted forks are geometrically disjoint from true divisions** — the mechanical account of
TP=5/FP=613: emitted fork parent distance median **2.47 µm** (p99 5.01) against a GT
far-daughter median of **7.13 µm**; emitted sister median **4.02 µm** against GT **10.57 µm**.
The deployed division triple admits **19 of 151** GT divisions (12.6%); the public 0.923 triple
admits **138 (91.4%)**. Gates are coupled: relaxing parent alone saturates at 44/151, sister alone
at 20/151, and the child gate removes **nothing**.

**The candidate threshold came back CLEAN.** τ sweep over the pre-ILP export: reach is monotone
decreasing above 0.50 (82.58% → 30.35% of detectable at 0.99). The deployed 0.48 is already at the
structural floor. The missing **17.42%** of detectable GT edges are absent because the column
argmax went elsewhere — **structural, not a tunable constant**. Corroborates the 15.65% in
`error_atlas_2026-08-19` and the standing "candidate generator is the frontier" conclusion.

**Negative result, on record:** there is no second `SAFE_DIV_MAX_UM`. Seven of the nine
GT-calibrated distance gates sit **above the 95th percentile** of their own true distributions —
slack, not tight. The below-the-biology pathology is confined to the division stage and the
density reference.

**Ranked action** (distance × transfer class × cheapness): **(1) sweep `OUTPUT_MIN_TRACK_LEN`
{4,5,6,7,8}** — the only constant that is both in the transferring detection-surface class and
never swept by us; deletes **263 of 4,435 GT components (5.93%)** at the deployed 6, and 19.2% of
all FN per the earlier 5,311-GT-edge measurement; min_len 4 costs 2.95%. (2) `GAP_DENSITY_REFERENCE_UM`
→ 10.36. (3) the global division cap. (4–5) the division gates — **deferred to `p8_loosefilter`**,
which is already live-testing exactly this; division-class transfer is measured 0.000.
Explicitly NOT recommended: tightening `OUTPUT_EDGE_MAX_UM`/`MOTION_RELINK_RELAXED_UM`/`GAP_CLOSE_REUSE_UM`
— all >95th percentile slack and all edge-permutation class.

Instruments (reusable, CPU): `scripts/win_bet/constant_audit.py`,
`scripts/win_bet/constant_audit_pred.py`, `scripts/win_bet/candidate_threshold_sweep.py`.

---

## 2026-08-22 — Constants audit: two contradictions resolved by measurement, one new gate found

### Resolved: `DUAL_SEED_EDGE_THRESHOLD = 0.48` is INERT (host-verified)

`audit_config` claimed 0.48 "breaks the in-degree ≤ 1 theorem" by admitting two column entries
where 0.5 admits one. `audit_constants` claimed it is inert. **Measured over the full fold-1
candidate graph (2,162,040 edges): min `edge_prob` = 0.500000 exactly; edges with prob < 0.50 =
ZERO.** So any threshold ≤ 0.5 is identical to 0.5. **`audit_constants` is right.**

All three positions now reconcile: the flag **does** exist and is set to 0.48 (`audit_config`
right, and independently confirmed by a live P0-B run manifest recording
`"edge_candidate_threshold": 0.48`); it is **inert** (`audit_constants` right); and in-degree ≤ 1
holds universally (host measurement right). The 2026-08-18 record's "hard-coded 0.5, no CLI flag
exists" is **half wrong** — the flag exists — and **half right in effect** — setting it below 0.5
changes nothing.

### Verified: `OUTPUT_MIN_TRACK_LEN` — the #1 free lever

Host-measured over 4,435 GT tracks (199 crops):

| `min_len` | GT tracks deleted | share |
|---|---|---|
| 4 | 134 | **3.02%** |
| 5 | 198 | 4.46% |
| **6 (deployed)** | **276** | **6.22%** |
| 7 | 346 | 7.80% |
| 8 | 435 | 9.81% |

(Agent reported 263/5.93% — host measures 276/6.22%; the small gap is track-vs-component counting.
Direction and magnitude agree.) **Lowering 6 → 4 returns ~3.2% of GT tracks.** It is the only
constant that is both in the transferring detection-surface class and never swept by us. CPU replay,
no GPU, no slot.

### NEW: the dual-seed retention guard binds hard, and is one-sided

From `_evidence/agent_runs/armb_deploy_2026-08-01/data/p0b_live/dual_seed_frame_retention_guard_report.json`
(md5-identical to the agent1 copy, so it is the deployed P0-B config):

| movie | frames | fallback frames | median retention | min |
|---|---|---|---|---|
| 44b6_0113de3b | 100 | 0 | 1.0072 | 0.9871 |
| **44b6_0b24845f** | 100 | **60** | **0.8876** | **0.5784** |
| 6bba_05b6850b | 100 | 0 | 0.9858 | 0.9000 (exactly on the gate) |
| 6bba_05db0fb1 | 100 | 0 | 1.0028 | 0.9774 |

**The guard fires on 60% of one movie's frames, discarding the entire 0.475-weighted secondary
detector there.** It is one-sided by construction — it can only shrink the detection surface, never
expand it — which places it in the 3.5×-transfer class. One movie sits **exactly** on the 0.9000
gate, so the boundary is live.

**Selection constraint, correctly flagged by the agent:** this telemetry is over the four visible
placeholder movies, so CLAUDE.md rule 2 admits it as **diagnosis that the mechanism binds** but
**not** as a basis for choosing a new value — that must come from LOEO.

### Other results

- **`GAP_DENSITY_REFERENCE_UM = 6.5` sits at the 0.8th percentile** of detector local neighbour
  spacing (median 10.36 µm, n=3.8M) with no record of it existing. The mechanism is *degenerate*:
  49.1% of nodes pinned at the +0.125 clip, 0.8% ever contract — so `GAP_DENSITY_ADAPTIVE=1` is a
  near-constant +0.2 µm widening. **Prize is small** (109 residual unclosed gap opportunities across
  199 crops). GT was the wrong ruler here (6.7 nodes/frame vs the detector's 154), which is likely
  why it escaped notice.
- **Candidate-threshold sweep RUN, came back clean.** Reach is monotone decreasing above 0.50 and
  the deployed 0.48 is already at the floor — the missing **17.42%** of detectable GT edges is
  **structural** (the column argmax went elsewhere), not threshold-tunable. Independently
  corroborates `error_atlas`'s 15.65%.
- **Negative result recorded:** there is no second `SAFE_DIV_MAX_UM`. Seven of nine GT-calibrated
  distance gates sit **above** the 95th percentile of their true distributions — slack, not tight.
- **The division budget binds and was never chosen:** `SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375` — 26 of
  199 crops land on **exactly** the cap, 48.2% at or above, fold-1 median forks/cap = **1.000**.
- **Mechanical account of TP=5/FP=613:** emitted forks are geometrically **disjoint** from true
  divisions — fork parent distance median **2.47 µm** (p99 5.01) against a GT far-daughter median of
  **7.13 µm**. The deployed gate triple admits **19 of 151** GT divisions; the public 0.923 triple
  admits **138**.
- **Discipline noted:** the agent ranked the division gates 4–5 rather than 1 despite their being
  furthest from the data (13.2nd/29.1st percentile), because division-class transfer measures 0.000
  and `p8_loosefilter` is live-testing exactly that — deliberately declining to pre-empt a running
  experiment.
