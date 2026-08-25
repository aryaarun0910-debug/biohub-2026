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

## 2026-08-24 — COMPUTE: Colab withdrawn. H1 survives on its own numbers; the S0-before-S1 gate inverts

- **Host change:** the two-pool budget is void. One pool remains — Kaggle GPU, host-stated ~45 h/wk
  (published quota is 30; plan against 30). It now covers training, inference and submission alike.
- **H1 was never compute-bound and its own spec says so.** Restating `h1_execution_spec_2026-08-18`
  §7 against one pool: S0+S0b+S0c 6–10 h, S1+S1b+S2 2–3.5 h, S3+S4 6–8 h, S5+S6 7–11 h —
  **21–33 GPU-h and 2 slots for the whole programme**, under one week even at 30 h. **Calendar
  remains the binding constraint.** The plan does not need re-planning; it needs re-sequencing.
- **DECISION INVERTED — S0 no longer gates S1.** Spec §6.6 ordered S0 first so a cheap proxy could
  pause H1 *before Colab hours were spent*. In one pool **S0+S0c costs 6–10 GPU-h + 1 slot while
  the S1 detector arm costs 0.5–2 GPU-h + 0 slots** — the gate is 3–5× more expensive than what it
  protected. Sequencing S1 behind S0 now buys nothing and spends calendar. **S1 runs in the same
  week as S0.** S0 is kept (it is the only leg producing LB evidence, the only evidence that counts
  since the LOEO suspension); only S3/S4 stay gated.
- **Two spec open items CLOSED by the move, not by work.** (#5) Colab's non-deterministic
  T4/L4/A100 allocation is gone — Kaggle is deterministically T4×2, CC 7.5, so fp16+`GradScaler` is
  correct and the hard-coded `torch.float16` needs no bf16 branch. (#1) The P2 OME-Zarr metadata
  re-audit and the `h1r_fetch_imaging.py` 3×4-chunk bug leave the critical path, because the edge
  half's data is **already an attachable Kaggle dataset** — `kkunizaw/biohub-zh001r` verified live
  2026-08-24: `zh001r_iso.npy` 377 MB, `zh001r_tgt.npy` 377 MB, `zh001r_nodes.npz` 9.4 MB.
- **One risk PROMOTED.** The `nn.DataParallel`/autocast trap (spec F5/F8) was conditional on Colab
  allocating 2 GPUs; on T4×2 it is guaranteed every run. The patch already autocasts *inside*
  `TemporalUNet3D.forward` (the only placement that survives DataParallel's worker threads) but is
  **still unverified on GPU (open item #4)**. S1b is now mandatory, not optional: an unmeasured run
  can be ~4× slow and look fine. Escape hatch: Kaggle P100 is single-GPU.
- **NEW BLOCKER, zero GPU, on the critical path:** `data/external/zebrahub/zh001r_identity.npz`
  (8.7 MB, the 1,258,182-edge sidecar) is **not** a Kaggle dataset — verified against
  `kaggle datasets list --mine`. Colab read it from local disk; a Kaggle kernel cannot.
  **Nothing in H1 runs until it is pushed.**
- **Contract fixed (the 2026-08-22 audit's top item, now done).** `CLAUDE.md` named `src/biotrack/`
  as "the deployed wrapper"; re-verified this session — `DEEPCENTER` **0× in `wrapper.py` vs 73× in
  the built notebook**, `VOLUME_GUARD` 4× vs 0×. `CLAUDE.md` now names
  `notebooks/kaggle_<name>/biohub-<name>.ipynb` as the audited artifact and labels `src/biotrack/`
  a partial mirror.
- **Baseline re-verified 2026-08-24:** `pytest -q` → 530 passed / 3 skipped / 29 failures, and with
  the two known files ignored **466 passed, 3 skipped, 0 failures** — the failures are confirmed
  confined, 0 new. `validate_research_tree.py` OK (65 area docs); `claims_table.py --check` OK
  (53 claims resolve).

## 2026-08-24 — S0 (xiaoleilian A/B) is DEAD on a tensor-level premise check. Zero GPU, zero slots.

`h1_execution_spec_2026-08-18` §6.3 designs S0 as a **drop-in weights swap** — "our deployed P3
pipeline, unchanged, with only the detector weights path repointed at their checkpoint" — costed at
2–4 GPU-h. The spec flagged the risk itself: *"their architecture may not match
`TemporalUNet3D([32,64,128])` … that count is itself the first measurement."* **Measured 2026-08-24,
locally, on CPU:**

- **`xiaoleilian/biohub-m001-ens3-sm6-sim2` returns 403 Forbidden** — one of the two named assets is
  no longer accessible. `biohub-unet3d-weights-v2models` is live (4 files;
  `unet3d_v2_tophat_b32.pt`, 22,453,556 B).
- **Key-name overlap with our `TemporalUNet3D`: 0 of 106.** `load_state_dict(strict=False)` →
  **missing=64, unexpected=106**.
- **It is a different architecture, not a renamed one.** Theirs: plain UNet3D, blocks
  `e1/e2/e3/bott/u1-u3/d1-d3/out`, **4 pooling levels**, **5,605,359 params**, no attention.
  Ours: `encoder_blocks/decoder_blocks/temporal_blocks/head`, **3 levels**, **1,497,610 params**,
  with multi-head **temporal attention** (`temporal_blocks.{1,2}.attn`, 64- and 128-dim).
  **They have no analogue for the temporal blocks at all**, and the depth differs — so even a
  hand-written key remap does not align. (`shape exists anywhere in theirs: 65/74` is BN-vector
  coincidence, not correspondence.)

**Consequence:** S0 is not a 2–4 GPU-h weights swap. The only valid version is the spec's fallback —
run their detector standalone for a node set and feed our linker — which is a real integration job
(their `preproc='tophat'`, `norm='p50-p99.5'`, neither of which our pipeline does) carrying exactly
the class-C/D risk of getting someone else's preprocessing subtly wrong.

**And the payoff evidence is weak, from their own checkpoint metadata:** `val_recall = 0.5789`,
`epoch_best = 8` of `epochs = 120`, `n_train_movies = 171`. A detector recording **57.9% validation
recall**, from a team **+0.003 ahead of us**. **RECOMMENDATION: kill S0**, release its 6–10 GPU-h
and 1 slot to S1. This is a class-C kill verified at tensor level, and the ledger's own rule applies
— *code-level reasoning is 1/1 for kills, 0/2 for opportunities*; this is a kill.

**Free intel extracted for S1 (the reason the check was worth running anyway).** xiaoleilian's full
recipe, read out of the checkpoint: `base=32`, `pool=4`, `norm='p50-p99.5'`, `preproc='tophat'`,
`aug='flip'`, `fpm=0`, `seed=0`, `epochs=120`, **`epoch_best=8`**.
- **`epoch_best=8` of 120 corroborates our S1 plan.** The spec's 20-epoch cosine fine-tune is the
  right order of magnitude; long training overfits this data. Select checkpoints aggressively.
- **`pool=4` vs our 3 encoder levels** is a depth lever no report has considered.
- **`preproc='tophat'`** is a detection-surface change — the one class with measured LB transfer.

**PROCESS NOTE — the disproving evidence was on local disk the whole time.** All three
xiaoleilian checkpoints have sat in `data/external/public_weights/` since **2026-08-18**
(`unet3d_v2_tophat_b32.pt`, md5 `d86d476a86cbed15ad0fa88b9120b06b`, byte-identical to the copy
re-downloaded today) — the same day `h1_execution_spec` was written naming S0 the first move and
the gate for the whole programme. The check that killed it is `torch.load` + `load_state_dict`,
one line, no GPU, no network. **This is the recurring failure mode again** — sound planning on an
unverified premise, with the machine holding the refutation and never re-reading it. It is the
same shape as the `src/biotrack/` mismatch fixed today. Concrete guard: **a bet that names an
external artifact must load it and assert compatibility before it is allowed to be costed.**

**Also cleared this session:** `aryaarun07/biohub-zh001r-identity` created **private**
(8,696,665 B, byte-count matches local; absent from public search). The H1 prerequisite blocker is
gone — every Kaggle training kernel can now attach the sidecar alongside `kkunizaw/biohub-zh001r`.

## 2026-08-24 — H1 execution surface built; corrected S1 smoke passed

- Real association-loader gate passed on the 72×20 pack: **1,192,441 continuation + 65,741
  division-daughter = 1,258,182**; nine corruption/ambiguity contracts pass.
- S1 detector defects fixed before GPU: 1.625 µm authority, no validation truncation, strict
  checkpoint init, zero-epoch seven-point threshold sweep, full optimizer/scheduler/scaler/RNG
  resume, and paired fp32/fp16 DataParallel telemetry with a 1.3× abort gate.
- Reproducible specs built: `h1r_det_s1_smoke` and `h1r_det_s1`; after integrating the edge lane,
  the focused suite is 42/42 and the non-D1 suite is 506 passed / 3 skipped.
- Kaggle `aryaarun07/biohub-h1r-detector-s1-smoke` v1 failed during import, before touching model
  or data: `ModuleNotFoundError: biohub_tracking`. Root cause was the vendored trainer's `src/`
  directory missing from `sys.path`. Both required import roots are now inserted. Corrected v2
  completed on T4x2: fp32 4.4975 s vs fp16 1.1549 s for two paired steps (**3.894x speedup**);
  selected-threshold validation F1 **0.6335 -> 0.7375** and deployed-threshold F1 **0.5538 ->
  0.7008**. This is a plumbing smoke, not promotion evidence. No competition submission. The
  full S1 run has passed its technical gate and awaits explicit host authorization.
- Embedding play implemented but not promoted: opt-in zero-gated 64-D cosine residual with
  continuation-only spatial-KNN hard triplets, plus a threshold-union-top-k candidate sidecar.
  Both are legacy-parity when disabled and require held-out association evidence before use.
- S5 now has a runnable edge trainer: deterministic adjacent-window sampling, strict public
  initialization, frozen/bypassed detector, continuation-only focal parental-softmax, exact zero
  gradient on division-daughter columns, zero-epoch baseline, top-1/candidate-recall metrics, and
  optimizer/scheduler/scaler/RNG resume. It is implementation readiness, not promotion evidence.
- S5 packaging is now complete: `h1r_edge_s5_smoke.json` and `h1r_edge_s5.json` both attach
  `kkunizaw/biohub-zh001r` and private `aryaarun07/biohub-zh001r-identity`, replace inference with
  artifact-only training cells, and declare `expects_submission=false`. Neither was launched.

## 2026-08-24 — SEVEN-AGENT CYCLE: the 0.926 is real, the naive port is dead, both GPU lanes are broken

### A. The 0.926 is LB-VERIFIED and its delta is one atomic transplant

`rockerritesh/0-926-biohub-divsub` and `kunaldesale2408/biohub-cell-tracking` are the **same notebook**
(code diff = 14 lines, all whitespace) and **both teams sit on the leaderboard at exactly 0.926**
(Rocker Ritesh 2026-08-23, Kunal Desale 2026-08-24). Their notebooks' internal markdown still says
"Public score: 0.913" and `metadata.codex.displayed_public_score = 0.913` — that is **stale
kernel-version-1 metadata**, not a refutation. Two agents raised a false alarm on it; the leaderboard
settles it. `divport` appears nowhere in our repo, so the "this is our own comment" claim was wrong.

**Lineage (verified by full config diff, 94 of 100 env keys byte-identical):**
- shared base = `pilkwang` 0.913 "Dual Seed Frame Retention Guard V1"
- theirs = base + **division transplant** -> **0.926 (+0.013)**
- ours  = base + bidirectional harmonic + out-degree guard + gap guards -> **0.915 (+0.002)**

**We are siblings, not followers.** Nothing differs in detection, secondary-seed, ILP, gap-density,
DeepCenter or short-track config. Their `BIDIRECTIONAL_EDGE_WEIGHT` is ABSENT; it is ours alone.

**The transplant is five coupled items, not one lever:**
radii **8.0 / 11.0 / 10.0** (vs our 4.66 / 8.5 / 7.65) + **C1** mid-track parent
(`source_id in incoming`) + **C2** mutual-nearest-orphan sisters (cKDTree) + **C3** t+2 divergence
(`d(_n1,_n2) - sister_dist >= 2.25`). Their own comment: the wide radii without the constraints
"produced hundreds of forks and never a single TP" — **that is exactly p8, and exactly our -0.003.**
Trap: their *markdown* still documents 4.66/8.5/7.65; a second env block overwrites it with 8/11/10.
Porting the prose ports the wrong numbers. The DeepCenter safe-div veto is **OFF in theirs**,
same as ours — it was never the missing discriminator.

### B. C3 ALONE IS DEAD — measured on our own GT, zero GPU

Statistic `delta = d(daughters at t+2) - d(daughters at t+1)` over `data/train` (199 crops, 151 GT divisions):

| | n | p10 | Q1 | median | Q3 |
|---|---|---|---|---|---|
| GT division delta (um) | 147 | **0.00** | 0.91 | **2.35** | 4.12 |

**At the public tau=2.25, only 76 of 151 (50.3%) true divisions survive — in PERFECT ground truth.**
Separation AUC vs our scored false forks = **0.632** (weak). Our shipped LOEO graphs emit
**11,582 forks of which 2 are true divisions**. Best trade is tau~2.0-2.25 at ~7.8 FPs cut per TP lost;
FP reduction is real (722 -> 118 scored FPs, -84%) but lifts div_jaccard only **0.0023 -> 0.0074**.
**A precision filter downstream of a 2/151 proposer cannot add a true division. DO NOT ship C3 alone.**

### C. ...but the same measurement VALIDATES the radii half, which is the load-bearing half

**GT median sister separation at birth = 10.57 um. Our sister gate is 8.5 um. Theirs is 11.0 um.**
In perfect GT, **only 19 of 151 divisions satisfy our geometric gates at all.** Our gates are tuned
below where divisions actually live; theirs are tuned to it. This is independent, GT-based
corroboration that 8/11/10 is right and 4.66/8.5/7.65 is wrong — arrived at without reference to
their notebook.

**The dominant residual lever is upstream of every gate.** `div_proposal_funnel.py`, fold 0, 26 GT
divisions: mother detected 26 -> both daughters detected 25 -> mother out-degree 1: 24 -> linked child is
a true daughter 23 -> **other daughter is ORPHAN: 4 (loses 19)** -> parent_dist <= 4.7: **0 (loses 4)**.
**Zero of 26 are proposable.** Condition (e) — the second daughter already having a parent — kills
19/26 and **no gate setting touches it.**

### D. Both GPU lanes are defective. Neither may launch as specified.

**S1 smoke performed ZERO optimiser steps — VERIFIED, not inferred.** `kernel.log:136,139` carries
`UserWarning: Detected call of 'lr_scheduler.step()' before 'optimizer.step()'`, which PyTorch emits
only when the optimiser has never stepped: GradScaler started at 65536, hit inf/nan, skipped both
steps. `det_loss=4.6878` is the loss of the **unmodified** initialisation. The entire **+0.104 F1 is
BatchNorm running-statistic adaptation** (AdaBN), doubled per step because gradient checkpointing
recomputes conv blocks in backward. **The smoke is not evidence that training works.**
Discriminating control, no code change: `H1R_LR=0, H1R_EPOCHS=1, H1R_MAX_STEPS=0, H1R_BENCH_STEPS=0`
~ **0.1 GPU-h**.

**S1 further:** checkpoint selection is `max F1 over ALL thresholds` (`h1r_det_train.py:68`, `:424`)
while the docstring claims the deployed operating point — the sweep is monotone decreasing so it
always selects the loosest probe (p0.5, 0.469 away from deployed 0.96875). `eval_detection` uses ONE
forward while deployment uses **8-view TTA**, so the number labelled "deployed" is not the deployed
operating point. Training normalises per-crop over uint8; deployment over whole-video zarr attrs over
uint16. `H1R_RESUME=1` is dead code on Kaggle (`/kaggle/working` starts empty, no `kernel_sources`).
**SEV-1: the fine-tuned UNet trunk IS the association representation** — `unet_out` feeds
`predict_edges` in deployment, so moving the detector rewrites every edge feature under a transformer
trained on the old one. That trunk is the 0.915 substrate and nothing measures it.

**S5 would crash on step 1 and, if fixed, would train the wrong objective.**

1. `F.binary_cross_entropy` (`h1r_edge_train.py:177`) inside CUDA autocast (`:336`) — banned op, hard
   RuntimeError. **`h1r_trainer_patch.py:12-13` documents this exact trap**; the edge lane never
   applies that patch. A CPU smoke cannot reproduce it.
2. **ALL division supervision is deleted** (`:95-97` zeroes division columns, `:165-167` drops them
   from the loss). The model never sees one mother->daughter positive. **This discards the entire
   65,741-link asset that is H1's whole rationale**, and inverts the repo's own P3 division upweight.
3. "Frozen detection" freezes only `detect_head` (one 1x1 conv); the shared UNet trains 20 epochs.
   The post-run guard (`:351-353`) is tautological. The saved checkpoint is the full model + config ->
   silently ships a drifted detector.
4. `node_cap=256` against ~942 nodes/frame with **independent** source/target sampling: a target's
   true parent survives ~27% of the time, so training sees ~73% of columns as "no parent" while
   deployment has ~92% *with* one — background prior ~9x off, softmax denominator 256 vs ~900.
5. Checkpoint selection is `accuracy x recall` — the exact rule `h1r_trainer_patch.py:19-22` already
   rejected for having no precision term.

Verified clean: both specs DO attach `kkunizaw/biohub-zh001r` AND `aryaarun07/biohub-zh001r-identity`;
label construction and t->t+1 indexing are correct.

### E. Free intel, LB-measured by other people

- `evgendvorkin/biohub-0-923-lb` annotates its config with results: **`ILP_DIVISION_WEIGHT` at
  0.3 / 1.0 / 2.0 / 3.0 ALL scored 0.915.** Independently confirms our "ILP is structurally inert"
  finding and closes that lever for free.
- **`leevvin` README claims the public support pack was trained with `test` inside `train` over all 199
  labelled movies.** If true this is a mechanistic explanation of the entire LOEO->LB transfer
  failure — instrument contamination, not "no offline instrument can predict the LB". UNVERIFIED;
  `splits_movie_heldout.json` is already on disk at `data/external/public_weights/leevvin/`.
- 0.926 and 0.923 lineages **disagree on the DeepCenter checkpoint**: `checkpoint_last.pt` (epoch 500)
  vs `best.pt` (epoch 2), same dataset. One-flag A/B.
- Highest live public score across ~900 kernels is **0.926**; the two 0.950 titles are pre-rescore
  metric-hack-era artifacts and are dead. Top-3 is now **0.953** (TWEAK), leader 0.962.

### F. THE PROCESS FINDING — the repo does not inherit its own fixes

S5 defects 1, 2 and 5 are each a mistake **already found, documented and fixed in
`h1r_trainer_patch.py`**, then re-made in a new file that does not apply that patch. Same shape as the
two other recurrences today: `CLAUDE.md` pointing at `src/biotrack` while the real program was the
notebook, and xiaoleilian's weights sitting unread on disk for six days while a spec costed an
experiment around them. **A capability ledger would not catch any of these.** What is needed is a
**defect ledger** — known trap -> where found -> what fixes it -> which files must inherit it — enforced
as a `kaggle_factory` build gate, the same way it already asserts exact-match on edits.

## 2026-08-25 — POST-P9 CYCLE: LOEO contamination confirmed in our own specs; the contested-daughter loss is STRUCTURAL

### A. TASK 1 — the "leak". leevvin's claim is UNVERIFIABLE. Ours is WORSE and it is VERIFIED.

**leevvin's claim (public support pack trained with `test` inside `train`, all 199 movies) cannot be
checked and the evidence points against it.** The support pack publishes no split manifest and no
main training script (only the DeepCenter one) — verified by listing
`pilkwang/biohub-tracking-support-pack-50ep-v1` (no `run_main_training_oneshot.sh`, no
`split_manifest.json`). The vendored trainer's DEFAULT split is **clean and disjoint**:
`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:1050` →
`folds = [{"split": 0, "train": stems[n_val:], "test": stems[:n_val]}]`. A leak would require a
custom splits file that is not published. **Record this as UNVERIFIED, not as fact.**

**But the sharper question is answerable and the answer is yes — our LOEO split the CROPS and never
the MODEL.** All nine LOEO specs attach `pilkwang/biohub-tracking-support-pack-50ep-v1` and evaluate
on competition training movies. The weights are fixed public weights trained on some portion of the
same 199 labelled movies. **Any post-processing tuned on that instrument was tuned against a model
that had already seen its evaluation set.** This is a mechanistic explanation of the LOEO→LB gap
that does not depend on leevvin at all.

**AND THE TWO FOLDS ARE NOT THE SAME INSTRUMENT — verified:**

| spec | own weights attached |
|---|---|
| `p3_base_loeo_f0`, `p3_armb_loeo_f0` | **none** (public weights only) |
| `p3_base_loeo_f1`, `p3_armb_loeo_f1` | **`aryaarun07/biohub-oof-weights`** |

Fold 0 is public-weights-only; fold 1 attaches our own out-of-fold weights. **We have been pooling a
contaminated arm with a possibly-clean one and reading the average**, which plausibly explains the
fold-0/fold-1 MDE asymmetry (±0.0043 vs ±0.0057) we had attributed to noise.
**LIMIT OF THIS FINDING: attachment is verified, USE is not.** Confirming the f1 notebook actually
loads `biohub-oof-weights` is the next step and it is a grep, not an experiment.

### B. TASK 2 — the contested-daughter loss is STRUCTURAL, caused by one line, and has a known fix

**Verified in the DEPLOYED P9 notebook (not the `src/biotrack/wrapper.py` mirror):
`motion_relink_edges` solves `linear_sum_assignment(cost)` — 3 occurrences.** That returns a
**bijection**, so **out-degree ≤ 1 is a hard structural invariant of the linker.** "The second
daughter already has a parent" is therefore unreachable by ANY radius, threshold or constraint.
This unifies three previously separate observations: the relink discards 99.87% of solver output;
100% of output divisions are post-processor artifacts; and the orphan requirement appears in BOTH
`coupled_division_transplant.py` and `restore_learned_divisions.py` — it is forced, not chosen.

**Therefore: no gate tuning can recover the 19/26. Stop tuning C3's 2.25 µm threshold.**

**THE FIX (M1, top-ranked): duplicated-source LAP.** Augment the cost matrix from `(|S|,|T|)` to
`(|S|+|S_div|,|T|)` with a duplicate row block for division-eligible mothers at `cost + δ_div`, then
solve the SAME `linear_sum_assignment`. Out-degree 2 becomes a first-class outcome chosen on cost;
there is no "already linked" node because a daughter is just a column. Restrict `S_div` to
C1-satisfying mid-track sources to bound the runtime (LAP is O(n²m), `MOTION_RELINK_MAX_FRAME_NODES`
= 2600). δ_div is in µm, same units as `motion + 0.05*raw - 0.75*prob`.

**Corroboration from the literature:** every tracker that handles division well relaxes exactly this
bijection — Linajea's ILP (`Σ next_edges − 2·selected ≤ 0`), Padfield's coupled min-cost flow,
Haubold's generalised successive shortest paths, and the duplicated-column trick in arXiv 2403.15011.
**Jaqaman's second-LAP is NOT the right model** — it requires the daughter to be a segment start,
i.e. it shares our defect. TrackMate-Oneat measured branching accuracy **0.122 → 0.328** via exactly
this kind of contested re-linking.

**PRE-REGISTERED CHEAP TEST, zero GPU, NOT YET RUN.** Replay the augmented LAP offline against
`C:/temp/preilp_f1_v2/preilp_split1.parquet` (4,685,519 rows; columns dataset/row_type/node_id/
t/z/y/x/source_id/target_id/edge_prob) and count how many of the 19 contested daughters flip to the
correct mother. **KILL CRITERION, fixed in advance: fewer than 8 of 19 flip at any δ_div ⇒ M1 is
dead.**

**Two supporting levers found with it:**
- **`H1R_DIV_WEIGHT` is 3.0; Trackastra's published λ_div gives weight 11** — a 3.7× under-weighting
  of division rows, one env var.
- **The proposer writes `edge_prob: None` on every fork** — it never receives the discriminant that
  measured median 0.9188 at true divisions, while C3's divergence statistic has AUC only 0.632.
  Replacing C3's gate with an `edge_prob` gate is a one-script AUC comparison on existing dumps.

### C. THE PUBLIC FRONTIER MOVED AGAIN — 0.927, and still with NO retraining

`evgendvorkin/biohub-0-927-lb` (ran 2026-08-24 21:25, 14 votes) attaches only the three stock
pilkwang datasets. Deltas vs the 0.926 lineage: **8-view D4 detection TTA**, a **second temporal
model ensemble**, `BIDIRECTIONAL_EDGE_WEIGHT` **0.20 → 0.30**, `DEEPCENTER_SAFE_DIV_VETO` **ON**.

**This contradicts our P9 interference finding and the contradiction is live.** We inferred from
0.913+0.013 vs 0.915+0.010 that our harmonic is subsumed and should be *removed*; the 0.927 raises
it to 0.30 instead. Both cannot be right. One slot settles it. `xstargate/biohub-v17-e1..e4`
(2026-08-23) are published ablations of exactly these mechanisms — free attribution data.

### D. FIRST PUBLISHED LOEO ASSETS — the two halves of our own planned pipeline, now public

- **`poonszesen/biohub-c1-honest-fold1-e25-probe`** (2026-08-24, 7.7 MB) — LOEO fold-1 edge
  predictor, `training_embryo 44b6` / `held_out_embryo 6bba`, 25 epochs, seed 314159. Caveat:
  `skip_validation: true`, author calls it a calibration probe.
- **`daifanhao/biohub-loeo-synth-code-public-v1`** (2026-08-23, 4.7 KB) — a complete LOEO trainer
  that **asserts** train/held-out disjointness (`raise RuntimeError("LOEO train/held-out overlap")`),
  `EXPECTED_COUNTS {"44b6": 71, "6bba": 128}`, and warm-starts the UNet+detector head **without
  touching edge weights**. Pairs with `daifanhao/biohub-synthetic-detector-pretrain-public-v1`.
- **`raykkretzschmar/biohub-robust-motion-proposals-v2`** — the only asset publishing held-out
  numbers on a **movie-disjoint** split (clean proposal recall @2 µm 0.9035, corrupted 0.8987).
- `yuki0731/biohub-coord-inject` (2026-08-25, 83 MB) — 4-seed detector ensemble + coordinate
  refiners, **but epoch-0/1/6 checkpoints and no protocol or metrics**. Unvalidated.

### E. OPEN / NOT DONE — carry these forward

1. **The M1 LAP replay has NOT been run.** It is the decisive test for the largest identified loss
   channel and costs only CPU. Kill criterion is pre-registered above.
2. **Confirm the f1 LOEO notebook actually LOADS `biohub-oof-weights`** (grep, not experiment).
3. **Metric headroom analysis was still running at close** — the question it answers is whether the
   0.1-weighted division term can mathematically deliver +0.028 at all, or whether it cannot and the
   budget must go to detection/association. **Do not commit GPU before this is answered.**
4. Top-solution gap analysis stalled twice; its "major intelligence" finding was not recovered.
5. S5 remains unlaunchable (CUDA-autocast BCE crash; all division labels masked out). S1 has three
   SEV-1 defects. Neither may promote anything.

## 2026-08-25 — SWARM RESULTS (cited). The metric says DIVISIONS CAN reach top-3; two top-3 teams say the gap is NOT the edge model.

### A. COMPETITOR INTELLIGENCE — verbatim, cited

**TWEAK (rank 3, LB 0.953)** — https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352
> "we are not focused on 0.0001 or Division J; we are working on a **universal plugin that the bio cell
> team can plug into their current pipeline with minimal changes. We have tested our plugin with every
> available unique public notebook and model, with gains ranging from 0.030, 0.040, to 0.050 instantly
> just attaching our plugin. We've seen gains from a single public model reach a score of 0.940
> untuned.** We are not focused on the 0.0001 or tuning to the hidden."

Existence and magnitude VERIFIED (their words); **mechanism NOT disclosed** — Sergio Alvarez asked the
disambiguating question in-thread and TWEAK never answered. This is +0.028 and more, **without new
weights**, and explicitly **not division work**.

**Soheil Ayati (rank 2, LB 0.959)** — https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101
> "Break your missed edges into two categories: **missing endpoint nodes and incorrect associations. In
> my case, many 'linking' issues actually originated earlier during node selection**, so improving the
> edge model won't necessarily help. **Always validate complete movies using the official scorer and
> movie-level OOF splits. Edge-level random CV can be highly misleading.**"

**Tang (rank 7, LB 0.946)** — https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604
> "the current ckpt has kind of hit a wall, it's hard to get more gain from post-processing alone."
> "the common approach ... is basically two parts: **modeling and track optimization, aka
> post-processing.** ... i'm not working on that part."
Tang is at 0.946 **without** the track-optimisation half — consistent with TWEAK's plugin being a
separable +0.03–0.05.

**Mendrika Ramarlina (rank 87)**, same thread — the cleanest public decomposition:
> "**Edge recall ≈ node recall² × conditional linking accuracy.** This means **detector misses are
> especially expensive.**" and "keep the detector fixed while comparing linkers ... otherwise
> post-processing can appear to improve locally by changing the candidate set and graph simultaneously."

**mikelou1 (rank 28)**, discussion/735352 and /737101 — calibration points: retrained **from scratch**
and reached only **0.928**; his division J ≈ **0.3** is considered good; and he reports **0.919 CV vs
0.895 LB** — a 0.024 inversion, the exact pathology Soheil warns about.

### B. LEADERBOARD STRUCTURE — the body is one forked artifact, the top is bespoke

Downloaded 2026-08-25, 2,693 teams. Exact-value spikes: **0.926 → 106 teams**, 0.915 → 198, 0.913 → 105,
0.917 → 92, 0.923 → 42. **Above 0.927 no score value has more than 7 teams**; from 0.940 up it is 1–3.

| band | teams |
|---|---|
| ≥ 0.926 | 202 |
| 0.926 ≤ x < 0.947 | 196 |
| **≥ 0.947** | **6** |
| ≥ 0.953 (top-3) | 3 |

**There is no cluster at the top to reverse-engineer.** Note also Dylan Gallagher reached 0.944 in
**7 submissions** and Corwin 0.943 in **9** — unreachable without a trusted offline signal.

### C. METRIC HEADROOM — computed, and it OVERTURNS my earlier statement

Scorer verified at `.venv/Lib/site-packages/tracking_cellmot/metrics.py`:
`adj_i = max(0, J_i·(1 − 0.1·r_i))` (`:446-451`), `ADJ = Σw_i·adj_i / Σw_i` (`:498-504`, edge-volume
weighted mean), `D` = **micro**-pooled division Jaccard (`:508-521`), `SCORE = ADJ + 0.1·D` (`:522`).
`node_recall` is reported but **is NOT in the score** (verified absent from the expression).
Hand-recomputation matched the stored value bit-exactly (0.9001767463).

**Pooled LOEO baseline 0.733456** (f0 0.903318 / f1 0.704231; div TP/FP/FN = 5/613/146; D = 0.006545).

| oracle scenario | Δ pooled |
|---|---|
| **division J → 1.0** | **+0.099346** |
| division reach-limited oracle (all 89 reachable, FP=0) | **+0.058286** |
| division FP → 0 ONLY | **+0.002657** |
| division FN → 0 (current FP kept) | +0.019110 |
| edge ADJ → 1.0 | +0.277081 |
| edge FN → 0 | +0.167171 |
| edge FP → 0 | +0.089494 |

**I PREVIOUSLY SUGGESTED THE 0.1-WEIGHTED DIVISION TERM MIGHT MATHEMATICALLY NOT REACH +0.028. THAT IS
FALSE.** A perfect division Jaccard is worth **+0.0993 — 3.5× the gap**. Even the reach-limited oracle
is **+0.0583, 2× the gap**. Required: div J **0.006545 → 0.2865**.

**Exact frontier for +0.028** (151 GT divisions, 89 currently reachable):

| div FP | div TP needed | feasible? |
|---|---|---|
| 0 | 44 | YES |
| 50 | 58 | YES |
| 100 | 72 | YES |
| ≥170 | >89 | **NO** |

Current: **TP=5, FP=613, precision 0.8%**.

**RULED OUT — division FP suppression alone (+0.0027).** Any fork-precision project that does not also
raise TP is mathematically incapable of closing the gap. **P9 was a precision fix; the next one must
raise TP.**

**RULED OUT — node-count / N_est manipulation.** Two independent grounds: (i) `estimated_number_of_nodes`
lives only in `data/train/*.geff/zarr.json` and is **not observable at inference**; (ii) measured — the
node-budget sweep earned +0.014437 in ratio bonus against −0.049805 Jaccard damage, **net −0.035368**.

**Marginal costs (pooled):** a missing GT-matched node costs **−1.40e-5**; a spurious node in
unannotated space costs **−3.33e-8**. **Missing costs 420× more than spurious — over-detect.**
One division TP = **+1.31e-4** = worth **19 edge FNs**.

**Dead zones (verified):** 98.1% of predicted edges are never scored (p9: 117,901 predicted, **2,190**
`pred_valid`); **97.95% of predicted forks are never charged** (f0: 5,279 forks, 108 charged). A fork is
charged **iff** it sits on an annotated GT cell with ≥1 GT child. **Global fork suppression is therefore
pure downside** — it cannot preferentially reduce charged FPs but does destroy TPs.

### D. NOMINATION ARCHITECTURE — Ultrack uses TOP-K; we use a bare threshold

**Ultrack** (Nature Methods 2025, DOI 10.1038/s41592-025-02778-0; arXiv 2308.04526), verbatim:
> "For each segmentation i ∈ H ... compute its candidates **2k-nearest neighbors within a predefined
> radius** ... The association score ... **w(i,j) = IoU(i,j)^γ** ... γ is consistently set to four ...
> **Include into E_T the k pairs per segment in t with the largest IoU, using their distance as a
> tie-breaker.**"

Code: `ultrack/core/linking/processing.py` — `KDTree(...).query(target_pos, k=2*config.max_neighbors,
distance_upper_bound=config.max_distance)` then `sorted(neighborhood, reverse=True)[:config.max_neighbors]`.
Defaults (`ultrack/config/config.py`): **`max_neighbors = 5`, `max_distance = 15.0`**.
**Retrieve 2k geometrically → rerank by appearance (masked IoU⁴) → hard-truncate to k.**
The search ball is **movement-displaced** (`target_pos += target_shift`) when optical flow is on.
Ultrack also ships an **oracle instrument for candidate-set ceiling** — MTIoU, `ultrack/core/match_gt.py`
+ CLI `ultrack/cli/match_gt.py`.

**Linajea** (Nature Biotech 41, 44–49 (2023), DOI **10.1038/s41587-022-01427-7** — note a DOI given
earlier in this project was wrong and resolved to a COVID vaccine paper) uses a **fixed radius, no k**:
`extract_edges_blockwise.py` → `pre_kd_tree.query_ball_point(nex_parent_center, edge_move_threshold)`,
ball centred on the **movement-displaced** position (`use_mv_distance` default True). Out-degree is
unbounded. Published θ ∈ {25, 30, 40, 45} **world units, not µm**. Node recall reported >0.99 (mouse,
Drosophila) / >0.96 (zebrafish); **edge recall of the candidate graph is NOT published**, though the
instrument exists (`linajea/evaluation/analyze_candidates.py::get_edge_recall`).

**Abstention:** both put the null option **inside the same "exactly one" sum** as the real candidates —
Linajea `Σ_{p∈P_v} y_p + y_v^T − y_v = 0` (appear indicator competes with every parent edge); Ultrack
`x_α`/`x_β` slack variables. **Neither normalises abstention against candidate scores.** Our column
softmax has **no null column at all** — this is the structural gap.

### E. WHAT THIS IMPLIES — three cheap CPU tests that gate the GPU spend

1. **Oracle-node substitution** — replace predicted nodes with GT nodes on a held-out movie, keep linker
   and solver fixed, score officially. Gives the exact detection headroom. Motivated by Soheil (#2) and
   by `edge recall ≈ node recall² × linking accuracy`.
2. **Candidate-width sweep** — recompute the oracle ceiling as a function of candidate radius and top-K
   only. If it rises ≫0.03 as K widens, TWEAK's plugin is localised to candidate generation. Our own
   `error_atlas` says a perfect solver over *today's* candidates is worth only +0.0012, so H1 must be a
   candidate **widening**, not a better solver.
3. **LOEO↔LB regression** — regress our historical LOEO scores against achieved LB. Given §A's finding
   that fold 0 uses public weights and fold 1 attaches our own, and mikelou1's 0.919/0.895 inversion,
   this decides whether our decision loop is load-bearing at all. Zero compute.

**Do not commit the 21–33 GPU-h until tests 1 and 2 have run.** mikelou1 retrained from scratch and
landed at 0.928 — retraining is necessary-but-nowhere-near-sufficient.

## 2026-08-25 — RETRIEVAL TEARDOWN: the nominator is a hard ARGMAX, 70.6% of the loss is RANKING, and the ILP cannot prune

Measured on `c:/temp/error_atlas/*.parquet` and `c:/temp/preilp_f1_v2/preilp_split1.parquet`
(fold 1, 128 crops, 2,523,479 nodes / 2,162,040 candidate edges), scale [1.625, 0.40625, 0.40625] µm.

### A. The deployed nominator is a per-target ARGMAX, not "at most 2 parents"

| candidate in-degree per target | count |
|---|---|
| **1** | **2,162,040** |
| 2 | **0** |

`min(edge_prob) = 0.5000002`, p01 = 0.5103. The column softmax is so peaked at deployment node counts
that **only the column winner ever clears the 0.48 gate, and it always clears 0.5**. The rule is
operationally `argmax_i p[:,j] if max > 0.5 else abstain`. Out-degree (source axis) is unconstrained:
{1: 1,795,844, 2: 165,148, 3: 10,911, 4: 715, 5: 59, 6: 2}.

**335,210 of 2,497,250 non-first-frame nodes (13.42%) receive ZERO candidate parents** → forced track
starts. This supersedes the earlier "3 × 0.48 = 1.44 > 1 so at most 2 parents" argument — directionally
right, but the measured reality is stronger: it is always exactly one.

### B. The 17,001 missing GT edges are TWO different failures, and the big one is RANKING

| branch | count | share |
|---|---|---|
| **target already has a candidate — the WRONG parent** | **11,996** | **70.6%** |
| target has zero candidates | 5,005 | 29.4% |

**This kills the "lower the threshold / add abstention" family as a PRIMARY fix.** 70.6% of the loss is
a ranking error, not a thresholding error.

**The true parent is geometrically CLOSER than the nominated one in 77.3% of the 11,996 cases:**

| | median | p75 |
|---|---|---|
| GT parent → child displacement | **2.19 µm** | 3.83 |
| nominated parent → child displacement | **4.77 µm** | 6.13 |

**Divisions are hit ~3× harder:** 49.6% of division-daughter links are un-nominated vs 17.3% of
continuations.

### C. kNN-3 is a near-SUPERSET of today's nominations and recovers 92.5% of the misses

cKDTree per (dataset, t−1), all 17,001 missing edges evaluated:

| gate | missing GT edges contained | contains today's nominated source |
|---|---|---|
| k = 1 | 24.7% (4,203) | — |
| **k = 3** | **92.5% (15,726)** | **99.4%** |
| k = 5 | 97.7% (16,609) | 99.7% |

(k=2 and k=10 NOT computed. Superset figure is a 24,000-edge sample over 8 of 128 datasets.)

So **kNN-3 ≈ (today's set, 99.4% retained) ∪ (92.5% of the misses)** at **3.5× the candidate count**
(2.16 M → ~7.49 M). The k=1 → k=3 jump is the finding: **nearest-neighbour alone is NOT the fix (24.7%)**,
and combined with §B neither probability alone nor distance alone ranks correctly — **retrieval and
ranking must be separated.**

### D. THE BLOCKER — VERIFIED IN THE DEPLOYED P9 NOTEBOOK, not the mirror

`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:555-563`:
```python
solver = td.solvers.ILPSolver(
    edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),   # ilp_edge_weight = -1.0
    appearance_weight=cfg.ilp_appearance_weight,
    disappearance_weight=cfg.ilp_disappearance_weight,
    division_weight=cfg.ilp_division_weight)
```
Deployed env, verified in `notebooks/kaggle_p9_coupled_division/biohub-p9-coupled-division.ipynb`:
**`BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"`**, `BIOHUB_ILP_DISAPPEARANCE_WEIGHT = "1.5"`.

1. **The ILP ranks by `edge_prob`, which IS the column softmax. Widening the candidate set WITHOUT
   rescoring is a NO-OP** — the ILP re-selects the same argmax, because in-degree ≤ 1 makes this a
   replacement decision and every added candidate is by construction lower-probability.
2. **With `appearance_weight = 0.0` there is NO abstention price.** Selecting an edge changes the
   objective by −p; leaving the target to appear costs 0. **Every feasible candidate is accepted.**
   The ILP is a maximiser, not a filter — so **"over-nominate and let the ILP prune", the
   Ultrack/Linajea contract, DOES NOT HOLD in our deployment.** Restoring appearance weight is a
   PRECONDITION for any candidate-widening work, not an independent lever.

### E. Ranked mechanisms

- **M2 (run FIRST, trivial):** `BIOHUB_ILP_APPEARANCE_WEIGHT` 0.0 → 0.1 (vendor default). Kill if it does
  not hold or improve adjusted Jaccard on the *current* candidate set. Precondition for M1; the two
  interact and are NOT additive.
- **M1 (rank 1):** kNN-3 geometric gate ∪ today's set, then **rescore with a signal orthogonal to the
  column softmax** — Ultrack's shape exactly (geometric retrieve → different-signal rerank → ILP select).
  Candidate features already measured in `error_atlas_2026-08-19.md:255-268`: `rev_margin_um` AUC 0.7277,
  `nn_margin_um` 0.7152, combined 0.7752. No retrain for the gate. CPU-only kill test on the pre-ILP
  parquet; **distance-alone is already falsified (24.7%)**.
- **M3 (rank 3, information):** `scripts/kaggle_edits/h1r_candidate_export_patch.py` already exports
  top-k raw logits per target (`H1R_EDGE_TOPK`, default 5), is unit-tested, and **has never been run.**
  It is the only way to learn whether the true parent is in the *model's* top-3 logits — which decides
  whether M1 can use model score at all or must lean on geometry. Kill the learned-rerank lane if top-5
  logit recall over the misses is below ~50% (geometry already gives 97.7%).
- **M4 (rank 4, deferred):** null/background column in the parental softmax. Already implemented in
  `h1r_edge_loss_patch.py` (E1, `H1R_BG_TERM`). **Do NOT ship inference-only** — it is a monotone
  per-column rescale that strictly lowers every probability, so at a fixed 0.48 threshold it can only
  SHRINK the candidate set and make recall worse. Needs the retrain to pay. It fixes the 29.4%
  calibration branch, not the 70.6% ranking branch.
- **M6 — ALREADY FALSIFIED, do not build:** blanket top-1 fallback for zero-candidate targets is
  ≤5,005 TP for ~335,210 added edges ≈ **1.5% precision against a ~47% break-even**.

**CORRECTION carried into this record:** the metric fact "a missing detection costs ~420× a spurious one"
applies to **DETECTIONS**, not edges. The edge term is Jaccard-like and symmetric, so it does **not**
rescue over-nomination of edges. Do not transfer that asymmetry across terms.

### F. Literature, cited

- **Ultrack** — retrieve-2k-then-rerank; `KDTree.query(target_pos, k=2*max_neighbors,
  distance_upper_bound=max_distance)` then rerank by masked IoU⁴, truncate to k. Defaults
  `max_neighbors=5`, `max_distance=15.0`. https://arxiv.org/pdf/2308.04526 ·
  https://raw.githubusercontent.com/royerlab/ultrack/main/ultrack/core/linking/processing.py ·
  Nature Methods 2025 DOI 10.1038/s41592-025-02778-0
- **Linajea** — fixed radius, NO k; `query_ball_point(nex_parent_center, edge_move_threshold)`,
  unbounded out-degree, ball centred on the **movement-displaced** position. Nature Biotechnology 41,
  44–49 (2023), **DOI 10.1038/s41587-022-01427-7** (a DOI used earlier in this project was wrong and
  resolved to a COVID-19 vaccine paper). https://github.com/funkelab/linajea
- **Trackastra** — "parental softmax": block-wise sum of possible parent associations **at most one**;
  candidate graph built by averaging over a sliding temporal window. ECCV 2024,
  https://arxiv.org/abs/2405.15700 · https://github.com/weigertlab/trackastra
- Both Ultrack and Linajea place the abstention/null option **inside the same "exactly one parent" sum**
  as the real candidates. **Our column softmax has no null column** — the structural gap.
- Cross-domain (cited from memory, URLs NOT fetched — verify before quoting): two-stage retrieval
  doctrine, Covington RecSys 2016; Huang KDD 2020 https://arxiv.org/abs/2006.11632; dustbin/null
  options SuperGlue https://arxiv.org/abs/1911.11763, LightGlue https://arxiv.org/abs/2306.13643,
  DETR https://arxiv.org/abs/2005.12872; Deformable DETR https://arxiv.org/abs/2010.04159;
  calibration Guo https://arxiv.org/abs/1706.04599.

### G. Caveats
Fold 1 only; fold 0 untested. The kNN gate measured is **static**, Linajea's is motion-displaced — a
displaced gate would likely do better at the same k, untested. EmbedTrack / PAC-MAP / CellTrackFormer /
CTC-winner nomination rules remain an open gap.

## 2026-08-25 — DETECTION TEARDOWN: the NMS rule is EXONERATED; the one-hot delta target explains the threshold pathology

### A. The peak/NMS rule costs at most 0.12 pp of recall — MEASURED, model-independent

`predict_unet_transformer.py:284-286` `is_peak = (logits == max_pool3d(logits, pool_kernel, stride=1)) &
sigmoid(logits) > det_threshold`. Kernel resolves via `pool_kernel_from_um` (`:229-251`):
`voxel_size = ds.scale x downsample = (1.625,0.40625,0.40625) x (1,4,4) = (1.625,1.625,1.625)` isotropic,
`round(3.0/1.625) = 2 -> forced odd -> 3`. **Kernel (3,3,3), suppression radius exactly 1 voxel =
1.625 um.** Training uses `pool_kernel_um=5.0` (`train_unet_transformer.py:626`) -> `round(5/1.625)=3`
-> also (3,3,3). **No train/inference mismatch.** The plausible bug (raw anisotropic `ds.scale` leaking
in, giving (3,7,7) ~ 11.4 um lateral suppression) was checked and is **NOT present**.

**Competition GT, all 199 movies, 131,641 nuclei:** GT pairs in the same 1.625 um voxel = **0**;
nuclei sharing a (3,3,3) block = **0 (0.000%)**. NN distance p1 = 11.87, median **43.42 um**.
**Recall ceiling of the peak rule on scored data = 100.000%.**

**Dense Zebrahub GT, `ZSNS001_tracks.csv`, 9 timepoints, 245,998 nuclei:** same-voxel **0**;
sharing a (3,3,3) block **287 (0.12%)**; NN p1 = 4.32, median **7.13 um**. **Ceiling >= 99.88%.**

**Nuclei are ~7 um apart; the NMS radius is 1.625 um — a 4.3x margin. Pool-kernel tuning is a DEAD
LEVER; do not spend a GPU run on it.** Sub-voxel refinement is likewise not a recall lever (the metric
matches at 7 um and median NN is 7.13 um, so +-0.8 um quantisation almost never flips a match) —
consistent with the measured bilateral -0.0004/-0.0009 that closed `bet-subvoxel-refine`.

**Therefore all of the missing ~55% recall is in the heatmap values and the 0.96875 threshold.**

### B. THE CAUSE: a one-hot delta target the network cannot fit

`train_unet_transformer.py:528-573`:
```python
target = torch.zeros_like(logits)
zi = gt_coords[:,0].long().clamp(...)   # float um position TRUNCATED, not rounded
target[b, zi, yi, xi] = 1.0             # ONE-HOT, single voxel, no spatial extent
```
A delta target with 1-voxel support in a 64^3 = 262,144-voxel volume, at a **truncated** index
(`.long()` not `.round()` -> systematic half-voxel bias, `:562-564`).

1. **The target is not a learnable function of the image** — the net cannot know which of the ~8 voxels
   under a 7 um nucleus the annotator's float coordinate truncated to. The optimal hedge under BCE is a
   **broad low-amplitude blob**, whose argmax is ill-defined and whose max sigmoid sits well below
   0.96875. **This explains the monotone-decreasing F1-vs-threshold curve as a SYMPTOM OF THE TARGET,
   not an independent bug.**
2. **Zero gradient credit for near-misses** — one voxel off scores as both a FN and a FP.
3. `neg_weight=0.1` makes positives dominate 10:1, which *should* over-produce detections, yet recall is
   0.445 — only explicable if peaks are flattened by (1) then guillotined by the 0.96875 gate.

**Honest limit:** the parameterisation explains the SHAPE of our failure but **not** a dense-packing loss,
because §A shows we do not have a dense-packing problem. PAC-MAP's proximity-adjusted amplitude,
StarDist's polyhedral NMS and Spotiflow's flow refinement are all solutions to *crowding*, which costs us
<= 0.12 pp. **Ours is a calibration / target-smoothness problem.**

### C. Ranked, with the cheapest also the largest

1. **Move the operating point down the existing PR curve (0.96875 -> ~0.50). ZERO GPU.**
   Measured: recall 0.445 -> 0.563 (+0.118), precision -0.008. Since `edge recall ~ node_recall^2`, that
   is **x1.60 on edge recall**. **`scripts/kaggle_specs/p4_detsweep_export_f0.json` already exists,
   exports the full local-max superset at sigmoid > 0.5 plus raw logits, and HAS NEVER BEEN PUSHED.**
   The local-max test is threshold-independent, so the whole [0.5, 1.0] curve is a **CPU-only replay**.
   Risk: the 3.5x LB amplification — but that was measured for a node *cut*, not an addition.
4. **TTA aggregation destroys peaks.** `predict_unet_transformer.py:375-388` averages 4 flipped logit
   volumes and divides by 4. **Averaging is a low-pass filter — if the views peak +-1 voxel apart, the
   mean has no strict local maximum and the peak is annihilated.** Replace `/4` with `torch.maximum`
   accumulation, or emit peaks per view and union with a 1.625 um dedup. Union >= any single view by
   construction. **Most under-examined line in the inference path.**
12. **z-depth recall diagnostic (CPU, trivial):** stratify measured Zebrahub recall by depth. If flat,
    the whole depth-dependent-SNR / PSF / restoration branch dies for free.
10. **Temporal attention may be dead weight.** The LB-0.918 rival's plain 4-level UNet3D (5.6 M params,
    NO attention, val_recall 0.5789) matches our best recall 0.563 with 1.5 M params + temporal MHA.
    Ablation: zero the temporal-attention output (identity residual) at inference, re-measure F1 on
    held-out Zebrahub. If F1 moves < 0.01 the attention is dead weight and its parameter budget should
    move into a 4th encoder level. Time helps *association*, not detection — Trackastra and Ultrack both
    put temporal reasoning in the linker.
2+3. **Gaussian heatmap target + Adaptive Wing loss** (one retrain, A/B against one-hot BCE on identical
    crops/epochs, compare **best-over-threshold** F1). PAC-MAP F1 **0.774** vs StarDist-3D **0.640** on
    mouse-brain LSFM; **0.793** vs SAU-Net 0.750 on dense spheroids. AWing is ~30 lines
    (theta=0.5, alpha=2.1, omega=14, eps=1).
11. **Candidate-hypothesis detection into the ILP (Ultrack pattern)** — pass peaks from several
    thresholds as competing node hypotheses with an exclusivity constraint. Converts the threshold from
    a guess into a decision variable. Strategically correct end-state, but only after #1.

**Surveyed and DECLINED, with reasons:** StarDist-3D (needs instance masks, we have points only);
Cellpose/Omnipose (2D-flow compositing noisy on anisotropic data); foundation models (LSFM FM pretrained
on only 1,023 96^3 patches — **we are label-rich with 1.357 M positions; FMs pay off when label-poor**);
Spotiflow stereographic flow (localisation not detection; small for us given 7.13 um NN vs 7 um match).

### D. Sources
[Spotiflow, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02662-x) ·
[spotiflow repo](https://github.com/weigertlab/spotiflow) ·
[Adaptive Wing Loss, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.pdf) ·
[StarDist-3D, WACV 2020](https://arxiv.org/abs/1908.03636) ·
[PAC-MAP, Comput Biol Med 2024](https://www.sciencedirect.com/science/article/pii/S0010482524016469) +
[repo](https://github.com/DeVosLab/PAC-MAP) · [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) ·
[Omnipose, Nature Methods 2022](https://www.nature.com/articles/s41592-022-01639-4) ·
[uSAM, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02580-4) ·
[3D LSFM foundation model](https://arxiv.org/html/2605.26026v1) ·
[Ultrack, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) ·
[Trackastra, ECCV 2024](https://arxiv.org/abs/2405.15700) ·
[DeAbe](https://www.nature.com/articles/s41467-024-55267-x) ·
[Kaggle discussion 737101, rank-2 on node selection](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101)
Code: `train_unet_transformer.py:528-573,626` · `predict_unet_transformer.py:229-251,284-286,334-335,375-388`
· `h1r_zh001r_register.py:17-21` · `p4_detsweep_export_f0.json` · `data/train/*.geff` ·
`data/external/zebrahub/ZSNS001_tracks.csv`

## 2026-08-25 — TRAINING PROTOCOL: tune EARLY layers not the head; and the assert that would have caught our phantom gain

### A. Can a student exceed the Ultrack teacher? SPLIT VERDICT

Ultrack is an **integer-linear-program over segmentation hypotheses** (arXiv 2308.04526, VERIFIED) — a
*deterministic* selection rule, so **its errors are structured and input-determined**, which is the case
where noise-averaging does NOT work.

| label type | volume | expected noise | verdict |
|---|---|---|---|
| nucleus detections | 1,357,051 | low (centroid localisation << the tracking ILP) | **student can exceed** |
| continuation edges | ~1,192,441 | moderate | marginal |
| **division links** | **65,741** | **highest — division topology is the weakest part of every ILP tracker, and it is exactly what our metric charges** | **student will NOT exceed without cleaning** |

**PRESCRIPTION: trust Ultrack's detections, distrust Ultrack's divisions.** Down-weight division links
to ~0.5 and clean them before they touch the loss. **Anyone promising "fine-tune on 1.26 M edges and
beat the teacher" is relying on a mechanism that does not apply to a deterministic ILP teacher.**

Mechanisms that DO transfer: cross-domain wash-out (we must beat Ultrack on the *competition* embryos,
not on ZSNS001 — its Zebrahub-specific tuning is baked into the labels but is not transferable), and
different hypothesis class (we are a learned detector, not an ILP). But Burns et al. (weak-to-strong,
arXiv 2312.09390, VERIFIED) found **plain fine-tuning underdelivers this** — it needed an auxiliary
**confidence** loss letting the student trust its own confident predictions over the weak label.
Counter-evidence to watch: Stanton et al. (arXiv 2106.05945, VERIFIED) — **matching the teacher more
closely does not improve student generalisation**; fidelity and generalisation are distinct axes.

**Kill test T1.1 (<1 GPU-h, go/no-go):** run the deployed 0.915 model on Zebrahub and compare its
detections to Ultrack's labels. **If the disagreements are STRUCTURED (e.g. Ultrack systematically drops
dim/deep nuclei), we are about to train TOWARD a worse teacher.**
**T1.4 (decisive):** after fine-tuning, compare the student against **Ultrack run on the target data**.
If we cannot beat Ultrack on target, we have merely distilled it.

### B. THE COUNTER-INTUITIVE FINDING — tune EARLY layers, not the head

- **Bai et al., PES (arXiv 2106.15853, VERIFIED):** *"the latter layers in a DNN are much more sensitive
  to label noise, while their former counterparts are quite robust."* **Later layers memorise noise
  first.**
- Lee et al., Surgical Fine-Tuning (arXiv 2210.11466, INFERRED): appearance/input shift → tune **early**
  layers; label/output shift → tune **later**.

Ours is appearance shift (different embryo) **and** noisy labels. **Both readings converge: tune the
ENCODER more and the HEADS less. The universal reflex — "freeze the backbone, train the head" — is the
WORST choice under this combination.**

**COROLLARY: do NOT apply layer-wise LR decay.** ULMFiT-style discriminative fine-tuning prescribes
*lower* LR for early layers (arXiv 1801.06146, INFERRED) — the exact opposite. Applying LLRD by habit
pushes the wrong way.

Other arms worth one run each: **BN-affine-only** training (arXiv 2003.00152, INFERRED) — a few thousand
params, near-zero overfit risk, genuinely attractive with one embryo. **LP-FT** (arXiv 2202.10054,
INFERRED) — full FT distorts pretrained features and underperforms OOD; our deployment IS OOD, so adopt
linear-probe-then-fine-tune as the default. **Skip LoRA** — it exists for LLM-scale parameter efficiency;
our whole model (1.5 M params) is smaller than one LLM layer.
**Check first what `TemporalUNet3D` normalises with:** if BatchNorm at 3D batch sizes of 1–2, the running
statistics are near-garbage, which alone may explain both the fragility and the size of the phantom +0.104.

### C. Representation drift — the highest-risk item, with a 1-hour gate

`L_total = L_det + λ·KL(assoc_new ‖ assoc_frozen)`. **The frozen teacher for the association head IS our
deployed 0.915 model** — we have it, and unlimited unlabelled input; cost is one extra forward pass.
**Distil the association head's OUTPUT LOGITS, not trunk features** — feature matching over-constrains the
trunk and blocks the detection gain we are paying for. Add a **hard drift gate**: evaluate the association
head on a fixed frozen batch every N steps, assert AUC stays within ε, kill the run otherwise.

**T3.1 — RUN THIS BEFORE ANY OF THE MACHINERY (<1 GPU-h):** fine-tune detection 200 steps with NO drift
control and measure the association AUC drop. **<1% ⇒ all of the drift apparatus is unnecessary and must
not be paid for. 10% ⇒ the problem is quantified and the machinery is justified.** Either outcome is worth
an hour. PCGrad / GradNorm / uncertainty-weighting do **not** apply out of the box (only one supervised
loss) — they become relevant only after the distillation term exists and only if gradient cosine < 0.

### D. THE ASSERT THAT WOULD HAVE CAUGHT OUR PHANTOM +0.104, ON ITS OWN

**`assert optimizer.state[p]['step'] > 0` for every trainable parameter.** Adam's internal counter
increments only on a real step.

Plus, all cheap and all permanent:
- Count skipped steps: `before = scaler.get_scale(); scaler.step(opt); scaler.update();
  skipped += (scaler.get_scale() < before)`. **Assert skipped/total < 0.05; hard-fail above 0.20.**
- Snapshot `w0`; log `‖p − w0‖₂ / ‖w0‖₂` **per layer group** each epoch; assert max relative delta > 1e-6.
  **Bonus: that table IS the "which layers actually moved" diagnostic for §B, free.**
- **Always run three arms. Arm A — no-op control: LR = 0, model still in `train()` so norm buffers update,
  same forward passes. ANY gain in Arm A is pure AdaBN — this is the arm that would have caught +0.104.**
  Arm B — real fine-tune with all norm layers in `eval()`. Arm C — the actual run.
  **Report `gain(C) − gain(A)`.**
- Log pre-clip global grad norm every step (assert finite and > 0; `inf` on 100% of steps is the
  fp16-overflow signature) and `scaler.get_scale()` (monotone decay to floor = permanent overflow).
- **Overfit-one-batch gate:** one crop, 200 steps, assert train loss < 10% of initial. **If the model
  cannot overfit one batch, no gain from the full run is real.** Two minutes of GPU.
- Root cause fix: T4 is Turing sm_75 — **no efficient bf16**, so fp16 is forced and the guards are
  mandatory. Mitigate with `GradScaler(init_scale=2**10)`, loss in fp32, and hunt the overflow source
  (usually an exp/softmax in the temporal attention, or a divide-by-small-count in the detection loss).
- AdaBN (arXiv 1603.04779, INFERRED) **is a real, free effect** — it must be run as a separate, explicitly
  labelled baseline arm every time, never folded into a fine-tuning result.

### E. Validation and selection — never select on Zebrahub

Crop-modulo over one embryo at the same timepoints is the textbook **spatial-autocorrelation leak**
(Roberts 2017, doi:10.1111/ecog.02881, INFERRED). Ranked, worst → best predictor of cross-embryo transfer:
(1) crop-modulo *(what we have — predicts essentially nothing)*; (2) **spatially blocked with a buffer
shell ≥ max displacement + nucleus diameter, ≥10 voxels at 1.625 µm**; (3) **temporal holdout** train
t=0..13 / val t=16..19 with a 2-frame gap; (4) anatomical-region holdout.
**Run the 2×2 of (2)×(3); the gap between crop-modulo and blocked+temporal IS our leakage estimate** —
that number is a deliverable in its own right.

**HARD RULE: never select a checkpoint on Zebrahub validation** — it is a training-health monitor only.
Select on the target objective: export `.geff`, score against `data/train` with the patched scorer via
`scripts/core/score_oof.py`, pooled, **both embryo directions separately**. Evidence: the rival's
`epochs=120 / epoch_best=8 / val_recall=0.5789` was selected on same-domain validation, and mikelou1's
from-scratch retrain landed at **0.928** — at the public frontier, not above it.
**Do not select on the LB either** — three of our graphs scored exactly 0.915.

**WiSE-FT is the best compute-to-candidate ratio available to us** (arXiv 2109.01903, INFERRED):
interpolate in weight space, `θ_α = (1−α)·θ_deployed + α·θ_ft`, α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}.
**One training run yields ~10 candidates for the price of ~10 CPU scorer runs.** Expect the optimum at
α < 1. **Kill criterion: if best α = 0, the fine-tune added nothing — log to `failed-experiments.md`
and stop.**

### F. 1.36 M nuclei is A LITTLE, not a lot

The *effective* sample size is 1 embryo × 1 developmental stage × 20 timepoints — independent units ≈ 20,
arguably 1. Nuclei within a frame are near-duplicates for a detector (same illumination, PSF, stage,
chromatin appearance). **Our risk is overfitting to one embryo's appearance, not underfitting.**
Consequences: spend the augmentation budget on **intensity / noise / blur / PSF**, not more crops; and
**one additional Zebrahub embryo is worth more than 10× more crops of ZSNS001.** Heavy input noise is also
exactly what Noisy Student (arXiv 1911.04252, VERIFIED) requires for a student to exceed its teacher —
two independent arguments converging on the same action.

### G. Citation ledger
**VERIFIED this session (abstract-level, not full text):** arXiv 2312.09390 · 2007.00151 · 1804.06872 ·
1911.00068 · 2106.05945 · 1911.04252 · 2106.15853 · 2308.04526
**INFERRED (recalled, URL given, NOT re-read — verify before quoting):** arXiv 2210.11466 · 1603.04779 ·
2003.00152 · 2106.09685 · 2202.10054 · 1801.06146 · 1612.00796 · 2001.06782 · 1711.02257 · 1705.07115 ·
1606.09282 · 2109.01903 · 2203.05482 · 2207.07048 · Roberts 2017 doi:10.1111/ecog.02881
**Method caveat:** the session's WebSearch budget was exhausted, so verification was by direct WebFetch
against arXiv abstract pages — sufficient for the attributed claims, but methods sections were not read.

**Per the standing rule, all of the above is THESIS, not result.** Stage 0 — overfit-one-batch gate,
T3.1 drift measurement, Arm-A no-op AdaBN control, T1.1 teacher-noise audit — is **~2–4 GPU-h and
converts the three highest-risk theses into measurements. Any one of them can cancel the project cheaply.**

## 2026-08-25 — EXPERIMENT: M1 (duplicated-source LAP) is DEAD; divisions need k=5, not k=3

Two CPU-only experiments, pre-registered, run on fold 1
(`C:/temp/preilp_f1_v2/preilp_split1.parquet`, `C:/temp/error_atlas/*_pre1.parquet`).
Scripts kept outside the repo; the reusable half is now
`scripts/win_bet/candidate_gate_eval.py` (8 contract tests).

### A. M1 IS DEAD — killed analytically, not marginally

M1 proposed augmenting `motion_relink_edges`' cost matrix with duplicate rows for
division-eligible mothers so one mother could win two targets, making the contested
daughter reachable. **Strict precondition:** `linear_sum_assignment` can only choose among
candidates present in the matrix, so M1 needs the TRUE MOTHER to be a candidate parent of
the contested daughter.

**P1 — the candidate set gives every target EXACTLY ONE parent:**

| candidate parents per target | count |
|---|---|
| **1** | **2,162,040** |
| 2+ | **0** |

335,210 of 2,497,250 eligible targets (13.42%) get **zero** candidates.

**P2 — 125 GT divisions, 123 mothers mapped, 236 mapped daughter rows:**

| daughter state | count | share |
|---|---|---|
| true mother IS the candidate | 119 | 50.4% |
| **contested** (a candidate exists, but not the mother) | **76** | **32.2%** |
| zero candidates | 41 | 17.4% |
| **contested daughters where the mother was available** | **0** | **0%** |

**Since each target has exactly one candidate, a contested daughter's sole candidate is BY
DEFINITION the wrong parent — the true mother is never in the cost matrix. No augmentation
of the assignment problem can select a candidate that was never nominated. M1 CANNOT WORK.**
This is stronger than the pre-registered "<8 of 19 flip ⇒ dead" criterion: the mechanism is
impossible, not merely unprofitable. **Do not build it.**

**Cross-check:** 76 + 41 = **117/236 = 49.6%** of division daughters need a nomination-stage
fix — reproducing the independently-derived "49.6% of division-daughter links are
un-nominated" to the decimal, from an entirely different computation.

**Consequence: candidate widening at NOMINATION is not *a* path to division recovery, it is
the ONLY one.** Everything at or below the relink is downstream of a set that already
committed to one parent per target.

### B. DIVISIONS NEED k=5, NOT k=3 — a uniform gate under-serves them

Containment of the true parent within the child's k nearest frame-(t−1) predicted nodes:

| k | **division daughters (n=117)** | continuations (n=16,884) | candidate-set size |
|---|---|---|---|
| 1 | 31.6% | 24.6% | 1.16x |
| 2 | 57.3% | 80.4% | 2.31x |
| **3** | **75.2%** | **92.7%** | 3.47x |
| **5** | **93.2%** | 97.7% | 5.78x |
| 10 | 97.4% | 99.6% | 11.55x |
| 20 | 99.2% | 99.95% | 23.1x |

**At k=3 — the previously recommended value — continuations reach 92.7% but divisions only
75.2%, a 17.5-point gap.** A uniform k=3 gate would systematically under-serve exactly the
class carrying the largest untouched metric headroom (+0.0993 division oracle).
**Recommend k=5**, or an asymmetric/adaptive k that spends the extra breadth only where a
division is plausible.

**Mechanism (consistent, not merely fitted):** after a division the mother sits roughly
equidistant from BOTH daughters with other cells intervening, so she is rarely the nearest
neighbour of either. **Divisions are intrinsically a higher-k problem.** Note the control at
k=1 (24.6%) reproduces the prior independent measurement of 24.7%.

**Cost:** k=5 is a **5.8x** candidate set. That is a real ILP cost and it interacts with the
appearance-weight blocker — with `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"` the solver pays
nothing to leave a target unlinked, so **a wider gate is NOT pruned; every feasible
candidate is accepted.** Restoring appearance weight remains a precondition.

### C. NEW TOOL — `scripts/win_bet/candidate_gate_eval.py`

Built because this session hand-rolled the same containment measurement twice. Given a
pre-ILP export and the error atlas it reports, for any k: containment **split by
continuation vs division**, the in-degree histogram, the count needing a nomination fix, and
the candidate-set size multiplier. **It always reports the class split**, because pooling
hides the division gap that is the whole point. The docstring carries the two standing
caveats (one-parent-per-target; appearance weight 0.0 ⇒ the solver does not prune) so the
next reader cannot mistake an oracle ceiling for a predicted score.
8 contract tests in `tests/test_candidate_gate_eval.py`, including that the µm scale is the
deployed geometry and that unevaluable pairs are skipped rather than silently counted as
misses.

## 2026-08-25 — THE BIJECTION DESTROYS 29 REAL DIVISIONS; but the two-population fix FAILS its pre-registered bar

Three CPU-only experiments on fold 1 (`C:/temp/preilp_f1_v2/preilp_split1.parquet`,
`C:/temp/error_atlas/*_pre1.parquet`, 128 crops, 125 GT divisions). Nothing in the repo modified.

### A. 29 of 125 GT divisions are ALREADY in the candidate set and the 1:1 relink destroys them

The candidate set is NOT a matching. Source out-degree:
`{1: 1,795,844, 2: 165,148, 3: 10,911, 4: 715, 5: 59, 6: 2}` — **176,835 sources carry a division
hypothesis**, and `motion_relink_edges` solves a `linear_sum_assignment` **bijection**, forcing
out-degree <= 1 and destroying **189,361 excess edges**.

| GT division state | count | share of 125 |
|---|---|---|
| **BOTH daughters nominated to the true mother** | **29** | **23.2%** |
| exactly one daughter nominated | 57 | 45.6% |
| neither nominated | 27 | 21.6% |
| unmapped | 12 | 9.6% |

**29 real divisions are fully formed in the candidate set and are thrown away by the relink** —
and the deployed pipeline then tries to bolt divisions back on afterwards by hunting for orphans.
We currently emit **5 division TPs across 199 crops**. This is a strictly worse route to the same
goal, and it is a post-processing fact — no nomination change, no retraining.

### B. But P9's OWN gates reject 22 of those 29

Applying the shipped P9 gates (radii 8.0 / 11.0 / 10.0, C1 mid-track, C3 t+2 divergence >= 2.25)
to the 29:

| outcome | count |
|---|---|
| **PASS** | **7** |
| C3: no unique successor | 9 |
| radius_parent (> 8.0 um) | 8 |
| radius_sister (> 11.0 um) | 4 |
| C3: no divergence | 1 |

**P9's gates — which scored +0.010 — reject 76% of the true divisions that are already fully
nominated.** They were calibrated against ORPHAN proposals, where the prior that a pair is a real
division is ~1:2400. For the already-nominated population the model has already committed both
daughters to this mother, so the prior is far higher and the same radii are miscalibrated.

### C. TWO-POPULATION HYPOTHESIS — TESTED AND FAILED ITS PRE-REGISTERED BAR

**Pre-registered bar: >= 20 of 29 TP at an admitted count within the range P9 already tolerates
(P9 ships 406 divisions).** Sweep over the already-nominated population (29 true; 176,835
multi-child sources as the FP pool):

| C3 successor test | parent <= | sister <= | TP / 29 | sources admitted |
|---|---|---|---|---|
| ON | 8.0 | 11.0 | **7** | 9,263 |
| ON | 10.0 | 14.0 | 10 | 10,449 |
| ON | 15.0 | 18.0 | **10 (ceiling)** | 10,905 |
| OFF | 8.0 | 11.0 | 17 | 135,519 |
| OFF | 8.0 | 14.0 | **21** | 137,638 |
| OFF | 10.0 | 14.0 | 28 | 150,723 |
| OFF | 12.0 | 14.0 | **29** | 154,409 |

**VERDICT: FAILED.** The only settings reaching >= 20 TP admit **137,638+ sources — 339x more than
the 406 divisions P9 ships.** No setting meets the bar. **Do not build the two-population gate as
specified.**

**The diagnostic that survives, and it is the useful part: C3, NOT the radii, is the binding
filter.** With C3 ON, TP ceilings at **10 of 29 no matter how wide the radii go** (8->15 um parent,
11->18 um sister changes nothing beyond 10). With C3 OFF at the SAME 8.0/11.0 radii, TP jumps
**7 -> 17**. So the thing rejecting already-nominated true divisions is C3's requirement that
**both daughters have exactly one successor at t+1** — a harsh structural demand in a candidate set
where 13.42% of targets have zero parents at all. It is a candidate-sparsity artifact being read as
a biological signal.

### D. Honest caveats
- **Fold 1 only** (125 GT divisions); pooled GT is 151.
- **C2 (mutual-nearest-orphan) was NOT applied** — it is P9's strongest precision filter, so every
  "admitted" figure is an UPPER bound on FP. TP counts are unbiased.
- "Admitted" counts SOURCES with >= 1 qualifying pair, not final emitted divisions. P9's global
  frac cap (0.00375 x edges ~ 8,100 here), frame cap and score-ordered selection would truncate
  heavily. **Whether the 21 TP survive score-ordering into the top ~8,100 of 137,638 is UNTESTED
  and is the one live question that could revive this lane.**
- 97.95% of predicted forks are never charged (a fork only costs on an annotated GT cell with >= 1
  GT child), so raw FP counts overstate metric damage — but not by the ~339x needed here.

### E. What this changes
1. **M1 (duplicated-source LAP) stays dead** — for CONTESTED daughters the true mother is not a
   candidate at all (0 of 76).
2. **The bijection is a real and quantified loss** (29 divisions, 23.2% of GT) — recorded as a
   mechanism, not yet as a lever.
3. **C3's unique-successor clause is the highest-value thing to re-examine in the shipped P9
   design.** It is cheap to test: the same sweep with C3 divergence retained but the
   unique-successor requirement relaxed to "at least one successor".

## 2026-08-25 — THE BIJECTION FORBIDS DIVISIONS. divJ ceiling is ZERO. +0.0232 is structurally unreachable.

**This supersedes the standing "+0.0012 perfect solver" figure, which was computed UNDER the
out-degree <= 1 constraint and therefore never bounded this.**

### Verified in the DEPLOYED artifact, not the mirror

`notebooks/kaggle_p9_coupled_division/biohub-p9-coupled-division.ipynb` contains **two bare
`linear_sum_assignment(cost)` calls** — no block matrix, no initiation/termination dummy rows.
Both association stages are BIJECTIONS. (The mirror `src/biotrack/wrapper.py:343,:522` agrees, but
the mirror is not load-bearing; the notebook is.)

### The measurement (fold 1, 128 crops, 109,057 GT edges, 125 GT divisions)

```
reachable (true parent IS the target's nominated candidate) : 81,055 / 109,057  (74.32%)
reachable children per source                               : {1: 80,997, 2: 29}

O1  out-degree <= 1 : 81,026 edges  ->   0 complete division events
O2  out-degree <= 2 : 81,055 edges  ->  29 complete division events

divJ CEILING   out-deg<=1 : 0.0000
divJ CEILING   out-deg<=2 : 0.2320
SCORE delta from the division term alone : +0.0232
```

**Under the bijection the SOLVER'S division Jaccard ceiling is literally ZERO.** A division event
requires out-degree 2; `linear_sum_assignment` forbids it by construction. Therefore **100% of the
divisions we emit come from the post-processor bolt-on, and the solver can never contribute one at
any parameter setting.** This is a feasibility-set failure, not a modelling failure — it explains
every null division result in the ledger (divfix 0.000, ILP division weight 0.3/1.0/2.0/3.0 all
0.915 on someone else's LB) without appealing to any of them.

**The bijection buys almost nothing in exchange:** lifting it changes the adjacency term by
**29 edges out of 81,055**. It forfeits the entire division term to save nothing.

**Realistic gain:** current divJ ~0.0152 -> ceiling 0.232 is **+0.1 x (0.232 - 0.0152) = +0.0217**
of SCORE. Our gap to top-3 is +0.028.

### Why the current bolt-on cannot rescue it — measured the same day

Under P9's real global cap (0.00375 x 2,162,040 = **8,108**) and its shipped ordering
`score = parent_dist + 0.15 * sister_dist` ascending:

| C3 variant | radii | proposals | emitted | **TP in cap** | est divJ |
|---|---|---|---|---|---|
| A (shipped) | 8.0 / 11.0 | 9,263 | 8,108 | **4** | 0.0137 |
| A | 10.0 / 14.0 | 10,449 | 8,108 | 3 | 0.0103 |
| B (>=1 successor) | 8.0 / 11.0 | 11,397 | 8,108 | 2 | 0.0069 |
| C (abstain on missing successor) | 8.0 / 11.0 | 90,836 | 8,108 | **0** | 0.0000 |
| D (no C3) | 10.0 / 14.0 | 150,723 | 8,108 | **0** | 0.0000 |

**Loosening the gates makes it WORSE, not better.** True divisions are not the geometrically closest
pairs, so ranking by `parent_dist + 0.15*sister_dist` buries them: at 90,836 proposals against a
cap of 8,108, **zero** true divisions survive. The shipped variant A wins this comparison — which
validates P9's tightness as a local optimum **of a fundamentally wrong architecture**.

### THREE stacked structural failures, all now measured

1. **The bijection forbids out-degree 2** -> solver divJ ceiling 0.
2. **The bolt-on ranks by a non-discriminative score** -> true divisions are buried by distance.
3. **The global cap truncates at 8,108** -> whatever survives 1 and 2 is cut anyway.

Fixing any one alone does not help. Fixing 1 makes 2 and 3 irrelevant, because a division-aware
global solve chooses jointly instead of propose-rank-cap.

### External corroboration (VERIFIED by agent, quantitative)

The Cell Tracking Challenge ran a **Cell Linking Benchmark** (ISBI 2024) — a linking-only benchmark
holding segmentation FIXED and swapping only the linker (https://celltrackingchallenge.net/ctc-vii/,
results spreadsheet http://public.celltrackingchallenge.net/documents/CellLinkingBenchmark.xlsx,
parsed). Over 13 datasets, linker swap alone spans **LNK 0.070 / BIO 0.189**; on
**Fluo-N3DH-CE**, a dense dividing 3D embryo — the closest analogue to our data — the spread is
**LNK 0.1313 and BIO 0.449** (PAST-FR 0.9822/0.8619 vs KTH-SE 0.8510/0.4128).
**The division-sensitive measure has ~2.7x the spread of the pure linking measure: linker choice
dominates specifically on divisions.** The winner, PAST-FR, is Institut Pasteur — the TrackMate
group — running a **two-stage LAP with explicit gap-closing and SPLITTING blocks**.
So a +0.03-0.05 gain from swapping only the linker is the documented norm, not an extraordinary
claim. It also makes TWEAK's unanswered "refines the graph or replaces the method?" legible: the
answer is both, and saying so gives the method away.

### THE ARCHITECTURE TO BUILD

Two-stage LAP (Jaqaman / TrackMate form, https://imagej.net/plugins/trackmate/trackers/lap-trackers):
- **Stage 1, frame-to-frame:** solve the `(n+m) x (n+m)` BLOCK matrix rather than the bare `n x m` —
  top-left link costs, top-right diagonal **termination**, bottom-left diagonal **initiation**,
  bottom-right transpose block. This alone legalises the **13.42% of targets that currently get zero
  candidates**: they become track starts instead of corrupting the assignment.
- **Stage 2, segment linking:** a second LAP over segment endpoints with a **gap-closing** block, a
  **SPLITTING** block (segment start -> interior spot of another segment: the ONLY operation that
  creates out-degree 2), a **merging** block set to infinity (biologically invalid for cells, and it
  will otherwise eat divisions), and diagonal **rejection** blocks at alternative cost
  `1.05 x max(C)` — data-derived, which is why it works UNTUNED.
- Pure `scipy.optimize.linear_sum_assignment`. No weights, no internet, no new dependency — it drops
  straight into a Kaggle kernel.

Equivalent alternative: min-cost flow with coupled division arcs
(https://pubmed.ncbi.nlm.nih.gov/20864383/).

### Composition with candidate widening
The 0.232 ceiling counts only the 29 divisions where BOTH daughters are already reachable. The other
96 are not nominated. **kNN widening at k=5 (measured: 93.2% division-daughter containment vs 75.2%
at k=3) raises the reachable count, and the division-aware solver then claims it.** The two levers
compose: widening raises the ceiling, the solver reaches it. Neither works alone — widening without
a division-aware solver is a no-op (the ILP re-selects the same argmax and appearance_weight = 0.0
means it never prunes), and the solver without widening caps at 0.232.

### Caveats
Fold 1 only (125 GT divisions; pooled GT is 151). The 0.232 ceiling is an upper bound a real solver
would not fully attain. divJ 0.0152 is taken from the existing ledger, not re-measured here.

### Not adopted: napari-chatgpt
Royer lab's LLM-driven napari assistant (https://github.com/royerlab/napari-chatgpt) is an
interactive GUI agent. We run no napari, have no interactive step, and already have the LLM layer.
**Recorded as declined.** The load-bearing Royer-lab artifact for us is **Ultrack**, which implements
exactly the two mechanisms this entry proves we lack: multi-hypothesis candidate generation and an
ILP with **division arcs** rather than a bijection.

## 2026-08-25 — THE BLIND SPOT: we could see 9.5% of the public corpus. All three mechanisms we lack are in the 90.5%.

### A. Quantified blind spot (host's instinct, confirmed)

**719 unique public kernels** (not ~900), enumerated exhaustively across 4 sort orders x 8 pages, all
converging. Saved at `C:/tmp/nb/all_kernels.csv`.

| | count | share |
|---|---|---|
| title contains a score/LB token | **68** | **9.5%** |
| title has NO score token | **651** | **90.5%** |

**Every score-keyed search this project has run could see under one tenth of the corpus** — and the
score-titled tenth is almost entirely forks of the same 0.926/0.927 lineage. **All three
mechanism-bearing notebooks below have NO number in their titles.** The frontier is in the 9.5%;
the mechanisms are in the 90.5%.

### B. FIND 1 — `beicicc/biohub-exp041-vmerckle-relink-division-slot` (4 votes, 2026-07-07)

**Raw code for an out-degree-2 relink — the exact mechanism our oracle proved we need.**
Confirmed ABSENT from our `notebooks/` and `src/` (grep, 0 matches). ~48 lines, self-contained,
`exp041.flat.py:1730-1776`. Runs AFTER the Hungarian pass, over `unmatched_targets`, letting a
source that holds exactly one match reserve a SECOND daughter — upstream of gap recovery, not a
post-hoc geometric add.

```python
score = -prob + 0.015 * raw + 0.006 * sister + 0.004 * motion
```

**THE CRITICAL DIFFERENCE FROM OUR P9 BOLT-ON: this ranks by PROBABILITY with distance as a small
tiebreak. P9 ranks by `parent_dist + 0.15*sister_dist` — pure geometry.** Our own capped experiment
measured why that matters: under P9's cap and distance ordering, **zero** true divisions survive out
of 90,836 proposals, because true divisions are not the geometrically closest pairs. A
probability-led score is the discriminative signal we have been missing.

**It targets exactly the population our oracle identified.** The 29 GT divisions with both daughters
nominated to the mother: the Hungarian takes one, the other becomes an unmatched target — which is
precisely what this slot reclaims.

Gates: `MIN_PROB 0.42`, `PARENT_MAX_UM 7.2`, `SISTER_MAX_UM 8.5`, `FRAME_FRAC_CAP 0.006`, one slot
per source. Note the radii are TIGHTER than P9's 8.0/11.0 — the selectivity comes from probability,
not geometry.

### C. FIND 2 — `dalloliogm/biohub-exp227-divergence-mutualnn-wide` (0 votes, 2026-08-23)

Carries three things we lack:
- **Learned local association ranker** (`:2600-2790`): per-candidate feature matrix scored by
  `LOCAL_ASSOCIATION_RANKER.predict_proba`, folded into the LSA cost as
  `0.85*ranker + 0.15*primary`. **This is retrieve-then-rerank** — the primitive the retrieval
  analysis said we need. (It still ends in `linear_sum_assignment`, so it is NOT a multi-parent
  nominator.)
- **KD-tree geometric gating** (`:2528-2534`): `cKDTree(...).query_ball_point(points, r=7.0,
  return_length=True)` → a `density_7um` feature per node per frame.
- **Mutual-NN division gate** (`:3297-3340`).

**PUBLIC WEIGHTS, 18.5 KB:** `pilkwang/biohub-local-association-ranker-unet300-v1` (downloaded,
`C:/tmp/rk/`). `model/local_association_ranker.pt` = 20,283 bytes. MLP hidden=64, dropout=0.05,
best_epoch 7, **best_score 0.9777**, trained on 385,648 rows / 108,876 groups, split by dataset,
22 named features. Author's own constraint, verbatim: *"Use only as a constrained local association
tie-breaker, not as a global edge veto."* A 34-vote carrier of the same ranker exists at
`lonnieqin/biohub-gap2-joint-node-budget`.

### D. CORRECTION — the ILP appearance weight is NOT an oversight. Lever CLOSED.

I queued `BIOHUB_ILP_APPEARANCE_WEIGHT` 0.0 -> 0.1 as a "precondition for candidate widening".
**That was wrong.** `dalloliogm/biohub-exp110-ilp-birth-death-cost` (15 votes) sets it deliberately:
```
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"]    = "0.0"
| ILP appearance cost | 0.0 | Avoid over-penalizing new track starts. |
```
The CODE DEFAULT is 0.1, so 0.0 is a deliberate override — and **all 26 pulled kernels set 0.0**.
(The 1.5/1.55/1.575 values a naive grep finds are `DISAPPEARANCE_WEIGHT` matching as a substring.)
**Our 0.0 is the tuned public consensus. Treat as a closed lever absent a new mechanism.**

### E. The public ceiling is 0.927 and there is nothing above it

`evgendvorkin/biohub-0-927-lb` re-ran 2026-08-24 21:25. Diff vs the 0.926: DeepCenter checkpoint
`checkpoint_last.pt` epoch 500 -> **`best.pt` epoch 2**; `BIDIRECTIONAL_EDGE_WEIGHT` **0.30**;
`BIDIRECTIONAL_FUSION_MODE harmonic_probability`. **The epoch-500 -> epoch-2 DeepCenter swap is the
substantive change.** Its inline comment records `ILP_DIVISION_WEIGHT` 0.3/1.0/2.0/3.0 **all scoring
0.915 on the real leaderboard** — independently closing that knob.

Last-wins audit: 0.927 assigns `SAFE_DIV_MAX_UM` twice (plain `12.0` then env `"8.0"`); the read is
`float(os.environ.get(...))` so **env wins, 8.0/11.0/10.0**, and `DIVERGE_UM` has no env at all →
code default 2.25. `backtracking/biohub-general-v11` assigns
`BIOHUB_MOTION_RELINK_LEARNED_BONUS` twice — `"1.0"` then `"2.0"`; **last wins = 2.0**.

### F. THE CHECK ON OUR DIVISION THESIS — read this before over-investing

mikelou1, 2026-08-18, discussion 735352: *"I trained mine from scratch since my division score is
quite good [0.3] and edge is really bad [~0.01 below public notebooks]"* — and mikelou1 scores
**0.928**. So **divJ 0.30 is demonstrably attainable, and on its own it yields 0.928, not 0.947.**
INFERRED but well-supported: **the >=0.947 cluster is winning on EDGE, not on division.**

This does NOT retire the division work — our own oracle says the bijection forfeits +0.0232 for
almost nothing (29 edges of 81,055) — but it caps the expectation. **Divisions are necessary, not
sufficient.** The edge term (oracle headroom: FN->0 is +0.167) is where the top of the board lives.

### G. Mechanisms NOT found anywhere in 719 kernels — searched SOURCE, not titles

- **Multi-parent nominator (>1 candidate parent per target): NOT FOUND.** Every kernel terminates in
  `scipy.optimize.linear_sum_assignment`, a strict bijection. Out-degree 2 is only ever reached by a
  post-assignment second-daughter add (exp041) or a post-hoc geometric add (the 0.926 lineage).
- **Dustbin / no-parent / abstention column: ZERO hits** for
  `dustbin|no_parent|null_column|abstain|background_class|no_match` across every kernel pulled.
- **Learned division classifier: NOT FOUND.** All division logic is geometric gates or mutual-NN.

**So the entire public field shares our structural ceiling.** Nobody has a multi-parent nominator and
nobody has abstention. That is simultaneously the best news of the sweep — the ceiling is not
knowledge we lack, it is a mechanism nobody has built — and the reason the frontier is stuck at 0.927.

### H. Order to run
1. **exp041 second-daughter slot** — ~48 lines, no new weights, no new dataset, probability-led
   scoring, directly attacks the measured `{1: 2,162,040}` in-degree histogram and the 29.
2. **exp227 local association ranker** — 20 KB public weights, needs the 22-feature port. Higher
   ceiling, higher integration cost. Honour the author's tie-breaker-not-veto constraint.
3. ~~ILP appearance weight~~ — CLOSED, see D.
4. Do not chase `flexonafft/biohub-harmonic-fusion` or the backtracking forks — 0.926 clones with
   single-knob deltas.

## 2026-08-25 — END-TO-END: the bijection costs +0.0018 in ADJACENCY, not divisions. Blanket relaxation is HARMFUL.

Three graphs built from the SAME pre-ILP candidate set (fold 1), differing ONLY in the out-degree
rule, each scored by the OFFICIAL patched scorer on 25 crops vs `data/train`. Because in-degree is
exactly 1 for every target, the frame assignment degenerates and the out-degree cap is the only
variable — so the deltas are cleanly attributable.

| graph | rule | edges | SCORE | adjJ | divJ | div TP / FP |
|---|---|---|---|---|---|---|
| **G1** | out-degree <= 1 (the deployed bijection) | 1,972,679 | **0.5944** | 0.5944 | 0.0000 | **0 / 0** |
| **G2** | out-degree <= 2, BLANKET (top-2 by edge_prob) | 2,149,514 | **0.5863** | 0.5860 | 0.0027 | 6 / 2,160 |
| **G3** | out-degree <= 2, gated exactly as exp041 | 1,988,076 | **0.5962** | 0.5962 | 0.0000 | 0 / 370 |

(Absolute scores are low because these are RAW candidate graphs with none of the deployed
post-processing — gap closing, short-track filter, safe divisions. Only the deltas are meaningful.)

### THREE RESULTS, ALL PRE-REGISTERED

1. **The bijection emits literally ZERO forks** — G1 divJ 0.0000 with **FP=0**, i.e. not one
   out-degree-2 node exists. This is the direct end-to-end confirmation of the oracle: the solver's
   division ceiling is not low, it is *zero*.
2. **Blanket out-degree <=2 is HARMFUL: -0.0081.** It buys 6 division TPs and pays 2,160 division FPs
   plus **-0.0084 of adjacency**. **The oracle's +0.0232 is a ceiling under PERFECT selection; naive
   relaxation lands far below zero.** Taking the second-best child unconditionally adds a wrong edge
   for most of the 176,835 eligible sources. This is the single most important calibration of the
   division lane and it kills any "just allow out-degree 2" proposal.
3. **exp041's gating PASSES its bar: G3 = 0.5962 vs G1 = 0.5944, +0.0018** — with 130,832 candidates
   passing the gates and only **15,397** admitted after the 0.6% per-frame cap.

### THE SURPRISE — the gain is ADJACENCY, not divisions

G3 recovers **zero** true divisions (TP=0, divJ 0.0000, same as G1). Its entire +0.0018 is
`adj_edge_jaccard`. **The bijection's real cost on this substrate is CORRECT CONTINUATION EDGES that
it rejects when two targets compete for one source** — not mitosis. The second-daughter slot is
mis-named for what it actually buys us.

Caveat on the division reading: with TP=0 in both arms, divJ = 0/(0+FP+FN) = 0 either way, so G3's
370 division FPs are currently free. **They would NOT be free once TPs exist** — a future arm that
recovers real divisions must re-measure this, because 370 FPs would then dilute divJ hard.

### WHAT THIS MEANS FOR THE RANKED PLAN
- exp041's second-daughter slot is a **real but small** lever (+0.0018 on a raw substrate, 25 crops,
  fold 1), and it works through a different mechanism than its author's framing implies.
- The division lane still needs a *selector*, not a *relaxation*. The oracle ceiling (+0.0232) stands,
  but nothing yet reaches it: blanket relaxation is -0.0081, exp041 gating gets +0.0018 with no
  divisions at all.
- Consistent with the sweep's mikelou1 datapoint: **divJ 0.30 yields 0.928, so divisions are
  necessary but not sufficient; the >=0.947 cluster wins on EDGE.** G3 gaining purely on adjacency
  points the same way.

## 2026-08-25 — Local association ranker VENDORED and locally validated (build in progress)

`scripts/kaggle_edits/ranker_block.py` (425 lines, 7 contract tests in
`tests/test_ranker_block.py`). Extracted verbatim from the public notebook
`dalloliogm/biohub-exp227-divergence-mutualnn-wide` (flattened lines 322-703), loading the public
CC dataset `pilkwang/biohub-local-association-ranker-unet300-v1` (18.5 KB).

**VERIFIED locally on CPU 2026-08-25:** loads the real checkpoint, reports `input_dim` 22 and 22
feature names, returns finite probabilities in [0,1] on probe rows. Architecture from the state dict:
`net.0 Linear(22,64)` -> norm -> `net.4 Linear(64,32)` -> `net.6 Linear(32,1)`; the checkpoint also
carries `median`/`mean`/`std` for its own feature normalisation, so it is fully self-contained.
Manifest: `best_epoch 7`, `best_score 0.9777`, 385,648 rows / 108,876 groups.

**The integration, verbatim from exp227 (`:2547-2551`):**
```python
evidence = 0.85 * ranker_matrix[i, col] + 0.15 * primary_matrix[i, col]
cost[i, col] = motion_dist[i, col] + 0.05 * raw_dist[i, col] - MOTION_RELINK_LEARNED_BONUS * evidence
```
It REPLACES the bare edge probability in the relink cost with a blend. Ours uses bare `prob`.
So the patch is a one-line swap at a cost line we already have.

**WHY THIS LANE:** our measured bottleneck is the EDGE term, not divisions — oracle headroom
FN->0 is **+0.167** vs **+0.0993** for divisions, and mikelou1 reached **divJ 0.30 and still only
scores 0.928**. This ranker is the *rerank* half of the retrieve-then-rerank primitive that Ultrack
and Linajea both implement and that no kernel in the 719-notebook corpus has.

### TWO CONSTRAINTS RECORDED IN THE MODULE DOCSTRING AND ENFORCED BY TESTS

1. **Author's own limit, verbatim:** *"Use only as a constrained local association tie-breaker, not
   as a global edge veto."*
2. **LOCAL EVALUATION IS CONTAMINATED.** Manifest: `datasets_seen: 199`, `split_column: "dataset"`,
   `val_fraction: 0.15` — trained on ~169 of the 199 competition TRAINING movies, so it has seen most
   of `data/train`. **Any local score using this ranker is inflated and must NOT gate promotion.**
   It is legitimate for the leaderboard (the hidden set is not among the 199), so **the only honest
   instrument for this lever is a submission slot.** Same leak pattern `leevvin` documented for the
   public support pack. `tests/test_ranker_block.py` fails if this caveat is ever removed from the
   docstring.

**REMAINING FOR THE KERNEL:** the feature-extraction call site. 20 of the 22 features are computable
from the pre-ILP export (probs, degrees, frame counts, KD-tree `density_7um`, candidate rank/count,
the six displacement terms, `has_prev`/`has_next`, `best_next_prob`, `t_norm`); the two motion
features (`motion_dist_um`, `motion_gain_um`) need the flow model's predicted position, which exists
only inside the kernel. exp227's `_ranker_feature_aliases_from_semantics` (already inside the
vendored block) builds the alias dict; what still needs porting is the ~100-line context builder at
exp227 `:2450-2535`.

## 2026-08-25 — FORUM BREAKTHROUGH: over-detection is nearly free, and ONE SLOT reads our exact adj_edge

Read via the Kaggle SDK's undocumented `discussion_api_client.get_topic` / `list_comments`. The CLI
`forums topics list --competition` 403s and the `-s` fallback silently caps at 20 and omits threads;
WebFetch returns only page titles on this SPA. **34 threads confirmed to exist, 21 read in full.**
Recipe + harvested JSON: `%TEMP%\kbatch.py`, `%TEMP%\klist.py`, `%TEMP%\threads\*.json`.
Rate limit ~100 rapid calls then 429; 6 s spacing is stable.

### A. THE METRIC PAYS FOR RECALL AND BARELY CHARGES FOR OVER-DETECTION

Luka Duvanov, discussion/733877, 2026-08-08, VERIFIED verbatim:
> "Over-detection is almost free. A predicted node that matches no ground-truth node is never a false
> positive; the whole cost is the adjusted-Jaccard term. The line is exactly **1 - 0.1 x
> over-prediction**, so 10% more nodes has to buy only a 1% relative gain in edge Jaccard to break
> even. Recall on detection is worth far more than precision here, and I think most peoples
> thresholds are too high."

> "**Duplicating a detection costs ~9% and buys nothing.** Node matching is a one-to-one bipartite
> assignment, so a twin one voxel away matches nothing. If you take the union of two models
> detections without a merge pass, you pay this in full."

Mendrika Ramarlina, discussion/734604, +5:
> "**Edge recall ~ node recall^2 x conditional linking accuracy**"

**This is the quantitative rule behind our own measurement** that detection F1 is monotone decreasing
in threshold with the optimum at or below p0.50 (recall 0.445 -> 0.563 at p0.5). Node recall enters
edge recall SQUARED while over-prediction is taxed only 0.1x linearly.
**`scripts/kaggle_specs/p4_detsweep_export_f0.json` is built and has NEVER been pushed.**
Mandatory companion: a merge/NMS pass, or near-duplicates cost the full ~9% for nothing.

Corroborating, Alan Thanickal discussion/724917:
> "raising the detection threshold to cut nodes 17% costs ~0.18 edge Jaccard"

Counter-signal (INFERRED, linker capacity not detector): Moawiz — "the detector is finding the real
cells, but it is also producing an enormous junk candidate pool. The linker then has to choose among
those candidates which ends up with wrong linking." **So the recall push pays only if the linker can
reject junk.** That is exactly what the local association ranker is for. The two levers are coupled.

### B. ONE SLOT READS OUR EXACT adj_edge — no modelling work

Arul Prasad S P, discussion/734192, VERIFIED:
> "summarise() **drops the division term entirely when a submission contains no divisions at all**
> (score = edge_jaccard if not has_divisions). So a fork-free submission returns your pure adjusted
> edge Jaccard, and the division term follows by subtraction."

mikelou1's decomposition: **0.928 = adj_edge 0.898 + 0.030 (divJ 0.30)**, and "my edge algorithms are
really bad and has ~0.895 on actual lb and 0.919 CV" — **a -0.024 CV->LB gap**, about the whole
distance we need.
**A 0.953 team with mikelou1 divisions needs adj_edge ~0.923. The frontier is an EDGE race,
arithmetically now, not by inference.**
**ACTION: a fork-free submission is the cheapest decisive diagnostic we have.**

### C. INDEPENDENT PUBLIC CONFIRMATION OF OUR BIJECTION RESULT

Luka Duvanov, discussion/733877, VERIFIED:
> "A Hungarian linker forfeits the whole division term by construction. One successor per cell means
> no node ever has two outgoing edges, so **division Jaccard is exactly 0.000 - in my test, 40 FN out
> of 40** - and 0.1 of the available 1.1 is gone before the tracker sees an image."

Measured independently, in public, on 2026-08-08. Our G1 arm reproduced it exactly (divJ 0.0000,
FP=0). Two independent derivations of the same structural fact.

### D. HOST RULINGS AND DATA FACTS

- **ZEBRAHUB IS EXPLICITLY CLEARED.** Thibgolds (organizer), discussion/734330, 2026-08-13:
  "Yes you are free to use the data and all resources in Zebrahub for this competition! **There is no
  overlap with the test set.**" Green-lights the entire `scripts/win_bet/` Zebrahub workstream and the
  1.26M GT association edges — for the term that actually separates the frontier.
- **No embryo overlap train/test** (Thibgolds, discussion/716793); public LB ~29% of test, private 71%.
- **The four visible clips are a placeholder only** (Thibgolds, discussion/723921) — independent host
  confirmation of our standing rule.
- **GT edges span exactly one frame.** An edge between non-consecutive frames is structurally
  unmatchable and scores 0.0 with no diagnostic (Diana Daher, discussion/728613).
  **Never stride frames — budget a contiguous block from t=0.**
- **Voxels are 1.625 x 0.40625 x 0.40625 um, so a radius expressed in VOXELS is 4x wrong in z**
  (Luka Duvanov, discussion/733973). Our `candidate_gate_eval.py` uses um and a test pins the scale.
  Same thread: displacement median 1.82 um, p95 5.34, p99 8.38 - an 8.4 um NN radius reaches 99% of
  true links. **151 divisions total, one link in 853, and 112 of 199 movies contain NONE** - so
  division CV on random splits is noise.
- **Frozen frames: 57.3% of training videos have >=1 frozen transition; 89.1% of 6bba (114/128, 947
  duplicate pairs) vs 0% of 44b6** (discussion/724283). A large embryo-asymmetric artifact - any
  6bba/44b6 asymmetry in our results may be this, not biology.
- **Any local CV from before 2026-07-22 is VOID** - the division-exploit metric patch
  (discussion/727154, organizer).

### E. METHOD INTEL FROM THE TOP

- Soheil Ayati (**rank 2, 0.959**), discussion/737101 - MORE than we had:
  "Always validate complete movies using the official scorer and **movie-level OOF splits**.
  Edge-level random CV can be highly misleading."
- Tang (0.946): "retrain the model instead of just using the public ckpt... the current ckpt has kind
  of hit a wall, it is hard to get more gain from post-processing alone."
- Mendrika's instruction, directly relevant to how we run arms: **"keep the detector fixed while
  comparing linkers, and keep the linker fixed while comparing detectors. Otherwise post-processing
  can appear to improve locally by changing the candidate set and graph simultaneously."**
- ISAKA Tsuyoshi (7th/344 then), discussion/716952, +47: "Detection was the biggest lever... DoG at
  multiple scales... jumped LB 0.786 -> 0.826 (+0.040). In contrast, division edges hurt and gap
  closing was roughly neutral." And: **"detection improvements track CV ~ LB (~1:1). Graph-level
  add-ons (like divisions) can raise CV yet lower LB."**
- **adj_edge plateaus ~0.90-0.91 across the shared split_0 stack** (Arul, discussion/728551) -
  passing it means leaving that stack, not tuning it.
- **HOCT is a dead end for point detections** - Arul measured it under-performing a tuned ILP,
  over-linking, ~45 min/movie; the Ultrack author confirms "we haven't tried on the competition
  dataset". Closed, and paid for by someone else.
- hengck23, discussion/723655, +23: the solver forces wrong links on large displacements because
  "link cost < termination cost + new-track cost" - the termination/disappearance cost is the knob.
  And **<1% of links are labelled**, so pseudo-labelling short tracks from an ensemble of open-source
  trackers is his stated route to a better edge model.

### F. GAPS
**No team at >=0.947 has described their method** beyond Soheil's node-selection line. TWEAK posted
once with substance and never answered the direct follow-up. **Nobody has publicly stated an adj_edge
above ~0.90** - the >=0.947 cluster's edge number is not in the forum. No thread mentions Trackastra,
TrackMate, Linajea or motile at all.

### G. WHAT THIS REORDERS
1. **Fork-free probe (1 slot)** - returns our exact adj_edge. Decisive, no modelling work, tells us
   whether we are edge- or division-limited before any further spend.
2. **Detection threshold reduction + merge pass** - `p4_detsweep_export_f0.json` is already built and
   unpushed; the metric line is 1 - 0.1 x over-prediction and node recall enters edge recall squared.
3. **Local association ranker** - now doubly motivated: it is the only thing that lets the linker
   reject the junk pool that a recall push creates (Moawiz's counter-signal).

## 2026-08-25 — HOST DECISION: Kaggle for ALL GPU compute. Colab DECLINED. Both training lanes REPAIRED.

### A. Compute decision — single pool, final

Host, 2026-08-25: *"actually just use Kaggle for all GPU Compute tasks"*. Colab Pro and the
Colab VS Code extension are **declined**. This reinstates the single-pool plan.

Investigated before declining (recorded so it is not re-litigated): the extension
(https://marketplace.visualstudio.com/items?itemName=Google.colab) "exposes Colab servers
directly in VS Code", i.e. **cloud runtimes, not a local runtime** — so it would have been
genuinely additive GPU that does not touch the Kaggle quota, and being single-GPU it would have
sidestepped the DataParallel autocast trap entirely. It was declined anyway. **Two practical
limits made it a poor fit regardless:** it is an interactive editor integration with a browser
sign-in step and **cannot be driven from this session**, so every Colab run would be a manual
hand-off; and the marketplace page documents no GPU types, no session limits and no Pro
specifics.

**Consequences of Kaggle-only, restated:**
- Device is deterministically **T4x2, CC 7.5** — fp16 + `GradScaler` is correct and no bf16
  branch is needed. Colab's non-deterministic T4/L4/A100 allocation stays closed.
- **The DataParallel/autocast trap is guaranteed on every run**, not conditional.
- Training and inference now compete for the same ~30-45 GPU-h/week.
- **NEW OPERATIONAL LIMIT, measured today: Kaggle permits a maximum of 2 concurrent GPU BATCH
  sessions.** `p17_det094` was refused — *"Maximum batch GPU session count of 2 reached"* —
  while `p15_forkfree_probe` and `p16_det090` were running. **Kernel pushes must be pipelined
  two at a time.** This was not previously in the ledger and it bounds how many arms can be in
  flight.

### B. S5 (`h1r_edge_train.py`) — all six audit defects REPAIRED

1. **CUDA autocast crash.** `F.binary_cross_entropy` is banned under CUDA autocast and raised at
   runtime on step 1. The loss now runs in **fp32 with autocast explicitly disabled**, while the
   UNet/transformer forward keeps its fp16. **No CPU smoke could ever have caught this** — the
   CPU autocast path permits the op, which is exactly why it survived a passing smoke.
2. **ALL division supervision was deleted — the worst defect, now fixed.** Root cause was subtle:
   `_transition_indices` (`h1r_edge_data.py:190-203`) **already** resolves a daughter to her
   mother's row via `div_i = source_row.get(parent_id)`, so the correct target existed all along.
   The trainer destroyed it twice — zeroing those columns out of the target and dropping them
   from the loss — so the model never saw one mother->daughter positive, discarding the
   **65,741-link asset the lane exists for**. The original intent was legitimate (avoid a false
   negative when the node cap drops a mother) but the remedy was wrong: **that is a SAMPLING
   problem, not a DIVISION problem, and it applies identically to continuations.** The mask now
   keys on the sampling condition alone, and divisions are supervised and upweighted 3x
   (`H1R_DIV_WEIGHT`, matching this repo's own P3 patch; Trackastra uses 11).
3. **Node-cap sampling bias.** Source and target frames were drawn INDEPENDENTLY, so with ~942
   nuclei and a cap of 256 a target's true parent survived only ~27% of draws — making ~73% of
   columns look parentless while deployment has ~92% *with* a parent, a **~9x background-prior
   error on the exact quantity the parental softmax must calibrate**. Targets are now drawn
   first and their true parents retained in the source draw. **The cap is unchanged; only which
   nodes fill it.**
4. **Checkpoint selection was `link_top1 * candidate_recall`** — the exact product
   `h1r_trainer_patch.py:19-22` already rejected once for having no precision term and
   "selecting the junkiest detector". Replaced with the harmonic mean.
5. **"Frozen detection" froze only `detect_head` (one 1x1 conv)** while the shared UNet trained.
   The contract is now EXPLICIT: `H1R_TRUNK_MODE` = `frozen` (DEFAULT) | `adapt` | `distill`.
   `frozen` freezes the trunk itself — because every detection logit is `detect_head(unet(x))`
   **and `unet_out` is the exact tensor deployment feeds to `predict_edges`**, so training the
   trunk rewrites every edge feature under a transformer trained on the old representation, and
   that representation IS the 0.915 substrate.
6. **The post-run guard was TAUTOLOGICAL** — it compared `detect_head`'s state_dict to its
   initial values, parameters that were never in the optimizer and cannot change. Replaced with
   `detection_drift()`, which measures detection **behaviour** against a frozen reference on a
   fixed probe batch, reports every epoch, and hard-fails if `frozen` mode ever drifts >1e-4.
   **This is also the T3.1 pre-flight gate made runnable:** run a short `adapt` fine-tune, read
   `max_abs`, and if detection barely moves the entire drift apparatus is unnecessary.
7. **Resume was unreachable** (`/kaggle/working` starts empty). Now discovers an attached
   prior-run checkpoint under `/kaggle/input`.

### C. S1 (`h1r_det_train.py`) — the three SEV-1 defects were ALREADY FIXED. Verified, not redone.

A previous session repaired them between the audit and now. **Verified at file:line rather than
assumed:**
- Selection binds to `selection_at_threshold(val, "deployed")`, and `best_operating_point` now
  carries the docstring *"Diagnostic best point in a sweep; never used for checkpoint
  promotion."* `is_best` uses `>` with `min_delta`. The module docstring's claim about selecting
  at the deployed operating point is now **true**.
- `encode_detection_logits` implements the full 8-view planar TTA set; `H1R_EVAL_TTA` defaults
  to "1" and `eval_detection` uses it.
- `H1R_TRUNK_CONTRACT` defaults to `freeze` with a real byte-level `assert_trunk_unchanged`
  guard on a trunk snapshot (a genuine check, unlike S5's old head-bytes version, because the
  trunk IS in the optimizer under adapt mode).
- SEV-2 "20 epochs, no early stopping" is also closed: `H1R_PATIENCE` 3, `H1R_MIN_DELTA` 1e-4.

**One defect DID remain and is now fixed:** resume was still local-only, the same dead code as
S5 — a preempted 12 h kernel silently restarted from epoch 0. It now searches `/kaggle/input`.
**And the latent corruption the audit flagged alongside it is closed:** resuming a checkpoint
without its `metrics.json` left `history == []`, which makes the downstream zero-epoch branch
label a **partially trained** model as the baseline and overwrite `edge_predictor_best.pth` with
it, turning every later `improvement_over_baseline` into fiction. It now fails closed.

### D. Test-quality findings worth keeping

- **Three S5 tests encoded the defects as contracts**, one literally named
  `test_division_columns_have_exactly_zero_loss_gradient`. All rewritten to assert the corrected
  behaviour. A fourth, `test_division_columns_stay_masked_when_node_cap_drops_the_mother`, failed
  *because the fix worked* — parent-retention now rescues the mother — and was renamed
  accordingly.
- **A pre-existing FLAKY test was exposed.** `test_synthetic_end_to_end...` asserts the projector
  receives gradient, but the triplet loss is margin-based: on an unlucky unseeded draw every
  triplet already satisfies the margin and the gradient is legitimately zero. It failed alone and
  passed in-suite — the signature of RNG order dependence. Verified across three seeds that
  gradient always flows when seeded, then seeded the test. **The flake predates this work;
  changing the sampler merely shifted the RNG stream enough to expose it.**
- **One of my own new tests was wrong** and passed for the wrong reason: I wrote `0.60 x 0.60`
  against `0.20 x 1.00` as "equal products" (0.36 vs 0.20). Corrected to `0.36 x 1.00` so the
  test actually isolates the harmonic mean's preference.

### E. Verification
`pytest -q` -> **761 passed**, 0 failures. `validate_research_tree.py` OK (65 docs);
`claims_table.py --check` OK (53 claims).

### F. Still open on S5, deliberately
The edge lane has **no DataParallel at all**, so the second T4 is idle — but `H1R_EDGE_BS=1`
means it could not split anything anyway. If DataParallel is ever added, note that the training
autocast at the call site is main-thread and **would then hit the thread-local trap**; the fix is
the same one applied to the loss.

## 2026-08-25 — THE `leevvin` LEAK CLAIM: UNVERIFIED, and the vendored default CONTRADICTS it. The practical conclusion survives anyway.

This closes an item that has been open since it was first raised and was repeatedly cited as a
possible explanation for the whole LOEO->LB validation crisis. **It does not hold as stated.**

### The claim

`leevvin/biohub-movie-heldout-edge-predictor-v1` README, VERIFIED verbatim (file on disk at
`data/external/public_weights/leevvin/README.md`):
> "The widely used public support-pack checkpoint was trained with `train=199` videos and a test
> list that is a *subset* of those 199 - so it has seen every labelled movie. Anyone validating
> locally on movies from `train/` is therefore scoring a model that memorised them, which makes
> offline post-processing sweeps point the wrong way."

### What the code actually does — VERIFIED at file:line

`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:1036-1051`, the split construction:
```python
stems = sorted(p.name[:-5] for p in data_dir.glob("*.zarr") if (data_dir / f"{p.name[:-5]}.geff").exists())
random.Random(0).shuffle(stems)
n_val = max(1, len(stems) // 10)
folds = [{"split": 0, "train": stems[n_val:], "test": stems[:n_val]}]
```
**The default split is 90/10 and strictly DISJOINT** — `train = stems[n_val:]`, `test = stems[:n_val]`,
by construction non-overlapping. Producing `test` as a SUBSET of `train` would require a custom
`dataset_splits.json`, and **the support pack does not ship one for `unet_transformer`**: its
weights directory contains only `checkpoint_last.pth`, `config.json` and `edge_predictor_best.pth`
(verified against the dataset file listing). The `split_manifest.json` referenced by
`source_scripts/run_full_frame_center_training.sh:248` belongs to the **DeepCenter full-frame
detector**, a different model.

Our own `data/dataset_splits.json` is also clean: two folds, 128/71 and 71/128, **overlap 0** in
both directions.

**VERDICT: the specific `test is a subset of train` mechanism is UNVERIFIED and the default code
path contradicts it.** It cannot be confirmed without the support pack's actual splits file, which
is not published. Do not cite it as established.

### BUT the practical conclusion survives, for a different and simpler reason

A 90/10 split over 199 movies still puts **~179 of 199 movies in training**. So any given movie in
`data/train` has roughly a **90% chance of having been trained on**, and our four placeholder crops
are almost certainly among them. **Local scoring against `data/train` with the public support-pack
weights is therefore contaminated regardless of whether `test` was a strict subset of `train`.**

The distinction matters:
- **Wrong reason (leevvin's):** a deliberate `test`-inside-`train` construction. Unverified.
- **Right reason:** an ordinary 90/10 split means almost everything is in train, so validating on
  `train/` movies scores a model that saw them.

The consequence for us is unchanged — **local post-processing sweeps on `data/train` are not a
trustworthy promotion gate** — but the mechanism should be stated correctly, because the wrong
mechanism implies a deliberate defect in the public artifact and the right one implies only that
we are validating on the wrong movies.

### What this does and does not license

- It does **not** retire the standing suspension of LOEO as a promotion instrument; that was
  measured directly (arm-B +0.0144/+0.0090 local -> +0.000 LB), independently of any leak.
- It does **not** justify reopening closed levers on the grounds that they "were closed on
  contaminated evidence". Abstention in particular was closed on a 58.3%-vs-58.88% margin, and
  nothing here shows that margin was inflated by contamination.
- It **does** reinforce rank-2 Soheil Ayati's forum advice (discussion/737101, VERIFIED):
  *"Always validate complete movies using the official scorer and movie-level OOF splits.
  Edge-level random CV can be highly misleading."*
- It **does** raise the value of `leevvin`'s actual artifact, whatever the reasoning behind it:
  that checkpoint holds out exactly our four placeholder crops, which makes it the only public
  edge predictor that is uncontaminated on our scoring substrate.

### Process note
This is the fourth time a premise cited as established turned out not to be verified at
`file:line`. It cost nothing to check — two greps into the vendored trainer.
