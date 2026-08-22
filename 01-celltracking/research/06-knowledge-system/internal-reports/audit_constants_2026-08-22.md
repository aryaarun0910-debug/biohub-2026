# Constant audit — was it SELECTED, or merely INHERITED?

Every numeric constant in the deployed P3-harmonic configuration, measured against the
distribution of the quantity it actually gates. `[MEASURED]` = computed this session from
GT / deployed artifacts; `[RECORD]` = found in the research machine; `[INFERENCE]` = reasoning
on top of those. Reproduce with the commands in §8.

**Substrates.** GT: `data/train`, 199 crops, 133,318 nodes, 128,883 edges, **151 divisions**.
Deployed output: the two LOEO strict exports (`c:/temp/subvoxel_f{0,1}`), 199 crops,
3.65 M emitted edges. Candidate surface: `c:/temp/preilp_f1_v2/preilp_split1.parquet`,
128 crops, 2.16 M candidate edges.

---

## 1. THE TABLE — where each gate sits in its own true distribution

Percentile = fraction of the TRUE population that the deployed value admits. A gate at the
13th percentile discards 87% of what it exists to admit.

### 1a. Gates measured against ground truth

| constant | deployed | true median | **44b6 pct** | **6bba pct** | **POOLED pct** | verdict |
|---|---|---|---|---|---|---|
| `SAFE_DIV_MAX_UM` (far daughter) | **4.66** | 7.13 µm | 26.9% | 10.4% | **13.2%** | SELECTED¹ |
| `SAFE_DIV_SISTER_MAX_UM` | **8.5** | 10.57 µm | 42.3% | 26.4% | **29.1%** | SELECTED¹ |
| `MOTION_RELINK_TIGHT_UM` | 6.0 | 1.82 µm | 98.4% | 96.2% | 96.5% | INHERITED |
| `GAP_CLOSE_REUSE_UM` | 3.2 | 1.02 µm | 97.7% | 95.4% | 95.7% | INHERITED |
| `GAP_REFINE_MAX_SHIFT_UM` | 3.2 | 0.88 µm | 98.5% | 97.9% | 98.0% | INHERITED |
| `GAP_CLOSE_UM` ×2 (t→t+2 gate) | 11.6 | 3.07 µm | 99.3% | 98.5% | 98.6% | SELECTED |
| `MOTION_RELINK_RELAXED_UM` | 10.0 | 1.82 µm | 99.8% | 99.7% | 99.7% | INHERITED |
| `OUTPUT_EDGE_MAX_UM` | 14.0 | 1.82 µm | 100.0% | 99.9% | 99.9% | INHERITED |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` (near daughter) | 7.65 | 4.08 µm | 100.0% | 95.2% | 96.0% | PARTIAL |

¹ Swept and rejected in `experimental-records.md:857-911, 1293-1299` — but on the LOEO/replay
instrument that `loeo_lb_gap_2026-08-18` later falsified. See §4.

### 1b. Gates measured against the PREDICTED graph

GT is a *sparse subset* annotation — 6.7 nodes/frame against the detector's 154 `[MEASURED]`.
Density and frame-size gates must be calibrated on the detector's output, not on GT.

| constant | deployed | data-implied | percentile | verdict |
|---|---|---|---|---|
| `GAP_DENSITY_REFERENCE_UM` | **6.5** | **10.36 µm** (median local spacing) | **0.8%** | **INHERITED, no record** |
| `MOTION_RELINK_MAX_FRAME_NODES` | 2600 | max observed **872** | 100% (3× headroom) | INHERITED, no record |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.00375 | binds on **48.2%** of crops | 87.9%² | INHERITED |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.0076 | ≈1 division/frame allowed | 99.2%² | INHERITED |
| `GAP_CLOSE_MAX_ADDED_FRAC` / `_ABS` | 0.05 / 2000 | residual need **109** vs budget **180,238** | not binding | INHERITED |

² percentile within the true per-crop / per-frame division-rate distribution.

### 1c. Count gate

| constant | deployed | measured cost |
|---|---|---|
| `OUTPUT_MIN_TRACK_LEN` | **6** | deletes **5.93%** of GT connected components (**263 of 4,435**) |

Curve `[MEASURED]` — GT components destroyed by each setting:

| min_len | 2 | 4 | **6 (deployed)** | 8 | 10 |
|---|---|---|---|---|---|
| GT components deleted | 0 (0.00%) | 131 (2.95%) | **263 (5.93%)** | 425 (9.58%) | 604 (13.62%) |

GT component sizes: median 21, p5 = 5, p10 = 8, **zero of length 1**, max 187.

---

## 2. THE HEADLINE FINDING — a new constant on the wrong side, with no record at all

**`BIOHUB_GAP_DENSITY_REFERENCE_UM = 6.5` sits at the 0.8th percentile of the neighbour
spacing it references.** The detector's actual median local spacing is **10.36 µm**
`[MEASURED, n = 3,800,683 nodes]`. The records audit found **no measurement of this constant
anywhere in the repo** — not in `experimental-records.md`, `bets.yaml`, `submissions.md`, or
`research.sqlite`.

The consequence is mechanical, and it is worse than a mis-set number: **the gate's adaptivity
is degenerate.** The wrapper computes
`step_delta = clip(0.040 × (local_spacing − 6.5), −0.125, +0.125)`, after blending the local
spacing 80/20 toward the same 6.5 reference. Because the reference sits below essentially the
entire distribution, the term is one-sided `[MEASURED]`:

- **49.1%** of nodes are pinned at the **+0.125 ceiling**
- **0.8%** of nodes ever produce a *negative* delta (the contracting half of the mechanism)
- **0.0%** reach the −0.125 floor
- mean applied delta **+0.101 µm/step** = **+1.75%** on the 11.6 µm gate

So `GAP_DENSITY_ADAPTIVE=1` is not adapting to density. It is a **near-constant +0.2 µm
widening** of the gap gate with a ±0.125 µm wobble. Setting the reference to the data-implied
10.36 µm would restore the two-sided behaviour the mechanism was written for.

**Honest headroom caveat `[MEASURED]`:** the residual unclosed gap opportunity across all 199
crops is only **109** (median 0 per crop). Gap-close has already consumed nearly everything
within reach, so repairing this term is *correct* but small. The diagnosis is clean; the prize
is not large.

---

## 3. INERT CONSTANTS — knobs that cannot do anything

Five deployed settings are structurally incapable of taking effect. Each is a place where a
config value implies a behaviour the code does not deliver.

| constant | deployed | why it is inert |
|---|---|---|
| `GAP_CLOSE_MAX_GAP` | **2** | `wrapper.py`: `effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)` — **hard-clamped to 1** `[RECORD: inventory/biprop_SUMMARY.json:83]` |
| `DUAL_SEED_EDGE_THRESHOLD` | **0.48** | **no exported candidate edge has prob < 0.500** `[MEASURED]` — the candidate rule is a column softmax > 0.5, so any value ≤ 0.5 yields an identical candidate set |
| `DIV_PARENT_MAX_UM` / `DIV_SISTER_MAX_UM` | 10.5 / 8.0 | `OUTPUT_DIVISION_GEOMETRY_FILTER = 0` |
| `GAP2_*` (6 constants) | — | `OUTPUT_GAP2_RECOVERY = 0` |
| `DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.12 | `DEEPCENTER_SAFE_DIV_VETO = 0` (being flipped in `p8_loosefilter`) |
| `SHORT_TRACK_RESCUE_*` (6 constants) | — | `ADAPTIVE_SHORT_TRACK_RESCUE = 0` |

**A latent cliff, not currently firing.** `MOTION_RELINK_MAX_FRAME_NODES = 2600` is not a
soft limit: if *any single frame* exceeds it, `motion_relink_edges` returns `[]` for the
**entire crop**, silently deleting every relinked edge in that movie. Max observed frame is
**872** `[MEASURED]`, so there is 3× headroom today — but the failure mode is total and silent,
and the constant has no record of ever being chosen.

---

## 4. THE DIVISION GATES — biggest distance from the data, smallest expected transfer

The deployed division triple admits **19 of 151 GT divisions — 12.6%** `[MEASURED]`. This is
the joint geometric admissibility, before any detector or linker condition:

| setting | parent | child | sister | GT divisions admissible |
|---|---|---|---|---|
| **DEPLOYED** | 4.66 | 7.65 | 8.5 | **19 / 151 (12.6%)** |
| public 0.923 (`kimi-notebook-v17`) | 12.0 | 12.0 | 15.0 | **138 / 151 (91.4%)** |
| +1 step | 6.0 | 8.0 | 10.0 | 40 / 151 (26.5%) |
| +2 step | 8.0 | 10.0 | 12.0 | 78 / 151 (51.7%) |
| uncapped | ∞ | ∞ | ∞ | 151 / 151 (100%) |

One-at-a-time relaxation from the deployed point `[MEASURED]` shows the gates are **coupled**,
and that relaxing either one alone saturates fast:

- `parent` 4.66 → 6.0 → 8.0 → ∞ : 19 → 30 → 43 → **44** (saturates at 44; sister then binds)
- `sister` 8.5 → 10.0 → ∞ : 19 → 20 → **20** (saturates at 20; parent then binds)
- `child` 7.65 → ∞ : 19 → **19** (this gate removes **nothing** — corroborates
  `experimental-records.md:856`, stage (f) removes zero)

**Why this is ranked low anyway.** The transfer law measured 2026-08-19 puts division-term
changes at **0.000 LB transfer** (divfix: local +0.0071, LB 0.000), and these exact gates were
swept and rejected once already. The distance from the data is enormous and the expected
payoff is still near zero. `p8_loosefilter` is the live test of whether a loose-propose +
appearance-veto design breaks that pattern; **this report is not a reason to pre-empt it.**

### 4a. The mechanical explanation for TP=5 / FP=613

The emitted fork population is **geometrically disjoint from the true division population**
`[MEASURED]`:

| quantity | emitted forks (deployed) | true divisions (GT) |
|---|---|---|
| parent→daughter distance, median | **2.47 µm** | 5.75 µm (pooled) / **7.13 µm** (far daughter) |
| parent→daughter, p99 | 5.01 µm | — |
| sister distance, median | **4.02 µm** | **10.57 µm** |
| sister distance, max | 9.07 µm | — |

99% of emitted forks sit at parent distance ≤ 5.01 µm, while 86.8% of true divisions have a
far-daughter distance **above** 4.66 µm. The pipeline is emitting forks in a distance regime
where true divisions barely exist. `[INFERENCE]` The division FP/FN disjointness is not a
ranking failure; it is the gates selecting a different physical population.

---

## 5. THE DIVISION BUDGET IS BINDING — and was never chosen

`SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375` caps safe-division edges at
`round(n_edges × 0.00375)` per crop. Measured across all 199 deployed crops `[MEASURED]`:

- **26 crops** land on **exactly** the cap (chance alone would give ~2–4)
- **96 crops (48.2%)** are at or above it
- median forks/cap ratio **0.97**; fold-1 median is exactly **1.000**

So for roughly half the crops the division count is set by an inherited budget constant, not
by the geometry or the model. `[RECORD]` `public_code_teardown_2026-08-19.md:144` finds
`0.0076 / 0.00375` **identical to the public stack** — "confirms the shared origin". Neither
value was ever selected here.

The companion `SAFE_DIV_FRAME_FRAC_CAP = 0.0076` resolves to `max(1, round(154 × 0.0076)) = 1`
division per frame at the observed frame size — generous against the true rate of
**1.13e-3 per cell-frame**, and not binding. The global cap binds first.

---

## 6. SELECTED vs INHERITED — the full ledger

Of the 40 constants audited, **26 have no sweep or measurement of any kind on record.**
Fourteen are never even *set* in the deployed notebook — they are `wrapper.py` / vendor CLI
defaults that arrived silently.

**SELECTED (a sweep with compared values is on record):** `DET_THRESHOLD` (0.96875, full
offline τ sweep + a 0.999 LB control), `ILP_APPEARANCE_WEIGHT` (0.0, break-even confirmed to
the boundary), `ILP_DIVISION_WEIGHT` (1.0, 0.55 shipped → LB flat), `GAP_CLOSE_UM` (5.8, own
upward sweep 12→30 µm monotone worse), `MOTION_RELINK_LEARNED_BONUS` (1.0, 3-point sweep,
measured ~inert: 8,263 / 8,264 / 8,269 GT edges), `SAFE_DIV_MAX_UM` and
`SAFE_DIV_SISTER_MAX_UM` (swept, rejected on a since-falsified instrument).

**PARTIAL (touched once, never swept):** `BIDIRECTIONAL_EDGE_WEIGHT` (0.20 — "the
source-defined w=0.20, **no blend sweep**"; public 0.923 uses 0.30), `OUTPUT_MIN_TRACK_LEN`
(cost measured, our sweep never run), `OUTPUT_EDGE_MAX_UM` (slack verified, never swept),
`GAP_CLOSE_MAX_GAP`, `SAFE_DIV_EXISTING_CHILD_MAX_UM`.

**INHERITED with literally zero record found:** `SECONDARY_LOW_MARGIN_MAX` (0.35),
`UNET_BATCH_SIZE` (4), `MOTION_RELINK_MAX_FRAME_NODES` (2600), `GAP_DENSITY_REFERENCE_UM`
(6.5), `GAP_DENSITY_MAX_STEP_DELTA_UM` (0.125), `GAP_DENSITY_NEIGHBORS` (3),
`GAP_REFINE_WIN_Z`/`WIN_YX` (1/3), `DEEPCENTER_GAP_THRESHOLD` (0.25),
`DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM` (8.5), `DEEPCENTER_SCORE_WIN_Z`/`WIN_YX` (1/2),
`GAP_CLOSE_MAX_ADDED_FRAC`/`_ABS`.

**Confirmed shared origin with the public stack** `[RECORD]` — identical to our deployed
values: `MOTION_RELINK_TIGHT/RELAXED 6.0/10.0`, `LINEFIT 0.8/2`, `GAP1_SNAP 3.2`,
`DIV_FRAME_CAP/GLOBAL_CAP 0.0076/0.00375`, velocity weight 0.5. These are not our numbers.

---

## 7. RANKING — (distance from data-implied) × (transfer class) × (cheapness)

The transfer law is the dominant term. Only **detection-surface / candidate-set** changes have
shown LB transfer (control: local −0.0091 → LB **−0.0320**, 3.5× amplified); division-term and
edge-permutation changes measured **0.000**.

**1. `OUTPUT_MIN_TRACK_LEN = 6` — sweep {4, 5, 6, 7, 8}.**
The only constant that is simultaneously (a) in the transferring class — it deletes **nodes**,
changing the detection surface; (b) never swept by us; (c) measured expensive: **263 GT
components (5.93%)** destroyed here, and `[RECORD]` **5,311 GT edges the relink had already
linked correctly — 19.2% of all FN**. Dropping to 4 recovers 132 components at 2.95% cost.
Cheap: a CPU replay over the cached graphs, no GPU.

**2. `GAP_DENSITY_REFERENCE_UM = 6.5 → 10.36`.**
Largest percentile displacement in the audit (**0.8th**), zero record, and in the transferring
class (gap-close inserts synthetic nodes). Repairs a provably degenerate mechanism. Ranked
second only because the measured residual headroom is small (109 unclosed opportunities).

**3. `SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375`.**
Genuinely binding on 48.2% of crops and never chosen — but division class, 0.000 measured
transfer. Worth knowing, not worth a slot.

**4–5. `SAFE_DIV_MAX_UM` / `SAFE_DIV_SISTER_MAX_UM`.**
Biggest distance from the data in the whole pipeline (13.2nd / 29.1st percentile; 12.6% of GT
divisions admissible vs 91.4% for the public 0.923 triple). Division class → 0.000 transfer,
already swept and rejected once. **Deferred to `p8_loosefilter`'s live result.**

**6. Documentation-only:** the six inert constants in §3. No action beyond recording that
`GAP_CLOSE_MAX_GAP=2` and `DUAL_SEED_EDGE_THRESHOLD=0.48` do not mean what they say.

**Explicitly NOT recommended:** tightening `OUTPUT_EDGE_MAX_UM` (14.0), `MOTION_RELINK_RELAXED_UM`
(10.0) or `GAP_CLOSE_REUSE_UM` (3.2). All sit above the 95th percentile of their true
distributions — they are slack, not wounds — and all are edge-permutation class, measured at
0.000 transfer.

---

## 8. Reproduce

```powershell
# GT-side distributions, percentiles, joint division-gate sweep
.\.venv\Scripts\python.exe scripts\win_bet\constant_audit.py --gt-dir data\train --json c:\temp\audit.json

# predicted-graph distributions, density degeneracy, cap-binding analysis
.\.venv\Scripts\python.exe scripts\win_bet\constant_audit_pred.py `
    --csv c:\temp\subvoxel_f0\loeo_split0_strict.csv.gz c:\temp\subvoxel_f1\loeo_split1_strict.csv.gz `
    --json c:\temp\audit_pred.json

# candidate-threshold GT-reach curve over the pre-ILP export
.\.venv\Scripts\python.exe scripts\win_bet\candidate_threshold_sweep.py `
    --parquet c:\temp\preilp_f1_v2\preilp_split1.parquet --gt-dir data\train --json c:\temp\tau.json
```

### Candidate-threshold curve `[MEASURED]` — the one sweep that came back clean

Fold-1 GT edges: 109,057. With both endpoints detected: **98,070 (89.93%)** — the detection
ceiling. Candidates reach **80,990 = 82.58%** of those at the deployed setting.

| τ | candidates | GT reached | % of detectable | Δ vs 0.50 |
|---|---|---|---|---|
| **0.50 (= deployed 0.48)** | 2,162,040 | **80,990** | **82.58%** | — |
| 0.60 | 1,945,160 | 77,220 | 78.74% | −3,770 |
| 0.70 | 1,701,405 | 72,174 | 73.59% | −8,816 |
| 0.80 | 1,413,866 | 65,298 | 66.58% | −15,692 |
| 0.90 | 1,049,110 | 54,867 | 55.95% | −26,123 |
| 0.99 | 470,600 | 29,765 | 30.35% | −51,225 |

Monotone: raising the candidate threshold only loses reach. The deployed 0.48 is already at
the structural floor and **cannot be improved downward** — the softmax > 0.5 rule admits at
most one candidate per column. The missing **17.42%** of detectable GT edges are absent
because the column argmax went elsewhere, not because of any threshold.
`[INFERENCE]` This corroborates `error_atlas_2026-08-19` (15.65% never nominated) and confirms
the candidate bottleneck is **structural, not a tunable constant** — consistent with the
standing conclusion that the candidate generator, not the solver, is the frontier.

---

## 9. What this audit did NOT find

Stated so the negative result is on record: **no second `SAFE_DIV_MAX_UM`.** Of the nine
GT-calibrated distance gates, seven sit **above the 95th percentile** of their own true
distributions — generously slack, exactly as a post-filter should be. The pathology of a gate
sitting below the biology is confined to the **division** stage (parent + sister) and to the
**density reference**. The rest of the wrapper's geometry is loose, not tight.

The real structural findings are elsewhere: six constants that cannot act at all (§3), a
budget constant setting the division count on half the crops (§5), and an emitted-fork
population that does not overlap the true one (§4a).
