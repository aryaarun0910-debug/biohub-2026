# Self-harm audit — cross-report synthesis and ranked patch list (2026-08-22)

**Role of this document.** Four sibling audits (config / dead code / vendor drift / constants) were
commissioned to inventory damage we inflicted on our own system. This file is the *across* layer:
one ranked, executable patch list, plus the interaction graph that says what may ship alone and
what must ship bundled. It is written to be **regenerated wholesale** as sibling reports land (§11).

**Ranking rule applied (not re-derived):** the measured transfer law of 2026-08-19
(`research/07-outputs/submissions.md:595-625`, `research/00-system/handoff.md:57-80`) —
detection-surface / candidate-set change → **LB ×3.5 amplified**; division-term change → **LB
0.000**; edge-permutation change → **LB 0.000**. A patch is ranked by **the class it acts in**, not
by how broken it looks.

---

## 0. SOURCE STATUS — READ FIRST

| expected sibling report | present? |
|---|---|
| `internal-reports/audit_config_2026-08-22.md` | **ABSENT** |
| `internal-reports/audit_deadcode_2026-08-22.md` | **ABSENT** |
| `internal-reports/audit_vendordrift_2026-08-22.md` | **ABSENT** |
| `internal-reports/audit_constants_2026-08-22.md` | **ABSENT** |

A listing of `research/06-knowledge-system/internal-reports/` on 2026-08-22 contains **no `audit_*`
file at all**. Every claim below is sourced directly from primary code, from the deployed notebook,
or from pre-existing reports — **none from a sibling audit.**

To keep this document useful rather than empty, the coordinator ran two primary sweeps in place of
the missing siblings: a full `BIOHUB_*` knob enumeration against the deployed kernel, and a
file:line trace of the four headline defects. Those cover roughly what `audit_config` and
`audit_vendordrift` would have covered. **Two gaps remain and are not closed here:**

- **`audit_constants`** — no percentile data exists for any constant outside the division geometry
  (§5 is 4 rows deep and mostly says "unmeasured").
- **`audit_deadcode`** — §4/§9 below list what fell out of the config sweep incidentally, not a
  systematic search for unreachable branches or discarded stage effects.

Notation: **`Cn:m`** = cell `n`, line `m` of
`notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb` (the deployed kernel). `wrapper.py:N` =
`src/biotrack/wrapper.py`, the partial offline mirror (see patch #6 — it is not the deployed code).

---

## 1. THE RANKED PATCH LIST

Cheapness: **C0** = no compute. **C1** = config-only + one GPU kernel. **C2** = code + one GPU
kernel. **C3** = code + training. "Slot?" = needs a submission slot.

| # | Patch | Kind | Class | LB-movable? | Cheap | Slot? | Independent? |
|---|---|---|---|---|---|---|---|
| **1** | Run the **built-but-never-launched `p4_detsweep_export_f0/f1`** — one GPU run makes the entire `[0.5,1.0]` det-threshold curve a **CPU replay** | CONFIG (already built) | **detection surface** | **YES ×3.5** | C1 | no | **INDEPENDENT** |
| **2** | **`BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48`** (vendor `cfg.threshold = 0.5`) — the column-softmax gate that *defines the candidate set*, loosened below vendor with **no ablation on record**. Sweep it. | CONFIG | **candidate set** | **YES ×3.5** | C1 | no (use placeholder gauge) | **INDEPENDENT** |
| **3** | **Re-operating-point the `leevvin` checkpoint** — `p7` collapsed 99.7% only because it ran at our inherited `DET_THRESHOLD` | CONFIG | **candidate set** | **YES ×3.5** | C1 | yes, after #1 | needs #1 |
| **4** | **Output node filter discards ~5,400 GT-matched nodes** (`error_atlas` §5) — audit via same-run A/B | CODE | **detection surface** | **YES ×3.5** | C2 | later | **ANTAGONIST-BOUND** (§2, B3) |
| **5** | **Sub-threshold candidate reservoir** (public mechanism 2) | CODE | **candidate set** | **YES ×3.5** | C2 | yes | **INDEPENDENT** |
| **6** | **`src/biotrack/wrapper.py` contains zero DeepCenter code** — the repo's declared "deployed wrapper" describes a pipeline in which the discriminator does not exist | CODE/DOC | **instrument** | no — gates everything | C0 | no | **INDEPENDENT** |
| **7** | **`loeo_retarget.py:127-141` force-disables the *entire* secondary + DeepCenter stack** in the strict arm — LOEO has never measured the deployed config | CODE | **instrument** | no — gates everything | C0 to tag | no | **INDEPENDENT** |
| **8** | **`BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` is never set anywhere** — p8 just shipped the discriminator at the untuned default `0.12` | CONFIG | division | **no — 0.000** | C0 | no | **MUST BUNDLE w/ p8** |
| **9** | **Division proposals ranked tightest-first** (`parent_dist + 0.15·sister_dist`) — true divisions are *wide*, so they rank last and the caps fill with false ones | CODE | division | **no — 0.000** | C2 | no | **MUST BUNDLE w/ #8,#10** |
| **10** | **Add the public stack's divergence gate** `d(succ(c1),succ(q)) − d(c1,q) ≥ 2.25 µm` **+ mutual-nearest-orphan test** | CODE | division | **no — 0.000** | C2 | no | **MUST BUNDLE w/ #8,#9** |
| **11** | **`BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER = 0`** — a *second* division filter we implemented and disabled, making 3 more knobs unreachable | CONFIG | division | **no — 0.000** | C0 | no | bundle w/ #9,#10 |
| **12** | **Gap-closing consumes orphans before the division proposer runs** (`wrapper.py:1189-1192`) | CODE | division | **no — 0.000** | C2 | no | bundle w/ #9,#10 |
| **13** | **`candidate_ids` requires no incoming edge** — 64% of *detected* divisions unproposable at **any** gate | CODE | division | **no — 0.000** | C2 | no | bundle w/ #9,#10 |
| **14** | **`BIOHUB_ILP_APPEARANCE_WEIGHT = 0.0`** vs vendor `0.1` — divisions strictly dominated | CONFIG | division | **no — one slot already spent, 0.000** | C0 | — | **DO NOT RE-SPEND** |
| **15** | **`BIOHUB_GAP_CLOSE_MAX_GAP = 2` is clamped by `min(…, 1)`** at `C6:627` — the deployed value is a literal no-op | CONFIG | dead code | no | C0 | no | INDEPENDENT |
| **16** | **`BIOHUB_OUTPUT_VOLUME_GUARD` exists ONLY in the offline mirror** (`wrapper.py:102,108,1033`) and has **no counterpart in the deployed kernel** — the mirror divergence runs in *both* directions | CODE/DOC | **instrument** | no — gates everything | C0 | no | **FOLD INTO #6** |
| **17** | **Stale comment `C3:122`** asserts DeepCenter "is disabled" in this run; it is enabled and the run **hard-fails** without it | DOC | doc | no | C0 | no | INDEPENDENT |
| **18** | **Vendor trainer division-upweight no-op** (`weight[div_rows] = 1.0`) | CODE | training | only via retrain | C3 | no | **ALREADY FIXED** (§4) |

### The one-line read

**Only rows 1–5 can move the leaderboard.** Rows 8–14 are the dramatic-looking self-harm — a
disabled discriminator, a second disabled filter, a gate below the biology, a strictly-dominated
solver term — and every one of them sits in the **division class, which has three zero-scoring
submissions behind it**. They are worth fixing because they are cheap, correct, and load-bearing
for the retrain lane. They are **not** worth a submission slot on their own evidence.

Rows 6 and 7 move nothing and are ranked above most of the list anyway, because they are **why the
rest of the list survived unnoticed for weeks**. See §10.

**The single most under-rated row is #2.** `BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48` is the column-
softmax gate that literally defines the candidate set (`C5:66` patch body; vendor default
`cfg.threshold = 0.5` at `predict_unet_transformer.py:73`; set at `C4:530`). It is a **divergence
from vendor, in the only class that transfers, with no ablation anywhere in the corpus**, and it is
config-only. It has never appeared in a bet, a report, or a sweep. Everything the campaign has said
for four days about "the candidate generator is the bottleneck" points at this knob, and nobody has
touched it.

---

## 2. BUNDLING / INTERACTION GRAPH

```
   INSTRUMENT LAYER  ── ship first, zero LB risk, unblocks reading everything below
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ #6  src/biotrack/wrapper.py has NO DeepCenter code at all                     │
   │ #7  loeo_retarget.py:127-141 strict arm force-disables:                       │
   │        SECONDARY_WEIGHTS=""  SECONDARY_EDGE_WEIGHT=0  SECONDARY_DET_WEIGHT=0  │
   │        USE_DEEPCENTER_VETO=0  REQUIRE=0  GAP_VETO=0  SAFE_DIV_VETO=0          │
   │ #17 C3:122 comment states the opposite of the truth                           │
   └───────────────────────────────┬──────────────────────────────────────────────┘
                                   │ until fixed, every LOEO number is blind to
                                   │ the secondary model AND all four vetoes
                                   ▼
 ┌─────────────────── DIVISION BUNDLE — transfer class 0.000 ───────────────────────────┐
 │                                                                                      │
 │  loose gates ──▶ DISCRIMINATOR ──▶ RANKING ──▶ CAPS ──▶ emitted divisions            │
 │  12/15/10        #8 threshold      #9 tightest-  frame 0.0076 / global 0.00375       │
 │  [p8 shipped]       unset @0.12       first,     (identical to public 0.917 —        │
 │                  #10 no diverge       wrong       NOT a divergence, do not touch)    │
 │                  #10 no mutual-NN     quantity                                       │
 │                  #11 2nd filter off                                                  │
 │                                                                                      │
 │  upstream leaks:  #12 gap-close eats orphans first  │  #13 orphan-only kills 64%     │
 │                                                                                      │
 │  ⚠ MEASURED HARD CONSTRAINT: loose gates WITHOUT a working discriminator =           │
 │    ~2,400–3,300 false candidates per true division. NEVER ship #12/#13 with          │
 │    relaxed gates unless #8 and #10 ship with them.                                   │
 └──────────────────────────────────────────────────────────────────────────────────────┘

 ┌────────── CANDIDATE-SET / DETECTION-SURFACE LANE — transfer class ×3.5 ──────────────┐
 │                                                                                      │
 │   #1 det-threshold curve ──▶ #3 leevvin re-operating-point                          │
 │      (built, never launched)                                                         │
 │                                                                                      │
 │   #2 DUAL_SEED_EDGE_THRESHOLD 0.48   ── INDEPENDENT of #1 ──▶ own sweep              │
 │      (a DIFFERENT gate: edge column-softmax, not detection sigmoid)                  │
 │                                                                                      │
 │   #5 sub-threshold reservoir         ── INDEPENDENT ──▶ own A/B                      │
 │                                                                                      │
 │   #4 output node filter ◀── ANTAGONIST ──▶ the measured +0.005/+0.018 node-count     │
 │      bonus. Adding nodes back erodes it. JOINT OPTIMUM, not a bug fix.               │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

**Hard bundling rules.**

- **B1 — Never relax gates without a live, *tuned* discriminator.** ~2,400–3,300 false candidates
  per true division (`experimental-records.md:883`; `bet_consolidation_2026-08-18.md:167`). p8
  satisfies this only if #8 also holds.
- **B2 — #8, #9, #10 are one patch, not three.** A discriminator at an unset operating point,
  filtering into a ranking sorted by the wrong quantity, is defeated by the caps *regardless of how
  good the discriminator is*. Shipping any one alone is uninterpretable.
- **B3 — #4 must be a same-run A/B.** `error_atlas_2026-08-19` §5 flags it **cross-run confounded**
  and explicitly declines to act. It also notes the node-count-bonus antagonism.
- **B4 — #3 depends on #1.** p7 v3 produced 400 nodes vs P3's 122,214 (`submissions.md:668-676`)
  purely because leevvin's detector ran at our inherited `0.96875`. The checkpoint is not dead; it
  has never been run at its own operating point.
- **B5 — #1 and #2 are genuinely independent.** They are two different gates: `DET_THRESHOLD`
  (detection sigmoid, `C2:7`) and `DUAL_SEED_EDGE_THRESHOLD` (edge column-softmax, `C4:530`). Do
  not conflate them; sweep separately or the interaction is unattributable.
- **B6 — #14 is closed by measurement, not argument.** One slot already spent
  (`bets.yaml:65-86`, submission `55616685`, 0.915 flat, mechanism verifiably fired: 703 divisions
  vs 307). Do not re-spend.

---

## 3. THE CASE THAT STARTED THIS — what the primary sources actually say

### MEASURED

| fact | primary source |
|---|---|
| `BIOHUB_DEEPCENTER_SAFE_DIV_VETO` default is **ON** (`!= "0"`); we override to `'0'` | read `C3:135`; override `C2:34` |
| Veto applied **before** the ranking score and **before** the caps — ordering is correct | `C6:1037-1048`, score at `C6:1044`, sort at `C6:1050` |
| What the veto computes: **max of the DeepCenter sigmoid heat-map in a ±(1,2,2) window around the candidate point**, reject if `< threshold` | `deepcenter_score_point` `C6:963-969`; `deepcenter_accept_repair_point` `C6:984-997` |
| The DeepCenter model **is loaded and paid for anyway** — `USE_DEEPCENTER_VETO=1`, `GAP_VETO=1`. The safe-div veto was free compute. | `C2:27-28`, `C2:32` |
| `SAFE_DIV_MAX_UM = 4.66` vs measured true parent-distance median **7.42 µm (44b6) / 8.87 µm (6bba)**; 135/151 GT divisions unproposable | `experimental-records.md:868`; `linker_division_capacity_2026-08-18.md:273-280` |
| `ILP_APPEARANCE_WEIGHT = 0.0` vs vendor `0.1` → conversion cost `1.0 − p ≥ 0` ∀ `p ≤ 1`; division strictly dominated | `vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:78,558,629`; `scripts/kaggle_edits/pre_ilp_export.py:6-11`; `linker_division_capacity_2026-08-18.md:243-248` |
| 176,835 out-degree≥2 sources offered on fold 1 → **0** divisions emitted; 29/125 GT divisions had *both* true daughters offered at median `edge_prob` 0.9188 | `bets.yaml:81-86` |
| Trainer division upweight is a literal no-op | `vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:68-70` |
| **A second division filter is also disabled**: `OUTPUT_DIVISION_GEOMETRY_FILTER = 0` (default), making `DIV_PARENT_MAX_UM` / `DIV_SISTER_MAX_UM` / `DIV_DROP_TO_SINGLE_IF_BAD` unreachable | `C3:71-74`, `wrapper.py:50-53`, call site `C6:1471` |

### INFERENCE — and this is the most important thing in the document

> **The DeepCenter veto and the public stack's divergence gate are not the same mechanism, and they
> are not substitutes.**
>
> The veto asks *"is there a real cell centre at this point?"* — a **spurious-node detector**. The
> public 0.917 filter asks *"do these two children move apart afterwards?"*
> (`d(succ(c1),succ(q)) − d(c1,q) ≥ 2.25 µm`) plus *"are they mutual nearest orphans?"* — those test
> **sisterhood**, which is the quantity that actually separates a division from a neighbouring cell
> (`public_code_teardown_2026-08-19.md:22`).
>
> At 2,400:1, the overwhelming majority of false candidates admitted by a 12 µm gate **are real
> cells** — they are simply not sisters. A cell-ness veto passes nearly all of them.
> **Therefore `DEEPCENTER_SAFE_DIV_VETO=1` is a materially weaker discriminator than the public
> design it is standing in for, and p8 may substantially under-deliver against the 0.923 reference
> even if the division class did transfer.** Graded INFERENCE: nobody has measured the veto's
> rejection rate on division candidates, because — see §10 — the only instrument that could is the
> one with the veto switched off.

### Correction to the framing of the commission

Three of the four headline defects are **not self-inflicted**:

- `SAFE_DIV_MAX_UM = 4.66` / `SISTER = 8.5` / `EXISTING_CHILD = 7.65` are **INHERITED** from the
  public 0.914 base preset (`C2:21-23`). `quickwins_internal_2026-08-17.md:32-33,142` recorded them
  a week ago as *"tuned to 2–3 sig figs on an unknown substrate"* and ranked them **low-EV**.
- The trainer no-op is a **vendor** defect.
- `DEEPCENTER_SAFE_DIV_VETO='0'` sits in the same inherited preset cell.

Note also: **`p3_harmonic.json` and `p7_cleanedge.json` contain zero `kind:"env"` edits** — both
inherit the base preset verbatim. Only `p8_loosefilter.json:32-43` overrides it. In fourteen
submissions we changed the mechanism repeatedly and the **configuration essentially never**.

The self-harm is not that we chose these values. It is that we **read them, wrote them down,
labelled them low-EV, and moved on** — twice, five days apart. Different failure, different fix (§10).

---

## 4. WHAT THE FLAGGED DEFECTS ARE *NOT* — false positives, recorded so nobody re-spends

| flagged as | actually |
|---|---|
| `SAFE_DIV_FRAME_FRAC_CAP=0.0076` / `GLOBAL=0.00375` look arbitrarily self-tuned | **Byte-identical to the public 0.917 stack's `DIV_FRAME_CAP`/`DIV_GLOBAL_CAP`** (`public_code_teardown_2026-08-19.md:126-152`) — shared origin. **NOT a divergence. Do not touch.** |
| `BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR = 0` is "a mechanism shipped disabled" (teardown mechanism 8) | **Name collision.** Our implementation (`wrapper.py:1176-1185`) keeps *only the best child per source* — it **deletes** divisions, and runs **before** `add_safe_divisions_postlink`. `0` is correct; `restore_learned_divisions.py:21` already says so. |
| The trainer no-op is unfixed | Fixed at `scripts/kaggle_edits/h1r_trainer_patch.py:85-94` (`weight[div_rows] = _H1R_DIV_WEIGHT`, default 3.0; Trackastra uses 11), with a unit test at `h1r_edge_loss_patch.py:438`. Inert only because we have not retrained. |
| The trainer no-op is a live scoring defect | **Training only.** The deployed weights were trained by the organizer *with* the no-op. Zero effect on any inference config. |
| The `GAP_DENSITY_*` block is a tuned mechanism | Four of its five knobs (`REFERENCE_UM` `6.5`, `GAIN` `0.040`, `MAX_STEP_DELTA_UM` `0.125`, `NEIGHBORS` `3`, `C2:14-17`) are **set to exactly their own defaults** — no-op sets. Only `GAP_DENSITY_ADAPTIVE=1` is a real change. |

---

## 5. CONSTANTS — where the deployed value sits in its own true distribution

Only the division geometry has percentile data. **The rest of the constant surface is unmeasured**
and awaits `audit_constants_2026-08-22.md`.

| constant | deployed | vendor/default | true distribution | verdict |
|---|---|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.66 | 4.7 | parent-dist median 7.42 / 8.87 µm; p90 8.4–9.9 | **below the median — excludes >50% outright** · INHERITED, wrong |
| `SAFE_DIV_SISTER_MAX_UM` | 8.5 | 7.2 | sister-dist median 9.01 / 10.35 µm | below the median · INHERITED, wrong |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.65 | 7.8 | unmeasured | INHERITED |
| **`DEEPCENTER_SAFE_DIV_THRESHOLD`** | **never set → 0.12** | 0.12 | unmeasured on division candidates | **UNSET — the operating point of the discriminator p8 just enabled** |
| `DEEPCENTER_GAP_THRESHOLD` | 0.25 | 0.10 | unmeasured | SELECTED — **note the asymmetry** |
| `DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM` | 8.5 | 0 | unmeasured | SELECTED — narrows the gap veto to long gaps only |
| `DET_THRESHOLD` | 0.96875 | 0.99 | `metric_forensics` break-even p*≈0.50 | **~2× the claimed break-even** · INHERITED, never re-swept on our stack (patch #1) |
| **`DUAL_SEED_EDGE_THRESHOLD`** | **0.48** | **0.5 (vendor `cfg.threshold`)** | unmeasured | **SELECTED, no ablation anywhere · candidate-set class (patch #2)** |
| `ILP_APPEARANCE_WEIGHT` | 0.0 | 0.1 | degenerate — makes divisions strictly dominated | INHERITED, then **rationalised post-hoc as our own win, "A-11 already shipped"** (`quickwins_internal_2026-08-17.md:25,42,158`) |
| `ILP_DISAPPEARANCE_WEIGHT` | 1.5 | 0.1 | unmeasured | **UNJUSTIFIED — 15× vendor, no ablation on record**, same "A-11 already shipped" label |
| `MOTION_RELINK_LEARNED_BONUS` | 1.0 | 0.75 | unmeasured | SELECTED, no ablation on record |
| `SECONDARY_LOW_MARGIN_MAX` | 0.35 | 0.2 | unmeasured | SELECTED, no ablation on record |
| `SAFE_DIV_*_FRAC_CAP` | 0.0076 / 0.00375 | 0.008 / 0.004 | matches public 0.917 exactly | INHERITED, **correct** |

**The `DEEPCENTER_GAP_THRESHOLD` vs `DEEPCENTER_SAFE_DIV_THRESHOLD` asymmetry is the tell.** Same
model, same accept function, two call sites. One got a deliberately tuned operating point (0.10 →
0.25) *and* a span narrowing (0 → 8.5 µm). The other got **neither a threshold nor an enabled
flag**. That is what "we built a filter and disabled it" looks like at file:line.

---

## 6. VENDOR DRIFT — classified

Pending `audit_vendordrift_2026-08-22.md`. What is establishable now:

| divergence | vendor | ours | class |
|---|---|---|---|
| `ilp_appearance_weight` | 0.1 | 0.0 | **UNJUSTIFIED** (measured to fire and to strictly dominate divisions; LB-neutral) |
| `ilp_disappearance_weight` | 0.1 | 1.5 | **UNJUSTIFIED** (15×, no ablation) |
| `cfg.threshold` → `DUAL_SEED_EDGE_THRESHOLD` | 0.5 | 0.48 | **UNJUSTIFIED** — and it is the **candidate-set gate** (patch #2) |
| `ilp_division_weight` / `ilp_edge_weight` | 1.0 / −1.0 | unchanged | NEUTRAL |
| trainer division upweight | no-op | no-op (fix staged, unapplied) | **vendor LOSS**; ours matches |
| harmonic bidirectional blend | absent | 0.20 | **MEASURED-WIN** (P3, deployed 0.915) |
| `add_safe_divisions_postlink`, all `SAFE_DIV_*`, all `DEEPCENTER_*`, dual-seed, gap stages | **no vendor counterpart** | ours | **entirely our own layer** — "vendor default" is meaningless here |
| division proposal *filter* (diverge + mutual-NN) | absent | absent | **the gap vs the public 0.917 stack** (mechanism 1) |

**Structural note for whoever writes the vendor-drift audit: four baselines are in play** — the
vendored `tracking_cellmot` package; the public 0.914 base notebook we forked (`C2`, the preset);
the public 0.917/0.923 frontier; and `src/biotrack/wrapper.py`, which matches **none of them**
(patch #6). Most of our "drift" is against the second, not the first.

---

## 7. DEAD CODE / NO-OPS found incidentally

Not a substitute for `audit_deadcode_2026-08-22.md`.

| item | evidence |
|---|---|
| `BIOHUB_GAP_CLOSE_MAX_GAP = 2` is clamped by `effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)` | `C6:627`; set at `C2:11` — **the deployed value buys nothing** |
| `ADAPTIVE_SHORT_TRACK_RESCUE = 0` renders **6** `SHORT_TRACK_RESCUE_*` knobs unreachable | `C2:26`, `C3:96-101`, call site `C6:1142` |
| `OUTPUT_GAP2_RECOVERY = 0` renders **6** `GAP2_*` knobs unreachable | `C2:20`, `C3:108-113`, call site `C6:830` |
| `OUTPUT_DIVISION_GEOMETRY_FILTER = 0` renders **3** `DIV_*` knobs unreachable | `C3:71-74`, call site `C6:1471` |
| `DEEPCENTER_SAFE_DIV_VETO = 0` renders `DEEPCENTER_SAFE_DIV_THRESHOLD` inert | `C2:34`, `C3:139` |
| **11 env sets write exactly the existing default** | `GAP_DENSITY_{REFERENCE_UM,GAIN,MAX_STEP_DELTA_UM,NEIGHBORS}`, `OUTPUT_FILTER_SHORT_TRACKS`, `OUTPUT_MIN_TRACK_LEN`, `OUTPUT_KEEP_DIVISION_COMPONENTS`, `USE_DEEPCENTER_VETO`, `REQUIRE_DEEPCENTER_VETO`, `DEEPCENTER_GAP_VETO`, `DUAL_SEED_MIN_CANDIDATE_RETENTION` |
| Trainer division upweight | `vendor/.../train_unet_transformer.py:68-70` |

**Roughly 16 of ~103 knobs are structurally unreachable in the deployed run, and 11 more are
no-op sets. That is ~26% of the config surface that cannot affect the output at all** — which is
precisely the noise that let three real defects hide inside it.

---

## 8. BRANCH ON `p8_loosefilter` (live)

Kernel `aryaarun07/biohub-p8-loosefilter` v1. Pre-registered prediction on record
(`submissions.md:717-720`): central 0.918–0.923, upside 0.924–0.928 (~20%), flat 0.915 (~25%).

### If p8 ≥ 0.918 — the division class DOES transfer

- Promote #8, #9, #10, #11 to **TIER 1**, above #5.
- Ship **#8 first and alone** (config-only, C0): highest-leverage single knob in the bundle, and
  p8's own margin becomes the baseline to beat.
- Re-open `bet-division-proposal` (**closed**, `bets.yaml:39`) and `bet-ilp-division-economics`
  (**parked**, `bets.yaml:65`). Re-examine #14 — the divfix zero may have been *over-division*
  (703 divisions ≈ 176/crop vs true ~27/crop), which is the alternative `bets.yaml:71-74` left
  explicitly unseparated, rather than class-zero-transfer.
- **Do not conclude the transfer law is wrong.** p8 changes `BIDIRECTIONAL_EDGE_WEIGHT` 0.20→0.30
  and the DeepCenter checkpoint (`checkpoint_last.pt`/ep500 → `best.pt`/ep2) **simultaneously with**
  the division changes. It is a 6-way confound. A ≥0.918 result buys a follow-up ablation, not a
  conclusion.

### If p8 = 0.915 flat — two readings survive, and they are separable for free

- **(a) the class genuinely does not transfer** — the placeholder/hidden sets hold ~3 GT divisions;
  `0.1×divJ` has no headroom. Rows 8–14 drop to housekeeping, the division lane closes for good,
  everything moves to #1–#5.
- **(b) the discriminator was too weak** — the §3 inference: a cell-ness veto at an untuned `0.12`
  cannot separate sisters from neighbours at 2,400:1, so p8 degenerated to *pure relaxation*, which
  `div_proposal_funnel.py` already killed.
- **DISTINGUISHING TEST, ZERO COST — do this before writing anything down about p8.** Fetch
  `run_stats.csv` and read `deepcenter_safe_div_rejected` / `deepcenter_safe_div_accepted` and
  `safe_divisions_added`.
  - Veto rejected a *small* fraction **and** `safe_divisions_added` exploded → **(b)**: the veto was
    a near-no-op and the loose-propose/strict-filter design was **never actually tested**.
  - Veto rejected hard **and** divisions stayed near P3's 307 → **(a)**.
  - Under **(b)** the correct next move is **#10** (the real divergence gate), not another threshold
    guess.
- **VERIFIED 2026-08-22 — the test is live.** `deepcenter_safe_div_accepted` and
  `deepcenter_safe_div_rejected` are initialised in the main stats dict at `C6:1374-1375` and
  incremented at `C6:408`; the gate state and its threshold are echoed into the run manifest at
  `C3:220,224`. They are **not** behind `RUN_OUTPUT_DIAGNOSTICS = 0` (`C2:35`). An earlier draft of
  this document raised that as a caveat; it was wrong and is withdrawn. **No precondition remains —
  run the test the moment p8 lands.**

---

## 9. MEASURED vs INFERENCE — ledger

**MEASURED (primary source cited above, reproducible):**
transfer law ×3.5 / 0.000 / 0.000 · veto default ON, overridden to `'0'` at `C2:34` · veto applied
before ranking and caps (`C6:1037-1050`) · 4.66 vs 7.42/8.87 µm · `ILP_APPEARANCE_WEIGHT=0.0` makes
division strictly dominated · 176,835 offers → 0 divisions · 2,400–3,300 false per true · 17,067 GT
edges (15.65%) never nominated · perfect solver over today's candidates = +0.0012 · trainer no-op at
`train_unet_transformer.py:68-70` · `src/biotrack/` contains no DeepCenter code · LOEO strict arm
disables secondary + all four vetoes (`loeo_retarget.py:127-141`) · frac caps identical to public
0.917 · `GAP_CLOSE_MAX_GAP` clamped at `C6:627` · p7 400 nodes vs 122,214 · `SINGLE_CHILD_REPAIR`
deletes rather than adds divisions · `DUAL_SEED_EDGE_THRESHOLD=0.48` vs vendor 0.5 ·
`p3_harmonic.json`/`p7_cleanedge.json` carry zero env edits.

**INFERENCE (reasoned, not measured — do not cite as fact):**
the DeepCenter veto tests cell-ness not sisterhood and is a weak substitute for the divergence gate
(§3) · p8 may degenerate to pure relaxation at threshold 0.12 (§8b) · gap-closing consumes orphans
before the division proposer and so suppresses proposals (#12) · the single-quoted `'0'` on the veto
override amid double-quoted neighbours suggests a later hand-edit rather than an inherited line ·
the public stack's two-pass Hungarian may leave more orphans than ours, partially relieving #13 on
their side.

**UNKNOWN — nobody has measured:**
the veto's rejection rate on division candidates · every constant's percentile outside the division
geometry · whether `ILP_DISAPPEARANCE_WEIGHT=1.5`, `MOTION_RELINK_LEARNED_BONUS=1.0`,
`SECONDARY_LOW_MARGIN_MAX=0.35` or `DUAL_SEED_EDGE_THRESHOLD=0.48` have **ever** been ablated ·
whether #4 (the ~5,400-node filter) is net-positive once the node-count bonus is priced in ·
whether the DeepCenter veto counters survive `RUN_OUTPUT_DIAGNOSTICS=0`.

---

## 10. THE META-QUESTION — what process failure produced this

Blunt version: **the research machine measured the system, and the system it measured was not the
system we ship.**

Four compounding failures, in the order they mattered.

### F1 — The only local instrument is structurally blind to the mechanism

`scripts/kaggle_edits/loeo_retarget.py:127-141`, for the `strict` LOEO arm, sets
`BIOHUB_SECONDARY_WEIGHTS=""`, `SECONDARY_EDGE_WEIGHT=0`, `SECONDARY_DETECTION_WEIGHT=0`,
`USE_DEEPCENTER_VETO=False`, `REQUIRE_DEEPCENTER_VETO=False`, `DEEPCENTER_GAP_VETO=False`,
**`DEEPCENTER_SAFE_DIV_VETO=False`**. The reasons printed in the code are legitimate — the secondary
model saw every train crop; *"no fold variant exists"* for DeepCenter. But the consequence was never
carried forward: **every division-lane LOEO measurement we have ever made was taken on a pipeline
where the discriminator cannot fire, and where the secondary model does not exist.**

`div_proposal_funnel.py` — which produced the decisive *"1 of 151 GT divisions admitted"* result
that **closed `bet-division-proposal`** (`bets.yaml:39-50`) — was walking a veto-free pipeline. The
funnel was right about the *gates* and structurally incapable of seeing the *filter*. Nobody wrote
down "this instrument cannot see DeepCenter", so nobody could notice.

### F2 — The kill note named the fix, then re-scoped it to the most expensive thing on the board

`bets.yaml:48-49` closes `bet-division-proposal` with:

> *"Divisions require a learned appearance-based sister discriminator + a non-stealing linker = the
> edge model. Redirects INTO `bet-zebrahub-retrain`."*

An appearance-based discriminator existed, in our own kernel, at `C6:1037`, switched off by `C2:34`.
The kill note correctly identified the required mechanism and routed to a GPU retrain **without an
inventory check of what we already had**. There is no step in the bet lifecycle that asks
*"before we build it — do we already own it, and is it on?"*

### F3 — Inherited values were audited once, labelled low-EV, and thereby immunised

`quickwins_internal_2026-08-17.md:32-33` lists `SAFE_DIV_MAX_UM = 4.66` and
`SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65` as *"3-sig-fig, substrate unknown"*, and `:142` concludes
they are **low-EV**. That judgement predates the 2026-08-18 funnel measuring the true parent-distance
median at 7.42/8.87 µm. When the new measurement landed, **nothing went back and re-scored the old
audit.**

Worse: `:25-26,42,158` records `ILP_APPEARANCE_WEIGHT=0.0` and `ILP_DISAPPEARANCE_WEIGHT=1.5` as
*"A-11 **already shipped**"* — inherited values retrospectively adopted as our own win and thereby
removed from the candidate list, **six days before** the same value was measured to emit zero
divisions from 176,835 opportunities. **Calling an inherited default a shipped win is the specific
move that hides it.**

### F4 — The repo's declared source of truth is not the deployed artifact

`CLAUDE.md` names `src/biotrack/` the *"immutable scorer/graph core and deployed wrapper"*.
`grep -i deepcenter src/biotrack/` returns **nothing**. All DeepCenter logic — score function,
accept function, both call sites, ten `BIOHUB_DEEPCENTER_*` knobs — exists **only in the generated
notebooks**. An agent auditing the division path from `wrapper.py:771-860` finds
`add_safe_divisions_postlink` with **no filter stage whatsoever** and correctly concludes the design
has none. That is exactly what happened. `C3:122` compounds it with a comment asserting DeepCenter
is disabled in this run, when the run **hard-fails** without it.

**And the divergence runs in both directions.** `BIOHUB_OUTPUT_VOLUME_GUARD` exists *only* in the
mirror (`wrapper.py:102,108,1033`, with a comment instructing that it be set to `1` "for any
artifact that must pass validation") and has **no counterpart anywhere in the deployed kernel**. So
the mirror describes a safety mechanism the shipped pipeline does not contain, while the shipped
pipeline contains a discriminator the mirror does not. **Neither artifact is a superset of the
other, and nothing in the repo says so.** Any audit that picks one file and trusts it will be wrong
in a direction it cannot detect from inside that file — which is the precise mechanism by which
this entire incident stayed invisible.

### What should change

Five things. All cheap. Ranked by how much of the above each would have caught.

1. **Every instrument declares what it cannot see.** `loeo_retarget.py` already *prints*
   `"LOEO: DeepCenter add-only gate DISABLED"`. Promote that to a machine-readable
   `disabled_mechanisms: [...]` field in `LOEO_MANIFEST`, and require any result row citing a LOEO
   number to carry it. **A funnel that cannot see the veto must not be allowed to close a bet about
   filtering.** This alone catches F1, the root of the whole incident.

2. **A closing bet must answer "do we already have this, and is it on?" at file:line.** Add one
   mandatory `already_owned:` field to `bets.yaml`, populated with a grep result, not a judgement.
   The existing screening rules R1–R8 (`handoff.md:158-166`) test whether a lever *should* work.
   **None tests whether it is already built and disabled.** Catches F2.

3. **Delete the "already shipped" category.** An inherited default is INHERITED until an ablation
   exists. Every knob gets exactly one of `MEASURED-WIN` (ablation cited), `INHERITED`, or
   `SELECTED` (sweep cited). `ILP_APPEARANCE_WEIGHT`, `ILP_DISAPPEARANCE_WEIGHT`,
   `DUAL_SEED_EDGE_THRESHOLD`, `MOTION_RELINK_LEARNED_BONUS` and all four `SAFE_DIV_*` constants
   should be relabelled **today**. Catches F3.

4. **Make the deployed artifact the audited artifact.** Either vendor the DeepCenter block into
   `src/biotrack/`, or amend `CLAUDE.md` to state plainly that the deployed pipeline is
   `notebooks/kaggle_p0b_clean913_revtime/*.ipynb` and `src/biotrack/wrapper.py` is a partial
   mirror. The second is a one-line change and removes a trap that has now cost weeks. Catches F4.

5. **Every enabled gate must have an explicitly set threshold.** Add a build-time assertion to
   `kaggle_factory.py`: if a spec enables any `*_VETO` or `*_GATE` flag, the corresponding
   `*_THRESHOLD` must be set explicitly in the same spec. Mechanical, and it would have caught
   patch #8 at build time this morning, before the kernel was pushed.

### The uncomfortable part

103 knobs, ~55 attempted levers, 13 bets, 33 internal reports, a claims table with drift detection,
and a validated research-tree schema — and the finding that mattered was **six env-var comparisons
against a public notebook, done by hand, in an afternoon.**

The machine is optimised for producing *new measurements* and has **no routine that re-reads its own
configuration**. Every one of F1–F4 is a failure of **inventory**, not of **analysis**. The analysis
in this corpus is repeatedly correct — `bets.yaml:48` named the exact missing mechanism and then
pointed away from it.

**Concrete implication for the next month:** before any GPU is committed on 08-24, spend one
afternoon on the ~64 knobs still unaudited by anyone. The measured cost of not doing it is this
entire document. And note what it competes against: `handoff.md` currently plans to open the retrain
lane first. Retraining is the right long lane — it is also the lane F2 wrongly redirected into once
already, for exactly this reason.

---

## 11. REGENERATION PROTOCOL

This file is disposable and should be regenerated, not amended, when the sibling audits land.

Inputs, in order: the four `audit_*_2026-08-22.md` reports (if present); `research/07-outputs/submissions.md`
(tail); `research/00-system/handoff.md`; `research/01-research-direction/bets.yaml`; cells 2–6 of
`notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb`; `src/biotrack/wrapper.py:38-138` and
`:771-860`; `scripts/kaggle_edits/loeo_retarget.py:127-141`;
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:76-80,555-563`.

Rules: re-rank **strictly** by the transfer-law class table in `handoff.md` — never by severity.
Fold in the p8 result via §8's branch. Carry forward no ranking that was not re-derived from the
class table. Keep §9's MEASURED/INFERENCE/UNKNOWN split intact; anything promoted from INFERENCE to
MEASURED needs a file:line or an evidence path.

---

## 12. VERIFICATION STATUS OF THIS DOCUMENT (2026-08-22)

Because the four sibling audits are absent, the coordinator re-verified this document's own
load-bearing citations directly against primary sources rather than relying on delegated search.

**Verified exactly as cited:** `C2:11`, `C2:34`, `C3:71`, `C3:122`, `C3:135`, `C3:139`, `C4:530`,
`C6:627`, `C6:1037` (all against `notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb`);
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:73` (`threshold = 0.5`), `:78`
(`ilp_appearance_weight = 0.1`), `:79`, `:558`, `:629`;
`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:68-70` (the no-op);
`src/biotrack/wrapper.py:771-860`, `:1170-1200`; `scripts/kaggle_edits/loeo_retarget.py:127-141`.

**Two errors found in the first draft and corrected:**
1. Patch #16 was ranked as a deployed-config risk. `BIOHUB_OUTPUT_VOLUME_GUARD` **does not exist in
   the deployed kernel at all** — it is mirror-only. Reclassified as instrument, folded into #6, and
   it strengthens F4 (§10) rather than standing alone.
2. §8 carried a caveat that the DeepCenter veto counters might be suppressed by
   `RUN_OUTPUT_DIAGNOSTICS = 0`. **False** — they sit in the main stats dict (`C6:1374-1375`,
   incremented `C6:408`) with the gate state echoed to the manifest (`C3:220,224`). Caveat withdrawn;
   the p8 distinguishing test has no precondition.

**Still unverified by anyone**, and inherited from the coordinator's own sweep rather than a
sibling audit: the counts in §7 ("16 structurally unreachable knobs, 11 no-op env sets") are a
spot tally, not an exhaustive enumeration; the vendor defaults for `MOTION_RELINK_LEARNED_BONUS`
(0.75) and `SECONDARY_LOW_MARGIN_MAX` (0.2) in §5 are **not** re-verified and should be treated as
ASSERTED until `audit_constants` lands. Do not cite either as MEASURED.
