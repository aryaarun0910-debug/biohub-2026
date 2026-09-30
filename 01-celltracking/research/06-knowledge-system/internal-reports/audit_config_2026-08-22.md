# Configuration audit: self-inflicted damage across the BIOHUB_* knob surface

**Date:** 2026-08-22
**Scope:** every `os.environ.get("BIOHUB_...")` reachable in the deployed pipeline, plus every
`os.environ[...] =` assignment in the deployed notebook.
**Deployed artifact audited:** `notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb`
(the P3-harmonic 0.915 lineage; P3/P5/P6/P7/P8 all inherit cell 2 verbatim).
**Comparator:** `yunusgmsoy/kimi-notebook-v17` (public, 0.923), pulled to `c:/temp/kimi17b`.
**Constraints honoured:** no GPU launch, no submission, no commit, `.claude/settings.json` untouched.

Everything under **MEASURED** is a number read out of code, a notebook cell, or an archived run
artifact. Everything under **INFERENCE** is reasoning on top of those, clearly separated.

---

## 0. Headline

Six findings of the shape the audit was looking for — *we built a mechanism and then disabled it,
or bounded it so it can never fire, or let a downstream stage throw its output away.* Ranked by the
2026-08-19 transfer law (only detection-surface / candidate-set levers have demonstrated LB
transfer; division-term and edge-permutation levers both measured 0.000).

| # | finding | class | status |
|---|---|---|---|
| **A** | `ADAPTIVE_SHORT_TRACK_RESCUE=0` while the short-track filter deletes **11,028 nodes (5.9% of raw)** — and its trigger is set at 0.10 when the measured removal rate is **0.029–0.085 on every crop**, so the rescue could not fire even if switched on | **detection surface** | NEW |
| **B** | `motion_relink_edges` replaces **99.87%** of the solver's edges with a strict 1:1 `linear_sum_assignment`, so `ILP_APPEARANCE/DISAPPEARANCE/DIVISION_WEIGHT` and `USE_ILP` cannot affect output topology at all; **100% of output divisions (557/557) come from the safe-div post-processor**, none from the model or the ILP | edge permutation (explains two dead lanes) | NEW |
| **C** | `DEEPCENTER_SAFE_DIV_VETO=0` overrides a code default of `1`, disabling the appearance gate on the only stage that emits divisions — while the gates it exists to filter are set below the biology | division term | KNOWN (the motivating find) |
| **D** | `MOTION_RELINK_LEARNED_BONUS=1.0` buys the learned association model at most **1 µm of tie-breaking** in a cost function whose geometric terms run to 6–10 µm; the safe-div proposer holds `edge_prob` on every existing child edge and ranks on pure geometry instead | edge permutation | NEW |
| **E** | `DET_THRESHOLD=0.96875` still never selected; `DUAL_SEED_EDGE_THRESHOLD=0.48` **supersedes the 2026-08-18 "hard-coded 0.5, no flag exists" record** — the flag exists and we already set it | **detection surface** | PARTLY NEW (correction) |
| **F** | `DUAL_SEED_MIN_CANDIDATE_RETENTION=0.90` is a one-sided ratchet that throws away the whole `0.475`-weighted secondary detector on any frame retaining <90% of primary candidates. **Measured binding on 60 of 400 placeholder frames (15%) — 60% of one movie.** It can only shrink the detection surface, never grow it | **detection surface** | NEW (first draft wrongly called this dormant; corrected) |

---

## 1. MEASURED — the knob surface

**Counts.** 89 distinct `BIOHUB_*` names exist across `src/biotrack/wrapper.py`,
`scripts/kaggle_edits/*.py`, `vendor/kaggle-cell-tracking/scripts/*.py`. The deployed notebook
**sets 41** of them and **reads 55 more that it never sets**, which therefore run at code default.

**Assignment sites in the deployed notebook** (last-assignment-wins verified programmatically —
no knob is assigned twice; `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` is assigned once only, at
`cell5:171`, *after* the env cell, and is read at `cell5:172` and `cell10:20`):

- `cell2:6–35` — the 30-knob env block
- `cell4:519–530` — 7 secondary-model knobs
- `cell5:85`, `cell5:171` — 2 knobs set inside the runtime-patch cell

**Reader sites.** All post-processing knobs are read in `cell3:24–142`, a verbatim copy of
`src/biotrack/wrapper.py:38–142` with a constant line offset of **+21** (wrapper `:38` = notebook
`cell3:59`). Code-default claims below cite the wrapper; both were checked and agree.

### 1.1 MEASURED — deployed graph statistics

Source: `_evidence/kaggle_runs/p3_d1_pilot_f1_v1/run_stats.csv`, 10 crops, LOEO fold 1 (the
deployed post-processing chain). Caveat: the DeepCenter gate is reported DISABLED in that run's log
(`"DeepCenter add-only gate DISABLED (no fold variant exists)"`), so the `deepcenter_*` counters are
0 for a reason unrelated to the veto flag.

| quantity | value |
|---|---|
| `raw_nodes` | 186,281 |
| `gap_added_nodes` | +2,942 |
| `pruned_isolated_nodes` | −180 |
| **`short_track_nodes_removed`** | **−11,028** |
| final `nodes` | 178,015 (arithmetic closes exactly: 186,281 + 2,942 − 180 − 11,028 = 178,015) |
| `short_track_components_removed` | 2,779 |
| `raw_edges` → `edges` | 164,677 → 169,089 |
| **`motion_relink_replaced_raw_edges`** | **164,470 of 164,677 = 99.87%** |
| `motion_relink_tight_edges` / `relaxed_edges` | 161,227 / 9,670 |
| `motion_relink_fallback_raw` / `skipped_large_frame` | 0 / 0 |
| `dropped_long_edges` (`OUTPUT_EDGE_MAX_UM=14`) | 207 (0.13%) |
| `safe_division_candidates` → `safe_divisions_added` | 865 → **557 (64.4% admitted)** |
| `safe_division_skipped_cap` | 35 |
| **`division_like_sources`** | **557 — identical to `safe_divisions_added`** |
| pre-ILP candidate rows (fold 1) | **4,685,519** (`p4_preilp_loeo_f1_v1` log) vs 1,976,653 post-ILP edges = **2.37× shrink at the solver** |
| **retention-guard fallback frames** (deployed P0-B, 4 placeholder movies) | **60 of 400 = 15%** — all 60 on `44b6_0b24845f` (60 of its 100 frames); other 3 movies 0 |

### 1.2 MEASURED — per-crop short-track removal vs the rescue trigger

`ADAPTIVE_SHORT_TRACK_RESCUE` fires only if `removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC`
(`cell6:1134-1136`, default `0.10`).

| crop | nodes pre-filter | removed | removed_frac | ≥ 0.10 trigger? |
|---|---|---|---|---|
| 6bba_312f0dc3 | 8,530 | 650 | 0.0762 | no |
| 6bba_3db54e20 | 17,851 | 1,135 | 0.0636 | no |
| 6bba_57b7cc1e | 69,888 | 4,236 | 0.0606 | no |
| 6bba_6feb10f0 | 9,331 | 791 | 0.0848 | no |
| 6bba_91951b3a | 11,402 | 416 | 0.0365 | no |
| 6bba_969618f6 | 16,708 | 896 | 0.0536 | no |
| 6bba_d3da753b | 14,937 | 434 | 0.0291 | no |
| 6bba_debd7bfa | 11,215 | 716 | 0.0638 | no |
| 6bba_f17befbc | 21,381 | 1,132 | 0.0529 | no |
| 6bba_fc83837d | 7,800 | 622 | 0.0797 | no |

**Zero of ten crops reach the trigger.** Max observed 0.085 against a 0.10 gate.

---

## 2. MEASURED — the damage findings in detail

### A. The short-track filter / rescue pair — the largest node-surface operation in the pipeline, with its counterweight triple-disabled

**Deleting side (ON, aggressive):**

- `OUTPUT_FILTER_SHORT_TRACKS = "1"` — `cell2:6`; code default `"1"` (`wrapper.py:72`)
- `OUTPUT_MIN_TRACK_LEN = "6"` — `cell2:18`; code default `"6"` (`wrapper.py:73`)
- Mechanism: `filter_short_track_components`, `wrapper.py:867` ff. — union-find over the edge set;
  every connected component shorter than 6 frames is **deleted entirely, nodes and edges**.
- `OUTPUT_KEEP_DIVISION_COMPONENTS = "1"` (`cell2:19`) spares components containing a fork only.
- **Measured cost: 11,028 nodes (5.9% of `raw_nodes`), 8,249 edges, 2,779 components.** That is
  **61× larger than `pruned_isolated_nodes` (180)** and **3.7× larger than everything gap-close
  adds (2,942)**.

**Restoring side (OFF, and bounded below the observed regime):**

- `ADAPTIVE_SHORT_TRACK_RESCUE = "0"` — `cell2:26`. Code default is also `"0"` (`cell3:95`).
- Even if flipped to `"1"`, three further bounds apply, all unset and running at default:
  1. `SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = 0.10` (`cell3:96`) — **measured max 0.085; never
     reached on any crop** (§1.2).
  2. Budget `min(SHORT_TRACK_RESCUE_MAX_NODES_ABS=180, 0.018 × n_nodes)` (`cell3:100-101`, used at
     `cell6:1137-1140`) — a ceiling of **~180 nodes against 11,028 removed = 1.6%**.
  3. Eligibility `SHORT_TRACK_RESCUE_MIN_LEN=4 ≤ len < OUTPUT_MIN_TRACK_LEN=6` (`cell6:1147`) —
     only components of size 4 or 5 are ever candidates; sizes 1–3 are structurally unrescuable.

**This is the exact shape of the motivating find, in the class that transfers.** We built the
rescue, switched it off, set its trigger above the operating regime, and capped its budget at 1.6%
of what the deleting side removes. The public 0.923 notebook also sets `"0"` — so this is not
evidence against us, but it is equally untested by anyone.

**Recorded justification:** none. Neither `experimental-records.md` nor `bets.yaml` contains any
entry for `ADAPTIVE_SHORT_TRACK_RESCUE`, `SHORT_TRACK_RESCUE_*`, or a selection of
`OUTPUT_MIN_TRACK_LEN = 6`. Inherited-and-unexamined.

### B. `motion_relink_edges` discards 99.87% of the solver output — the ILP knobs are structurally dead

`wrapper.py:1147-1161`:

```
    if OUTPUT_MOTION_RELINK:
        learned_edge_probs = { (src,tgt): prob for edge in edges ... }   # :1148-1159
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges                                          # :1160-1161
```

`motion_relink_edges` (`wrapper.py:~295-405`) rebuilds the edge set **from `nodes_by_id` alone**.
Per consecutive frame pair it runs `scipy.optimize.linear_sum_assignment` over
`cost[i,j] = motion + 0.05*raw - MOTION_RELINK_LEARNED_BONUS * prob` (`wrapper.py:339`), in two
passes (`MOTION_RELINK_TIGHT_UM=6.0`, then `MOTION_RELINK_RELAXED_UM=10.0`).

Consequences, all measured or directly readable:

- `linear_sum_assignment` is a **1:1 matching**. Out-degree ≤ 1 and in-degree ≤ 1 hold by
  construction, not by any threshold argument. **Every division the ILP emits is destroyed here.**
- The ILP contributes only `edge_prob` values, used as a bonus term. Its *topology* is discarded.
  Measured `motion_relink_replaced_raw_edges = 164,470 / 164,677 = 99.87%`,
  `motion_relink_fallback_raw = 0`.
- **`division_like_sources = 557` equals `safe_divisions_added = 557` exactly.** 100% of divisions
  in the deployed output are manufactured by `add_safe_divisions_postlink` *after* the relink; zero
  survive from the learned model or from the solver.

`BIOHUB_OUTPUT_MOTION_RELINK` is **never set by the notebook** and runs at its code default `"1"`
(`wrapper.py:43`). No recorded decision to enable it exists.

This retro-explains two closed lanes: `p5_divfix` (ILP division economics) measured **+0.0071 local
/ 0.000 LB**, and `bet-ilp-division-economics` is parked. Any lever whose effect is an ILP edge-cost
change is thrown away 27 lines later unless it also re-injects post-relink.

### C. `DEEPCENTER_SAFE_DIV_VETO = 0` — the motivating find, confirmed and quantified

- Deployed: `os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = '0'` — `cell2:34`.
- **Code default is `"1"` (ON)** — `cell3:135`:
  `os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"`.
- Single use site: `cell6:1033-1044`, inside the safe-division proposer, immediately after the
  sister-distance gate and immediately before `proposals.append(...)`. It calls
  `deepcenter_accept_repair_point(...)` at `DEEPCENTER_SAFE_DIV_THRESHOLD`, which is **never set**
  and runs at our default **0.12** (`cell3:139`); kimi-v17's code default is **0.08**
  (`cell3:145`).
- The gate it should be filtering is set below the biology: `SAFE_DIV_MAX_UM = 4.66` (`cell2:21`)
  against a measured true second-daughter parent-distance median of **7.42 µm (44b6) / 8.87 µm
  (6bba)** (`experimental-records.md:868`). Measured funnel: **1 of 151 GT divisions admitted**
  (stage (h) of the same record).
- Meanwhile the proposer admits **64.4% of everything it proposes** (865 → 557 measured), and emits
  557 divisions on a fold whose GT contains ~125. kimi-v17's own comment at `cell1:39` reports the
  same observation independently: *"funnel telemetry showed safe-div accepting 81-100% of its own
  candidates -- turning on the one gate built spec[ifically for it]"*.

**Recorded justification for the `0`:** none found. `bets.yaml:44` records the *diagnosis* of the
tight gate, not a decision to disable the veto.

### D. The learned association model is worth ≤ 1 µm, and the division proposer ignores probability entirely

- `MOTION_RELINK_LEARNED_BONUS = "1.0"` (`cell2:8`); code default `"0.75"` (`wrapper.py:47`).
- Cost function `wrapper.py:339`: `cost = motion + 0.05*raw - 1.0*prob`, with `motion`/`raw` in µm
  and `prob ∈ (0.48, 1]`. The learned term's **full dynamic range is ≤ 1.0**, against geometric
  terms gated at 6.0 µm (tight) and 10.0 µm (relaxed).
- The safe-division proposer (`wrapper.py:812-834`) holds `existing_child_edge` — which carries a
  live `edge_prob`, because relinked edges do keep it (`wrapper.py:~394`) — and ranks purely on
  `score = parent_dist + 0.15 * sister_dist` (`wrapper.py:833`). Probability is never consulted.
  Added edges are written with `"edge_prob": None` (`wrapper.py:854`).
- `experimental-records.md` already notes the ranker "sorts tightest-first — so true divisions,
  being *wide* (7.4–8.9 µm), rank near LAST of thousands." The probability signal that could break
  that tie sits in the same data structure, unused.

### E. Detection / candidate-set thresholds

- `DET_THRESHOLD = "0.96875"` (`cell2:7`); vendor CLI default **0.99**
  (`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:618`), dataclass default 0.5
  (`:68`). Provenance still unresolved: `scripts/d1/d1f_probe.py:1253` states *"NO threshold
  selection, deployed threshold 0.96875"* and `:92` repeats it. **No selection experiment exists.**
  kimi-v17 sets the identical 0.96875 (`cell1:10`), so both notebooks inherit the same unselected
  number.
- `DUAL_SEED_EDGE_THRESHOLD = "0.48"` (`cell4:530`) — **this supersedes a claim in the 2026-08-18
  record.** That record states the edge candidate threshold is *"a hard-coded `threshold = 0.5` on a
  column-softmax … no CLI flag exists for it. In-degree ≤ 1 is then a theorem."* In the deployed
  notebook the flag **does** exist and **is** set: `cell5:278-280` reads
  `BIOHUB_DUAL_SEED_EDGE_THRESHOLD` (defaulting to `str(cfg.threshold)` = 0.5) and `cell5:310`
  assigns `cfg.threshold = edge_candidate_threshold` — reached because `secondary_weights_text` is
  non-empty (`BIOHUB_SECONDARY_WEIGHTS` set at `cell4:519`). Candidate selection at
  `predict_unet_transformer.py:460-466` then uses 0.48, not 0.5. Since the softmax is over `dim=0`
  (columns sum to 1), two entries can exceed 0.48 but only one can exceed 0.5 — **the in-degree ≤ 1
  "theorem" is already broken by our own config.** The observed post-ILP `edge_prob` floor of
  0.5000001 is therefore a property of the solver's output, not of the candidate threshold.
- `DUAL_SEED_MIN_CANDIDATE_RETENTION = "0.90"` (`cell5:85`) is a **one-sided ratchet** on the
  detection surface: `cell5:385-391` discards the entire `0.475`-weighted secondary blend for any
  frame where blended candidates fall below 90% of primary, with no symmetric cap on the upside.
  **CORRECTION (added after first draft): it binds, and hard.** My first pass grepped kernel stdout
  for `BIOHUB_RETENTION_GUARD` and found 0 — that was the wrong instrument. Those logs are 6bba-only
  LOEO folds, and the stdout line prints only for first-seen frames; the real telemetry is the JSON
  sidecar. Measured, deployed P0-B config, on the four placeholder movies
  (`_evidence/agent_runs/armb_deploy_2026-08-01/data/p0b_live/dual_seed_frame_retention_guard_report.json`,
  byte-identical to `_evidence/agent_runs/agent1/out_p0b/...`, md5 `30cdcdf61414158258ba7675f28aa931`):

  | movie | frames | fallback_frames | median_retention | min_retention |
  |---|---|---|---|---|
  | 44b6_0113de3b | 100 | 0 | 1.0072 | 0.9871 |
  | **44b6_0b24845f** | 100 | **60** | **0.8876** | **0.5784** |
  | 6bba_05b6850b | 100 | 0 | 0.9858 | 0.9000 (exactly on the gate) |
  | 6bba_05db0fb1 | 100 | 0 | 1.0028 | 0.9774 |
  | **total** | **400** | **60 (15%)** | — | — |

  On `44b6_0b24845f` the secondary detector is thrown away on **60% of frames**, and the ratchet is
  one-sided by construction — it can only ever *reduce* the detection surface, never expand it. This
  is the class with 3.5× measured LB amplification, so it is now the third-ranked lever, not a
  latent curiosity. **Selection constraint:** these are the four visible placeholder movies, which
  `CLAUDE.md` rule 2 forbids using for model or threshold selection. This telemetry is admissible as
  *diagnosis* (the mechanism binds) but **not** as a basis for choosing a new value — that must come
  from LOEO.
- The same JSON independently confirms §E's correction from a **live P0-B run**: its `configuration`
  block records `"edge_candidate_threshold": 0.48` (alongside `detector_threshold: 0.96875`,
  `ilp_appearance_weight: 0.0`, `ilp_disappearance_weight: 1.5`). Its `topology` block records
  `max_indegree: 1, max_outdegree: 2` on all four movies — consistent with finding B (the relink is
  1:1, and the only out-degree-2 nodes are the ones safe-div adds afterwards).

---

## 3. MEASURED — mechanisms that exist and are simply never turned on

Read by the deployed pipeline, never set by the notebook, code default OFF:

| knob | default | reader | mechanism | note |
|---|---|---|---|---|
| `BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR` | `"0"` | `wrapper.py:41` | `:1176-1185` keeps only the best out-edge per source | **Correctly off** — it would delete every division. Enabling it would be damage, not repair. |
| `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER` | `"0"` | `wrapper.py:50` | `:1195-1226` validates forks on `DIV_PARENT_MAX_UM=10.5` / `DIV_SISTER_MAX_UM=8.0`, drops to single if bad | A *second*, looser division geometry filter, fully built, never used. Its bounds (10.5 / 8.0) sit far closer to measured GT geometry than the active safe-div gates (4.66 / 8.5). |
| `BIOHUB_OUTPUT_VOLUME_GUARD` | `"0"` | `wrapper.py:108` | `:1033` | Never exercised. |
| `BIOHUB_OUTPUT_GAP2_RECOVERY` | `"0"` (also explicitly set `"0"`, `cell2:20`) | `wrapper.py:114` | `:627` — the whole two-frame gap recovery stage, plus 6 tuned sub-knobs (`GAP2_*`) | A complete stage with tuned constants, dead. The explicit `"0"` is redundant with the default. |

`BIOHUB_RESTORE_LEARNED_DIVISIONS` and `BIOHUB_ARMB_FLOW_GATE` exist in `scripts/kaggle_edits/` but
are absent from the deployed notebook (`RESTORE_LEARNED_DIVISIONS` appears only in `p5_divfix`,
`cell2:43`).

---

## 4. MEASURED — full deployed-value table with code default and public-0.923 comparator

`code default` = the literal in the notebook's own `environ.get`. `<<<` marks a difference from
kimi-v17.

| knob | ours | code default | kimi-v17 (0.923) | verdict |
|---|---|---|---|---|
| `DEEPCENTER_SAFE_DIV_VETO` | **`0`** | **`1`** | **`1`** `<<<` | **overrides a default-ON gate to OFF; no recorded justification** |
| `SAFE_DIV_MAX_UM` | `4.66` | `4.7` | **`12.0`** `<<<` | tightened below GT median 7.42/8.87 µm |
| `SAFE_DIV_SISTER_MAX_UM` | `8.5` | `7.2` | **`15.0`** `<<<` | loosened vs default, still far below kimi |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | `7.65` | `7.8` | **`10.0`** `<<<` | tightened vs default |
| `BIDIRECTIONAL_EDGE_WEIGHT` | `0.20` | `0` | **`0.30`** `<<<` | kimi's own comment: *"was 0.20 (0.915 reference value, never tuned)"* |
| `DEEPCENTER_EXPECTED_EPOCH` | `500` | `0` | **`2`** `<<<` | paired with the checkpoint choice below |
| `DEEPCENTER_CHECKPOINT` | `.../checkpoint_last.pt` | `""` | `.../best.pt` `<<<` | kimi: *"best.pt is epoch 2, not checkpoint_last.pt's epoch 500"* — we use the epoch-500 checkpoint of a center-prior model whose best validation was epoch 2 |
| `ILP_APPEARANCE_WEIGHT` | `0.0` | **`0.1`** (vendor `:78`, argparse `:629`) | `0.0` | differs from vendor; **but see finding B — structurally irrelevant** |
| `ILP_DISAPPEARANCE_WEIGHT` | `1.5` | **`0.1`** (vendor `:79`) | `1.5` | **15× vendor.** Track starts free, track ends cost 1.5 — a strong time asymmetry. No recorded justification. Also irrelevant per B. |
| `ILP_DIVISION_WEIGHT` | *(unset → 1.0)* | `1.0` | `1.0` explicit, with note *"0.3/1.0/2.0/3.0 all scored 0.915 on the real leaderboard"* | independent corroboration of our own division-term null result |
| `DET_THRESHOLD` | `0.96875` | `0.99` (vendor CLI `:618`) | `0.96875` | **never selected** (`d1f_probe.py:1253`) |
| `MOTION_RELINK_LEARNED_BONUS` | `1.0` | `0.75` | `1.0` | see finding D |
| `DEEPCENTER_GAP_THRESHOLD` | `0.25` | `0.10` | `0.25` | tightened 2.5× vs default |
| `DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM` | `8.5` | `0` | `8.5` | |
| `GAP_CLOSE_MAX_GAP` | `2` | `1` | `2` | |
| `GAP_CLOSE_UM` | `5.8` | `6.0` | `5.8` | |
| `GAP_DENSITY_ADAPTIVE` | `1` | `0` | `1` | |
| `SECONDARY_EDGE_WEIGHT` | `0.15` | `0` | `0.15` | mechanism off by default, correctly enabled |
| `SECONDARY_DETECTION_WEIGHT` | `0.475` | `0` | `0.475` | as above |
| `SECONDARY_LINK_MODE` | `low_margin_consensus` | `raw` | `low_margin_consensus` | |
| `SECONDARY_LOW_MARGIN_MAX` | `0.35` | `0.2` | `0.35` | |
| `DUAL_SEED_EDGE_THRESHOLD` | `0.48` | `str(cfg.threshold)` = `0.5` | `0.48` | see finding E |
| `DUAL_SEED_MIN_CANDIDATE_RETENTION` | `0.90` | `0.90` | `0.90` | **BINDING** — 60/400 placeholder frames (15%); see §2.E |
| `ADAPTIVE_SHORT_TRACK_RESCUE` | `0` | `0` | `0` | **finding A** |
| `SAFE_DIV_FRAME_FRAC_CAP` | `0.0076` | `0.008` | `0.0076` | kimi: *"never bound at 0.920 (cap_skipped=0)"* |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | `0.00375` | `0.004` | `0.00375` | our measured `safe_division_skipped_cap = 35` — **this one does bind** |
| `RUN_OUTPUT_DIAGNOSTICS` | `0` | `1` | `0` | we run blind by default |
| 16 more, identical to kimi | — | — | — | `OUTPUT_FILTER_SHORT_TRACKS`, `OUTPUT_MIN_TRACK_LEN`, `OUTPUT_KEEP_DIVISION_COMPONENTS`, `OUTPUT_GAP2_RECOVERY`, `USE_DEEPCENTER_VETO`, `REQUIRE_DEEPCENTER_VETO`, `DEEPCENTER_GAP_VETO`, `GAP_DENSITY_{REFERENCE_UM,GAIN,MAX_STEP_DELTA_UM,NEIGHBORS}`, `SECONDARY_MIX_TEMPERATURE`, `ILP_{APPEARANCE,DISAPPEARANCE}_WEIGHT` |

### 4.1 MEASURED — kimi-v17 has *code* we do not have

Three mechanisms exist in the 0.923 notebook with **no counterpart anywhere in our tree** (checked
against `wrapper.py` and all of `scripts/kaggle_edits/`):

| kimi knob | kimi default | reader | what it does |
|---|---|---|---|
| `BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE` | `1` | `cell3:125` | requires the two daughters to be moving apart |
| `BIOHUB_SAFE_DIV_DIVERGE_UM` | `1.5` | `cell3:124` | with comment *"was defaulting to 2.25, carried over from 0.917 which operates at a much wider base radius"* |
| `BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN` | `1` | `cell3:126` | requires candidate and existing child to be mutual nearest neighbours |
| `BIOHUB_VALIDATOR_*` (5 knobs) | on | `cell8:11-15` | an in-kernel scored validator |
| `BIOHUB_BIDIRECTIONAL_FUSION_MODE` | `harmonic_probability` | `cell10:22` | explicit fusion-mode selector |

**This materially qualifies `p8_loosefilter`.** `p8` copies kimi's *numbers*
(`SAFE_DIV_MAX_UM=12.0`, `SISTER=15.0`, `EXISTING_CHILD=10.0`, `SAFE_DIV_VETO=1`,
`BIDIRECTIONAL=0.30`, `best.pt` @ epoch 2 — `notebooks/kaggle_p8_loosefilter/*.ipynb` cell2:41-47)
but **not the three extra precision filters kimi runs alongside them**. Our own measured sweep says
the 12.0/15.0 setting opens **103,008 false candidates for 31 true ones**
(`experimental-records.md`, gate-sweep table). kimi survives that pool with divergence +
mutual-NN + a *looser* deepcenter threshold (0.08 vs our 0.12) doing the filtering; p8 has only the
deepcenter veto.

---

## 5. Ranked action table

Ranked by the 2026-08-19 transfer law: detection-surface / candidate-set first (measured 3.5× LB
amplification), division-term and edge-permutation last (both measured 0.000 LB).

| rank | knob(s) | our value | code default | kimi-0.923 | what it disables | evidence | cheapness to flip |
|---|---|---|---|---|---|---|---|
| **1** | `ADAPTIVE_SHORT_TRACK_RESCUE` + `SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC` + `SHORT_TRACK_RESCUE_MAX_NODES_ABS` | `0` / 0.10 / 180 | `0` / 0.10 / 180 | `0` (untested by them either) | recovery of any part of **11,028 deleted nodes (5.9%)**; triple-bounded so it cannot fire at all | run_stats fold-1 + §1.2; `cell6:1134-1185` | **free** — CPU replay on existing geffs; no GPU, no slot |
| **2** | `OUTPUT_MIN_TRACK_LEN` | `6` | `6` | `6` | the deletion itself. Never selected; no record anywhere | `wrapper.py:73`, `:867` ff.; 11,028 nodes | **free** — sweep 2/3/4/5/6 as CPU post-processing on cached graphs |
| **3** | `DUAL_SEED_MIN_CANDIDATE_RETENTION` | `0.90` | `0.90` | `0.90` | a one-sided ratchet that **discards the entire `0.475`-weighted secondary detection blend** on any frame retaining <90% of primary candidates. **BINDING: 60 of 400 placeholder frames (15%), all 60 on `44b6_0b24845f`** | `cell5:376-391`; `_evidence/agent_runs/armb_deploy_2026-08-01/data/p0b_live/dual_seed_frame_retention_guard_report.json` | free flag change, but **must be selected on LOEO, never on the placeholders** (CLAUDE.md rule 2) |
| **4** | `DET_THRESHOLD` | `0.96875` | `0.99` vendor | `0.96875` | node recall. **Never selected** — stated twice in our own probe | `d1f_probe.py:92, 1253`; error_atlas: 17,067 (15.65%) detectable GT edges never nominated | **one GPU export** (`p4_detsweep_export_f{0,1}` already written and validated, never pushed) turns the whole curve into a CPU replay |
| **5** | `DUAL_SEED_EDGE_THRESHOLD` | `0.48` | `0.5` | `0.48` | candidate set. Pre-ILP graph is already 2.37× the post-ILP graph | `cell5:278-280, 310`; vendor `:460-466`; 4,685,519 vs 1,976,653 | needs a kernel run, but the pre-ILP export already exists for fold 1 |
| **6** | `DEEPCENTER_SAFE_DIV_VETO` (+ `SAFE_DIV_*` gates, + the three missing kimi filters) | `0` | **`1`** | **`1`** | the appearance gate on the only stage that emits divisions | `cell2:34` vs `cell3:135`; `cell6:1033`; 865→557 admitted | built as `p8_loosefilter`, but **incomplete** — see §4.1. Division class: 0.000 measured LB transfer to date |
| **7** | `OUTPUT_MOTION_RELINK` | *(unset → `1`)* | `1` | `1` | 99.87% of the solver's output topology, and every learned division | `wrapper.py:1147-1161`, `:295-405`; 557/557 | expensive to change; **flag it before any future ILP-side lever is designed** |
| **8** | `MOTION_RELINK_LEARNED_BONUS` | `1.0` | `0.75` | `1.0` | caps the learned model at ≤1 µm of influence in a 6–10 µm cost function | `wrapper.py:339` | free CPU replay; edge-permutation class (0.000 LB) |
| **9** | `DEEPCENTER_CHECKPOINT` / `DEEPCENTER_EXPECTED_EPOCH` | `checkpoint_last.pt` / `500` | `""` / `0` | `best.pt` / `2` | uses an epoch-500 checkpoint of a model whose best epoch was 2 | `cell2:29,31` vs kimi `cell1:34,36` | free flag change; only matters once a veto is actually enabled |
| **10** | `OUTPUT_DIVISION_GEOMETRY_FILTER` | *(unset → `0`)* | `0` | *(unset)* | a fully built second division filter at 10.5/8.0 µm — far closer to GT geometry than the active 4.66 gate | `wrapper.py:50, 1195-1226` | free CPU replay |
| **11** | `OUTPUT_GAP2_RECOVERY` + 6 `GAP2_*` | `0` | `0` | `0` | an entire tuned two-frame gap-recovery stage | `wrapper.py:114, 627` | free CPU replay; node-adding, so **detection-surface class** |
| 12 | `ILP_DISAPPEARANCE_WEIGHT` | `1.5` | **`0.1`** | `1.5` | 15× vendor; free starts, expensive ends. No recorded justification | vendor `:79` | irrelevant while §B holds |
| 13 | `ILP_APPEARANCE_WEIGHT` | `0.0` | **`0.1`** | `0.0` | division strictly dominated in the solver | vendor `:78` | irrelevant while §B holds; **kimi at 0.923 sets the identical 0.0, so this is not the gap** |

---

## 6. INFERENCE (separated from the above)

1. **Finding A is the best remaining candidate in the class that transfers.** Node recall is the one
   thing measured to move the LB (3.5× amplified), and 11,028 deliberately deleted nodes is by far
   the largest node-surface decision in the pipeline that has never been sweep-tested. The
   `p6_control_degraded` result also shows the node-count multiplier is *unclamped above 1*, so
   under-producing earns a partial bonus — meaning the true cost of the short-track filter is masked
   on the placeholder substrate and is probably understated there.
2. **Finding B explains, in one line of code, why three separate division lanes all returned 0.000
   on the LB.** Any lever expressed as an ILP edge cost is overwritten 27 lines later. Treat this as
   a standing design constraint, not a knob: future division work must act on `nodes_by_id` before
   the relink, re-inject after it (as `p5_divfix`'s `RESTORE_LEARNED_DIVISIONS` does), or disable
   the relink.
3. **`ILP_APPEARANCE_WEIGHT=0.0` is confirmed as *not* the 0.923 gap.** kimi-v17 sets the identical
   `0.0`. The 2026-08-18 "root cause" analysis remains correct about the objective's economics, but
   the economics are moot given B, and the public evidence shows a 0.923 notebook living happily
   with the same setting.
4. **`p8_loosefilter` should not be pushed as built.** It replicates the loose gates without the
   three precision filters kimi pairs with them, against our own measurement of a 103,008:31
   false-to-true candidate ratio at exactly those bounds. Either port `SAFE_DIV_REQUIRE_DIVERGENCE`
   / `SAFE_DIV_DIVERGE_UM` / `SAFE_DIV_REQUIRE_MUTUAL_NN` and drop `DEEPCENTER_SAFE_DIV_THRESHOLD`
   to 0.08, or accept that p8 tests a strictly weaker design than the one that scored 0.923. Note
   also that on the transfer law this is division-class, where three consecutive submissions
   measured 0.000.
5. **Two of the three cheapest detection-surface levers are one-line flag changes.** Findings A and
   F are both "a mechanism we built, pointed at the detection surface, then bounded out of the
   operating range" — A can never fire, F fires only in the shrinking direction. Neither has a
   recorded justification, and neither has ever been swept. They are also independent: A adds nodes
   back after linking, F changes which nodes exist before linking, so they can be tested separately
   and stacked.
6. **The safe-div ranker is fixable without any env knob.** It has `edge_prob` on the existing child
   edge and sorts on geometry alone. Ranking on `-prob` (or `parent_dist − k·prob`) is a two-line
   change introducing no new mechanism.

---

## 7. What I could not determine

- Whether `DUAL_SEED_EDGE_THRESHOLD=0.48` measurably increases in-degree in the *pre-ILP* graph. The
  `p4_preilp_loeo_f1_v1` parquet (4,685,519 rows) is named in the log but is **not** in
  `_evidence/`; only the log survived. Settling §E fully needs that file.
- Whether kimi-v17 is upstream or downstream of our lineage. It uses our `BIOHUB_*` names and 31 of
  41 values match, so one derives from the other's public ancestor. This does not weaken the
  evidence — a notebook above our score sets the knobs differently — but it means the two configs
  are not independent samples.
- Provenance of `ILP_DISAPPEARANCE_WEIGHT = 1.5`. No record, no bet, no comment anywhere in the
  tree.
