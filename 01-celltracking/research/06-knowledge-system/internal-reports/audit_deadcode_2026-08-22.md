# Dead-code / no-op audit — 2026-08-22

Scope: `src/biotrack/wrapper.py` (deployed post-processing), `scripts/kaggle_edits/*.py`
(+ `scripts/kaggle_specs/*.json`, `scripts/core/kaggle_factory.py`), and the vendored
trainer/predictor `vendor/kaggle-cell-tracking/scripts/`.

Every claim below is either provable by reading the code or measured. Measurements come from
`c:/temp/preilp_f1_v2/preilp_split1.parquet` (fold-1 pre-ILP graph: 2,523,479 nodes /
2,162,040 candidate edges, 128 crops) and `c:/temp/subvoxel_f1/loeo_split1_strict.csv.gz`
(the corresponding final submission graph: 1,957,978 nodes / 1,877,129 edges). Node ids are
consistent between the two files (`t` matches on 100% of the 6,164 common ids in the first
crop), so pre/post comparison is valid.

Ranking uses the measured **transfer law (2026-08-19)**: only detection-surface /
candidate-set changes have shown LB transfer (control: local -0.0091 -> LB -0.0320, 3.5x
amplified). Division-term and edge-permutation changes measured **0.000**. Findings that
cannot touch the candidate set are marked `NO-TRANSFER` and ranked low no matter how ugly
they are.

---

## 1. Pipeline order — `filter_output_graph`, `src/biotrack/wrapper.py:1069-1241`

```
  raw_edges (ILP output: 2,162,040 edges, 1,972,677 distinct sources)
      |
  [A] pre-filter loop                                    :1129-1143   *** EFFECT DISCARDED ***
      |    - OUTPUT_ENFORCE_NEXT_FRAME (on)  drop dt!=1     :1135  measured 0 drops
      |    - OUTPUT_EDGE_MAX_UM=14 (on)      drop d>14um    :1140  measured 5,796 drops
      |    -> produces `edges`, which is thrown away 18 lines later
      |    -> only surviving product: `learned_edge_probs` dict           :1145-1157
      |
  [B] motion_relink_edges                                 :1158-1161  *** DISCARDS [A] ***
      |    strict 1-1 Hungarian per frame pair, gates 6.0 / 10.0 um
      |    `edges = motion_edges`                                :1161
      |    -> out-degree <= 1 AND in-degree <= 1 by construction  :373-386
      |    -> 189,363 multi-child ILP edges destroyed (measured)
      |
  [C] OUTPUT_SINGLE_PARENT_REPAIR (on)                    :1165-1174  *** DEAD (no-op) ***
      |    in-degree already <= 1 after [B]; also measured {1: 2,162,040}
      |    pre-relink, so it had nothing to do even in the fallback path
      |
  [D] OUTPUT_SINGLE_CHILD_REPAIR (off)                    :1176-1185  *** GATED OFF + DEAD ***
      |    out-degree already <= 1 after [B]
      |
  [E] assert_degree_invariants("motion relink + ...")     :1187       *** UNFIREABLE ***
      |
  [F] close_single_frame_gaps                             :1188  LIVE (adds synthetic nodes)
      |    but effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)     :482
      |    *** deployed kernels set BIOHUB_GAP_CLOSE_MAX_GAP="2" and it is discarded ***
      |    GAP_DENSITY_* block: dead locally (default 0), LIVE on Kaggle (env "1")
      |
  [G] recover_strict_gap2                                 :1189       *** GATED OFF ***
      |    OUTPUT_GAP2_RECOVERY default "0", kernel explicitly sets "0" (:621-628, ~150 lines)
      |
  [H] assert_degree_invariants("gap close + gap2")        :1191       *** UNFIREABLE ***
      |
  [I] add_safe_divisions_postlink                         :1192  LIVE
      |    the ONLY surviving source of out-degree 2. Purely geometric; sets edge_prob=None.
      |    measured output: 6,303 divisions / 1,877,129 edges
      |
  [J] assert_degree_invariants("safe divisions")          :1193       *** UNFIREABLE ***
      |
  [K] OUTPUT_DIVISION_GEOMETRY_FILTER (off)               :1195-1226  *** GATED OFF ***
      |    already documented as never-enabled at scripts/d1/gt_division_gates.py:13
      |
  [L] OUTPUT_PRUNE_ISOLATED (on)                          :1228-1234  LIVE, detection-surface
      |
  [M] filter_short_track_components                       :1236  LIVE, detection-surface
      |    [L]+[M] together remove 565,501 of 2,523,479 nodes (22.4%)
      |
  [N] linefit_smooth_output_graph                         :1237  LIVE, rewrites coordinates
      |    OUTPUT_VOLUME_GUARD (off) inside it                    :1035
      |
  [O] assert_degree_invariants("export")                  :1240       *** UNFIREABLE ***
      |
  final graph: 1,957,978 nodes / 1,877,129 edges
```

### Stages whose effect is discarded, overwritten, or gated off

| # | Stage | file:line | Why it does nothing | Class |
|---|---|---|---|---|
| W1 | Raw-edge pre-filter (both filters) | `wrapper.py:1129-1143` | Output `edges` replaced wholesale at `:1161`; the residue (`learned_edge_probs`) is provably unaffected | candidate-set (already inert) |
| W2 | `OUTPUT_SINGLE_PARENT_REPAIR` | `wrapper.py:1165-1174` | in-degree already <=1 both before and after relink | edge-perm, NO-TRANSFER |
| W3 | `OUTPUT_SINGLE_CHILD_REPAIR` | `wrapper.py:1176-1185` | off, and would be a no-op if on | division, NO-TRANSFER |
| W4 | `recover_strict_gap2` (~150 lines) | `wrapper.py:621-770` | `OUTPUT_GAP2_RECOVERY="0"` in kernel | candidate-set (dormant) |
| W5 | `OUTPUT_DIVISION_GEOMETRY_FILTER` | `wrapper.py:1195-1226` | never enabled anywhere | division, NO-TRANSFER |
| W6 | `BIOHUB_GAP_CLOSE_MAX_GAP="2"` | set in kernel, capped at `wrapper.py:482` | `min(GAP_CLOSE_MAX_GAP, 1)` silently halves the configured gap horizon | **detection-surface, CAN TRANSFER** |
| W7 | 4x `assert_degree_invariants` | `:1187, :1191, :1193, :1240` | provably cannot fire in the deployed config | contract only |
| W8 | `GAP_REFINE_MAX_SHIFT_UM=3.2` rejection | `wrapper.py:255` | window geometry bounds the shift at 3.16 um < 3.2 | detection-surface (dormant) |
| W9 | `GAP_DENSITY_*` adaptive block | `wrapper.py:439-520` | dead in the library default (`"0"`), **live on Kaggle** (`"1"`) — local and deployed disagree | detection-surface |
| W10 | `refine_synthetic_midpoint` w/ `TEST_DIR=None` | `wrapper.py:219-262` | fails into `except` and increments `gap_refine_failed` | detection-surface |
| W11 | `OUTPUT_VOLUME_GUARD` | `wrapper.py:105, 1035` | default off, deliberately (documented `:81-104`) | not a defect |
| W12 | Never-tripping guards | `:517`, `:1029`, `:433` | `np.isfinite` on values that are finite by construction; ABS cap above the FRAC cap | curiosity |

---

## 2. Wrapper findings in detail

### W1 — the entire raw-edge pre-filter is inert (`wrapper.py:1129-1143`)

**Meant to do:** drop non-consecutive and over-long ILP edges before linking.

**Why it does nothing:**

1. `edges` built by this loop is overwritten at `:1161` (`edges = motion_edges`) whenever motion
   relink returns anything — which is always (see W7).
2. The only thing that survives is `learned_edge_probs` (`:1146-1157`), keyed by
   `(source_id, target_id)` and consumed at `:283-293`. But `motion_relink_edges` only ever
   evaluates pairs with `t_target == t_source + 1` (`:369-371`) and `raw <= gate_um` where
   `gate_um <= MOTION_RELINK_RELAXED_UM = 10.0` (`:328-330`). Since `OUTPUT_EDGE_MAX_UM = 14.0
   > 10.0`, **no edge the pre-filter removes could ever have been looked up.**

**Measured (fold-1 candidate graph):**

- `dropped_nonconsecutive_edges`: **0 / 2,162,040** — every candidate edge already has dt=1.
- `dropped_long_edges`: **5,796** (0.27%) — the counter is non-zero, which is why this stage
  looks live in the stats block.
- Edges dropped by the pre-filter that were within the relaxed gate (`d <= 10 um` and `dt == 1`): **0**.

Both filters are provably unable to change the output. `OUTPUT_EDGE_MAX_UM` reads as a
detection-surface lever and is not one.

**Fix:** code change (move the knob into `motion_relink_edges`'s gate, or raise the gates above
14 um). **Transfer:** as written, changing `OUTPUT_EDGE_MAX_UM` anywhere in `[10, inf)` is a
guaranteed 0.000. Below 10 um it *would* start biting — meaning any past sweep above 10
measured nothing. Rank: **high value as a warning, zero as a lever.**

### W2/W3 — parent and child repair are doubly dead (`wrapper.py:1165-1185`)

`motion_relink_edges` removes both endpoints from `unmatched_sources`/`unmatched_targets` on
every match (`:381-383`), so its output is a strict one-to-one matching: in-degree <= 1 and
out-degree <= 1 by construction. `OUTPUT_SINGLE_PARENT_REPAIR` (on) therefore keeps every edge
and `dropped_multi_parent_edges` is structurally 0.

Independently: the **pre-relink** in-degree histogram over the whole fold-1 candidate graph is
`{1: 2,162,040}` — the ILP already emits at most one parent per node, so parent repair would be
a no-op even on the fallback path. `NO-TRANSFER` (edge permutation). Fix: delete.

### W2b — quantifying the division kill (`wrapper.py:1158-1161`)

The known finding, now with numbers:

| | fold-1 measurement |
|---|---|
| ILP sources with out-degree >= 2 | **176,833** (9.0% of 1,972,677 linked sources) |
| second/third/... child edges destroyed by the relink | **189,363** |
| out-degree histogram, pre-relink | `{1: 1,795,844, 2: 165,148, 3: 10,911, 4: 715, 5: 59}` |
| out-degree histogram, final submission | `{1: 1,864,523, 2: 6,303}` |
| divisions in the final submission carrying a learned `edge_prob` | **0** |
| division edges (12,606) that exist in the pre-ILP candidate set | **10,179 (80.7%)** |

Every division that reaches a submission is manufactured by `add_safe_divisions_postlink` from
geometry alone (`edge_prob: None`, `wrapper.py:838`). 80.7% of those edges were *also* proposed
by the learned model — the pipeline re-derives them without the probability it already had. The
learned division signal is not weakened; it is **absent from the output by construction**.

`scripts/kaggle_edits/restore_learned_divisions.py` is the correct fix and is already built and
pushed (`notebooks/kaggle_p5_divfix/build_manifest.json` has `pushed_slug: biohub-p5-divfix`;
the spec sets `BIOHUB_ILP_DIVISION_WEIGHT=0.55`, `BIOHUB_OUTPUT_SAFE_DIVISIONS=0`,
`BIOHUB_RESTORE_LEARNED_DIVISIONS=1`). **Transfer: division-term = measured 0.000. Rank low.**

### W6 — the deployed 2-frame gap-close is silently downgraded to 1 (`wrapper.py:482`)

```python
effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)      # wrapper.py:482
```

**Every** deployed kernel sets `os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"` — verified in
`notebooks/kaggle_p0b_clean913_revtime`, `kaggle_p3_armb`, `kaggle_p3_subvoxel_loeo_f1`,
`kaggle_p5_divfix`, `kaggle_p7_cleanedge` (all carry both the `= "2"` and the `min(..., 1)` cap).
The configured value is read into `GAP_CLOSE_MAX_GAP` and then thrown away.

Consequences, all silent:

- the `t -> t+3` gap-2 bridge never runs;
- `threshold_um = GAP_CLOSE_UM * (gap + 1)` (`:490`) evaluates at gap=1, i.e. 11.6 um with the
  deployed `GAP_CLOSE_UM = 5.8`, not the 17.4 um a gap-2 pass would use;
- `stats["gap_close_effective_max_gap"]` (`:483`) does record 1, so the evidence was always there
  but is never surfaced next to the config.

**This is the highest-ranked wrapper finding.** Gap close *inserts synthetic nodes* into the
output graph (`:557-566`) — it is a detection-surface / candidate-set change, the only class
that has shown LB transfer. Whether raising the cap helps or hurts is unknown, but it is the
only wrapper no-op that can plausibly move the LB. Note `recover_strict_gap2` (W4) is the
already-written, already-gated-off gap-2 mechanism, so the cheap experiment is to enable that
rather than to touch `:482`.

**Fix:** one-line code change (or delete the cap and let the env var mean what it says).

### W8 — the gap-refine shift rejection can never trip (`wrapper.py:255`)

```python
if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:   # 3.2 um
```

`refined` is an intensity-weighted centroid confined to the window
`z +/- GAP_REFINE_WIN_Z (=1)`, `y,x +/- GAP_REFINE_WIN_YX (=3)` around `round(midpoint)`
(`:243-249`). Worst-case displacement from the unrounded midpoint is
`sqrt((1.5*1.625)^2 + 2*(3.5*0.40625)^2) = sqrt(5.9414 + 2*2.0217) = 3.160 um < 3.2 um`.
Clipping at the frame boundary can only reduce it. `stats["gap_refine_rejected_shift"]` is
therefore always 0 and the guard is decorative. Fix: code change (widen the window, or lower the
threshold into the reachable range). Detection-surface but tiny — it only affects synthetic gap
nodes.

### W9 — local and deployed disagree on the density-adaptive gap threshold

`wrapper.py:58` defaults `GAP_DENSITY_ADAPTIVE` to `"0"`, under which `adaptive_threshold` stays
equal to `threshold_um` everywhere (`:489, :495-511`), `adaptive_mask == base_mask`, and the
whole `local_spacing_by_id` / `PREFIX_DENSITY_BLEND` / `PREFIX_DENSITY_PRIOR_UM` machinery
(`:439-478`, ~40 lines) never executes — `gap_density_nodes_scored`,
`prefix_density_nodes_blended`, `gap_density_candidates_expanded/restricted` are all
structurally 0.

But every deployed kernel sets `os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"` before the
constants are read. So this block is **dead in every local replay driver that does not set the
env var** and **live on Kaggle**. That is a local/LB behavioural divergence in a
detection-surface stage — worth checking against the LOEO-LB gap memo. Fix: config change (make
the library default match the deployment, or have the replay drivers import the kernel's env
block). `scripts/win_bet/e0c_run.py:42-44` enumerates the overridable constants and does *not*
include `GAP_DENSITY_ADAPTIVE`, which is how the divergence survived.

### W10 — `refine_synthetic_midpoint` fails silently when `TEST_DIR` is unset

`GAP_REFINE_SYNTHETIC` defaults on (`:69`), but `read_test_frame:199` does `TEST_DIR / f"..."`
with `TEST_DIR = None` (`:26`), raising `TypeError`, which `:260-262` swallows into
`stats["gap_refine_failed"] += 1`. The module docstring says gap-refine is "disabled here", so
this is intentional — but the failure mode is a counter that reads like an image problem rather
than a configuration one. Both real drivers set it (`scripts/win_bet/e0c_run.py:54` ->
`data/train`, `scripts/win_bet/e0c_parity.py:30` -> `data/test`, kernel -> `LOEO_DATA_DIR`), so
this only bites new drivers. Fix: none required; flagged so `gap_refine_failed` is not misread.

### W7 — all four degree assertions are unfireable

`assert_degree_invariants` raises on out-degree > 2 or in-degree > 1 (`:1046-1067`).

- `:1187` — after a strict 1-1 matching both degrees are <= 1. The only path that could reach it
  with raw edges is the `motion_relink_fallback_raw` branch, which requires either
  `not nodes_by_id` or `max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES = 2600` (`:299-302`).
  **Measured: the largest frame in the fold-1 graph has 1,153 nodes** (p99 = 871, mean = 197)
  over 12,800 (dataset, t) frames — 0 frames exceed 2,600, on 0 of 128 crops. The fallback is
  unreachable and the cap sits 2.25x above the observed maximum. This is also the proof that
  motion relink fires on 100% of crops.
- `:1191` — gap close only links a source with no outgoing edge to a target with no incoming edge
  via a fresh middle node (`:522-527`); gap2 is off. Degrees cannot rise above 1.
- `:1193` — `add_safe_divisions_postlink` guards out-degree explicitly at `:855-857`.
- `:1240` — between `:1193` and `:1240` edges are only ever *filtered* (K off, L and M are
  subsets), and filtering cannot raise a degree.

Contract-only, no scientific consequence. Worth keeping as regression armour, but they are not
evidence that any stage is behaving.

### W12 — guards that cannot trip (curiosities)

- `wrapper.py:517` `if not np.isfinite(d).any(): continue` — `d` is filled by `point_distance_um`
  on finite coordinates; always finite.
- `wrapper.py:1029` `if not np.isfinite(fitted).all()` — `np.polyfit` on finite, distinct `dts`
  cannot return non-finite.
- `wrapper.py:433-436` `max_synthetic = min(GAP_CLOSE_MAX_ADDED_ABS=2000, round(N*0.05))` —
  crops run ~15-20k nodes, so the FRAC cap (750-1000) always binds first and the ABS cap is dead.

---

## 3. `scripts/kaggle_edits/` and the spec machinery

### The mechanism first — failed anchors cannot become silent no-ops

`scripts/core/kaggle_factory.py:171` `apply_edit` **hard-fails** on every count mismatch:
`:183` (env), `:205-206` (insert_before/after), `:219-220` (replace), `:234-235` (replace_cell),
`:241-242` (unknown kind), and every touched cell must `compile()` (`:244-250`). Edits apply
strictly in list order (`:286-287`).

**Therefore a stale anchor breaks the build rather than silently doing nothing** — the whole
"patch anchor no longer matches" class is ruled out for all 19 built specs. Two gaps:

- `kaggle_factory.py:222-228` — `append_cell` is the one edit kind with **no `expect` check and
  no anchor**; it always succeeds. 20 spec edits use it (`armb_provenance.py` x12,
  `d1_aggregate.py` x8). This is the remaining blind spot.
- `kaggle_factory.py:146-151` — `_edit_code` prefers inline `code` over `code_file`; an edit
  carrying both silently ignores the file. No current spec does this.

### Module status

| Module | Referencing specs | Built | Verdict |
|---|---|---|---|
| `degree_invariants.py` | all 25 | 19 | LIVE |
| `loeo_export.py` | 19 | yes | LIVE |
| `loeo_retarget.py` | 19 | yes | LIVE |
| `armb_flow_gate.py` | 12 | 10 | LIVE (absent from p7/p8) |
| `armb_provenance.py` | 12 | 10 | LIVE |
| `pre_ilp_rollup.py` | `p4_preilp_loeo_f1` | pushed | LIVE |
| `restore_learned_divisions.py` | `p5_divfix` | pushed | LIVE |
| `d1_inject.py` | 8 x `p3_d1_*` | 2 of 8 | LIVE-STALE |
| `d1_aggregate.py` | 8 x `p3_d1_*` | 2 of 8 | LIVE-STALE |
| `loeo_pregraph_export.py` | 8 x `p3_d1_*` | 2 of 8 | LIVE-STALE |
| `d1_response_audit.py` | none (generator input) | n/a | BUILD-INPUT, misfiled |
| `detpeak_export.py` | `p4_detsweep_export_f0/f1` | built, **never pushed** | **NEVER EXECUTED** |
| `pre_ilp_export.py` | **none** — body inlined into spec JSON | ships inline | **ORPHAN FILE** |
| `subvoxel_refine.py` | **none** — body inlined | ships inline | **ORPHAN + latent no-op** |
| `swap_edge_weights.py` | **none** — body inlined | ships inline | **ORPHAN FILE** |
| `h1r_det_train.py` / `h1r_trainer_patch.py` / `h1r_edge_loss_patch.py` | none | never | NOT SPEC-DRIVEN |
| `node_budget_stage.py` | **none** | never | **DEAD ORPHAN** |
| `volume_guard_linefit.py` | **none** | never | **DEAD ORPHAN** |

### K1 — three modules are inlined duplicates the factory never opens (highest-value here)

`pre_ilp_export.py`, `subvoxel_refine.py`, `swap_edge_weights.py` appear in **no** spec's
`code_file`. Their bodies are copy-pasted into spec JSON as `replace.new` strings:

- `scripts/kaggle_specs/p4_preilp_loeo_f1.json` edit 17 == `pre_ilp_export.py:31-47`
- `scripts/kaggle_specs/p3_subvoxel_loeo_f0.json` / `_f0_smoke` / `_f1` edit 18 ==
  `subvoxel_refine.py:8-40`
- `scripts/kaggle_specs/p7_cleanedge.json` edit 10 == `swap_edge_weights.py` (whole file)

All three are currently byte-identical to their inlined copies, but nothing enforces it.
**Editing any of those `.py` files has zero effect on any build.** They are documentation shaped
like code.

Worse, `subvoxel_refine.py:8-40` wraps its payload in `PATCH = r'''...'''`. If someone "fixed"
the duplication by wiring it in as a `code_file`, the injected cell would define a string
constant and execute nothing — a genuine silent no-op that would pass the build **and** the
`compile()` check and produce an unchanged kernel. This is the one place in the spec machinery
where the hard-fail discipline would not save us.

**Fix:** code/config change in the specs. **Transfer:** none directly — this is a correctness
hazard for future work, not a lever.

### K2 — two fully dead modules

- `scripts/kaggle_edits/node_budget_stage.py` — tokens `node_budget` / `_NB_TARGET` appear in no
  spec and in no base notebook. Its own header (`:3-11`) marks it PROVISIONAL. Last relevant
  commit message: "Both deployable mechanisms are NO-GO on the 0.914 base".
- `scripts/kaggle_edits/volume_guard_linefit.py` — `_VG_VOLUME_ZYX` (`:27`) and
  `linefit_volume_fallback` appear in no spec and not in the p0b base notebook. It implements the
  A3 volume-guard fix, which is therefore **in no current kernel**. This is the same mechanism as
  `OUTPUT_VOLUME_GUARD` (W11), which is also off — so the out-of-volume smoothing leak documented
  at `wrapper.py:81-104` is unmitigated in every deployed kernel.

### K3 — `detpeak_export.py` has never run

`p4_detsweep_export_f0.json` / `_f1.json` edit 18 is its only reference. Both notebooks were built
but their manifests carry **no `pushed_slug`** — and `cmd_push` writes that key unconditionally on
success (`kaggle_factory.py:355-357`). `notebooks/kaggle_p3_base_loeo_f1/` is in the same
never-pushed state, which matters because it is the fold-1 half of the arm-B-off control pair.
**This is the detection-threshold sweep lane** — the one lane whose class the transfer law says
*can* move the LB — and it has produced nothing.

### K4 — the d1 lane is 75% unbuilt

Six of eight `p3_d1_*` specs have no `out_dir` and no manifest (`p3_d1_smoke_f0/f1`,
`p3_d1_smoke_f0_src/f1_src`, `p3_d1_pilot_f0_src`, `p3_d1_pilot_f1_src`). Only `p3_d1_pilot_f0`
and `p3_d1_pilot_f1` built (2026-08-07). `d1_inject.py`, `d1_aggregate.py` and
`loeo_pregraph_export.py` are referenced only by this lane, so their live surface is those two
kernels. `d1_response_audit.py` is not an injected edit at all — it is the generator input for
`d1_inject.py` (`scripts/d1/gen_d1_inject.py:15-16`), correctly used but filed in the wrong
directory by convention.

### K5 — one confirmed env shadow, correctly neutralised by a second edit

`p8_loosefilter.json` edit 0 (`env`) writes `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT="0.30"` into cell 2;
the base notebook's cell 5 unconditionally reassigns the same variable. Env bodies are *appended*
to the matched cell (`kaggle_factory.py:186`), so cell 2's write is clobbered — the env edit alone
is a no-op. It is rescued by edit 11, a `replace` that rewrites cell 5 from `"0.20"` to `"0.30"`.
The built kernel is correct, but only because two hand-maintained edits agree. This was the
**only** shadowing instance across all built specs and all env vars.

### K6 — supersession: clean

Cross-referencing every `replace_cell` edit's touched cells against all prior edits' touched cells,
using the recorded `edits[].cells` in all 19 manifests: **zero collisions**. No injected module is
overwritten by a later edit in the same spec.

### K7 — the promoted mechanism is missing from the newest specs

`armb_flow_gate.py:32` defaults `BIOHUB_ARMB_FLOW_GATE` **ON**, and seven built specs inject the
module then set the flag to `"0"` (`p3_base_loeo_f0/f1` intentionally, plus
`p3_subvoxel_loeo_f0/_f0_smoke/_f1`, `p4_detsweep_export_f0/f1`, `p4_preilp_loeo_f1`). Meanwhile
commit `3b75f5b` promoted arm B, and the two newest specs — `p7_cleanedge.json` and
`p8_loosefilter.json` — do not inject `armb_flow_gate.py` at all. The promoted mechanism is absent
from the head of the spec line. Not a no-op, but a measurement-substrate inconsistency worth
resolving before the next comparison.

Also: `scripts/kaggle_specs/live_p0b.json` has an **empty edit list** and an `out_dir` pointing at
the base notebook's own directory — a pass-through spec that builds a byte-copy.

---

## 4. Vendored trainer/predictor — `vendor/kaggle-cell-tracking/scripts/`

### Tier 1 — touches the detection surface / candidate set (CAN TRANSFER)

**V1. `predict_unet_transformer.py:70` vs `train_unet_transformer.py:1067` — the trained NMS
radius is silently discarded.** `pool_kernel_um` is written into `config.json` at train time (5.0)
but `load_model` (`predict:154, :200`) never reads it back, and `PredictConfig.pool_kernel_um`
defaults to **3.0**. No notebook in `notebooks/` patches it; only
`scripts/kaggle_preflight/dryrun_predict.py:19-20` sets 3.0 -> 5.0, so **preflight and the
deployed kernel run different NMS radii.** This directly changes how many peaks survive local-max
suppression. Fix: code change (read the key in `load_model`).

**V2. `train_unet_transformer.py:681` vs `predict_unet_transformer.py:285` — train thresholds RAW
LOGITS, predict thresholds SIGMOID.**

```python
# train:681
is_peak = (det_logits == pooled) & (det_logits > det_threshold)          # 0.3 raw logit = p 0.574
# predict:285
is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)   # 0.99 prob     = logit 4.60
```

The detector is trained against a candidate set an order of magnitude denser than the one it is
deployed with; the transformer never saw the inference-time candidate distribution. Fix: code
change.

**V3. `predict_unet_transformer.py:456, :465, :478` — `max_parents_per_node` cannot bind (the
known theorem, now with its mechanism).** `probs = torch.softmax(raw, dim=0)` normalises over
**sources**, so each column sums to 1 and at most one row can exceed the 0.5 threshold. Under the
deployed `USE_ILP=1`, `__post_init__:89` leaves both greedy limits `None` so `:476/:478` are
skipped entirely. Consequence: **the candidate set handed to the ILP is already injective on
targets — the ILP can only delete edges, never re-route a parent.** This is why the measured
in-degree histogram is `{1: 2,162,040}`.

**V4. `predict_unet_transformer.py:73` — `threshold = 0.5` has no CLI arg and no env override.**
No `--threshold` in `main()` (`:594-635`); the only assignment anywhere
(`cfg.threshold = edge_candidate_threshold` in the p3_armb notebook) is gated behind
`if secondary_weights_text:` and is dead without a secondary model. Every submission uses a
hardcoded 0.5 on a source-normalised softmax: for a frame with N sources the column mass is
spread, and any target with a diffuse parent distribution yields **zero** candidate edges. This is
the dominant recall limiter on the candidate set, and it is not sweepable. Fix: config change (add
CLI/env).

**V5. `predict_unet_transformer.py:293, :495-496` — sub-voxel detection precision destroyed.**
Coordinates are cast to `int16` on the downsampled grid (`:293`) and only then multiplied by the
stride `(1,4,4)` (`:495`), so every exported Y/X coordinate is an exact multiple of 4 original
voxels. Nothing downstream recovers it. Fix: code change. (This is the gap the sub-voxel refine
lane is attacking from the far end.)

**V6. `train_unet_transformer.py:625, :627` — `det_threshold` and `max_match_distance` are never
passed by any caller.** Both call sites (`train_epoch:848-854`, `evaluate:948-954`) pass only
`voxel_size`, `pool_kernel_um`, `frame_index`, `window_size`; there is no `--det-threshold` /
`--match-um` CLI arg. `max_match_distance` is set only in
`scripts/kaggle_edits/h1r_det_train.py:166`, a different trainer. The matching radius that defines
**every edge target** is a frozen literal (5.0).

**V7. `train_unet_transformer.py:336-340` — a second unreachable `F.interpolate`, the twin of the
known `_load_frame` one, this time in the training loader.** `target_shape = vm.image_shape[1:]`
and `io.py:96` builds `image_shape` as `ceil(s/d)` per axis — exactly the strided shape. Fix:
delete.

**V8. `predict_unet_transformer.py:347-353, :410-412` — the last-window top-up and the
`seen_pairs` dedupe are both dead at the deployed `W=2`.** `stride = max(W-1, 1) = 1`, so
`window_starts[-1] + W < T` is always False and every `(t_src, t_tgt)` pair is unique. Harmless
today; they are the only thing standing between a `W>2` experiment and a silent frame gap.

**V9. `train_unet_transformer.py:1014 vs :1237` — `det_loss_weight` default differs 10x between
the function (1e1) and the CLI (1e0), and the CLI help text claims 1e1.** Same drift at
`:1007 vs :1224` (`lr` 1e-3 vs 1e-4) and `:1009 vs :1228` (`num_workers` 4 vs 8, help says 4).
This is the weight on the detection head relative to the edge loss. Fix: config change.

### Tier 2 — NO-TRANSFER (edge-weight / division / plumbing)

**V10. `predict_unet_transformer.py:304, :512, :551, :670` — `unet_batch_size` is fully plumbed
CLI -> `predict_video()` and never used** (zero references in the body; every window is encoded at
batch size 1 at `:370`). It is also threaded through the deployed notebook as
`BIOHUB_UNET_BATCH_SIZE` and logged into the run manifest — a manifest field recording a knob that
does nothing.

**V11. `train_unet_transformer.py:817, :936` — `pos_feats` is built, padded, collated, moved to
GPU and never read** (`detect_and_match:742` replaces them with `_pos_embed_torch`). Dead all the
way back to `extract_pos_features:230` and `pad_window:264, :271`. **This dead code masks a live
bug:** `augmentations.py:57-61` `flip_augment` flips `coords` but leaves `pos_feats` computed from
the unflipped coords (`train:271`). Anyone who "fixes" the unused-variable warning by wiring
`pos_feats` back in immediately starts feeding mirrored positional embeddings.

**V12. `train_unet_transformer.py:60-61` — the empty-target edge loss returns a fresh leaf tensor,
so those batches contribute no gradient.** `torch.tensor(0.0, requires_grad=True, ...)` is not
connected to the graph, and `compute_batch_loss:88` averages this detached constant into the batch
loss — the edge branch trains on fewer samples than `n_samples` reports (`:897, :899`).

**V13. `train_unet_transformer.py:985-991` + `:98-99` — `test_loss` is diluted.** Pairs with no GT
contribute 0.0 to the numerator but 1 to `n_pairs`, so `test_loss` is deflated by exactly the
empty-pair fraction — the quantity that changes as the detector improves. Combined with the
already-known degenerate `score = test_acc * test_recall` (`:1180`), **no reported validation
number in this trainer is a usable model-selection signal.** Add `:1181`
`is_best = score >= best_score` with `best_score = 0.0`: `>=` means every tie overwrites, so with
a near-constant `test_acc` this is effectively "save the last epoch".

**V14. `predict_unet_transformer.py:136, :142, :482-485` — `edge_dist` is computed per candidate
edge and has zero consumers anywhere.** `geffs_to_csv.py:37-47` drops it, `csv_to_geffs.py:36-43`
never restores it, the ILP (`:556-561`) uses only `edge_prob`, and `src/biotrack/wrapper.py:166`
**recomputes** it as `edge_distance_um`. It is also computed *before* the
`coords[:,1:] *= ds_arr` rescale at `:495`, so it is in downsampled-voxel units — wrong by 4x in
Y/X even if read. Fix: delete.

**V15. `predict_unet_transformer.py:72, :455-458` — the `sigmoid` edge-activation branch is
unreachable.** `edge_activation` defaults `"softmax"`, has no CLI arg, and is set nowhere.

**V16. `predict_unet_transformer.py:88-93` — comment says 1/1, code sets parents 1 / children 2.**
The class docstring at `:63`/`:65` also disagrees with itself. Divisions *are* permitted in the
non-ILP path.

**V17. `predict_unet_transformer.py:120-127` — the docstring says it "avoids `add_node_attr_key`"
and then calls it three times**; the `-999999.0` sentinel is dead because `bulk_add_nodes`
(`:129-132`) supplies z/y/x for every node. Same dead sentinel at `csv_to_geffs.py:23`.

**V18. `train_unet_transformer.py:1018, 1101-1108, 1113, 1119` — the seeding path is
unreachable.** No `--seed` CLI arg and `main():1270-1290` never passes it, so `g` and
`worker_init_fn` are always `None`. Compounding it, `FrameWindowDataset.__getitem__:343` builds a
fresh unseeded `np.random.default_rng()` per item. **Every training run in this repo is
unreproducible.** Fix: config change (`--seed`) plus code change at `:343`.

**V19. `train_unet_transformer.py:1019 -> :1079` — `max_frames` is plumbed but always `None`**,
making the truncation at `:387-389` unreachable. Same for `augmentations` (`:1021`, always
`DEFAULT_AUGMENTATIONS`, no way to ablate) and `brightness_augment(shift_range=0.1)`
(`augmentations.py:13`, never passed).

**V20. Guards that cannot trip.** `train:559` `if n_gt <= 0: continue`
(`get_window_data:221-222` already returns `None` for zero-GT frames); `train:667-671`
`max(1, k if k % 2 == 1 else k + 1)` where `k` is already `max(1, ...)`; `train:308-309`
`if max_nodes is None` (both construction sites at `:1097, :1098` pass it);
`simple_node_transformer.py:135-140, 207-208` unbatched `(N, D)` path (both callers pass batched).

---

## 5. Ranked action list

Ranked by whether fixing it could plausibly move the LB under the measured transfer law.

**Can transfer (detection surface / candidate set):**

1. **V4 + V3** — `threshold = 0.5` hardcoded on a `dim=0` softmax. The candidate set reaching the
   ILP is already injective on targets and its recall is set by an unsweepable literal. Config
   change (add CLI/env) then a sweep. Largest candidate-set lever found.
   `predict_unet_transformer.py:73, :456, :465`.
2. **V1** — `pool_kernel_um` trained at 5.0, deployed at 3.0, preflight at 5.0. Code change (a few
   lines in `load_model`). Changes how many detections exist.
   `predict_unet_transformer.py:70, :154, :200`.
3. **W6** — every kernel asks for `GAP_CLOSE_MAX_GAP=2` and `wrapper.py:482` silently gives it 1.
   Gap close inserts synthetic nodes, so this is a candidate-set change. The cheap version of the
   experiment is to enable the already-written `recover_strict_gap2` (W4) rather than to edit
   `:482`. One-line code change either way.
4. **V2** — logit-vs-sigmoid threshold mismatch between train (0.3 logit) and predict (0.99 prob).
   Code change; large, and it explains a train/deploy distribution shift in the detector.
5. **K3** — `detpeak_export.py` / the detection-threshold sweep lane was built and **never
   pushed**. This is the only lane in the transferring class and it has produced zero data. No
   code change needed — just run it.
6. **V5** — 4-voxel Y/X quantisation at `predict:293, :495`. Code change; the sub-voxel lane is
   currently attacking this from downstream, which is strictly harder.
7. **W9** — `GAP_DENSITY_ADAPTIVE` is off in the library and on in every kernel. Config change. A
   local/LB divergence in a detection-surface stage is exactly the shape of the LOEO-LB gap.

**Cannot transfer (curiosities — record, do not spend GPU on):**

8. **W1** — the raw-edge pre-filter is provably inert. Important as a *negative* result: any past
   or future sweep of `OUTPUT_EDGE_MAX_UM` above 10 um measured, and will measure, exactly 0.000.
   Do not re-run it.
9. **W2b / restore_learned_divisions** — 189,363 ILP division edges destroyed, 0 learned divisions
   in any submission, all 6,303 output divisions manufactured geometrically (80.7% of them
   duplicating an edge the model already proposed). The fix exists and is pushed. Division term,
   measured 0.000 transfer. **Low.**
10. **W2/W3/W5/W7/W12, V10-V20** — dead by construction, no scientific consequence.
11. **V13 + the known `test_acc * test_recall`** — no validation number this trainer prints is a
    model-selection signal. This does not move the LB by itself, but it means every "best"
    checkpoint we have selected was effectively the last epoch.

**Hygiene (no LB effect, real future-bug risk):**

12. **K1** — three `kaggle_edits/*.py` files are inlined duplicates the factory never opens;
    `subvoxel_refine.py`'s `PATCH = r'''...'''` wrapper would become a genuine silent no-op if
    someone wired it in. Either wire them in properly (stripping the wrapper) or delete them.
13. **K2** — delete or archive `node_budget_stage.py` and `volume_guard_linefit.py`; note that the
    latter means the A3 volume-guard fix is in no kernel, matching `OUTPUT_VOLUME_GUARD=0`.
14. Add an `expect` check to `append_cell` (`kaggle_factory.py:222-228`), the one unvalidated edit
    kind, and a `--check` that fails when a `kaggle_edits/*.py` is referenced by no spec.
15. **K7** — arm B was promoted (`3b75f5b`) but is absent from `p7_cleanedge` / `p8_loosefilter`
    and explicitly off in seven measurement lanes. Resolve the substrate before the next
    comparison.
