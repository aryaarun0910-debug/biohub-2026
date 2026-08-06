# D1 v5 — corrected status. Three blockers, one of them fatal to D1-F.

**Written:** 2026-08-06 · Supersedes the "v5 smoke PASSED 3/3, D1-F is next" framing in the
previous handoff. **No full-199 export was launched.** `scripts/d1f_probe.py` was never run.

---

## Triage

| area | status |
|---|---|
| kernel wiring, immutable manifests, aggregation | **PASS** |
| raw spherical neighbourhood statistics | **PASS** |
| feature serialization (shape, finiteness, atomicity) | **PASS** |
| M/C/T/L/D partition | **NOT RUN in v5** — now derived CPU-side, see below |
| D1-F representation-vs-head experiment | **INVALID / UNPROVEN** |

---

## The three blockers

### B1 — the kernel never emitted the partition

`{crop}__rows.parquet` has **no `d1_class` column and no `matched` column**. It carries raw
statistics only (`n_lm_7um`, `n_acc_7um`, `n_lm_15um`, `n_acc_15um`, `best7_*`, `best7_15_*`,
`near15_*`, `best15_*`, `k0..k6_*`). `matched` can only come from the scorer's own bipartite
matching, which is a CPU step. The two headline acceptance checks had therefore never been
run on anything.

**Resolved** by `scripts/d1_postprocess.py` (below). Not a defect in the export — a missing
stage that was believed to exist.

### B2 — `scripts/d1f_probe.py` cannot load a v5 artifact and its arms are not distinct

- reads `{crop}__feat_near.npy`; the kernel writes `{crop}__feat_max.npy`
- reads columns `d1_class` and `near_dist_um`; neither exists
- `main()` **fits** H0 instead of reading the checkpoint head, so H0 is not a parity reference
- `build_arm(..., pi_crop=None)` is hardcoded and no temporal logic exists, so **H1, H2, H3
  and H4 are byte-identical arms**. Four identical rows would read as convergent evidence.

**Not repairable by a filename/column patch.** The script is being rewritten with a
capability registry that hard-fails unimplemented arms.

### B3 — features are identity-view, logits are post-TTA. **This is the fatal one.**

`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py`:

```
372:  unet_out, det_logits = model.encode(imgs)          # identity view
379:  if cfg.det_tta:
380:      tta_flips = [(-1,), (-2,), (-2, -1)]
383:          _, det_flip = model.encode(imgs_flip)      # features DISCARDED
385:          det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
388:      det_logits[f] = det_logits[f] / 4
```

The deployed view list is **exactly 4** (identity + 3 planar flips) and **only logits** are
accumulated. `unet_out` is bound once from the identity view and never touched again.
`scripts/kaggle_edits/d1_inject.py:264` then passes `det_logits[f_idx][0]` and
`unet_out[0, f_idx]` to the audit **as though they were a matched pair** — and
`d1_response_audit.py:5` documents the misconception in a comment:
*"post-TTA det_logits[f_idx][0] and the 32-channel unet_out[:, f_idx] are both in scope."*

**Consequence:** `checkpoint_detect_head(feat_max)` reproduces the identity-view logit, not
the deployed post-TTA logit. **H0 parity is mathematically unavailable from a v5 export**, so
the representation-versus-head question is unproven, not merely mislabelled.

**Scope of the damage is exactly bounded:** every neighbourhood statistic is computed from
`logits_1zyx`, which *is* the post-TTA tensor. So **the partition is sound and only the 32-D
features are contaminated.** Locked by `tests/test_d1_postprocess.py`.

---

## The derived partition (v5 smoke, 3 crops)

`scripts/d1_postprocess.py` — versioned (`d1-derived-1`), deterministic, manifest-driven,
treats the raw export as read-only, and calls the scorer's own matching rather than
reimplementing it. Invariants asserted per crop: `M+C+T+L+D == GT` and `M == scorer-matched`.

**Authority for `matched` is measured, not assumed.** Both the post-wrapper submission graph
and the pre-wrapper pregraph are scored. The pregraph reproduces the recorded parity target
**52/52 on `44b6_0113de3b`**; the submission does not. `matched` therefore follows the
pregraph, the detection-honest substrate.

| crop | GT | M | C | T | L | D | submission-matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| `44b6_0113de3b` | 52 | **52** | 0 | 0 | 0 | 0 | 49 |
| `6bba_57b7cc1e` | 1,659 | 1,314 | 242 | 39 | 64 | 0 | 1,307 |
| `6bba_6feb10f0` | 1,368 | 155 | 47 | 810 | 356 | 0 | 149 |
| **total** | **3,027** | **1,469** | **289** | **849** | **420** | **0** | **1,505** |

Of 1,558 unmatched GT nodes: **C 18.5% · T 54.5% · L 27.0% · D 0.0%.**

**Basis tag: 3-crop smoke, IN-FAMILY, diagnostic only.** One crop was deliberately selected
as "extreme". This is *not* a corpus census and must not be quoted as one. Class counts are
diagnostic and may never authorise encoder retraining on their own.

### What it nonetheless says

- **Class D is exactly zero.** Every missed GT node has a local maximum within 15 µm. At this
  radius there is **no evidence of a response-poor representation** on these crops.
- **T dominates (54.5%)** — the response exists within 7 µm but nothing clears
  `det_threshold = 0.96875` (logit 3.434). That is the **calibration / operating point**
  cause, the third of the roadmap's three, and the one that explicitly does *not* justify
  encoder work.
- `6bba_6feb10f0` is the extreme case: 155/1368 matched (11.3%), 810 T (59.2%).

These point *away* from encoder retraining. That is the cheap direction, so the bar for
believing it should be higher, not lower — hence v6 before any conclusion.

### Side finding — the wrapper destroys matched GT nodes

Pregraph-matched **1,521** vs submission-matched **1,505**: the post-wrapper stage
(short-track filtering, isolated-node pruning) **loses 16 GT matches, 1.05%**, on every crop
measured (52→49, 1314→1307, 155→149). Recorded here; not chased this cycle.

---

## Acceptance checks — all passed on the 3-crop v5 smoke

- `M+C+T+L+D == GT` and `M == scorer-matched`, per crop
- every distance column **≤ 15 µm** (observed max **14.982 µm**) — the v4 cube-corner leak
  that produced 24.4 µm is genuinely fixed
- `best7_dist_um ≤ 7`; `best7_15_dist_um` strictly in (7, 15]
- peak parity: `voxel_accepted == is_local_max & over_threshold`;
  `over_threshold == prob > 0.96875`; `is_local_max == (logit == pooled)`
- monotonicity: `n_lm_15um ≥ n_lm_7um ≥ n_acc_7um`, `n_acc_15um ≥ n_acc_7um`
- `subthr_localmax` rows are all local maxima and all under threshold
- features `(n_rows, 32)`, finite, both arrays
- terminal per-crop manifests agree with the aggregate on every shared field
- downloaded manifests are **sha256-identical** to the archived copies

---

## Next — v6, then a single combined full-199 launch

1. **v6 TTA-consistent feature export.** Inverse-transform every encoder feature map to base
   coordinates and accumulate over the **same view list as `det_logits`**, derived from the
   generated notebook rather than assumed. Because `detect_head` is a shared linear 1×1×1
   conv, `detect_head(mean(aligned_features)) == mean(aligned_logits)` exactly — assert that
   numerically inside the smoke and report max absolute error and dtype. Export
   `__feat_tta_mean_gt.npy` / `__feat_tta_mean_max.npy`. Original-view features may be kept
   only under a separate name and must never be called the post-TTA representation. Measure
   the feature-accumulator memory cost on the 3-crop smoke; **no silent fallback**.
2. **Rewrite `d1f_probe.py`** with a capability registry: H0 frozen and fold-routed with a
   hard parity abort; H2 requires a real `pi_crop` (null must raise); H3 requires a real
   temporal table and nonzero temporal loss or it is **BLOCKED, not emitted**; H4 requires
   both. Fixtures must prove each auxiliary term changes the fitted objective — distinct
   labels or config hashes are insufficient. Ranking and calibration reported separately;
   sampled GT/background F1 cannot promote an arm.
3. **Re-run the 3-crop smoke as v6** and gate on: manifests 3/3 · partition parity ·
   checkpoint-head/TTA-feature logit parity · H0 frozen · no byte-identical implemented arms ·
   required-input failures tested · hashes recorded · runtime and peak VRAM acceptable.
4. **Only then** launch the combined full-199 export **once**. Do not run full v5 and then
   repeat 199 crops for v6.

Research lanes are notified: v5 response/probe conclusions are frozen. Nobody may cite v5 as
M/C/T/L/D evidence or consume `d1f_probe.py` output.
