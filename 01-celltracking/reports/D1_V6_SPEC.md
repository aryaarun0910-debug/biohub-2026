# v6 export spec — TTA-consistent features, without touching association

**Written:** 2026-08-06 · Prerequisite reading: `reports/D1_V5_STATUS.md`.
**Status:** specification. Not yet implemented, not pushed.

---

## 0. The trap this spec exists to avoid

Every research lane independently proposed the same v6 fix: *accumulate `unet_out` over the
same views as `det_logits`, then divide*. One lane wrote it as literal code:

```python
unet_flip, det_flip = model.encode(imgs.flip(dims))   # was: _, det_flip
unet_out = unet_out + unet_flip.flip(dims)            # <-- WRONG
...
unet_out = unet_out / 4
```

**That silently destroys the 0.915 association substrate.** `unet_out` is not a
detection-only tensor:

```
L372: unet_out, det_logits = model.encode(imgs)
L442:     unet_out[:, f_idx],     p_coords_src, p_mask_src,     -> _index_features -> predict_edges
L445:     unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,     -> _index_features -> predict_edges
L490: del unet_out
```

Both association reads happen **after** the TTA block. Overwriting `unet_out` therefore
changes **every edge feature in the pipeline**, which is exactly the roadmap's single largest
named downside risk — *"gaining detections while destroying the working P3 association
representation"* — reintroduced by the repair for B3.

It would also be nearly invisible: detection metrics would move, node counts would move, and
the association change would be attributed to the new node population rather than to a
silently altered feature basis.

> **Rule for v6: `unet_out` is read-only. The TTA-mean feature lives in a separate tensor and
> is visible only to the D1 audit.** Association stays bit-identical, and that is asserted,
> not assumed.

---

## 1. The corrected TTA block

Replaces the primary TTA block in base-notebook cell 5 (`notebooks/kaggle_p0b_clean913_revtime/`).
The **secondary** block is left alone — it is inert under `BIOHUB_LOEO_ARM=strict`, which
resets `BIOHUB_SECONDARY_DETECTION_WEIGHT` to `"0"` (`loeo_retarget.py:127-130`).

```python
if cfg.det_tta:
    _nv = 1
    _unet_tta = unet_out.clone()                    # identity view; unet_out NEVER reassigned
    for dims in [(-1,), (-2,), (-2, -1)]:
        imgs_flip = imgs.flip(dims)
        _u_flip, det_flip = model.encode(imgs_flip)
        for f in range(W):
            det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
        _unet_tta = _unet_tta + _u_flip.flip(dims)
        del imgs_flip, det_flip, _u_flip
        _nv += 1
    for _k in (1, 3):
        imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
        _u_rot, det_rot = model.encode(imgs_rot)
        for f in range(W):
            det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
        _unet_tta = _unet_tta + torch.rot90(_u_rot, -_k, dims=(-2, -1))
        del imgs_rot, det_rot, _u_rot
        _nv += 1
    imgs_t = imgs.transpose(-1, -2)
    _u_t, det_t = model.encode(imgs_t)
    for f in range(W):
        det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
    _unet_tta = _unet_tta + _u_t.transpose(-1, -2)
    del imgs_t, det_t, _u_t
    _nv += 1
    imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
    _u_at, det_at = model.encode(imgs_at)
    for f in range(W):
        det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
    _unet_tta = _unet_tta + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))
    del imgs_at, det_at, _u_at
    _nv += 1
    for f in range(W):
        det_logits[f] = det_logits[f] / _nv
    _unet_tta = _unet_tta / _nv
else:
    _unet_tta = unet_out
```

### Why the inverse transforms are correct on a 6-D tensor

`det_logits[f]` is `(1,1,Z,Y,X)`; `unet_out` is `(1,W,C,Z,Y,X)`. Every operation acts only on
the trailing `(-2,-1) = (Y,X)` axes, so each applies unchanged to the 6-D tensor. Each feature
inverse mirrors its logit inverse exactly:

| forward view | logit inverse | feature inverse |
|---|---|---|
| `flip(dims)` | `.flip(dims)` | `.flip(dims)` |
| `rot90(k)` | `rot90(-k)` | `rot90(-k)` |
| `transpose(-1,-2)` | `.transpose(-1,-2)` | `.transpose(-1,-2)` |
| `rot90(1)∘transpose` | `transpose` then `rot90(-1)` | `transpose` then `rot90(-1)` |

The last row is a composition, so its inverse applies in **reverse order**. Rotations and
transposes swap Y and X; the output grid is 64×64 so this is well-defined, but the code must
not depend on squareness — assert `Y == X` at entry.

---

## 2. The parity assertion — the whole point of v6

`detect_head` is `Conv3d(32,1,kernel_size=1)`, a pointwise affine map, so it commutes with
both the spatial permutations and the mean:

```
detect_head(mean_v  aligned_feature_v)  ==  mean_v  aligned_logit_v
```

Inside the v6 smoke, apply the **fold-routed checkpoint head** to `_unet_tta` and assert
numerical equality with the deployed post-TTA `det_logits`. Report **max abs error, p99.9,
median, and the dtype of every operand.** Expected float32 rounding is ~1e-6 on a 32-term dot
product; gate at `max_abs_err <= 1e-4` with a **hard abort**, not a warning.

Interpretation of a near-miss, which is the dangerous case:

- residual small **and uncorrelated** with the logit ⇒ float noise, proceed, record envelope;
- residual small but **systematically signed or correlated** with the logit ⇒ real mismatch
  (partial accumulation, dtype cast, stale head) ⇒ **abort**;
- `max_abs_err >> 1e-4` ⇒ wrong tensor pairing ⇒ **abort and name the suspect**.

There is no "proceed with caveat" branch.

**This is achievable in the smoke because the secondary blend is off.** Under
`BIOHUB_LOEO_ARM=strict` the 8-view average is the *only* difference between `w·x + b` and
`rows.logit`. It would **not** hold on the public deployment path, where the field is
`0.525·primary + 0.475·aligned_secondary` behind a per-frame retention guard — a separate,
recorded transfer hazard, not a parity blocker here.

---

## 3. Audit wiring

`d1_inject.py` anchor A2 currently passes `unet_out[0, f_idx]`. v6 passes **both**:

```python
_d1_audit_frame(
    ds_path.stem, ds_path.parent, t, det_logits[f_idx][0],
    _unet_tta[0, f_idx],        # post-TTA detector representation
    unet_out[0, f_idx],         # identity-view = the ASSOCIATION representation
    cfg.det_threshold, pool_k, voxel_size, downsample,
)
```

Emitted as four arrays per crop:

| file | meaning |
|---|---|
| `{crop}__feat_tta_mean_gt.npy` | 32-D at the sampled voxel, **8-view TTA mean** |
| `{crop}__feat_tta_mean_max.npy` | 32-D at the strongest nearby local max, TTA mean |
| `{crop}__feat_idview_gt.npy` | identity view — **the association representation** |
| `{crop}__feat_idview_max.npy` | identity view at the strongest nearby local max |

**The identity-view arrays must never be described as the post-TTA detector representation.**
They are retained because `predict_edges` genuinely reads the identity view, so they are the
correct substrate for association work — v5 exported them under the detector's name, which is
the whole of B3.

---

## 4. Remaining v6 requirements

| # | requirement | why |
|---|---|---|
| 1 | **`raise`, not `print`**, if the TTA patch fails to apply | today the guard prints `"TTA WARNING: block not found - using default 4-way"` and silently degrades the view set |
| 2 | manifest records `tta_view_set`, `n_views`, `grid_zyx`, `n_frames`, `n_uniform_per_frame`, `estimated_number_of_nodes`, `checkpoint_sha256`, `split` | the view count is currently in **no** manifest field; the loader must gate on it |
| 3 | replace the `feat_max -> feat_gt` fallback (`d1_response_audit.py:213`) with an explicit **NaN sentinel** | today non-GT rows silently get "feature at a random voxel" while GT rows get "feature at a local max" — a ~100% label-correlated artefact if the two are ever contrasted |
| 4 | add counters `n_local_max_in_frame`, `n_subthr_localmax_in_frame`, `n_accepted_in_frame` | fixes the ranking denominator; the required AUC bar swings ~0.86 -> ~0.99 across its plausible range |
| 5 | add `dist_to_nearest_gt_um` per row | the masked-loss arm cannot be built without it |
| 6 | assert `Y == X` before any rot90/transpose view | the inverse transforms assume it |
| 7 | declared parquet schema + column contract | **DONE** (trap 21) |

**Not in v6:** per-node `det_logit` and per-target pre-threshold probabilities for the
association work. They are ~0 extra GPU and would be valuable, but they belong to the
association track, which is blocked pending a detector change. Recorded so the decision is
deliberate.

---

## 5. Memory

The accumulator `_unet_tta` is one extra `(1, W, 32, Z, Y, X)` float32 tensor. At W=2 and a
64x64x64 output grid that is **32 MiB/frame, 64 MiB/window**. The per-view `_u_*` tensors were
already materialised — they were simply bound to `_` and discarded — so the marginal increase
is the accumulator alone, against a measured deployed peak of ~601 MB.

**Measure it on the 3-crop smoke anyway and record peak VRAM.** If it binds, solve it
explicitly (e.g. accumulate in-place, or fp16 accumulation with an fp32 master).
**No silent fallback to identity-view features under any circumstance.**

---

## 6. Gate before any full-199 launch

All of:

- immutable manifests complete 3/3
- CPU M/C/T/L/D postprocessor parity (`scripts/d1_postprocess.py`), 52/52 on `44b6_0113de3b`
- **checkpoint-head vs TTA-feature logit parity**, max abs error reported with dtypes
- **association bit-identity**: pregraph node and edge sets identical to v5 for all 3 crops.
  This is the assertion that proves `unet_out` was not disturbed. Any difference means §0's
  trap was walked into
- H0 truly frozen and fold-routed
- no byte-identical implemented arms
- required-input failures tested (a null `pi_crop` must raise; a missing temporal table must
  BLOCK the arm, not degrade it)
- source, schema, checkpoint and scorer hashes recorded
- runtime and peak VRAM acceptable

Then launch the combined full-199 export **once**. Do not run full v5 and then repeat 199
crops for v6.
