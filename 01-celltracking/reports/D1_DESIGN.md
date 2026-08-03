# D1 design — the A/B/D partition is exactly computable, and much cheaper than specified

**Status:** design complete, kernel NOT built, NOT pushed. No GPU spent.

## 1. The partition needs no prominence heuristics

`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py::_detect_cells_pooled`:

```python
logits = det_logits.unsqueeze(0)                      # (1,1,Z,Y,X) raw logits
pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
```

Acceptance is exactly two independent conditions. So at any GT coordinate the D1 class follows
directly, with no local-prominence modelling:

| class | condition at the GT voxel | meaning |
|---|---|---|
| **A** | `logits == pooled` **and** `sigmoid(logits) <= det_threshold` | a genuine local maximum, rejected **only** by the threshold |
| **B** | `sigmoid(logits) > det_threshold` **and** `logits != pooled` | above threshold, **suppressed by the pool** (NMS merge) |
| **D** | neither | no usable response |
| **C** | accepted peak that the wrapper later removed | **already measured at ~0** by Lane D0 |

The exported statistics reduce to: `logit@GT`, `pooled@GT`, `sigmoid(logit)`, `det_threshold`,
plus `max logit within {3,5,7,10,15} µm` and the offset to that maximum. Everything else in the
original D1 brief (prominence, accepted/rejected peak counts, contrast) is *secondary* — useful
for the T/N/TN/S/P arms, not needed for the A/B/D verdict.

## 2. The grid is isotropic in physical space — which makes class B geometrically unlikely

`config["downsample"] = [1, 4, 4]` and `VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)`.

| axis | voxel µm | downsample | **grid step µm** |
|---|---:|---:|---:|
| z | 1.625 | 1 | **1.625** |
| y | 0.40625 | 4 | **1.625** |
| x | 0.40625 | 4 | **1.625** |

**The 4× voxel anisotropy is exactly cancelled by the 4× y/x downsample.** The detector's grid is
isotropic at 1.625 µm, so `max_pool3d` with the default `(3,3,3)` kernel suppresses only within
**±1.625 µm** on every axis.

Local cell spacing is ~6.5 µm (`GAP_DENSITY_REFERENCE_UM`). A suppression radius of 1.625 µm
cannot merge two distinct cells 6.5 µm apart. **Class B should therefore be small**, and D1's real
question is the A-versus-D split.

This also predicts that **lane N (density/scale-aware NMS) has little headroom**: the pool is
already far tighter than cell spacing, and the anisotropy correction the brief asks for is
already present in the downsample. Widening the pool would only *destroy* peaks.

Consistent with Lane D0, which found 40.5% of 6bba misses have no accepted peak within 15 µm — a
distance no ±1.625 µm suppression can explain.

**Revised expectation, stated before the measurement:** A + D dominate, B is small, C ≈ 0. If that
holds, the arms worth replaying are **T** (threshold) and **S** (secondary agreement) and **P**
(track-conditioned); **N** and **TN** are likely dead on arrival.

## 3. Consequence for the decision rule

The brief's branch "if A/B dominate, fix threshold/NMS rather than train a network" should be read
as **A dominates → threshold work; B dominates → NMS work**, and the analysis above says B is
unlikely to dominate. If the measurement returns mostly **D**, M2 is authorised.

A caveat that must survive into M2: the threshold arm **T** is not free. `total_node_ratio` is
−0.1397 and we currently collect a **+0.008633** count-adjustment bonus for under-predicting.
Lowering the threshold adds nodes and gives that back, so T must clear break-even *after* the
bonus is surrendered, not before.

## 4. Kernel plan (not yet built)

Base `notebooks/kaggle_p0a_clean913/biohub-p0a-clean913-repro.ipynb` via `kaggle_factory`, as
`loeo_f1_strict_pregraph` does — it already carries the fold-specific LOEO weights
(`aryaarun07/biohub-oof-weights`, `edge_predictor_best_split_{0,1}.pth`) and the training data
(`kms111201/biohub-cell-tracking-data`).

Patch anchor is the one the retention guard already uses inside `predict_unet_transformer.py`:

```
                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )
```

At that point `det_logits[f]`, `frame_indices[f]`, `ds_path.stem`, `cfg.det_threshold` and
`pool_k` are all in scope. GT geffs are attached, so per-frame GT coordinates can be mapped to the
grid by `(z, y//4, x//4)` and sampled. Export one compact parquet, no heatmap volumes.

**Two views, never mixed:** the deployment checkpoint (diagnostic only, carries training-family
leakage) and the fold-specific LOEO checkpoints (split_0 → held-out 44b6, split_1 → held-out
6bba) for the leak-free generalisation claim.

---

## 5. M2 premise VERIFIED in the deployed training code (no GPU needed)

`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py::compute_detection_loss`:

```python
target = torch.zeros_like(logits)          # everything is background by default
...
target[b, zi, yi, xi] = 1.0                # ONE voxel per GT node, no Gaussian
...
w_neg = (neg_weight / n_neg)               # neg_weight = 0.1
return F.binary_cross_entropy_with_logits(logits, target, weight=weight, ...)
```

**There is no ignore mask and no exclusion radius.** Every voxel that is not an annotated GT
centre receives an explicit "you are background" gradient at weight `0.1/n_neg`. Since annotation
covers **0.655% of cells on 44b6 and 8.529% on 6bba**, that means **91.5–99.3% of real nuclei are
actively trained as background.**

This is exactly the failure mode the sparse/PU literature predicts, confirmed in the deployed code
rather than hypothesised. It converges with the measurements:

- **D0**: 6bba misses 13.28% of GT nodes, 40.5% with no accepted peak within 15 µm (class D).
- **D1 design**: NMS cannot explain it — the pool suppresses only within ±1.625 µm against ~6.5 µm
  cell spacing.
- **Training code**: the detector is explicitly supervised to suppress the cells it then fails to
  detect, from single-voxel targets.

**The two M2 changes in the brief — soft/Gaussian centre targets and a PU-compatible loss — target
precisely the two confirmed defects.** The premise no longer rests on D1's outcome.

### One collapse hypothesis measured and RULED OUT

`compute_detection_loss` warns that GT nodes can collapse to duplicate voxels under the `[1,4,4]`
downsample, which would make them structurally undetectable. Measured over all 199 crops:

| | GT nodes | unique target voxels | collided |
|---|---:|---:|---:|
| ALL | 133,318 | 133,318 | **0 (0.00%)** |
| 44b6 | 20,197 | — | 0 |
| 6bba | 113,121 | — | 0 |

**Zero collisions.** Downsample-induced target collapse is not a contributor and needs no
mitigation in M2.

### Revised M2 ordering, cheapest-first

1. **Masked / ignore-radius loss** (Linajea, *Nat Biotechnol* 2022, **MIT**): compute the detection
   BCE only inside a radius of each annotated centre; unannotated nuclei contribute zero gradient
   instead of a negative one. This is validated in our exact modality — 3D light-sheet whole-embryo
   cell-centre detection from sparse annotations — and is a near-free change to the loss above.
2. **Gaussian/soft centre targets** replacing the single-voxel target.
3. **nnPU risk estimator** only if 1–2 are insufficient. It carries a hard blocker: every source
   treats the class prior `π` as a validation-searched hyperparameter, and we have no densely
   annotated validation region to search it on.

Note the framing correction that matters for any PU attempt: our 0.655–8.529% is the **labelling
frequency**, not the class prior `π` the estimator consumes. Conflating them is the dominant
implementation failure. The strongest quantitative evidence (F1 0.522 vs 0.102 at 1–9% retained
annotation) is 2D region-level, **not** 3D voxel heatmaps, so it is unproven at our sparsity.
