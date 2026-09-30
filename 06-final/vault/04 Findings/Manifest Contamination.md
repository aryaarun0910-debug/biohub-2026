---
tags:
  - finding
  - gotcha
  - key
---

# The All-Train Manifest Claim

A discussion post argued the public stack's detectors saw all 199 train videos,
so any local hold-out is in-sample and local measurement cannot be trusted.

**Verified true**, from the manifests in `weights/`:

```
unet_transformer/split_0/split_manifest.json
  method: unet_transformer_alltrain_seed314159_v1
  train:  199        test: 40 -- a strict SUBSET of train (40/40 overlap)
```

All 8 [[Validator Films]] are in that 199.

**But the inference does not follow.** All **4** [[Test Films]] — the films the
submission is actually scored on — are in the *same* 199 train list. In-sample-ness
is **symmetric**, so it cannot by itself explain a local-vs-board sign flip.

What differs is **composition**:
| | validator | scored |
|---|---|---|
| films | 8 | 4 |
| GT divisions | **12** | **3** |
| weighting | roughly even | dominated by one 70k-node film |

**A sharper fact nobody mentioned:** the [[DeepCenter]] split is not a hold-out at
all — it is **entirely embryo-level**. Train = all 71 `44b6` films, val = all 128
`6bba` films. DeepCenter has never seen a single 6bba frame, and 6bba carries 125
of 151 divisions and most of the scored weight.

**Their counter-example was right about the mechanism, though.** A sub-voxel peak
refinement gained locally and lost on the board — a coordinate-localisation
change, the same family as [[s09]], which [[Scored Films Measurement]]
independently shows flipping sign.

**Verdict:** the fact is real, the causal story is wrong, the warning lands.
Fixed by [[Scored Films Measurement]], which removes the need to argue about it.
