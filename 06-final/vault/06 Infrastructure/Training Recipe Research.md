---
tags:
  - infra
  - key
---

# Training Recipe — what the discussions, the weights and the web actually say

## The public model's own config (it ships with the weights)

`weights/biohub-temporal-unet3d-seed314159-v1/.../training_config.json`:

| | value |
|---|---|
| `batch_size` | **8** |
| `epochs_target` | **500** |
| `learning_rate` | 1e-4 |
| `unet_layers` | [32, 64, 128], out 32 |
| `downsample` | [1, 4, 4] · `window_size` 2 · `pool_kernel_um` **5.0** |
| `det_loss_weight` | 1.0 · `det_neg_weight` 0.01 |
| augmentations | `brightness_augment`, `flip_augment` |
| seed | 314159 · `train_datasets` 199 · `validation_datasets` 40 |

## The single most important line in the discussions

From `741749`, by someone running a 24-video stratified hold-out for three weeks:

> **"post-processing settings chosen locally DO carry over to the LB, but the
> local ranking of two trained models does NOT"**

That is the justification for this whole detour. Our post-processing conclusions
failed not because post-processing is unmeasurable locally, but because we
measured against training labels ([[Why Every Offline Tier Was Wrong]]). With an
honest hold-out they should transfer. Their local CV had **r ≈ −0.2** with the LB
for ranking *models* in the 0.93–0.95 band — so never rank weights locally.

Also from that thread, matching our own [[Error Budget]]: **node-count
calibration predicted LB movement far better than missed detections did.**

## Deliberately NOT doing: synthetic pretraining

Worth **+0.012–0.018** and described as "the only big lever I found" — pretrain
on the CC0 synthetic set (discussion `732103`, 18.5 GB, 165k labelled divisions),
then fine-tune. Notably **497 sequences beat all 2,174** ("what helps is having
some of that data, not having more of it").

**We are not using it**, because it is right for a better *detector* and wrong
for an *instrument*: it would make our error profile differ from the public
stack, which is exactly what we need to match. Revisit only if we chase score
directly.

## MPS, measured rather than assumed

| batch | steady state | samples/s |
|---|---|---|
| **8** | **5.19 s/it** | **1.54** |
| 32 | ~80 s/it | 0.40 |

**Batch 8 wins by ~4×** — large batches exceed unified-memory bandwidth on 3D
volumes. So the public's batch size is also the fastest here.

⚠ **Benchmarks need warm-up.** An 8-iteration probe measured 52 → 31 → 19 → 14
s/it and never reached steady state, which made the real 5.19 s/it look like 12.
Discard the first ~10 iterations or the number is cold-launch overhead.

Related: [[Local Pipeline On The Mac]], [[Evidence Tiers]]
