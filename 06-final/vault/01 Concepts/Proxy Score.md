---
tags:
  - concept
---

# Proxy Score

The name for [[The Metric]] evaluated **offline**, on films where ground truth
is held — the 8 [[Validator Films]].

Two instruments compute it:
- Kaggle's **in-kernel validator**, inside the notebook run. This is the
  instrument that correctly rejected [[s01]].
- The [[Local Harness]], in seconds instead of 1.75 h.

They agree closely but not exactly: the harness omits [[Single Parent Repair]],
the short-track rescue and the [[DeepCenter]] vetoes, which costs it one extra
division false positive. See [[Harness Validated Against Kernel]].

**Proxy is not the board.** The board has spoken 3-for-3 on the division axis and
0-for-3 on the edge axis — see [[Axis Priors]].
