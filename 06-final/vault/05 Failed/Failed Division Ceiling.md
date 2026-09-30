---
tags:
  - failed
---

# Failed: the division detection ceiling

**Refuted — the premise was arithmetically wrong.**

Hypothesis: the detector's pooling kernel merges sister cells that are too close,
imposing a hard floor on division recall.

What is actually true: `pool_kernel_um` **quantises**. Values 3.0, 4.0 and 5.0 all
produce the identical (3,3,3) kernel, and the true floor is **3.25 µm Chebyshev**.
Only **1 of 151** divisions sits below it, and **zero** are actually lost to it.

The instance that motivated the whole investigation turned out to be a *linker*
failure with all three nodes correctly detected.

A good reminder to check the arithmetic of a mechanism before building on it.
