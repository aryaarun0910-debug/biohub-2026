---
tags:
  - finding
  - key
---

# The Gates Are The Wrong Functional Form

The first change in this project to be on the right axis, resolved by a CI, and
validated with no real labels fit.

## Synthetic candidate triples (`scripts/124`)

| arm | AUC | precision | recall |
|---|---|---|---|
| deployed GATES | 0.7126 | 0.171 | 0.153 |
| **learned GEOM** | **0.9233** | **0.641** | **0.572** |
| learned GEOM+IMAGE | 0.9231 | 0.637 | 0.569 |

On the **same seven geometric quantities the gates already read**, a small MLP
nearly **quadruples precision and recall** at a matched operating point. The
gates are not badly tuned — they are **axis-aligned thresholds on a joint
distribution**, which is exactly why §5's sweep could not improve any of them in
any direction.

**The image adds −0.0002 — nothing.** The 6.5 GB synthetic download was not
needed for its pixels; only for its *labels*, which let a model learn the joint
geometry that 304 real events never could.

## Transfer to real divisions (`scripts/125`)

| arm (real) | AUC |
|---|---|
| deployed GATES | 0.6651 |
| **learned GEOM (synthetic-trained)** | **0.8558** |

**AUC difference +0.1907, 95% CI [+0.0907, +0.2740] — excludes zero.** It passes
the [[Both-Sets Rule]]'s successor rule from [[The Board Verdict]]: a resolved
CI, not merely a positive point estimate.

No real label is fit anywhere — trained purely on synthetic — so this is honest
in the same way the [[s08]] instrument test was.

## Caveats, stated up front

- **Only 21 real positives.** GT sparsity (1.63%) means both daughters are
  rarely labelled, so most of the 151 divisions cannot form a valid triple. The
  CI is wide but excludes zero.
- The gate baseline **omits `SAFE_DIV_DIVERGE_UM`** (it needs the frame after
  next), so the real gates are stricter than this baseline and the gap is
  somewhat inflated.
- These are **GT-derived** triples. Deployment runs on *predicted* graphs, which
  is the next test.

Related: [[The Division Lever]], [[Safe Division]], [[Axis Priors]]
