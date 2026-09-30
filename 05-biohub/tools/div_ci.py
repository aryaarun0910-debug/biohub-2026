#!/usr/bin/env python3
"""Confidence intervals for division Jaccard -- so the assay can tell signal from noise.

The whole strategy rests on measuring division changes the 0.947 family cannot see. But our own
corpus holds only 151 divisions across 199 datasets, and 26 of those are 44b6. A change of "+5
true positives" sounds decisive and may be nothing. Without an interval we would chase noise with
better instruments than anyone else -- which is worse than not measuring, because it looks rigorous.

Bootstrap is over DATASETS, not divisions: divisions within a dataset share a detector, a linker
and an embryo, so resampling them independently would understate the variance badly.
"""
from __future__ import annotations
import numpy as np


def divJ(tp, fp, fn) -> float:
    d = tp + fp + fn
    return tp / d if d else float("nan")


def bootstrap_divJ(per_dataset, n=10000, alpha=0.05, seed=0):
    """per_dataset: list of (tp, fp, fn) per dataset. Returns (point, lo, hi)."""
    a = np.asarray(per_dataset, float)
    if not len(a):
        return float("nan"), float("nan"), float("nan")
    point = divJ(*a.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    s = a[idx].sum(1)
    boot = np.where(s.sum(1) > 0, s[:, 0] / np.maximum(s.sum(1), 1e-9), np.nan)
    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_delta(before, after, n=10000, alpha=0.05, seed=0):
    """Paired bootstrap on the SAME datasets -- the only honest way to compare two configs.

    Returns (delta, lo, hi, p_worse) where p_worse is the bootstrap fraction in which the change
    was not an improvement. Pairing matters: dataset-to-dataset variance dwarfs the effect, so an
    unpaired interval would hide a real gain in shared noise.
    """
    A, B = np.asarray(before, float), np.asarray(after, float)
    assert A.shape == B.shape, "paired comparison needs the same datasets in the same order"
    delta = divJ(*B.sum(0)) - divJ(*A.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), (n, len(A)))
    da = A[idx].sum(1); db = B[idx].sum(1)
    ja = da[:, 0] / np.maximum(da.sum(1), 1e-9)
    jb = db[:, 0] / np.maximum(db.sum(1), 1e-9)
    d = jb - ja
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(delta), float(lo), float(hi), float((d <= 0).mean())


def _demo():
    """What our current numbers can and cannot resolve."""
    rng = np.random.default_rng(0)
    # 199 datasets, 151 divisions, roughly our observed 126/32/36 split
    base = []
    for _ in range(199):
        k = rng.poisson(151 / 199)
        tp = rng.binomial(k, 0.83); fn = k - tp; fp = rng.poisson(32 / 199)
        base.append((tp, fp, fn))
    p, lo, hi = bootstrap_divJ(base)
    print(f"  divJ {p:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  width {hi-lo:.4f}")
    print(f"  -> with 151 divisions the assay resolves changes of about {(hi-lo)/2:.3f} divJ,")
    print(f"     i.e. {(hi-lo)/2*0.1:.4f} of score. Smaller claimed gains are not measurable here.")
    # a +5 TP change, paired
    after = [(tp + (1 if i < 5 else 0), fp, max(0, fn - (1 if i < 5 else 0)))
             for i, (tp, fp, fn) in enumerate(base)]
    d, dlo, dhi, pw = paired_delta(base, after)
    print(f"\n  PAIRED +5 TP: delta {d:+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]  "
          f"P(not an improvement) {pw:.3f}")
    print(f"  -> pairing is what makes a 5-division change detectable at all.")


if __name__ == "__main__":
    _demo()
