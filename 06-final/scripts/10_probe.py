"""G3b: does the frozen encoder already know about mitosis?

L2 logistic probe on UNet features at labelled division parents vs controls
matched, within the SAME film and frame, on (log intensity, local detection
density). Unmatched, a probe relearns "this region is bright" and returns the
AUC 0.73 raw intensity already gives -- the matching is the experiment.

Leave-one-embryo-out, because train/test are embryo-disjoint and a random split
measures a strictly easier question.

Offsets t-2..t+2 are scored separately: two independent observers believed the
anaphase signature precedes the labelled split frame and neither tested it.

Gate (ABORT_RULES G3b):
  AUC >= 0.85   signal is in the frozen features -> skip the CNN
  0.70 - 0.85   build the patch model, with evidence it is needed
  AUC <  0.70   a learned division classifier is no longer the main bet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

df = pd.read_csv("artifacts/probe_dataset.csv.gz")
OFFS = [-2, -1, 0, 1, 2]
print(f"rows {len(df):,} | positives {int(df.label.sum())} | films {df.film.nunique()}")
print(df.groupby("embryo").label.agg(["size", "sum"]).to_string(), "\n")


def auc_loeo(cols, label="", n_boot=200):
    """Fit on one embryo, score the other. Report both directions and pooled."""
    per, preds, truth = {}, [], []
    for held in sorted(df.embryo.unique()):
        tr, te = df[df.embryo != held], df[df.embryo == held]
        if te.label.nunique() < 2 or tr.label.nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(tr[cols].values), tr.label.values)
        p = clf.predict_proba(sc.transform(te[cols].values))[:, 1]
        per[held] = roc_auc_score(te.label.values, p)
        preds.append(p); truth.append(te.label.values)
    y = np.concatenate(truth); p = np.concatenate(preds)
    pooled = roc_auc_score(y, p)
    rng = np.random.default_rng(0)
    bs = [roc_auc_score(y[i], p[i]) for i in
          (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
          if len(np.unique(y[i])) > 1]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    fold_str = "  ".join(f"{k}:{v:.3f}" for k, v in per.items())
    print(f"  {label:<34} AUC {pooled:.3f}  [{lo:.3f}-{hi:.3f}]   per-fold {fold_str}")
    return pooled


print("=== baselines (what the corpus already measured) ===")
auc_loeo(["intensity"], "raw peak intensity")
auc_loeo(["intensity", "density"], "intensity + local density")

print("\n=== frozen UNet features, one offset at a time ===")
best = {}
for o in OFFS:
    cols = [f"f{o:+d}_{j}" for j in range(32)]
    best[o] = auc_loeo(cols, f"features at t{o:+d}" + ("   <- labelled split" if o == 0 else ""))

print("\n=== combinations ===")
auc_loeo([f"f{o:+d}_{j}" for o in OFFS for j in range(32)], "all offsets t-2..t+2")
auc_loeo([f"f{o:+d}_{j}" for o in (-1, 0) for j in range(32)], "t-1 and t only")
auc_loeo([f"f{o:+d}_{j}" for o in OFFS for j in range(32)] + ["intensity", "density"],
         "all offsets + intensity + density")

peak = max(best, key=best.get)
print(f"\nsignal peaks at offset t{peak:+d} (AUC {best[peak]:.3f}); "
      f"labelled split frame t+0 gives {best[0]:.3f}")
if peak < 0:
    print("  -> the anaphase hypothesis is SUPPORTED: shift every division window earlier.")
else:
    print("  -> no evidence the signal precedes the labelled split frame.")
