"""Freeze the leave-one-embryo-out split. Run ONCE. Never retuned on.

Train/test are embryo-disjoint and train holds exactly two embryos, so LOEO is
the only split that asks the question the board asks. A random film split
measures within-embryo transfer, which is strictly easier.
"""
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

OUT = Path("artifacts/loeo_split.json")
if OUT.exists():
    sys.exit(f"{OUT} already exists. The split is frozen; refusing to overwrite.")

films = pd.read_csv("artifacts/films.csv").set_index("film")

folds = {}
for emb, grp in films.groupby("embryo"):
    folds[emb] = {
        "held_out_films": sorted(grp.index),
        "n_films": len(grp),
        "D_labelled_divisions": int(grp.divisions.sum()),
        "labelled_nodes": int(grp.labelled_nodes.sum()),
        "labelled_edges": int(grp.labelled_edges.sum()),
        "event_resolution_0.1_over_D": round(0.1 / max(grp.divisions.sum(), 1), 5),
        "median_labelled_pct": round(float(grp.labelled_fraction.median() * 100), 2),
    }

# A fast-iteration subset: division-bearing films from BOTH embryos, largest
# counts first (no selection of the same size carries more divisions).
dev = []
for emb, grp in films.groupby("embryo"):
    d = grp[grp.divisions > 0].sort_values(["divisions", "labelled_nodes"], ascending=False)
    dev += list(d.index[:10])
dev = sorted(dev)
dev_D = int(films.loc[dev].divisions.sum())

split = {
    "frozen_on": "2026-09-19",
    "rule": "leave-one-embryo-out; gate reports BOTH folds separately, never pooled",
    "folds": folds,
    "dev_subset": {
        "films": dev,
        "n_films": len(dev),
        "D": dev_D,
        "event_resolution": round(0.1 / max(dev_D, 1), 5),
        "purpose": "fast iteration only. Never the gate. Never a promotion criterion.",
        "embryo_balance": films.loc[dev].embryo.value_counts().to_dict(),
    },
    "warnings": [
        "Fold 44b6 carries D=26: one division event moves total score by 0.0038. "
        "Any division delta smaller than that is unreadable on this fold.",
        "Label density differs ~12x between folds (0.77% vs 9.71%), so edge Jaccard "
        "is NOT comparable across folds. Report per fold, never averaged into one number.",
        "Deleting nodes looks free on 44b6 and is not. The node-count term is forbidden.",
    ],
}
OUT.write_text(json.dumps(split, indent=2))
print(f"wrote {OUT}")
print("sha256:", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
print()
for emb, f in folds.items():
    print(f"  fold hold-out {emb}: {f['n_films']:>3} films, D={f['D_labelled_divisions']:>3}, "
          f"resolution {f['event_resolution_0.1_over_D']}, {f['median_labelled_pct']}% labelled")
print(f"  dev subset:        {len(dev):>3} films, D={dev_D}, resolution {0.1/dev_D:.5f}, "
      f"balance {split['dev_subset']['embryo_balance']}")
