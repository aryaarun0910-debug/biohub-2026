"""Build embryo-grouped cross-validation splits.

CRITICAL: the hidden test is embryo-DISJOINT, and "dataset names begin with the
embryo identifier" (the prefix before the first underscore, e.g. 44b6 / 6bba).
Each embryo is sliced into many ~100-frame crops. Splitting folds by individual
crop would leak the same embryo into both train and validation (the dossier's
"validation trap"), giving optimistic scores that collapse on the private 71%.

So the honest, hidden-test-faithful CV is LEAVE-ONE-EMBRYO(-FAMILY)-OUT: one fold
per embryo family, testing on that family and training on the rest. With the two
training embryos (44b6, 6bba) this yields 2 folds. This is exactly the
"leave-one-embryo-out where feasible" the dossier recommends.

Writes data/dataset_splits.json in the organizer format (list of folds, each
{"train": [...], "test": [...]}), consumed by vendor/.../scripts/evaluate.py.

Usage: python scripts/build_splits.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "reports" / "inventory" / "embryo_stats.csv"
OUT = ROOT / "data" / "dataset_splits.json"


def family(embryo: str) -> str:
    return embryo.split("_")[0]


def main() -> None:
    rows = list(csv.DictReader(STATS.open(encoding="utf-8")))
    for r in rows:
        r["n_edges"] = int(r["n_edges"])
        r["n_nodes"] = int(r["n_nodes"])
        r["n_div"] = int(r["n_div"])

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[family(r["embryo"])].append(r)

    fams = sorted(by_fam)
    all_emb = sorted(r["embryo"] for r in rows)

    folds = []
    for f in fams:
        test = sorted(r["embryo"] for r in by_fam[f])
        train = [e for e in all_emb if e not in set(test)]
        folds.append({"train": train, "test": test})

    OUT.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}  (leave-one-embryo-out: {len(fams)} folds over {len(all_emb)} crops)")
    for i, f in enumerate(fams):
        rs = by_fam[f]
        e = sum(r["n_edges"] for r in rs)
        n = sum(r["n_nodes"] for r in rs)
        d = sum(r["n_div"] for r in rs)
        print(f"  fold {i} (test=embryo {f}): {len(rs):3d} crops | edges {e:6d} | nodes {n:6d} | divisions {d}")


if __name__ == "__main__":
    main()
