"""Compute per-crop statistics from the LOCAL extracted train pairs (ground truth).

Scans data/train/*.geff directly -- NOT the API manifest, which was incomplete
(captured 129 of the true 199 pairs). Writes reports/inventory/embryo_stats.csv with,
per crop: n_nodes, n_edges, n_div (out-degree>=2), t range, estimated_n_nodes, label_fraction.
"""

import csv
from pathlib import Path

import tracksdata as td
from geff import GeffMetadata

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "reports" / "inventory" / "embryo_stats.csv"


def local_pairs() -> list[str]:
    zarr = {p.stem for p in TRAIN.glob("*.zarr")}
    geff = {p.stem for p in TRAIN.glob("*.geff")}
    return sorted(zarr & geff)


def stats(name: str) -> dict:
    geff_dir = TRAIN / f"{name}.geff"
    res = td.graph.IndexedRXGraph.from_geff(geff_dir)
    g = res[0] if isinstance(res, tuple) else res
    na = g.node_attrs()
    ids = g.node_ids()
    n_div = int(sum(1 for d in g.out_degree(ids) if d >= 2))
    meta = GeffMetadata.read(geff_dir)
    est = (meta.extra or {}).get("estimated_number_of_nodes")
    est = float(est) if est is not None else float("nan")
    n_nodes = g.num_nodes()
    return {
        "embryo": name,
        "n_nodes": n_nodes,
        "n_edges": g.num_edges(),
        "n_div": n_div,
        "t_min": int(na["t"].min()),
        "t_max": int(na["t"].max()),
        "n_timepoints": na["t"].n_unique(),
        "estimated_n_nodes": est,
        "label_fraction": (n_nodes / est) if est and est == est else float("nan"),
    }


def main() -> None:
    names = local_pairs()
    print(f"{len(names)} LOCAL train pairs")
    rows = []
    for i, name in enumerate(names, 1):
        try:
            rows.append(stats(name))
        except Exception as ex:
            print(f"  ERROR {name}: {ex}")
        if i % 25 == 0 or i == len(names):
            print(f"  {i}/{len(names)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    fam = Counter(r["embryo"].split("_")[0] for r in rows)
    tot_e = sum(r["n_edges"] for r in rows)
    tot_d = sum(r["n_div"] for r in rows)
    print(f"\nWrote {OUT}: {len(rows)} crops, {tot_e} edges, {tot_d} divisions")
    print("by family:", dict(fam))


if __name__ == "__main__":
    main()
