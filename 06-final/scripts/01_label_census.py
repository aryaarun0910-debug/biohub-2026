"""Tabulate every training .geff. Graph arrays only -- no image pixels.

Produces artifacts/films.csv, the table every later decision keys on.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from biohub import io

root = io.dataset_root()
rows, skipped = [], []
t0 = time.perf_counter()

for stem in io.films("train"):
    geff = root / "train" / (stem + ".geff")
    try:
        g = io.read_geff(geff)
    except Exception as e:
        skipped.append((stem, str(e)[:70]))
        continue
    src, out_deg = np.unique(g["edges"][:, 0], return_counts=True)
    # a labelled division is a node with >= 2 outgoing edges
    rows.append({
        "film": stem,
        "embryo": stem.split("_")[0],
        "divisions": int((out_deg >= 2).sum()),
        "max_out_degree": int(out_deg.max()) if len(out_deg) else 0,
        "labelled_nodes": int(len(g["ids"])),
        "labelled_edges": int(len(g["edges"])),
        "estimated_nodes": g["estimated_number_of_nodes"],
        "frames": int(g["t"].max()) + 1 if len(g["t"]) else 0,
    })

films = pd.DataFrame(rows).set_index("film")
films["labelled_fraction"] = films.labelled_nodes / films.estimated_nodes
films.to_csv("artifacts/films.csv")

print(f"read {len(films)} films in {time.perf_counter()-t0:.1f}s   skipped={len(skipped)}")
for s in skipped[:5]: print("  SKIP", s)
print()
print("TOTAL labelled divisions:", int(films.divisions.sum()))
print("TOTAL labelled nodes:    ", int(films.labelled_nodes.sum()))
print("TOTAL labelled edges:    ", int(films.labelled_edges.sum()))
print("max out-degree anywhere: ", int(films.max_out_degree.max()))
print()
agg = films.groupby("embryo").agg(
    films=("divisions", "size"), divisions=("divisions", "sum"),
    films_with_div=("divisions", lambda s: int((s > 0).sum())),
    labelled_nodes=("labelled_nodes", "sum"), est_nodes=("estimated_nodes", "sum"),
)
agg["median_labelled_pct"] = films.groupby("embryo").labelled_fraction.median() * 100
print(agg.to_string(float_format=lambda v: f"{v:,.2f}"))
print()
print("films with zero labelled divisions:", int((films.divisions == 0).sum()), "of", len(films))
print("division-event resolution 0.1/D if you use ALL films:", f"{0.1/films.divisions.sum():.5f}")
