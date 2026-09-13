#!/usr/bin/env python3
"""EXP-21 -- audit EVERY gate in the 0.947 division selector against 76,217 real divisions.

Their selector rejects a proposal on any of six conditions. We have Zebrahub: the same organism,
densely tracked, with 76k divisions where we know the answer. So each gate can be priced in the
only currency that matters -- what fraction of REAL divisions it throws away -- without a GPU,
their graphs, or a submission.

TEMPORAL NORMALISATION, and it matters: Zebrahub frames are closer together than Kaggle's.
Measured per-frame displacement is median 1.724um (Kaggle) against 1.315um (Zebrahub), a factor
of 1.31. Every distance a gate tests scales with that, so Zebrahub distances are reported BOTH
raw and multiplied by 1.31 into Kaggle-equivalent units. A gate that looks generous on raw
Zebrahub numbers may be tight on Kaggle.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl

ZSCALE = np.array([1.24, 0.439, 0.439])       # Zebrahub voxel size, um (from its ome-zarr .zattrs)
TEMPORAL = 1.724 / 1.315                      # Kaggle / Zebrahub per-frame displacement (EXP-13)
SRC = Path("data/zebrahub")

# their published gates
G = dict(max_um=9.0, sister_max_um=14.0, existing_child_max_um=10.0,
         symmetry_tau=0.6, diverge_um=2.25)


def collect(path):
    df = pl.read_csv(path)
    # schemas differ across the five files: ZSNS001_tail uses ParentTrackID, the rest
    # parent_track_id. Normalise rather than assume -- EXP-13 silently saw only 2 of 5.
    if "parent_track_id" not in df.columns:
        for alt in ("ParentTrackID", "parent_trackid"):
            if alt in df.columns:
                df = df.rename({alt: "parent_track_id"}); break
    assert "parent_track_id" in df.columns, (path.name, df.columns)
    if df.schema["parent_track_id"] == pl.String:          # ZSNS001_tail types it as text
        df = df.with_columns(pl.col("parent_track_id")
                             .cast(pl.Float64, strict=False).fill_null(-1).cast(pl.Int64))
    for c in ("track_id", "t"):
        if df.schema[c] == pl.String:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64))
    kids = (df.filter(pl.col("parent_track_id") != -1)
              .group_by("parent_track_id").agg(pl.col("track_id").unique().alias("ch")))
    kids = kids.filter(pl.col("ch").list.len() == 2)
    pos = {}
    for r in df.iter_rows(named=True):
        pos[(r["track_id"], r["t"])] = np.array([r["z"], r["y"], r["x"]]) * ZSCALE
    first, last = {}, {}
    for r in df.sort("t").iter_rows(named=True):
        tid = r["track_id"]
        p = np.array([r["z"], r["y"], r["x"]]) * ZSCALE
        first.setdefault(tid, (r["t"], p))
        last[tid] = (r["t"], p)

    rows = []
    for r in kids.iter_rows(named=True):
        p = r["parent_track_id"]; a, b = r["ch"]
        if p not in last or a not in first or b not in first:
            continue
        tp, P = last[p]; ta, A = first[a]; tb, B = first[b]
        if ta != tb or ta != tp + 1:
            continue
        da, db = np.linalg.norm(A - P), np.linalg.norm(B - P)
        if da < 1e-9 or db < 1e-9:
            continue
        sis = np.linalg.norm(A - B)
        g1, g2 = pos.get((a, ta + 1)), pos.get((b, tb + 1))
        div = (np.linalg.norm(g1 - g2) - sis) if (g1 is not None and g2 is not None) else np.nan
        # their 'existing child' is whichever daughter the linker already had; both orderings
        # occur, so score the gate on each in turn and take the kinder one.
        rows.append((max(da, db), min(da, db), sis, div))
    return np.array(rows)


allr = []
for f in sorted(SRC.glob("*_tracks.csv")):
    r = collect(f)
    if len(r):
        allr.append(r)
        print(f"  {f.stem}: {len(r):,} clean divisions", flush=True)
R = np.vstack(allr)
arc_max, arc_min, sis, div = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
print(f"\n  TOTAL {len(R):,} divisions\n")

def rate(mask, label, note=""):
    print(f"    {label:<46} rejects {100*mask.mean():>5.1f}% of real divisions {note}")

for scale, tag in ((1.0, "raw Zebrahub"), (TEMPORAL, f"x{TEMPORAL:.2f} Kaggle-equivalent")):
    am, an, ss, dv = arc_max*scale, arc_min*scale, sis*scale, div*scale
    print(f"  === {tag} ===")
    rate(am > G["max_um"],            f"SAFE_DIV_MAX_UM = {G['max_um']} (parent arc)")
    rate(ss > G["sister_max_um"],     f"SAFE_DIV_SISTER_MAX_UM = {G['sister_max_um']}")
    rate(an > G["existing_child_max_um"], f"SAFE_DIV_EXISTING_CHILD_MAX_UM = {G['existing_child_max_um']}")
    den = np.maximum((an + am) / 2.0, 1e-6)
    rate(np.abs(an - am) / den > G["symmetry_tau"], f"SAFE_DIV_SISTER_SYMMETRY_TAU = {G['symmetry_tau']}")
    ok = ~np.isnan(dv)
    rate(dv[ok] < G["diverge_um"],    f"SAFE_DIV_DIVERGE_UM = {G['diverge_um']}", f"(of {ok.sum():,} observable)")
    # everything at once
    comb = ((am > G["max_um"]) | (ss > G["sister_max_um"]) | (an > G["existing_child_max_um"])
            | (np.abs(an - am) / den > G["symmetry_tau"]))
    comb_full = comb.copy(); comb_full[ok] |= (dv[ok] < G["diverge_um"])
    rate(comb, "ALL DISTANCE+SYMMETRY GATES COMBINED")
    rate(comb_full, "ALL GATES INCLUDING DIVERGENCE")
    print()

print("  cap calibration:")
print(f"    their SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375 of edges")
print(f"    Zebrahub true division rate    = 0.00391 of nodes  (15,868/4,057,611)")
print(f"    -> the cap sits ~4% BELOW the biological rate, so it binds even with perfect ranking")

# ---- the same audit IN DOMAIN, on Kaggle's own 151 ground-truth divisions ----
# Zebrahub is Ultrack output on a different embryo at finer temporal sampling, so a rejection
# rate measured there is a hypothesis, not a result. 151 is a small n but it is the real target.
sys.path.insert(0, "reference/royerlab-baseline/src")
import tracksdata as td
KSCALE = np.array([1.625, 0.40625, 0.40625])
rows = []
for p in sorted(Path("data/train_geff").glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    P = {int(r["node_id"]): np.array([r["z"], r["y"], r["x"]]) * KSCALE
         for r in n.iter_rows(named=True)}
    T = {int(r["node_id"]): int(r["t"]) for r in n.iter_rows(named=True)}
    out = {}
    for r in e.iter_rows(named=True):
        out.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    for s, kk in out.items():
        if len(kk) != 2 or s not in P:
            continue
        a, b = kk
        if a not in P or b not in P:
            continue
        da, db = np.linalg.norm(P[a] - P[s]), np.linalg.norm(P[b] - P[s])
        if da < 1e-9 or db < 1e-9:
            continue
        sis = np.linalg.norm(P[a] - P[b])
        ga = out.get(a, []); gb = out.get(b, [])
        dv = np.nan
        if len(ga) >= 1 and len(gb) >= 1 and ga[0] in P and gb[0] in P:
            dv = np.linalg.norm(P[ga[0]] - P[gb[0]]) - sis
        rows.append((max(da, db), min(da, db), sis, dv))
K = np.array(rows)
am, an, ss, dv = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
print(f"\n  === KAGGLE ground truth, IN DOMAIN: {len(K)} divisions ===")
den = np.maximum((an + am) / 2.0, 1e-6)
ok = ~np.isnan(dv)
for label, mask in (
    (f"SAFE_DIV_MAX_UM = {G['max_um']}", am > G["max_um"]),
    (f"SAFE_DIV_SISTER_MAX_UM = {G['sister_max_um']}", ss > G["sister_max_um"]),
    (f"SAFE_DIV_EXISTING_CHILD_MAX_UM = {G['existing_child_max_um']}", an > G["existing_child_max_um"]),
    (f"SAFE_DIV_SISTER_SYMMETRY_TAU = {G['symmetry_tau']}", np.abs(an - am) / den > G["symmetry_tau"]),
):
    print(f"    {label:<46} rejects {100*mask.mean():>5.1f}%")
print(f"    {'SAFE_DIV_DIVERGE_UM = 2.25':<46} rejects {100*(dv[ok] < G['diverge_um']).mean():>5.1f}%"
      f"  (of {ok.sum()} observable)")
comb = ((am > G["max_um"]) | (ss > G["sister_max_um"]) | (an > G["existing_child_max_um"])
        | (np.abs(an - am) / den > G["symmetry_tau"]))
cf = comb.copy(); cf[ok] |= (dv[ok] < G["diverge_um"])
print(f"    {'ALL DISTANCE+SYMMETRY COMBINED':<46} rejects {100*comb.mean():>5.1f}%")
print(f"    {'ALL GATES INCLUDING DIVERGENCE':<46} rejects {100*cf.mean():>5.1f}%")
