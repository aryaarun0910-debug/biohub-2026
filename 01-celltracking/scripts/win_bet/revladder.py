r"""THE REVERSED CEILING LADDER: what NODE correction is worth at TODAY's association.

WHY THIS EXISTS
---------------
``FACT-0368``'s ladder perfects ASSOCIATION FIRST and then nodes, so its detection rung is what
remains AFTER association is already perfect. That is an ORDER-DEPENDENT oracle increment and it
is NOT the value of a detector swap. ``FACT-0371`` computed the other half of the same forward
order - perfect association on today's NODES. Nobody had computed the REVERSE: perfect NODES on
today's EDGES. That is the quantity a detector swap actually buys under a frozen consumer, and
this module measures it.

THE ARMS - all five scored by the OFFICIAL patched scorer, on the complete deployed chain output
------------------------------------------------------------------------------------------------
``control``
    the deployed champion export, untouched. A control that does not reproduce the registry's
    recorded numbers HALTS the packet; nothing else here is readable without it.

``perfect_localisation``
    every node that the official one-to-one 7 um matcher already pairs with a GT cell is MOVED to
    that cell's exact GT coordinate. The node set, the node COUNT and the entire edge list are
    untouched, so this arm is count-fair by construction. It prices the share of deployed edge
    error charged only because an endpoint sits far enough from its cell to match the wrong one,
    or none - the FACT-0270 "mislinked endpoints are geometrically wrong" claim, read as a
    ceiling. It cannot relink anything, because the topology is frozen.

``perfect_membership_additive``
    every ANNOTATED GT cell that no deployed node matches is INSERTED at its GT coordinate.
    Nothing is deleted and no edge is changed. This is the honest, identifiable half of node
    correction - a better detector finding cells the deployed one misses - and it is very nearly
    count-fair. Inserted nodes arrive with NO edges, because the consumer is frozen; that is the
    measurement, not an oversight.

``perfect_membership_full``
    the node set becomes EXACTLY the annotated GT set: unmatched deployed nodes are deleted,
    missing GT cells inserted, matched nodes KEEP THEIR DEPLOYED COORDINATES (localisation is the
    other arm's business, and holding them apart is what makes the two arms readable together).
    Today's edges are transported wherever both endpoints survive.
    READ THIS ARM COUNT-FAIR AND NOWHERE ELSE. It deletes ~99% of predicted nodes, because the
    annotation is sparse (``FACT-0354``: 861 annotated cells against an estimate of 6,362 on one
    crop), so its adjusted score collects an enormous under-production bonus (``FACT-0191``) that
    no method can earn. The panel reports its count channel separately for exactly this reason.
    It is ALSO GENEROUS IN A SECOND WAY that must be stated: deleting an unmatched node removes
    every false-positive edge that pointed at it, so an ASSOCIATION error whose victim happens to
    be an unannotated node is scored here as if node correction had fixed it. Both biases inflate
    this arm, which is the right direction for a disqualifier.

``calibration_perfect``
    GT nodes with GT edges. ``FACT-0368``'s harness scores this at exactly 1.0000 raw on both
    folds, so it is THIS instrument's own calibration and it is checked before any other arm is
    read.

WHAT THIS BOUNDS, AND WHAT IT DOES NOT
--------------------------------------
NODE CORRECTION UNDER THE FROZEN CONSUMER, only. A real detector also changes localisation, node
features, candidate availability and association decisions. A miss here does NOT kill
detector-derived representations or candidate-generation effects, and must never be quoted as if
it did.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The official-scorer path, byte-for-byte the one ceiling_ladder.py used for FACT-0368/FACT-0371.
from ceiling_ladder import _rows, match_gt_to_pred, oracle_edges_for, score_rows  # noqa: E402

HEARTBEAT = "REVLADDER_CROP_COMPLETE"
ARMS = (
    "control",
    "perfect_localisation",
    "perfect_membership_additive",
    "perfect_membership_full",
    "calibration_perfect",
)
FOLDS = {
    0: {"prefix": "44b6", "csv": Path(r"C:\temp\p28_f0\loeo_split0_champion.csv.gz"), "n_crops": 71},
    1: {"prefix": "6bba", "csv": Path(r"C:\temp\p28_f1\loeo_split1_champion.csv.gz"), "n_crops": 128},
}


def load_gt(gt_geff: Path):
    """GT node array (t,z,y,x) and GT edges as row-index pairs."""
    from biotrack.metric import load_graph

    g = load_graph(gt_geff)
    n = g.node_attrs().to_pandas()
    ids = n["node_id"].to_numpy().astype(np.int64)
    row = {int(v): i for i, v in enumerate(ids)}
    arr = n[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    e = g.edge_attrs().to_pandas()
    edges = [
        (row[int(s)], row[int(t)])
        for s, t in zip(e["source_id"], e["target_id"])
        if int(s) in row and int(t) in row
    ]
    return arr, edges


def _as_float_coords(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("z").cast(pl.Float64), pl.col("y").cast(pl.Float64), pl.col("x").cast(pl.Float64)
    )


def _new_node_rows(crop: str, coords: np.ndarray, first_id: int, template: pl.DataFrame) -> pl.DataFrame:
    """Node rows for inserted GT cells, in *template*'s column order."""
    k = len(coords)
    frame = pl.DataFrame({
        "id": [-1] * k,
        "dataset": [crop] * k,
        "row_type": ["node"] * k,
        "node_id": [first_id + i for i in range(k)],
        "t": coords[:, 0].astype("int64").tolist(),
        "z": coords[:, 1].astype("float64").tolist(),
        "y": coords[:, 2].astype("float64").tolist(),
        "x": coords[:, 3].astype("float64").tolist(),
        "source_id": [-1] * k,
        "target_id": [-1] * k,
    })
    return frame.select([c for c in template.columns])


def crop_arms(crop: str, gt_geff: Path, deployed: pl.DataFrame) -> dict[str, dict]:
    """Score all five arms for one crop. Every arm goes through the official scorer."""
    gt, gt_edges = load_gt(gt_geff)
    node_rows = deployed.filter(pl.col("row_type") == "node")
    edge_rows = deployed.filter(pl.col("row_type") == "edge")

    node_f = _as_float_coords(node_rows)
    edge_f = _as_float_coords(edge_rows)
    dep_ids = node_f["node_id"].to_numpy().astype(np.int64)
    dep = node_f.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)

    # THE matcher: official one-to-one, 7 um, at (1.625, 0.40625, 0.40625). Shared with
    # FACT-0354/0355/0357 and FACT-0368, so "matched" means here what it means there.
    mapping = match_gt_to_pred(gt, dep)                 # gt row -> deployed row
    matched_pred_rows = sorted(set(mapping.values()))
    missing_gt_rows = [g for g in range(len(gt)) if g not in mapping]
    next_id = int(dep_ids.max()) + 1 if len(dep_ids) else 0

    out: dict[str, dict] = {}

    # ---- ARM 1: the control, byte-identical to the shipped export --------------------------
    out["control"] = score_rows(deployed, gt_geff)

    # ---- ARM 2: perfect localisation, topology and node count frozen -----------------------
    zyx = dep[:, 1:].copy()
    for g, p in mapping.items():
        zyx[p] = gt[g, 1:]
    loc_nodes = node_f.with_columns(
        pl.Series("z", zyx[:, 0]), pl.Series("y", zyx[:, 1]), pl.Series("x", zyx[:, 2])
    )
    out["perfect_localisation"] = score_rows(pl.concat([loc_nodes, edge_f]), gt_geff)

    added = _new_node_rows(crop, gt[missing_gt_rows], next_id, node_f) if missing_gt_rows else None

    # ---- ARM 3a: perfect membership, additive half (identifiable, ~count-fair) --------------
    add_nodes = pl.concat([node_f, added]) if added is not None else node_f
    out["perfect_membership_additive"] = score_rows(pl.concat([add_nodes, edge_f]), gt_geff)

    # ---- ARM 3b: perfect membership, full (count-inflated; read count-fair) -----------------
    keep_mask = np.zeros(len(dep_ids), dtype=bool)
    keep_mask[matched_pred_rows] = True
    kept = node_f.filter(pl.Series(keep_mask))
    kept_ids = kept["node_id"].to_list()
    surviving = edge_f.filter(
        pl.col("source_id").is_in(kept_ids) & pl.col("target_id").is_in(kept_ids)
    )
    full_nodes = pl.concat([kept, added]) if added is not None else kept
    out["perfect_membership_full"] = score_rows(pl.concat([full_nodes, surviving]), gt_geff)

    # ---- ARM 4: the instrument's own calibration - must be raw edge Jaccard 1.0000 ----------
    ident = {i: i for i in range(len(gt))}
    out["calibration_perfect"] = score_rows(
        _rows(crop, gt, oracle_edges_for(gt_edges, ident)), gt_geff
    )

    for arm in list(out):
        out[arm] = {"dataset": crop, **out[arm]}
    out["_diag"] = {
        "dataset": crop,
        "n_gt_nodes": int(len(gt)),
        "n_gt_edges": int(len(gt_edges)),
        "n_deployed_nodes": int(len(dep_ids)),
        "n_deployed_edges": int(edge_rows.height),
        "n_matched": int(len(mapping)),
        "n_missing_gt": int(len(missing_gt_rows)),
        "n_unmatched_deployed": int(len(dep_ids) - len(matched_pred_rows)),
        "n_edges_surviving_full": int(surviving.height),
    }
    return out


def run_crop(fold: int, crop: str, out: Path, slice_dir: Path, gt_dir: Path) -> None:
    src = slice_dir / f"f{fold}_{crop}.parquet"
    if not src.is_file():
        raise SystemExit(f"missing pre-sliced crop {src} - run `prep` first")
    deployed = pl.read_parquet(src)
    gt_geff = gt_dir / f"{crop}.geff"
    if not gt_geff.exists():
        raise SystemExit(f"missing GT for {crop}")
    t0 = time.time()
    arms = crop_arms(crop, gt_geff, deployed)
    payload = {
        "heartbeat": HEARTBEAT,
        "fold": fold,
        "crop": crop,
        "seconds": time.time() - t0,
        "arms": arms,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    tmp.replace(out)      # atomic: two workers meeting on one crop cost a duplicate, never a tear


def prep(fold: int, slice_dir: Path) -> int:
    """Split the champion export into per-crop parquet ONCE.

    Profiled first, per the packet's PROFILE-BEFORE-REFACTORING rule: the gzipped champion CSV
    takes ~3.7 s to read whole, so reloading it inside every crop would burn ~4.4 minutes per
    worker per fold and hold ~300 MB resident in each - on a box that is memory-bound above three
    workers (FACT-0440). Slicing once removes both.
    """
    cfg = FOLDS[fold]
    slice_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_csv(cfg["csv"])
    crops = sorted(frame["dataset"].unique().to_list())
    if len(crops) != cfg["n_crops"]:
        raise SystemExit(f"fold {fold}: export holds {len(crops)} crops, expected {cfg['n_crops']}")
    for crop in crops:
        frame.filter(pl.col("dataset") == crop).write_parquet(slice_dir / f"f{fold}_{crop}.parquet")
    print(f"PREP_COMPLETE fold={fold} crops={len(crops)} -> {slice_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fold", type=int, required=True, choices=(0, 1))
    p.add_argument("--slice-dir", type=Path, default=Path(r"C:\temp\revladder\slices"))
    r = sub.add_parser("crop")
    r.add_argument("--fold", type=int, required=True, choices=(0, 1))
    r.add_argument("--crop", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--slice-dir", type=Path, default=Path(r"C:\temp\revladder\slices"))
    r.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    args = ap.parse_args()

    if args.cmd == "prep":
        return prep(args.fold, args.slice_dir)
    run_crop(args.fold, args.crop, args.out, args.slice_dir, args.gt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
