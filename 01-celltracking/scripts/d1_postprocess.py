"""D1 CPU postprocessor -- derives the M/C/T/L/D response partition from a D1 export.

WHY THIS EXISTS. The D1 kernel does NOT emit the partition. It emits raw spherical
neighbourhood statistics per GT node (`n_lm_7um`, `n_acc_7um`, `n_lm_15um`, ...) and no
`matched` column at all. `matched` can only come from the scorer's own bipartite matching,
which is a CPU step. So the partition is DERIVED here, once, deterministically, from
immutable raw inputs.

Partition, over ALL GT nodes:

    M  matched by the scorer's own bipartite matching
    then, for scorer-unmatched GT only:
    C  an ACCEPTED local maximum exists within MATCH_UM
    T  a local maximum exists within MATCH_UM but none is accepted
    L  no local maximum within MATCH_UM, but one exists within SEARCH_UM
    D  no local maximum within SEARCH_UM

Invariants asserted per crop: M+C+T+L+D == GT count, and M == the authoritative
scorer-matched count.

WHICH GRAPH DEFINES `matched` IS ITSELF MEASURED. The deployed submission graph is
post-wrapper and contains synthetic gap-fill nodes that can match a GT node no detector
peak ever found; the pregraph is pre-wrapper. Both are scored and both are reported, so
the choice is evidence rather than assumption. `matched` in the derived rows follows
--authority (default: pregraph, the detection-honest substrate).

The raw export is treated as READ-ONLY. Nothing is written back into it.

Usage
-----
  .venv\\Scripts\\python.exe scripts\\d1_postprocess.py ^
      --fold-dir <downloaded fold dir> --out-dir <derived dir>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402
import tracksdata as td  # noqa: E402

from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph  # noqa: E402
from biotrack.submission import submission_to_graphs  # noqa: E402
from tracking_cellmot.metrics import evaluate  # noqa: E402

DERIVED_SCHEMA_VERSION = "d1-derived-1"
MATCH_UM = 7.0
SEARCH_UM = 15.0
NID = td.DEFAULT_ATTR_KEYS.NODE_ID
MID = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID

# Columns the derivation actually reads. Presence is asserted, so a schema drift in the
# kernel export fails loudly here instead of silently changing the partition.
REQUIRED_ROW_COLUMNS = (
    "dataset", "t", "kind", "n_lm_7um", "n_acc_7um", "n_lm_15um", "gt_z", "gt_y", "gt_x",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scorer_fingerprint() -> dict:
    """Hash the exact scoring code that defines `matched`, plus the matching library."""
    vend = ROOT / "vendor" / "kaggle-cell-tracking" / "src" / "tracking_cellmot"
    files = {p.name: sha256_file(p) for p in sorted(vend.glob("*.py"))}
    return {
        "tracking_cellmot_files": files,
        "tracksdata_version": getattr(td, "__version__", "unknown"),
        "match_um": MATCH_UM,
        "search_um": SEARCH_UM,
        "scale": list(DEFAULT_SCALE),
    }


def classify(matched: bool, n_lm_7: int, n_acc_7: int, n_lm_15: int) -> str:
    """M/C/T/L/D for one GT node. `n_lm_15um` is a SUPERSET of `n_lm_7um` (both spherical)."""
    if matched:
        return "M"
    if n_acc_7 > 0:
        return "C"
    if n_lm_7 > 0:
        return "T"
    if n_lm_15 > 0:
        return "L"
    return "D"


def complete_crops(audit_dir: Path) -> tuple[dict, list[str]]:
    """Resolve crops through the TERMINAL manifests, never by globbing filenames."""
    agg = json.loads((audit_dir / "d1_manifest.json").read_text(encoding="utf-8"))
    if not agg.get("COMPLETE"):
        raise SystemExit(f"aggregate manifest COMPLETE is not true in {audit_dir}")
    per_crop_dir = audit_dir / "manifests"
    crops = []
    for name, entry in sorted(agg["crops"].items()):
        if entry.get("status") != "complete":
            raise SystemExit(f"{name}: aggregate status {entry.get('status')!r}, refusing partial")
        terminal = per_crop_dir / f"{name}.complete.json"
        if not terminal.exists():
            raise SystemExit(f"{name}: no terminal manifest {terminal}")
        t = json.loads(terminal.read_text(encoding="utf-8"))
        for k in ("dataset", "status", "fold", "checkpoint_sha256", "n_rows", "gt_rows"):
            if t.get(k) != entry.get(k):
                raise SystemExit(
                    f"{name}: terminal manifest disagrees with aggregate on {k!r}: "
                    f"{t.get(k)!r} vs {entry.get(k)!r}"
                )
        crops.append(name)
    if not crops:
        raise SystemExit(f"no complete crops in {audit_dir}")
    return agg, crops


def open_maybe_gz(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    tmp = Path(tempfile.mkdtemp(prefix="d1pp_")) / path.name[:-3]
    with gzip.open(path, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return tmp


def matched_gt_ids(pred, gt) -> set[int]:
    """The scorer's OWN matching. Never reimplemented here."""
    if pred.num_nodes() == 0 or pred.num_edges() == 0:
        return set()
    evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    na = pred.node_attrs(attr_keys=[NID, MID])
    out = set()
    for row in na.iter_rows(named=True):
        g = row[MID]
        if g is not None and int(g) != -1:
            out.add(int(g))
    return out


def gt_index(gt) -> dict[tuple[int, int, int, int], int]:
    """(t,z,y,x) -> GT node id, asserted collision-free."""
    na = gt.node_attrs(attr_keys=["t", "z", "y", "x", NID])
    idx: dict[tuple[int, int, int, int], int] = {}
    for r in na.iter_rows(named=True):
        key = (int(r["t"]), int(round(r["z"])), int(round(r["y"])), int(round(r["x"])))
        if key in idx:
            raise SystemExit(f"GT coordinate collision at {key}; join is not 1:1")
        idx[key] = int(r[NID])
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True,
                    help="downloaded kernel output dir (contains d1_audit/ and the graphs)")
    ap.add_argument("--out-dir", required=True, help="derived output dir (created)")
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--authority", choices=("pregraph", "submission"), default="pregraph",
                    help="which graph defines `matched` in the derived rows")
    a = ap.parse_args()

    fold_dir, out_dir, gt_dir = Path(a.fold_dir), Path(a.out_dir), Path(a.gt_dir)
    audit_dir = fold_dir / "d1_audit"
    agg, crops = complete_crops(audit_dir)
    fold = str(agg["fold"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_csv = fold_dir / f"loeo_split{fold}_strict.csv.gz"
    pre_pq = fold_dir / f"pregraphs_split{fold}.parquet"
    for p in (sub_csv, pre_pq):
        if not p.exists():
            raise SystemExit(f"missing required graph artifact: {p}")

    print(f"fold {fold}: {len(crops)} complete crop(s) -> {crops}")
    sub_graphs = submission_to_graphs(pl.read_csv(open_maybe_gz(sub_csv)))
    pre_df = pl.read_parquet(pre_pq).drop("edge_prob")
    pre_graphs = submission_to_graphs(pre_df)

    input_hashes = {
        "d1_manifest.json": sha256_file(audit_dir / "d1_manifest.json"),
        sub_csv.name: sha256_file(sub_csv),
        pre_pq.name: sha256_file(pre_pq),
    }

    census, frames = [], []
    for crop in crops:
        rows_p = audit_dir / f"{crop}__rows.parquet"
        if not rows_p.exists():
            raise SystemExit(f"{crop}: missing {rows_p}")
        input_hashes[rows_p.name] = sha256_file(rows_p)

        df = pl.read_parquet(rows_p)
        missing = [c for c in REQUIRED_ROW_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{crop}: raw rows missing required columns {missing}")
        gtr = df.filter(pl.col("kind") == "gt_centre").sort("t", "gt_z", "gt_y", "gt_x")

        gt = load_graph(gt_dir / f"{crop}.geff")
        n_gt = gt.num_nodes()
        if gtr.height != n_gt:
            raise SystemExit(f"{crop}: {gtr.height} gt_centre rows vs {n_gt} GT nodes")

        idx = gt_index(gt)
        gt_ids = []
        for r in gtr.iter_rows(named=True):
            key = (int(r["t"]), int(round(r["gt_z"])), int(round(r["gt_y"])), int(round(r["gt_x"])))
            if key not in idx:
                raise SystemExit(f"{crop}: gt_centre row {key} has no GT node")
            gt_ids.append(idx[key])
        if len(set(gt_ids)) != len(gt_ids):
            raise SystemExit(f"{crop}: gt_centre rows map to duplicate GT nodes")

        matched: dict[str, set[int]] = {}
        for kind, graphs in (("submission", sub_graphs), ("pregraph", pre_graphs)):
            if crop not in graphs:
                raise SystemExit(f"{crop}: absent from {kind} graph artifact")
            matched[kind] = matched_gt_ids(graphs[crop], load_graph(gt_dir / f"{crop}.geff"))

        auth = matched[a.authority]
        cls = [
            classify(gid in auth, int(r["n_lm_7um"]), int(r["n_acc_7um"]), int(r["n_lm_15um"]))
            for gid, r in zip(gt_ids, gtr.iter_rows(named=True))
        ]
        counts = {k: cls.count(k) for k in ("M", "C", "T", "L", "D")}

        # ---- invariants -------------------------------------------------------------
        total = sum(counts.values())
        if total != n_gt:
            raise SystemExit(f"{crop}: M+C+T+L+D = {total} != {n_gt} GT")
        n_auth = len(auth & set(gt_ids))
        if counts["M"] != n_auth:
            raise SystemExit(f"{crop}: M = {counts['M']} != {n_auth} scorer-matched GT")
        unmatched = counts["C"] + counts["T"] + counts["L"] + counts["D"]
        if unmatched != n_gt - n_auth:
            raise SystemExit(f"{crop}: C+T+L+D = {unmatched} != {n_gt - n_auth} unmatched GT")

        derived = gtr.select(["dataset", "t", "gt_z", "gt_y", "gt_x",
                              "n_lm_7um", "n_acc_7um", "n_lm_15um"]).with_columns([
            pl.Series("gt_node_id", gt_ids, dtype=pl.Int64),
            pl.Series("matched", [g in auth for g in gt_ids], dtype=pl.Boolean),
            pl.Series("matched_submission",
                      [g in matched["submission"] for g in gt_ids], dtype=pl.Boolean),
            pl.Series("matched_pregraph",
                      [g in matched["pregraph"] for g in gt_ids], dtype=pl.Boolean),
            pl.Series("d1_class", cls, dtype=pl.Utf8),
            pl.lit(DERIVED_SCHEMA_VERSION).alias("derived_schema_version"),
            pl.lit(a.authority).alias("match_authority"),
        ])
        frames.append(derived)

        row = {"dataset": crop, "family": crop.split("_")[0], "n_gt": n_gt, **counts,
               "matched_submission": len(matched["submission"] & set(gt_ids)),
               "matched_pregraph": len(matched["pregraph"] & set(gt_ids))}
        census.append(row)
        print(f"  {crop:<16} GT={n_gt:>5}  M={counts['M']:>5} C={counts['C']:>4} "
              f"T={counts['T']:>4} L={counts['L']:>4} D={counts['D']:>5}   "
              f"[sub {row['matched_submission']} | pre {row['matched_pregraph']}]")

    all_derived = pl.concat(frames, how="vertical_relaxed")
    out_pq = out_dir / f"d1_derived_split{fold}.parquet"
    all_derived.write_parquet(out_pq)

    report = {
        "derived_schema_version": DERIVED_SCHEMA_VERSION,
        "fold": fold,
        "match_authority": a.authority,
        "checkpoint_sha256": agg.get("checkpoint_sha256"),
        "crops": crops,
        "census": census,
        "totals": {k: sum(c[k] for c in census) for k in ("n_gt", "M", "C", "T", "L", "D")},
        "scorer": scorer_fingerprint(),
        "input_sha256": input_hashes,
        "derived_sha256": sha256_file(out_pq),
    }
    out_json = out_dir / f"d1_derived_split{fold}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = report["totals"]
    print(f"\ntotals  GT={t['n_gt']}  M={t['M']} C={t['C']} T={t['T']} L={t['L']} D={t['D']}")
    print(f"wrote {out_pq}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
