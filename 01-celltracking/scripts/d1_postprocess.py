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

THE SIX v6 REPAIRS (correction C6). Each has a lock in tests/test_d1_postprocess.py.

  1. ROW/FEATURE ALIGNMENT. GT rows are a NON-CONTIGUOUS SUBSET of the emitted rows (96
     non-GT rows follow every frame's GT rows) and the derivation additionally sorts them.
     Both facts break a positional join to the `.npy` feature arrays, silently and with
     plausible-looking output. Measured on the v5 smoke: a naive positional join is wrong
     for 51/52, 1646/1659 and 1354/1368 GT rows; the sort alone moves 22.6% and 46.7% of
     the rows of the two 6bba crops. `sort_rows_with_features` is now the ONLY sanctioned
     reorder and it gathers every feature array through the identical permutation.
  2. STABLE `row_id`. Consumed from the export (0-based, +1 per emitted row, feature-array
     row i == row_id i), asserted, and threaded through raw rows, gathered feature arrays,
     the derived parquet and the report. v5 exports predate it; that case is refused unless
     the operator declares emission order EXPLICITLY with --v5-emission-order-row-id.
  3. MATCH RADIUS BINDING. `MATCH_UM` is bound to the scorer's `MAX_DISTANCE` and checked at
     import against the vendored `evaluate` default, `biotrack.d1_partition`, and (when the
     export records them) the radii the export actually used. A scorer change can no longer
     desynchronise the partition silently.
  4. ZERO-EDGE NODE MATCHING. The vendored `_evaluate` short-circuits when a prediction has
     no edges: it warns, returns SCORE 0.0, and NEVER calls `graph.match`, so
     MATCHED_NODE_ID is never written. The old code read that as "no GT matched" and marked
     every GT of such a crop unmatched. The score is about EDGES; node matching is not. We
     now run the scorer's own `DistanceMatching` with identical parameters in that branch.
  5. `feat_max_valid` SEMANTICS. all-NaN iff invalid, all-finite iff valid, NO infinities
     anywhere -- asserted on load. v5 wrote a COPY OF feat_gt instead of NaN and shipped no
     validity column; that fallback is reconstructed, VERIFIED row by row, and repaired to
     NaN in the derived copy so the derived artifact always satisfies the contract.
  6. ATOMIC OUTPUT, NO MIXED DIRECTORIES. Every output goes through `<name>.partial` +
     os.replace, the JSON report is written last as the completion marker, and an output
     directory holding artifacts from another schema version, another match authority,
     another scorer build, an orphaned artifact or an aborted run is REFUSED, not merged.

Usage
-----
  .venv\\Scripts\\python.exe scripts\\d1_postprocess.py ^
      --fold-dir <downloaded fold dir> --out-dir <derived dir>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import tracksdata as td  # noqa: E402

from biotrack import d1_partition as _d1p  # noqa: E402
from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph  # noqa: E402
from biotrack.submission import submission_to_graphs  # noqa: E402
from tracking_cellmot.metrics import evaluate  # noqa: E402

# Bumped from d1-derived-1: rows now carry `row_id`/`feat_row`/`feat_max_valid`, aligned
# feature arrays are emitted alongside, and zero-edge predictions match their nodes. Old
# artifacts must NOT silently mix with new ones.
DERIVED_SCHEMA_VERSION = "d1-derived-2"

# NOT A LITERAL. The scorer owns the match radius; MATCH_UM follows it and
# assert_radius_binding() below refuses to run if any other component disagrees.
MATCH_UM: float = float(MAX_DISTANCE)
SEARCH_UM: float = 15.0

NID = td.DEFAULT_ATTR_KEYS.NODE_ID
MID = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID

ROW_ID = "row_id"
FEAT_ROW = "feat_row"
FEAT_MAX_VALID = "feat_max_valid"
FEATURE_ARRAYS = ("feat_gt", "feat_max")
SORT_KEYS = ("t", "gt_z", "gt_y", "gt_x")

# Columns the derivation actually reads. Presence is asserted, so a schema drift in the
# kernel export fails loudly here instead of silently changing the partition.
REQUIRED_ROW_COLUMNS = (
    "dataset", "t", "kind", "n_lm_7um", "n_acc_7um", "n_lm_15um", "gt_z", "gt_y", "gt_x",
)

_DERIVED_STEM = "d1_derived_split"
_FOLD_RE = re.compile(rf"^{_DERIVED_STEM}(\d+)")


# ------------------------------------------------------------------ radius binding (C6.3)
def assert_radius_binding() -> dict:
    """Bind MATCH_UM to the scorer and refuse to run if any component has drifted.

    Four independent definitions of the same radius exist in this codebase; if any two
    disagree the partition silently stops meaning what it says. This ties them together.
    """
    vend = inspect.signature(evaluate).parameters["max_distance"].default
    problems = []
    if float(MATCH_UM) != float(MAX_DISTANCE):
        problems.append(f"MATCH_UM {MATCH_UM} != biotrack.metric.MAX_DISTANCE {MAX_DISTANCE}")
    if float(vend) != float(MATCH_UM):
        problems.append(
            f"vendored evaluate(max_distance={vend}) != MATCH_UM {MATCH_UM}")
    if float(_d1p.MATCH_UM) != float(MATCH_UM):
        problems.append(
            f"biotrack.d1_partition.MATCH_UM {_d1p.MATCH_UM} != MATCH_UM {MATCH_UM}")
    if float(_d1p.SEARCH_UM) != float(SEARCH_UM):
        problems.append(
            f"biotrack.d1_partition.SEARCH_UM {_d1p.SEARCH_UM} != SEARCH_UM {SEARCH_UM}")
    if not SEARCH_UM > MATCH_UM:
        problems.append(f"SEARCH_UM {SEARCH_UM} must exceed MATCH_UM {MATCH_UM}")
    if problems:
        raise SystemExit("match radius desynchronised:\n  " + "\n  ".join(problems))
    return {
        "match_um": float(MATCH_UM),
        "search_um": float(SEARCH_UM),
        "scorer_max_distance": float(MAX_DISTANCE),
        "vendored_evaluate_default": float(vend),
        "d1_partition_match_um": float(_d1p.MATCH_UM),
        "d1_partition_search_um": float(_d1p.SEARCH_UM),
    }


RADII = assert_radius_binding()


def assert_export_radii(agg: dict) -> dict:
    """If the export recorded the radii it used, they must equal ours. v5 recorded none."""
    rec: dict = {}
    for key, want in (("match_um", MATCH_UM), ("search_um", SEARCH_UM)):
        got = agg.get(key, agg.get(f"d1_{key}"))
        if got is None:
            rec[key] = None
            continue
        if float(got) != float(want):
            raise SystemExit(
                f"export recorded {key}={got} but this postprocessor uses {want}; the "
                "neighbourhood statistics were computed at a different radius")
        rec[key] = float(got)
    rec["recorded"] = any(v is not None for k, v in rec.items() if k != "recorded")
    return rec


# ----------------------------------------------------------------------------- utilities
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, write_fn) -> Path:
    """Write through `<name>.partial` then os.replace. No reader ever sees a half file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    try:
        write_fn(tmp)
        if not tmp.exists():
            raise SystemExit(
                f"{path.name}: the writer did not produce {tmp.name}. Some writers rename "
                f"what they are given -- np.save APPENDS '.npy' to a PATH argument -- which "
                f"silently defeats the .partial rename. Pass an open file handle instead")
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def save_npy(arr: np.ndarray):
    """A writer for `atomic_write`. MUST take a handle: `np.save(Path)` appends '.npy'."""
    def _w(p: Path) -> None:
        with open(p, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
    return _w


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
        "radius_binding": RADII,
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


# ------------------------------------------------------------- scorer matching (C6.4)
def _match_nodes_only(pred, gt) -> None:
    """Run the matcher the scorer would have run, with identical parameters.

    This mirrors `tracking_cellmot.metrics._evaluate` exactly -- same DistanceMatching
    class, same max_distance, same scale, same reset-before-rematch and same progress
    suppression -- for the branch in which `_evaluate` returns early without matching.
    """
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    if MID in pred.node_attr_keys():
        ids = pred.node_ids()
        pred.update_node_attrs(
            node_ids=ids,
            attrs={MID: -1, td.DEFAULT_ATTR_KEYS.MATCH_SCORE: 0.0},
        )
    prev = get_options().show_progress
    set_options(show_progress=False)
    try:
        pred.match(gt, matching=DistanceMatching(max_distance=MATCH_UM, scale=DEFAULT_SCALE))
    finally:
        set_options(show_progress=prev)


def matched_gt_ids(pred, gt) -> tuple[set[int], str]:
    """The scorer's OWN node matching. Never reimplemented here.

    ZERO-EDGE PREDICTIONS (C6.4). `tracking_cellmot.metrics._evaluate` short-circuits when
    `graph.num_edges() == 0 or graph.num_nodes() == 0`: it warns, returns SCORE 0.0, and
    returns BEFORE `graph.match(...)`, so MATCHED_NODE_ID is never written at all. The old
    code turned that into an empty matched set, which marked every GT node of such a crop
    unmatched and corrupted the partition. The zero score is about EDGES; node matching is
    a separate question and is perfectly well defined -- a node-only prediction whose
    coordinates sit on the GT matches every GT node. Verified against the real 44b6 GT:
    a zero-edge prediction at GT coordinates matches 52/52 via `graph.match`, while
    `evaluate` leaves MATCHED_NODE_ID absent.

    Returns the matched GT node ids and which code path produced them.
    """
    if pred.num_nodes() == 0:
        return set(), "no-nodes"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pred.num_edges() == 0:
            _match_nodes_only(pred, gt)
            path = "distance-matching-zero-edge"
        else:
            evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MATCH_UM)
            path = "evaluate"
    if MID not in pred.node_attr_keys():
        raise SystemExit(
            f"matching contract broken: {MID!r} absent after the {path!r} path")
    na = pred.node_attrs(attr_keys=[NID, MID])
    out = {int(g) for g in na[MID].to_list() if g is not None and int(g) != -1}
    return out, path


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


# ------------------------------------------------------------------- row_id contract (C6.2)
def attach_row_id(df: pl.DataFrame, crop: str, *,
                  allow_v5_emission_order: bool) -> tuple[pl.DataFrame, str]:
    """Consume the export's `row_id`; never guess it.

    Export contract (Agent 2, v6): `row_id` is 0-based, increases by exactly one per
    emitted row, and the i-th row of every feature array is the row with `row_id == i`.
    v5 artifacts have no such column -- that is refused unless the operator declares the
    emission-order assumption explicitly on the command line.
    """
    if ROW_ID in df.columns:
        rid = df.get_column(ROW_ID)
        if rid.null_count():
            raise SystemExit(f"{crop}: {ROW_ID} contains {rid.null_count()} null(s)")
        arr = rid.to_numpy()
        if arr.dtype.kind not in "iu":
            raise SystemExit(f"{crop}: {ROW_ID} dtype {arr.dtype} is not integral")
        ident = np.arange(df.height, dtype=arr.dtype)
        if not np.array_equal(np.sort(arr), ident):
            raise SystemExit(
                f"{crop}: {ROW_ID} is not a bijection onto 0..{df.height - 1} -- it must be "
                f"0-based, dense and unique, because feature-array row i IS the row with "
                f"{ROW_ID} == i")
        # Parquet row ORDER is not part of the contract: every feature read in this module
        # gathers BY row_id, so a re-ordered parquet is still exactly recoverable. Only the
        # bijection above is load-bearing. Which case we are in is recorded, not assumed.
        source = "export" if np.array_equal(arr, ident) else "export-reordered"
        return df.with_columns(pl.col(ROW_ID).cast(pl.Int64)), source

    if not allow_v5_emission_order:
        raise SystemExit(
            f"{crop}: raw rows carry no {ROW_ID!r} column. v5 exports predate the row_id "
            f"contract, so features cannot be joined to labels without an assumption. "
            f"Re-export with the v6 audit, or pass --v5-emission-order-row-id to declare "
            f"EXPLICITLY that this artifact's parquet row order is its emission order. "
            f"Refusing to guess.")
    return (df.with_columns(pl.arange(0, df.height, dtype=pl.Int64).alias(ROW_ID)),
            "v5-emission-order-declared")


# ------------------------------------------------------- feature arrays + contract (C6.5)
def load_features(audit_dir: Path, crop: str, n_rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in FEATURE_ARRAYS:
        p = audit_dir / f"{crop}__{name}.npy"
        if not p.exists():
            raise SystemExit(f"{crop}: missing feature array {p}")
        arr = np.load(p)
        if arr.ndim != 2:
            raise SystemExit(f"{crop}/{name}: expected a 2-D array, got shape {arr.shape}")
        if arr.shape[0] != n_rows:
            raise SystemExit(
                f"{crop}/{name}: {arr.shape[0]} feature rows vs {n_rows} raw rows -- the "
                f"positional row_id contract cannot hold")
        out[name] = arr
    dims = {n: int(a.shape[1]) for n, a in out.items()}
    if len(set(dims.values())) != 1:
        raise SystemExit(f"{crop}: feature arrays disagree on width: {dims}")
    return out


def check_feature_contract(crop: str, name: str, arr: np.ndarray,
                           valid: np.ndarray | None = None) -> np.ndarray:
    """all-NaN iff invalid, all-finite iff valid, and NO infinities anywhere.

    Returns the per-row validity implied by the array itself.
    """
    if np.isinf(arr).any():
        bad = int(np.isinf(arr).any(axis=1).sum())
        raise SystemExit(
            f"{crop}/{name}: {bad} row(s) contain +/-inf. The contract admits exactly two "
            f"row states -- all finite (valid) or all NaN (invalid); infinities are neither")
    finite = np.isfinite(arr).all(axis=1)
    allnan = np.isnan(arr).all(axis=1)
    mixed = ~(finite | allnan)
    if mixed.any():
        raise SystemExit(
            f"{crop}/{name}: {int(mixed.sum())} row(s) mix NaN with finite values; a row is "
            f"valid (all finite) or invalid (all NaN), never partial")
    if valid is not None and not np.array_equal(finite, np.asarray(valid, dtype=bool)):
        n = int((finite != np.asarray(valid, dtype=bool)).sum())
        raise SystemExit(
            f"{crop}/{name}: {n} row(s) disagree with {FEAT_MAX_VALID} -- all-NaN iff invalid")
    return finite


def feat_max_validity(crop: str, df: pl.DataFrame,
                      feats: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Per-row validity of `feat_max`, from the export column when it exists.

    v5 has no `feat_max_valid` and, where no maximum existed, wrote a COPY OF feat_gt
    instead of NaN. That fallback is reconstructed from the exported statistics and then
    VERIFIED row by row: every row this reconstruction calls invalid must literally carry
    the feat_gt vector. If a single row disagrees the v5 semantics are not what the repair
    assumes and we refuse rather than fabricate a validity mask.
    """
    if FEAT_MAX_VALID in df.columns:
        s = df.get_column(FEAT_MAX_VALID)
        if s.null_count():
            raise SystemExit(f"{crop}: {FEAT_MAX_VALID} contains {s.null_count()} null(s)")
        return s.cast(pl.Boolean).to_numpy().astype(bool), "export"

    is_gt = (df.get_column("kind") == "gt_centre").to_numpy()
    n15 = df.get_column("n_lm_15um").fill_null(0).cast(pl.Int64).to_numpy()
    valid = is_gt & (n15 > 0)
    same = np.all(feats["feat_max"] == feats["feat_gt"], axis=1)
    broken = (~valid) & (~same)
    if broken.any():
        raise SystemExit(
            f"{crop}: {int(broken.sum())} row(s) reconstructed as feat_max-invalid do not "
            f"carry the v5 feat_gt fallback vector; v5 feat_max semantics are not what this "
            f"repair assumes")
    return valid, "v5-reconstructed-and-verified"


def apply_validity(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a copy whose invalid rows are all-NaN, so the contract holds downstream."""
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
    out = np.array(arr, dtype=dtype, copy=True)
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


# ------------------------------------------------------------- the only sanctioned sort (C6.1)
def sort_rows_with_features(
    df: pl.DataFrame,
    feats: dict[str, np.ndarray],
    by=SORT_KEYS,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Reorder rows AND gather every feature array through the identical permutation.

    C6: never sort rows without applying the identical permutation to the feature arrays.
    `df` may be any SUBSET of the emitted rows; `feats` are the FULL raw arrays indexed by
    `row_id`, so this single call fixes both misalignment mechanisms -- the non-contiguous
    GT subset and the sort itself. `row_id` is appended as the final sort key so the order
    is total and reproducible even when the spatial keys tie.
    """
    if ROW_ID not in df.columns:
        raise SystemExit(f"cannot reorder rows without {ROW_ID!r}; call attach_row_id first")
    out = df.sort([*by, ROW_ID])
    perm = out.get_column(ROW_ID).to_numpy()
    gathered: dict[str, np.ndarray] = {}
    for name, arr in feats.items():
        if perm.size and (perm.min() < 0 or perm.max() >= arr.shape[0]):
            raise SystemExit(
                f"{name}: row_id range [{perm.min()}, {perm.max()}] escapes the feature "
                f"array of {arr.shape[0]} rows")
        gathered[name] = arr[perm]
    return out, gathered


# ------------------------------------------------------------- output directory guard (C6.6)
def _fold_of(name: str) -> str | None:
    m = _FOLD_RE.match(name)
    return m.group(1) if m else None


def guard_out_dir(out_dir: Path, fold: str, authority: str,
                  scorer: dict, *, overwrite: bool = False) -> list[str]:
    """Refuse a derived directory that would end up mixing runs.

    Rejected: leftovers from an aborted run (`*.partial`), a different
    `derived_schema_version`, a different `--authority`, a different scorer build, an
    artifact with no report (orphan/stale), and re-running a fold already present unless
    --overwrite is given. Different FOLDS may legitimately share a directory -- that is the
    intended way to assemble both splits -- so folds alone are not a conflict.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    partials = sorted(p.name for p in out_dir.glob("*.partial"))
    if partials:
        raise SystemExit(
            f"{out_dir}: an aborted run left {partials}; the directory is in an unknown "
            f"state. Remove it and re-run rather than merging into it")

    reports = sorted(out_dir.glob(f"{_DERIVED_STEM}*.json"))
    seen: list[str] = []
    for rp in reports:
        try:
            r = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"{rp}: unreadable derived report ({exc}); refusing to merge")
        rf = str(r.get("fold"))
        seen.append(rf)
        if r.get("derived_schema_version") != DERIVED_SCHEMA_VERSION:
            raise SystemExit(
                f"{rp.name}: schema {r.get('derived_schema_version')!r} != "
                f"{DERIVED_SCHEMA_VERSION!r}; refusing to mix schema versions in {out_dir}")
        if r.get("match_authority") != authority:
            raise SystemExit(
                f"{rp.name}: match authority {r.get('match_authority')!r} != {authority!r}; "
                f"refusing to mix match authorities in {out_dir}")
        prev = (r.get("scorer") or {}).get("tracking_cellmot_files")
        if prev is not None and prev != scorer.get("tracking_cellmot_files"):
            raise SystemExit(
                f"{rp.name}: was derived with a different scorer build; refusing to mix "
                f"scorer versions in {out_dir}")
        if rf == str(fold) and not overwrite:
            raise SystemExit(
                f"{out_dir} already holds fold {rf}. Pass --overwrite to replace it")

    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or not p.name.startswith(_DERIVED_STEM) or p.suffix == ".json":
            continue
        af = _fold_of(p.name)
        if af is None or af not in seen:
            raise SystemExit(
                f"{p.name}: derived artifact with no matching report in {out_dir}; the "
                f"directory holds output from an incomplete or foreign run")
    return sorted(set(seen))


def _int_col(df: pl.DataFrame, name: str, crop: str) -> np.ndarray:
    s = df.get_column(name)
    if s.null_count():
        raise SystemExit(f"{crop}: {name} has {s.null_count()} null(s) on gt_centre rows")
    return s.cast(pl.Int64).to_numpy()


# ------------------------------------------------------------------------------- driver
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True,
                    help="downloaded kernel output dir (contains d1_audit/ and the graphs)")
    ap.add_argument("--out-dir", required=True, help="derived output dir (created)")
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--authority", choices=("pregraph", "submission"), default="pregraph",
                    help="which graph defines `matched` in the derived rows")
    ap.add_argument("--v5-emission-order-row-id", action="store_true",
                    help="declare EXPLICITLY that a pre-row_id (v5) export's parquet row "
                         "order is its emission order; without this such exports are refused")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace this fold's artifacts in an existing derived dir")
    a = ap.parse_args(argv)

    warnings.filterwarnings("ignore")

    fold_dir, out_dir, gt_dir = Path(a.fold_dir), Path(a.out_dir), Path(a.gt_dir)
    audit_dir = fold_dir / "d1_audit"
    agg, crops = complete_crops(audit_dir)
    fold = str(agg["fold"])
    export_radii = assert_export_radii(agg)
    scorer = scorer_fingerprint()
    guard_out_dir(out_dir, fold, a.authority, scorer, overwrite=a.overwrite)
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
    feat_out: dict[str, list[np.ndarray]] = {n: [] for n in FEATURE_ARRAYS}
    provenance: dict[str, dict] = {}
    feat_cursor = 0

    for crop in crops:
        rows_p = audit_dir / f"{crop}__rows.parquet"
        if not rows_p.exists():
            raise SystemExit(f"{crop}: missing {rows_p}")
        input_hashes[rows_p.name] = sha256_file(rows_p)

        df = pl.read_parquet(rows_p)
        missing = [c for c in REQUIRED_ROW_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{crop}: raw rows missing required columns {missing}")

        # --- row_id + feature arrays, before ANY reordering ---------------------------
        df, rid_source = attach_row_id(
            df, crop, allow_v5_emission_order=a.v5_emission_order_row_id)
        feats = load_features(audit_dir, crop, df.height)
        for name in FEATURE_ARRAYS:
            input_hashes[f"{crop}__{name}.npy"] = sha256_file(audit_dir / f"{crop}__{name}.npy")

        valid, valid_source = feat_max_validity(crop, df, feats)
        check_feature_contract(crop, "feat_gt", feats["feat_gt"],
                               np.ones(df.height, dtype=bool))
        if valid_source == "export":
            check_feature_contract(crop, "feat_max", feats["feat_max"], valid)
        else:
            check_feature_contract(crop, "feat_max", feats["feat_max"], None)
            feats["feat_max"] = apply_validity(feats["feat_max"], valid)
            check_feature_contract(crop, "feat_max", feats["feat_max"], valid)

        # --- GT subset, sorted TOGETHER with its features ------------------------------
        gt_mask = (df.get_column("kind") == "gt_centre").to_numpy()
        gtr, gfeat = sort_rows_with_features(df.filter(pl.col("kind") == "gt_centre"), feats)
        gvalid = valid[gtr.get_column(ROW_ID).to_numpy()]

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
        match_path: dict[str, str] = {}
        for kind, graphs in (("submission", sub_graphs), ("pregraph", pre_graphs)):
            if crop not in graphs:
                raise SystemExit(f"{crop}: absent from {kind} graph artifact")
            matched[kind], match_path[kind] = matched_gt_ids(
                graphs[crop], load_graph(gt_dir / f"{crop}.geff"))

        auth = matched[a.authority]
        n7 = _int_col(gtr, "n_lm_7um", crop)
        a7 = _int_col(gtr, "n_acc_7um", crop)
        n15 = _int_col(gtr, "n_lm_15um", crop)
        cls = [classify(g in auth, int(u), int(v), int(w))
               for g, u, v, w in zip(gt_ids, n7, a7, n15)]
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

        derived = gtr.select([
            pl.col("dataset").cast(pl.Utf8),
            pl.col(ROW_ID).cast(pl.Int64),
            pl.col("t").cast(pl.Int64),
            pl.col("gt_z").cast(pl.Float64), pl.col("gt_y").cast(pl.Float64),
            pl.col("gt_x").cast(pl.Float64),
            pl.col("n_lm_7um").cast(pl.Int64), pl.col("n_acc_7um").cast(pl.Int64),
            pl.col("n_lm_15um").cast(pl.Int64),
        ]).with_columns([
            pl.Series(FEAT_ROW, np.arange(feat_cursor, feat_cursor + gtr.height),
                      dtype=pl.Int64),
            pl.Series("gt_node_id", gt_ids, dtype=pl.Int64),
            pl.Series("matched", [g in auth for g in gt_ids], dtype=pl.Boolean),
            pl.Series("matched_submission",
                      [g in matched["submission"] for g in gt_ids], dtype=pl.Boolean),
            pl.Series("matched_pregraph",
                      [g in matched["pregraph"] for g in gt_ids], dtype=pl.Boolean),
            pl.Series("d1_class", cls, dtype=pl.Utf8),
            pl.Series(FEAT_MAX_VALID, gvalid, dtype=pl.Boolean),
            pl.lit(DERIVED_SCHEMA_VERSION).alias("derived_schema_version"),
            pl.lit(a.authority).alias("match_authority"),
        ])
        frames.append(derived)
        for name in FEATURE_ARRAYS:
            feat_out[name].append(gfeat[name])
        feat_cursor += gtr.height

        provenance[crop] = {
            "row_id_source": rid_source,
            "feat_max_valid_source": valid_source,
            "n_raw_rows": int(df.height),
            "n_gt_rows": int(gtr.height),
            "n_non_gt_rows": int(df.height - int(gt_mask.sum())),
            "feat_dim": int(feats["feat_gt"].shape[1]),
            "n_feat_max_valid": int(gvalid.sum()),
            "match_path": match_path,
        }

        row = {"dataset": crop, "family": crop.split("_")[0], "n_gt": n_gt, **counts,
               "matched_submission": len(matched["submission"] & set(gt_ids)),
               "matched_pregraph": len(matched["pregraph"] & set(gt_ids))}
        census.append(row)
        print(f"  {crop:<16} GT={n_gt:>5}  M={counts['M']:>5} C={counts['C']:>4} "
              f"T={counts['T']:>4} L={counts['L']:>4} D={counts['D']:>5}   "
              f"[sub {row['matched_submission']} | pre {row['matched_pregraph']}]")

    all_derived = pl.concat(frames, how="vertical")
    stacked = {n: np.concatenate(feat_out[n], axis=0) for n in FEATURE_ARRAYS}
    for name, arr in stacked.items():
        if arr.shape[0] != all_derived.height:
            raise SystemExit(
                f"{name}: {arr.shape[0]} rows vs {all_derived.height} derived rows")

    # The published join key. `row_id` is PER-CROP, so only (dataset, row_id) identifies a
    # raw row; `feat_row` is the fold-global index into the emitted feature arrays and must
    # be exactly 0..N-1 in parquet order or the shipped contract is false.
    if all_derived.select(["dataset", ROW_ID]).n_unique() != all_derived.height:
        raise SystemExit(
            f"(dataset, {ROW_ID}) is not unique across the derived rows; the join key "
            f"cannot address a raw row")
    if not np.array_equal(all_derived.get_column(FEAT_ROW).to_numpy(),
                          np.arange(all_derived.height)):
        raise SystemExit(
            f"{FEAT_ROW} is not 0..{all_derived.height - 1} in parquet order; the derived "
            f"feature arrays cannot be joined positionally as documented")
    check_feature_contract("derived", "feat_gt", stacked["feat_gt"],
                           np.ones(all_derived.height, dtype=bool))
    check_feature_contract("derived", "feat_max", stacked["feat_max"],
                           all_derived.get_column(FEAT_MAX_VALID).to_numpy().astype(bool))

    # ---- atomic output; the JSON report is the completion marker, written LAST --------
    out_pq = out_dir / f"{_DERIVED_STEM}{fold}.parquet"
    atomic_write(out_pq, lambda p: all_derived.write_parquet(p))
    out_files = {out_pq.name: sha256_file(out_pq)}
    for name in FEATURE_ARRAYS:
        p = out_dir / f"{_DERIVED_STEM}{fold}__{name}.npy"
        atomic_write(p, save_npy(stacked[name]))
        out_files[p.name] = sha256_file(p)

    report = {
        "derived_schema_version": DERIVED_SCHEMA_VERSION,
        "fold": fold,
        "match_authority": a.authority,
        "checkpoint_sha256": agg.get("checkpoint_sha256"),
        "crops": crops,
        "census": census,
        "totals": {k: sum(c[k] for c in census) for k in ("n_gt", "M", "C", "T", "L", "D")},
        "scorer": scorer,
        "export_radii": export_radii,
        "provenance": provenance,
        "feature_alignment": {
            "row_id_column": ROW_ID,
            "feat_row_column": FEAT_ROW,
            "sort_keys": [*SORT_KEYS, ROW_ID],
            "contract": "derived feature array row i == derived parquet row with feat_row == i",
        },
        "input_sha256": input_hashes,
        "derived_sha256": out_files,
    }
    out_json = out_dir / f"{_DERIVED_STEM}{fold}.json"
    atomic_write(out_json, lambda p: Path(p).write_text(
        json.dumps(report, indent=2), encoding="utf-8"))

    t = report["totals"]
    print(f"\ntotals  GT={t['n_gt']}  M={t['M']} C={t['C']} T={t['T']} L={t['L']} D={t['D']}")
    for name in sorted(out_files):
        print(f"wrote {out_dir / name}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
