# --- D1 PARENT AGGREGATOR ------------------------------------------------------------
# Runs in the NOTEBOOK, after every predict subprocess has exited. Reads the immutable
# per-crop records under d1_audit/manifests/ and builds the single global manifest.
#
# WHY THIS EXISTS. v4 kept one shared manifest and rewrote it on every flush, so when the
# audit flushed more than once only the last crop survived and the others looked as though
# they had never run -- fold 1 wrote BOTH crops' artifacts but reported 1/2. A shared mutable
# manifest cannot be made safe by merging either, because concurrent writers still race. Each
# crop now owns a write-once terminal record and this aggregator is the only reader.
#
# It FAILS NONZERO rather than merely setting COMPLETE=false, so a broken audit cannot be
# mistaken for a finished one by anything downstream.
import json as _agg_json
from pathlib import Path as _AggPath

_AGG_DIR = _AggPath("/kaggle/working/d1_audit")
_AGG_MAN = _AGG_DIR / "manifests"
_agg_expected = sorted(_agg_json.loads(os.environ["BIOHUB_LOEO_STEMS"]))
_agg_fold = os.environ.get("BIOHUB_D1_FOLD")
_agg_ckpt = os.environ.get("BIOHUB_D1_CKPT_SHA")

_agg_problems: list[str] = []
_agg_crops: dict = {}

for _stem in _agg_expected:
    _c = _AGG_MAN / f"{_stem}.complete.json"
    _e = _AGG_MAN / f"{_stem}.error.json"
    _terminals = [p for p in (_c, _e) if p.exists()]
    if len(_terminals) == 0:
        _agg_problems.append(f"{_stem}: NO terminal record (never reached the flush)")
        _agg_crops[_stem] = {"dataset": _stem, "status": "not_reached"}
        continue
    if len(_terminals) > 1:
        _agg_problems.append(f"{_stem}: MULTIPLE terminal records {[p.name for p in _terminals]}")
    _rec = _agg_json.loads(_terminals[0].read_text())
    _agg_crops[_stem] = _rec
    if _rec.get("status") != "complete":
        _agg_problems.append(f"{_stem}: status={_rec.get('status')} exc={_rec.get('exception')}")
    if str(_rec.get("fold")) != str(_agg_fold):
        _agg_problems.append(f"{_stem}: fold {_rec.get('fold')} != expected {_agg_fold}")
    if _rec.get("checkpoint_sha256") != _agg_ckpt:
        _agg_problems.append(f"{_stem}: checkpoint mismatch")
    if not _rec.get("n_rows"):
        _agg_problems.append(f"{_stem}: zero rows exported")
    if _rec.get("feat_dim") != 32:
        _agg_problems.append(f"{_stem}: feat_dim {_rec.get('feat_dim')} != 32")
    if _rec.get("feat_rows") != _rec.get("n_rows"):
        _agg_problems.append(
            f"{_stem}: feat_rows {_rec.get('feat_rows')} != n_rows {_rec.get('n_rows')}")
    if not _rec.get("feat_finite"):
        _agg_problems.append(f"{_stem}: non-finite features")
    if _rec.get("gt_load_error"):
        _agg_problems.append(f"{_stem}: GT load error {_rec['gt_load_error']}")
    if not _rec.get("gt_rows"):
        _agg_problems.append(f"{_stem}: ZERO GT rows -- the audit has no subject")

# a stem set that differs from what was declared is a routing/config failure, not a warning
_agg_actual = sorted(p.name.split(".")[0] for p in _AGG_MAN.glob("*.start.json"))
if _agg_actual != _agg_expected:
    _agg_problems.append(f"stem set differs: started={_agg_actual} expected={_agg_expected}")

_agg_manifest = {
    "fold": _agg_fold, "checkpoint_sha256": _agg_ckpt,
    "expected_crops": len(_agg_expected), "expected_stems": _agg_expected,
    "started_stems": _agg_actual,
    "n_complete": sum(1 for v in _agg_crops.values() if v.get("status") == "complete"),
    "crops": _agg_crops, "problems": _agg_problems,
    "COMPLETE": not _agg_problems and len(_agg_crops) == len(_agg_expected),
}
_agg_tmp = _AGG_DIR / "d1_manifest.json.partial"
_agg_tmp.write_text(_agg_json.dumps(_agg_manifest, indent=2, default=str), encoding="utf-8")
_agg_tmp.replace(_AGG_DIR / "d1_manifest.json")

print("D1 AGGREGATOR:", _agg_manifest["n_complete"], "/", len(_agg_expected),
      "COMPLETE=", _agg_manifest["COMPLETE"], flush=True)
for _p in _agg_problems:
    print("  PROBLEM:", _p, flush=True)
if _agg_problems:
    raise RuntimeError(
        f"D1 aggregation FAILED with {len(_agg_problems)} problem(s); "
        "a partial audit must not be mistaken for a complete one"
    )
print("D1 AGGREGATOR: all crops complete and consistent", flush=True)
