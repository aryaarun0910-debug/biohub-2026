# =====================================================================================
# LOEO RETARGET BLOCK  (injected by scripts/build_loeo_analogue.py)
#
# P0-A is a TEST-only notebook: it discovers stems from the competition test dir and
# emits one submission. Nothing about its substrate -- node recall, and above all how
# many GT divisions are even REACHABLE in its graph -- has ever been measured, because
# measuring anything requires fold-specific graphs over the 199 labelled TRAIN crops.
#
# This block retargets the identical pipeline at one LOEO fold of the train set.
# It runs at module level inside the cell that patches predict_unet_transformer.py,
# i.e. AFTER every artifact/sha256 assertion in the base notebook has already passed and
# BEFORE `test_stems` is computed, so it can rebind the notebook globals the predict
# command and the wrapper both read (TEST_DIR, WEIGHTS_RELATIVE, the DeepCenter flags).
#
# LEAKAGE, stated up front because it decides which arm is meaningful:
#   * fold 0 = held-out 44b6 (71 crops); fold 1 = held-out 6bba (128 crops).
#   * the support pack ships ONLY weights/unet_transformer/split_0 -- trained on 6bba,
#     held out 44b6. It is therefore LOEO-CLEAN on fold 0 and LEAKY on fold 1.
#   * the secondary temporal model is `unet_transformer_alltrain_seed314159_v1` with
#     "train_datasets": 199 in its own training_config.json. It has seen every train
#     crop. With it enabled, NO measurement on the train set is clean.
#   * DeepCenter has no fold variant either.
# Hence arm `strict` (fold 0, pack primary only, secondary and DeepCenter OFF) is the
# only configuration that yields a number comparable with the E0c / clean903 / v122 OOF
# substrates. Arm `asis` reproduces P0-A verbatim and is an upper bracket only.
# =====================================================================================
import glob as _loeo_glob
import hashlib

LOEO_FOLD = int(os.environ["BIOHUB_LOEO_FOLD"])
LOEO_ARM = os.environ["BIOHUB_LOEO_ARM"].strip()
LOEO_LIMIT = int(os.environ.get("BIOHUB_LOEO_LIMIT", "0"))  # >0 => smoke on N crops
LOEO_STEMS_DECLARED = json.loads(os.environ["BIOHUB_LOEO_STEMS"])

if LOEO_ARM not in {"strict", "asis", "hybrid", "champion"}:
    raise ValueError(f"unknown LOEO arm {LOEO_ARM!r}")

# ---------------------------------------------------------------- 1. train crop mount
# ONLY the .zarr images are mounted. The GT .geff files are deliberately NOT linked:
# the kernel must be incapable of reading a label, exactly as it is on the real test
# set. Scoring happens locally against data/train with the pinned patched scorer.
_loeo_hits: list[str] = []
for _pat in (
    "/kaggle/input/*/train/*.zarr",
    "/kaggle/input/*/*/train/*.zarr",
    "/kaggle/input/*/*/*/train/*.zarr",
):
    _loeo_hits += _loeo_glob.glob(_pat)
_loeo_by_stem = {Path(p).name[:-5]: Path(p) for p in sorted(set(_loeo_hits))}
if not _loeo_by_stem:
    # Trap 9: an empty /kaggle/input is a broken kernel, not a broken notebook.
    # Print the actual mount tree first so the two failure modes are distinguishable.
    for _lvl in ("/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*"):
        print(_lvl, "->", sorted(_loeo_glob.glob(_lvl))[:40])
    raise RuntimeError(
        "No train .zarr found under /kaggle/input. If the listing above is EMPTY the "
        "kernel was created in an SSL-error window: re-create it under a FRESH SLUG. "
        "If it is non-empty, the train crops sit at an unexpected depth -- widen "
        "the glob patterns above."
    )

LOEO_STEMS = [s for s in LOEO_STEMS_DECLARED if s in _loeo_by_stem]
_loeo_missing = [s for s in LOEO_STEMS_DECLARED if s not in _loeo_by_stem]
if _loeo_missing:
    raise RuntimeError(f"fold {LOEO_FOLD}: {len(_loeo_missing)} declared stems absent: {_loeo_missing[:10]}")
if LOEO_LIMIT > 0:
    LOEO_STEMS = LOEO_STEMS[:LOEO_LIMIT]

LOEO_DATA_DIR = WORKING_DIR / f"loeo_data_split{LOEO_FOLD}"
if LOEO_DATA_DIR.exists():
    shutil.rmtree(LOEO_DATA_DIR)
LOEO_DATA_DIR.mkdir(parents=True)
for _stem in LOEO_STEMS:
    _src = _loeo_by_stem[_stem]
    _dst = LOEO_DATA_DIR / _src.name
    try:
        os.symlink(_src, _dst, target_is_directory=True)
    except OSError:
        shutil.copytree(_src, _dst)

# The wrapper's read_test_frame(), the base notebook's stem discovery and its final
# audit all read TEST_DIR. Rebinding it here retargets every one of them at once.
TEST_DIR = LOEO_DATA_DIR
print(f"LOEO fold {LOEO_FOLD} arm {LOEO_ARM}: {len(LOEO_STEMS)} crops -> {TEST_DIR}")

# ------------------------------------------------------------------- 2. fold weights
# fold 0 keeps the pack's split_0 (clean on 44b6); fold 1 needs our own split_1.
def _loeo_find(pattern: str) -> list[str]:
    """Resolve an input glob at whatever depth Kaggle mounted the dataset.

    2026-08-01: a fold-1 kernel died at t=628 s because the declared glob was
    `/kaggle/input/*/<file>` while datasets were mounted at
    `/kaggle/input/datasets/<owner>/<slug>/<file>` -- the same run resolved the support
    pack at that nested path, so /kaggle/input was NOT empty and this was not trap 9.
    Try the declared pattern first, then a bounded depth ladder on its basename (the
    same defence the .zarr mount above already uses). Bounded, never recursive: a
    `**` walk over /kaggle/input would descend the 79 GB competition zarr tree.
    """
    _hits = list(_loeo_glob.glob(pattern))
    _base = Path(pattern).name
    for _d in (1, 2, 3, 4):
        _hits += _loeo_glob.glob("/kaggle/input/" + "*/" * _d + _base)
    return sorted(set(_hits))


_loeo_weight_override = os.environ.get("BIOHUB_LOEO_WEIGHTS_GLOB", "").strip()
if _loeo_weight_override:
    _cand = _loeo_find(_loeo_weight_override)
    if len(_cand) != 1:
        for _lvl in ("/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*"):
            print(_lvl, "->", sorted(_loeo_glob.glob(_lvl))[:40])
        raise RuntimeError(f"weights glob {_loeo_weight_override!r} matched {_cand}")
    _cfg = _loeo_find(os.environ["BIOHUB_LOEO_CONFIG_GLOB"])
    if len(_cfg) != 1:
        raise RuntimeError(f"config glob matched {_cfg}")
    _wdir = WORKING_DIR / "loeo_weights"
    _wdir.mkdir(exist_ok=True)
    shutil.copy(_cand[0], _wdir / "edge_predictor.pth")
    shutil.copy(_cfg[0], _wdir / "config.json")
    WEIGHTS_RELATIVE = str(_wdir / "edge_predictor.pth")
    print(f"LOEO primary weights OVERRIDDEN -> {_cand[0]} "
          f"(sha256 {hashlib.sha256(Path(_cand[0]).read_bytes()).hexdigest()[:16]})")
else:
    print(f"LOEO primary weights: pack default {WEIGHTS_RELATIVE}")

# --------------------------------------------------- 3. arm-specific contamination cut
if LOEO_ARM in {"strict", "hybrid", "champion"}:
    os.environ["BIOHUB_SECONDARY_WEIGHTS"] = ""
    os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0"
    os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0"
    print("LOEO: secondary all-199-train model DISABLED (it has seen every train crop)")
if LOEO_ARM == "champion":
    # P24-lineage (0.928) control: keep the DeepCenter best.pt + safe-div veto bundle ON, driven by
    # the spec's cell-2 env (the proven P24 path). CAVEAT: DeepCenter has NO fold variant (FACT-0310,
    # trained on all 44b6 / validated on all 6bba), so on FOLD 0 (44b6 held out) it is IN-SAMPLE and
    # this control's ABSOLUTE score / offline->LB calibration is contaminated. Use `champion` for
    # PAIRED lever deltas on the 0.928 base, NOT for the calibration anchor. Fold 1 is the cleaner fold.
    print("LOEO champion: DeepCenter best.pt + safe-div veto KEPT ON (P24 bundle from spec env); "
          "fold-0 DeepCenter is in-sample (FACT-0310) - paired deltas only, not calibration.")
if LOEO_ARM == "strict":
    USE_DEEPCENTER_VETO = False
    REQUIRE_DEEPCENTER_VETO = False
    DEEPCENTER_GAP_VETO = False
    DEEPCENTER_SAFE_DIV_VETO = False
    os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
    os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"
    os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
    os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
    print("LOEO: DeepCenter add-only gate DISABLED (no fold variant exists)")

LOEO_MANIFEST = {
    "fold": LOEO_FOLD,
    "arm": LOEO_ARM,
    "n_crops": len(LOEO_STEMS),
    "crops": LOEO_STEMS,
    "weights": str(WEIGHTS_RELATIVE),
    "secondary_enabled": bool(os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()),
    "deepcenter_enabled": bool(USE_DEEPCENTER_VETO),
    "det_threshold": DET_THRESHOLD,
    "limit": LOEO_LIMIT,
    "experiment_tag": EXPERIMENT_TAG,
}
(WORKING_DIR / "loeo_manifest.json").write_text(json.dumps(LOEO_MANIFEST, indent=2))
print(json.dumps(LOEO_MANIFEST, indent=2)[:800])
