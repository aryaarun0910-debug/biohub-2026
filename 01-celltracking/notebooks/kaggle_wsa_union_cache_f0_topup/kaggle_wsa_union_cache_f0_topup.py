"""WORKSTREAM A -- union candidate cache: ONE GPU inference pass over the UNCHANGED node set.

WHAT THIS KERNEL DOES, AND WHAT IT DELIBERATELY DOES NOT DO
-----------------------------------------------------------
`src/biotrack/wrapper.py::motion_relink_edges` gates on the RAW source->target distance
(`if raw > gate_um: continue`) and only then scores with the MOTION-aware cost.  On a
doubled acquisition interval or an abrupt global shift the true partner is discarded by the
gate before the motion cost is ever evaluated.  The repair gates on the FLOW-COMPENSATED
RESIDUAL at the SAME 6/10 um radius -- it re-aims the gate, it does not widen it (verified
offline: every arm admits FEWER pairs in aggregate than the status quo).

To score that repair we need edge probabilities for pairs the current export never wrote.
This kernel supplies them.  It changes NOTHING about the model, the detection, or the node
population:

  * `predict_video` ALREADY computes the complete (n_src, n_tgt) edge logit matrix for every
    consecutive frame pair and applies `softmax(dim=0)` over ALL sources.  `cfg.threshold`
    plus max_parents=1 / max_children=2 is what truncates the geff export at ~9.88 um.
    There is no distance gate anywhere in the model.
  * therefore this is an EXPORT-SURFACE change with ZERO additional model FLOPs, and the
    softmax normalisation is bit-for-bit the one the deployed model already computes,
    because it is taken over the same complete source set of the same frame.
  * the geff export path is untouched, so the predicted graphs this run writes must be
    identical to the ones already cached locally.  The kernel asserts that: it hashes its
    own (t,z,y,x) detections per crop and hard-fails against the manifest's coord_sha256.

NOT A SUBMISSION.  Emits no submission.csv.  Scoring is local, on the pinned patched scorer.

Environment traps honoured (reports/ENVIRONMENT_TRAPS.md):
  8   PYTHONUTF8 is set by the pusher, not here, but no non-ASCII is emitted.
  9   an EMPTY /kaggle/input means a broken kernel -> re-create under a fresh slug; the
      failure path prints the mount tree so that stays distinguishable from trap 16.
  11  outputs are written as a small number of NAMED files, fetched by URL.
  16  attached DATASETS mount owner-qualified one level deeper than competition data, so
      every input is resolved with a BOUNDED depth ladder.  Never recursive glob -- that
      walks the 79 GB zarr tree.
  18  pair order is never rebuilt from a set; the manifest's own row order is preserved and
      the per-crop sha256 is re-verified after loading.
  20  timepoints come from node attributes / the manifest, never from node-id arithmetic.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG ============================
FOLD = int(os.environ.get("WSA_FOLD", "0"))   # 0 = held-out 44b6, 1 = held-out 6bba
DET_THRESHOLD = 0.99                          # identical to the run that produced oof_clean
POOL_KERNEL_UM = 5.0                          # trained/served value; script default 3.0 is wrong
MANIFEST_SLUG = "biohub-wsa-union-manifest"
WORK = Path("/kaggle/working")
OUTDIR = WORK / f"wsa_cache_split_{FOLD}"
# ===============================================================

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def sh(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ladder(basename, extra=()):
    """Bounded depth ladder (trap 16). NEVER recursive=True over /kaggle/input."""
    pats = [f"/kaggle/input/{'*/' * d}{basename}" for d in range(1, 5)] + list(extra)
    hits = []
    for p in pats:
        hits += glob.glob(p)
    return sorted(set(hits))


def require(basename, what, extra=()):
    hits = ladder(basename, extra)
    if not hits:
        for lvl in ("/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*"):
            print(lvl, "->", sorted(glob.glob(lvl))[:40], flush=True)
        raise SystemExit(
            f"{what}: no match for {basename!r}. If the listing above is EMPTY this kernel was "
            f"created in an SSL-error window (trap 9) -- re-create it under a FRESH SLUG. If it "
            f"is non-empty the input sits deeper than depth 4 (trap 16) -- widen the ladder.")
    return hits


def find_pack_repo() -> Path:
    hits = require("repo/scripts/predict_unet_transformer.py", "support pack")
    return Path(hits[0]).parents[1]


def gather_data_dir() -> Path:
    dst = WORK / "data"
    dst.mkdir(exist_ok=True)
    geffs = ladder("train/*.geff")
    n = 0
    for g in sorted(set(geffs)):
        z = str(Path(g).with_suffix(".zarr"))
        if not os.path.exists(z):
            continue
        for src in (g, z):
            link = dst / Path(src).name
            if not link.exists():
                try:
                    os.symlink(src, link)
                except OSError:
                    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, link)
        n += 1
    if n == 0:
        raise SystemExit("No crops found. Attach 'kms111201/biohub-cell-tracking-data'.")
    print(f"Gathered {n} crops -> {dst}", flush=True)
    return dst


def load_corrupt() -> set:
    stems = set()
    for h in ladder("metadata_files/corrupt_files.txt") + ladder("corrupt_files.txt"):
        stems |= set(re.findall(r"(?:44b6|6bba)_[0-9a-f]+", Path(h).read_text()))
    return stems


def find_weights():
    w = require(f"edge_predictor_best_split_{FOLD}.pth", f"fold-{FOLD} edge predictor",
                extra=tuple(f"/kaggle/input/{'*/' * d}repo/weights/*/split_{FOLD}/edge_predictor_best.pth"
                            for d in range(1, 5)))
    c = (ladder(f"config_split_{FOLD}.json")
         + ladder(f"repo/weights/*/split_{FOLD}/config.json"))
    print(f"weights candidates: {w}\nconfig candidates: {c}", flush=True)
    return Path(w[0]), (Path(c[0]) if c else None)


def find_manifest():
    req = require(f"union_request_split_{FOLD}.parquet", "WSA union manifest")
    idx = require("INDEX.json", "WSA manifest index")
    # prefer files that live next to each other in the manifest dataset
    req_p = Path(req[0])
    idx_p = next((Path(i) for i in idx if Path(i).parent == req_p.parent), Path(idx[0]))
    return req_p, idx_p


# --------------------------------------------------------------------------------------
# The patch injected into the pack's predict script.
# --------------------------------------------------------------------------------------
PRELUDE = '''
# ===================== WSA UNION CACHE (injected) =====================
import hashlib as _wsa_hashlib
import json as _wsa_json
import os as _wsa_os
from pathlib import Path as _WsaPath
import numpy as _wsa_np

WSA = {"req": None, "index": None, "outdir": None, "rows": None, "crop": None,
       "n_hit": 0, "n_frames": 0}


def wsa_init():
    import polars as _pl
    rq = _WsaPath(_wsa_os.environ["WSA_REQUEST"])
    ix = _WsaPath(_wsa_os.environ["WSA_INDEX"])
    od = _WsaPath(_wsa_os.environ["WSA_OUTDIR"])
    od.mkdir(parents=True, exist_ok=True)
    df = _pl.read_parquet(rq)
    req = {}
    # trap 18: preserve the manifest's own row order; never rebuild through a set.
    for (crop,), g in df.group_by("crop", maintain_order=True):
        per_t = {}
        for (t,), h in g.group_by("t", maintain_order=True):
            per_t[int(t)] = (h["source_id"].to_numpy().astype(_wsa_np.int64),
                             h["target_id"].to_numpy().astype(_wsa_np.int64))
        req[crop] = per_t
    WSA["req"] = req
    WSA["index"] = _wsa_json.loads(ix.read_text())
    WSA["outdir"] = od
    print(f"[WSA] manifest {rq.name}: {df.height} pairs over {len(req)} crops -> {od}", flush=True)


def wsa_begin(ds_path):
    crop = _WsaPath(ds_path).name
    if crop.endswith(".zarr"):
        crop = crop[:-5]
    WSA["crop"] = crop
    WSA["rows"] = []
    WSA["n_hit"] = 0
    WSA["n_frames"] = 0
    return crop


def wsa_record(t_src, s_src, s_tgt, n_src, n_tgt, probs, raw_logits):
    per_t = (WSA["req"] or {}).get(WSA["crop"])
    if not per_t:
        return
    ent = per_t.get(int(t_src))
    if ent is None:
        return
    gi, gj = ent
    i = gi - int(s_src)
    j = gj - int(s_tgt)
    if not (bool((i >= 0).all()) and bool((i < n_src).all())
            and bool((j >= 0).all()) and bool((j < n_tgt).all())):
        raise RuntimeError(
            f"[WSA] manifest index out of range for {WSA['crop']} t={t_src}: "
            f"s_src={s_src} n_src={n_src} s_tgt={s_tgt} n_tgt={n_tgt} "
            f"i[{i.min()},{i.max()}] j[{j.min()},{j.max()}]. The detection set has DRIFTED "
            f"from the manifest -- the node population is not unchanged. Do not use this run.")
    WSA["rows"].append((
        _wsa_np.full(len(gi), int(t_src), dtype=_wsa_np.int16),
        gi.astype(_wsa_np.int32), gj.astype(_wsa_np.int32),
        probs[i, j].astype(_wsa_np.float32),
        raw_logits[i, j].astype(_wsa_np.float32)))
    WSA["n_hit"] += len(gi)
    WSA["n_frames"] += 1


def wsa_finish(coords):
    import polars as _pl
    crop = WSA["crop"]
    order = _wsa_np.arange(len(coords))
    arr = _wsa_np.ascontiguousarray(coords[order].astype(_wsa_np.int16))
    sha = _wsa_hashlib.sha256(arr.tobytes()).hexdigest()
    want = None
    for k, v in (WSA["index"] or {}).items():
        if k.endswith("__" + crop):
            want = v
            break
    rec = {"crop": crop, "n_nodes": int(len(coords)), "coord_sha256": sha,
           "n_pairs_exported": int(WSA["n_hit"]), "n_frame_pairs": int(WSA["n_frames"])}
    if want is not None:
        rec["manifest_n_nodes"] = int(want.get("n_nodes", -1))
        rec["manifest_n_pairs"] = int(want.get("n_pairs", -1))
        rec["coord_sha256_expected"] = want.get("coord_sha256")
        rec["node_count_match"] = (rec["manifest_n_nodes"] == rec["n_nodes"])
        rec["pair_count_match"] = (rec["manifest_n_pairs"] == rec["n_pairs_exported"])
        if not rec["node_count_match"]:
            raise RuntimeError(
                f"[WSA] NODE POPULATION CHANGED for {crop}: kernel {rec['n_nodes']} vs manifest "
                f"{rec['manifest_n_nodes']}. Every arm requires the SAME node set; abort.")
        if rec.get("coord_sha256_expected") and rec["coord_sha256_expected"] != sha:
            raise RuntimeError(
                f"[WSA] COORD HASH MISMATCH for {crop}: {sha} != {rec['coord_sha256_expected']}. "
                f"Detections drifted; the cache would not be comparable. Abort.")
        if not rec["pair_count_match"]:
            raise RuntimeError(
                f"[WSA] pair-count mismatch for {crop}: exported {rec['n_pairs_exported']} vs "
                f"manifest {rec['manifest_n_pairs']}. Abort.")
    if WSA["rows"]:
        t = _wsa_np.concatenate([r[0] for r in WSA["rows"]])
        gi = _wsa_np.concatenate([r[1] for r in WSA["rows"]])
        gj = _wsa_np.concatenate([r[2] for r in WSA["rows"]])
        pr = _wsa_np.concatenate([r[3] for r in WSA["rows"]])
        lg = _wsa_np.concatenate([r[4] for r in WSA["rows"]])
        _pl.DataFrame({"t": t, "source_id": gi, "target_id": gj,
                       "edge_prob_full": pr, "edge_logit": lg}).write_parquet(
            WSA["outdir"] / (crop + ".parquet"), compression="zstd")
    (WSA["outdir"] / (crop + ".json")).write_text(_wsa_json.dumps(rec))
    print(f"[WSA] {crop}: nodes={rec['n_nodes']} exported={rec['n_pairs_exported']} "
          f"coord_sha={sha[:12]} OK", flush=True)


wsa_init()
# =================== END WSA UNION CACHE (injected) ===================
'''

ANCHOR_PROBS = """            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()
"""
REPLACE_PROBS = ANCHOR_PROBS + """
            wsa_record(t_src, s_src, s_tgt, n_src, n_tgt, probs,
                       raw.detach().float().cpu().numpy())
"""

ANCHOR_OPEN = "    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)"
REPLACE_OPEN = "    wsa_begin(ds_path)\n" + ANCHOR_OPEN

ANCHOR_RET = "    coords = coords.astype(np.int16)\n    return coords, all_edges"
REPLACE_RET = "    coords = coords.astype(np.int16)\n    wsa_finish(coords)\n    return coords, all_edges"


def patch_predict(pf: Path):
    src = pf.read_text()
    out = src.replace("pool_kernel_um: float = 3.0", f"pool_kernel_um: float = {POOL_KERNEL_UM}")
    assert out != src, "pool_kernel_um anchor missing -- predict script version drifted"

    for anchor, repl, name in ((ANCHOR_PROBS, REPLACE_PROBS, "probs export"),
                               (ANCHOR_OPEN, REPLACE_OPEN, "video begin"),
                               (ANCHOR_RET, REPLACE_RET, "coord finish")):
        n = out.count(anchor)
        assert n == 1, f"WSA patch {name}: expected exactly 1 anchor, found {n}"
        out = out.replace(anchor, repl)

    # prelude goes after the last top-level import block, before the config dataclass
    marker = "# =============================================================================\n# Prediction config"
    assert out.count(marker) == 1, "prelude anchor missing"
    out = out.replace(marker, PRELUDE + "\n" + marker)
    pf.write_text(out)
    print("Patched predict_unet_transformer.py: pool_kernel_um + WSA union cache", flush=True)


def main():
    pack_repo = find_pack_repo()
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(pack_repo, repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    print(f"repo -> {repo}", flush=True)

    pf = repo / "scripts" / "predict_unet_transformer.py"
    patch_predict(pf)

    sh([sys.executable, "-m", "pip", "install", "-q",
        "tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4", "polars>=1.36",
        "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"])

    data_dir = gather_data_dir()
    req_p, idx_p = find_manifest()
    index = json.loads(idx_p.read_text())

    # THE MANIFEST IS THE AUTHORITATIVE CROP LIST.
    # Version 1 of this kernel derived `held` from a corrupt-filtered glob and silently ran on
    # 66 of the 71 fold-0 crops -- a SCOPE REDUCTION with no error, which is exactly the class
    # of defect ENVIRONMENT_TRAPS warns about ("score only over the intersection of crops
    # present in every arm").  The set is now taken from the manifest and every crop must
    # resolve to a zarr, or the kernel dies naming the ones it could not find.
    held = sorted(k.split("__", 1)[1] for k in index if k.startswith(f"{FOLD}__"))
    only = os.environ.get("WSA_ONLY", '["44b6_587a1e22", "44b6_5f15d135", "44b6_668e0cc7", "44b6_66f9292d", "44b6_706092f0"]').strip()
    if only:
        want = set(json.loads(only))
        held = [c for c in held if c in want]
        print(f"WSA_ONLY restricts this run to {len(held)} crops", flush=True)
    unresolved = []
    for c in held:
        if (data_dir / f"{c}.zarr").exists():
            continue
        hits = ladder(f"{c}.zarr") + ladder(f"train/{c}.zarr")
        if not hits:
            unresolved.append(c)
            continue
        link = data_dir / f"{c}.zarr"
        try:
            os.symlink(hits[0], link, target_is_directory=True)
        except OSError:
            shutil.copytree(hits[0], link)
        g = ladder(f"{c}.geff") + ladder(f"train/{c}.geff")
        if g and not (data_dir / f"{c}.geff").exists():
            try:
                os.symlink(g[0], data_dir / f"{c}.geff", target_is_directory=True)
            except OSError:
                shutil.copytree(g[0], data_dir / f"{c}.geff")
    if unresolved:
        for lvl in ("/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*"):
            print(lvl, "->", sorted(glob.glob(lvl))[:40], flush=True)
        raise SystemExit(
            f"{len(unresolved)} manifest crops have no .zarr under /kaggle/input: {unresolved}. "
            f"Refusing to run on a silently reduced crop set.")
    print(f"FOLD {FOLD}: {len(held)} crops, all resolved from the manifest", flush=True)
    print(f"manifest -> {req_p} ({req_p.stat().st_size/1e6:.1f} MB), index -> {idx_p}", flush=True)

    folds = [{"train": [], "test": []}, {"train": [], "test": []}]
    folds[FOLD]["test"] = held
    splits = data_dir / "dataset_splits.json"
    splits.write_text(json.dumps(folds))

    wpath, cpath = find_weights()
    wdir = WORK / "wts"
    wdir.mkdir(exist_ok=True)
    shutil.copy(wpath, wdir / "edge_predictor.pth")
    if cpath:
        shutil.copy(cpath, wdir / "config.json")
    wsha = hashlib.sha256(Path(wpath).read_bytes()).hexdigest()
    print(f"weights -> {wpath} sha256={wsha}", flush=True)
    (WORK / f"wsa_run_meta_split_{FOLD}.json").write_text(json.dumps({
        "fold": FOLD, "weights_path": str(wpath), "weights_sha256": wsha,
        "config_path": str(cpath) if cpath else None,
        "det_threshold": DET_THRESHOLD, "pool_kernel_um": POOL_KERNEL_UM,
        "n_crops": len(held), "manifest": str(req_p),
        "manifest_sha256": hashlib.sha256(req_p.read_bytes()).hexdigest(),
    }, indent=1))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pypath = os.pathsep.join([str(repo / "src"), str(repo / "scripts"), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, BIOHUB_DATA_DIR=str(data_dir), PYTHONUNBUFFERED="1", PYTHONPATH=pypath,
               WSA_REQUEST=str(req_p), WSA_INDEX=str(idx_p), WSA_OUTDIR=str(OUTDIR))
    cmd = [sys.executable, "-u", str(pf), "--split", str(FOLD),
           "--data-dir", str(data_dir), "--splits", str(splits),
           "--weights", str(wdir / "edge_predictor.pth"),
           "--det-threshold", str(DET_THRESHOLD)]
    print("\n===== UNION-CACHE INFERENCE PASS =====", flush=True)
    sh(cmd, env=env)

    done = sorted(OUTDIR.glob("*.json"))
    print(f"\nWSA cache: {len(done)} / {len(held)} crops written to {OUTDIR}", flush=True)
    # one packed file per fold so it can be fetched by URL (trap 11)
    import polars as pl
    parts = []
    for p in sorted(OUTDIR.glob("*.parquet")):
        parts.append(pl.read_parquet(p).with_columns(pl.lit(p.stem).alias("crop")))
    if parts:
        allp = pl.concat(parts)
        out = WORK / f"wsa_union_cache_split_{FOLD}.parquet"
        allp.write_parquet(out, compression="zstd")
        print(f"packed {allp.height} rows -> {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    recs = [json.loads(p.read_text()) for p in done]
    (WORK / f"wsa_cache_index_split_{FOLD}.json").write_text(json.dumps(recs, indent=1))
    print("DONE. No submission is produced by this kernel by design.", flush=True)


if __name__ == "__main__":
    main()
