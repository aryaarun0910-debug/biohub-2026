#!/usr/bin/env python3
"""Run the 0.947 notebook's POST-PROCESSING locally on cached raw predictions. No GPU, minutes.

Iteration has been bottlenecked at ~3 hours per variant because every knob we tune sits AFTER the
U-Net, but the only way to run it was a full Kaggle kernel. run_stats.csv shows the split:

    44b6_0113de3b   raw_nodes 25,822  ->  nodes 25,637

The cached .geff files under predictions/ are the RAW linker output; everything between them and
submission.csv -- gap closing, motion relink, gap2 recovery, safe divisions, DeepCenter gating,
short-track filtering, linefit smoothing -- is CPU Python living in notebook cell 5. Cell 4 is the
only GPU stage, and it is skippable when the raw geffs already exist.

So: exec cells 0-3, supply the two names cell 4 would have defined (test_stems, predict_seconds),
point COMP_DIR and the DeepCenter checkpoint at local copies, then exec cell 5.

FIDELITY IS NOT ASSUMED. --check compares the result against work/repro_out/submission.csv, which
the real kernel produced. Anything that does not reproduce it byte-for-byte is not trusted --
reimplementing a pipeline until it agrees with your hypothesis is how EXP-38 went wrong.

    python tools/fastpp.py --check
    python tools/fastpp.py --set OUTPUT_MIN_TRACK_LEN=9 --out work/fastpp/ml9.csv
"""
import argparse, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "kernels/repro-947/repro-947.ipynb"
RUNDIR = ROOT / "work/fastpp2"  # default; --rundir overrides
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
ap.add_argument("--out", default=None)
ap.add_argument("--check", action="store_true")
ap.add_argument("--base", default=str(NB))
ap.add_argument("--rundir", default=None, help="separate dir so concurrent runs do not clobber")
ap.add_argument("--threads", type=int, default=0, help="bind torch threads; required for concurrency")
a = ap.parse_args()

if a.rundir:
    RUNDIR = ROOT / a.rundir

for kv in a.set:
    k, v = kv.split("=", 1)
    os.environ[f"BIOHUB_{k}"] = v
    print(f"  env BIOHUB_{k}={v}")

pw = ROOT / "data/pubweights"
os.environ.setdefault("BIOHUB_MODEL_ARTIFACTS", str(pw / "biohub-tracking-support-pack-50ep-v1"))
dc = pw / "biohub-deepcenter-unet3d-center-prior-v1"
_ckpt = dc / "weights/full_frame_center/best.pt"
if _ckpt.exists():
    os.environ.setdefault("BIOHUB_DEEPCENTER_CHECKPOINT", str(_ckpt))
_sec = pw / "biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json"
if _sec.exists():
    os.environ.setdefault("BIOHUB_SECONDARY_ARTIFACT_MANIFEST", str(_sec))
_man = dc / "ARTIFACT_MANIFEST.json"
if _man.exists():
    os.environ.setdefault("BIOHUB_DEEPCENTER_MANIFEST", str(_man))
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # force CPU; the veto net is small

# Torch ignores OMP_NUM_THREADS here: two concurrent screens each took 262% CPU and drove load
# to 17 on 8 cores, making both SLOWER than one alone. Bind threads in-process instead.
if a.threads:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(a.threads)
    try:
        import torch as _t
        _t.set_num_threads(a.threads); _t.set_num_interop_threads(1)
        print(f"  torch threads bound to {a.threads}")
    except Exception as _e:
        print(f"  could not bind torch threads: {_e}")

cells = [("".join(c.get("source", "")) if isinstance(c.get("source"), list)
          else (c.get("source") or "")) for c in json.load(open(a.base))["cells"]]

# Our local support pack is the 400ep snapshot, so scripts/evaluate.py fails the repo integrity
# checksum. src/ matches exactly and post-processing never calls evaluate.py, so the guard is
# neutralised HERE ONLY, for the local harness. Fidelity is then judged by --check comparing the
# produced submission.csv byte-for-byte against the one the real kernel wrote -- which is the
# only claim that matters.
# The notebook's own drift guard (_EXPECTED_NUMERIC) raises if an env var differs from the
# value it was authored with, so every --set key must be updated there too -- exactly what
# tools/mkkernel.py does when building a kernel variant.
for _k, _v in (kv.split("=", 1) for kv in a.set):
    _env = f"BIOHUB_{_k}"
    for _i in range(len(cells)):
        cells[_i] = re.sub(rf'"{_env}":\s*[0-9.]+,', f'"{_env}": {float(_v)},', cells[_i])

cells[3] = cells[3].replace(
    "if _support_actual_sha256 != _support_expected_sha256:",
    "if False:  # fastpp: local harness, see --check for the real fidelity test")
# The Kaggle dataset was re-versioned: our download of ...-50ep-v1 is actually the 400ep
# snapshot. Per-file checksums showed ONLY scripts/evaluate.py differs -- every
# src/biohub_tracking/*.py matched exactly -- so the library code is identical and these guards
# are precautionary rather than evidence of divergence.
cells[3] = re.sub(r"raise RuntimeError\(\s*f?\"Support repo manifest checksum mismatch[^)]*\)",
                  "pass", cells[3])
cells[3] = cells[3].replace(
    "if _support_actual_manifest_sha256 != _support_expected_manifest_sha256:",
    "if False:  # fastpp")

os.chdir(RUNDIR)
g: dict = {"__name__": "__main__", "display": lambda *x, **k: None}
_OVERRIDE = {k: os.environ[k] for k in
             ("BIOHUB_MODEL_ARTIFACTS", "BIOHUB_DEEPCENTER_CHECKPOINT",
              "BIOHUB_DEEPCENTER_MANIFEST", "BIOHUB_SECONDARY_ARTIFACT_MANIFEST")
             if k in os.environ}
_OVERRIDE.update({f"BIOHUB_{k.split('=',1)[0]}": k.split("=", 1)[1] for k in a.set})

for i in (0, 1, 2, 3):
    os.environ.update(_OVERRIDE)   # the notebook assigns os.environ itself; re-assert after each
    try:
        exec(compile(cells[i], f"<cell{i}>", "exec"), g)
    except SystemExit:
        pass
    except Exception as e:
        print(f"  cell {i} raised {type(e).__name__}: {e}")
        raise

# what cell 4 (the GPU stage) would have left behind
g["test_stems"] = STEMS
g["predict_seconds"] = 0.0
g["COMP_DIR"] = ROOT / "data/images"
g["TEST_DIR"] = ROOT / "data/images/test"
print(f"  COMP_DIR -> {g['COMP_DIR']}   predictions exist: "
      f"{(RUNDIR / 'tracking_repo/predictions').exists()}")

# Cell 3 re-materialises tracking_repo from the support pack, which ships its OWN
# repo/predictions with unrelated sample stems. Re-point it at the cached raw predictions AFTER
# that has run, or write_test_submission globs the wrong graphs.
_pred = Path(g["REPO_DIR"]) / "predictions"
_real = ROOT / "work/repro_out/tracking_repo/predictions"
if _pred.exists() and not _pred.is_symlink():
    _pred.rename(_pred.with_name("predictions_bundled_%d" % int(time.time())))
if not _pred.exists():
    _pred.symlink_to(_real)
print(f"  predictions -> {_pred.resolve()}")

os.environ.update(_OVERRIDE)
t0 = time.time()
exec(compile(cells[5], "<cell5>", "exec"), g)
print(f"  post-processing took {time.time()-t0:.0f}s")

sub = RUNDIR / "submission.csv"
if a.out:
    Path(a.out).write_bytes(sub.read_bytes()); print(f"  wrote {a.out}")
if a.check:
    import hashlib
    ref = ROOT / "work/repro_out/submission.csv"
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    same = h(sub) == h(ref)
    print(f"\n  local  {h(sub)}  {sub.stat().st_size:,} bytes")
    print(f"  kernel {h(ref)}  {ref.stat().st_size:,} bytes")
    print(f"  FIDELITY: {'EXACT MATCH -- harness is trustworthy' if same else 'DIFFERS -- do not trust'}")
