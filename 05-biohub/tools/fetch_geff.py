#!/usr/bin/env python3
"""Download ONLY the .geff ground-truth graphs (~2.4 MB total) from the competition.

The 98 GB is image volumes; the ground truth is a rounding error beside it. The division
re-gating work - the highest-value item in the sprint - is pure graph geometry and needs
none of the voxels.

Kaggle's per-file download flattens to the basename, so the zarr directory structure is
reconstructed here. Resumable: already-present files are skipped.
"""
import json, os, shutil, sys, time, tempfile

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
from kaggle.api.kaggle_api_extended import KaggleApi

COMP  = "biohub-cell-tracking-during-development"
PATHS = "/home/aryaarun/.claude/jobs/f43a4343/tmp/geff_paths.json"
DEST  = os.path.expanduser("~/Work/biohub/data/train_geff")
THROTTLE = float(os.environ.get("GEFF_THROTTLE", "0.35"))

def main():
    paths = json.load(open(PATHS))
    api = KaggleApi(); api.authenticate()
    os.makedirs(DEST, exist_ok=True)
    done = skipped = failed = 0
    t0 = time.time(); backoff = 2.0
    for i, rel in enumerate(paths, 1):
        out = os.path.join(DEST, rel.split("/", 1)[1])       # strip the leading "train/"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1; continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        while True:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    api.competition_download_file(COMP, rel, path=tmp, force=True, quiet=True)
                except Exception as e:
                    if "429" in str(e):
                        backoff = min(backoff * 2, 60)
                        print(f"  429 at {i}/{len(paths)} - sleeping {backoff:.0f}s", flush=True)
                        time.sleep(backoff); continue
                    print(f"  FAIL {rel}: {type(e).__name__}", flush=True)
                    failed += 1; break
                got = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp) for f in fs]
                if not got:
                    failed += 1; break
                shutil.move(got[0], out); done += 1
            break
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}  new={done} skip={skipped} fail={failed}  {time.time()-t0:.0f}s", flush=True)
        time.sleep(THROTTLE)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(DEST) for f in fs)
    ds = len([d for d in os.listdir(DEST) if d.endswith(".geff")])
    print(f"\nDONE new={done} skipped={skipped} failed={failed}")
    print(f"  {ds} .geff datasets, {size/1e6:.2f} MB at {DEST}")

if __name__ == "__main__":
    sys.exit(main())
