r"""Batch runner for PKT-0046's reversed ceiling ladder. One worker, resumable, dynamically claimed.

Holds no science of its own: it calls ``revladder.run_crop`` per crop and writes one payload each,
so a killed run loses at most one crop and a rerun skips what is already complete.

DYNAMIC CLAIMING, not static sharding - the same choice ``finaledge_batch.py`` made and for the
same measured reason: crop cost varies (a crop's cost tracks its node count, which spans an order
of magnitude), so a static split leaves the run hostage to whichever shard drew the big crops.
With ``--claim`` every worker takes the next unclaimed crop, so the workers finish together. The
claim is an ``O_EXCL`` create, atomic on Windows.

STALE-CLAIM PROTECTION IS FOUR HOURS by packet order, so a thermally throttled large crop is never
duplicated by a sibling that assumed its owner had died.

A refusal is RECORDED, never skipped silently - a silently missing crop is invisible in the panel,
which is the ``all_passed`` trap ``FACT-0417`` records one build over.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from revladder import FOLDS, HEARTBEAT, run_crop  # noqa: E402

STALE_S_DEFAULT = 14400.0      # four hours, per PKT-0046 worker_policy


def crops_for(fold: int, gt_dir: Path) -> list[str]:
    cfg = FOLDS[fold]
    names = sorted(p.stem for p in gt_dir.glob(f"{cfg['prefix']}_*.geff"))
    if len(names) != cfg["n_crops"]:
        raise SystemExit(
            f"fold {fold}: found {len(names)} GT crops, expected {cfg['n_crops']} - refusing to "
            "run a fold that is not complete"
        )
    return names


def _claim(out_dir: Path, fold: int, crop: str, stale_s: float) -> bool:
    """Take this crop, or report that someone else has it. O_EXCL create, atomic on Windows."""
    claim = out_dir / f".claim_f{fold}_{crop}"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            return False
        if age < stale_s:
            return False
        try:
            claim.unlink()
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return False
    os.write(fd, f"{os.getpid()} {time.time()}".encode())
    os.close(fd)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slice-dir", type=Path, default=Path(r"C:\temp\revladder\slices"))
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--worker", type=int, default=0, help="label only; claiming does the routing")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--claim-stale-s", type=float, default=STALE_S_DEFAULT)
    args = ap.parse_args()

    names = crops_for(args.fold, args.gt_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = 0
    t_start = time.time()
    for i, crop in enumerate(names):
        out = args.out_dir / f"f{args.fold}_{crop}.json"
        if out.is_file():
            try:
                if json.loads(out.read_text(encoding="utf-8")).get("heartbeat") == HEARTBEAT:
                    skipped += 1
                    continue
            except Exception:
                pass
        if args.claim and not _claim(args.out_dir, args.fold, crop, args.claim_stale_s):
            skipped += 1
            continue
        t0 = time.time()
        try:
            run_crop(args.fold, crop, out, args.slice_dir, args.gt_dir)
            done += 1
        except Exception as exc:              # noqa: BLE001 - recorded, never swallowed
            failed += 1
            (args.out_dir / f"REFUSED_f{args.fold}_{crop}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            print(f"REFUSED fold={args.fold} crop={crop} {type(exc).__name__}: {exc}", flush=True)
        print(f"  [w{args.worker}] {i + 1}/{len(names)} {crop} {time.time() - t0:.0f}s "
              f"done={done} failed={failed} skipped={skipped}", flush=True)
    elapsed = time.time() - t_start
    rate = (done / elapsed * 3600.0) if elapsed > 0 and done else 0.0
    print(f"REVLADDER_BATCH_COMPLETE fold={args.fold} worker={args.worker} done={done} "
          f"failed={failed} skipped={skipped} elapsed_s={elapsed:.0f} crops_per_hour={rate:.1f}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
