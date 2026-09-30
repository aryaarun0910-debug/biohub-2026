r"""Batch runner for PKT-0042's CPU gates. One worker, a shard of crops, resumable.

Holds no science of its own: it calls ``finaledge_gates.run_crop`` per crop and writes one
payload each, so a killed run loses at most one crop and a rerun skips what is already complete.
A crop whose control does not reproduce raises and is recorded as a refusal - never skipped
silently, because a silently missing crop is invisible in the panel (the ``all_passed`` trap
FACT-0417 records one build over).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.win_bet.finaledge_gates import FOLDS, HEARTBEAT, run_crop  # noqa: E402


def crops_for(fold: int) -> list[str]:
    frame = pl.read_parquet(FOLDS[fold]["preilp"], columns=["dataset"])
    names = sorted(frame["dataset"].unique().to_list())
    if len(names) != FOLDS[fold]["n_crops"]:
        raise SystemExit(
            f"fold {fold}: pre-ILP export holds {len(names)} crops, expected "
            f"{FOLDS[fold]['n_crops']} - refusing to run a fold that is not complete"
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
        # A worker that died mid-crop must not strand its crop forever - and a stranded crop is
        # invisible in the panel, which is the failure this whole runner exists to avoid.
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--bonuses", default="0,0.5,1,2,4,8")
    ap.add_argument("--gateb-ranks", default="")
    ap.add_argument("--gateb-bonuses", default="")
    ap.add_argument("--no-ledger", action="store_true")
    # Static sharding is badly unbalanced here because crop cost varies by an order of magnitude
    # (one fold-1 crop took 4,181s while another took 137s), so the run is hostage to its slowest
    # shard. A reverse worker walks the SAME list from the other end, skipping anything already
    # complete, which drains the tail without a scheduler. Payload writes are atomic, so if two
    # workers meet on one crop the cost is a duplicated crop and never a torn file.
    ap.add_argument("--reverse", action="store_true")
    # DYNAMIC CLAIMING beats static sharding here because crop cost varies by an order of
    # magnitude (137 s to 4,181 s on fold 1), so a static split leaves the whole run hostage to
    # whichever shard drew the big crops - measured: one shard sat at 4/22 while its siblings
    # were at 10/22. With --claim, every worker takes the next crop nobody has claimed, so the
    # workers finish together. The claim is an O_EXCL create, which is atomic on Windows, and a
    # stale claim from a killed worker is reclaimed after --claim-stale-s.
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--claim-stale-s", type=float, default=7200.0)
    args = ap.parse_args()

    names = crops_for(args.fold)
    if not args.claim:
        names = names[args.shard::args.shards]
    if args.reverse:
        names = list(reversed(names))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = 0
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
            run_crop(
                args.fold, crop, out, args.scores,
                tuple(float(v) for v in args.bonuses.split(",") if v != ""),
                tuple(int(v) for v in args.gateb_ranks.split(",") if v != ""),
                tuple(float(v) for v in args.gateb_bonuses.split(",") if v != ""),
                not args.no_ledger,
            )
            done += 1
        except Exception as exc:                       # noqa: BLE001 - recorded, never swallowed
            failed += 1
            (args.out_dir / f"REFUSED_f{args.fold}_{crop}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            print(f"REFUSED fold={args.fold} crop={crop} {type(exc).__name__}: {exc}", flush=True)
        print(f"  [shard {args.shard}/{args.shards}] {i + 1}/{len(names)} {crop} "
              f"{time.time() - t0:.0f}s done={done} failed={failed} skipped={skipped}", flush=True)
    print(f"FINALEDGE_BATCH_COMPLETE fold={args.fold} shard={args.shard} "
          f"done={done} failed={failed} skipped={skipped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
