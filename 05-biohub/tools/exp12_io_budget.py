#!/usr/bin/env python3
"""EXP-12 -- measure the REAL kernel bottleneck: zarr read + decompress.

EXP-11 showed the network is 13 minutes of T4 compute at base_c=64 against a 720-minute budget,
so compute does not decide the submission. I/O does: the hidden test set is ~199 datasets of
(100,64,256,256) uint16 -- 167GB uncompressed, ~82GB on disk. This times an honest streaming
read of what is already extracted and extrapolates to the full test set.
"""
import sys, time
from pathlib import Path
import numpy as np, zarr

SRC = Path("data/images/train")
N_TEST, T = 199, 100

files = sorted(SRC.glob("*.zarr"))
if not files:
    raise SystemExit("nothing extracted yet")
sample = files[:min(8, len(files))]
print(f"  timing {len(sample)} datasets of {len(files)} extracted\n")

tot_frames = tot_bytes = 0
t0 = time.time()
for p in sample:
    z = zarr.open(str(p), mode="r")
    arr = z["0"] if "0" in z else z
    for i in range(arr.shape[0]):
        v = np.asarray(arr[i])                 # read + decompress one frame
        tot_bytes += v.nbytes; tot_frames += 1
el = time.time() - t0

fps = tot_frames / el
mbs = tot_bytes / el / 1e6
print(f"  {tot_frames} frames, {tot_bytes/1e9:.2f} GB uncompressed in {el:.1f}s")
print(f"  -> {fps:.1f} frames/s, {mbs:.0f} MB/s decompressed (single process)")
full = N_TEST * T / fps
print(f"\n  FULL TEST SET ({N_TEST} datasets x {T} frames = {N_TEST*T:,} frames):")
print(f"    single process      {full/3600:>6.2f} h")
for w in (2, 4, 8):
    print(f"    {w} workers (ideal)   {full/w/3600:>6.2f} h")
NET = 0.22                                   # EXP-11, base_c=64, T4, whole test set
total = full/3600 + NET
print(f"\n  kernel budget 12.00 h; EXP-11 network cost at base_c=64 is {NET} h")
print(f"  I/O {full/3600:.2f} h + network {NET} h = {total:.2f} h  ->  {12/total:.0f}x HEADROOM")
print(f"  NOTHING here is the binding constraint. That headroom buys roughly {int(12/total)} full")
print(f"  passes over the test set: test-time augmentation, ensembling, or both.")
