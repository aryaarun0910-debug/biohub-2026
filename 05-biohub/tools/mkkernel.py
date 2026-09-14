#!/usr/bin/env python3
"""Build a variant of the reproduced 0.947 kernel with env overrides that survive their guard.

Two things have to be patched together or the notebook raises before it starts: the os.environ
assignment AND the matching entry in _EXPECTED_NUMERIC, which exists specifically to catch
configuration drift. machine_shape must be NvidiaTeslaT4 -- a P100 (sm_60) cannot run the pinned
torch at all, and acceleratorType is accepted and silently ignored.

    python tools/mkkernel.py <slug> KEY=VALUE [KEY=VALUE ...]
"""
import json, re, sys
from pathlib import Path

slug, kv = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
nb = json.load(open("kernels/repro-947/repro-947.ipynb"))
anchor = 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"'
patched = 0
for c in nb["cells"]:
    s = c.get("source")
    if not s: continue
    t = "".join(s) if isinstance(s, list) else s
    before = t
    for k, v in kv:
        env = f"BIOHUB_{k}"
        if re.search(rf'os\.environ\["{env}"\]\s*=', t):
            t = re.sub(rf'os\.environ\["{env}"\]\s*=\s*"[^"]*"',
                       f'os.environ["{env}"] = "{v}"', t)
        elif anchor in t:
            t = t.replace(anchor, anchor + f'\nos.environ["{env}"] = "{v}"')
        # the drift guard must agree or it raises before anything runs
        t = re.sub(rf'"{env}":\s*[0-9.]+,', f'"{env}": {float(v)},', t)
    if t != before: c["source"] = t; patched += 1

d = Path("kernels") / slug; d.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(d / "v.ipynb", "w"))
json.dump({"id": f"aryaarun07/{slug}", "title": slug, "code_file": "v.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_internet": False,
           "machine_shape": "NvidiaTeslaT4", "kernel_sources": [],
           "dataset_sources": ["pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                               "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                               "pilkwang/biohub-tracking-support-pack-50ep-v1"],
           "competition_sources": ["biohub-cell-tracking-during-development"]},
          open(d / "kernel-metadata.json", "w"), indent=2)

src = "".join("".join(c.get("source", "")) if isinstance(c.get("source"), list)
              else (c.get("source") or "") for c in nb["cells"])
print(f"  {slug}: patched {patched} cell(s)")
for k, _ in kv:
    env = f"BIOHUB_{k}"
    for m in re.finditer(re.escape(env), src):
        seg = src[max(0, m.start()-38):m.start()+len(env)+22].replace("\n", " | ")
        if "environ[" in seg or '":' in seg: print(f"     {seg}")
