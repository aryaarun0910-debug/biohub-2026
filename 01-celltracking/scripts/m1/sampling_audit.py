"""Zero-GPU sampling audit: does the epoch-1 sampler actually cover the training population?

The resume test proved repeatability, not coverage. The smoke reported
`unique_crops_sampled = 1`, which 40 draws over 59 crops cannot explain (uniform draws
would touch ~29). That figure came from a broken window->crop attribution with a "?"
fallback, not from the sampler. This audit rebuilds attribution from canonical
`VideoMeta.zarr_path` and enumerates the exact first 800 epoch-1 sampler positions.

Reports: unique windows/crops, steps-per-crop min/median/max, counts across the
preregistered density strata, and hard assertions that no validation crop, no 6bba path and
no unresolved crop ID can appear.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/src"))
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/scripts"))
sys.path.insert(0, str(ROOT / "scripts/m1"))
os.environ.setdefault("BIOHUB_DATA_DIR", str(ROOT / "data/train"))

import m1_driver as MD  # noqa: E402
from m1_config import M1_CONFIG  # noqa: E402

DET = ROOT / "artifacts/kaggle/coupled_cache/det"
STEPS = M1_CONFIG["max_iters_per_epoch"]
SEED = M1_CONFIG["seed"]
EPOCH = 1


def crop_strata(crops: list[str]) -> dict:
    """Reproduce the manifest's 3x2x2 deployment-observable grid."""
    stats = {}
    for c in crops:
        z = np.load(DET / f"{c}__tta-4view__det-0.969.npz")
        co = z["coords"]
        n = int(co.shape[0])
        tmax = int(co[:, 0].max()) + 1 if n else 1
        src, tgt = z["edge_src"], z["edge_tgt"]
        sz, sy, sx = 1.625, 0.40625, 0.40625
        if len(src):
            p = co[:, 1:].astype(float) * np.array([sz, sy, sx])
            motion = float(np.median(np.linalg.norm(p[tgt.astype(int)] - p[src.astype(int)], axis=1)))
        else:
            motion = 0.0
        stats[c] = {"density": n / tmax, "cand": len(src) / max(n, 1), "motion": motion}
    d = np.array([stats[c]["density"] for c in crops])
    ca = np.array([stats[c]["cand"] for c in crops])
    mo = np.array([stats[c]["motion"] for c in crops])
    de = np.quantile(d, [1 / 3, 2 / 3])
    cm, mm = np.median(ca), np.median(mo)
    for c in crops:
        s = stats[c]
        di = 0 if s["density"] <= de[0] else (1 if s["density"] <= de[1] else 2)
        stats[c]["stratum"] = (di, int(s["cand"] > cm), int(s["motion"] > mm))
    return stats


def main() -> None:
    import train_unet_transformer as T

    man = json.loads((ROOT / "scripts/m1/val_manifests.json").read_text())
    d1 = man["directions"]["1"]
    train_crops = sorted(d1["train_crops"])
    val_crops = set(d1["inner_val_crops"])

    # ---- index exactly as the kernel does, attributing by canonical zarr_path
    owner: list[str] = []
    per_crop_windows: dict[str, int] = {}
    for c in train_crops:
        vm, w = T.load_dataset_windows(ROOT / "data/train" / c, window_size=2,
                                       max_frames=None, downsample=(1, 4, 4))
        name = Path(str(vm.zarr_path)).name
        if name.endswith(".zarr"):
            name = name[:-5]
        if name != c:
            raise SystemExit(f"ABORT -- attribution mismatch: indexed {c} but metadata says {name}")
        owner += [name] * len(w)
        per_crop_windows[name] = len(w)
    n_windows = len(owner)
    unresolved = [o for o in owner if not o or o == "?"]
    print(f"indexed {len(train_crops)} crops -> {n_windows} windows "
          f"(unresolved ids: {len(unresolved)})")

    # ---- enumerate the exact epoch-1 sampler positions
    sampler = MD.ResumableSampler(n_windows, seed=SEED, epoch=EPOCH, start=0, length=STEPS)
    drawn = list(sampler)
    crops_drawn = [owner[i] for i in drawn]
    counts = Counter(crops_drawn)
    per = np.array([counts.get(c, 0) for c in train_crops])

    strata = crop_strata(train_crops)
    stratum_counts: Counter = Counter()
    for c, k in counts.items():
        stratum_counts[str(strata[c]["stratum"])] += k
    all_strata = {str(strata[c]["stratum"]) for c in train_crops}

    # expected exposure if draws were uniform over windows
    exp = {c: STEPS * per_crop_windows[c] / n_windows for c in train_crops}
    dev = {c: counts.get(c, 0) - exp[c] for c in train_crops}
    worst = sorted(dev.items(), key=lambda kv: -abs(kv[1]))[:5]

    print()
    print(f"=== EPOCH-{EPOCH} SAMPLER AUDIT (first {STEPS} positions, seed {SEED}) ===")
    print(f"  unique windows drawn : {len(set(drawn))} / {STEPS} draws")
    print(f"  unique crops touched : {len(counts)} / {len(train_crops)} training crops")
    print(f"  steps per crop       : min={per.min()} median={int(np.median(per))} max={per.max()}")
    print(f"  crops with zero draws: {int((per == 0).sum())}")
    print()
    print(f"  strata present in training pool : {len(all_strata)}")
    print(f"  strata covered by the 800 draws : {len(stratum_counts)}")
    for s in sorted(all_strata):
        print(f"    stratum {s}: {stratum_counts.get(s, 0)} draws")
    print()
    print("  largest |realised - expected| exposure deviations:")
    for c, dv in worst:
        print(f"    {c}: realised={counts.get(c, 0)} expected={exp[c]:.1f} dev={dv:+.1f}")

    leak_val = sorted(set(crops_drawn) & val_crops)
    leak_fam = sorted({c for c in crops_drawn if not c.startswith("44b6")})
    print()
    print(f"  validation crops drawn : {leak_val}  -> {'NONE (correct)' if not leak_val else 'LEAK'}")
    print(f"  non-44b6 crops drawn   : {leak_fam}  -> {'NONE (correct)' if not leak_fam else 'LEAK'}")
    print(f"  unresolved crop ids    : {len(unresolved)}")

    ok = (not leak_val and not leak_fam and not unresolved
          and len(counts) == len(train_crops) and len(stratum_counts) == len(all_strata))
    print()
    print(f"AUDIT: {'PASS' if ok else 'FAIL'}")

    out = ROOT / "reports/inventory/m1_sampling_audit.json"
    out.write_text(json.dumps({
        "seed": SEED, "epoch": EPOCH, "steps": STEPS,
        "n_train_crops": len(train_crops), "n_windows": n_windows,
        "unique_windows_drawn": len(set(drawn)), "unique_crops_touched": len(counts),
        "steps_per_crop": {"min": int(per.min()), "median": int(np.median(per)),
                           "max": int(per.max()), "zero_draw_crops": int((per == 0).sum())},
        "windows_per_crop": per_crop_windows,
        "draws_per_crop": dict(counts),
        "strata_in_pool": sorted(all_strata), "strata_draws": dict(stratum_counts),
        "expected_vs_realised_worst": [{"crop": c, "realised": counts.get(c, 0),
                                        "expected": round(exp[c], 1), "dev": round(dv, 1)}
                                       for c, dv in worst],
        "validation_crops_drawn": leak_val, "non_train_family_drawn": leak_fam,
        "unresolved_ids": len(unresolved), "verdict": "PASS" if ok else "FAIL",
    }, indent=2))
    print(f"wrote {out}")
    if not ok:
        raise SystemExit("SAMPLING AUDIT FAILED")


if __name__ == "__main__":
    main()
