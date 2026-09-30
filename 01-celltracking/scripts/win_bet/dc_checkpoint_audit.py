r"""DeepCenter checkpoint audit (PKT-0017 substrate check) - three measurements, one script.

1. SPLIT: parse the pack's split_manifest.json by embryo prefix. Answers "is fold k in-sample
   for the centre prior?" (FACT-0310).
2. HISTORY: read the training history stored inside checkpoint_last.pt and print validation
   loss at selected epochs. Answers "did the deployed epoch-500 snapshot generalise?" (FACT-0311).
3. HEATMAP: run each checkpoint on named (crop, t) frames with the deployed preprocessing and
   report heatmap max / mean / fraction above the 0.10 gap-veto threshold, plus the heatmap value
   at the GT nodes of that frame (from an ea_atlas gtnodes parquet).

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\dc_checkpoint_audit.py ^
      --split-manifest <split_manifest.json> ^
      --checkpoints C:\temp\biohub_deepcenter_p10_audit\checkpoint_last.pt C:\temp\biohub_deepcenter_p10_audit\best.pt ^
      --frames 6bba_05b6850b:0 6bba_05b6850b:50 44b6_0113de3b:0 44b6_0113de3b:50 ^
      --gt-parquets C:\temp\edge_atlas_strict\gtnodes_s1.parquet C:\temp\edge_atlas_strict\gtnodes_s0.parquet ^
      --json-out C:\temp\dc_refine\checkpoint_audit.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_spec = importlib.util.spec_from_file_location("dc_subvoxel_refine", ROOT / "scripts" / "win_bet" / "dc_subvoxel_refine.py")
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

GAP_VETO_THRESHOLD = 0.10   # BIOHUB_DEEPCENTER_GAP_THRESHOLD default (deployed notebook cell 3:136)


def split_by_embryo(manifest: Path) -> dict:
    m = json.loads(manifest.read_text())
    out = {"seed": m.get("seed"), "val_fraction_declared": m.get("val_fraction")}
    for key in ("all", "train", "val"):
        counts: dict[str, int] = {}
        for crop in m.get(key, []):
            counts[crop[:4]] = counts.get(crop[:4], 0) + 1
        out[key] = counts
    return out


def history_table(checkpoint: Path, epochs=(1, 2, 3, 6, 11, 51, 101, 201, 500)) -> list[dict]:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hist = ck.get("history")
    if not hist:
        return []
    df = pd.DataFrame(hist)
    best = df.loc[df["val_loss"].idxmin()]
    rows = [{"epoch": int(r.epoch), "train_loss": float(r.train_loss), "val_loss": float(r.val_loss)}
            for r in df.itertuples() if int(r.epoch) in epochs]
    rows.append({"epoch": int(best.epoch), "train_loss": float(best.train_loss), "val_loss": float(best.val_loss),
                 "best_val": True})
    return rows


def heatmap_stats(bundle: dict, crop: str, t: int, gt: pd.DataFrame | None, data_dir: Path) -> dict:
    frame = dc.read_frame(data_dir / f"{crop}.zarr", t)
    hm = dc.heatmap_for_frame(bundle, frame)
    out = {"crop": crop, "t": t, "hm_max": float(hm.max()), "hm_mean": float(hm.mean()),
           "frac_ge_gap_threshold": float((hm >= GAP_VETO_THRESHOLD).mean())}
    if gt is not None:
        g = gt[(gt["dataset"] == crop) & (gt["t"] == t)]
        pool = int(getattr(bundle["cfg"], "pool_factor", 4))
        vals = [float(hm[min(hm.shape[0] - 1, int(round(z))), min(hm.shape[1] - 1, int(round(y / pool))),
                        min(hm.shape[2] - 1, int(round(x / pool)))])
                for z, y, x in g[["z", "y", "x"]].to_numpy()]
        out["n_gt"] = len(vals)
        out["hm_at_gt_median"] = float(np.median(vals)) if vals else None
        out["hm_at_gt_min"] = float(np.min(vals)) if vals else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--frames", nargs="*", default=[], help="crop:t pairs")
    ap.add_argument("--gt-parquets", nargs="*", default=[])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    result: dict = {}
    if args.split_manifest:
        result["split"] = split_by_embryo(Path(args.split_manifest))
        print("split by embryo:", json.dumps(result["split"]))
    gt = pd.concat([pd.read_parquet(p) for p in args.gt_parquets], ignore_index=True) if args.gt_parquets else None
    result["checkpoints"] = {}
    for ck in args.checkpoints:
        ckp = Path(ck)
        entry = {"path": str(ckp), "history": history_table(ckp), "frames": []}
        if entry["history"]:
            print(f"{ckp.name} history:", "  ".join(f"ep{h['epoch']} val {h['val_loss']:.4f}" for h in entry["history"]))
        if args.frames:
            bundle = dc.load_deepcenter(ckp, args.threads)
            entry["epoch"] = bundle["epoch"]
            for spec in args.frames:
                crop, t = spec.split(":")
                s = heatmap_stats(bundle, crop, int(t), gt, Path(args.data_dir))
                entry["frames"].append(s)
                print(f"  {ckp.name} ep{bundle['epoch']} {crop} t={t}: max {s['hm_max']:.3f} mean {s['hm_mean']:.4f} "
                      f"frac>=0.10 {s['frac_ge_gap_threshold']:.4f} | at GT (n={s.get('n_gt')}) median "
                      f"{s.get('hm_at_gt_median')}")
        result["checkpoints"][ckp.name] = entry
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
