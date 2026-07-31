"""Build P0-A's LOEO analogue: the same pipeline, run over one fold of the TRAIN crops.

THE GAP THIS CLOSES
-------------------
The division layer's ceiling is substrate-dependent -- reachable GT divisions per family:

    E0c        20/26   93/125   ->  +0.0641 / +0.0646
    clean903   20/26   96/125   ->  best measured substrate
    v122       15/26   68/125   ->  +0.0419 / +0.0461

P0-A (public clean 0.913) has never been measured, because it is a TEST-only notebook:
it discovers stems from the competition test directory and emits a single submission.
Measuring anything requires fold-specific graphs over the 199 labelled train crops.
This script produces exactly those, as Kaggle kernels, via scripts/kaggle_factory.py.

ARMS -- read before choosing one
--------------------------------
Fold 0 = held-out 44b6 (71 crops); fold 1 = held-out 6bba (128 crops).

  strict  Pack primary only. Secondary and DeepCenter OFF.
          The support pack ships ONLY weights/unet_transformer/split_0, trained on 6bba
          with 44b6 held out, so on FOLD 0 this arm is genuinely LOEO-clean and its
          numbers are directly comparable with the E0c / clean903 / v122 substrates.
  asis    P0-A verbatim (secondary + DeepCenter on). CONTAMINATED on the train set: the
          secondary is `unet_transformer_alltrain_seed314159_v1`, whose own
          training_config.json records "train_datasets": 199. Upper bracket only.
  hybrid  Secondary off, DeepCenter on. Isolates the DeepCenter gate's contribution.

FOLD 1 IS NOT DIRECTLY COMPARABLE without --weights-glob pointing at our own
edge_predictor_best_split_1.pth, and even then the two folds run different-vintage
models (400ep public vs our own), so cross-fold differences are confounded.
Recommended first measurement: fold 0, arm strict.

USAGE
-----
  # 1. generate the spec + notebook (local, free)
  .\.venv\Scripts\python.exe scripts\build_loeo_analogue.py --fold 0 --arm strict --limit 3
  .\.venv\Scripts\python.exe scripts\build_loeo_analogue.py --fold 0 --arm strict

  # 2. push / poll / fetch / score (Kaggle GPU; ~2.5 h for the full 71-crop fold 0)
  $env:PYTHONUTF8=1
  .\.venv\Scripts\python.exe scripts\kaggle_factory.py push   --spec scripts\kaggle_specs\loeo_f0_strict.json
  .\.venv\Scripts\python.exe scripts\kaggle_factory.py status --spec scripts\kaggle_specs\loeo_f0_strict.json
  .\.venv\Scripts\python.exe scripts\kaggle_factory.py fetch  --spec scripts\kaggle_specs\loeo_f0_strict.json ^
        --files loeo_split0_strict.csv.gz loeo_split0_strict.json
  .\.venv\Scripts\python.exe scripts\score_loeo_submission.py --csv <fetched.csv.gz> --gt-dir data\train
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "scripts" / "kaggle_specs"
BASE_NB = "notebooks/kaggle_p0a_clean913/biohub-p0a-clean913-repro.ipynb"
BASE_SHA = "35681256f355bc98552d9cbdc4d208a9670657f18b7e7c4ce5637e6619e5e570"

# kms111201/biohub-cell-tracking-data supplies train/*.zarr; the rest is P0-A's own set.
DATASETS = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
    "thtennant/taaf-kaggle-source-share-fork",
    "kms111201/biohub-cell-tracking-data",
]
OOF_WEIGHTS_DATASET = "aryaarun07/biohub-oof-weights"

ANCHOR_DEF = "def list_test_stems() -> list[str]:"
ANCHOR_CALL = "test_stems = list_test_stems()"
AUDIT_CELL_MARK = "_guard_submission = Path("
PATCH_CELL_MARK = "_ps = REPO_DIR"


def fold_stems(fold: int) -> list[str]:
    splits = json.loads((REPO / "data" / "dataset_splits.json").read_text())
    return sorted(splits[fold]["test"])


def build_spec(fold: int, arm: str, limit: int, weights_glob: str | None) -> dict:
    stems = fold_stems(fold)
    tag = f"f{fold}_{arm}" + (f"_smoke{limit}" if limit else "")
    slug = f"biohub-loeo-{tag.replace('_', '-')}"
    env_vars = {
        "BIOHUB_LOEO_FOLD": str(fold),
        "BIOHUB_LOEO_ARM": arm,
        "BIOHUB_LOEO_LIMIT": str(limit),
        "BIOHUB_LOEO_STEMS": json.dumps(stems),
    }
    datasets = list(DATASETS)
    if weights_glob:
        env_vars["BIOHUB_LOEO_WEIGHTS_GLOB"] = weights_glob
        env_vars["BIOHUB_LOEO_CONFIG_GLOB"] = weights_glob.replace(
            f"edge_predictor_best_split_{fold}.pth", f"config_split_{fold}.json"
        )
        datasets.append(OOF_WEIGHTS_DATASET)

    return {
        "name": f"loeo_{tag}",
        "slug": slug,
        # Kaggle slugifies the TITLE; keep it word-for-word equal to the slug so the two
        # cannot diverge (trap 12).
        "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.ipynb",
        "out_dir": f"notebooks/kaggle_loeo_{tag}",
        "base_notebook": BASE_NB,
        "base_sha256": BASE_SHA,
        "datasets": datasets,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "expects_submission": False,
        "purpose": (
            "LOEO analogue of P0-A: identical pipeline retargeted at fold "
            f"{fold} of the labelled train crops, to measure node recall and reachable "
            "GT divisions on P0-A's substrate."
        ),
        "edits": [
            {"kind": "env", "cell_match": "BIOHUB_PRESET", "vars": env_vars},
            {"kind": "insert_before", "anchor": ANCHOR_DEF,
             "cell_match": PATCH_CELL_MARK,
             "code_file": "scripts/kaggle_edits/loeo_retarget.py"},
            {"kind": "replace", "old": ANCHOR_CALL,
             "new": "test_stems = list(LOEO_STEMS)  # LOEO retarget", "expect": 1},
            {"kind": "replace_cell", "cell_match": AUDIT_CELL_MARK,
             "code_file": "scripts/kaggle_edits/loeo_export.py"},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, choices=[0, 1], required=True)
    ap.add_argument("--arm", choices=["strict", "asis", "hybrid"], default="strict")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N crops (Kaggle smoke test)")
    ap.add_argument("--weights-glob", default=None,
                    help="e.g. /kaggle/input/*/edge_predictor_best_split_1.pth "
                         "(REQUIRED for fold 1: the pack has no split_1)")
    ap.add_argument("--no-build", action="store_true", help="write the spec only")
    args = ap.parse_args()

    if args.fold == 1 and not args.weights_glob and args.arm != "asis":
        print("REFUSING: fold 1 held out 6bba, but the support pack ships only split_0, "
              "which was TRAINED on 6bba. Pass --weights-glob "
              "'/kaggle/input/*/edge_predictor_best_split_1.pth' (dataset "
              f"{OOF_WEIGHTS_DATASET}) or use --arm asis and label the result LEAKY.",
              file=sys.stderr)
        return 2

    spec = build_spec(args.fold, args.arm, args.limit, args.weights_glob)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = SPEC_DIR / f"{spec['name']}.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"spec -> {spec_path}")
    print(f"  fold {args.fold} | arm {args.arm} | crops "
          f"{args.limit or len(fold_stems(args.fold))} | slug {spec['slug']}")

    if args.no_build:
        return 0
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "kaggle_factory.py"),
         "build", "--spec", str(spec_path)],
        cwd=REPO,
    )
    if r.returncode:
        return r.returncode
    print("\nNext:")
    print(f"  $env:PYTHONUTF8=1")
    print(f"  .\\.venv\\Scripts\\python.exe scripts\\kaggle_factory.py verify --spec {spec_path.relative_to(REPO)}")
    print(f"  .\\.venv\\Scripts\\python.exe scripts\\kaggle_factory.py push   --spec {spec_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
