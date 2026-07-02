"""Targeted downloader for the Biohub competition.

We never mirror the full 87 GB. Instead we pull exactly what a task needs:
- a GEFF label graph for an embryo (KB-scale) -> ``--kind geff``
- the image zarr metadata only (scale/shape) -> ``--kind zmeta``
- the full image zarr volume for an embryo (~GB) -> ``--kind zarr``

Files are resolved against the cached manifest (reports/inventory/file_manifest.txt)
and downloaded individually via the Kaggle API into ``data/``.

Usage:
    python scripts/fetch.py --embryo 44b6_0113de3b --kind geff
    python scripts/fetch.py --embryo 44b6_0113de3b --kind zmeta
    python scripts/fetch.py --list-embryos train
"""

import argparse
from pathlib import Path

from kaggle import KaggleApi

COMP = "biohub-cell-tracking-during-development"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "inventory" / "file_manifest.txt"
DATA = ROOT / "data"


def load_manifest() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def embryos(split: str) -> list[str]:
    import re

    ids = set()
    for n in load_manifest():
        m = re.match(rf"{split}/([^/]+?)\.(zarr|geff)", n)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def select(embryo: str, kind: str, split: str) -> list[str]:
    names = load_manifest()
    geff_pre = f"{split}/{embryo}.geff/"
    zarr_pre = f"{split}/{embryo}.zarr/"
    if kind == "geff":
        return [n for n in names if n.startswith(geff_pre)]
    if kind == "zarr":
        return [n for n in names if n.startswith(zarr_pre)]
    if kind == "zmeta":
        # only the small json metadata, no chunk data
        return [n for n in names if n.startswith(zarr_pre) and n.endswith(".json")]
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo")
    ap.add_argument("--kind", choices=["geff", "zarr", "zmeta"], default="geff")
    ap.add_argument("--split", default="train")
    ap.add_argument("--list-embryos", metavar="SPLIT")
    args = ap.parse_args()

    if args.list_embryos:
        ids = embryos(args.list_embryos)
        print(f"{len(ids)} {args.list_embryos} embryos:")
        for e in ids:
            print(" ", e)
        return

    api = KaggleApi()
    api.authenticate()
    files = select(args.embryo, args.kind, args.split)
    if not files:
        print(f"No files matched embryo={args.embryo} kind={args.kind} split={args.split}")
        return
    print(f"Downloading {len(files)} files for {args.embryo} ({args.kind})...")
    for i, name in enumerate(files, 1):
        dest = DATA / Path(name).parent
        dest.mkdir(parents=True, exist_ok=True)
        api.competition_download_file(COMP, name, path=str(dest), quiet=True)
        # Kaggle saves under dest/<basename>; flatten any nesting it created.
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")
    print(f"Done -> {DATA}")


if __name__ == "__main__":
    main()
