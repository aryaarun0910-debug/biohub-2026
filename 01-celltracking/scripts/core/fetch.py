"""Targeted downloader for the Biohub competition.

CANONICAL BULK DOWNLOAD (all 199 train + test, ~87 GB): use the whole-competition endpoint,
which is one streamed archive and NOT subject to the per-file rate limit:
    kaggle competitions download -c biohub-cell-tracking-during-development -p data/_full_zip
    unzip -o -q data/_full_zip/*.zip -d data/

This script is for TARGETED per-embryo pulls (e.g. re-fetch one crop). It builds a COMPLETE
file manifest by full API pagination (the previously cached manifest was truncated to 129 of
199 embryos), caches it, then downloads individual files.

Usage:
    python scripts/core/fetch.py --embryo 44b6_0113de3b --kind geff
    python scripts/core/fetch.py --embryo 44b6_0113de3b --kind zmeta
    python scripts/core/fetch.py --refresh-manifest
    python scripts/core/fetch.py --list-embryos train
"""

import argparse
from pathlib import Path

from kaggle import KaggleApi

COMP = "biohub-cell-tracking-during-development"
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
MANIFEST = ROOT / "research" / "06-knowledge-system" / "inventory" / "file_manifest.txt"
DATA = ROOT / "data"


def refresh_manifest() -> list[str]:
    """Enumerate ALL competition files via complete pagination (follow token until exhausted)."""
    api = KaggleApi(); api.authenticate()
    names, token, pages = [], None, 0
    while True:
        res = api.competition_list_files(COMP, page_token=token, page_size=1000)
        for f in getattr(res, "files", res):
            names.append(str(f.ref if hasattr(f, "ref") else f.name))
        token = getattr(res, "nextPageToken", None) or getattr(res, "token", None)
        pages += 1
        if not token or pages > 500:   # generous cap; ~24k files / 1000 -> ~24 pages
            break
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(names) + "\n", encoding="utf-8")
    n_emb = len({n.split("/")[1].rsplit(".", 1)[0] for n in names if n.startswith("train/") and "." in n.split("/")[1]})
    print(f"manifest: {len(names)} files, {n_emb} train embryos -> {MANIFEST}")
    return names


def load_manifest() -> list[str]:
    if not MANIFEST.exists():
        return refresh_manifest()
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
    ap.add_argument("--refresh-manifest", action="store_true", help="rebuild the full file manifest")
    args = ap.parse_args()

    if args.refresh_manifest:
        refresh_manifest()
        return

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
