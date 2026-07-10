"""Pack inherited OOF GEFF directory outputs into two transport archives."""

from __future__ import annotations

import glob
import shutil
from pathlib import Path


WORK = Path("/kaggle/working")


def main() -> None:
    found: dict[int, Path] = {}
    for fold in (0, 1):
        hits = sorted(glob.glob(f"/kaggle/input/**/pred_geffs_split_{fold}", recursive=True))
        if not hits:
            sample = sorted(glob.glob("/kaggle/input/**/*.geff", recursive=True))[:20]
            raise FileNotFoundError(f"fold {fold} prediction dir missing; sample={sample}")
        found[fold] = Path(hits[0])

    for fold, source in found.items():
        n = len(list(source.glob("*.geff")))
        if n == 0:
            raise RuntimeError(f"no GEFFs under {source}")
        base = WORK / f"pred_geffs_split_{fold}"
        archive = Path(shutil.make_archive(str(base), "zip", root_dir=source.parent, base_dir=source.name))
        print(f"fold={fold} geffs={n} source={source} archive={archive} bytes={archive.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
