"""Build a complete Linux/Python 3.12 wheelhouse for offline Trackastra inference."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


WORK = Path("/kaggle/working")
WHEELS = WORK / "trackastra_wheelhouse"
REQUIREMENTS = [
    "trackastra==0.5.2",
    "tracksdata",
    "geff>=1.1.3.1.1",
    "zarr>=3.0.10,<4",
    "polars>=1.36",
]


def main() -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--dest", str(WHEELS), *REQUIREMENTS],
        check=True,
    )
    archive = Path(shutil.make_archive(str(WORK / "trackastra_wheelhouse"), "zip", WHEELS))
    files = []
    for path in sorted(WHEELS.iterdir()):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "python": sys.version,
        "requirements": REQUIREMENTS,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": files,
    }
    (WORK / "trackastra_wheelhouse_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(WHEELS)
    print(json.dumps({key: manifest[key] for key in ("archive", "archive_bytes", "archive_sha256")}, indent=2))


if __name__ == "__main__":
    main()
