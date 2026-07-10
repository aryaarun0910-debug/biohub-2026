"""Build a lean Linux/Python 3.12 wheel pack for offline Trackastra inference.

Kaggle already ships torch/CUDA, numpy, scipy and the core scientific stack. The
learned-detector support pack supplies GEFF/tracksdata. Pulling Trackastra's full
dependency closure would add 3+ GiB and risk replacing the working CUDA runtime.
"""

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
    "edt",
    "lz4",
]


def main() -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--dest",
            str(WHEELS),
            *REQUIREMENTS,
        ],
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
