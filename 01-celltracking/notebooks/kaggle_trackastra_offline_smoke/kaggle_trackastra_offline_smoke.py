"""Internet-off import and checkpoint smoke test for the deployment stack."""

from __future__ import annotations

import glob
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


WORK = Path("/kaggle/working")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    archives = sorted(glob.glob("/kaggle/input/**/trackastra_wheelhouse.zip", recursive=True))
    expected_archive = "31317ff2c670837f096ff42773e5443d96549d48b506a3a4b2de1533b4d15911"
    if archives:
        if len(archives) != 1:
            raise FileNotFoundError(f"expected one lean wheel archive, found {archives}")
        archive = Path(archives[0])
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != expected_archive:
            raise RuntimeError(f"wheel archive hash mismatch: {digest}")
        lean = WORK / "lean_wheels"
        lean.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(lean)
    else:
        # Kaggle normally unpacks uploaded ZIP datasets at mount time.
        trackastra_wheels = sorted(glob.glob("/kaggle/input/**/trackastra-0.5.2-*.whl", recursive=True))
        if len(trackastra_wheels) != 1:
            raise FileNotFoundError(f"expected one unpacked Trackastra wheel, found {trackastra_wheels}")
        lean = Path(trackastra_wheels[0]).parent
        expected_wheels = {
            "trackastra-0.5.2-py3-none-any.whl": "373c8fa3840174050183a238b3977326662f5ef830dd340c95168bb4e468f89e",
            "edt-3.1.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "24de82036a0c0ace3572eb03852510e2b9993e7701402f885591562844067527",
            "lz4-4.4.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": "24092635f47538b392c4eaeff14c7270d2c8e806bf4be2a6446a378591c5e69e",
        }
        for name, expected in expected_wheels.items():
            path = lean / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"wheel hash mismatch for {name}: {actual}")
        digest = expected_archive

    # The learned-detector support pack carries the competition graph/Zarr
    # wheel stack. Resolve those packages offline first, then add Trackastra's
    # three lean wheels without replacing Kaggle's torch/CUDA runtime.
    support_wheels = sorted(Path("/kaggle/input").glob("**/*.whl"))
    wheel_dirs = sorted({str(path.parent) for path in support_wheels} | {str(lean)})
    find_links = [item for directory in wheel_dirs for item in ("--find-links", directory)]
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            *find_links,
            "tracksdata",
            "geff>=1.1.3.1.1",
            "zarr>=3.0.10,<4",
            "polars>=1.36",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(next(lean.glob("trackastra-0.5.2-*.whl"))),
            str(next(lean.glob("edt-*.whl"))),
            str(next(lean.glob("lz4-*.whl"))),
        ]
    )

    import geff
    import polars
    import torch
    import trackastra
    import tracksdata
    import zarr
    from trackastra.model import Trackastra

    model_files = sorted(glob.glob("/kaggle/input/**/ctc/ctc/model.pt", recursive=True))
    if len(model_files) != 1:
        raise FileNotFoundError(f"expected one checkpoint, found {model_files}")
    model_dir = Path(model_files[0]).parent
    Trackastra.from_folder(model_dir, device="cpu", batch_size=1)
    print(
        {
            "trackastra": trackastra.__version__,
            "torch": torch.__version__,
            "cuda_runtime_preserved": torch.version.cuda,
            "tracksdata": getattr(tracksdata, "__version__", "unknown"),
            "geff": getattr(geff, "__version__", "unknown"),
            "zarr": zarr.__version__,
            "polars": polars.__version__,
            "wheel_archive_sha256": digest,
            "checkpoint": str(model_dir),
        },
        flush=True,
    )
    print("OFFLINE TRACKASTRA SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
