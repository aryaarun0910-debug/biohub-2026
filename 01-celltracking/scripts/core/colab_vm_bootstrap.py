"""VM-side bootstrap for CLI-driven Colab jobs (uploaded and executed by scripts/core/colab_cli.py).

Runs inside the session's system Python (3.13 on Colab) via `colab exec`. It:
  1. writes the Kaggle access token, materialises <input_root>/<slug> and <input_root>/<competition>
     with kagglehub (Colab's own /kaggle/input is read-only, FACT-0320),
  2. builds a Python 3.12 Jupyter kernel with uv (Kaggle kernels are 3.12 and the support pack ships
     cp312 wheels), launched with a clean environment (Colab exports PYTHONPATH=/env/python) and
     self-tested through papermill,
  3. launches papermill DETACHED on the job notebook, so the `colab exec` call returns and the laptop
     polls /content/runs/<job>/status.json.

Everything it does is idempotent per VM: inputs, the venv and the kernel are cached across jobs.
Configuration arrives as JSON at /content/jobs/<job>/job.json (written by the laptop) plus
/content/.kaggle_token (the KGAT_ access token, uploaded separately, never printed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONTENT = Path("/content")
KAGGLE_WORKING = Path("/kaggle/working")
POS_MARK = "[bootstrap]"


def log(msg: str) -> None:
    print(f"{POS_MARK} {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sh(cmd, check=True, env=None, cwd=None) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if check and cp.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} -> {cp.returncode}\n{cp.stdout[-800:]}\n{cp.stderr[-800:]}")
    return cp


def clean_env(extra=None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["PYTHONNOUSERSITE"] = "1"
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


# ------------------------------------------------------------------ kaggle auth + inputs
def setup_kaggle_auth() -> None:
    tok = (CONTENT / ".kaggle_token").read_text().strip()
    assert tok.startswith("KGAT_"), "token file does not hold a KGAT_ access token"
    os.environ["KAGGLE_API_TOKEN"] = tok
    kd = Path.home() / ".kaggle"
    kd.mkdir(exist_ok=True)
    (kd / "access_token").write_text(tok)
    os.chmod(kd / "access_token", 0o600)
    sh([sys.executable, "-m", "pip", "install", "-q", "kagglehub", "kaggle", "papermill"])
    chk = sh(["kaggle", "competitions", "files", "-c", "biohub-cell-tracking-during-development", "--csv", "--page-size", "1"], check=False)
    if chk.returncode != 0 or "401" in chk.stdout + chk.stderr:
        raise RuntimeError("KAGGLE AUTH FAILED: " + (chk.stdout + chk.stderr)[-400:])
    log("kaggle auth OK (access token)")


def layout_plan(job: dict) -> list[dict]:
    root = (job.get("input_root") or "/content/kaggle/input").rstrip("/")
    plan = []
    for ds in job.get("datasets", []) or []:
        owner, slug = ds.split("/", 1)
        plan.append({"kind": "dataset", "ref": ds, "path": f"{root}/{slug}", "alias": f"{root}/datasets/{owner}/{slug}"})
    comp = job.get("competition") or ""
    if comp:
        plan.append({"kind": "competition", "ref": comp, "path": f"{root}/{comp}", "alias": f"{root}/competitions/{comp}"})
    return plan


def ensure_inputs(job: dict) -> None:
    import kagglehub

    for item in layout_plan(job):
        target, alias = Path(item["path"]), Path(item["alias"])
        if target.exists():
            log(f"input present: {target}")
            continue
        t0 = time.time()
        if item["kind"] == "dataset":
            src = Path(kagglehub.dataset_download(item["ref"]))
        else:
            src = Path(kagglehub.competition_download(item["ref"]))
        for link in (target, alias):
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(src, target_is_directory=True)
        log(f"input ready: {target} <- {src} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ python 3.12 kernel
def ensure_kernel(version: str) -> str | None:
    if not version:
        return None
    tag = "biohub" + version.replace(".", "")
    venv = CONTENT / ("venv" + version.replace(".", ""))
    py = venv / "bin" / "python"
    marker = venv / ".ready"
    if marker.exists():
        log(f"kernel {tag} present")
        return tag
    t0 = time.time()
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh([sys.executable, "-m", "uv", "venv", str(venv), "--python", version, "--seed"])
    sh([str(py), "-m", "pip", "install", "-q", "ipykernel", "ipython", "numpy", "scipy", "pandas"])
    sh([str(py), "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cu126"])
    sh([str(py), "-m", "ipykernel", "install", "--user", "--name", tag, "--env", "PYTHONNOUSERSITE", "1", "--env", "PYTHONPATH", ""])
    ver = sh([str(py), "-c", "import sys,torch,ipykernel,zmq;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"],
             env=clean_env()).stdout.strip()
    log(f"kernel {tag} built in {time.time() - t0:.0f}s: {ver}")
    probe = CONTENT / "kernel_probe.ipynb"
    probe.write_text(json.dumps({"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                                            "source": ["import sys, torch; print('kernel ok', sys.version.split()[0], torch.cuda.is_available())"]}],
                                 "metadata": {"kernelspec": {"name": tag, "display_name": tag, "language": "python"}},
                                 "nbformat": 4, "nbformat_minor": 5}))
    cp = sh(["papermill", str(probe), str(CONTENT / "kernel_probe_out.ipynb"), "--kernel", tag, "--log-output"], check=False, env=clean_env())
    if cp.returncode != 0:
        raise RuntimeError("kernel self-test FAILED:\n" + (cp.stdout + cp.stderr)[-1500:])
    log("kernel self-test OK: " + " ".join(l for l in cp.stdout.splitlines() if "kernel ok" in l)[:100])
    marker.write_text(ver)
    return tag


# ------------------------------------------------------------------ launch
def launch(job: dict) -> dict:
    jdir = CONTENT / "jobs" / job["id"]
    run_dir = CONTENT / "runs" / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if KAGGLE_WORKING.exists():
        sh(["rm", "-rf", str(KAGGLE_WORKING)])
    KAGGLE_WORKING.mkdir(parents=True)
    kernel = ensure_kernel(job.get("python", "3.12"))
    env = clean_env(job.get("env") or {})
    cmd = ["papermill", str(jdir / job["notebook"]), str(KAGGLE_WORKING / "executed.ipynb"), "--log-output", "--cwd", str(KAGGLE_WORKING)]
    if kernel:
        cmd += ["--kernel", kernel]
    log_path = run_dir / "log.txt"
    wrapper = run_dir / "run.sh"
    wrapper.write_text("#!/bin/bash\n" + " ".join(f"'{c}'" for c in cmd) + f" > '{log_path}' 2>&1\necho $? > '{run_dir / 'exit_code'}'\n")
    os.chmod(wrapper, 0o755)
    proc = subprocess.Popen(["nohup", "bash", str(wrapper)], env=env, cwd=str(KAGGLE_WORKING),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    status = {"state": "running", "pid": proc.pid, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "kernel": kernel, "cmd": cmd}
    (run_dir / "status.json").write_text(json.dumps(status, indent=1))
    log(f"papermill launched detached, pid {proc.pid}, log {log_path}")
    return status


def main() -> None:
    job_id = os.environ.get("BIOHUB_JOB_ID") or sys.argv[1]
    job = json.loads((CONTENT / "jobs" / job_id / "job.json").read_text())
    t0 = time.time()
    setup_kaggle_auth()
    ensure_inputs(job)
    st = launch(job)
    log(f"bootstrap done in {time.time() - t0:.0f}s: {json.dumps(st)}")


if __name__ == "__main__":
    main()
