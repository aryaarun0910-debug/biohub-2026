r"""Colab job factory - drive a Colab GPU from the laptop through a private GitHub relay.

    init             create/clone the private relay repo (C:/temp/colab_relay by default)
    session          record the HOST's budget + scope (rule 8) -> session.json ; --stop ends it
    queue --spec     build a kaggle spec with kaggle_factory and push it as a job
    smoke            queue the plumbing smoke job (nvidia-smi + tiny matmul, ~0.1 units)
    status           pull the relay; show session, queued jobs, running heartbeats, results
    fetch --job      pull the relay; copy runs/<id>/ (status, log tail, small outputs) locally
    worker-notebook  (re)generate notebooks/colab_worker/biohub-colab-worker.ipynb

Never launches anything. A job runs only when the host has (1) opened the worker notebook on a
Colab GPU and pressed Run all, and (2) recorded a budget with `session`. The worker enforces
both (`scripts/core/colab_relay.py::job_allowed`).

One-time Colab setup (host): Colab Secrets -> GH_TOKEN (repo-scoped PAT for the relay repo),
KAGGLE_USERNAME, KAGGLE_KEY (kagglehub reads these). Then File -> Open notebook -> GitHub ->
this repo -> notebooks/colab_worker/biohub-colab-worker.ipynb, pick the GPU, Run all.

Usage
-----
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py init --repo aryaarun0910-debug/biohub-colab-relay
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py session --units 30 --hours 12 --gpu L4 ^
      --scope smoke,notebook --host-approved "30 units on L4 for calibration stage 2, inference only"
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py smoke
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py queue --spec scripts\kaggle_specs\p20_relink_sweep_f0.json --max-hours 3
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py status
  .\.venv\Scripts\python.exe scripts\core\colab_factory.py fetch --job p20_relink_sweep_f0-20260826t2210
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))
import colab_relay as relay  # noqa: E402

DEFAULT_RELAY_DIR = Path(os.environ.get("BIOHUB_COLAB_RELAY_DIR", "C:/temp/colab_relay"))
WORKER_NB = REPO / "notebooks" / "colab_worker" / "biohub-colab-worker.ipynb"
RELAY_CONFIG = REPO / "scripts" / "core" / "colab_relay.json"     # {"repo": "owner/name"}


# --------------------------------------------------------------------------------------
# git plumbing (laptop side)
# --------------------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path | None = None, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env)
    if check and cp.returncode != 0:
        raise SystemExit(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stdout}\n{cp.stderr}")
    if not quiet and cp.stdout.strip():
        print(cp.stdout.strip())
    return cp


def relay_repo() -> str:
    cfg = relay.read_json(RELAY_CONFIG, {})
    if not cfg.get("repo"):
        raise SystemExit(f"no relay configured - run `init --repo owner/name` first ({RELAY_CONFIG})")
    return cfg["repo"]


def relay_sync(relay_dir: Path, message: str, push: bool = True) -> None:
    """Commit everything under the relay checkout, rebase on the remote, push (retry x3)."""
    _run(["git", "add", "-A"], relay_dir, quiet=True)
    cp = _run(["git", "commit", "-q", "-m", message], relay_dir, check=False, quiet=True)
    if not push:
        return
    for attempt in range(3):
        _run(["git", "pull", "--rebase", "-q"], relay_dir, check=False, quiet=True)
        cp = _run(["git", "push", "-q"], relay_dir, check=False, quiet=True)
        if cp.returncode == 0:
            return
    raise SystemExit(f"relay push failed after 3 attempts:\n{cp.stderr}")


def relay_pull(relay_dir: Path) -> None:
    _run(["git", "pull", "--rebase", "-q"], relay_dir, check=False, quiet=True)


# --------------------------------------------------------------------------------------
# init / session
# --------------------------------------------------------------------------------------
README = """# biohub-colab-relay

Control channel between the laptop (Claude Code) and a Colab worker notebook.
See `scripts/core/colab_relay.py` in the main repo for the contract. Nothing here is a result;
results are registered in the main repo's `research/00-system/registry/`.
"""


def cmd_init(args) -> int:
    relay_dir = Path(args.relay_dir)
    owner_repo = args.repo
    exists = _run(["gh", "repo", "view", owner_repo, "--json", "name"], check=False, quiet=True).returncode == 0
    if not exists:
        if args.no_create:
            raise SystemExit(f"{owner_repo} does not exist and --no-create was given")
        _run(["gh", "repo", "create", owner_repo, "--private", "--description",
              "Biohub Colab job relay (control channel only)"], quiet=True)
        print(f"created private repo {owner_repo}")
    if not (relay_dir / ".git").exists():
        relay_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["gh", "repo", "clone", owner_repo, str(relay_dir)], quiet=True)
        print(f"cloned to {relay_dir}")
    else:
        relay_pull(relay_dir)
    for d in ("jobs/pending", "jobs/running", "jobs/done", "jobs/failed", "jobs/refused", "runs", "lib"):
        (relay_dir / d).mkdir(parents=True, exist_ok=True)
        (relay_dir / d / ".keep").touch()
    (relay_dir / "README.md").write_text(README, encoding="utf-8")
    shutil.copy(REPO / "scripts" / "core" / "colab_relay.py", relay_dir / "lib" / "colab_relay.py")
    if not (relay_dir / "session.json").exists():
        relay.write_json(relay_dir / "session.json", dict(relay.SESSION_SCHEMA))
    relay.write_json(RELAY_CONFIG, {"repo": owner_repo, "relay_dir": str(relay_dir)})
    relay_sync(relay_dir, "init relay", push=not args.no_push)
    print(f"relay ready: {owner_repo} @ {relay_dir}; session INACTIVE until `session` records a host budget")
    return 0


def cmd_session(args) -> int:
    relay_dir = Path(args.relay_dir)
    relay_pull(relay_dir)
    if args.stop:
        s = relay.read_json(relay_dir / "session.json", dict(relay.SESSION_SCHEMA))
        s["stop"] = True
        s["active"] = False
        relay.write_json(relay_dir / "session.json", s)
        relay_sync(relay_dir, "session stop", push=not args.no_push)
        print("session stopped: the worker will finish its current job and exit")
        return 0
    if not args.host_approved:
        raise SystemExit("rule 8: an active session needs --host-approved \"<the host's words>\"")
    scope = [k.strip() for k in args.scope.split(",") if k.strip()]
    rates = {}
    for kv in (args.rate or []):
        k, v = kv.split("=")
        rates[k.strip().upper()] = float(v)
    s = relay.new_session(args.host_approved, args.units, scope, args.hours, gpu=args.gpu, rates=rates)
    errs = relay.validate_session(s)
    if errs:
        raise SystemExit("invalid session: " + "; ".join(errs))
    relay.write_json(relay_dir / "session.json", s)
    relay_sync(relay_dir, f"session: {args.units} units, {args.hours} h, {scope}", push=not args.no_push)
    print(json.dumps(s, indent=2))
    return 0


# --------------------------------------------------------------------------------------
# queue / smoke
# --------------------------------------------------------------------------------------
def _job_id(stem: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M")
    base = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem.lower())[:40].strip("-")
    return f"{base}-{ts}"


def _load_kaggle_factory():
    spec = importlib.util.spec_from_file_location("kaggle_factory", REPO / "scripts" / "core" / "kaggle_factory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_job(relay_dir: Path, job: dict, notebook: Path | None) -> Path:
    errs = relay.validate_job(job)
    if errs:
        raise SystemExit("invalid job: " + "; ".join(errs))
    jdir = relay_dir / "jobs" / "pending" / job["id"]
    if jdir.exists():
        raise SystemExit(f"job {job['id']} already queued")
    jdir.mkdir(parents=True)
    if notebook is not None:
        nb = json.loads(Path(notebook).read_text(encoding="utf-8"))
        root = job.get("input_root") or ""
        job["input_rewrites"] = relay.rewrite_input_root(nb, root) if root and root != relay.KAGGLE_INPUT else 0
        (jdir / job["notebook"]).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        job["notebook_sha256"] = hashlib.sha256((jdir / job["notebook"]).read_bytes()).hexdigest()
    relay.write_json(jdir / "job.json", job)
    return jdir


def cmd_queue(args) -> int:
    relay_dir = Path(args.relay_dir)
    kf = _load_kaggle_factory()
    spec_path = Path(args.spec)
    spec = kf.load_spec(spec_path)
    if not args.no_build:
        kf.cmd_build(spec)
    built = kf.built_nb(spec)
    if not built.exists():
        raise SystemExit(f"built notebook missing: {built}")
    job = relay.new_job(
        args.job_id or _job_id(spec["name"]), "notebook",
        spec=str(spec_path.relative_to(REPO)) if spec_path.is_absolute() else str(spec_path),
        priority=args.priority, datasets=list(spec.get("datasets", [])),
        competition=(spec.get("competition_sources") or [relay.COMPETITION])[0] if not args.no_competition else "",
        max_hours=float(args.max_hours), publish_kaggle=bool(args.publish_kaggle),
        kaggle_dataset_slug=args.kaggle_dataset_slug or "", note=args.note or spec.get("purpose", ""),
        env=dict(kv.split("=", 1) for kv in (args.env or [])), input_root=args.input_root,
    )
    relay_pull(relay_dir)
    jdir = write_job(relay_dir, job, built)
    relay_sync(relay_dir, f"queue {job['id']}", push=not args.no_push)
    print(f"queued {job['id']} -> {jdir} (datasets {len(job['datasets'])}, max {job['max_hours']} h, "
          f"input_root {job['input_root']}, {job['input_rewrites']} literals rewritten)")
    print("NOTE: it runs only if the worker is up AND session.json admits it (rule 8).")
    return 0


def cmd_smoke(args) -> int:
    relay_dir = Path(args.relay_dir)
    job = relay.new_job(args.job_id or _job_id("smoke"), "smoke", priority=0, max_hours=0.1,
                        note="plumbing smoke: nvidia-smi, torch matmul, metrics.json round-trip")
    relay_pull(relay_dir)
    jdir = write_job(relay_dir, job, None)
    relay_sync(relay_dir, f"queue {job['id']}", push=not args.no_push)
    print(f"queued smoke {job['id']} -> {jdir}")
    return 0


# --------------------------------------------------------------------------------------
# status / fetch
# --------------------------------------------------------------------------------------
def cmd_status(args) -> int:
    relay_dir = Path(args.relay_dir)
    relay_pull(relay_dir)
    s = relay.read_json(relay_dir / "session.json", {})
    st = relay.read_json(relay_dir / "session_state.json", {})
    print(f"session: active={s.get('active')} budget={s.get('budget_units')} scope={s.get('scope_kinds')} "
          f"expires={s.get('expires_at')} stop={s.get('stop')}")
    print(f"worker : gpu={st.get('gpu')} spent~{st.get('spent_units', 0):.2f} units last_seen={st.get('last_seen')} "
          f"state={st.get('state')}")
    for state in ("pending", "running", "done", "failed", "refused"):
        ids = sorted(p.name for p in (relay_dir / "jobs" / state).glob("*") if p.is_dir())
        if ids:
            print(f"{state:8s}: {', '.join(ids)}")
    for run in sorted((relay_dir / "runs").glob("*")):
        if not run.is_dir():
            continue
        status = relay.read_json(run / "status.json", {})
        hb = relay.read_json(run / "heartbeat.json", {})
        line = f"  {run.name}: {status.get('state')} "
        if status.get("reason"):
            line += f"({status['reason']}) "
        if status.get("elapsed_s") is not None:
            line += f"{status['elapsed_s'] / 60:.0f} min, ~{status.get('units', 0):.2f} units "
        if hb.get("at"):
            line += f"heartbeat {hb['at']} {hb.get('note', '')}"
        print(line)
    return 0


def cmd_fetch(args) -> int:
    relay_dir = Path(args.relay_dir)
    relay_pull(relay_dir)
    run = relay_dir / "runs" / args.job
    if not run.exists():
        raise SystemExit(f"no run for {args.job}")
    dest = Path(args.dest) if args.dest else Path("C:/temp/colab_runs") / args.job
    dest.mkdir(parents=True, exist_ok=True)
    for p in run.rglob("*"):
        if p.is_file():
            target = dest / p.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, target)
    status = relay.read_json(run / "status.json", {})
    print(json.dumps(status, indent=2))
    print(f"copied to {dest}")
    if status.get("kaggle_dataset"):
        print(f"large artefacts: python -m kaggle datasets download -d {status['kaggle_dataset']} -p <dest> --unzip")
    if status.get("drive_dir"):
        print(f"Drive copy: {status['drive_dir']}")
    return 0


# --------------------------------------------------------------------------------------
# worker notebook generator
# --------------------------------------------------------------------------------------
WORKER_CELLS = [
    ("markdown", """# Biohub Colab worker

Runs jobs queued by the laptop through the private relay repo `{RELAY_REPO}`.

**Before Run all (once per account):** Colab Secrets → `GH_TOKEN` (PAT with *contents: read/write* on the
relay repo), `KAGGLE_USERNAME`, `KAGGLE_KEY`. Pick the GPU the session budget names (Runtime → Change runtime type).

**Rule 8 is enforced here:** a job runs only if `session.json` carries a host-recorded budget whose scope admits it
and whose remaining units cover the job's worst case. Everything else is refused and written back as such.
The loop exits after `IDLE_EXIT_MIN` minutes with nothing to do, so an idle runtime does not burn units.
"""),
    ("code", """RELAY_REPO = "{RELAY_REPO}"
RELAY_DIR = "/content/relay"
KAGGLE_ROOT = "/kaggle"
DRIVE_ROOT = "/content/drive/MyDrive/biohub"
POLL_S = 60
IDLE_EXIT_MIN = 30
USE_DRIVE = True
"""),
    ("code", """import os, sys, json, time, shutil, subprocess, threading, glob, traceback
from pathlib import Path

from google.colab import userdata
GH_TOKEN = userdata.get("GH_TOKEN").strip()
assert GH_TOKEN.startswith(("github_pat_", "ghp_")), "GH_TOKEN does not look like a GitHub token"
Path.home().joinpath(".kaggle").mkdir(exist_ok=True)

def _secret(name):
    try:
        v = userdata.get(name)
        return v.strip() if v else ""
    except Exception:
        return ""

# Kaggle's current API authenticates with a KGAT_ ACCESS TOKEN (Colab secret KAGGLE_API_TOKEN, the
# contents of ~/.kaggle/access_token on the laptop). Username+key is legacy and is REJECTED by the
# api.kaggle.com endpoints kagglehub and the kaggle CLI now use (measured 2026-08-26: 401 with a
# verified-correct key; 200 with the bearer token). kagglehub and kagglesdk both read
# KAGGLE_API_TOKEN before anything else.
_api_token = _secret("KAGGLE_API_TOKEN")
_kuser, _kkey = _secret("KAGGLE_USERNAME"), _secret("KAGGLE_KEY")
if _api_token:
    assert _api_token.startswith("KGAT_"), "KAGGLE_API_TOKEN should start with KGAT_"
    os.environ["KAGGLE_API_TOKEN"] = _api_token
    Path.home().joinpath(".kaggle", "access_token").write_text(_api_token)
    os.chmod(Path.home() / ".kaggle" / "access_token", 0o600)
    print("Kaggle auth: access token (KGAT_) from secret KAGGLE_API_TOKEN")
if _kuser and _kkey:
    assert len(_kkey) == 32, f"KAGGLE_KEY should be 32 chars, got {len(_kkey)}"
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = _kuser, _kkey
    Path.home().joinpath(".kaggle", "kaggle.json").write_text(json.dumps({"username": _kuser, "key": _kkey}))
    os.chmod(Path.home() / ".kaggle" / "kaggle.json", 0o600)
    print("Kaggle auth: legacy username/key also present (fallback only)")
if not _api_token and not (_kuser and _kkey):
    raise SystemExit("No Kaggle credentials: add Colab secret KAGGLE_API_TOKEN (contents of ~/.kaggle/access_token)")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill", "kagglehub", "kaggle"], check=True)

# FAIL FAST on Kaggle credentials: a bad secret must stop the worker here, not 30 minutes into a
# download. Public datasets download anonymously, so only a COMPETITION call proves the account.
_comp = "biohub-cell-tracking-during-development"
_chk = subprocess.run(["kaggle", "competitions", "files", "-c", _comp, "--csv", "--page-size", "1"],
                      capture_output=True, text=True)
if _chk.returncode != 0 or "401" in _chk.stdout + _chk.stderr or "403" in _chk.stdout + _chk.stderr:
    raise SystemExit("KAGGLE AUTH FAILED for competition data. Add the Colab secret KAGGLE_API_TOKEN = the KGAT_ "
                     "token from ~/.kaggle/access_token on the laptop (username/key is rejected by the current API); "
                     "the account must have accepted the competition rules.\\n" + (_chk.stdout + _chk.stderr)[-600:])
print("Kaggle auth OK", f"(user {os.environ['KAGGLE_USERNAME']})" if os.environ.get("KAGGLE_USERNAME") else "(access token)", "| competition listing:",
      (_chk.stdout.strip().splitlines() or ["?"])[-1][:80])
try:
    import kagglehub
    print("kagglehub", kagglehub.__version__, "whoami:", getattr(kagglehub, "whoami", lambda: "n/a")())
except Exception as _e:
    print("kagglehub whoami unavailable:", _e)

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
        print("Drive mounted at", DRIVE_ROOT)
    except Exception as e:
        print("Drive mount failed - artefacts will only go to the relay / Kaggle:", e)
        USE_DRIVE = False
"""),
    ("code", """def sh(cmd, cwd=None, check=True, env=None):
    cp = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if check and cp.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {cp.returncode}\\n{cp.stdout}\\n{cp.stderr}")
    return cp

if not Path(RELAY_DIR, ".git").exists():
    sh(["git", "clone", "-q", f"https://x-access-token:{GH_TOKEN}@github.com/{RELAY_REPO}.git", RELAY_DIR])
sh(["git", "config", "user.email", "colab-worker@relay"], cwd=RELAY_DIR)
sh(["git", "config", "user.name", "colab-worker"], cwd=RELAY_DIR)
sys.path.insert(0, f"{RELAY_DIR}/lib")
import colab_relay as relay

def relay_pull():
    sh(["git", "pull", "--rebase", "-q"], cwd=RELAY_DIR, check=False)

def relay_push(message):
    sh(["git", "add", "-A"], cwd=RELAY_DIR)
    sh(["git", "commit", "-q", "-m", message], cwd=RELAY_DIR, check=False)
    for _ in range(4):
        sh(["git", "pull", "--rebase", "-q"], cwd=RELAY_DIR, check=False)
        if sh(["git", "push", "-q"], cwd=RELAY_DIR, check=False).returncode == 0:
            return True
        time.sleep(5)
    print("WARNING: relay push failed 4x")
    return False

gpu_name = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False).stdout.strip() or "unknown"
print("GPU:", gpu_name, "| family:", relay.gpu_family(gpu_name))
relay_pull()
session = relay.read_json(Path(RELAY_DIR, "session.json"), {})
print("session:", json.dumps(session, indent=1))
state = relay.read_json(Path(RELAY_DIR, "session_state.json"), {}) or {}
if state.get("session_approved_at") != session.get("approved_at"):
    state = {"session_approved_at": session.get("approved_at"), "spent_units": 0.0}
state.update({"gpu": gpu_name, "rate_units_per_h": relay.rate_for(gpu_name, session.get("rates")),
              "started": relay.utcnow(), "last_seen": relay.utcnow(), "state": "idle"})
relay.write_json(Path(RELAY_DIR, "session_state.json"), state)
relay_push("worker up")
"""),
    ("code", """def ensure_inputs(job):
    \"\"\"Materialise <input_root>/<slug> and <input_root>/<competition> from kagglehub downloads.
    Colab's own /kaggle/input is a READ-ONLY mount, so jobs are rewritten to /content/kaggle/input.\"\"\"
    import kagglehub
    plan = relay.kaggle_layout_plan(job)
    for item in plan:
        target = Path(item["path"]); alias = Path(item["alias"])
        if target.exists():
            print("  input present:", target); continue
        t0 = time.time()
        if item["kind"] == "dataset":
            src = Path(kagglehub.dataset_download(item["ref"]))
        else:
            try:
                src = Path(kagglehub.competition_download(item["ref"]))
            except Exception as e:
                print(f"  kagglehub competition_download failed ({type(e).__name__}); falling back to the Kaggle CLI")
                dl = Path("/content/dl"); dl.mkdir(parents=True, exist_ok=True)
                src = Path("/content/comp") / item["ref"]
                if not src.exists():
                    sh(["kaggle", "competitions", "download", "-c", item["ref"], "-p", str(dl)])
                    zips = sorted(dl.glob("*.zip"))
                    if not zips:
                        raise RuntimeError("Kaggle CLI download produced no zip")
                    src.mkdir(parents=True, exist_ok=True)
                    sh(["unzip", "-q", "-o", str(zips[-1]), "-d", str(src)])
                    zips[-1].unlink()
        for link in (target, alias):
            try:
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(src, target_is_directory=True)
            except OSError as e:
                raise RuntimeError(f"cannot link {link} -> {src}: {e}. Is input_root writable? "
                                   f"(Colab's /kaggle/input is read-only; queue with --input-root /content/kaggle/input)")
        print(f"  input ready: {target} <- {src} ({time.time() - t0:.0f}s)")
    if plan:
        print("  layout:", ", ".join(i["path"] for i in plan))

def heartbeat_loop(run_dir, stop_evt, note_fn):
    while not stop_evt.wait(relay.HEARTBEAT_S):
        try:
            relay.write_json(run_dir / "heartbeat.json", {"at": relay.utcnow(), "note": note_fn()})
            st = relay.read_json(Path(RELAY_DIR, "session_state.json"), {}); st["last_seen"] = relay.utcnow()
            relay.write_json(Path(RELAY_DIR, "session_state.json"), st)
            relay_push("heartbeat")
        except Exception as e:
            print("heartbeat error:", e)

def run_job(jdir, job):
    run_dir = Path(RELAY_DIR, "runs", job["id"]); run_dir.mkdir(parents=True, exist_ok=True)
    working = Path(KAGGLE_ROOT, "working")
    if working.exists(): shutil.rmtree(working)
    working.mkdir(parents=True)
    log_path = Path("/content", f"{job['id']}.log")
    t0 = time.time()
    status = {"state": "running", "started": relay.utcnow(), "gpu": gpu_name, "kind": job["kind"]}
    relay.write_json(run_dir / "status.json", status); relay_push(f"start {job['id']}")
    stop_evt = threading.Event()
    hb = threading.Thread(target=heartbeat_loop, args=(run_dir, stop_evt, lambda: f"{(time.time()-t0)/60:.0f} min"), daemon=True)
    hb.start()
    outcome = "done"; reason = ""
    try:
        if job["kind"] == "smoke":
            import torch
            a = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t1 = time.time()
            for _ in range(10): a = a @ a / 4096.0
            torch.cuda.synchronize()
            (working / "metrics.json").write_text(json.dumps({"gpu": gpu_name, "matmul_10x4096_s": time.time() - t1,
                "torch": torch.__version__, "cuda": torch.version.cuda}, indent=1))
            log_path.write_text(sh(["nvidia-smi"], check=False).stdout)
        else:
            ensure_inputs(job)
            env = dict(os.environ, **{k: str(v) for k, v in (job.get("env") or {}).items()})
            with open(log_path, "w") as fh:
                proc = subprocess.Popen(["papermill", str(jdir / job["notebook"]), str(working / "executed.ipynb"),
                                         "--log-output", "--cwd", str(working)], stdout=fh, stderr=subprocess.STDOUT, env=env)
                try:
                    rc = proc.wait(timeout=float(job["max_hours"]) * 3600.0)
                except subprocess.TimeoutExpired:
                    proc.kill(); rc = -9; outcome = "timeout"; reason = f"exceeded max_hours {job['max_hours']}"
            if rc != 0 and outcome == "done":
                outcome = "failed"; reason = f"papermill exit {rc}"
    except Exception as e:
        outcome = "failed"; reason = f"{type(e).__name__}: {e}"; log_path.open("a").write(traceback.format_exc())
    finally:
        stop_evt.set()
    elapsed = time.time() - t0
    # outputs: small -> relay, everything -> Drive, optionally -> a private Kaggle dataset
    out_dir = run_dir / "out"; out_dir.mkdir(exist_ok=True)
    for p in relay.select_relay_outputs(working, job.get("relay_outputs") or []):
        tgt = out_dir / p.relative_to(working); tgt.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, tgt)
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        (run_dir / "log_tail.txt").write_text("\\n".join(lines[-relay.LOG_TAIL_LINES:]))
    drive_dir = ""
    if USE_DRIVE:
        drive_dir = f"{DRIVE_ROOT}/runs/{job['id']}"
        try:
            shutil.copytree(working, f"{drive_dir}/working", dirs_exist_ok=True)
            if log_path.exists(): shutil.copy(log_path, f"{drive_dir}/log.txt")
        except Exception as e:
            print("Drive copy failed:", e); drive_dir = f"FAILED: {e}"
    kaggle_ds = ""
    if job.get("publish_kaggle"):
        slug = job.get("kaggle_dataset_slug") or f"colab-{job['id']}"
        meta = {"title": slug, "id": f"{os.environ['KAGGLE_USERNAME']}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
        (working / "dataset-metadata.json").write_text(json.dumps(meta))
        cp = sh(["kaggle", "datasets", "create", "-p", str(working), "-r", "zip", "--dir-mode", "zip"], check=False)
        kaggle_ds = meta["id"] if cp.returncode == 0 else f"FAILED: {cp.stderr[-300:]}"
    units = relay.estimate_units(gpu_name, elapsed, session.get("rates"))
    status.update({"state": outcome, "reason": reason, "finished": relay.utcnow(), "elapsed_s": elapsed,
                   "units": units, "drive_dir": drive_dir, "kaggle_dataset": kaggle_ds,
                   "relay_outputs": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())})
    relay.write_json(run_dir / "status.json", status)
    return outcome, units
"""),
    ("code", """import importlib
idle_since = time.time()
while True:
    relay_pull()
    relay = importlib.reload(relay)   # pick up lib changes pushed from the laptop without a restart
    session = relay.read_json(Path(RELAY_DIR, "session.json"), {})
    state = relay.read_json(Path(RELAY_DIR, "session_state.json"), {}) or state
    if session.get("stop"):
        print("session stop flag set - exiting"); state["state"] = "stopped"; relay.write_json(Path(RELAY_DIR, "session_state.json"), state); relay_push("worker stopped"); break
    pending = sorted(Path(RELAY_DIR, "jobs", "pending").glob("*/job.json"),
                     key=lambda p: (relay.read_json(p, {}).get("priority", 100), relay.read_json(p, {}).get("created_at", "")))
    if not pending:
        if time.time() - idle_since > IDLE_EXIT_MIN * 60:
            print(f"idle for {IDLE_EXIT_MIN} min - exiting to save units"); state["state"] = "idle-exit"; relay.write_json(Path(RELAY_DIR, "session_state.json"), state); relay_push("worker idle exit"); break
        state["last_seen"] = relay.utcnow(); state["state"] = "idle"; relay.write_json(Path(RELAY_DIR, "session_state.json"), state); relay_push("idle")
        time.sleep(POLL_S); continue
    idle_since = time.time()
    jpath = pending[0]; jdir = jpath.parent; job = relay.read_json(jpath, {})
    allowed, reason = relay.job_allowed(job, session, float(state.get("spent_units", 0.0)), gpu_name)
    print(relay.utcnow(), job.get("id"), "->", "RUN" if allowed else "REFUSE", reason)
    if not allowed:
        dest = Path(RELAY_DIR, "jobs", "refused", jdir.name); shutil.move(str(jdir), str(dest))
        relay.write_json(Path(RELAY_DIR, "runs", job.get("id", jdir.name), "status.json"), {"state": "refused", "reason": reason, "at": relay.utcnow()})
        relay_push(f"refuse {jdir.name}"); continue
    dest = Path(RELAY_DIR, "jobs", "running", jdir.name); shutil.move(str(jdir), str(dest)); jdir = dest
    state["state"] = f"running {job['id']}"; relay.write_json(Path(RELAY_DIR, "session_state.json"), state)
    outcome, units = run_job(jdir, job)
    state["spent_units"] = float(state.get("spent_units", 0.0)) + units; state["state"] = "idle"; state["last_seen"] = relay.utcnow()
    relay.write_json(Path(RELAY_DIR, "session_state.json"), state)
    final = Path(RELAY_DIR, "jobs", "done" if outcome == "done" else "failed", jdir.name); shutil.move(str(jdir), str(final))
    relay_push(f"{outcome} {job['id']} ({units:.2f} units)")
    print(f"{job['id']}: {outcome}, {units:.2f} units, spent {state['spent_units']:.2f}/{session.get('budget_units')}")
"""),
]


def build_worker_notebook(relay_repo: str) -> dict:
    cells = []
    for kind, src in WORKER_CELLS:
        text = src.replace("{RELAY_REPO}", relay_repo)
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def cmd_worker_notebook(args) -> int:
    nb = build_worker_notebook(args.repo or relay_repo())
    out = Path(args.out) if args.out else WORKER_NB
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells) - open it in Colab from GitHub and Run all")
    return 0


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relay-dir", default=str(DEFAULT_RELAY_DIR))
    ap.add_argument("--no-push", action="store_true", help="commit locally only (tests)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("init"); p.add_argument("--repo", required=True); p.add_argument("--no-create", action="store_true")
    p.set_defaults(func=cmd_init)
    p = sp.add_parser("session")
    p.add_argument("--units", type=float, default=0.0); p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--gpu", default="L4"); p.add_argument("--scope", default="smoke,notebook")
    p.add_argument("--host-approved", default=""); p.add_argument("--rate", action="append", help="GPU=units_per_hour")
    p.add_argument("--stop", action="store_true")
    p.set_defaults(func=cmd_session)
    p = sp.add_parser("queue")
    p.add_argument("--spec", required=True); p.add_argument("--max-hours", type=float, default=3.0)
    p.add_argument("--priority", type=int, default=100); p.add_argument("--job-id")
    p.add_argument("--publish-kaggle", action="store_true"); p.add_argument("--kaggle-dataset-slug")
    p.add_argument("--no-competition", action="store_true"); p.add_argument("--no-build", action="store_true")
    p.add_argument("--env", action="append"); p.add_argument("--note")
    p.add_argument("--input-root", default=relay.COLAB_INPUT_ROOT,
                   help="rewrite /kaggle/input literals to this writable root (Colab's /kaggle/input is read-only)")
    p.set_defaults(func=cmd_queue)
    p = sp.add_parser("smoke"); p.add_argument("--job-id"); p.set_defaults(func=cmd_smoke)
    p = sp.add_parser("status"); p.set_defaults(func=cmd_status)
    p = sp.add_parser("fetch"); p.add_argument("--job", required=True); p.add_argument("--dest"); p.set_defaults(func=cmd_fetch)
    p = sp.add_parser("worker-notebook"); p.add_argument("--repo"); p.add_argument("--out"); p.set_defaults(func=cmd_worker_notebook)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
