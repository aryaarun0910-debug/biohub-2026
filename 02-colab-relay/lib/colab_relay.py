r"""Colab job relay - the contract shared by the laptop CLI and the Colab worker.

WHY THIS EXISTS
---------------
The host pays Colab compute units (FACT-0318) and wants Claude to drive them from the laptop.
Colab runtimes cannot be reached from outside (and remote shells are disallowed by Colab's
terms), so the control channel is a PRIVATE GITHUB REPO ("the relay"):

    laptop  --push job (notebook + manifest)-->  relay repo  <--pull--  Colab worker notebook
    laptop  <--pull status / heartbeat / metrics--  relay repo  <--push--  Colab worker

Large artefacts go to Google Drive (mounted on the worker) and, when asked, to a private Kaggle
dataset the laptop can download with the Kaggle API it already has.

RULE 8 (no GPU launch is automatic) is enforced MECHANICALLY on the runtime, not by convention:
a job runs only if `session.json` carries a host-authorised budget whose scope admits the job's
kind, and the job's worst-case unit cost fits the remaining budget. `job_allowed()` below is that
gate; the worker refuses everything else and says so in `runs/<job>/status.json`.

This module has NO Colab or Kaggle imports so it can be unit-tested on the laptop and copied
verbatim into the relay repo as `lib/colab_relay.py`.

Relay layout
------------
    session.json                host-authorised budget (see SESSION_SCHEMA)
    session_state.json          worker-written: units spent (estimate), gpu, started, last_seen
    jobs/pending/<id>/job.json  + notebook.ipynb  (queued by the laptop)
    jobs/running/<id>/          (claimed by the worker)
    jobs/done/<id>/  jobs/failed/<id>/  jobs/refused/<id>/
    runs/<id>/status.json       state, timings, gpu, units, reason
    runs/<id>/heartbeat.json    updated every HEARTBEAT_S while running
    runs/<id>/log_tail.txt      last LOG_TAIL_LINES of the notebook log
    runs/<id>/out/              small declared outputs (<= RELAY_OUTPUT_MAX_BYTES each)
    lib/colab_relay.py          this file
"""
from __future__ import annotations

import fnmatch
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

RELAY_VERSION = 1
COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_INPUT = "/kaggle/input"            # what factory-built notebooks hard-code
COLAB_INPUT_ROOT = "/content/kaggle/input"  # Colab mounts /kaggle/input READ-ONLY (measured 2026-08-26,
                                            # EROFS on symlink) so jobs are rewritten to this writable root
KAGGLE_WORKING = "/kaggle/working"          # writable on Colab (the smoke wrote there)

# Compute-unit rates per GPU-hour. UNVERIFIED defaults, deliberately rounded UP (FACT-0318):
# the host overrides them in session.json["rates"] from the Colab resources panel.
DEFAULT_RATES = {"T4": 2.0, "L4": 5.0, "A100": 12.0, "unknown": 12.0}

JOB_KINDS = ("notebook", "smoke")
HEARTBEAT_S = 120
LOG_TAIL_LINES = 400
RELAY_OUTPUT_MAX_BYTES = 5 * 1024 * 1024

JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# --------------------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------------------
SESSION_SCHEMA = {
    "version": RELAY_VERSION,
    "active": False,                 # False -> the worker runs nothing
    "host_approval": "",             # the host's words, verbatim; REQUIRED when active
    "approved_at": "",               # utc
    "expires_at": "",                # utc; the worker stops claiming jobs after this
    "budget_units": 0.0,             # compute units the host authorised for this session
    "scope_kinds": ["smoke"],        # job kinds admitted
    "gpu": "L4",                     # the GPU the host intends to attach (advisory)
    "rates": DEFAULT_RATES,          # units per GPU-hour, host-verified
    "stop": False,                   # set True from the laptop to end the worker loop
}

JOB_SCHEMA = {
    "version": RELAY_VERSION,
    "id": "",                        # JOB_ID_RE
    "kind": "notebook",              # JOB_KINDS
    "created_at": "",
    "priority": 100,                 # lower runs first
    "spec": "",                      # repo-relative kaggle spec path (provenance)
    "notebook": "notebook.ipynb",    # file next to job.json
    "notebook_sha256": "",
    "datasets": [],                  # owner/slug, materialised at /kaggle/input/<slug>
    "competition": COMPETITION,      # or "" when the job needs no competition data
    "input_root": COLAB_INPUT_ROOT,  # where the worker materialises <slug>/ and <competition>/;
                                     # the queued notebook has its /kaggle/input literals rewritten to it
    "input_rewrites": 0,             # how many literals were rewritten (0 = notebook untouched)
    "max_hours": 3.0,                # hard timeout AND the unit estimate used by the gate
    "python": "3.12",                # kernel for notebook jobs. Kaggle kernels run 3.12 and the support
                                     # pack ships cp312 wheels; Colab's system Python is 3.13 (measured
                                     # 2026-08-26: the pack's offline install fails there). "" = system.
    "relay_outputs": ["submission.csv", "*.json", "*.csv", "*.log"],   # small files -> relay
    "drive_outputs": ["**/*"],       # everything -> Drive runs/<id>/working/
    "publish_kaggle": False,         # also upload /kaggle/working to a private Kaggle dataset
    "kaggle_dataset_slug": "",       # default colab-<id>
    "env": {},                       # extra env for the notebook process
    "note": "",
}

STATUS_STATES = ("queued", "refused", "running", "done", "failed", "timeout")


def validate_job(job: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(job, dict):
        return ["job is not an object"]
    if job.get("version") != RELAY_VERSION:
        errs.append(f"version must be {RELAY_VERSION}")
    if not JOB_ID_RE.match(str(job.get("id", ""))):
        errs.append("id must match ^[a-z0-9][a-z0-9_-]{2,63}$")
    if job.get("kind") not in JOB_KINDS:
        errs.append(f"kind must be one of {JOB_KINDS}")
    if job.get("kind") == "notebook" and not job.get("notebook"):
        errs.append("notebook jobs need a notebook file")
    try:
        if float(job.get("max_hours", 0)) <= 0:
            errs.append("max_hours must be > 0")
    except (TypeError, ValueError):
        errs.append("max_hours must be a number")
    for key in ("datasets", "relay_outputs", "drive_outputs"):
        if not isinstance(job.get(key, []), list):
            errs.append(f"{key} must be a list")
    for ds in job.get("datasets", []) or []:
        if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", str(ds)):
            errs.append(f"dataset must be owner/slug: {ds!r}")
    if not isinstance(job.get("env", {}), dict):
        errs.append("env must be an object")
    return errs


def validate_session(session: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(session, dict):
        return ["session is not an object"]
    if session.get("version") != RELAY_VERSION:
        errs.append(f"version must be {RELAY_VERSION}")
    if session.get("active"):
        if not str(session.get("host_approval", "")).strip():
            errs.append("an active session must record host_approval verbatim")
        try:
            if float(session.get("budget_units", 0)) <= 0:
                errs.append("an active session needs budget_units > 0")
        except (TypeError, ValueError):
            errs.append("budget_units must be a number")
        if not session.get("expires_at"):
            errs.append("an active session needs expires_at")
        else:
            try:
                parse_utc(session["expires_at"])
            except ValueError:
                errs.append("expires_at must be YYYY-MM-DDTHH:MM:SSZ")
        kinds = session.get("scope_kinds", [])
        if not isinstance(kinds, list) or not kinds or any(k not in JOB_KINDS for k in kinds):
            errs.append(f"scope_kinds must be a non-empty subset of {JOB_KINDS}")
    rates = session.get("rates", DEFAULT_RATES)
    if not isinstance(rates, dict) or any(float(v) <= 0 for v in rates.values()):
        errs.append("rates must map gpu -> positive units/hour")
    return errs


# --------------------------------------------------------------------------------------
# units and the rule-8 gate
# --------------------------------------------------------------------------------------
def gpu_family(gpu_name: str) -> str:
    n = (gpu_name or "").upper()
    for fam in ("A100", "L4", "T4"):
        if fam in n:
            return fam
    return "unknown"


def rate_for(gpu_name: str, rates: dict | None = None) -> float:
    rates = dict(DEFAULT_RATES, **(rates or {}))
    fam = gpu_family(gpu_name)
    return float(rates.get(fam, rates.get("unknown", DEFAULT_RATES["unknown"])))


def estimate_units(gpu_name: str, seconds: float, rates: dict | None = None) -> float:
    return rate_for(gpu_name, rates) * max(0.0, float(seconds)) / 3600.0


def job_allowed(job: dict, session: dict, spent_units: float, gpu_name: str,
                now_utc: str | None = None) -> tuple[bool, str]:
    """The rule-8 gate. Returns (allowed, reason). Refuses by default."""
    if validate_job(job):
        return False, "invalid job: " + "; ".join(validate_job(job))
    if validate_session(session):
        return False, "invalid session: " + "; ".join(validate_session(session))
    if not session.get("active"):
        return False, "session inactive - no host authorisation on record"
    if session.get("stop"):
        return False, "session stop flag set"
    now = parse_utc(now_utc) if now_utc else time.time()
    if now > parse_utc(session["expires_at"]):
        return False, f"session expired at {session['expires_at']}"
    if job["kind"] not in session.get("scope_kinds", []):
        return False, f"kind {job['kind']!r} outside authorised scope {session.get('scope_kinds')}"
    worst = estimate_units(gpu_name, float(job["max_hours"]) * 3600.0, session.get("rates"))
    budget = float(session["budget_units"])
    if spent_units + worst > budget + 1e-9:
        return False, (f"worst-case cost {worst:.1f} units on {gpu_family(gpu_name)} + spent "
                       f"{spent_units:.1f} exceeds budget {budget:.1f}")
    return True, f"ok: worst-case {worst:.1f} units, {budget - spent_units:.1f} remaining"


# --------------------------------------------------------------------------------------
# kaggle layout the worker must recreate
# --------------------------------------------------------------------------------------
def kaggle_layout_plan(job: dict) -> list[dict]:
    """What must exist under /kaggle/input for this job's notebook to run unmodified.

    Built notebooks resolve datasets at /kaggle/input/<slug> (and the nested
    /kaggle/input/datasets/<owner>/<slug>), and the competition at
    /kaggle/input/<competition> or /kaggle/input/competitions/<competition>.
    """
    root = (job.get("input_root") or KAGGLE_INPUT).rstrip("/")
    plan = []
    for ds in job.get("datasets", []) or []:
        owner, slug = ds.split("/", 1)
        plan.append({"kind": "dataset", "ref": ds, "path": f"{root}/{slug}",
                     "alias": f"{root}/datasets/{owner}/{slug}"})
    comp = job.get("competition") or ""
    if comp:
        plan.append({"kind": "competition", "ref": comp, "path": f"{root}/{comp}",
                     "alias": f"{root}/competitions/{comp}"})
    return plan


def rewrite_input_root(nb: dict, new_root: str, old_root: str = KAGGLE_INPUT) -> int:
    """Replace every `old_root` literal in the notebook's code cells with `new_root`. Returns the count."""
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if old_root in src:
            n += src.count(old_root)
            cell["source"] = src.replace(old_root, new_root).splitlines(keepends=True)
    return n


def select_relay_outputs(working: Path, patterns: list[str], max_bytes: int = RELAY_OUTPUT_MAX_BYTES) -> list[Path]:
    """Files under `working` matching any pattern (by name or relative path), small enough for git."""
    out: list[Path] = []
    if not working.exists():
        return out
    for p in sorted(working.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(working).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns):
            if p.stat().st_size <= max_bytes:
                out.append(p)
    return out


# --------------------------------------------------------------------------------------
# small file helpers
# --------------------------------------------------------------------------------------
def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def new_session(host_approval: str, budget_units: float, scope_kinds: list[str], hours: float,
                gpu: str = "L4", rates: dict | None = None, now_utc: str | None = None) -> dict:
    now = parse_utc(now_utc) if now_utc else time.time()
    exp = datetime.fromtimestamp(now + hours * 3600.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s = dict(SESSION_SCHEMA)
    s.update({
        "active": True, "host_approval": host_approval, "approved_at": now_utc or utcnow(),
        "expires_at": exp, "budget_units": float(budget_units), "scope_kinds": list(scope_kinds),
        "gpu": gpu, "rates": dict(DEFAULT_RATES, **(rates or {})), "stop": False,
    })
    return s


def new_job(job_id: str, kind: str = "notebook", **fields) -> dict:
    j = dict(JOB_SCHEMA)
    j.update({"id": job_id, "kind": kind, "created_at": utcnow()})
    j.update(fields)
    if kind == "smoke":
        j["notebook"] = ""
        j["datasets"] = []
        j["competition"] = ""
        j["max_hours"] = min(float(j.get("max_hours", 0.1)), 0.25)
    return j
