r"""Drive a Colab GPU session from the laptop with the OFFICIAL Colab CLI (google-colab-cli), via WSL.

The CLI is Linux/macOS only, so it lives in WSL Ubuntu (root) with ADC auth (gcloud, four scopes -
see `colab skill`). Every call here is `wsl -d Ubuntu -u root -- bash <script>` with a script file
written under C:/temp/colab_cli/ (no shell-quoting across the Windows/WSL boundary).

Rule 8 (no automatic GPU launch) is the SAME gate as the relay: `colab new` is refused unless the
relay's session.json carries a host-recorded budget whose scope and remaining units admit the job
(`colab_relay.job_allowed`). Units are accounted by wall-clock x the session's rate table in
C:/temp/colab_cli/ledger.json, from `new` to `stop`. `stop` is always attempted unless --keep.

    check                         whoami + sessions
    run   --job <relay job id>    gate -> new (or reuse) -> upload -> bootstrap (inputs, 3.12 kernel) ->
                                  papermill detached -> poll -> download outputs -> stop
    poll  --job <id> --session S  one status read of a running job
    fetch --job <id> --session S  download /kaggle/working (zip) + log
    stop  --session S             release the VM and close the ledger entry
    sessions                      list server-side sessions

Usage
-----
  .\.venv\Scripts\python.exe scripts\core\colab_cli.py check
  .\.venv\Scripts\python.exe scripts\core\colab_cli.py run --job p21-parity-f0-a6 --session biohub-l4 --gpu L4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))
import colab_relay as relay  # noqa: E402

WORK = Path(os.environ.get("BIOHUB_COLAB_CLI_DIR", "C:/temp/colab_cli"))
RELAY_DIR = Path(relay.read_json(REPO / "scripts" / "core" / "colab_relay.json", {}).get("relay_dir", "C:/temp/colab_relay"))
LEDGER = WORK / "ledger.json"
DISTRO = os.environ.get("BIOHUB_WSL_DISTRO", "Ubuntu")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BOX = re.compile("[\u2500-\u257f]")
BOOTSTRAP = REPO / "scripts" / "core" / "colab_vm_bootstrap.py"


def to_wsl(path: str | Path) -> str:
    p = str(Path(path).resolve()).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else p


def clean(text: str) -> str:
    return "\n".join(l.rstrip() for l in BOX.sub("", ANSI.sub("", text)).splitlines() if l.strip())


_n = 0


def colab(args: list[str], stdin_text: str | None = None, timeout: int = 7200, check: bool = True) -> subprocess.CompletedProcess:
    """Run `colab --auth=adc <args>` inside WSL through a script file; returns the completed process (cleaned)."""
    global _n
    _n += 1
    WORK.mkdir(parents=True, exist_ok=True)
    script = WORK / f"cmd_{os.getpid()}_{_n}.sh"
    stdin_file = None
    body = "export PATH=/root/google-cloud-sdk/bin:/root/.local/bin:$PATH\nexport COLUMNS=160\n"
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in args)
    if stdin_text is not None:
        stdin_file = WORK / f"stdin_{os.getpid()}_{_n}.py"
        stdin_file.write_text(stdin_text, encoding="utf-8", newline="\n")
        body += f"colab --auth=adc {quoted} < '{to_wsl(stdin_file)}'\n"
    else:
        body += f"colab --auth=adc {quoted}\n"
    script.write_text(body, encoding="utf-8", newline="\n")
    cp = subprocess.run(["wsl", "-d", DISTRO, "-u", "root", "--", "bash", to_wsl(script)],
                        capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    cp.stdout, cp.stderr = clean(cp.stdout or ""), clean(cp.stderr or "")
    if check and cp.returncode != 0:
        raise SystemExit(f"colab {' '.join(args)} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


def exec_py(session: str, code: str, timeout_s: int) -> str:
    cp = colab(["exec", "-s", session, "--timeout", str(timeout_s)], stdin_text=code, timeout=timeout_s + 120, check=False)
    out = cp.stdout + ("\n" + cp.stderr if cp.stderr else "")
    if cp.returncode != 0:
        raise SystemExit(f"colab exec failed ({cp.returncode}):\n{out[-3000:]}")
    return out


# ------------------------------------------------------------------ ledger (units by wall-clock)
def ledger() -> dict:
    return relay.read_json(LEDGER, {"sessions": {}, "spent_units": 0.0})


def save_ledger(l: dict) -> None:
    relay.write_json(LEDGER, l)


def spent_units(l: dict, rates: dict, now: float | None = None) -> float:
    now = now or time.time()
    total = float(l.get("spent_units", 0.0))
    for name, s in l.get("sessions", {}).items():
        if s.get("open"):
            total += relay.estimate_units(s.get("gpu", "unknown"), now - float(s["started_ts"]), rates)
    return total


# ------------------------------------------------------------------ commands
def cmd_check(args) -> int:
    print(colab(["whoami"], check=False).stdout)
    print(colab(["sessions"], check=False).stdout)
    print(f"ledger: {json.dumps(ledger())}")
    return 0


def cmd_sessions(args) -> int:
    print(colab(["sessions"], check=False).stdout)
    return 0


def session_exists(name: str) -> bool:
    """`colab sessions` prints owned sessions as `[<name>] <endpoint> | Hardware: ...` and orphans as `[?] ...`."""
    out = colab(["sessions"], check=False).stdout
    return any(line.strip().startswith(f"[{name}]") for line in out.splitlines())


def load_job(job_id: str) -> tuple[dict, Path]:
    for state in ("pending", "running", "done", "failed", "refused"):
        jdir = RELAY_DIR / "jobs" / state / job_id
        if (jdir / "job.json").exists():
            return relay.read_json(jdir / "job.json"), jdir
    raise SystemExit(f"job {job_id} not found under {RELAY_DIR}/jobs/*/")


def cmd_run(args) -> int:
    job, jdir = load_job(args.job)
    session_cfg = relay.read_json(RELAY_DIR / "session.json", {})
    rates = session_cfg.get("rates", relay.DEFAULT_RATES)
    l = ledger()
    gpu = args.gpu
    ok, reason = relay.job_allowed(job, session_cfg, spent_units(l, rates), gpu)
    print(f"gate: {'RUN' if ok else 'REFUSE'} - {reason}")
    if not ok:
        return 2
    name = args.session
    if name in l["sessions"] and l["sessions"][name].get("open") and session_exists(name):
        print(f"reusing session {name}")
    else:
        print(f"colab new -s {name} --gpu {gpu} ...")
        out = colab(["new", "-s", name, "--gpu", gpu], timeout=900).stdout
        print(out[-400:])
        l["sessions"][name] = {"gpu": gpu, "started_ts": time.time(), "started": relay.utcnow(), "open": True, "jobs": []}
        save_ledger(l)
    l["sessions"][name].setdefault("jobs", []).append(job["id"])
    save_ledger(l)

    # upload job files + token + bootstrap
    exec_py(name, f"import os; os.makedirs('/content/jobs/{job['id']}', exist_ok=True); os.makedirs('/content/runs/{job['id']}', exist_ok=True); print('dirs ok')", 60)
    colab(["upload", "-s", name, to_wsl(jdir / "job.json"), f"/content/jobs/{job['id']}/job.json"], timeout=300)
    colab(["upload", "-s", name, to_wsl(jdir / job["notebook"]), f"/content/jobs/{job['id']}/{job['notebook']}"], timeout=600)
    token = Path.home() / ".kaggle" / "access_token"
    colab(["upload", "-s", name, to_wsl(token), "/content/.kaggle_token"], timeout=120)
    colab(["upload", "-s", name, to_wsl(BOOTSTRAP), "/content/colab_vm_bootstrap.py"], timeout=120)
    print("uploaded job, notebook, token, bootstrap")

    # bootstrap: inputs + kernel + detached papermill (long: downloads + torch)
    t0 = time.time()
    out = exec_py(name, f"import os, runpy; os.environ['BIOHUB_JOB_ID'] = '{job['id']}'; "
                        f"runpy.run_path('/content/colab_vm_bootstrap.py', run_name='__main__')", int(args.bootstrap_timeout))
    print(out[-2500:])
    print(f"bootstrap took {time.time() - t0:.0f}s")
    (WORK / f"{job['id']}_bootstrap.txt").write_text(out, encoding="utf-8")
    if args.no_wait:
        print("launched; poll with: colab_cli.py poll --job", job["id"], "--session", name)
        return 0
    return wait_and_fetch(job, name, args, l)


POLL_SNIPPET = """
import json, os
r = '/content/runs/{job}'
st = json.load(open(r + '/status.json')) if os.path.exists(r + '/status.json') else {{}}
ec = open(r + '/exit_code').read().strip() if os.path.exists(r + '/exit_code') else None
tail = ''
if os.path.exists(r + '/log.txt'):
    lines = open(r + '/log.txt', errors='replace').read().splitlines()
    tail = '\\n'.join(lines[-{n}:])
alive = False
if st.get('pid'):
    try:
        os.kill(int(st['pid']), 0); alive = True
    except OSError:
        alive = False
print('POLLJSON ' + json.dumps({{'state': st.get('state'), 'exit_code': ec, 'alive': alive, 'started': st.get('started'), 'tail': tail}}))
"""


def poll_once(job_id: str, session: str, n_tail: int = 12) -> dict:
    out = exec_py(session, POLL_SNIPPET.format(job=job_id, n=n_tail), 120)
    for line in out.splitlines():
        if line.startswith("POLLJSON "):
            return json.loads(line[len("POLLJSON "):])
    raise SystemExit("poll: no POLLJSON line in output:\n" + out[-1500:])


def cmd_poll(args) -> int:
    p = poll_once(args.job, args.session, args.tail)
    print(json.dumps({k: v for k, v in p.items() if k != "tail"}))
    print(p.get("tail", ""))
    return 0


def fetch_outputs(job: dict, session: str) -> Path:
    dest = Path("C:/temp/colab_runs") / job["id"]
    dest.mkdir(parents=True, exist_ok=True)
    exec_py(session, f"import shutil; shutil.make_archive('/content/runs/{job['id']}/working', 'zip', '/kaggle/working'); print('zipped')", 900)
    colab(["download", "-s", session, f"/content/runs/{job['id']}/working.zip", to_wsl(dest / "working.zip")], timeout=1800)
    colab(["download", "-s", session, f"/content/runs/{job['id']}/log.txt", to_wsl(dest / "log.txt")], timeout=600, check=False)
    print(f"downloaded to {dest}")
    return dest


def wait_and_fetch(job: dict, name: str, args, l: dict) -> int:
    t0 = time.time()
    deadline = t0 + float(job["max_hours"]) * 3600.0
    last_bucket = -1
    while True:
        p = poll_once(job["id"], name)
        elapsed = time.time() - t0
        if p["exit_code"] is not None or not p["alive"]:
            break
        bucket = int(elapsed // 600)
        if bucket != last_bucket:
            print(f"[{elapsed / 60:.0f} min] running - " + (p["tail"].splitlines()[-1][:120] if p["tail"] else ""), flush=True)
            last_bucket = bucket
        if time.time() > deadline:
            print("max_hours exceeded - killing papermill")
            exec_py(name, f"import os, signal, json; st=json.load(open('/content/runs/{job['id']}/status.json')); os.killpg(int(st['pid']), signal.SIGKILL)", 60)
            p = {"exit_code": "timeout", "alive": False, "tail": p["tail"]}
            break
        time.sleep(120)
    outcome = "done" if str(p.get("exit_code")) == "0" else ("timeout" if p.get("exit_code") == "timeout" else "failed")
    print(f"papermill finished: {outcome} (exit {p.get('exit_code')}) after {(time.time() - t0) / 60:.0f} min")
    print(p.get("tail", "")[-2000:])
    dest = fetch_outputs(job, name)
    result = {"job": job["id"], "session": name, "outcome": outcome, "exit_code": p.get("exit_code"),
              "papermill_minutes": (time.time() - t0) / 60, "dest": str(dest), "finished": relay.utcnow()}
    (dest / "result.json").write_text(json.dumps(result, indent=1))
    if not args.keep:
        do_stop(name, l)
    else:
        print(f"session {name} kept open (--keep); remember colab_cli.py stop --session {name}")
    return 0 if outcome == "done" else 1


def do_stop(name: str, l: dict | None = None) -> None:
    l = l or ledger()
    cp = colab(["stop", "-s", name], check=False, timeout=300)
    print(cp.stdout[-300:] or cp.stderr[-300:])
    s = l["sessions"].get(name)
    if s and s.get("open"):
        rates = relay.read_json(RELAY_DIR / "session.json", {}).get("rates", relay.DEFAULT_RATES)
        units = relay.estimate_units(s.get("gpu", "unknown"), time.time() - float(s["started_ts"]), rates)
        s.update({"open": False, "stopped": relay.utcnow(), "units": units})
        l["spent_units"] = float(l.get("spent_units", 0.0)) + units
        save_ledger(l)
        print(f"session {name} closed: ~{units:.2f} units ({(time.time() - float(s['started_ts'])) / 60:.0f} min on {s.get('gpu')}); ledger total {l['spent_units']:.2f}")


def cmd_stop(args) -> int:
    do_stop(args.session)
    return 0


def cmd_fetch(args) -> int:
    job, _ = load_job(args.job)
    fetch_outputs(job, args.session)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check").set_defaults(func=cmd_check)
    sp.add_parser("sessions").set_defaults(func=cmd_sessions)
    p = sp.add_parser("run")
    p.add_argument("--job", required=True); p.add_argument("--session", default="biohub-l4"); p.add_argument("--gpu", default="L4")
    p.add_argument("--keep", action="store_true"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--bootstrap-timeout", type=int, default=5400)
    p.set_defaults(func=cmd_run)
    p = sp.add_parser("poll"); p.add_argument("--job", required=True); p.add_argument("--session", default="biohub-l4"); p.add_argument("--tail", type=int, default=12)
    p.set_defaults(func=cmd_poll)
    p = sp.add_parser("fetch"); p.add_argument("--job", required=True); p.add_argument("--session", default="biohub-l4"); p.set_defaults(func=cmd_fetch)
    p = sp.add_parser("stop"); p.add_argument("--session", default="biohub-l4"); p.set_defaults(func=cmd_stop)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
