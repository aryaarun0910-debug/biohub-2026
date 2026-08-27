r"""Kaggle queue runner - keep both GPU slots busy through an ORDERED list of built specs.

WHY THIS EXISTS (2026-08-27)
----------------------------
The campaign has ~19 GPU-hours of Kaggle quota, two concurrent batch slots (FACT-0061), and a
queue of built specs that each take 1.5-3 h: submission A/Bs (P24, p23), a recipe-parity LOEO
(p25), the S5 edge training run and its LOEO consumer (which may only start once its producer
kernel is COMPLETE, because it mounts that kernel's output). Driving that by hand means idle
slots between sessions. This runner does the bookkeeping: push in order as slots free, honour
producer -> consumer dependencies, poll status, fetch the named outputs on completion, run the
release audit for submission-bearing specs, and submit ONLY the specs explicitly named with
``--submit``. It never re-pushes a kernel that already ran, and it stops on its own when the
queue drains or ``--max-hours`` elapses.

Every Kaggle side effect goes through scripts/core/kaggle_factory.py (push / status / fetch /
audit / submitcmd), so the release-receipt contract is unchanged. State lives in a JSON file so
a killed runner resumes where it stopped.

Usage
-----
  .\.venv\Scripts\python.exe scripts\core\kaggle_queue.py run ^
      --specs scripts/kaggle_specs/p24_deepcenter_best_veto.json ^
              scripts/kaggle_specs/p25_recipe_parity_loeo_f0.json ^
              scripts/kaggle_specs/p23_icom155.json ^
              scripts/kaggle_specs/h1r_edge_s5.json ^
              scripts/kaggle_specs/deploy_h1r_edge_s5_loeo_f0.json ^
      --submit p24_deepcenter_best_veto,p23_icom155 --state C:/temp/queue/state.json --poll 300
  .\.venv\Scripts\python.exe scripts\core\kaggle_queue.py status --state C:/temp/queue/state.json

Without ``--submit`` a completed submission-bearing spec is fetched and audited and the
receipt-bound submit command is PRINTED, never run (CLAUDE.md rule 8).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))
import kaggle_factory as KF  # noqa: E402

MAX_RUNNING_DEFAULT = 2          # FACT-0061: two concurrent GPU batch sessions
TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


@dataclass
class Item:
    spec_path: str
    name: str
    slug: str
    expects_submission: bool
    deps: list[str] = field(default_factory=list)      # producer slugs this item mounts
    fetch: list[str] = field(default_factory=list)
    status: str = "pending"        # pending | running | complete | fetched | audited | submitted | error
    kernel_version: int = 1
    kaggle_status: str = ""
    pushed_at: float | None = None
    finished_at: float | None = None
    note: str = ""


# ------------------------------------------------------------------------------ pure logic
def default_fetch(spec: dict) -> list[str]:
    if spec.get("queue_fetch"):
        return list(spec["queue_fetch"])
    env = KF._spec_env(spec)
    if spec.get("expects_submission"):
        return ["submission.csv", "run_stats.csv"]
    if "BIOHUB_LOEO_FOLD" in env:
        fold, arm = env["BIOHUB_LOEO_FOLD"], env.get("BIOHUB_LOEO_ARM", "strict")
        return [f"loeo_split{fold}_{arm}.csv.gz", "loeo_manifest.json", "run_stats.csv"]
    return ["metrics.json", "summary.json", "config.json"]


def item_from_spec(spec_path: Path, spec: dict, queue_slugs: set[str], owner: str = KF.OWNER) -> Item:
    deps = []
    for src in spec.get("kernel_sources", []):
        o, _, slug = str(src).partition("/")
        if o == owner and slug in queue_slugs:
            deps.append(slug)
    return Item(spec_path=str(spec_path), name=spec["name"], slug=spec["slug"],
                expects_submission=bool(spec.get("expects_submission")), deps=deps,
                fetch=default_fetch(spec))


def deps_ready(item: Item, by_slug: dict[str, Item]) -> bool:
    return all(by_slug[d].status in {"complete", "fetched", "audited", "submitted"}
               for d in item.deps if d in by_slug)


def plan_pushes(items: list[Item], max_running: int) -> list[Item]:
    """Which pending items to push now: in queue order, dependency-ready, within the slot cap."""
    running = sum(1 for i in items if i.status == "running")
    by_slug = {i.slug: i for i in items}
    out: list[Item] = []
    for it in items:
        if running + len(out) >= max_running:
            break
        if it.status == "pending" and deps_ready(it, by_slug):
            out.append(it)
    return out


def tick(items: list[Item], *, status_fn, push_fn, complete_fn, max_running: int, now: float) -> list[str]:
    """One scheduler step. Returns human-readable events. All side effects go through the
    injected callables so the logic is unit-testable."""
    events: list[str] = []
    for it in items:
        if it.status != "running":
            continue
        st = status_fn(it.slug)
        it.kaggle_status = st
        head = st.split(" ")[0]
        if head in TERMINAL_OK:
            it.status = "complete"
            it.finished_at = now
            events.append(f"{it.name}: COMPLETE after {(now - (it.pushed_at or now)) / 3600:.2f} h")
            try:
                complete_fn(it)
            except Exception as exc:  # fetch/audit/submit failures must not kill the queue
                it.note = f"post-complete step failed: {exc}"
                events.append(f"{it.name}: {it.note}")
        elif head in TERMINAL_BAD:
            it.status = "error"
            it.finished_at = now
            events.append(f"{it.name}: {st}")
    for it in plan_pushes(items, max_running):
        try:
            rc = push_fn(it)
        except Exception as exc:
            rc = 1
            it.note = f"push raised: {exc}"
        if rc == 0:
            it.status = "running"
            it.pushed_at = now
            events.append(f"{it.name}: pushed -> {KF.OWNER}/{it.slug} v{it.kernel_version}")
        else:
            it.status = "error"
            it.finished_at = now
            events.append(f"{it.name}: push FAILED ({it.note or rc}) - skipped")
    return events


def all_done(items: list[Item]) -> bool:
    return all(i.status not in {"pending", "running"} for i in items)


# ------------------------------------------------------------------------------ state
def load_state(path: Path) -> list[Item]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Item(**row) for row in raw]


def save_state(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(i) for i in items], indent=2), encoding="utf-8")
    tmp.replace(path)


def build_queue(spec_paths: list[Path], existing: list[Item], versions: dict[str, int]) -> list[Item]:
    """Merge the requested specs with saved state; saved items keep their status."""
    by_name = {i.name: i for i in existing}
    specs = [(p, KF.load_spec(p)) for p in spec_paths]
    queue_slugs = {s["slug"] for _p, s in specs}
    items: list[Item] = []
    for p, s in specs:
        if s["name"] in by_name:
            items.append(by_name[s["name"]])
            continue
        if not KF.built_nb(s).exists():
            raise SystemExit(f"{s['name']}: not built ({KF.built_nb(s)}); run kaggle_factory build first")
        it = item_from_spec(p, s, queue_slugs)
        it.kernel_version = int(versions.get(s["name"], 1))
        items.append(it)
    return items


# ------------------------------------------------------------------------------ side effects
def real_push(it: Item) -> int:
    spec = KF.load_spec(Path(it.spec_path))
    rc = KF.cmd_push(spec)
    if rc == 0:
        man = json.loads(KF.manifest_path(spec).read_text(encoding="utf-8"))
        pushed = man.get("pushed_slug")
        if pushed and pushed != it.slug:
            it.note = f"SLUG DIVERGENCE {it.slug} -> {pushed}; spec must be updated"
            it.slug = pushed
    return rc


def make_complete_fn(submit_names: set[str], dest_root: Path):
    def complete(it: Item) -> None:
        spec = KF.load_spec(Path(it.spec_path))
        dest = dest_root / it.name
        KF.cmd_fetch(spec, it.fetch, dest)
        it.status = "fetched"
        if not it.expects_submission:
            return
        KF.cmd_audit(spec, dest, it.kernel_version)
        it.status = "audited"
        message = f"{it.name}: {spec.get('purpose', '')[:160]}"
        KF.cmd_submitcmd(spec, message, dest, it.kernel_version)   # receipt-bound gate; prints the command
        if it.name not in submit_names:
            it.note = "audited; submit command printed, NOT submitted (not in --submit)"
            return
        r = KF.run([sys.executable, "-m", "kaggle", "competitions", "submit",
                    "-c", KF.COMPETITION, "-k", f"{KF.OWNER}/{it.slug}", "-v", str(it.kernel_version),
                    "-f", "submission.csv", "-m", message])
        out = (r.stdout or "") + (r.stderr or "")
        print(out.strip())
        if r.returncode == 0:
            it.status = "submitted"
            it.note = "submitted"
        else:
            it.note = f"submit failed: {out.strip()[:300]}"
    return complete


def print_status(items: list[Item]) -> None:
    for it in items:
        extra = f" [{it.kaggle_status}]" if it.kaggle_status else ""
        deps = f" deps={it.deps}" if it.deps else ""
        print(f"  {it.status:<10} {it.name:<34} {KF.OWNER}/{it.slug} v{it.kernel_version}{extra}{deps} {it.note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--specs", nargs="+", required=True)
    r.add_argument("--submit", default="", help="comma-separated spec NAMES to submit after audit PASS")
    r.add_argument("--state", default="C:/temp/queue/state.json")
    r.add_argument("--dest-root", default="C:/temp/queue")
    r.add_argument("--poll", type=int, default=300)
    r.add_argument("--max-running", type=int, default=MAX_RUNNING_DEFAULT)
    r.add_argument("--max-hours", type=float, default=20.0)
    r.add_argument("--version", action="append", default=[], help="name=N kernel version override (default 1)")
    s = sub.add_parser("status")
    s.add_argument("--state", default="C:/temp/queue/state.json")
    args = ap.parse_args()

    state_path = Path(args.state)
    if args.cmd == "status":
        print_status(load_state(state_path))
        return 0

    versions = {}
    for row in args.version:
        k, _, v = row.partition("=")
        versions[k] = int(v)
    spec_paths = [Path(p) if Path(p).is_absolute() else REPO / p for p in args.specs]
    items = build_queue(spec_paths, load_state(state_path), versions)
    submit_names = {n.strip() for n in args.submit.split(",") if n.strip()}
    complete_fn = make_complete_fn(submit_names, Path(args.dest_root))
    t0 = time.time()
    print(f"queue: {len(items)} items, max_running={args.max_running}, submit={sorted(submit_names) or 'none'}")
    print_status(items)
    while True:
        events = tick(items, status_fn=KF.kernel_status, push_fn=real_push, complete_fn=complete_fn,
                      max_running=args.max_running, now=time.time())
        save_state(state_path, items)
        for e in events:
            print(time.strftime("%H:%M:%S"), e, flush=True)
        if all_done(items):
            print("queue drained")
            print_status(items)
            return 0
        if (time.time() - t0) / 3600 > args.max_hours:
            print("max-hours reached; leaving running kernels to finish on Kaggle")
            print_status(items)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
