"""Materialise the PINNED external checkouts a fresh clone does not carry.

WHY THIS EXISTS
---------------
`vendor/` is the organizer's repository. It is gitignored on purpose (`.gitignore:6-7`, "vendored
organizer repo - reproducible via git clone"), so a fresh clone does not contain it. The intent
was sound and the mechanism was missing: nothing materialised it, and nothing told you that was
why things had failed. Measured on a clean clone of `e5db190`:

  * `validate_registry.py` exited 1 with four R3 errors - FACT-0295, FACT-0336, FACT-0349 and
    FACT-0353 cite instruments under `vendor/kaggle-cell-tracking/`;
  * `pytest --collect-only` exited 2, INTERRUPTED, because `tests/test_h1r_edge_train.py` and
    `tests/test_h1r_rope4d.py` import `train_unet_transformer` from that checkout.

The irony is the point. R3 exists to enforce "a MEASURED fact must be reproducible from committed
code", and R3 passed on the development machine ONLY because of files that are not committed. It
was green here and red everywhere else - the same shape as FACT-0043, the anchor that proved
unreproducible because its script had never been committed.

THE REPAIR IS A PIN PLUS A BOOTSTRAP, NOT A COMMIT
--------------------------------------------------
Committing `vendor/` would fight `.gitignore` and vendor a second copy of somebody else's
repository into ours. The dependency is already PINNED - `config/requirements.txt` and
`config/requirements.lock.txt` both name commit 075fc5f5... and the three champion-lineage
notebooks embed the same sha - so it is genuinely reproducible. What was missing is a step that
performs the reproduction and a diagnostic that names it when it has not been performed.

`--check` never touches the network, so it is safe to call from a validator or a test.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
HEARTBEAT_OK = "BOOTSTRAP_VENDOR_OK"
HEARTBEAT_MISSING = "BOOTSTRAP_VENDOR_MISSING"

#: Where the pin is DECLARED. Read, never restated - a second copy of a commit sha is a second
#: chance to go stale, which is the whole reason this repository cites ids instead of values.
PIN_SOURCES = ("config/requirements.txt", "config/requirements.lock.txt")

CHECKOUTS = {
    "vendor/kaggle-cell-tracking": {
        "url": "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
        "pin_package": "tracking-cellmot",
        "why": "the organizer's own repository: the official scorer, the vendored converter and "
               "the reference predictor. Four facts cite instruments inside it and two test "
               "modules import from it.",
        "sentinels": ("src/tracking_cellmot/division_metrics.py",
                      "scripts/csv_to_geffs.py",
                      "scripts/predict_unet_transformer.py",
                      "scripts/train_unet_transformer.py"),
    },
}

_PIN = re.compile(r"@([0-9a-f]{7,40})\s*$")


class BootstrapRefusal(RuntimeError):
    """Raised when the pin cannot be established. Never guessed, never defaulted."""


def declared_pin(package: str) -> tuple[str, str]:
    """Return ``(commit, source)`` for a package, read from the declaring files.

    Both sources must agree. A disagreement is a refusal rather than a preference, because
    picking one silently is how a build starts reproducing a commit nobody chose.
    """
    found: dict[str, str] = {}
    for rel in PIN_SOURCES:
        p = REPO / rel
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(package):
                m = _PIN.search(line.strip())
                if m:
                    found[rel] = m.group(1)
    if not found:
        raise BootstrapRefusal(
            f"no pin for {package!r} in any of {list(PIN_SOURCES)}. An external checkout with no "
            f"pin is not reproducible, so it FAILS CLOSED rather than cloning a moving branch."
        )
    commits = set(found.values())
    if len(commits) > 1:
        raise BootstrapRefusal(
            f"the pin for {package!r} DISAGREES across its sources: {found}. Refusing rather than "
            f"choosing one - a build must not reproduce a commit nobody selected."
        )
    return commits.pop(), ", ".join(sorted(found))


def status(name: str) -> dict:
    """Is this checkout present, at the right commit, and complete? No network."""
    spec = CHECKOUTS[name]
    path = REPO / name
    commit, source = declared_pin(spec["pin_package"])
    rec: dict = {"checkout": name, "path": str(path), "url": spec["url"],
                 "declared_commit": commit, "pin_source": source, "present": path.is_dir()}
    if not path.is_dir():
        rec["state"] = "ABSENT"
        rec["missing_sentinels"] = list(spec["sentinels"])
        return rec
    missing = [s for s in spec["sentinels"] if not (path / s).is_file()]
    rec["missing_sentinels"] = missing
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
        rec["head"] = head.stdout.strip() or None
    except (OSError, subprocess.SubprocessError) as exc:
        rec["head"] = None
        rec["head_error"] = f"{type(exc).__name__}: {exc}"
    if missing:
        rec["state"] = "INCOMPLETE"
    elif rec.get("head") and not rec["head"].startswith(commit):
        rec["state"] = "WRONG_COMMIT"
    else:
        rec["state"] = "OK"
    return rec


def is_ok(name: str) -> bool:
    try:
        return status(name)["state"] == "OK"
    except BootstrapRefusal:
        return False


def diagnostic(name: str) -> str:
    """The sentence a validator or a skipped test should print instead of a bare ImportError."""
    try:
        rec = status(name)
    except BootstrapRefusal as exc:
        return f"{name}: {exc}"
    return (
        f"{name} is {rec['state']} at {rec['path']}. This is a PINNED EXTERNAL CHECKOUT, not "
        f"missing committed code: it is gitignored by design and reproduced by\n"
        f"    .venv/Scripts/python.exe scripts/core/bootstrap_vendor.py --apply\n"
        f"which clones {rec['url']} at commit {rec['declared_commit']} "
        f"(pinned in {rec['pin_source']})."
    )


def apply(name: str) -> dict:
    """Clone or fetch the checkout to its pinned commit. The only path that touches the network."""
    spec = CHECKOUTS[name]
    path = REPO / name
    commit, _ = declared_pin(spec["pin_package"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", spec["url"], str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--force", commit], check=True)
    rec = status(name)
    if rec["state"] != "OK":
        raise BootstrapRefusal(
            f"bootstrap did not reach a usable state: {rec}. Refusing to report success on a "
            f"checkout whose sentinels are absent - a half-materialised vendor tree is exactly "
            f"how a guard goes quietly missing."
        )
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="clone/checkout to the pinned commit (touches the network)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    report = {}
    ok = True
    for name in CHECKOUTS:
        try:
            rec = apply(name) if args.apply else status(name)
        except (BootstrapRefusal, subprocess.CalledProcessError) as exc:
            rec = {"checkout": name, "state": "REFUSED", "error": str(exc)}
        report[name] = rec
        ok = ok and rec.get("state") == "OK"
        print(f"  {name:<32} {rec.get('state')}"
              + (f"  head={str(rec.get('head'))[:12]}" if rec.get("head") else ""))
        if rec.get("state") != "OK":
            print("    " + diagnostic(name).replace("\n", "\n    "))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(HEARTBEAT_OK if ok else HEARTBEAT_MISSING)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
