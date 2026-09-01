"""Measure the notebook line-ending contract in DISPOSABLE clones. Never touches the repo.

THE QUESTION. FACT-0446 pins the champion notebook at a sha256 that is the CRLF WORKING-TREE
digest; the git blob is different. `core.autocrlf` comes from the SYSTEM config here (true), so a
clone on this machine reproduces the pin and a Linux/macOS/CI checkout does not. The repository
has no `*.ipynb` attribute, and the obvious fix (`eol=lf`, copying the existing
`data/d1_factorial/*.json` rule) would rewrite the working tree to LF and invalidate every
recorded notebook digest immediately.

THE ARMS. Three, each in its own throwaway clone:
  A  current attributes (no *.ipynb rule)
  B  *.ipynb -text                (git stops converting; worktree becomes the blob)
  C  *.ipynb text eol=crlf        (every platform materialises CRLF)

WHAT IS MEASURED, per arm and per probe notebook: the git blob sha, the Windows worktree sha, the
sha a non-Windows checkout would produce (simulated with core.autocrlf=false, which is the
default off-Windows), whether the recorded spec/manifest digests still resolve, and whether
`git add --renormalize` would rewrite tracked blobs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("c:/Users/aryaa/Documents/Biohub-CellTracking-2026")
WORK = Path("C:/temp/rejig/le_arms")

PROBES = [
    "notebooks/kaggle_p35_dcveto_on_931/biohub-p35-dcveto-on-931.ipynb",
    "notebooks/kaggle_p32_public931_exact/biohub-p32-public931-exact.ipynb",
    "notebooks/kaggle_p38_relink_bonus_b2/biohub-p38-relink-bonus-b2.ipynb",
]

ARMS = {
    "A_current": None,
    "B_binary": "*.ipynb -text\n",
    "C_eol_crlf": "*.ipynb text eol=crlf\n",
}


def sh(*args, cwd=None, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{args} -> {r.returncode}\n{r.stderr[-500:]}")
    return r.stdout


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def recorded_digests() -> dict:
    """What the repository currently believes each probe notebook hashes to."""
    out = {}
    for rel in PROBES:
        name = Path(rel).parent.name
        man = REPO / "notebooks" / name / "build_manifest.json"
        rec = {"build_manifest_built_sha256": None, "spec_base_sha256_referencing_it": []}
        if man.is_file():
            rec["build_manifest_built_sha256"] = json.loads(
                man.read_text(encoding="utf-8")).get("built_sha256")
        for sp in sorted((REPO / "scripts" / "kaggle_specs").glob("*.json")):
            try:
                s = json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            base = str(s.get("base_notebook") or s.get("base") or "").replace("\\", "/")
            if base == rel:
                rec["spec_base_sha256_referencing_it"].append(
                    {"spec": sp.name, "base_sha256": s.get("base_sha256")})
        out[rel] = rec
    return out


def run_arm(name: str, attributes: str | None) -> dict:
    root = WORK / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    sh("git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(root))

    ga = root / ".gitattributes"
    original_ga = ga.read_text(encoding="utf-8") if ga.is_file() else ""

    # THE ATTRIBUTES MUST BE COMMITTED. The first version of this experiment wrote .gitattributes
    # into the worktree only, so the clone-of-clone below checked out HEAD's attributes and every
    # arm produced identical numbers - three different rules, one answer, which should have been
    # the tell. A migration would commit the rule, so the experiment commits it too.
    arm = {"attributes_added": attributes, "probes": {}}
    if attributes:
        ga.write_text(original_ga + "\n# EXPERIMENT ARM\n" + attributes, encoding="utf-8")
        sh("git", "add", ".gitattributes", cwd=root)
        sh("git", "-c", "user.email=x@y", "-c", "user.name=exp",
           "commit", "--quiet", "-m", f"experiment arm {name}", cwd=root)
        # re-materialise the worktree under the committed attributes
        sh("git", "rm", "--cached", "-r", "--quiet", ".", cwd=root)
        sh("git", "reset", "--hard", "--quiet", cwd=root)

    # does renormalisation rewrite tracked blobs under these attributes?
    sh("git", "add", "--renormalize", ".", cwd=root, check=False)
    changed = [ln[3:] for ln in sh("git", "status", "--porcelain", cwd=root).splitlines()
               if ln[:2].strip()]
    arm["renormalize_changes_tracked_files"] = len(changed)
    arm["renormalize_changed_ipynb"] = sorted(c for c in changed if c.endswith(".ipynb"))[:6]
    arm["renormalize_changed_ipynb_count"] = sum(1 for c in changed if c.endswith(".ipynb"))
    sh("git", "reset", "--quiet", cwd=root, check=False)

    # a second clone checked out as a NON-WINDOWS client would (autocrlf=false)
    nonwin = WORK / (name + "_nonwindows")
    if nonwin.exists():
        shutil.rmtree(nonwin, ignore_errors=True)
    sh("git", "clone", "--quiet", "--no-hardlinks", "-c", "core.autocrlf=false",
       str(root), str(nonwin))

    for rel in PROBES:
        blob = sh("git", "show", f"HEAD:{rel}", cwd=root).encode("utf-8", "surrogateescape")
        blob_raw = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root,
                                  capture_output=True).stdout
        win = (root / rel).read_bytes()
        nw = (nonwin / rel).read_bytes() if (nonwin / rel).is_file() else b""
        arm["probes"][rel] = {
            "blob_sha256": sha_bytes(blob_raw),
            "blob_bytes": len(blob_raw),
            "windows_worktree_sha256": sha_bytes(win),
            "windows_worktree_bytes": len(win),
            "nonwindows_checkout_sha256": sha_bytes(nw),
            "nonwindows_checkout_bytes": len(nw),
            "windows_equals_nonwindows": sha_bytes(win) == sha_bytes(nw),
        }
    shutil.rmtree(nonwin, ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    return arm


def main() -> int:
    rec = recorded_digests()
    report = {"recorded_digests": rec, "arms": {}}
    for name, attrs in ARMS.items():
        print(f"--- arm {name} ---", flush=True)
        arm = run_arm(name, attrs)
        # does each recorded digest still resolve, on each platform?
        for rel, probe in arm["probes"].items():
            want = rec[rel]["build_manifest_built_sha256"]
            probe["recorded_digest"] = want
            probe["recorded_resolves_on_windows"] = (want == probe["windows_worktree_sha256"])
            probe["recorded_resolves_off_windows"] = (want == probe["nonwindows_checkout_sha256"])
        arm["all_recorded_resolve_on_windows"] = all(
            p["recorded_resolves_on_windows"] for p in arm["probes"].values())
        arm["all_recorded_resolve_off_windows"] = all(
            p["recorded_resolves_off_windows"] for p in arm["probes"].values())
        report["arms"][name] = arm
        print(f"    windows ok={arm['all_recorded_resolve_on_windows']}  "
              f"off-windows ok={arm['all_recorded_resolve_off_windows']}  "
              f"renormalize touches {arm['renormalize_changes_tracked_files']} files "
              f"({len(arm['renormalize_changed_ipynb'])} ipynb shown)", flush=True)
    out = Path("C:/temp/rejig/lineending_experiment.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
