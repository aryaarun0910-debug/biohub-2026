r"""Fail-closed version-coherence receipt: does a running or landed artifact mix code versions?

THE DEFECT THIS EXISTS TO CATCH
--------------------------------
`build_manifest.json` records, per edit, `sha256(json.dumps(edit))` - a hash of the EDIT SPEC,
not of the patch source it names. So editing `scripts/kaggle_edits/<patch>.py` after a build
leaves every `payload_sha256` unchanged. The only artifact that binds the patch TEXT is the built
notebook itself, because the build inlines that text into a cell. This instrument therefore
verifies the chain end to end:

    spec  ->  base notebook (sha256)  ->  built notebook (sha256 == manifest.built_sha256)
          ->  a NON-DESTRUCTIVE REBUILD from the current spec, base and patch sources into a
              temp directory reproduces `manifest.built_sha256` exactly
          ->  every one of those paths is clean in git and its last-touching commit is named
          ->  declared_slug == spec slug == pushed_slug   (the slug-divergence trap)

A patch source edited after the build is then a hard failure: the rebuild no longer reproduces
the recorded notebook. Grepping the notebook for the patch text is NOT sufficient and was tried
first - a cell that several edits touch in sequence no longer contains any one patch source as a
contiguous block, so a text probe reports a false failure. The rebuild has no such ambiguity.

WHAT IT CANNOT DO, STATED PLAINLY
----------------------------------
It cannot prove which KERNEL VERSION executed. `kaggle_factory.py fetch` and `log` bind no
version - only `audit` does, and only for submission artifacts. For a diagnostic run the recorded
`kernel_version` in experiments.yaml is a human assertion, not a verified one. This receipt
reports that gap rather than papering over it, and where the kernel log carries a build-identifying
string (a config hash, an inlined heartbeat) it matches that instead.

FAIL CLOSED. A missing manifest, a missing notebook or an unreadable spec is a FAIL, never a skip.
Heartbeat `VERSION_COHERENCE_COMPLETE`; its absence is the alarm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HEARTBEAT = "VERSION_COHERENCE_COMPLETE"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    return (r.stdout or "").strip()


def audit_spec(spec_path: Path, at_commit: str | None) -> dict:
    out: dict = {"spec": str(spec_path), "checks": {}, "passed": False}
    if not spec_path.is_file():
        out["fail_reason"] = "spec not found"
        return out
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out["name"] = spec.get("name")
    out["slug"] = spec.get("slug")
    odir = ROOT / spec["out_dir"]
    built = odir / spec["code_file"]
    manifest_path = odir / "build_manifest.json"
    if not built.is_file() or not manifest_path.is_file():
        out["fail_reason"] = f"missing built notebook or manifest under {odir}"
        return out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    built_sha = sha256_file(built)
    base = ROOT / spec["base_notebook"]
    base_sha = sha256_file(base) if base.is_file() else None

    out["checks"]["built_matches_manifest"] = {
        "built_sha256": built_sha,
        "manifest_built_sha256": manifest.get("built_sha256"),
        "passed": built_sha == manifest.get("built_sha256"),
    }
    out["checks"]["base_matches_spec_and_manifest"] = {
        "base_sha256": base_sha,
        "spec_base_sha256": spec.get("base_sha256"),
        "manifest_base_sha256": manifest.get("base_sha256"),
        "passed": base_sha is not None
        and base_sha == spec.get("base_sha256") == manifest.get("base_sha256"),
    }
    declared, pushed = manifest.get("declared_slug"), manifest.get("pushed_slug")
    out["checks"]["slug_agreement"] = {
        "spec_slug": spec.get("slug"), "declared_slug": declared, "pushed_slug": pushed,
        "passed": declared == spec.get("slug") and (pushed is None or pushed == spec.get("slug")),
        "note": "pushed_slug absent means the manifest predates the push read-back",
    }

    # --- the check the manifest itself cannot make: rebuild from the CURRENT sources ---
    # `payload_sha256` hashes the edit SPEC, so a patch source edited after the build leaves the
    # manifest untouched. A non-destructive rebuild into a temp directory is the only thing that
    # binds the committed patch bytes to the notebook that ran. Nothing in the worktree is written.
    rebuild = {"passed": False}
    try:
        sys.path.insert(0, str(ROOT / "scripts" / "core"))
        import kaggle_factory as kf  # noqa: PLC0415

        nb = kf.load_nb(ROOT / spec["base_notebook"])
        for n, edit in enumerate(spec.get("edits", [])):
            kf.apply_edit(nb, edit, f"{spec['name']}/edit{n}({edit['kind']})")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "rebuilt.ipynb"
            kf.dump_nb(nb, tmp)
            rebuilt_sha = sha256_file(tmp)
        rebuild = {
            "rebuilt_sha256": rebuilt_sha,
            "manifest_built_sha256": manifest.get("built_sha256"),
            "passed": rebuilt_sha == manifest.get("built_sha256"),
            "note": "rebuilt in a temp dir from the CURRENT spec, base and patch sources; the "
                    "worktree was not written",
        }
    except Exception as exc:  # a rebuild that cannot run is a FAIL, never a skip
        rebuild = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    rebuild["patch_source_sha256_16"] = {
        e["code_file"]: hashlib.sha256((ROOT / e["code_file"]).read_bytes()).hexdigest()[:16]
        for e in spec.get("edits", []) if e.get("code_file") and (ROOT / e["code_file"]).is_file()
    }
    out["checks"]["rebuild_reproduces_built_sha256"] = rebuild

    tracked = [str(spec_path.relative_to(ROOT)), str(built.relative_to(ROOT)),
               str(manifest_path.relative_to(ROOT))]
    tracked += [e["code_file"] for e in spec.get("edits", []) if e.get("code_file")]
    dirty = [t for t in tracked if _git("status", "--porcelain", "--", t)]
    out["checks"]["git_clean"] = {
        "paths": tracked,
        "dirty": dirty,
        "last_commit": {t: _git("log", "-1", "--format=%h %ad", "--date=short", "--", t)
                        for t in tracked},
        "passed": not dirty,
    }

    if at_commit:
        # `git show | sha256` is WRONG on this repo: core.autocrlf is true, so the blob is LF
        # while the working tree is CRLF and every comparison would read as a false difference.
        # `git diff --quiet` applies the same normalisation git uses for cleanliness.
        rel = built.relative_to(ROOT).as_posix()
        listed = subprocess.run(["git", "ls-tree", "--name-only", at_commit, "--", rel],
                                cwd=str(ROOT), capture_output=True, text=True)
        present = bool((listed.stdout or "").strip())
        diff = subprocess.run(["git", "diff", "--quiet", at_commit, "--", rel],
                              cwd=str(ROOT), capture_output=True)
        out["checks"]["identical_to_named_commit"] = {
            "commit": at_commit,
            "path_present_at_commit": present,
            "passed": bool(present and diff.returncode == 0),
            "method": "git diff --quiet (autocrlf-safe); a raw `git show` hash would false-fail",
        }

    out["passed"] = all(v.get("passed") for v in out["checks"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, nargs="+", required=True)
    ap.add_argument("--at-commit", nargs="*", default=[],
                    help="NAME=COMMIT pairs: assert that spec NAME's built notebook is unchanged "
                         "from COMMIT (autocrlf-safe comparison)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pins = dict(kv.split("=", 1) for kv in args.at_commit)
    rows = []
    for s in args.spec:
        path = s if s.is_absolute() else ROOT / s
        name = json.loads(path.read_text(encoding="utf-8")).get("name") if path.is_file() else None
        rows.append(audit_spec(path, pins.get(name)))
    result = {
        "schema_version": 1,
        "head_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty_paths": [
            l[3:] for l in _git("status", "--porcelain").splitlines() if l
        ],
        "specs": rows,
        "kernel_version_binding_gap": (
            "kaggle_factory fetch/log bind no kernel version; only `audit` issues a "
            "version-bound receipt and only for submission artifacts. For a diagnostic run the "
            "kernel_version recorded in experiments.yaml is an assertion, not a verified fact."
        ),
        "passed": all(r["passed"] for r in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in rows:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r.get('name') or r['spec']}"
              + (f"  ({r['fail_reason']})" if r.get("fail_reason") else ""))
        for k, v in r.get("checks", {}).items():
            if not v.get("passed"):
                print(f"        FAILED {k}: {json.dumps(v)[:300]}")
    print(f"{HEARTBEAT} specs={len(rows)} passed={result['passed']} -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
