r"""Notebook-only Kaggle submission factory.

WHY THIS EXISTS
---------------
This competition **only accepts submissions from notebooks**.  A locally produced
``submission.csv`` can never be submitted (``CreateSubmission`` returns
``FAILED_PRECONDITION``, which the CLI truncates to a bare ``400``).  Therefore every
candidate -- however well audited locally -- must first exist as a COMPLETED Kaggle
kernel.  This module is the reusable path from *a local graph-edit function* to
*a completed, audited kernel plus the exact submit command*.

It never submits.  ``submitcmd`` prints the command; a human runs it.

Each stage hard-fails on one of the documented environment traps
(``research/03-experimentation/quality-control.md``):

  build   every edit asserts an exact match count, so a silently-skipped patch is an
          error rather than a warning (trap: "do not preflight, modify, then scale");
          the build manifest records base/built sha256 + a config hash.
  push    ``PYTHONUTF8=1`` is forced (trap 8, locale codec) and the **slug is read back
          from the push response** (trap 12, Kaggle slugifies the TITLE not the id).
  status  typed ``get_kernel_session_status(...).status.name`` only -- never substring
          matching on CLI text (trap 10, a transient SSLError reads as "ERROR").
  fetch   named files pulled by URL via ``list_kernel_session_output`` (trap 11,
          ``kaggle kernels output`` pulls the whole 168-file working dir and times out).
  audit   delegates to ``scripts/d1/audit_submission_structure.py`` and propagates its
          exit code.  An artifact that fails the audit gets no submit command.

SPEC FORMAT (JSON, see scripts/kaggle_specs/*.json)
---------------------------------------------------
{
  "name":            "p0d_example",
  "slug":            "biohub-p0d-example",          # what YOU want; verified after push
  "title":           "Biohub P0D Example",          # Kaggle slugifies THIS
  "code_file":       "biohub-p0d-example.ipynb",
  "out_dir":         "notebooks/kaggle_p0d_example",
  "base_notebook":   "notebooks/kaggle_p0a_clean913/biohub-p0a-clean913-repro.ipynb",
  "base_sha256":     "<optional, asserted if present>",
  "datasets":        ["owner/slug", ...],
  "competition_sources": ["biohub-cell-tracking-during-development"],
  "enable_gpu": true, "enable_internet": false, "machine_shape": "NvidiaTeslaT4",
  "docker_image": "<pinned digest>",
  "expects_submission": true,
  "edits": [
    {"kind": "env",           "vars": {"BIOHUB_X": "1"}, "cell_match": "BIOHUB_PRESET"},
    {"kind": "replace",       "old": "...", "new": "...", "expect": 1},
    {"kind": "insert_before", "anchor": "...", "code_file": "scripts/kaggle_edits/x.py"},
    {"kind": "insert_after",  "anchor": "...", "code": "print('hi')\n"},
    {"kind": "append_cell",   "code_file": "scripts/kaggle_edits/y.py"}
  ]
}

``code_file`` paths are repo-relative.  Keeping the injected code in a real ``.py`` file
is the point: it can be imported, linted and unit-tested locally before it ever touches
a kernel.

USAGE
-----
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py build     --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py push      --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py status    --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py log       --spec <spec.json> [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py fetch     --spec <spec.json> [--files submission.csv ...] [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py audit     --spec <spec.json> [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py submitcmd --spec <spec.json> -m "message"
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py verify    --spec <spec.json>   # cell-level diff vs base

Always run with ``PYTHONUTF8=1`` set (this module sets it for its own child processes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
KAGGLE_EXE = REPO / ".venv" / "Scripts" / "kaggle.exe"
DEFAULT_DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
COMPETITION = "biohub-cell-tracking-during-development"
OWNER = os.environ.get("BIOHUB_KAGGLE_OWNER", "aryaarun07")
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


# --------------------------------------------------------------------------- utils
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "slug", "title", "code_file", "out_dir", "base_notebook"):
        if key not in spec:
            raise SystemExit(f"spec {path} is missing required key {key!r}")
    spec["_spec_path"] = str(path)
    return spec


def out_dir(spec: dict) -> Path:
    return REPO / spec["out_dir"]


def built_nb(spec: dict) -> Path:
    return out_dir(spec) / spec["code_file"]


def manifest_path(spec: dict) -> Path:
    return out_dir(spec) / "build_manifest.json"


def cells_text(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"]]


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_nb(nb: dict, path: Path) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def env() -> dict:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(c) for c in cmd], cwd=REPO, env=env(),
        capture_output=True, text=True, encoding="utf-8", errors="replace", **kw,
    )


# --------------------------------------------------------------------------- build
def _edit_code(edit: dict) -> str:
    if "code" in edit:
        return edit["code"]
    if "code_file" in edit:
        return (REPO / edit["code_file"]).read_text(encoding="utf-8")
    raise SystemExit(f"edit {edit.get('kind')!r} needs 'code' or 'code_file'")


def _matching_cells(nb: dict, edit: dict) -> list[int]:
    """Indices of code cells eligible for this edit (optionally narrowed by cell_match)."""
    needle = edit.get("cell_match")
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if needle is not None and needle not in "".join(cell["source"]):
            continue
        out.append(i)
    return out


def _set_cell(nb: dict, i: int, text: str) -> None:
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


def apply_edit(nb: dict, edit: dict, tag: str) -> dict:
    """Apply one edit, asserting the exact expected match count.  Returns a record."""
    kind = edit["kind"]
    expect = int(edit.get("expect", 1))
    touched: list[int] = []

    if kind == "env":
        body = "".join(
            f'os.environ["{k}"] = {v!r}\n' for k, v in edit["vars"].items()
        )
        idx = _matching_cells(nb, edit)
        if len(idx) != expect:
            raise SystemExit(f"{tag}: env cell_match hit {len(idx)} cells, expected {expect}")
        for i in idx:
            text = "".join(nb["cells"][i]["source"])
            _set_cell(nb, i, text.rstrip("\n") + "\n\n# --- factory env overrides ---\n" + body)
            touched.append(i)

    elif kind in ("insert_before", "insert_after"):
        anchor = edit["anchor"]
        code = _edit_code(edit)
        hits = 0
        for i in _matching_cells(nb, edit):
            text = "".join(nb["cells"][i]["source"])
            n = text.count(anchor)
            if n == 0:
                continue
            hits += n
            if kind == "insert_before":
                text = text.replace(anchor, "\n" + code + "\n" + anchor, 1)
            else:
                text = text.replace(anchor, anchor + "\n" + code + "\n", 1)
            _set_cell(nb, i, text)
            touched.append(i)
        if hits != expect:
            raise SystemExit(f"{tag}: anchor matched {hits} times, expected {expect}")

    elif kind == "replace":
        old, new = edit["old"], edit["new"]
        hits = 0
        for i in _matching_cells(nb, edit):
            text = "".join(nb["cells"][i]["source"])
            n = text.count(old)
            if n == 0:
                continue
            hits += n
            _set_cell(nb, i, text.replace(old, new, n))
            touched.append(i)
        if hits != expect:
            raise SystemExit(f"{tag}: replace matched {hits} times, expected {expect}")

    elif kind == "append_cell":
        code = _edit_code(edit)
        nb["cells"].append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": code.splitlines(keepends=True),
        })
        touched.append(len(nb["cells"]) - 1)

    elif kind == "replace_cell":
        # Whole-cell replacement, selected by cell_match. Used when a base cell asserts
        # a contract the derived kernel deliberately no longer satisfies.
        idx = _matching_cells(nb, edit)
        if len(idx) != expect:
            raise SystemExit(f"{tag}: cell_match hit {len(idx)} cells, expected {expect}")
        code = _edit_code(edit)
        for i in idx:
            _set_cell(nb, i, code)
            touched.append(i)

    else:
        raise SystemExit(f"{tag}: unknown edit kind {kind!r}")

    # Every touched cell must still be valid Python.
    for i in touched:
        text = "".join(nb["cells"][i]["source"])
        try:
            compile(text, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"{tag}: cell {i} no longer compiles: {exc}") from exc

    return {"kind": kind, "cells": touched,
            "payload_sha256": sha256_bytes(json.dumps(edit, sort_keys=True).encode())}


def metadata_for(spec: dict) -> dict:
    return {
        "id": f"{OWNER}/{spec['slug']}",
        "title": spec["title"],
        "code_file": spec["code_file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(spec.get("is_private", True)),
        "enable_gpu": bool(spec.get("enable_gpu", True)),
        "enable_tpu": False,
        "enable_internet": bool(spec.get("enable_internet", False)),
        "keywords": spec.get("keywords", ["gpu"] if spec.get("enable_gpu", True) else []),
        "dataset_sources": spec.get("datasets", []),
        "kernel_sources": spec.get("kernel_sources", []),
        "competition_sources": spec.get("competition_sources", [COMPETITION]),
        "model_sources": [],
        "docker_image": spec.get("docker_image", DEFAULT_DOCKER),
        "machine_shape": spec.get("machine_shape", "NvidiaTeslaT4"),
    }


def cmd_build(spec: dict) -> int:
    base = REPO / spec["base_notebook"]
    base_sha = sha256_file(base)
    if spec.get("base_sha256") and spec["base_sha256"] != base_sha:
        raise SystemExit(
            f"base notebook drifted: expected {spec['base_sha256']}, got {base_sha}"
        )
    nb = load_nb(base)
    records = []
    for n, edit in enumerate(spec.get("edits", [])):
        records.append(apply_edit(nb, edit, f"{spec['name']}/edit{n}({edit['kind']})"))

    dest = out_dir(spec)
    dest.mkdir(parents=True, exist_ok=True)
    dump_nb(nb, built_nb(spec))
    (dest / "kernel-metadata.json").write_text(
        json.dumps(metadata_for(spec), indent=2) + "\n", encoding="utf-8"
    )

    built_sha = sha256_file(built_nb(spec))
    cfg = sha256_bytes(
        json.dumps({"base": base_sha, "edits": records,
                    "meta": metadata_for(spec)}, sort_keys=True).encode()
    )[:12]
    man = {
        "name": spec["name"], "spec": spec["_spec_path"],
        "base_notebook": spec["base_notebook"], "base_sha256": base_sha,
        "built_notebook": str(built_nb(spec).relative_to(REPO)),
        "built_sha256": built_sha, "config_hash": cfg,
        "n_cells": len(nb["cells"]), "edits": records,
        "declared_slug": spec["slug"], "title": spec["title"],
    }
    manifest_path(spec).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))
    return 0


def cmd_verify(spec: dict) -> int:
    """Cell-level diff of built vs base -- proves the blast radius of the edits."""
    a = cells_text(load_nb(REPO / spec["base_notebook"]))
    b = cells_text(load_nb(built_nb(spec)))
    print(f"cells: base={len(a)} built={len(b)}")
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x == y:
            continue
        if x is None:
            print(f"  cell {i}: ADDED ({len(y)} chars)")
        elif y is None:
            print(f"  cell {i}: REMOVED")
        else:
            print(f"  cell {i}: CHANGED {len(x)} -> {len(y)} chars (+{len(y) - len(x)})")
    return 0


# ---------------------------------------------------------------------------- push
def cmd_push(spec: dict) -> int:
    d = out_dir(spec)
    if not built_nb(spec).exists():
        raise SystemExit(f"not built: {built_nb(spec)} (run `build` first)")
    r = run([KAGGLE_EXE, "kernels", "push", "-p", str(d)])
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    if "successfully pushed" not in out.lower():
        return 1
    # Trap 12: Kaggle slugifies the TITLE, not the id. Read the real slug back.
    real = None
    for token in out.replace("\n", " ").split():
        if "/code/" in token:
            real = token.rstrip("/").split("/")[-1]
    if real is None:
        print("WARN could not parse the pushed URL; verify the slug manually")
    elif real != spec["slug"]:
        print(f"SLUG DIVERGENCE: declared {spec['slug']!r} but Kaggle created {real!r}. "
              f"Update the spec's 'slug' to {real!r} before status/fetch/submit.")
    else:
        print(f"slug confirmed: {OWNER}/{real}")
    man = json.loads(manifest_path(spec).read_text()) if manifest_path(spec).exists() else {}
    man["pushed_slug"] = real or spec["slug"]
    manifest_path(spec).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return 0


# -------------------------------------------------------------------------- status
def _client():
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    return api


def kernel_status(slug: str) -> str:
    """Typed status only.  Transport flakes report UNKNOWN so they are retried."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    api = _client()
    with api.build_kaggle_client() as k:
        req = ApiGetKernelSessionStatusRequest()
        req.user_name, req.kernel_slug = OWNER, slug
        try:
            resp = k.kernels.kernels_api_client.get_kernel_session_status(req)
            return resp.status.name + (f" ({resp.failure_message})" if resp.failure_message else "")
        except Exception as exc:  # transport flake -- never report as a kernel failure
            return f"UNKNOWN ({type(exc).__name__}: {str(exc)[:120]})"


def cmd_status(spec: dict) -> int:
    st = kernel_status(spec["slug"])
    print(f"{OWNER}/{spec['slug']}: {st}")
    return 0 if st.split(" ")[0] == "COMPLETE" else 1


# --------------------------------------------------------------------------- fetch
def session_outputs(slug: str) -> dict[str, str]:
    """List EVERY output file of a kernel session, following pagination.

    TRAP 22 (2026-08-01): this used to make ONE call and return `resp.files`. The endpoint
    caps a page at 500 entries, so for any kernel that also writes a zarr/geff tree the page
    filled with chunk files and the NAMED outputs were absent -- indistinguishable from "the
    kernel produced nothing". One workstream lost time to exactly that; following
    `next_page_token` turned 500 files into 1,943.

    Always paginate. A truncated listing is worse than an error because it looks like data.
    """
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = _client()
    out: dict[str, str] = {}
    token = None
    with api.build_kaggle_client() as k:
        for _ in range(200):                      # hard stop; 200 * 500 = 100k files
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = OWNER, slug
            req.page_size = 500
            if token:
                req.page_token = token
            resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
            out.update({f.file_name: f.url for f in resp.files})
            token = getattr(resp, "next_page_token", None) or None
            if not token:
                break
        else:
            raise RuntimeError(f"{slug}: output listing did not terminate after 200 pages")
    return out


def cmd_fetch(spec: dict, files: list[str], dest: Path) -> int:
    """Pull NAMED outputs by URL.  `kaggle kernels output` pulls all 168 files and times out."""
    urls = session_outputs(spec["slug"])
    dest.mkdir(parents=True, exist_ok=True)
    missing = [f for f in files if f not in urls]
    if missing:
        print(f"available ({len(urls)}): {sorted(urls)[:25]}")
        raise SystemExit(f"kernel produced no such output(s): {missing}")
    for name in files:
        target = dest / Path(name).name
        with urllib.request.urlopen(urls[name], timeout=600) as fh:
            target.write_bytes(fh.read())
        print(f"{name} -> {target}  ({target.stat().st_size:,} bytes, sha256 {sha256_file(target)})")
    return 0


# --------------------------------------------------------------------------- audit
def cmd_log(spec: dict, dest: Path) -> int:
    """Stream the kernel's execution log -- the only way to diagnose an ERROR precisely.

    A failed kernel with an EMPTY /kaggle/input listing is trap 9 (the kernel was created
    during an SSL-error window and is permanently broken): re-create it under a fresh
    slug rather than debugging the notebook.
    """
    api = _client()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "kernel.log"
    with api.build_kaggle_client() as k:
        stream = k.kernels.kernels_api_client.get_kernel_session_logs_stream(
            OWNER, spec["slug"]
        )
        body = stream if isinstance(stream, (str, bytes)) else b"".join(stream)
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    target.write_text(body, encoding="utf-8")
    print(f"log -> {target} ({len(body):,} chars)")
    tail = body.strip().splitlines()[-40:]
    print("\n".join(tail))
    if "/kaggle/input" in body and "-> []" in body:
        print("\nTRAP 9 SIGNATURE: /kaggle/input listed empty. Do not debug the notebook; "
              "re-create the kernel under a FRESH SLUG.")
    return 0


def cmd_audit(spec: dict, dest: Path) -> int:
    csv = dest / "submission.csv"
    if not csv.exists():
        raise SystemExit(f"no {csv}; run `fetch` first")
    r = run([sys.executable, "scripts/d1/audit_submission_structure.py", "audit", str(csv)])
    print((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        print("AUDIT FAILED -- no submit command will be issued for this artifact.")
    return r.returncode


def cmd_submitcmd(spec: dict, message: str, dest: Path) -> int:
    """Print the ONLY working submit form.  Never runs it."""
    csv = dest / "submission.csv"
    sha = sha256_file(csv) if csv.exists() else "<fetch first>"
    print("\n# Preconditions: kernel COMPLETE, audit exit 0, human owns the slot.")
    print(f"# artifact sha256 {sha}")
    print("$env:PYTHONUTF8=1; .\\.venv\\Scripts\\kaggle.exe competitions submit "
          f"-c {COMPETITION} -k {OWNER}/{spec['slug']} -v <VERSION> "
          f"-f submission.csv -m \"{message}\"")
    print("# -v is the kernel VERSION number, visible on the kernel's Output tab.")
    print("# A local CSV cannot be submitted: this competition accepts notebooks only.\n")
    return 0


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=[
        "build", "verify", "push", "status", "log", "fetch", "audit", "submitcmd"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--files", nargs="*", default=["submission.csv"])
    ap.add_argument("--dest")
    ap.add_argument("-m", "--message", default="")
    args = ap.parse_args()

    spec = load_spec(Path(args.spec) if Path(args.spec).is_absolute() else REPO / args.spec)
    dest = Path(args.dest) if args.dest else out_dir(spec) / "_out"

    if args.stage == "build":
        return cmd_build(spec)
    if args.stage == "verify":
        return cmd_verify(spec)
    if args.stage == "push":
        return cmd_push(spec)
    if args.stage == "status":
        return cmd_status(spec)
    if args.stage == "log":
        return cmd_log(spec, dest)
    if args.stage == "fetch":
        return cmd_fetch(spec, args.files, dest)
    if args.stage == "audit":
        return cmd_audit(spec, dest)
    if args.stage == "submitcmd":
        return cmd_submitcmd(spec, args.message or spec.get("name", ""), dest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
