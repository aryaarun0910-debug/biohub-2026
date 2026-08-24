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
  audit   delegates to ``scripts/d1/audit_submission_structure.py`` and writes a durable
          byte-bound release receipt. An artifact that fails gets no receipt.
  submitcmd refuses stale/missing receipts and requires the exact explicit kernel version
          recorded by ``audit``. It still never submits.

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
  "artifact_role": "training",                     # or diagnostic + diagnostic_reason
  "deploy_consumers": [{"artifact": "x.pth", "spec": "scripts/kaggle_specs/deploy.json"}],
  "consumes_artifacts": [{"producer_spec": "scripts/kaggle_specs/train.json",
                           "artifact": "x.pth"}],   # reciprocal, on deployment spec
  "resume_sources": [{"dataset": "owner/prior-run", "artifact": "last.pth"}],
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
a kernel.  ``scripts/core/experiment_defects.json`` scopes known-regression signatures
to affected spec families; both ``build`` and ``verify`` fail closed on a ledger violation.

USAGE
-----
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py build     --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py push      --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py status    --spec <spec.json>
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py log       --spec <spec.json> [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py fetch     --spec <spec.json> [--files submission.csv ...] [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py audit     --spec <spec.json> --kernel-version N [--dest DIR]
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py submitcmd --spec <spec.json> --kernel-version N -m "message"
  .\.venv\Scripts\python.exe scripts\core\kaggle_factory.py verify    --spec <spec.json>   # cell-level diff vs base

Always run with ``PYTHONUTF8=1`` set (this module sets it for its own child processes).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
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
DEFECT_LEDGER = REPO / "scripts" / "core" / "experiment_defects.json"
AUDIT_RECEIPT = "audit_receipt.json"
AUDIT_REPORT = "structural_audit.json"
RECEIPT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- utils
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def normalized_spec_sha256(spec: dict) -> str:
    """Hash only portable JSON spec content, excluding factory runtime metadata."""
    portable = {key: value for key, value in spec.items() if not key.startswith("_")}
    payload = json.dumps(
        portable, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except (OSError, ValueError):
        return path.resolve().as_posix()


def audit_script_path() -> Path:
    return REPO / "scripts" / "d1" / "audit_submission_structure.py"


def receipt_path(dest: Path) -> Path:
    return Path(dest) / AUDIT_RECEIPT


def _positive_kernel_version(version: int | None) -> int:
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise SystemExit(
            "an explicit positive --kernel-version is required; the factory does not infer "
            "a mutable latest version"
        )
    return version


def _release_materials(spec: dict, dest: Path, kernel_version: int) -> dict:
    """Resolve and validate every local byte string bound by a release receipt."""
    version = _positive_kernel_version(kernel_version)
    csv = Path(dest) / "submission.csv"
    if not csv.is_file():
        raise SystemExit(f"no {csv}; run `fetch` first")
    notebook = built_nb(spec)
    manifest = manifest_path(spec)
    auditor = audit_script_path()
    spec_path = Path(spec.get("_spec_path", ""))
    for label, path in (
        ("experiment spec", spec_path),
        ("built notebook", notebook),
        ("build manifest", manifest),
        ("defect ledger", DEFECT_LEDGER),
        ("structural auditor", auditor),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    try:
        disk_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unreadable experiment spec {spec_path}: {exc}") from exc
    spec_sha = normalized_spec_sha256(spec)
    if normalized_spec_sha256(disk_spec) != spec_sha:
        raise SystemExit("loaded spec no longer matches its current on-disk JSON")
    try:
        build = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unreadable build manifest {manifest}: {exc}") from exc
    notebook_sha = sha256_file(notebook)
    if build.get("built_sha256") != notebook_sha:
        raise SystemExit(
            "built notebook no longer matches build_manifest.json: "
            f"manifest={build.get('built_sha256')!r}, current={notebook_sha}"
        )
    if build.get("declared_slug") != spec.get("slug"):
        raise SystemExit(
            "build manifest slug does not match current spec: "
            f"manifest={build.get('declared_slug')!r}, spec={spec.get('slug')!r}"
        )
    return {
        "spec": {
            "path": _repo_relative(spec_path),
            "sha256": spec_sha,
        },
        "build": {
            "notebook_path": _repo_relative(notebook),
            "notebook_sha256": notebook_sha,
            "manifest_path": _repo_relative(manifest),
            "manifest_sha256": sha256_file(manifest),
        },
        "defect_ledger": {
            "path": _repo_relative(DEFECT_LEDGER),
            "sha256": sha256_file(DEFECT_LEDGER),
        },
        "submission": {
            "path": csv.name,
            "sha256": sha256_file(csv),
            "bytes": csv.stat().st_size,
        },
        "auditor": {
            "path": _repo_relative(auditor),
            "sha256": sha256_file(auditor),
        },
        "kernel": {
            "owner": OWNER,
            "slug": spec["slug"],
            "version": version,
        },
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


# --------------------------------------------------------------- defect build gate
def load_defect_ledger(path: Path = DEFECT_LEDGER) -> dict:
    """Load the small, machine-enforced ledger of previously observed defects."""
    ledger = json.loads(Path(path).read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1 or not isinstance(ledger.get("defects"), list):
        raise SystemExit(f"invalid defect ledger schema: {path}")
    ids = [d.get("id") for d in ledger["defects"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"defect ledger has missing/duplicate ids: {path}")
    return ledger


def _spec_env(spec: dict) -> dict[str, str]:
    """Flatten factory env edits; later edits intentionally take precedence."""
    out: dict[str, str] = {}
    for edit in spec.get("edits", []):
        if edit.get("kind") == "env":
            out.update({str(k): str(v) for k, v in edit.get("vars", {}).items()})
    return out


def _defect_applies(defect: dict, spec: dict) -> bool:
    scope = defect.get("scope", {})
    globs = scope.get("spec_name_globs", ["*"])
    return any(fnmatch.fnmatchcase(str(spec.get("name", "")), pattern) for pattern in globs)


def _reciprocal_consumer(spec: dict, artifact: str, consumer: dict) -> str | None:
    """Return an error unless a real consumer spec names this exact producer artifact."""
    rel = consumer.get("spec")
    if not isinstance(rel, str) or not rel:
        return f"artifact {artifact!r} has a consumer entry without a spec path"
    path = REPO / rel
    if not path.is_file():
        return f"artifact {artifact!r} consumer spec does not exist: {rel}"
    try:
        other = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"artifact {artifact!r} consumer spec is unreadable: {rel} ({exc})"
    if other.get("artifact_role") != "deployment":
        return f"artifact {artifact!r} consumer {rel} is not marked artifact_role=deployment"
    want_producer = spec.get("_spec_path")
    if want_producer:
        try:
            want_producer = str(Path(want_producer).resolve().relative_to(REPO.resolve()))
        except (ValueError, OSError):
            want_producer = str(want_producer)
        want_producer = want_producer.replace("\\", "/")
    if str(rel).replace("\\", "/") == want_producer:
        return f"artifact {artifact!r} cannot name its producer as its deployment consumer"
    consumes = other.get("consumes_artifacts", [])
    producer_role = spec.get("artifact_role")
    required_mode = "detection_only_bn_buffers" if producer_role == "calibration" else None
    expected_sha256 = consumer.get("sha256")
    if required_mode:
        if consumer.get("mode") != required_mode:
            return (f"calibration artifact {artifact!r} consumer {rel} must declare "
                    f"mode={required_mode!r}")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            return f"calibration artifact {artifact!r} requires a lowercase sha256 pin"
    matched = any(
        isinstance(row, dict)
        and str(row.get("producer_spec", "")).replace("\\", "/") == want_producer
        and row.get("artifact") == artifact
        and (required_mode is None or row.get("mode") == required_mode)
        and (required_mode is None or row.get("sha256") == expected_sha256)
        for row in consumes
    )
    if not matched:
        return (f"artifact {artifact!r} consumer {rel} lacks reciprocal "
                f"consumes_artifacts declaration for {want_producer}")
    return None


def validate_defect_gate(spec: dict, nb: dict, ledger: dict | None = None) \
        -> tuple[list[str], list[str]]:
    """Return (applicable defect ids, violations) for a fully assembled notebook.

    Regex checks are deliberately signatures of *known regressions*, not a general static
    analyser.  Structural facts (datasets, resume inputs, deployment consumers) are checked
    from the spec and the reciprocal consumer spec instead of trusting prose in code.
    """
    ledger = load_defect_ledger() if ledger is None else ledger
    source = "\n\n# --- defect-gate cell boundary ---\n\n".join(cells_text(nb))
    datasets = set(map(str, spec.get("datasets", [])))
    env_vars = _spec_env(spec)
    applicable: list[str] = []
    violations: list[str] = []

    for defect in ledger["defects"]:
        if not _defect_applies(defect, spec):
            continue
        defect_id = defect["id"]
        applicable.append(defect_id)
        for check in defect.get("checks", []):
            kind = check.get("kind")
            message: str | None = None
            if kind == "notebook_forbid_regex":
                if re.search(check["pattern"], source, flags=re.MULTILINE | re.DOTALL):
                    message = check.get("message", f"forbidden notebook signature {check['pattern']!r}")
            elif kind == "notebook_require_regex":
                if not re.search(check["pattern"], source, flags=re.MULTILINE | re.DOTALL):
                    message = check.get("message", f"required notebook signature {check['pattern']!r} missing")
            elif kind == "notebook_require_any_regex":
                patterns = check.get("patterns", [])
                if not any(re.search(p, source, flags=re.MULTILINE | re.DOTALL) for p in patterns):
                    message = check.get("message", "none of the required notebook signatures is present")
            elif kind == "spec_require_dataset":
                required = check["dataset"]
                if required not in datasets:
                    message = check.get("message", f"required dataset is not attached: {required}")
            elif kind == "spec_require_env":
                name, value = check["name"], str(check["value"])
                if env_vars.get(name) != value:
                    message = check.get(
                        "message", f"spec must explicitly set {name}={value!r}"
                    )
            elif kind == "resume_requires_source":
                names = [k for k, v in env_vars.items()
                         if re.fullmatch(check["env_pattern"], k) and v.lower() in {"1", "true", "yes"}]
                if names and not spec.get("resume_sources"):
                    message = check.get(
                        "message",
                        f"resume is enabled by {names}, but resume_sources is empty; /kaggle/working starts empty",
                    )
            elif kind == "artifacts_require_consumers":
                for artifact in check.get("artifacts", []):
                    if artifact not in source:
                        continue
                    role = spec.get("artifact_role")
                    if role == "diagnostic":
                        reason = spec.get("diagnostic_reason")
                        if not isinstance(reason, str) or len(reason.strip()) < 12:
                            message = (f"diagnostic artifact {artifact!r} requires a specific "
                                       "diagnostic_reason (at least 12 characters)")
                            break
                        if spec.get("deploy_consumers"):
                            message = (f"diagnostic artifact {artifact!r} cannot also declare "
                                       "deploy_consumers; choose a truthful artifact role")
                            break
                        continue
                    if role not in {"training", "calibration"}:
                        message = (f"produced artifact {artifact!r} must declare artifact_role as "
                                   "'training', 'calibration', or 'diagnostic'")
                        break
                    if role == "calibration":
                        reason = spec.get("calibration_reason")
                        if not isinstance(reason, str) or len(reason.strip()) < 12:
                            message = (f"calibration artifact {artifact!r} requires a specific "
                                       "calibration_reason (at least 12 characters)")
                            break
                    consumers = [row for row in spec.get("deploy_consumers", [])
                                 if isinstance(row, dict) and row.get("artifact") == artifact]
                    if not consumers:
                        message = (check.get("message") or
                                   f"produced artifact {artifact!r} has no declared deployment consumer")
                        break
                    errors = [_reciprocal_consumer(spec, artifact, row) for row in consumers]
                    errors = [e for e in errors if e]
                    if errors:
                        message = "; ".join(errors)
                        break
            else:
                raise SystemExit(f"defect {defect_id}: unknown check kind {kind!r}")
            if message:
                violations.append(f"{defect_id} [{defect.get('severity', 'UNKNOWN')}]: {message}")
    return applicable, violations


def enforce_defect_gate(spec: dict, nb: dict) -> list[str]:
    applicable, violations = validate_defect_gate(spec, nb)
    if violations:
        detail = "\n  - ".join(violations)
        raise SystemExit(f"defect gate failed for {spec.get('name', '<unnamed>')}:\n  - {detail}")
    return applicable


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

    defect_gate = enforce_defect_gate(spec, nb)

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
        "defect_gate": {"ledger": str(DEFECT_LEDGER.relative_to(REPO)),
                        "passed": defect_gate},
    }
    manifest_path(spec).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))
    return 0


def cmd_verify(spec: dict) -> int:
    """Cell-level diff of built vs base -- proves the blast radius of the edits."""
    a = cells_text(load_nb(REPO / spec["base_notebook"]))
    built = load_nb(built_nb(spec))
    passed = enforce_defect_gate(spec, built)
    print(f"defect gate: PASS ({len(passed)} applicable rules)")
    b = cells_text(built)
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
        # kaggle>=2.2.2 accepts a typed request and returns a requests.Response.
        # Keep the legacy fallbacks because older SDKs returned bytes/an iterator.
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionLogsStreamRequest,
        )

        request = ApiGetKernelSessionLogsStreamRequest()
        request.user_name = OWNER
        request.kernel_slug = spec["slug"]
        stream = k.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
        if hasattr(stream, "raise_for_status"):
            stream.raise_for_status()
            content_type = stream.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                events = stream.json()
                body = "".join(
                    event.get("data", "") if isinstance(event, dict) else str(event)
                    for event in events
                )
            else:
                body = stream.content
        else:
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


def cmd_audit(spec: dict, dest: Path, kernel_version: int | None = None) -> int:
    """Audit and atomically issue a receipt binding every local release input."""
    dest = Path(dest)
    receipt = receipt_path(dest)
    report = dest / AUDIT_REPORT
    report_tmp = report.with_name(report.name + ".tmp")
    # A new audit attempt invalidates any prior authorization before validation starts.
    receipt.unlink(missing_ok=True)
    report.unlink(missing_ok=True)
    report_tmp.unlink(missing_ok=True)

    materials = _release_materials(spec, dest, _positive_kernel_version(kernel_version))
    csv = dest / "submission.csv"
    r = run([
        sys.executable, "scripts/d1/audit_submission_structure.py", "audit", str(csv),
        "--json-out", str(report_tmp),
    ])
    print((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        report_tmp.unlink(missing_ok=True)
        print("AUDIT FAILED -- no submit command will be issued for this artifact.")
        return r.returncode

    try:
        reports = json.loads(report_tmp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report_tmp.unlink(missing_ok=True)
        raise SystemExit(f"auditor returned PASS without a readable JSON report: {exc}") from exc
    if not isinstance(reports, list) or len(reports) != 1:
        report_tmp.unlink(missing_ok=True)
        raise SystemExit("auditor JSON report must contain exactly one artifact")
    audited = reports[0]
    if audited.get("verdict") != "PASS" \
            or audited.get("sha256") != materials["submission"]["sha256"]:
        report_tmp.unlink(missing_ok=True)
        raise SystemExit(
            "auditor result does not bind the current submission bytes: "
            f"verdict={audited.get('verdict')!r}, sha256={audited.get('sha256')!r}"
        )
    current_materials = _release_materials(
        spec, dest, materials["kernel"]["version"],
    )
    if current_materials != materials:
        report_tmp.unlink(missing_ok=True)
        raise SystemExit("release inputs changed while the structural audit was running")
    report_tmp.replace(report)
    release_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "verdict": "PASS",
        **materials,
        "audit_report": {
            "path": report.name,
            "sha256": sha256_file(report),
        },
    }
    _write_json_atomic(receipt, release_receipt)
    print(f"release receipt -> {receipt} (sha256 {sha256_file(receipt)})")
    return 0


def cmd_submitcmd(
    spec: dict, message: str, dest: Path, kernel_version: int | None = None,
) -> int:
    """Print a receipt-bound submit command. Never runs it."""
    if spec.get("expects_submission") is not True:
        raise SystemExit(
            f"spec {spec.get('name', '<unnamed>')} is not submission-authorized "
            "(expects_submission must be true)"
        )
    version = _positive_kernel_version(kernel_version)
    dest = Path(dest)
    materials = _release_materials(spec, dest, version)
    receipt = receipt_path(dest)
    if not receipt.is_file():
        raise SystemExit(f"no {receipt}; run `audit --kernel-version {version}` first")
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unreadable release receipt {receipt}: {exc}") from exc
    if recorded.get("schema_version") != RECEIPT_SCHEMA_VERSION \
            or recorded.get("verdict") != "PASS":
        raise SystemExit("release receipt is not a supported PASS receipt")
    mismatches = [
        key for key, expected in materials.items() if recorded.get(key) != expected
    ]
    if mismatches:
        raise SystemExit(
            "release receipt no longer matches current release inputs: "
            + ", ".join(mismatches)
        )
    report_record = recorded.get("audit_report", {})
    if report_record.get("path") != AUDIT_REPORT:
        raise SystemExit("release receipt names an unexpected structural audit report")
    report = dest / AUDIT_REPORT
    if not report.is_file() or sha256_file(report) != report_record.get("sha256"):
        raise SystemExit("structural audit report is absent or changed since receipt issuance")
    try:
        reports = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unreadable structural audit report: {exc}") from exc
    if not isinstance(reports, list) or len(reports) != 1 \
            or reports[0].get("verdict") != "PASS" \
            or reports[0].get("sha256") != materials["submission"]["sha256"]:
        raise SystemExit("structural audit report does not authorize the current submission")

    print("\n# Receipt-bound local gates PASS; human still verifies remote kernel COMPLETE.")
    print(f"# receipt sha256 {sha256_file(receipt)}")
    print(f"# artifact sha256 {materials['submission']['sha256']}")
    print("$env:PYTHONUTF8=1; .\\.venv\\Scripts\\kaggle.exe competitions submit "
          f"-c {COMPETITION} -k {OWNER}/{spec['slug']} -v {version} "
          f"-f submission.csv -m \"{message}\"")
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
    ap.add_argument("--kernel-version", type=int)
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
        return cmd_audit(spec, dest, args.kernel_version)
    if args.stage == "submitcmd":
        return cmd_submitcmd(
            spec, args.message or spec.get("name", ""), dest, args.kernel_version,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
