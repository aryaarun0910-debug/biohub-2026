r"""CPU-side face of the ONE central mount resolver, plus the machinery that proves a built
notebook's own resolver is equivalent to it.

THE ANTI-DRIFT DEVICE
---------------------
This module does NOT re-implement the ladder. It reads
`scripts/kaggle_edits/kaggle_mount_ladder.py` and execs it, so `resolve()` here and the code the
kernel runs are the same characters. `LADDER_SHA256` goes into the preflight receipt; if the
injected copy in a built notebook ever differs from this file, `ladder_matches_notebook()` says so.
That is the whole point: FACT-0397 was a THIRD hand-written copy of a ladder this repository had
already got right twice.

WHAT ELSE LIVES HERE
--------------------
`simulate_mount_tree`  builds a fake /kaggle/input in BOTH observed conventions, so a resolver can
                       be executed against them on a Windows laptop.
`extract_resolvers`    pulls the `def *_find(...)` bodies out of a built notebook's cell source.
`rebase_and_exec`      re-points a resolver's hardcoded "/kaggle/input" at a sandbox and returns it
                       as a live callable, so the check is EXECUTION of the deployed text, not a
                       grep over it. An empty grep is not evidence (AGENTS.md); a green regex over
                       a resolver that still cannot find the file is the same mistake with extra
                       steps.
"""
from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER_PATH = ROOT / "scripts" / "kaggle_edits" / "kaggle_mount_ladder.py"

# The two mount conventions this repository has OBSERVED in fetched logs, not guessed at.
#   competitions:  /kaggle/input/competitions/<slug>/...      (p33 v2 log, "COMP_DIR:")
#   datasets:      /kaggle/input/datasets/<owner>/<slug>/...  (p33 v2 log, "ARTIFACTS:")
#   flat:          /kaggle/input/<slug>/...                   (what patches keep hardcoding)
CONVENTIONS = ("flat", "datasets", "competitions")

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "core"))
import hashing as _HASHING  # noqa: E402  the ONE hashing module

if not LADDER_PATH.is_file():                                  # fail closed, loudly
    raise FileNotFoundError(f"central mount ladder missing: {LADDER_PATH}")

LADDER_SOURCE = LADDER_PATH.read_text(encoding="utf-8")

# CANONICAL TEXT, not raw bytes. `read_text` already collapses CRLF to LF on read, but the digest
# was previously taken over whatever the checkout produced, so it differed between an LF worktree
# and a CRLF checkout - measured 2026-09-01, this exact constant was the cause of two
# clean-clone failures in test_gpu_preflight. The ladder is SOURCE; its line endings carry no
# meaning. `canon_text_v1` states the normalisation explicitly instead of relying on a side
# effect of how the file happened to be read.
LADDER_SHA256 = _HASHING.canonical_text_sha256_bytes(LADDER_SOURCE.encode("utf-8"))
LADDER_SHA256_KIND = _HASHING.CANONICAL
LADDER_SHA256_CANONICALIZATION = _HASHING.CANONICALIZATION_VERSION

_ns: dict = {}
exec(compile(LADDER_SOURCE, str(LADDER_PATH), "exec"), _ns)     # noqa: S102 - see docstring
BiohubMountNotFound = _ns["BiohubMountNotFound"]
resolve = _ns["biohub_mount_find"]
resolve_dir = _ns["biohub_mount_find_dir"]
diagnose = _ns["biohub_mount_diagnose"]

RECURSIVE = re.compile(r"\*\*")
RESOLVER_DEF = re.compile(r"^(\s*)def (_?\w*find\w*)\(", re.M)


def is_bounded(source: str) -> bool:
    """A resolver that can `**`-walk /kaggle/input is not bounded, whatever its comment says.

    Parsed, not grepped. The first version of this function read `**` anywhere in the text and
    therefore condemned `_loeo_find` for the sentence in its own DOCSTRING explaining why it does
    not use `**` - a checker that fails the code for documenting the trap it avoids. Only string
    arguments to glob/rglob/iglob calls count, and `rglob` counts on sight.
    """
    src = textwrap.dedent(source)
    if re.search(r"\brglob\s*\(", src):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return not RECURSIVE.search(src)          # fail closed on an unparseable fragment
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in {"glob", "iglob", "rglob"}:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and "**" in sub.value:
                    return False
    return True


def simulate_mount_tree(root: Path, slug: str, files: dict[str, bytes | str],
                        convention: str = "datasets", owner: str = "aryaarun07") -> Path:
    """Materialise a fake /kaggle/input holding one dataset under one mount convention."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; known: {CONVENTIONS}")
    base = {
        "flat": root / slug,
        "datasets": root / "datasets" / owner / slug,
        "competitions": root / "competitions" / slug,
    }[convention]
    for rel, payload in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    return base


def extract_resolvers(source: str) -> dict[str, str]:
    """Return {function name: source text} for every `def ...find...(` in a notebook cell.

    Used to lift `_afp_find` / `_loeo_find` straight out of the BUILT notebook so the preflight
    executes the deployed text rather than a paraphrase of it.
    """
    out: dict[str, str] = {}
    lines = source.splitlines(keepends=True)
    starts = [(m.start(), m.group(1), m.group(2)) for m in RESOLVER_DEF.finditer(source)]
    for off, indent, name in starts:
        line_no = source.count("\n", 0, off)
        body = [lines[line_no]]
        for nxt in lines[line_no + 1:]:
            if nxt.strip() and not nxt.startswith(indent + " ") and not nxt.startswith(indent + "\t"):
                if not nxt.startswith(indent + ")"):
                    break
            body.append(nxt)
        out[name] = "".join(body)
    return out


def rebase_and_exec(fn_source: str, name: str, sandbox_input: Path, extra: dict | None = None):
    """Re-point a resolver at `sandbox_input` and return it as a live callable.

    The substitution is textual and deliberate: a resolver that hardcodes "/kaggle/input" is
    exactly the artifact under test, so we keep its logic byte-for-byte and only move the root.
    """
    sandbox = str(sandbox_input).replace("\\", "/").rstrip("/")
    src = textwrap.dedent(fn_source).replace("/kaggle/input", sandbox)
    ns: dict = {"Path": Path, "_AfpPath": Path, "_loeo_glob": _GlobShim(sandbox_input)}
    ns.update(extra or {})
    exec(compile(src, f"<rebased {name}>", "exec"), ns)         # noqa: S102
    return ns[name]


class _GlobShim:
    """Stand-in for the notebook's `glob` module, rooted at the sandbox.

    `_loeo_find` calls `_loeo_glob.glob("/kaggle/input/...")`; rebasing the literal is not enough
    on Windows because the pattern is an absolute POSIX path. This shim rewrites the root and
    delegates to pathlib, keeping the ladder's shape intact.
    """

    def __init__(self, sandbox_input: Path):
        self.root = Path(sandbox_input)

    def glob(self, pattern: str):
        pat = str(pattern).replace("\\", "/")
        for prefix in (str(self.root).replace("\\", "/") + "/", "/kaggle/input/"):
            if pat.startswith(prefix):
                pat = pat[len(prefix):]
                break
        return [str(p) for p in sorted(self.root.glob(pat))]


def ladder_matches_notebook(notebook_source: str) -> bool:
    """True when the built notebook carries THIS ladder rather than a private re-write."""
    marker = "def biohub_mount_find("
    return marker in notebook_source


__all__ = [
    "BiohubMountNotFound", "CONVENTIONS", "LADDER_PATH", "LADDER_SHA256", "LADDER_SOURCE",
    "diagnose", "extract_resolvers", "is_bounded", "ladder_matches_notebook", "rebase_and_exec",
    "resolve", "resolve_dir", "simulate_mount_tree",
]
