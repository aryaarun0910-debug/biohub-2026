"""THE CANONICAL CATALOG. One generator, one schema, many views.

WHY ONE CATALOG
---------------
Obsidian and RAG are VIEWS. If each built its own index, they would disagree, and the project has
already paid for exactly that shape: FACT-0431 records two committed guards answering one question
differently, and whichever ran last looked authoritative. So there is one generator here and the
Maps of Content and the RAG index are both derived from its output.

THE REGISTRY REMAINS AUTHORITATIVE FOR NUMBERS. This catalog links to FACT/EXP/LEVER/PKT ids and
never copies a score. `test_catalog.py` enforces that by scanning the emitted JSON for
score-shaped literals - the same guard `baseline_roles.yaml` carries, for the same reason
(CLAUDE.md: the superseded score appeared 244 times across 36 files while the live one appeared
31 times across 7).

EVERY FILE GETS A STATUS, AND `unknown` IS A REAL ANSWER
-------------------------------------------------------
Invented provenance is forbidden. Where a binding cannot be established from the repository, the
field is `null` and the status says `unknown` or `unbound-historical`. A catalog that guesses is
worse than one that admits a gap, because the gap is what tells the next agent where to look.

WHAT IS DERIVED MECHANICALLY AND WHAT IS NOT
--------------------------------------------
Derived: paths, hashes, imports, callers, test->source edges, spec->notebook lineage, environment
variables, registry cross-references, feature deltas. Declared by a human and merely READ here:
the three baseline roles (`baseline_roles.yaml`), lifecycle overrides (`catalog_overrides.yaml`).
The split matters: anything derived can be regenerated and drift-checked; anything declared has to
be reviewed when it changes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

import yaml  # noqa: E402

import hashing as H  # noqa: E402  the ONE hashing module (Phase 1.5)


def _digest(path: Path, kind: str) -> dict:
    """A hash-bearing field, with its KIND declared.

    Bare `sha256` fields made this catalog checkout-dependent: 232 of the repository's 816
    recorded digests matched the WORKTREE bytes only, and the catalog was one of the places
    recording them. Source now carries CANONICAL_TEXT_SHA256, which is equal across an LF and a
    CRLF checkout; artifacts whose identity IS their bytes keep RAW_ARTIFACT_SHA256.
    """
    try:
        if kind == "DUAL":
            return H.dual_record(path, repo_root=REPO)
        return H.hash_record(path, kind, repo_root=REPO)
    except (H.HashRefusal, OSError) as exc:
        return {"hash_kind": None, "refused": str(exc)[:180]}

SCHEMA_VERSION = "catalog_v1"
HEARTBEAT_OK = "CATALOG_OK"
HEARTBEAT_DRIFT = "CATALOG_DRIFT"

OUT_DIR = REPO / "research" / "00-system" / "registry" / "generated" / "catalog"
REGISTRY = REPO / "research" / "00-system" / "registry"
OVERRIDES = REGISTRY / "catalog_overrides.yaml"

SCRIPT_ROOTS = ("scripts",)
TEST_ROOT = "tests"
NOTEBOOK_ROOT = "notebooks"
SPEC_ROOT = "scripts/kaggle_specs"

_ENVVAR = re.compile(r"""os\.environ\[\s*["']([A-Za-z0-9_]+)["']\s*\]\s*=""")
_FACT = re.compile(r"\b(FACT-\d{4})\b")
_EXP = re.compile(r"\b(EXP-\d{4})\b")
_LEVER = re.compile(r"\b(LEVER-\d{4})\b")
_PKT = re.compile(r"\b(PKT-\d{4})\b")


# ==============================================================================================
# helpers
# ==============================================================================================
def sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(8 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def git_tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=180)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def ids_in(text: str) -> dict:
    return {
        "facts": sorted(set(_FACT.findall(text))),
        "experiments": sorted(set(_EXP.findall(text))),
        "levers": sorted(set(_LEVER.findall(text))),
        "packets": sorted(set(_PKT.findall(text))),
    }


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def overrides() -> dict:
    if OVERRIDES.is_file():
        return yaml.safe_load(read_text(OVERRIDES)) or {}
    return {}


# ==============================================================================================
# registry entities - the spine everything else links to
# ==============================================================================================
def build_registry() -> dict:
    facts = yaml.safe_load(read_text(REGISTRY / "facts.yaml"))["facts"]
    exp_doc = yaml.safe_load(read_text(REGISTRY / "experiments.yaml"))
    exps = exp_doc["experiments"] if isinstance(exp_doc, dict) and "experiments" in exp_doc \
        else exp_doc
    levers = yaml.safe_load(read_text(REGISTRY / "levers.yaml"))["levers"]
    packets = [yaml.safe_load(read_text(p))
               for p in sorted((REGISTRY / "packets").glob("PKT-0*.yaml"))]

    ent = {"facts": {}, "experiments": {}, "levers": {}, "packets": {}}
    for f in facts:
        ent["facts"][f["id"]] = {
            "id": f["id"], "kind": "FACT",
            "provenance": f.get("provenance"), "validity": f.get("validity"),
            "superseded_by": f.get("superseded_by"),
            "guarded": bool(f.get("guard_state_docs")),
            "experiment": f.get("experiment"),
            "instrument": f.get("instrument"),
            "instrument_paths": sorted(set(re.findall(
                r"(?:scripts|src|tests|vendor|notebooks)/[\w./-]+\.(?:py|ipynb)",
                str(f.get("instrument") or "")))),
            # NO VALUE IS COPIED. The registry is the only place a number lives.
            "value_lives_in": "research/00-system/registry/facts.yaml",
        }
    for e in exps:
        ent["experiments"][e["id"]] = {
            "id": e["id"], "kind": "EXP", "status": e.get("status"),
            "kernel": e.get("kernel"), "spec": e.get("spec"),
            "submission": e.get("submission"), "facts": e.get("facts") or [],
            "has_score": e.get("lb") is not None,
            "score_lives_in": "research/00-system/registry/experiments.yaml",
        }
    for lv in levers:
        ent["levers"][lv["id"]] = {
            "id": lv["id"], "kind": "LEVER", "status": lv.get("status"),
            "closed_by": lv.get("closed_by") or [],
            "supporting": lv.get("supporting") or [],
            # THE CLAIM PROSE IS NOT COPIED. It routinely quotes scores, and a generated file is
            # the worst place for a stale number to sit: it looks authoritative and regenerates
            # without review. CLAUDE.md measured the cost - the superseded score appeared 244
            # times across 36 files. Read the claim in levers.yaml, which is where it lives.
            "claim_lives_in": "research/00-system/registry/levers.yaml",
        }
    for pk in packets:
        if not pk:
            continue
        inp = pk.get("inputs") or {}
        ent["packets"][pk["id"]] = {
            "id": pk["id"], "kind": "PKT", "lock": pk.get("lock"),
            "lever": pk.get("lever"), "owner": pk.get("owner"), "opened": pk.get("opened"),
            "input_facts": inp.get("facts") or [],
            "input_code": inp.get("code") or [],
            "input_artifacts": inp.get("artifacts") or [],
        }
    return ent


# ==============================================================================================
# scripts
# ==============================================================================================
def _imports_of(path: Path) -> list[str]:
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError:
        return []
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module)
    return sorted(names)


def build_scripts(reg: dict, tracked: set[str], ov: dict) -> dict:
    out: dict[str, dict] = {}
    files: list[Path] = []
    for root in SCRIPT_ROOTS:
        files += [p for p in (REPO / root).rglob("*.py") if "__pycache__" not in p.parts]
        files += [p for p in (REPO / root).rglob("*.json") if "__pycache__" not in p.parts]
    stem_index: dict[str, str] = {}
    for p in files:
        if p.suffix == ".py":
            stem_index.setdefault(p.stem, rel(p))

    # fact instruments -> script
    measures: dict[str, list[str]] = {}
    for fid, f in reg["facts"].items():
        for ip in f["instrument_paths"]:
            measures.setdefault(ip, []).append(fid)
    # packet code -> script
    owned: dict[str, list[str]] = {}
    for pid, pk in reg["packets"].items():
        for c in pk["input_code"]:
            owned.setdefault(str(c).strip(), []).append(pid)

    for p in sorted(files):
        r = rel(p)
        text = read_text(p)
        doc = None
        imports: list[str] = []
        if p.suffix == ".py":
            imports = _imports_of(p)
            try:
                doc = (ast.get_docstring(ast.parse(text)) or "").strip().splitlines()
                doc = doc[0] if doc else None
            except SyntaxError:
                doc = None
        entry = {
            "id": "SCRIPT:" + r, "path": r, "kind": p.suffix.lstrip("."),
            "purpose": doc,
            "tracked": r in tracked,
            # source under scripts/ - canonical, so the catalog is the same in any checkout
            "digest": _digest(p, H.CANONICAL if p.suffix == ".py" else H.CANONICAL),
            "bytes": p.stat().st_size,
            "imports": imports,
            "imports_first_party": [m for m in imports if m.split(".")[0] in stem_index
                                    or m.startswith("scripts.")],
            "measures_facts": sorted(measures.get(r, [])),
            "owning_packets": sorted(owned.get(r, [])),
            "ids_mentioned": ids_in(text),
            "reads_temp_paths": sorted(set(re.findall(r"C:/temp/[\w./-]+", text)))[:12],
            "subprocess_boundary": "subprocess" in imports,
            "dynamic_import": any(t in text for t in ("importlib", "__import__", "exec_module")),
            "callers": [],            # filled below
            "covered_by_tests": [],   # filled by build_tests
            "lifecycle": "unknown",   # refined below / overridden
            "lifecycle_basis": "not established",
        }
        out[r] = entry

    # callers: who imports whom (first-party, by module stem)
    by_stem = {Path(r).stem: r for r in out if r.endswith(".py")}
    for r, e in out.items():
        for m in e["imports"]:
            tgt = by_stem.get(m.split(".")[-1])
            if tgt and tgt != r:
                out[tgt]["callers"].append(r)
    for e in out.values():
        e["callers"] = sorted(set(e["callers"]))

    # lifecycle, from evidence only
    for r, e in out.items():
        if not e["tracked"]:
            e["lifecycle"], e["lifecycle_basis"] = "untracked-concurrent", "present but not in git"
        elif e["owning_packets"]:
            live = [p for p in e["owning_packets"]
                    if reg["packets"].get(p, {}).get("lock") in ("claimed", "running")]
            e["lifecycle"] = "active-experiment" if live else "historical"
            e["lifecycle_basis"] = f"named by packet(s) {e['owning_packets']}"
        elif e["measures_facts"]:
            e["lifecycle"], e["lifecycle_basis"] = "reusable-instrument", "named as a fact instrument"
        elif e["callers"]:
            e["lifecycle"], e["lifecycle_basis"] = "supporting", "imported by another script"
        elif not r.endswith(".py"):
            e["lifecycle"] = "data"
            e["lifecycle_basis"] = ("a JSON data file under scripts/ - specs, configs and edit "
                                    "payloads have no importers by construction")
        else:
            e["lifecycle"], e["lifecycle_basis"] = "unknown", "no packet, fact or importer found"
    for r, decl in (ov.get("scripts") or {}).items():
        if r in out:
            out[r]["lifecycle"] = decl.get("lifecycle", out[r]["lifecycle"])
            out[r]["lifecycle_basis"] = "declared in catalog_overrides.yaml: " + \
                str(decl.get("why", "no reason given"))
            out[r]["retirement_evidence"] = decl.get("evidence")
    return out


# ==============================================================================================
# tests
# ==============================================================================================
def build_tests(scripts: dict, tracked: set[str], node_ids: list[str] | None) -> dict:
    out: dict[str, dict] = {}
    by_stem = {Path(r).stem: r for r in scripts if r.endswith(".py")}
    nodes_by_file: dict[str, list[str]] = {}
    for n in (node_ids or []):
        nodes_by_file.setdefault(n.split("::")[0].replace("\\", "/"), []).append(n)

    for p in sorted((REPO / TEST_ROOT).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        r = rel(p)
        text = read_text(p)
        imports = _imports_of(p)
        protects = sorted({by_stem[m.split(".")[-1]] for m in imports
                           if m.split(".")[-1] in by_stem})
        klass = "unit"
        low = text.lower()
        if "mutation" in low or "proved by mutation" in low:
            klass = "mutation"
        elif "parity" in low:
            klass = "parity"
        elif "provenance" in low or "restriction" in low:
            klass = "provenance"
        elif "integration" in low:
            klass = "integration"
        elif "regression" in low:
            klass = "regression"
        skip_reasons = re.findall(r"skip\(\s*[\"'](.{0,90})", text)
        out[r] = {
            "id": "TEST:" + r, "path": r, "tracked": r in tracked,
            "digest": _digest(p, H.CANONICAL),
            "node_ids": sorted(nodes_by_file.get(r, [])),
            "n_nodes": len(nodes_by_file.get(r, [])),
            "protects_scripts": protects,
            "test_class": klass,
            "ids_mentioned": ids_in(text),
            "has_skip_guard": bool(skip_reasons),
            "skip_reasons": skip_reasons[:4],
            "needs_vendor": "vendor" in text,
            "needs_data_dir": bool(re.search(r"data[/\\](train|test)", text)),
            "fixture_parity": ("declared-unit-geometry"
                               if "ISOTROPIC_UNIT_FIXTURE_SCALE" in text else "not declared"),
            "blocking_scope": ("release" if "receipt" in low or "submission" in low
                               else "gpu" if "gpu" in low else "cpu"),
            "state": "active" if r in tracked else "untracked-concurrent",
        }
        # A test may exercise a script without importing it - through a subprocess, a path
        # literal or a generated patch. That edge is REAL but weaker than an import, so it is
        # recorded separately rather than folded in: conflating them would overstate coverage.
        named = sorted({sr for st, sr in by_stem.items()
                        if sr not in protects and re.search(rf"{re.escape(st)}", text)})
        out[r]["names_scripts_without_importing"] = named
        for s in protects:
            scripts[s]["covered_by_tests"].append(r)
        for s in named:
            scripts[s].setdefault("named_by_tests", []).append(r)
    for s in scripts.values():
        s["covered_by_tests"] = sorted(set(s["covered_by_tests"]))
        s["named_by_tests"] = sorted(set(s.get("named_by_tests", [])))
        s["test_coverage"] = ("imported-by-test" if s["covered_by_tests"]
                              else "named-only" if s["named_by_tests"] else "none")
    return out


# ==============================================================================================
# notebooks and specs
# ==============================================================================================
def _nb_lines(cell: dict) -> list[str]:
    src = cell.get("source", [])
    return src.splitlines() if isinstance(src, str) else [str(s).rstrip("\n") for s in src]


def _nb_env(p: Path) -> dict:
    try:
        nb = json.loads(read_text(p))
    except json.JSONDecodeError:
        return {}
    vals: dict[str, str] = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in _nb_lines(cell):
            m = _ENVVAR.search(line)
            if m:
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                vals[m.group(1)] = rhs.rstrip(",").strip()
    return vals


def build_notebooks(reg: dict, tracked: set[str], roles: dict) -> tuple[dict, dict]:
    specs: dict[str, dict] = {}
    for p in sorted((REPO / SPEC_ROOT).glob("*.json")):
        try:
            s = json.loads(read_text(p))
        except json.JSONDecodeError:
            continue
        specs[rel(p)] = {
            "id": "SPEC:" + rel(p), "path": rel(p), "tracked": rel(p) in tracked,
            "digest": _digest(p, H.CANONICAL), "name": s.get("name"), "slug": s.get("slug"),
            "out_dir": s.get("out_dir"),
            "base_notebook": str(s.get("base_notebook") or s.get("base") or "").replace("\\", "/")
                             or None,
            "base_sha256": s.get("base_sha256"),
            "declares_binding_restriction": "binding_restriction" in s,
            "edits": len(s.get("edits") or []),
        }

    spec_by_outdir = {v["out_dir"]: k for k, v in specs.items() if v.get("out_dir")}
    exp_by_spec: dict[str, list[str]] = {}
    for eid, e in reg["experiments"].items():
        if e.get("spec"):
            exp_by_spec.setdefault(str(e["spec"]).replace("\\", "/"), []).append(eid)

    # ONE NOTEBOOK CAN HOLD MORE THAN ONE ROLE, and here one does: P35 is both the leaderboard
    # champion and the operational base. The first version of this map used `role_by_path[nb] =
    # role`, so the second assignment silently overwrote the first and the champion was reported
    # as the operational base only. Collapsing the roles is the exact thing baseline_roles.yaml
    # exists to stop.
    role_by_path: dict[str, list[str]] = {}
    for role in ("leaderboard_champion", "operational_base", "scientific_control"):
        nb = (roles.get(role) or {}).get("notebook")
        if nb:
            role_by_path.setdefault(nb, []).append(role)

    notebooks: dict[str, dict] = {}
    for d in sorted((REPO / NOTEBOOK_ROOT).iterdir()):
        if not d.is_dir():
            continue
        nbs = sorted(d.glob("*.ipynb"))
        man = d / "build_manifest.json"
        kmeta = d / "kernel-metadata.json"
        manifest = {}
        if man.is_file():
            try:
                manifest = json.loads(read_text(man))
            except json.JSONDecodeError:
                manifest = {}
        km = {}
        if kmeta.is_file():
            try:
                km = json.loads(read_text(kmeta))
            except json.JSONDecodeError:
                km = {}
        nb_path = rel(nbs[0]) if nbs else None
        spec_rel = spec_by_outdir.get(rel(d))
        exps = exp_by_spec.get(spec_rel or "", [])
        nb_roles = role_by_path.get(nb_path or "", [])
        entry = {
            "id": "NOTEBOOK:" + d.name, "dir": rel(d), "notebook": nb_path,
            "tracked": (nb_path in tracked) if nb_path else False,
            # a built notebook needs BOTH: RAW is what was pushed to Kaggle and scored,
            # CANONICAL is what a cross-checkout source comparison is about.
            "digest": _digest(nbs[0], "DUAL") if nbs else None,
            "notebook_sha256": sha256_file(nbs[0]) if nbs else None,
            "producing_spec": spec_rel,
            "spec_sha256": specs.get(spec_rel, {}).get("sha256") if spec_rel else None,
            "base_notebook": (specs.get(spec_rel, {}) or {}).get("base_notebook")
                             if spec_rel else manifest.get("base_notebook"),
            "base_sha256": (specs.get(spec_rel, {}) or {}).get("base_sha256")
                           if spec_rel else manifest.get("base_sha256"),
            # a manifest records what the factory SHIPPED - raw
            "build_manifest_digest": _digest(man, H.RAW) if man.is_file() else None,
            "build_manifest_sha256": sha256_file(man) if man.is_file() else None,
            "built_sha256_recorded": manifest.get("built_sha256"),
            "built_sha256_matches_disk": (
                manifest.get("built_sha256") == sha256_file(nbs[0]) if nbs and manifest else None),
            "kernel_slug": km.get("id"),
            "experiments": exps,
            "facts": sorted({f for e in exps for f in reg["experiments"][e]["facts"]}),
            "submissions": [reg["experiments"][e]["submission"] for e in exps
                            if reg["experiments"][e].get("submission")],
            "release_receipt": rel(d / "_out" / "audit_receipt.json")
                               if (d / "_out" / "audit_receipt.json").is_file() else None,
            "release_receipt_tracked": False,   # _out/ is gitignored - see PHASE 1C envelopes
            "environment": _nb_env(nbs[0]) if nbs else {},
            "roles": nb_roles,
            "status": None,
            "status_basis": None,
        }
        if nb_roles:
            entry["status"] = "+".join(nb_roles)
            entry["status_basis"] = "declared in baseline_roles.yaml"
        elif exps:
            entry["status"] = "candidate"
            entry["status_basis"] = f"bound to {exps}"
        elif nb_path:
            entry["status"] = "unbound-historical"
            entry["status_basis"] = ("no experiment references its spec; provenance is NOT "
                                     "invented retrospectively")
        else:
            entry["status"], entry["status_basis"] = "unknown", "directory holds no .ipynb"
        notebooks[d.name] = entry

    # deltas from parent, computed from the environment maps
    by_path = {v["notebook"]: k for k, v in notebooks.items() if v["notebook"]}
    for name, e in notebooks.items():
        parent = e.get("base_notebook")
        pk = by_path.get(parent) if parent else None
        if not pk or pk == name:
            e["delta_from_parent"] = None
            continue
        a, b = e["environment"], notebooks[pk]["environment"]
        shared = sorted(set(a) & set(b))
        e["delta_from_parent"] = {
            "parent": notebooks[pk]["id"],
            "shared_variables": len(shared),
            "changed": {k: {"parent": b[k], "child": a[k]} for k in shared if a[k] != b[k]},
            "only_in_child": sorted(set(a) - set(b)),
            "only_in_parent": sorted(set(b) - set(a)),
        }
    return notebooks, specs


# ==============================================================================================
# research documents
# ==============================================================================================
def build_research(tracked: set[str]) -> dict:
    out: dict[str, dict] = {}
    for p in sorted((REPO / "research").rglob("*.md")):
        r = rel(p)
        text = read_text(p)
        fm = {}
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    fm = {}
        out[r] = {
            "id": "DOC:" + r, "path": r, "tracked": r in tracked,
            "doc_id": fm.get("id"), "title": fm.get("title"), "area": fm.get("area"),
            "status": fm.get("status"), "record_kind": fm.get("record_kind"),
            "owner": fm.get("owner"),
            "updated": str(fm.get("updated")) if fm.get("updated") is not None else None,
            "updated_is_a_string": isinstance(fm.get("updated"), str),
            "has_frontmatter": bool(fm),
            "certified_by_validator": bool(re.match(r"research/\d\d-[^/]+/[^/]+\.md$", r)),
            "supersedes": fm.get("supersedes"),
            "superseded_by": fm.get("superseded_by"),
            "entity_links": ids_in(text),
            "headings": len(re.findall(r"^#{1,6} ", text, flags=re.M)),
            "bytes": p.stat().st_size,
        }
    return out


# ==============================================================================================
# assembly
# ==============================================================================================
def collect_node_ids() -> list[str]:
    """Collect pytest node ids OURSELVES rather than accepting them as an argument.

    The first version took `--node-ids` as an optional input, so the emitted catalog depended on
    whether the caller happened to pass one - and `--check`, which did not, always reported drift
    against a catalog generated with it. A drift lock whose answer depends on an optional argument
    is not a lock. Collection is deterministic and costs about half a minute, so the generator
    does it itself and the output is a function of the repository alone.
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    return sorted(line.strip().replace("\\", "/")
                  for line in r.stdout.splitlines() if "::" in line)


def build(node_ids: list[str] | None = None) -> dict:
    if node_ids is None:
        node_ids = collect_node_ids()
    tracked = git_tracked()
    ov = overrides()
    roles = yaml.safe_load(read_text(REGISTRY / "baseline_roles.yaml")) or {}
    reg = build_registry()
    scripts = build_scripts(reg, tracked, ov)
    tests = build_tests(scripts, tracked, node_ids)
    notebooks, specs = build_notebooks(reg, tracked, roles)
    docs = build_research(tracked)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/core/catalog.py",
        "do_not_edit": "Generated. Regenerate; --check locks it against hand edits.",
        "authority": ("The REGISTRY is authoritative for every scientific number. This catalog "
                      "links by FACT/EXP/LEVER/PKT id and copies no values."),
        "counts": {
            "registry_facts": len(reg["facts"]), "registry_experiments": len(reg["experiments"]),
            "registry_levers": len(reg["levers"]), "registry_packets": len(reg["packets"]),
            "scripts": len(scripts), "tests": len(tests), "notebooks": len(notebooks),
            "specs": len(specs), "research_docs": len(docs),
        },
        "registry": reg, "scripts": scripts, "tests": tests,
        "notebooks": notebooks, "specs": specs, "research_docs": docs,
    }


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"


def write(payload: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in ("registry", "scripts", "tests", "notebooks", "specs", "research_docs"):
        p = out_dir / f"{key}.json"
        p.write_text(canonical({"schema_version": SCHEMA_VERSION, key: payload[key]}),
                     encoding="utf-8")
        written.append(p)
    idx = out_dir / "index.json"
    idx.write_text(canonical({k: v for k, v in payload.items()
                              if k not in ("registry", "scripts", "tests", "notebooks",
                                           "specs", "research_docs")}), encoding="utf-8")
    written.append(idx)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--node-ids", type=Path,
                    help="pre-collected pytest node list; omit and the generator collects its own")
    args = ap.parse_args(argv)

    nodes = None
    if args.node_ids and args.node_ids.is_file():   # an override for speed, not for content
        nodes = [l.strip() for l in args.node_ids.read_text(encoding="utf-8").splitlines()
                 if "::" in l]
    payload = build(nodes)

    if args.check:
        drift = []
        for key in ("registry", "scripts", "tests", "notebooks", "specs", "research_docs"):
            p = args.out / f"{key}.json"
            if not p.is_file():
                drift.append(f"{p.name} missing")
                continue
            want = canonical({"schema_version": SCHEMA_VERSION, key: payload[key]})
            if p.read_text(encoding="utf-8") != want:
                drift.append(f"{p.name} differs from a fresh generation")
        if drift:
            print(HEARTBEAT_DRIFT + " " + "; ".join(drift))
            return 1
        print(f"  catalog in sync: {payload['counts']}")
        print(HEARTBEAT_OK)
        return 0

    for p in write(payload, args.out):
        print(f"  wrote {rel(p)}")
    print(f"  {payload['counts']}")
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
