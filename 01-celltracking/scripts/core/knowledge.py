"""Generate the Obsidian navigation layer from the canonical catalog. Nothing is copied.

WHERE IT LIVES AND WHY THERE
----------------------------
`knowledge/` sits at the REPOSITORY ROOT, not under `research/`. That is deliberate and it is not
cosmetic:

  * `validate_research_tree.py` globs `research/[0-9][0-9]-*/*.md`, so a generated tree under
    `research/` would either need 200 manifest entries or would sit invisibly inside the validated
    area - and an unvalidated document inside a validated tree is worse than one outside it;
  * `validate_registry.py` R5 scans `research/**/*.md` for superseded values, and generated notes
    quoting registry prose would start tripping a guard aimed at hand-written drift.

So the vault is the repository root, the canonical documents stay exactly where they are, and this
layer only NAVIGATES. Open the repository root as an Obsidian vault and start at
`knowledge/00-command-center.md`.

IDENTITY IS THE ID, NOT THE TITLE
---------------------------------
Every entity note is named for its stable id - `FACT-0412.md`, `LEVER-0046.md`,
`NOTEBOOK-kaggle_p35_dcveto_on_931.md` - so renaming a display title cannot break an edge. The
note bodies carry NO VALUES: a fact note says what the fact is ABOUT, its provenance and validity,
and links to `facts.yaml` for the number. That is what keeps this layer from becoming the second
maintained truth the registry exists to prevent (CLAUDE.md: the superseded score appeared 244
times across 36 files).

EDGES ARE DERIVED, NOT DRAWN
----------------------------
Every relationship comes from the catalog, which comes from the registry and the filesystem. There
is no hand-maintained link list to go stale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

import yaml  # noqa: E402

SCHEMA_VERSION = "knowledge_v1"
HEARTBEAT_OK = "KNOWLEDGE_OK"
HEARTBEAT_DRIFT = "KNOWLEDGE_DRIFT"

CAT = REPO / "research" / "00-system" / "registry" / "generated" / "catalog"
OUT = REPO / "knowledge"
EDGES = REPO / "research" / "00-system" / "registry" / "generated" / "edges.json"
BANNER = ("> [!info] Generated navigation. Do not edit.\n"
          "> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the\n"
          "> only source of truth for numbers; every value below is a LINK, never a copy.\n")


def load(name: str) -> dict:
    return json.loads((CAT / f"{name}.json").read_text(encoding="utf-8"))[name]


def note_id(kind: str, key: str) -> str:
    """A note id is also a FILENAME, so it must not contain a path separator.

    `SCRIPT-scripts/win_bet/foo.py` would make Obsidian treat the link as a folder path and the
    note would never resolve. Slashes and dots become double underscores; the readable path is
    kept inside the note.
    """
    if kind in ("NOTEBOOK", "SCRIPT", "TEST", "DOC", "SPEC", "FEATURE"):
        safe = key.replace("/", "__").replace("\\", "__")
        return f"{kind}-{safe}"
    return key


def link(target: str, label: str | None = None) -> str:
    return f"[[{target}|{label}]]" if label else f"[[{target}]]"


# ==============================================================================================
# EDGES - every relationship the brief names, derived
# ==============================================================================================
def build_edges() -> list[dict]:
    reg, scripts = load("registry"), load("scripts")
    tests, notebooks = load("tests"), load("notebooks")
    specs, docs = load("specs"), load("research_docs")
    E: list[dict] = []

    def add(src, rel, dst, basis):
        E.append({"src": src, "rel": rel, "dst": dst, "basis": basis})

    for lid, lv in reg["levers"].items():
        for f in lv["closed_by"]:
            add(f, "refutes", lid, "levers.yaml closed_by")
        for f in lv["supporting"]:
            add(f, "supports", lid, "levers.yaml supporting")
    for eid, e in reg["experiments"].items():
        for f in e["facts"]:
            add(eid, "produces", f, "experiments.yaml facts")
    for pid, pk in reg["packets"].items():
        if pk["lever"]:
            add(pid, "claims", pk["lever"], "packet lever field")
        for f in pk["input_facts"]:
            add(pid, "consumes", f, "packet inputs.facts")
    for fid, f in reg["facts"].items():
        if f.get("experiment"):
            add(fid, "measured_in", f["experiment"], "fact experiment field")
        if f.get("superseded_by"):
            add(fid, "superseded_by", f["superseded_by"], "fact superseded_by")
    for path, s in scripts.items():
        for fid in s["measures_facts"]:
            add(note_id("SCRIPT", path), "measures", fid, "fact instrument path")
        for pid in s["owning_packets"]:
            add(pid, "owns", note_id("SCRIPT", path), "packet inputs.code")
    for path, t in tests.items():
        for sp in t["protects_scripts"]:
            add(note_id("TEST", path), "protects", note_id("SCRIPT", sp), "test imports source")
    for name, n in notebooks.items():
        if n.get("producing_spec"):
            add(note_id("SPEC", n["producing_spec"]), "builds",
                note_id("NOTEBOOK", name), "spec out_dir")
        parent = n.get("base_notebook")
        if parent:
            pname = next((k for k, v in notebooks.items() if v["notebook"] == parent), None)
            if pname and pname != name:
                add(note_id("NOTEBOOK", name), "descends_from",
                    note_id("NOTEBOOK", pname), "spec base_notebook")
        for eid in n["experiments"]:
            add(eid, "runs", note_id("NOTEBOOK", name), "experiment spec -> notebook")
        for var in n.get("environment", {}):
            add(note_id("NOTEBOOK", name), "enables", note_id("FEATURE", var),
                "notebook environment")
    for path, d in docs.items():
        if d.get("superseded_by"):
            add(note_id("DOC", path), "supersedes_reverse", str(d["superseded_by"]),
                "frontmatter superseded_by")
    return E


# ==============================================================================================
# ENTITY NOTES - filename is the id, body carries no values
# ==============================================================================================
def write_entities(out: Path, edges: list[dict]) -> int:
    reg = load("registry")
    by_src, by_dst = defaultdict(list), defaultdict(list)
    for e in edges:
        by_src[e["src"]].append(e)
        by_dst[e["dst"]].append(e)
    d = out / "entities"
    d.mkdir(parents=True, exist_ok=True)
    n = 0

    def emit(nid: str, front: dict, lines: list[str]):
        nonlocal n
        body = ["---"]
        for k, v in front.items():
            body.append(f"{k}: {json.dumps(v) if isinstance(v, (list, dict)) else v}")
        body += ["---", "", BANNER, ""] + lines
        for rel_name, group, arrow in (("outgoing", by_src.get(nid, []), "->"),
                                       ("incoming", by_dst.get(nid, []), "<-")):
            if not group:
                continue
            body += ["", f"## {rel_name}"]
            for e in sorted(group, key=lambda x: (x["rel"], x["dst"], x["src"]))[:60]:
                other = e["dst"] if rel_name == "outgoing" else e["src"]
                body.append(f"- `{e['rel']}` {arrow} {link(other)}  <sub>{e['basis']}</sub>")
        (d / f"{nid}.md").write_text("\n".join(body) + "\n", encoding="utf-8")
        n += 1

    for fid, f in reg["facts"].items():
        emit(fid, {"id": fid, "kind": "FACT", "provenance": f["provenance"],
                   "validity": f["validity"] or "unset",
                   "tags": "[registry, fact]"},
             [f"**Value lives in** `{f['value_lives_in']}` - not copied here.",
              f"**Instrument:** `{(f['instrument'] or 'none')[:160]}`"])
    for eid, e in reg["experiments"].items():
        emit(eid, {"id": eid, "kind": "EXP", "status": e["status"] or "unknown",
                   "tags": "[registry, experiment]"},
             [f"**Kernel:** `{e['kernel']}`", f"**Spec:** `{e['spec']}`",
              f"**Submission:** {e['submission']}",
              f"**Score lives in** `{e['score_lives_in']}` - not copied here."])
    for lid, lv in reg["levers"].items():
        emit(lid, {"id": lid, "kind": "LEVER", "status": lv["status"],
                   "tags": f"[registry, lever, lever-{lv['status']}]"},
             [f"**Claim lives in** `{lv['claim_lives_in']}` - not copied here."])
    for pid, pk in reg["packets"].items():
        emit(pid, {"id": pid, "kind": "PKT", "lock": pk["lock"],
                   "lever": pk["lever"] or "none", "tags": "[registry, packet]"},
             [f"**Owner:** {pk['owner']}", f"**Opened:** {pk['opened']}"])
    # EVERY ENDPOINT GETS A NOTE. The first version emitted notes only for registry entities and
    # notebooks while creating edges to FEATURE, SCRIPT, TEST and SPEC nodes - 3,507 dangling
    # wikilinks, which in Obsidian look like real nodes until you click one. Emitting a note for
    # every endpoint makes a broken link impossible by construction rather than by discipline.
    for name, nb in load("notebooks").items():
        nid = note_id("NOTEBOOK", name)
        roles = nb.get("roles") or []
        emit(nid, {"id": nid, "kind": "NOTEBOOK", "status": nb["status"],
                   "roles": roles, "tags": "[notebook]"},
             [f"**Path:** `{nb['notebook']}`", f"**Spec:** `{nb['producing_spec']}`",
              f"**Status basis:** {nb['status_basis']}",
              f"**Environment variables:** {len(nb.get('environment', {}))}"])

    emitted = {q.stem for q in d.glob("*.md")}
    endpoints = {e["src"] for e in edges} | {e["dst"] for e in edges}
    scripts_cat, tests_cat = load("scripts"), load("tests")
    for nid in sorted(endpoints - emitted):
        kind = nid.split("-", 1)[0]
        raw = nid.split("-", 1)[1].replace("__", "/") if "-" in nid else nid
        front = {"id": nid, "kind": kind, "tags": f"[{kind.lower()}]"}
        lines = [f"**Path / name:** `{raw}`"]
        if kind == "SCRIPT" and raw in scripts_cat:
            c = scripts_cat[raw]
            front["lifecycle"] = c["lifecycle"]
            lines += [f"**Lifecycle:** {c['lifecycle']} — {c['lifecycle_basis']}",
                      f"**Tests:** {len(c['covered_by_tests'])}"]
        elif kind == "TEST" and raw in tests_cat:
            c = tests_cat[raw]
            front["test_class"] = c["test_class"]
            lines += [f"**Class:** {c['test_class']}", f"**Nodes:** {c['n_nodes']}",
                      f"**Blocks:** {c['blocking_scope']}"]
        elif kind == "FEATURE":
            lines += ["An environment variable the built notebooks set. Its VALUES live in the "
                      "notebooks and in `generated/catalog/notebooks.json`, not here."]
        emit(nid, front, lines)
    return n


# ==============================================================================================
# MAPS OF CONTENT
# ==============================================================================================
def write_mocs(out: Path, edges: list[dict]) -> int:
    reg, scripts = load("registry"), load("scripts")
    tests, notebooks = load("tests"), load("notebooks")
    roles = yaml.safe_load(
        (REPO / "research" / "00-system" / "registry" / "baseline_roles.yaml").read_text("utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    written = 0

    def page(fname: str, title: str, lines: list[str], tags: str):
        nonlocal written
        body = ["---", f"title: {title}", f"tags: {tags}", "generated: true", "---", "",
                f"# {title}", "", BANNER, ""] + lines
        (out / fname).write_text("\n".join(body) + "\n", encoding="utf-8")
        written += 1

    # --- command center
    live = [p for p, v in reg["packets"].items() if v["lock"] in ("claimed", "running")]
    openlv = [l for l, v in reg["levers"].items() if v["status"] == "open"]
    held = {reg["packets"][p]["lever"] for p in live if reg["packets"][p]["lever"]}
    page("00-command-center.md", "Campaign command center", [
        "## Start here",
        "- " + link("10-baseline-roles", "Baseline roles and notebook lineage"),
        "- " + link("20-levers", "Levers: active, open, killed"),
        "- " + link("30-experiments", "Experiments and submissions"),
        "- " + link("40-scripts", "Scripts by lifecycle"),
        "- " + link("50-tests", "Tests by protected contract"),
        "- " + link("60-features", "Features and parameters"),
        "- " + link("70-models", "Models and representations"),
        "- " + link("80-defects", "Defect ledger"),
        "- " + link("90-pending-decisions", "Pending decisions"),
        "",
        "## Where to look before opening a new investigation",
        f"- **{len(openlv)}** levers are `open`; **{len(openlv & held if isinstance(openlv, set) else set(openlv) & held)}** "
        "of them are already HELD by a live packet and may not be claimed.",
        "- Genuinely free to claim: " + ", ".join(
            link(l) for l in sorted(set(openlv) - held)) or "_none_",
        "",
        "## Live packets",
    ] + [f"- {link(p)} `{reg['packets'][p]['lock']}` -> "
         f"{link(reg['packets'][p]['lever']) if reg['packets'][p]['lever'] else '_no lever_'}"
         for p in sorted(live)], "[moc, command-center]")

    # --- baseline roles + lineage
    lines = ["## The three roles", ""]
    for role in ("leaderboard_champion", "operational_base", "scientific_control"):
        r = roles.get(role) or {}
        nb = r.get("notebook")
        name = next((k for k, v in notebooks.items() if v["notebook"] == nb), None)
        lines += [f"### `{role}`",
                  f"- Artifact: {link(note_id('NOTEBOOK', name)) if name else r.get('substrate', 'n/a')}",
                  f"- Established by: {r.get('established_by', r.get('selection_rule', 'see manifest'))}"[:300],
                  ""]
    lines += ["## Lineage (descends_from)", ""]
    for e in sorted([x for x in edges if x["rel"] == "descends_from"], key=lambda x: x["src"]):
        lines.append(f"- {link(e['src'])} -> {link(e['dst'])}")
    page("10-baseline-roles.md", "Baseline roles and notebook lineage", lines,
         "[moc, baseline, lineage]")

    # --- levers
    by_status = defaultdict(list)
    for lid, lv in reg["levers"].items():
        by_status[lv["status"]].append(lid)
    lines = []
    for st in ("open", "running", "parked", "killed", "closed"):
        lines += [f"## {st} ({len(by_status[st])})", ""]
        for lid in sorted(by_status[st]):
            ev = [e["src"] for e in edges if e["rel"] == "refutes" and e["dst"] == lid]
            lines.append(f"- {link(lid)}" + (f" — killed by {', '.join(link(f) for f in ev)}"
                                             if ev else ""))
        lines.append("")
    page("20-levers.md", "Levers by status and killing mechanism", lines, "[moc, lever]")

    # --- experiments
    lines = ["| experiment | status | kernel | submission | notebook |", "|---|---|---|---|---|"]
    for eid, e in sorted(reg["experiments"].items()):
        nb = [x["dst"] for x in edges if x["rel"] == "runs" and x["src"] == eid]
        lines.append(f"| {link(eid)} | {e['status']} | `{e['kernel'] or ''}` | "
                     f"{e['submission'] or ''} | {link(nb[0]) if nb else ''} |")
    page("30-experiments.md", "Experiments and submissions", lines, "[moc, experiment]")

    # --- scripts by lifecycle
    bylc = defaultdict(list)
    for p, s in scripts.items():
        bylc[s["lifecycle"]].append(p)
    lines = []
    for lc in sorted(bylc):
        lines += [f"## {lc} ({len(bylc[lc])})", ""]
        lines += [f"- `{p}`" + (f" — tests: {len(scripts[p]['covered_by_tests'])}"
                                if scripts[p].get("covered_by_tests") else " — **no test**")
                  for p in sorted(bylc[lc])[:80]]
        lines.append("")
    page("40-scripts.md", "Scripts by lifecycle", lines, "[moc, script]")

    # --- tests
    bycls = defaultdict(list)
    for p, t in tests.items():
        bycls[t["test_class"]].append(p)
    lines = []
    for cls in sorted(bycls):
        lines += [f"## {cls} ({len(bycls[cls])})", ""]
        lines += [f"- `{p}` — protects {len(tests[p]['protects_scripts'])} script(s), "
                  f"{tests[p]['n_nodes']} node(s), blocks `{tests[p]['blocking_scope']}`"
                  for p in sorted(bycls[cls])]
        lines.append("")
    page("50-tests.md", "Tests by protected contract", lines, "[moc, test]")

    # --- features
    featnb = defaultdict(list)
    for e in edges:
        if e["rel"] == "enables":
            featnb[e["dst"]].append(e["src"])
    lines = ["| feature | notebooks setting it |", "|---|---|"]
    for feat, nbs in sorted(featnb.items(), key=lambda kv: -len(kv[1]))[:220]:
        lines.append(f"| `{feat.removeprefix('FEATURE-')}` | {len(nbs)} |")
    page("60-features.md", "Features and parameters", lines, "[moc, feature]")

    # --- models / topics
    topics = {
        "Biohub-X": ["biohubx", "LEVER-0046", "PKT-0049"],
        "Trackastra": ["trackastra", "LEVER-0005", "PKT-0050"],
        "FOCUS-3D": ["focus3d", "FACT-0423", "FACT-0424", "FACT-0425", "FACT-0460"],
        "HOCT": ["hoct", "FACT-0451", "FACT-0453", "FACT-0454", "FACT-0455", "FACT-0456"],
        "Detection": ["detpeak", "deepcenter", "DET_THRESHOLD"],
        "Association": ["assoc", "MOTION_RELINK", "FACT-0428"],
        "Division": ["divverify", "div_", "SAFE_DIV", "FACT-0380"],
        "Consumers and graph ownership": ["finaledge", "relink", "FACT-0428", "LEVER-0044"],
    }
    lines = []
    for topic, keys in topics.items():
        hits = sorted({p for p in scripts if any(k.lower() in p.lower() for k in keys)})
        ids = [k for k in keys if re.match(r"(FACT|LEVER|PKT|EXP)-\d{4}$", k)]
        lines += [f"## {topic}", ""]
        if ids:
            lines.append("Registry: " + ", ".join(link(i) for i in ids))
        lines += [f"- `{p}`" for p in hits[:25]] + [""]
    page("70-models.md", "Models, representations and consumers", lines, "[moc, model]")

    # --- defects
    defect_facts = [f for f, v in reg["facts"].items()
                    if v["validity"] in ("SUSPECT", "INVALID")]
    lines = ["Facts whose VALIDITY is not clean. Validity is orthogonal to provenance: a fact can "
             "be MEASURED and INVALID at once.", "",
             "| fact | provenance | validity |", "|---|---|---|"]
    for f in sorted(defect_facts):
        v = reg["facts"][f]
        lines.append(f"| {link(f)} | {v['provenance']} | {v['validity']} |")
    page("80-defects.md", "Defect ledger", lines, "[moc, defect]")

    # --- pending decisions
    page("90-pending-decisions.md", "Pending decisions", [
        "Decisions measured and routed, but NOT taken. Each names its evidence.", "",
        "## Notebook line-ending contract",
        "- Evidence: `research/00-system/registry/generated/lineending_experiment.json`",
        "- Measured: `*.ipynb text eol=crlf` makes every platform resolve the recorded digests and "
        "rewrites zero tracked blobs. `-text` breaks them everywhere, including here.",
        "- Blocked on: a migration decision. `.gitattributes` is unchanged.",
        "",
        "## Stale packet locks",
        "- 4 packets hold levers that are already `killed`/`closed`, so the anti-duplication lock "
        "reserves decided levers.",
        "- Blocked on: a coordinator pass over the packet locks.",
        "",
        "## Release receipts",
        "- All raw receipts are gitignored; bindings are now durable via "
        "`generated/receipts.json`, but the raw audit bundles still do not survive a clone.",
        "",
        "## Gate B",
        "- Stopped at its recorded resume boundary. Resume state: "
        "`C:/temp/finaledge/gateB_v2/RESUME_STATE.md`.",
    ], "[moc, pending]")
    return written


def build(out: Path) -> dict:
    edges = build_edges()
    EDGES.parent.mkdir(parents=True, exist_ok=True)
    EDGES.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "edges": edges},
                                indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n_ent = write_entities(out, edges)
    n_moc = write_mocs(out, edges)
    return {"edges": len(edges), "entity_notes": n_ent, "mocs": n_moc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.check:
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            build(tmp)
            diff = []
            for p in sorted(tmp.rglob("*.md")):
                target = args.out / p.relative_to(tmp)
                if not target.is_file() or target.read_text("utf-8") != p.read_text("utf-8"):
                    diff.append(target.name)
            existing = {q.relative_to(args.out).as_posix() for q in args.out.rglob("*.md")} \
                if args.out.is_dir() else set()
            fresh = {q.relative_to(tmp).as_posix() for q in tmp.rglob("*.md")}
            diff += sorted(existing - fresh)
            if diff:
                print(f"{HEARTBEAT_DRIFT} {len(diff)} note(s) differ: {diff[:6]}")
                return 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  knowledge layer in sync ({len(fresh)} notes)")
        print(HEARTBEAT_OK)
        return 0
    counts = build(args.out)
    print(f"  wrote {args.out.relative_to(REPO).as_posix()}/  {counts}")
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
