"""Local, deterministic RAG over the canonical catalog. Artifacts live in `.claude/rag/`.

WHY LEXICAL FIRST
-----------------
An embedding index is a black box whose ranking cannot be argued with. This project's failures are
almost all "a plausible answer that was wrong", so the first index is BM25-style lexical: given
the same corpus it returns the same ranking, and a bad result can be traced to a term. Embeddings
may be added later behind the same metadata contract - `search()` does not care where a candidate
came from, only that it carries the schema below.

WHAT IT REFUSES TO DO
---------------------
The registry is authoritative for numbers, and stale prose is the thing that made a registry
necessary (CLAUDE.md: the superseded score appeared 244 times across 36 files while the live one
appeared 31 times across 7). So:

  * every chunk carries `provenance`, `validity` and `superseded_by`;
  * SUPERSEDED, INVALID, SUSPECT, EXTERNAL and UNVERIFIED content is DEMOTED, never silently
    ranked beside a live VERIFIED fact;
  * a result whose chunk is superseded is returned with a REFUSAL BANNER rather than as an
    answer, so a caller cannot quote it as current state without seeing that it is not;
  * binary artifacts and generated notebook output are excluded entirely.

Every result carries its entity ID and its path, so an answer can always be checked at source.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

SCHEMA_VERSION = "rag_v1"
HEARTBEAT_OK = "RAG_INDEX_OK"
OUT = REPO / ".claude" / "rag"
CAT = REPO / "research" / "00-system" / "registry" / "generated" / "catalog"

#: rank multipliers. A live measured fact must outrank a paragraph of prose that mentions it.
TIER = {
    "VERIFIED": 1.00, "MEASURED": 0.95, "EXTERNAL": 0.55,
    "UNVERIFIED": 0.40, "SUPERSEDED": 0.15, None: 0.70,
}
VALIDITY_PENALTY = {"VALID": 1.0, None: 0.9, "SUSPECT": 0.5, "INVALID": 0.15}

_WORD = re.compile(r"[A-Za-z0-9_.-]{2,}")
SKIP_SUFFIX = (".ipynb", ".png", ".parquet", ".npz", ".npy", ".pth", ".pt", ".tif",
               ".geff", ".zip", ".csv", ".sqlite", ".log")


def tok(text: str) -> list[str]:
    """Emit the whole token AND its parts.

    Identifiers in this repository are snake/dotted - `BIOHUB_MOTION_RELINK_LEARNED_BONUS`,
    `scripts/win_bet/finaledge_gates.py`. Indexing them whole means no natural-language question
    can ever reach them: "which notebook introduced motion relink learned bonus" shares not one
    token with the identifier it is asking about. Splitting as well as keeping the whole form
    lets an exact identifier query still match exactly.
    """
    out: list[str] = []
    for w in _WORD.findall(text):
        w = w.lower()
        out.append(w)
        if "_" in w or "." in w or "-" in w:
            out.extend(part for part in re.split(r"[._-]+", w) if len(part) > 1)
    return out


def load_cat(name: str) -> dict:
    return json.loads((CAT / f"{name}.json").read_text(encoding="utf-8"))[name]


# ==============================================================================================
# chunking
# ==============================================================================================
def chunk_markdown(path: Path, doc_meta: dict) -> list[dict]:
    """Heading-aware chunks. The SECTION PATH travels with each chunk so a hit can be located."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    chunks, stack, buf, start = [], [], [], 0
    lines = text.splitlines()

    def flush(end):
        body = "\n".join(buf).strip()
        if len(body) < 40:
            return
        chunks.append({
            "id": f"{doc_meta['id']}#{'/'.join(stack) or 'body'}",
            "kind": "doc_section", "path": doc_meta["path"],
            "section_path": "/".join(stack), "text": body[:4000],
            "provenance": None, "validity": None, "superseded_by": doc_meta.get("superseded_by"),
            "entity_ids": doc_meta.get("entity_links", {}),
            "line_start": start + 1, "line_end": end,
            "record_kind": doc_meta.get("record_kind"),
            "status": doc_meta.get("status"),
        })

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush(i)
            depth = len(m.group(1))
            stack = stack[:depth - 1] + [m.group(2).strip()[:60]]
            buf, start = [], i
        else:
            buf.append(line)
    flush(len(lines))
    return chunks


def chunk_registry() -> list[dict]:
    """Each registry entity is its OWN chunk - the unit a question is actually about."""
    reg = load_cat("registry")
    out = []
    src = REPO / "research" / "00-system" / "registry"
    import yaml
    raw_facts = {f["id"]: f for f in
                 yaml.safe_load((src / "facts.yaml").read_text("utf-8"))["facts"]}
    raw_lev = {l["id"]: l for l in
               yaml.safe_load((src / "levers.yaml").read_text("utf-8"))["levers"]}
    for fid, f in reg["facts"].items():
        r = raw_facts.get(fid, {})
        body = " ".join(str(r.get(k, "")) for k in ("statement", "note", "instrument"))
        out.append({"id": fid, "kind": "FACT", "path": "research/00-system/registry/facts.yaml",
                    "section_path": fid, "text": body[:4000],
                    "provenance": f["provenance"], "validity": f["validity"],
                    "superseded_by": f["superseded_by"],
                    "entity_ids": {"facts": [fid]}, "record_kind": "registry", "status": None})
    for lid, lv in reg["levers"].items():
        r = raw_lev.get(lid, {})
        body = " ".join(str(r.get(k, "")) for k in ("claim", "falsifier", "outcome", "note"))
        out.append({"id": lid, "kind": "LEVER", "path": "research/00-system/registry/levers.yaml",
                    "section_path": lid, "text": body[:4000],
                    "provenance": "VERIFIED" if lv["status"] in ("killed", "closed") else None,
                    "validity": None, "superseded_by": None,
                    "entity_ids": {"levers": [lid], "facts": lv["closed_by"] + lv["supporting"]},
                    "record_kind": "registry", "status": lv["status"]})
    for eid, e in reg["experiments"].items():
        out.append({"id": eid, "kind": "EXP",
                    "path": "research/00-system/registry/experiments.yaml",
                    "section_path": eid,
                    "text": f"experiment {eid} status {e['status']} kernel {e['kernel']} "
                            f"spec {e['spec']} submission {e['submission']}",
                    "provenance": "MEASURED", "validity": None, "superseded_by": None,
                    "entity_ids": {"experiments": [eid], "facts": e["facts"]},
                    "record_kind": "registry", "status": e["status"]})
    for pid, pk in reg["packets"].items():
        out.append({"id": pid, "kind": "PKT",
                    "path": f"research/00-system/registry/packets/{pid}.yaml",
                    "section_path": pid,
                    "text": f"packet {pid} lock {pk['lock']} lever {pk['lever']} "
                            f"owner {pk['owner']}",
                    "provenance": None, "validity": None, "superseded_by": None,
                    "entity_ids": {"packets": [pid], "levers": [pk["lever"]] if pk["lever"] else []},
                    "record_kind": "registry", "status": pk["lock"]})
    return out


def chunk_roles() -> list[dict]:
    """The BASELINE ROLE MANIFEST, indexed as first-class entities.

    Omitting it was the reason "what is the operational base" - the single question this whole
    rejig exists to answer - returned a script docstring that happens to discuss the phrase. The
    authoritative answer must be IN the corpus, and ranked as VERIFIED, or the index sends the
    reader to prose about the answer instead of the answer.
    """
    import yaml
    src = REPO / "research" / "00-system" / "registry" / "baseline_roles.yaml"
    if not src.is_file():
        return []
    roles = yaml.safe_load(src.read_text("utf-8")) or {}
    out = []
    for role in ("leaderboard_champion", "operational_base", "scientific_control"):
        r = roles.get(role) or {}
        body = " ".join(f"{k} {v}" for k, v in r.items() if not isinstance(v, (dict, list)))
        rule = (roles.get("selection_rules") or {}).get(role, "")
        out.append({
            "id": f"ROLE-{role}", "kind": "ROLE",
            "path": "research/00-system/registry/baseline_roles.yaml",
            "section_path": role,
            "text": (f"{role} baseline role. {body} selection rule: {rule}")[:4000],
            "provenance": "VERIFIED", "validity": "VALID", "superseded_by": None,
            "entity_ids": {"experiments": [r["experiment"]] if r.get("experiment") else [],
                           "facts": [r["score_fact"]] if r.get("score_fact") else []},
            "record_kind": "registry", "status": role})
    return out


def chunk_artifacts() -> list[dict]:
    """Notebooks, scripts and tests as SEARCHABLE METADATA - never their bytes."""
    out = []
    for name, nb in load_cat("notebooks").items():
        d = nb.get("delta_from_parent") or {}
        out.append({
            "id": f"NOTEBOOK-{name}", "kind": "NOTEBOOK", "path": nb["notebook"] or nb["dir"],
            "section_path": name,
            "text": (f"notebook {name} status {nb['status']} roles {nb.get('roles')} "
                     f"spec {nb['producing_spec']} parent {nb['base_notebook']} "
                     f"experiments {nb['experiments']} "
                     f"features {' '.join(sorted(nb.get('environment', {})))} "
                     # the changed variable names are repeated: "which notebook introduced X" is
                     # a question about the DELTA, and one mention loses to any fact that
                     # discusses X at length.
                     f"changed_from_parent {' '.join(sorted((d.get('changed') or {}).keys()))} "
                     f"introduced {' '.join(sorted((d.get('changed') or {}).keys()))} "
                     f"{' '.join(sorted(d.get('only_in_child') or []))}"),
            "provenance": "MEASURED", "validity": None, "superseded_by": None,
            "entity_ids": {"experiments": nb["experiments"], "facts": nb["facts"]},
            "record_kind": "artifact", "status": nb["status"]})
    for path, s in load_cat("scripts").items():
        out.append({
            "id": f"SCRIPT-{path}", "kind": "SCRIPT", "path": path, "section_path": path,
            "text": (f"script {path} lifecycle {s['lifecycle']} {s['lifecycle_basis']} "
                     f"purpose {s.get('purpose') or ''} "
                     f"measures {s['measures_facts']} packets {s['owning_packets']} "
                     f"callers {len(s['callers'])} temp {s.get('reads_temp_paths')}"),
            "provenance": None, "validity": None, "superseded_by": None,
            "entity_ids": {"facts": s["measures_facts"], "packets": s["owning_packets"]},
            "record_kind": "artifact", "status": s["lifecycle"]})
    for path, t in load_cat("tests").items():
        out.append({
            "id": f"TEST-{path}", "kind": "TEST", "path": path, "section_path": path,
            "text": (f"test {path} class {t['test_class']} protects {t['protects_scripts']} "
                     f"blocks {t['blocking_scope']} nodes {t['n_nodes']} "
                     f"skip {t.get('skip_reasons')}"),
            "provenance": None, "validity": None, "superseded_by": None,
            "entity_ids": t["ids_mentioned"], "record_kind": "artifact", "status": t["state"]})
    return out


def build_corpus() -> list[dict]:
    chunks = chunk_roles() + chunk_registry() + chunk_artifacts()
    docs = load_cat("research_docs")
    for path, meta in docs.items():
        p = REPO / path
        if p.suffix.lower() in SKIP_SUFFIX or not p.is_file():
            continue
        chunks += chunk_markdown(p, {**meta, "id": f"DOC-{path}", "path": path})
    for p in sorted((REPO / "knowledge").rglob("*.md")) if (REPO / "knowledge").is_dir() else []:
        pass    # the knowledge layer is generated FROM this corpus; indexing it would be circular
    return chunks


# ==============================================================================================
# index + search
# ==============================================================================================
def build_index(chunks: list[dict]) -> dict:
    df: Counter = Counter()
    postings: dict[str, list] = defaultdict(list)
    lengths = []
    for i, c in enumerate(chunks):
        terms = tok(c["text"] + " " + c["id"] + " " + str(c.get("section_path", "")))
        tf = Counter(terms)
        lengths.append(len(terms) or 1)
        for t, n in tf.items():
            postings[t].append((i, n))
        df.update(tf.keys())
    return {"schema_version": SCHEMA_VERSION, "n": len(chunks),
            "avg_len": sum(lengths) / max(len(lengths), 1),
            "lengths": lengths, "df": dict(df),
            "postings": {t: v for t, v in postings.items()}}


def rank_multiplier(c: dict) -> float:
    m = TIER.get(c.get("provenance"), 0.7) * VALIDITY_PENALTY.get(c.get("validity"), 0.9)
    if c.get("superseded_by"):
        m *= 0.1
    return m


def search(idx: dict, chunks: list[dict], query: str, k: int = 8) -> list[dict]:
    N, avg = idx["n"], idx["avg_len"]
    k1, b = 1.4, 0.75
    scores: dict[int, float] = defaultdict(float)
    for t in tok(query):
        post = idx["postings"].get(t)
        if not post:
            continue
        n_t = idx["df"][t]
        idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        for i, f in post:
            dl = idx["lengths"][i]
            scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg))
    # A TITLE/ID MATCH IS EVIDENCE THE CHUNK IS *ABOUT* THE QUERY, not merely mentions it.
    # Without it `ROLE-scientific_control` outranked `ROLE-operational_base` for "what is the
    # operational base", because the scientific-control entry discusses the operational base at
    # length while the operational-base entry simply IS it.
    qterms = set(tok(query))
    def title_boost(c):
        own = set(tok(str(c.get("id", "")) + " " + str(c.get("section_path", ""))))
        overlap = len(qterms & own)
        return 1.0 + 0.6 * overlap

    ranked = sorted(((s * rank_multiplier(chunks[i]) * title_boost(chunks[i]), i)
                     for i, s in scores.items()), reverse=True)[:k]
    out = []
    for s, i in ranked:
        c = chunks[i]
        r = {"score": round(s, 3), "id": c["id"], "kind": c["kind"], "path": c["path"],
             "section": c.get("section_path"), "provenance": c.get("provenance"),
             "validity": c.get("validity"), "status": c.get("status"),
             "entity_ids": c.get("entity_ids"), "excerpt": c["text"][:280]}
        if c.get("superseded_by") or c.get("validity") in ("INVALID",):
            r["REFUSAL"] = (f"SUPERSEDED/INVALID - superseded_by={c.get('superseded_by')}. "
                            f"This may NOT be quoted as current state; read the successor.")
        out.append(r)
    return out


VALIDATION_QUERIES = [
    ("what is the operational base", "ROLE-operational_base"),
    ("which notebook introduced motion relink learned bonus", "NOTEBOOK-kaggle_p38_relink_bonus_b2"),
    ("which scripts can alter the final parent choice", None),
    ("which tests protect the champion notebook", None),
    ("which levers were killed by their own evidence", None),
    ("what consumes the FOCUS export", None),
    ("which files are unsafe to move while Gate B runs", None),
    ("provenance policy fold legitimacy one policy two guards", None),
    ("offline scoreable restriction general_v1", None),
    ("safe division geometry of the operational base", None),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    corpus_p, index_p = args.out / "corpus.json", args.out / "index.json"
    if args.build or not corpus_p.is_file():
        chunks = build_corpus()
        idx = build_index(chunks)
        args.out.mkdir(parents=True, exist_ok=True)
        corpus_p.write_text(json.dumps(chunks), encoding="utf-8")
        index_p.write_text(json.dumps(idx), encoding="utf-8")
        kinds = Counter(c["kind"] for c in chunks)
        print(f"  indexed {len(chunks)} chunks into {args.out.relative_to(REPO).as_posix()}/  "
              f"{dict(kinds)}")
    chunks = json.loads(corpus_p.read_text(encoding="utf-8"))
    idx = json.loads(index_p.read_text(encoding="utf-8"))
    idx["postings"] = {t: [tuple(x) for x in v] for t, v in idx["postings"].items()}

    if args.query:
        for r in search(idx, chunks, args.query, args.k):
            print(f"  [{r['score']:>7.2f}] {r['kind']:<9} {r['id']}")
            print(f"            {r['path']}  prov={r['provenance']} valid={r['validity']}")
            if "REFUSAL" in r:
                print(f"            !! {r['REFUSAL']}")
    if args.validate:
        bad = 0
        print("\n  VALIDATION QUERIES")
        for q, expect in VALIDATION_QUERIES:
            res = search(idx, chunks, q, 5)
            ids = [r["id"] for r in res]
            ok = (expect in ids) if expect else bool(res)
            bad += 0 if ok else 1
            print(f"   {'ok ' if ok else 'MISS'} {q!r}")
            print(f"        top: {ids[:3]}")
        print(f"\n  {len(VALIDATION_QUERIES) - bad}/{len(VALIDATION_QUERIES)} queries satisfied")
        if bad:
            return 1
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
