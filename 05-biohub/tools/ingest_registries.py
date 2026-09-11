#!/usr/bin/env python3
"""Load the prior campaigns' registries (facts.yaml, levers.yaml, submissions ledger)
into biohub_base.db so the new sprint starts from measured knowledge, not prose."""
import os, re, sqlite3, datetime, sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB   = os.path.join(ROOT, "db", "biohub_base.db")
MONO = os.path.join(ROOT, "repos", "Biohub-CellTracking-2026")
BX   = os.path.join(ROOT, "repos", "Biohub-X")
NOW  = datetime.date.today().isoformat()

# prior-campaign provenance -> our claim_type / confidence
PROV = {"VERIFIED":("observation","high"), "MEASURED":("observation","high"),
        "EXTERNAL":("claim","low"), "UNVERIFIED":("claim","low"),
        "SUPERSEDED":("observation","medium")}

def cx_():
    c = sqlite3.connect(DB); c.execute("PRAGMA foreign_keys=ON"); return c

def src_for(cx, path, note):
    rel = os.path.relpath(path, ROOT)
    cx.execute("INSERT OR IGNORE INTO source(url,sha256,method,fetched_at,note)"
               " VALUES(?,?,?,?,?)", (rel, "git:"+rel, "git", NOW, note))
    return cx.execute("SELECT id FROM source WHERE sha256=?", ("git:"+rel,)).fetchone()[0]

def load_facts(cx):
    p = os.path.join(MONO, "research/00-system/registry/facts.yaml")
    if not os.path.exists(p): print("  facts.yaml missing"); return 0
    sid = src_for(cx, p, "monolith facts registry")
    doc = yaml.safe_load(open(p))
    n = 0
    for f in (doc.get("facts") or []):
        fid = f.get("id");  stmt = f.get("statement","")
        if not fid: continue
        prov = (f.get("provenance") or "UNVERIFIED").upper()
        ctype, conf = PROV.get(prov, ("claim","low"))
        validity = (f.get("validity") or "VALID").upper()
        status = "active"
        if prov == "SUPERSEDED" or validity in ("INVALID",): status = "superseded"
        if validity == "SUSPECT": conf = "low"
        val = f.get("value")
        unit = f.get("unit","")
        value = f"{val} {unit}".strip() if val is not None else str(f.get("scope",""))
        quote = stmt
        if f.get("validity_reason"): quote += f"  [validity={validity}: {f['validity_reason']}]"
        cx.execute("INSERT OR REPLACE INTO fact"
                   "(topic,key,value,claim_type,confidence,source_id,quote,observed_at,status)"
                   " VALUES('prior-campaign',?,?,?,?,?,?,?,?)",
                   (f"{fid}: {stmt[:150]}", value, ctype, conf, sid, quote,
                    str(f.get("date") or NOW)[:10], status))
        n += 1
    print(f"  facts.yaml      -> {n} facts")
    return n

def load_levers(cx):
    p = os.path.join(MONO, "research/00-system/registry/levers.yaml")
    if not os.path.exists(p): print("  levers.yaml missing"); return 0
    doc = yaml.safe_load(open(p))
    n = 0
    for l in (doc.get("levers") or []):
        lid = l.get("id");  name = l.get("name") or l.get("statement") or ""
        if not lid: continue
        st = (l.get("status") or "").lower()
        outcome = "dead-end" if st in ("closed","killed","falsified","rejected") else \
                  ("win" if st in ("promoted","deployed","accepted") else "neutral")
        cx.execute("INSERT INTO experiment(repo_id,name,machine,approach,outcome,notes,source_ref)"
                   " SELECT id,?,?,?,?,?,? FROM repo WHERE name='Biohub-CellTracking-2026'",
                   (f"{lid} {name[:120]}", "hp-laptop",
                    str(l.get("mechanism") or l.get("hypothesis") or "")[:1000],
                    outcome,
                    f"status={st}; kill_condition={str(l.get('kill_condition') or l.get('falsifier') or '')[:400]}",
                    "levers.yaml:"+lid))
        n += 1
    print(f"  levers.yaml     -> {n} levers as experiments")
    return n

def load_failed(cx):
    """The graveyard table: closed levers with measured deltas."""
    p = os.path.join(MONO, "research/06-knowledge-system/failed-experiments.md")
    if not os.path.exists(p): return 0
    n = 0
    for line in open(p):
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*\*{0,2}([^|]+?)\*{0,2}\s*\|\s*([^|]*?)\s*\|\s*([^|]+?)\s*\|", line)
        if not m or m.group(1).strip() in ("lever","---"): continue
        lever, result, basis, why = (g.strip() for g in m.groups())
        cx.execute("INSERT INTO experiment(repo_id,name,machine,approach,outcome,notes,source_ref)"
                   " SELECT id,?,?,?,'dead-end',?,'failed-experiments.md' FROM repo WHERE name='Biohub-CellTracking-2026'",
                   (f"CLOSED: {lever}", "hp-laptop", why, f"result={result}; basis={basis}"))
        n += 1
    print(f"  failed-experiments -> {n} closed levers")
    return n

def load_submissions(cx):
    p = os.path.join(MONO, "research/07-outputs/submissions.md")
    if not os.path.exists(p): return 0
    txt = open(p).read()
    subs = set(re.findall(r"kaggle submission (\d{6,})", txt)) | set(re.findall(r"submission[_ ]?id[:=]\s*(\d{6,})", txt, re.I))
    scores = re.findall(r"(0\.9\d{2,4})", txt)
    sid = src_for(cx, p, "monolith submission ledger")
    cx.execute("INSERT OR REPLACE INTO fact"
               "(topic,key,value,claim_type,confidence,source_id,quote,observed_at,status)"
               " VALUES('prior-campaign','submission-ledger',?, 'observation','high',?,?,?,'active')",
               (f"{len(subs)} kaggle submission ids; score mentions {sorted(set(scores))[-6:]}",
                sid, txt[:400], NOW))
    print(f"  submissions.md  -> {len(subs)} submission ids referenced")
    return len(subs)

def main():
    cx = cx_()
    cx.execute("DELETE FROM experiment WHERE source_ref LIKE 'levers.yaml%' OR source_ref='failed-experiments.md'")
    load_facts(cx); load_levers(cx); load_failed(cx); load_submissions(cx)
    cx.commit()
    print(f"\n  total facts={cx.execute('SELECT COUNT(*) FROM fact').fetchone()[0]}"
          f"  experiments={cx.execute('SELECT COUNT(*) FROM experiment').fetchone()[0]}")
    cx.close()

if __name__ == "__main__":
    main()
