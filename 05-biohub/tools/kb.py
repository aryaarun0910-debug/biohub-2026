#!/usr/bin/env python3
"""kb -- the one way to read and write the knowledge base.

WHY THIS EXISTS. Facts were being written with hand-rolled SQL at every call site. In a single
session that cost four silently rolled-back transactions: a missing bind parameter (8 placeholders,
7 values), a `validity` value that failed a CHECK constraint, a `repo_id` column that does not
exist on `decision`, and a `superseded_by` given as a key when the column takes a fact id. Each
one printed a traceback AFTER the work was done and lost it. A schema you must remember is a
schema that will be got wrong.

    kb.py record  --topic division --key foo --value "..." --quote "..." --source tools/x.py
    kb.py supersede --key old_key --by new_key --why "..."
    kb.py ask division            # active facts on a topic, newest first
    kb.py open                    # open decisions
    kb.py check                   # provenance audit: every fact must have a source and a quote
"""
from __future__ import annotations
import argparse, re, sqlite3, subprocess, sys, textwrap
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "db" / "biohub_base.db"
def _allowed(col: str) -> tuple:
    """Read the CHECK constraint out of the schema instead of remembering it.

    The first version of this file hardcoded the claim_type values from memory and got them
    wrong -- 'external' and 'kaggle' are not in the constraint -- which is precisely the failure
    kb.py exists to prevent. A vocabulary the code guesses is a vocabulary it will guess wrong.
    """
    sql = sqlite3.connect(DB).execute(
        "select sql from sqlite_master where type='table' and name='fact'").fetchone()[0]
    m = re.search(rf"{col}\s+[A-Za-z]*\s*(?:NOT NULL\s*)?CHECK\s*\(\s*{col}\s+IN\s*\(([^)]*)\)",
                  sql, re.I)
    if not m:
        m = re.search(rf"CHECK\s*\(\s*{col}\s+IN\s*\(([^)]*)\)", sql, re.I)
    return tuple(v.strip().strip("'\"") for v in m.group(1).split(",")) if m else ()


def conn():
    c = sqlite3.connect(DB); c.execute("PRAGMA foreign_keys=ON"); return c


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=DB.parent.parent, text=True).strip()
    except Exception:
        return ""


def record(topic, key, value, quote, source, claim="observation", conf="high",
           observed="", note="") -> int:
    """Insert one fact with its provenance. Raises BEFORE writing if anything is missing."""
    claims, confs = _allowed("claim_type"), _allowed("confidence")
    if claims and claim not in claims:
        raise SystemExit(f"claim_type must be one of {claims}")
    if confs and conf not in confs:
        raise SystemExit(f"confidence must be one of {confs}")
    if not quote.strip():  raise SystemExit("a fact needs a verbatim quote -- that is the rule")
    if not source.strip(): raise SystemExit("a fact needs a retrievable source")
    observed = observed or __import__("datetime").date.today().isoformat()
    db = conn(); c = db.cursor()
    sha = _git_sha()
    c.execute("insert into source(url,method,fetched_at,note) values(?,?,?,?)",
              (source, "measured", observed, (note or "") + (f" @ {sha}" if sha else "")))
    sid = c.lastrowid
    c.execute("""insert into fact(topic,key,value,claim_type,confidence,source_id,quote,
                                  observed_at,status,validity)
                 values(?,?,?,?,?,?,?,?,'active','VALID')""",
              (topic, key, value, claim, conf, sid, quote, observed))
    fid = c.lastrowid
    db.commit()                                   # commit BEFORE printing, never after
    print(f"  fact {fid} recorded  [{topic}/{key}]  source {sid}" + (f"  @ {sha}" if sha else ""))
    return fid


def supersede(old_key, by_key, why):
    db = conn(); c = db.cursor()
    row = c.execute("select id from fact where key=? and status='active'", (by_key,)).fetchone()
    if not row: raise SystemExit(f"no active fact with key '{by_key}' to supersede by")
    n = c.execute("""update fact set status='superseded', superseded_by=?,
                     key = key || ' (SUPERSEDED: ' || ? || ')'
                     where key=? and status='active'""", (row[0], why, old_key)).rowcount
    db.commit()
    print(f"  superseded {n} fact(s) matching '{old_key}' -> fact {row[0]}")


def ask(topic, limit=12):
    db = conn()
    q = """select f.id,f.key,f.value,f.claim_type,f.confidence,f.observed_at,s.url
           from fact f left join source s on s.id=f.source_id
           where f.status='active' and (f.topic=? or f.key like ?)
           order by f.id desc limit ?"""
    rows = db.execute(q, (topic, f"%{topic}%", limit)).fetchall()
    if not rows: print(f"  nothing active for '{topic}'"); return
    for r in rows:
        print(f"\n  [{r[0]}] {r[1]}")
        print(f"       {r[3]}/{r[4]}  {r[5]}  src={r[6]}")
        print(textwrap.fill(r[2] or "", 96, initial_indent="       ", subsequent_indent="       "))


def open_decisions():
    db = conn()
    rows = db.execute("select id,made_at,question,choice,rationale from decision "
                      "where status='open' order by id").fetchall()
    print(f"  {len(rows)} OPEN decision(s)")
    for r in rows:
        print(f"\n  [{r[0]}] {r[1]}  {r[2]}")
        print(textwrap.fill(f"CHOICE: {r[3]}", 96, initial_indent="       ", subsequent_indent="       "))
        print(textwrap.fill(r[4] or "", 96, initial_indent="       ", subsequent_indent="       "))


def check():
    db = conn(); bad = 0
    # A fact written since the rule was enforced MUST carry a source and a quote. The pre-rule
    # facts are quarantined at confidence=low / validity=UNKNOWN instead of being given invented
    # provenance -- they are reported, but they do not fail the check.
    for label, q in [
        ("UNSOURCED but still high-conf",
         "select count(*) from fact where status='active' and confidence!='low' "
         "and (source_id is null or quote is null or trim(quote)='')"),
        ("superseded with no successor",
         "select count(*) from fact where status='superseded' and superseded_by is null"),
        ("superseded_by pointing nowhere",
         "select count(*) from fact f where f.superseded_by is not null "
         "and not exists(select 1 from fact g where g.id=f.superseded_by)"),
    ]:
        n = db.execute(q).fetchone()[0]
        print(f"  {label:<34} {n}")
        bad += n
    q = db.execute("select count(*) from fact where status='active' and confidence='low' "
                   "and (source_id is null or quote is null or trim(quote)='')").fetchone()[0]
    print(f"  {'quarantined pre-rule hints':<34} {q}   (reported, not a failure)")
    tot = db.execute("select count(*) from fact where status='active'").fetchone()[0]
    print(f"  {'active facts':<34} {tot}")
    print("\n  PROVENANCE OK" if bad == 0 else f"\n  {bad} PROBLEM(S)")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    for f in ("topic", "key", "value", "quote", "source"): r.add_argument(f"--{f}", required=True)
    r.add_argument("--claim", default="observation"); r.add_argument("--conf", default="high")
    r.add_argument("--note", default="")
    s = sub.add_parser("supersede")
    s.add_argument("--key", required=True); s.add_argument("--by", required=True)
    s.add_argument("--why", required=True)
    a = sub.add_parser("ask"); a.add_argument("topic"); a.add_argument("--limit", type=int, default=12)
    sub.add_parser("open"); sub.add_parser("check")
    n = ap.parse_args()
    if   n.cmd == "record":    record(n.topic, n.key, n.value, n.quote, n.source, n.claim, n.conf, note=n.note)
    elif n.cmd == "supersede": supersede(n.key, n.by, n.why)
    elif n.cmd == "ask":       ask(n.topic, n.limit)
    elif n.cmd == "open":      open_decisions()
    elif n.cmd == "check":     sys.exit(1 if check() else 0)
