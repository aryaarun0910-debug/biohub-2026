#!/usr/bin/env python3
"""Quick views over the development base.  Usage: python3 tools/ask.py <view> [arg]"""
import sqlite3, sys, os, textwrap
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "biohub_base.db")
cx = sqlite3.connect(DB); cx.row_factory = sqlite3.Row
def q(sql, *a): return cx.execute(sql, a).fetchall()
def table(rows, cols=None):
    if not rows: print("  (none)"); return
    cols = cols or rows[0].keys()
    w = {c: max(len(str(c)), max(len(str(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(str(c).ljust(w[c]) for c in cols))
    print("  " + "  ".join("-"*w[c] for c in cols))
    for r in rows: print("  " + "  ".join(str(r[c]).ljust(w[c]) for c in cols))

V = {}
def view(fn): V[fn.__name__] = fn; return fn

@view
def status(_=None):
    "One-screen situation report."
    f = dict((r["key"], r["value"]) for r in q("SELECT key,value FROM fact WHERE topic='competition'"))
    import datetime
    dl = f.get("deadline","")[:10]
    left = (datetime.date.fromisoformat(dl) - datetime.date.today()).days if dl else "?"
    print(f"\n  DEADLINE {dl}  ({left} days left)   entry/merge cutoff {f.get('prohibitNewEntrantsExplicitDeadline','')[:10]}")
    print(f"  kernels-only={f.get('onlyAllowKernelSubmissions')}  gpu_limit={f.get('maxGpuRuntimeMinutes')}min"
          f"  subs/day={f.get('maxDailySubmissions')}  final_subs={f.get('numScoredSubmissions')}"
          f"  public_LB={f.get('leaderboardPercentage')}%")
    print(f"  teams={f.get('totalTeams')}  prizes={f.get('numPrizes')}  score_decimals={f.get('scoreTruncationNumDecimals')}\n")
    print("  Leaderboard shape:")
    for k in (1,3,7,10,25,50,100,250,500,1000):
        r = q("SELECT score FROM lb_snapshot WHERE rank=? ORDER BY taken_at DESC LIMIT 1", k)
        if r: print(f"    rank {k:>5}: {r[0]['score']}")
    print("\n  Repos:"); table(q("SELECT name,status,last_ingest FROM repo"))

@view
def gap(_=None):
    "What separates the plateau from the podium."
    print("\n  Public-notebook families by best score vs runtime:")
    table(q("""SELECT family, COUNT(*) n, ROUND(MAX(best_score),4) best,
                      ROUND(AVG(runtime_s)/60.0,0) avg_min
               FROM public_kernel WHERE best_score IS NOT NULL
               GROUP BY family ORDER BY best DESC"""))
    print("\n  Teams at/above each score:")
    for t in (0.970,0.966,0.960,0.958,0.952,0.948,0.947):
        n = q("SELECT COUNT(*) c FROM lb_snapshot WHERE score>=? AND taken_at=(SELECT MAX(taken_at) FROM lb_snapshot)", t-1e-9)[0]["c"]
        print(f"    >= {t:.3f}: {n:>5} teams")

@view
def topics(arg=None):
    "Triaged discussion threads (arg: relevance filter)."
    sql = "SELECT relevance,votes,comments,is_host,id,title FROM forum_topic WHERE relevance IS NOT NULL"
    a=[]
    if arg: sql += " AND relevance=?"; a=[arg]
    for r in q(sql + " ORDER BY CASE relevance WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, votes DESC", *a):
        print(f"\n  [{r['relevance'].upper()}] {r['votes']}v {'HOST ' if r['is_host'] else ''}#{r['id']} {r['title']}")
        tk = q("SELECT takeaway FROM forum_topic WHERE id=?", r["id"])[0]["takeaway"]
        if tk: print(textwrap.fill(tk, 96, initial_indent="      -> ", subsequent_indent="         "))

@view
def kernels(arg=None):
    "Top public notebooks by score (arg: family)."
    sql="SELECT ROUND(best_score,4) score, votes, ROUND(runtime_s/60.0,0) min, family, title, url FROM public_kernel WHERE best_score IS NOT NULL"
    a=[]
    if arg: sql+=" AND family=?"; a=[arg]
    table(q(sql+" ORDER BY best_score DESC, votes DESC LIMIT 25", *a),
          ["score","votes","min","family","title"])

@view
def facts(arg=None):
    "Recorded facts (arg: topic)."
    sql="SELECT topic,key,value,claim_type,confidence FROM fact WHERE status='active'"
    a=[]
    if arg: sql+=" AND topic=?"; a=[arg]
    table(q(sql+" ORDER BY topic,key", *a))

@view
def expts(_=None):
    "Experiment log - what has already been tried."
    table(q("""SELECT e.name, e.machine, e.approach, e.cv_score, e.lb_public, e.outcome
               FROM experiment e ORDER BY e.lb_public DESC NULLS LAST"""))

@view
def todo(_=None):
    "Open decisions."
    table(q("SELECT id,question,choice,status FROM decision ORDER BY status,id"))

if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv)>1 else "status"
    arg = sys.argv[2] if len(sys.argv)>2 else None
    if v not in V:
        print("views:"); [print(f"  {k:<10} {f.__doc__}") for k,f in V.items()]; sys.exit(1)
    V[v](arg)
