#!/usr/bin/env python3
"""Populate biohub_base.db from frozen Kaggle evidence. Idempotent: re-run any time."""
import json, sqlite3, hashlib, glob, os, re, datetime, sys
from pathlib import Path
from evidence import verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB   = os.path.join(ROOT, "db", "biohub_base.db")
EV   = os.path.join(ROOT, "evidence", "raw")
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
RPC  = "https://www.kaggle.com/api/i/"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

MANIFEST = {}

def observed(path):
    return MANIFEST[Path(path).resolve()]["fetched_at"]

def src(cx, path, url, method="rpc", note=None):
    """Register a frozen artifact, return its source id."""
    s = sha(path)
    cx.execute("INSERT OR IGNORE INTO source(url,sha256,method,http_status,fetched_at,note)"
               " VALUES(?,?,?,?,?,?)", (url, s, method, 200, observed(path), note))
    return cx.execute("SELECT id FROM source WHERE sha256=?", (s,)).fetchone()[0]

def fact(cx, topic, key, value, ctype, conf, sid, quote=None, review=None):
    cx.execute("INSERT INTO fact"
               "(topic,key,value,claim_type,confidence,source_id,quote,observed_at,review_after,status)"
               " VALUES(?,?,?,?,?,?,?,?,?, 'active')"
               " ON CONFLICT(topic,key,observed_at) DO UPDATE SET value=excluded.value,"
               " claim_type=excluded.claim_type,confidence=excluded.confidence,"
               " source_id=excluded.source_id,quote=excluded.quote,review_after=excluded.review_after",
               (topic, key, str(value), ctype, conf, sid, quote, cx.execute("SELECT fetched_at FROM source WHERE id=?", (sid,)).fetchone()[0][:10], review))

def classify_kernel(title, score, runtime_s):
    t = title.lower()
    if "hack" in t: return "metric-hack"
    if "rule" in t and "base" in t: return "rule-based"
    if "gnn" in t or "graph" in t: return "graph-learned"
    if "ilp" in t: return "unet-ilp"
    if "unet" in t or "u-net" in t: return "unet"
    # fast + high score is the hack signature even when unlabelled
    if score and runtime_s and score >= 0.949 and runtime_s <= 15 * 60: return "metric-hack?"
    return "other"

CRITICAL = {
    727154: ("critical", "HOST patched a division-metric 'weakly connected component' exploit on 2026-07-18 and rescored. Precedent: the host WILL patch exploits mid-competition."),
    728324: ("critical", "Rescore was executed after the metric patch - scores moved."),
    732103: ("high", "Community-shared 18.5GB synthetic labelled 3D microscopy with 165k divisions. External data is rules-legal; divisions are the scarce signal."),
    734330: ("high", "Zebrahub (royerlab public zebrafish data) as external training data - check host answer."),
    716952: ("high", "A rule-based, no-learning method reached 7th/344 - classical tracking is competitive."),
    724283: ("high", "Ground-truth tracks contain jumps; GT is noisy."),
    729053: ("high", "Not all sparse GT edges are correct."),
    733973: ("high", "Linking radius ~8.4um; divisions are ~1 link in 853 - severe class imbalance."),
    734053: ("high", "Voxels are 4:1 anisotropic in Z - must be handled in the network and in distance metrics."),
    716793: ("high", "Only 2 embryo groups in train; test is embryo-disjoint -> generalisation is the core risk."),
    733877: ("high", "A one-to-one linker scores 0.000 on divisions - divisions need explicit modelling."),
    739018: ("high", "Node-count adjustment lets adj_edge_jaccard exceed 1.0 - the basis of the 'metric hack'."),
    728300: ("high", "You can score on train locally; clean predictions can exceed 1.0."),
    735352: ("high", "Shakeup expected: public LB is only 29% of test."),
    724917: ("medium", "Scoring timeouts driven by graph connectivity, not size - keep the predicted graph clean."),
    732345: ("medium", "Baseline training ~2h/epoch at batch 16 - our compute reference point."),
}

def main():
    global MANIFEST
    MANIFEST = verify(Path(ROOT) / "evidence")
    cx = sqlite3.connect(DB); cx.execute("PRAGMA foreign_keys=ON")

    # ---- competition metadata ----
    p = os.path.join(EV, "comp.json")
    sid = src(cx, p, RPC + "competitions.CompetitionService/GetCompetition")
    c = json.load(open(p))
    keep = ["deadline","teamMergerExplicitDeadline","prohibitNewEntrantsExplicitDeadline",
            "maxDailySubmissions","numScoredSubmissions","maxTeamSize","leaderboardPercentage",
            "onlyAllowKernelSubmissions","usesSynchronousReruns","maxGpuRuntimeMinutes",
            "maxCpuRuntimeMinutes","requiredSubmissionFilename","totalTeams","totalCompetitors",
            "totalSubmissions","scoreTruncationNumDecimals","requiresIdentityVerification","numPrizes"]
    for k in keep:
        if k in c:
            fact(cx,"competition",k,c[k],"claim","high",sid,
                 quote=f'"{k}":{json.dumps(c[k])}', review="2026-09-29")
    fact(cx,"competition","reward",json.dumps(c.get("reward")),"claim","high",sid,
         quote='"reward":' + json.dumps(c.get("reward")))

    # ---- pages (rules/description/evaluation/data/timeline/prizes/code-reqs) ----
    for md in sorted(glob.glob(os.path.join(EV,"pages","*.md"))):
        name = os.path.basename(md)[:-3]
        s2 = src(cx, md, "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development",
                 note=f"page:{name}")
        body = open(md).read()
        fact(cx,"page",name,f"{len(body)} chars @ {os.path.relpath(md, ROOT)}","claim","high",s2, quote=body[:400])

    # ---- leaderboard snapshot ----
    p = os.path.join(EV,"lb.json")
    sid = src(cx,p, RPC+"competitions.LeaderboardService/GetLeaderboard")
    lb = json.load(open(p)); teams = {t["teamId"]: t for t in lb.get("teams",[])}
    n=0
    for i, r in enumerate(lb.get("publicLeaderboard",[]), start=1):
        t = teams.get(r.get("teamId"), {})
        tl = (t.get("teamUpInfo") or {}).get("teamLeader") or {}
        cx.execute("INSERT INTO lb_snapshot"
                   "(taken_at,rank,team_id,team_name,score,n_subs,n_members,leader_tier,last_sub)"
                   " VALUES(?,?,?,?,?,?,?,?,?)"
                   " ON CONFLICT(taken_at,rank) DO UPDATE SET team_id=excluded.team_id,team_name=excluded.team_name,"
                   " score=excluded.score,n_subs=excluded.n_subs,n_members=excluded.n_members,"
                   " leader_tier=excluded.leader_tier,last_sub=excluded.last_sub",
                   (observed(p), r.get("rank", i), r.get("teamId"), t.get("teamName"),
                    float(r["displayScore"]) if r.get("displayScore") else None,
                    t.get("submissionCount"), len(t.get("teamMembers") or [])+1,
                    tl.get("tier"), (t.get("lastSubmissionDate") or "")[:10]))
        n+=1
    print(f"  leaderboard rows: {n}")

    # ---- forum topics ----
    tot=0
    for f in sorted(glob.glob(os.path.join(EV,"topics_p*.json"))):
        s2 = src(cx,f, RPC+"discussions.DiscussionsService/GetTopicListByForumId")
        for t in json.load(open(f)).get("topics",[]):
            rel,take = CRITICAL.get(t["id"], (None,None))
            cx.execute("INSERT INTO forum_topic"
                       "(id,title,votes,comments,is_host,url,harvested_at,relevance,takeaway)"
                       " VALUES(?,?,?,?,?,?,?,?,?)"
                       " ON CONFLICT(id) DO UPDATE SET title=excluded.title,votes=excluded.votes,"
                       " comments=excluded.comments,is_host=excluded.is_host,url=excluded.url,"
                       " harvested_at=excluded.harvested_at,"
                       " relevance=COALESCE(forum_topic.relevance,excluded.relevance),"
                       " takeaway=COALESCE(forum_topic.takeaway,excluded.takeaway)",
                       (t["id"], t.get("title"), t.get("votes"), t.get("commentCount"),
                        1 if t.get("authorType")=="HOST" else 0,
                        "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/%d"%t["id"],
                        observed(f), rel, take))
            tot+=1
    print(f"  forum topics: {cx.execute('SELECT COUNT(*) FROM forum_topic').fetchone()[0]} unique (from {tot} rows)")

    # ---- public kernels ----
    for f in sorted(glob.glob(os.path.join(EV,"kernels_p*.json"))):
        s2 = src(cx,f, RPC+"kernels.KernelsService/ListKernels")
        for k in json.load(open(f)).get("kernels",[]):
            score = float(k["bestPublicScore"]) if k.get("bestPublicScore") else None
            rt = int(k["lastRunExecutionTimeSeconds"]) if k.get("lastRunExecutionTimeSeconds") else None
            cx.execute("INSERT INTO public_kernel"
                       "(id,title,author,votes,best_score,runtime_s,url,harvested_at,family)"
                       " VALUES(?,?,?,?,?,?,?,?,?)"
                       " ON CONFLICT(id) DO UPDATE SET title=excluded.title,author=excluded.author,"
                       " votes=excluded.votes,best_score=excluded.best_score,runtime_s=excluded.runtime_s,"
                       " url=excluded.url,harvested_at=excluded.harvested_at,"
                       " family=COALESCE(public_kernel.family,excluded.family)",
                       (k["id"], k.get("title"), (k.get("author") or {}).get("userName"),
                        k.get("totalVotes"), score, rt,
                        "https://www.kaggle.com"+(k.get("scriptUrl") or ""), observed(f),
                        classify_kernel(k.get("title",""), score, rt)))
    print(f"  public kernels: {cx.execute('SELECT COUNT(*) FROM public_kernel').fetchone()[0]}")

    # ---- the three repos (placeholders until auth exists) ----
    for nm in ["Biohub-Sprint-2026","Biohub-CellTracking-2026","Biohub-X"]:
        cx.execute("INSERT OR IGNORE INTO repo(name,url,is_private,status)"
                   " VALUES(?,?,1,'pending-auth')",
                   (nm, f"https://github.com/aryaarun0910-debug/{nm}"))
    cx.commit()
    print(f"  sources: {cx.execute('SELECT COUNT(*) FROM source').fetchone()[0]}"
          f" | facts: {cx.execute('SELECT COUNT(*) FROM fact').fetchone()[0]}")
    cx.close()

if __name__ == "__main__":
    main()
