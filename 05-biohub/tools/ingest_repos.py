#!/usr/bin/env python3
"""
Clone/refresh the three private Biohub repos and load their history into biohub_base.db.

Needs GitHub auth first:  gh auth login        (or set GH_TOKEN / GITHUB_TOKEN)
Then:                     python3 tools/ingest_repos.py

Idempotent: re-run after every work session to keep the development base current.
"""
import json, os, re, sqlite3, subprocess, sys, datetime, hashlib

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB     = os.path.join(ROOT, "db", "biohub_base.db")
CHECKOUTS = os.path.join(ROOT, "repos")
OWNER  = "aryaarun0910-debug"
REPOS  = ["Biohub-Sprint-2026", "Biohub-CellTracking-2026", "Biohub-X"]
NOW    = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

LANG = {".py":"python",".ipynb":"notebook",".md":"markdown",".yaml":"yaml",".yml":"yaml",
        ".json":"json",".toml":"toml",".sh":"shell",".cpp":"cpp",".c":"c",".h":"c",
        ".txt":"text",".csv":"csv",".cfg":"config",".sql":"sql"}

# Files worth carrying forward into the new pipeline.
REUSABLE = re.compile(r"(train|model|dataset|loader|metric|eval|track|link|detect|unet|"
                      r"transformer|infer|predict|submit|post.?proc|config|util)", re.I)

def sh(args, cwd=None, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(f"{' '.join(args)}\n{r.stderr[:500]}")
    return r.stdout

def have_auth():
    if os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"): return True
    r = subprocess.run(["gh","auth","status"], capture_output=True, text=True)
    return r.returncode == 0

def clone_or_pull(name):
    dest = os.path.join(CHECKOUTS, name)
    if os.path.isdir(os.path.join(dest, ".git")):
        sh(["git","fetch","--all","--quiet"], cwd=dest, check=False)
        sh(["git","pull","--quiet","--ff-only"], cwd=dest, check=False)
    else:
        os.makedirs(CHECKOUTS, exist_ok=True)
        url = f"https://github.com/{OWNER}/{name}.git"
        r = subprocess.run(["gh","repo","clone",f"{OWNER}/{name}",dest],
                           capture_output=True, text=True)
        if r.returncode:
            r = subprocess.run(["git","clone","--quiet",url,dest], capture_output=True, text=True)
            if r.returncode: raise RuntimeError(f"clone {name}: {r.stderr[:400]}")
    return dest

def sha256(p):
    h=hashlib.sha256()
    try:
        with open(p,"rb") as f:
            for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    except OSError: return None
    return h.hexdigest()

def notebook_summary(path):
    """Pull the first markdown heading + any score-looking numbers out of a notebook."""
    try: nb=json.load(open(path))
    except Exception: return None
    head, scores = None, []
    for c in nb.get("cells",[]):
        srcs="".join(c.get("source",[]))
        if c.get("cell_type")=="markdown" and head is None:
            m=re.search(r"^#+\s*(.+)$", srcs, re.M)
            if m: head=m.group(1).strip()[:120]
        scores += re.findall(r"(?:score|cv|lb)\D{0,12}(0\.\d{3,5})", srcs, re.I)
    bits=[]
    if head: bits.append(head)
    if scores: bits.append("scores seen: "+", ".join(sorted(set(scores))[:8]))
    return " | ".join(bits) or None

def ingest(cx, name, path):
    cx.execute("INSERT OR IGNORE INTO repo(name,url,is_private,status) VALUES(?,?,1,'pending-auth')",
               (name, f"https://github.com/{OWNER}/{name}"))
    branch = sh(["git","rev-parse","--abbrev-ref","HEAD"], cwd=path).strip()
    cx.execute("UPDATE repo SET cloned_path=?, default_branch=?, last_ingest=?, status='ingested'"
               " WHERE name=?", (path, branch, NOW, name))
    rid = cx.execute("SELECT id FROM repo WHERE name=?", (name,)).fetchone()[0]

    # commits (with churn)
    log = sh(["git","log","--no-merges","--date=iso-strict",
              "--pretty=format:%H%x1f%ad%x1f%an%x1f%s%x1e","--shortstat"], cwd=path)
    nc = 0
    for chunk in log.split("\x1e"):
        if not chunk.strip(): continue
        parts = chunk.strip().split("\x1f")
        if len(parts) < 4: continue
        sha_, ad, an, rest = parts[0], parts[1], parts[2], parts[3]
        subj = rest.split("\n")[0]
        stat = rest
        f_ = re.search(r"(\d+) files? changed", stat)
        i_ = re.search(r"(\d+) insertions?", stat)
        d_ = re.search(r"(\d+) deletions?", stat)
        cx.execute("INSERT OR IGNORE INTO repo_commit"
                   "(repo_id,sha,authored,author,subject,files_changed,insertions,deletions)"
                   " VALUES(?,?,?,?,?,?,?,?)",
                   (rid, sha_.strip(), ad, an, subj,
                    int(f_.group(1)) if f_ else None,
                    int(i_.group(1)) if i_ else None,
                    int(d_.group(1)) if d_ else None))
        nc += 1

    # files
    nf = 0
    for rel in sh(["git","ls-files"], cwd=path).splitlines():
        ap = os.path.join(path, rel)
        if not os.path.isfile(ap): continue
        ext = os.path.splitext(rel)[1].lower()
        size = os.path.getsize(ap)
        last = sh(["git","log","-1","--pretty=%H","--",rel], cwd=path, check=False).strip() or None
        summ = notebook_summary(ap) if ext==".ipynb" and size < 40_000_000 else None
        cx.execute("INSERT OR REPLACE INTO repo_file"
                   "(repo_id,path,bytes,lang,sha256,last_commit,summary,reusable)"
                   " VALUES(?,?,?,?,?,?,?,?)",
                   (rid, rel, size, LANG.get(ext,ext.lstrip(".") or "none"),
                    sha256(ap) if size < 50_000_000 else None, last, summ,
                    1 if REUSABLE.search(rel) else 0))
        nf += 1
    print(f"  {name:<26} branch={branch:<12} commits={nc:<5} files={nf}")
    return nc, nf

def main():
    if not have_auth():
        print("NO GITHUB AUTH.\n"
              "  All three repos are private; nothing can be cloned yet.\n"
              "  Run:  gh auth login       (or export GH_TOKEN=...)\n"
              "  Then: python3 tools/ingest_repos.py", file=sys.stderr)
        return 2
    cx = sqlite3.connect(DB)
    for name in REPOS:
        try:
            ingest(cx, name, clone_or_pull(name))
        except Exception as e:
            print(f"  {name}: FAILED - {e}", file=sys.stderr)
            cx.execute("UPDATE repo SET status='error', last_ingest=? WHERE name=?", (NOW, name))
    cx.commit()
    print(f"\ncommits={cx.execute('SELECT COUNT(*) FROM repo_commit').fetchone()[0]}"
          f"  files={cx.execute('SELECT COUNT(*) FROM repo_file').fetchone()[0]}"
          f"  reusable={cx.execute('SELECT COUNT(*) FROM repo_file WHERE reusable=1').fetchone()[0]}")
    cx.close(); return 0

if __name__ == "__main__":
    sys.exit(main())
