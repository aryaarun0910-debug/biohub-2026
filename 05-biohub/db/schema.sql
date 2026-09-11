-- Biohub Cell Tracking 2026 :: development base
-- One DB that holds everything needed to resume work without starting from scratch.
PRAGMA journal_mode=WAL;

-- ---------- provenance ----------
CREATE TABLE IF NOT EXISTS source (
  id            INTEGER PRIMARY KEY,
  url           TEXT,
  sha256        TEXT UNIQUE,          -- frozen artifact in evidence/
  method        TEXT NOT NULL,        -- curl | rpc | git | measured | user-stated | unavailable
  http_status   INTEGER,
  fetched_at    TEXT NOT NULL,
  note          TEXT
);

-- Every durable fact carries a claim_type + a source. Mirrors the /research skill schema.
CREATE TABLE IF NOT EXISTS fact (
  id          INTEGER PRIMARY KEY,
  topic       TEXT NOT NULL,          -- 'metric','data','rules','timeline','leaderboard','hardware',...
  key         TEXT NOT NULL,
  value       TEXT NOT NULL,
  claim_type  TEXT NOT NULL CHECK(claim_type IN ('observation','claim','inference','decision','profile')),
  confidence  TEXT CHECK(confidence IN ('high','medium','low')),
  source_id   INTEGER REFERENCES source(id),
  quote       TEXT,                   -- verbatim span supporting the claim
  observed_at TEXT NOT NULL,
  review_after TEXT,
  status      TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','retracted')),
  superseded_by INTEGER REFERENCES fact(id),
  UNIQUE(topic,key,observed_at)
);

-- ---------- competition intel ----------
CREATE TABLE IF NOT EXISTS lb_snapshot (
  id         INTEGER PRIMARY KEY,
  taken_at   TEXT NOT NULL,
  rank       INTEGER NOT NULL,
  team_id    INTEGER,
  team_name  TEXT,
  score      REAL,
  n_subs     INTEGER,
  n_members  INTEGER,
  leader_tier TEXT,
  last_sub   TEXT,
  UNIQUE(taken_at, rank)
);

CREATE TABLE IF NOT EXISTS forum_topic (
  id         INTEGER PRIMARY KEY,     -- kaggle topic id
  title      TEXT NOT NULL,
  votes      INTEGER,
  comments   INTEGER,
  is_host    INTEGER DEFAULT 0,
  url        TEXT,
  harvested_at TEXT,
  relevance  TEXT,                    -- our triage: critical | high | medium | low | noise
  takeaway   TEXT                     -- what we concluded from it
);

CREATE TABLE IF NOT EXISTS public_kernel (
  id          INTEGER PRIMARY KEY,    -- kaggle kernel id
  title       TEXT NOT NULL,
  author      TEXT,
  votes       INTEGER,
  best_score  REAL,
  runtime_s   INTEGER,
  url         TEXT,
  harvested_at TEXT,
  family      TEXT,                   -- 'metric-hack' | 'unet-ilp' | 'rule-based' | 'gnn' | ...
  note        TEXT
);

-- ---------- development base (from the three repos) ----------
CREATE TABLE IF NOT EXISTS repo (
  id          INTEGER PRIMARY KEY,
  name        TEXT UNIQUE NOT NULL,
  url         TEXT,
  is_private  INTEGER,
  default_branch TEXT,
  cloned_path TEXT,
  last_ingest TEXT,
  status      TEXT                    -- 'pending-auth' | 'ingested' | 'error'
);

CREATE TABLE IF NOT EXISTS repo_commit (
  id        INTEGER PRIMARY KEY,
  repo_id   INTEGER NOT NULL REFERENCES repo(id),
  sha       TEXT NOT NULL,
  authored  TEXT,
  author    TEXT,
  subject   TEXT,
  files_changed INTEGER,
  insertions INTEGER,
  deletions  INTEGER,
  UNIQUE(repo_id, sha)
);

CREATE TABLE IF NOT EXISTS repo_file (
  id        INTEGER PRIMARY KEY,
  repo_id   INTEGER NOT NULL REFERENCES repo(id),
  path      TEXT NOT NULL,
  bytes     INTEGER,
  lang      TEXT,
  sha256    TEXT,
  last_commit TEXT,
  summary   TEXT,                     -- what this file does (filled on ingest/triage)
  reusable  INTEGER DEFAULT 0,        -- flag: carry forward to the new pipeline?
  UNIQUE(repo_id, path)
);

-- Anything that was already tried, so we never re-run a dead end.
CREATE TABLE IF NOT EXISTS experiment (
  id          INTEGER PRIMARY KEY,
  repo_id     INTEGER REFERENCES repo(id),
  name        TEXT NOT NULL,
  started_at  TEXT,
  machine     TEXT,                   -- 'hp-laptop' | 'macbook-m5pro' | 'kaggle'
  approach    TEXT,                   -- detector/linker/post-proc description
  config      TEXT,                   -- JSON blob of hyperparameters
  cv_score    REAL,
  lb_public   REAL,
  lb_private  REAL,
  runtime_min REAL,
  outcome     TEXT,                   -- 'win' | 'neutral' | 'dead-end'
  notes       TEXT,
  source_ref  TEXT                    -- commit sha / notebook version / path
);

CREATE TABLE IF NOT EXISTS artifact (
  id        INTEGER PRIMARY KEY,
  kind      TEXT NOT NULL,            -- 'weights' | 'dataset' | 'submission' | 'cache'
  name      TEXT NOT NULL,
  path      TEXT,
  bytes     INTEGER,
  sha256    TEXT,
  produced_by INTEGER REFERENCES experiment(id),
  kaggle_ref TEXT,                    -- kaggle dataset/model slug once uploaded
  created_at TEXT
);

-- Open questions / decisions, so context survives the machine switch.
CREATE TABLE IF NOT EXISTS decision (
  id         INTEGER PRIMARY KEY,
  made_at    TEXT,
  question   TEXT NOT NULL,
  choice     TEXT,
  rationale  TEXT,
  status     TEXT DEFAULT 'open'      -- open | decided | revisit
);

CREATE INDEX IF NOT EXISTS ix_fact_topic ON fact(topic,status);
CREATE INDEX IF NOT EXISTS ix_lb_score ON lb_snapshot(taken_at,score);
CREATE INDEX IF NOT EXISTS ix_kernel_score ON public_kernel(best_score);
CREATE INDEX IF NOT EXISTS ix_expt_outcome ON experiment(outcome);
