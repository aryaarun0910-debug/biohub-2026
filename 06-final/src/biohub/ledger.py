"""Experiment ledger. Every run writes an immutable record.

A run that emits one number teaches almost nothing at ~50 total submissions.
Each run records the full metric decomposition plus, for divisions, a per-event
cause of death -- so "why did we miss it" is read automatically rather than
measured by hand on seven events.
"""
from __future__ import annotations

import hashlib, json, sqlite3, time
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "artifacts" / "ledger" / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  ts         REAL NOT NULL,
  label      TEXT NOT NULL,
  kind       TEXT NOT NULL,              -- local | kaggle
  config     TEXT NOT NULL,              -- full config JSON
  config_sha TEXT NOT NULL,
  code_sha   TEXT,                       -- hash of src/ tree
  weights    TEXT,                       -- {name: sha256}
  wall_s     REAL,
  stage_s    TEXT,                       -- {stage: seconds} -- the 12h clock
  notes      TEXT
);

-- one row per film. Never pool folds into a single number.
CREATE TABLE IF NOT EXISTS scores (
  run_id TEXT NOT NULL, fold TEXT NOT NULL, film TEXT NOT NULL,
  J_edge REAL, adj_J_edge REAL, multiplier REAL,
  n_pred INTEGER, n_est REAL,
  edge_tp INTEGER, edge_fp INTEGER, edge_fn INTEGER,
  div_tp INTEGER, div_fp INTEGER, div_fn INTEGER,
  weight REAL,
  PRIMARY KEY (run_id, film)
);

-- the attribution table: why each ground-truth division was missed.
CREATE TABLE IF NOT EXISTS division_events (
  run_id TEXT NOT NULL, film TEXT NOT NULL, gt_parent INTEGER NOT NULL,
  outcome TEXT NOT NULL,                 -- TP | FN
  cause   TEXT,                          -- detector | gate | assignment | veto | unknown
  parent_matched INTEGER, d1_matched INTEGER, d2_matched INTEGER,
  arc1_present INTEGER, arc2_present INTEGER,
  d_parent_um REAL, d_sister_um REAL,
  PRIMARY KEY (run_id, film, gt_parent)
);

-- the board ledger: offline delta vs board delta, sign agreement per axis.
CREATE TABLE IF NOT EXISTS submissions (
  sub_id TEXT PRIMARY KEY, run_id TEXT, ts REAL,
  axis TEXT,                             -- edge | division | runtime | measurement
  change TEXT NOT NULL,                  -- ONE change
  offline_delta REAL, offline_component TEXT,
  board_score REAL, board_delta REAL,
  sign_agreement INTEGER,
  notes TEXT
);
"""


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def code_sha(src: Path | None = None) -> str:
    """Hash of the source tree, so a run is tied to the code that produced it."""
    src = src or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(src.rglob("*.py")):
        h.update(p.relative_to(src).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def start_run(label: str, config: dict, kind: str = "local", weights: dict | None = None,
              notes: str = "") -> str:
    blob = json.dumps(config, sort_keys=True)
    cfg_sha = hashlib.sha256(blob.encode()).hexdigest()
    run_id = f"{time.strftime('%m%d-%H%M%S')}-{cfg_sha[:8]}"
    with connect() as con:
        con.execute(
            "INSERT INTO runs (run_id, ts, label, kind, config, config_sha, code_sha, weights, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, time.time(), label, kind, blob, cfg_sha, code_sha(),
             json.dumps(weights or {}), notes),
        )
    return run_id


def finish_run(run_id: str, wall_s: float, stage_s: dict | None = None) -> None:
    with connect() as con:
        con.execute("UPDATE runs SET wall_s=?, stage_s=? WHERE run_id=?",
                    (wall_s, json.dumps(stage_s or {}), run_id))


def record_scores(run_id: str, rows: list[dict]) -> None:
    cols = ["fold", "film", "J_edge", "adj_J_edge", "multiplier", "n_pred", "n_est",
            "edge_tp", "edge_fp", "edge_fn", "div_tp", "div_fp", "div_fn", "weight"]
    with connect() as con:
        con.executemany(
            f"INSERT OR REPLACE INTO scores (run_id,{','.join(cols)}) "
            f"VALUES (?,{','.join('?' * len(cols))})",
            [(run_id, *[r.get(c) for c in cols]) for r in rows],
        )


def split_score(J: float, n_pred: int, n_est: float, a: float = 0.1) -> dict:
    """Break adj_edge_jaccard into Jaccard and node-count multiplier.

    If a change moves the multiplier while J sits still, you are being paid for
    deleting nodes, and that money is not in the bank until the board says so.
    """
    mult = 1.0 - a * (n_pred - n_est) / n_est
    return {"J": J, "multiplier": mult, "adj": max(0.0, J * mult),
            "node_delta_pct": 100.0 * (n_pred - n_est) / n_est}
