#!/usr/bin/env python3
"""Lossless, deterministic snapshot of the knowledge core (not disposable indexes).

Restore into an empty database; reimporting the identical snapshot is a no-op.
Never merge by integer IDs into a different database or overwrite unexported work.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'db/biohub_base.db'
OUT = ROOT / 'db/source_rows.jsonl'
TABLES = ('source', 'repo', 'fact', 'experiment', 'artifact', 'decision', 'lb_snapshot', 'forum_topic', 'public_kernel')
HEADER = '# biohub knowledge snapshot v2; restore before rebuilding disposable indexes\n'


def connect(path, readonly=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Database does not exist: {path}; run import to bootstrap it')
    cx = sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True)
    cx.row_factory = sqlite3.Row
    cx.execute('PRAGMA foreign_keys=ON')
    return cx


def rows(cx):
    for table in TABLES:
        for row in cx.execute(f'SELECT * FROM {table} ORDER BY id'):
            yield {'tbl': table, **dict(row)}


def encoded(records):
    return [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]


def read_snapshot(path):
    with open(path, encoding='utf-8') as f:
        if f.readline() != HEADER:
            raise ValueError('Legacy/incompatible snapshot: export v2 from the original DB first')
        records = [json.loads(line) for line in f if line.strip() and not line.startswith('#')]
    seen = set()
    for record in records:
        table = record.get('tbl')
        if table not in TABLES or type(record.get('id')) is not int:
            raise ValueError('Invalid snapshot table or row ID')
        key = table, record['id']
        if key in seen:
            raise ValueError(f'Duplicate snapshot identity: {key}')
        seen.add(key)
    return records


def export(db=DB, out=OUT):
    with connect(db, readonly=True) as cx:
        cx.execute('BEGIN')
        records = list(rows(cx))
        if cx.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Broken foreign keys; refusing to export')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=out.parent, prefix=out.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(HEADER)
            f.write('\n'.join(encoded(records)) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Exported {len(records)} knowledge rows to {out}')
    return records


def load(db=DB, out=OUT):
    records = read_snapshot(out)
    # Validate the entire payload in isolation before touching the destination.
    with sqlite3.connect(':memory:') as staged:
        staged.row_factory = sqlite3.Row
        staged.executescript((ROOT / 'db/schema.sql').read_text())
        staged.execute('PRAGMA foreign_keys=ON')
        with staged:
            staged.execute('PRAGMA defer_foreign_keys=ON')
            for record in records:
                table = record['tbl']
                cols = [c for c in record if c != 'tbl']
                allowed = {r[1] for r in staged.execute(f'PRAGMA table_info({table})')}
                if set(cols) != allowed:
                    raise ValueError(f'Wrong columns for {table}: {set(cols) ^ allowed}')
                staged.execute(f'INSERT INTO {table} ({",".join(cols)}) VALUES ({",".join("?" for _ in cols)})',
                               [record[c] for c in cols])
        if encoded(rows(staged)) != encoded(records):
            raise ValueError('Snapshot is not in canonical order')
        db = Path(db)
        db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db) as cx:
            cx.row_factory = sqlite3.Row
            cx.executescript((ROOT / 'db/schema.sql').read_text())
            cx.execute('PRAGMA foreign_keys=ON')
            cx.execute('BEGIN IMMEDIATE')
            existing = list(rows(cx))
            if existing:
                if encoded(existing) != encoded(records):
                    raise ValueError('Database differs from snapshot; export/save it first. Restore to a new --db path.')
                print('Already restored; no changes')
                return 0
            # External tables may contain references even if the core is empty.
            if any(cx.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                   for t in ('repo_commit', 'repo_file')):
                raise ValueError('Restore requires an empty database')
            cx.execute('PRAGMA defer_foreign_keys=ON')
            for record in records:
                table = record['tbl']
                cols = [c for c in record if c != 'tbl']
                cx.execute(f'INSERT INTO {table} ({",".join(cols)}) VALUES ({",".join("?" for _ in cols)})',
                           [record[c] for c in cols])
    print(f'Restored {len(records)} knowledge rows into {db}')
    return 0


def check(db=DB, out=OUT):
    saved = read_snapshot(out)
    with connect(db, readonly=True) as cx:
        cx.execute('BEGIN')
        live = list(rows(cx))
        if cx.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Broken foreign keys')
    if encoded(saved) != encoded(live):
        print('STALE: run python3 tools/sync_source_rows.py export')
        return 1
    print(f'Up to date ({len(live)} knowledge rows)')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('export', 'import', 'check'))
    parser.add_argument('--db', type=Path, default=DB)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    try:
        if args.command == 'export':
            export(args.db, args.out)
        else:
            raise SystemExit({'import': load, 'check': check}[args.command](args.db, args.out))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
