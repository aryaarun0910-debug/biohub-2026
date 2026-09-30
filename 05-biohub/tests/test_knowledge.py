"""Regression tests run entirely on disposable databases and frozen local evidence."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import sync_source_rows as sync
import ingest_kaggle as kaggle
from evidence import verify


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'test.db'
        self.out = Path(self.temp.name) / 'snapshot.jsonl'
        with contextlib.closing(sqlite3.connect(self.db)) as c, c:
            c.executescript((ROOT / 'db/schema.sql').read_text())
            c.execute("INSERT INTO source VALUES(7,'report','digest','measured',NULL,'2026-01-01','note')")
            c.execute("INSERT INTO repo(id,name) VALUES(3,'archive')")
            c.execute("INSERT INTO fact(id,topic,key,value,claim_type,source_id,quote,observed_at,status,superseded_by,validity,validity_reason) VALUES(10,'test','old','old','observation',7,'quote','2026-01-01','superseded',11,'INVALID','leak')")
            c.execute("INSERT INTO fact(id,topic,key,value,claim_type,source_id,quote,observed_at,validity) VALUES(11,'test','new','new','observation',7,'quote','2026-01-02','VALID')")
            c.execute("INSERT INTO experiment(id,repo_id,name,started_at,source_ref) VALUES(5,3,'run','2026-01-01','manual')")
            c.execute("INSERT INTO artifact(id,kind,name,produced_by) VALUES(9,'weights','model',5)")
            c.execute("INSERT INTO decision(id,question) VALUES(4,'what next?')")
            c.execute("INSERT INTO forum_topic(id,title,takeaway) VALUES(1,'topic','manual triage')")
        with contextlib.redirect_stdout(io.StringIO()):
            sync.export(self.db, self.out)

    def test_lossless_repeat_restore_and_atomic_export(self):
        target = Path(self.temp.name) / 'restored.db'
        sync.load(target, self.out)
        sync.load(target, self.out)
        self.assertEqual(sync.check(target, self.out), 0)
        with contextlib.closing(sqlite3.connect(target)) as c, c:
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(c.execute('SELECT COUNT(*) FROM decision').fetchone()[0], 1)
            self.assertEqual(c.execute('SELECT produced_by FROM artifact').fetchone()[0], 5)
        before = self.out.read_bytes()
        with patch.object(sync.os, 'replace', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                sync.export(target, self.out)
        self.assertEqual(self.out.read_bytes(), before)
        sync.export(target, self.out)
        self.assertEqual(self.out.read_bytes(), before)

    def test_refuses_drift_without_overwriting_work(self):
        with contextlib.closing(sqlite3.connect(self.db)) as c, c:
            c.execute("UPDATE decision SET choice='unexported' WHERE id=4")
        with self.assertRaisesRegex(ValueError, 'differs'):
            sync.load(self.db, self.out)
        self.assertEqual(sync.check(self.db, self.out), 1)
        with contextlib.closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('SELECT choice FROM decision').fetchone()[0], 'unexported')

    def test_invalid_foreign_key_never_creates_destination(self):
        records = sync.read_snapshot(self.out)
        next(r for r in records if r['tbl'] == 'artifact')['produced_by'] = 999
        self.out.write_text(sync.HEADER + '\n'.join(sync.encoded(records)) + '\n')
        target = Path(self.temp.name) / 'invalid.db'
        with self.assertRaises(sqlite3.IntegrityError):
            sync.load(target, self.out)
        self.assertFalse(target.exists())

    def test_missing_db_is_not_created_by_read(self):
        missing = Path(self.temp.name) / 'missing.db'
        with self.assertRaises(ValueError):
            sync.check(missing, self.out)
        self.assertFalse(missing.exists())

    def test_rejects_legacy_and_unknown_tables(self):
        self.out.write_text('# old export\n')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            sync.load(self.db, self.out)
        self.out.write_text(sync.HEADER + json.dumps({'tbl':'sqlite_master','id':1}) + '\n')
        with self.assertRaises(ValueError):
            sync.load(self.db, self.out)


class BootstrapTests(unittest.TestCase):
    def test_tracked_snapshot_bootstraps_and_replays_without_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / 'base.db'
            sync.load(db)
            self.assertEqual(sync.check(db), 0)
            with patch.object(kaggle, 'DB', str(db)), patch.object(kaggle, 'NOW', '2099-01-01T00:00:00Z'):
                kaggle.main()
                with contextlib.closing(sync.connect(db, readonly=True)) as c:
                    first = sync.encoded(sync.rows(c))
                kaggle.main()
                with contextlib.closing(sync.connect(db, readonly=True)) as c:
                    self.assertEqual(sync.encoded(sync.rows(c)), first)
                    self.assertFalse(c.execute("SELECT 1 FROM lb_snapshot WHERE taken_at LIKE '2099%'").fetchall())
                    self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(sync.check(db), 0)

    def test_evidence_tampering_fails_before_ingest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'evidence', root / 'evidence')
            with (root / 'evidence/raw/comp.json').open('a') as f:
                f.write(' ')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                verify(root / 'evidence')
            db = root / 'never-created.db'
            with patch.object(kaggle, 'ROOT', str(root)), patch.object(kaggle, 'DB', str(db)):
                with self.assertRaises(ValueError):
                    kaggle.main()
            self.assertFalse(db.exists())

    def test_failed_harvest_preserves_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            shutil.copy(ROOT / 'tools/harvest_kaggle.sh', root / 'tools')
            (root / 'evidence/raw').mkdir(parents=True)
            old = root / 'evidence/raw/comp.json'
            old.write_text('keep me')
            (root / 'bin').mkdir()
            curl = root / 'bin/curl'
            curl.write_text('#!/bin/sh\nexit 22\n')
            curl.chmod(0o755)
            import os
            result = subprocess.run(['bash', str(root / 'tools/harvest_kaggle.sh')],
                                    env={**os.environ,'PATH':str(root / 'bin') + ':' + os.environ['PATH']},
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(old.read_text(), 'keep me')
            self.assertFalse((root / 'evidence/.harvest-lock').exists())

    def test_manual_competition_annotations_survive_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / 'base.db'
            sync.load(db)
            with contextlib.closing(sqlite3.connect(db)) as c, c:
                topic = c.execute('SELECT id FROM forum_topic LIMIT 1').fetchone()[0]
                kernel = c.execute('SELECT id FROM public_kernel LIMIT 1').fetchone()[0]
                c.execute("UPDATE forum_topic SET relevance='reviewed',takeaway='manual conclusion' WHERE id=?", (topic,))
                c.execute("UPDATE public_kernel SET family='manually-reviewed',note='keep annotation' WHERE id=?", (kernel,))
            with patch.object(kaggle, 'DB', str(db)):
                kaggle.main()
            with contextlib.closing(sqlite3.connect(db)) as c:
                self.assertEqual(c.execute('SELECT relevance,takeaway FROM forum_topic WHERE id=?', (topic,)).fetchone(),
                                 ('reviewed','manual conclusion'))
                self.assertEqual(c.execute('SELECT family,note FROM public_kernel WHERE id=?', (kernel,)).fetchone(),
                                 ('manually-reviewed','keep annotation'))

    def test_complete_harvest_publishes_verified_batch(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'tools', root / 'tools', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(ROOT / 'db', root / 'db', ignore=shutil.ignore_patterns('*.db*','.receipts'))
            shutil.copytree(ROOT / 'evidence', root / 'evidence')
            sync.load(root / 'db/biohub_base.db')
            (root / 'bin').mkdir()
            curl = root / 'bin/curl'
            curl.write_text('#!' + sys.executable + "\nimport os,sys,shutil\nfrom pathlib import Path\nout=Path(sys.argv[sys.argv.index('-o')+1])\nshutil.copy(Path(os.environ['BIOHUB_FIXTURES'])/out.name,out)\n")
            curl.chmod(0o755)
            result = subprocess.run(['bash', str(root / 'tools/harvest_kaggle.sh')],
                                    env={**os.environ,'PATH':str(root / 'bin') + ':' + os.environ['PATH'],
                                         'BIOHUB_FIXTURES':str(ROOT / 'evidence/raw')},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(verify(root / 'evidence'))
            self.assertFalse((root / 'evidence/.harvest-lock').exists())


class RegistryTests(unittest.TestCase):
    def test_validity_supersession_and_artifact_links_survive_reingest(self):
        try:
            registry = importlib.import_module('ingest_registries')
        except ModuleNotFoundError:
            self.skipTest('optional PyYAML not installed')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'research/00-system/registry'
            path.mkdir(parents=True)
            (path / 'facts.yaml').write_text('''facts:
- id: F-1
  statement: old measurement
  value: 1
  date: '2026-01-01'
  validity: INVALID
  validity_reason: contaminated
  superseded_by: F-2
- id: F-2
  statement: corrected measurement
  value: 2
  date: '2026-01-02'
  validity: VALID
''')
            with contextlib.closing(sqlite3.connect(':memory:')) as c, c, patch.object(registry, 'MONO', str(root)):
                c.executescript((ROOT / 'db/schema.sql').read_text())
                c.execute('PRAGMA foreign_keys=ON')
                c.execute("INSERT INTO repo(id,name) VALUES(1,'Biohub-CellTracking-2026')")
                registry.load_facts(c)
                before = c.execute('SELECT * FROM fact ORDER BY id').fetchall()
                registry.load_facts(c)
                self.assertEqual(c.execute('SELECT * FROM fact ORDER BY id').fetchall(), before)
                self.assertEqual(c.execute("SELECT validity,validity_reason,superseded_by FROM fact WHERE key LIKE 'F-1:%'").fetchone(), ('INVALID','contaminated',2))
                registry.save_experiment(c,'run','test','neutral','note','levers.yaml:1')
                c.execute("INSERT INTO artifact(kind,name,produced_by) VALUES('weights','model',1)")
                registry.save_experiment(c,'run','changed','win','new note','levers.yaml:1')
                self.assertEqual(c.execute('SELECT COUNT(*) FROM experiment').fetchone()[0], 1)
                self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])


class RepositoryTests(unittest.TestCase):
    def test_commit_statistics_belong_to_correct_commit(self):
        import ingest_repos as repos
        with contextlib.closing(sqlite3.connect(':memory:')) as c, c:
            c.executescript((ROOT / 'db/schema.sql').read_text())
            def fake_sh(args, **kwargs):
                if args[1] == 'rev-parse':
                    return 'main'
                if args[1] == 'ls-files':
                    return ''
                return ('\x1e' + 'a'*40 + '\x1f2026-01-02\x1fAuthor\x1fnew commit\n'
                        ' 2 files changed, 7 insertions(+), 1 deletion(-)\n'
                        '\x1e' + 'b'*40 + '\x1f2026-01-01\x1fAuthor\x1fold commit\n'
                        ' 1 file changed, 3 insertions(+)\n')
            with patch.object(repos, 'sh', side_effect=fake_sh):
                repos.ingest(c, 'fixture', '/unused')
            self.assertEqual(c.execute('SELECT subject,files_changed,insertions,deletions FROM repo_commit ORDER BY authored DESC').fetchall(),
                             [('new commit',2,7,1),('old commit',1,3,None)])

    def test_failed_fetch_is_not_silently_accepted(self):
        import ingest_repos as repos
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'fixture/.git').mkdir(parents=True)
            def fail(args, **kwargs):
                self.assertTrue(kwargs.get('check', True))
                raise RuntimeError('network failure')
            with patch.object(repos, 'CHECKOUTS', directory), patch.object(repos, 'sh', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'network failure'):
                    repos.clone_or_pull('fixture')


class PreflightTests(unittest.TestCase):
    def test_exit_zero_without_cpu_report_does_not_pass(self):
        import preflight
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'script.py'
            script.write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'No explicit'):
                preflight.main(script)
            script.write_text('print(\'{"biohub_check":"preflight","device":"cpu","ok":true}\')\n')
            self.assertEqual(preflight.main(script), 0)


if __name__ == '__main__':
    unittest.main()
