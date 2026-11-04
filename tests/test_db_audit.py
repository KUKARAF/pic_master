"""scripts/db_audit.py: read-only audit of a data root."""
import hashlib
import importlib.util
import json
import os
import sqlite3
import time

import numpy as np
import pytest

from media_manager.database import Database
from media_manager.error_log import ErrorLog
from media_manager.manual_db import ManualDB

_SPEC = importlib.util.spec_from_file_location(
    'db_audit', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'db_audit.py'))
db_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(db_audit)


def _vec(seed, dim=8):
    return np.random.RandomState(seed).rand(dim).astype(np.float32).tobytes()


@pytest.fixture
def root(tmp_path):
    media = tmp_path / '.media'
    media.mkdir()
    db = Database(str(media / 'media.db'))
    ids = []
    for i in range(6):
        ids.append(db.upsert_file_path(f'/lib/a/b/img{i}.jpg', f'sum{i}', size=10, modified_time=1))
    db.upsert_file_path('/lib/a/c/dup0.jpg', 'sum0', size=10, modified_time=2)  # second path
    for i in ids[:4]:
        db.insert_embedding(i, _vec(i), 'clip-x')
    for i in ids[:2]:
        db.insert_siglip_embedding(i, _vec(i, 16), 'sig-x')
    db.close()
    m = ManualDB(str(media / 'manual.db'))
    m.close()
    e = ErrorLog(str(media / 'error.db'))
    e.close()
    return tmp_path


def _snapshot(root):
    snap = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, 'rb') as f:
                snap[p] = (hashlib.sha256(f.read()).hexdigest(), os.stat(p).st_mtime_ns)
    return snap


def test_report_sections_and_readonly(root):
    before = _snapshot(root)
    time.sleep(0.05)
    R = db_audit.run_audit(str(root), quick=False, timeout=30, runs=2)
    text = db_audit.render(R)
    for s in ('1. FILES AND PRAGMAS', '2. TABLES AND INDEXES', '3. EMBEDDING TABLE HEALTH',
              '4. INDEX AUDIT', '5. HOT QUERIES', '6. files_with_path ALTERNATIVE', '7. WAL GROWTH RISK',
              '8. FINDINGS'):
        assert s in text
    assert _snapshot(root) == before  # no file created, changed or touched
    p = R['sections']['pragmas']['media.db']
    assert p['page_size'] > 0 and p['journal_mode'] == 'wal'
    t = R['sections']['tables']['media.db']['tables']
    assert t['files']['rows'] == 6 and t['embeddings']['rows'] == 4
    h = R['sections']['embedding_health']
    assert h['tables']['embeddings']['distinct_dims'] == [8]
    assert h['files_without_primary_embedding']['clip'] == 2
    assert h['files_without_primary_embedding']['siglip'] == 4
    hq = R['sections']['hot_queries']
    assert hq['view_count']['rows'] == 1 and hq['view_id_path']['rows'] == 6
    assert any('CORRELATED' in f for f in hq['view_id_path']['plan_flags']) or hq['view_id_path']['plan']
    assert R['sections']['view_alternatives']['B_window_row_number']['rows'] == 6
    json.dumps(R, default=str)


def test_quick_skips_slow(root):
    R = db_audit.run_audit(str(root), quick=True)
    assert R['sections']['hot_queries']['view_id_path']['status'] == 'skipped (--quick)'
    assert R['sections']['hot_queries']['view_count']['status'] == 'ok'


def test_cli_json(root, tmp_path_factory, capsys):
    out = tmp_path_factory.mktemp('out') / 'r.json'
    assert db_audit.main(['--data-root', str(root), '--quick', '--json', str(out)]) == 0
    assert 'FINDINGS' in capsys.readouterr().out
    assert 'findings' in json.loads(out.read_text())


def test_missing_tables_degrade(root):
    media = root / '.media' / 'media.db'
    c = sqlite3.connect(str(media))
    c.execute('PRAGMA journal_mode=DELETE')
    for t in ('embeddings_dino', 'tile_embeddings', 'phashes', 'tags'):
        c.execute(f'DROP TABLE {t}')
    c.execute('DROP TABLE embeddings_siglip')
    c.commit()
    c.close()
    (root / '.media' / 'manual.db').unlink()
    R = db_audit.run_audit(str(root), quick=False, runs=1)
    text = db_audit.render(R)
    eh = R['sections']['embedding_health']
    assert eh['tables']['embeddings_dino'] == 'missing'
    assert eh['files_without_primary_embedding']['dino'] == 'missing'
    assert 'missing' in R['sections']['hot_queries']['matrix_siglip_ids']['status']
    assert R['sections']['manual_queries'] == 'missing'
    assert 'FINDINGS' in text


def test_no_media_db(tmp_path):
    (tmp_path / '.media').mkdir()
    R = db_audit.run_audit(str(tmp_path))
    assert R['sections']['errors']
    assert 'FINDINGS' in db_audit.render(R)
    assert os.listdir(tmp_path / '.media') == []


def test_live_wal_db_readonly(tmp_path):
    """App running: -wal/-shm exist and a writer holds the DB open. Audit must not alter
    media.db and must not create new files."""
    media = tmp_path / '.media'
    media.mkdir()
    db = Database(str(media / 'media.db'))
    i = db.upsert_file_path('/lib/x.jpg', 's', size=1, modified_time=1)
    db.insert_embedding(i, _vec(1), 'm')
    names = set(os.listdir(media))
    assert 'media.db-wal' in names and 'media.db-shm' in names
    main_sha = hashlib.sha256((media / 'media.db').read_bytes()).hexdigest()
    R = db_audit.run_audit(str(tmp_path), runs=1)
    assert R['sections']['tables']['media.db']['tables']['files']['rows'] == 1  # sees WAL data
    assert set(os.listdir(media)) == names
    assert hashlib.sha256((media / 'media.db').read_bytes()).hexdigest() == main_sha
    db.close()
