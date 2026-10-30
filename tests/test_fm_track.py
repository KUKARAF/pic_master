"""The /files "Track this folder" job (POST /api/fm/track/start, GET /api/fm/track/status)
and the CLI scan path it shares with `media add` (media_manager.scan_folder / fast_scan).

A temp library with untracked media in nested folders, an already-tracked file, a
byte-identical copy of a tracked file (a duplicate), a .mediaignore'd file, a .noface
marker and an untracked file OUTSIDE the tracked subtree. Tracks a subfolder via the web
job and checks: only that subtree got tracked, the duplicate was counted + reported and
NOT tracked, the ignored file skipped, the status fields/phase, fm_listing states, the
.noface marker applied, 409 on a concurrent start and the path jail.

CLI unchanged: fast_scan's / start_scan's positional signatures are as before (new
arguments are keyword-only), `media add` run as a real CLI subprocess gives the same
stdout/stderr, duplicates.txt and DB rows as before, it still pipes through pv when pv
is installed (a fake pv on PATH), and the web path (callback, no pv) produces the same
DB result for the same tree.

Run: python tests/test_fm_track.py   (or python -m pytest tests/test_fm_track.py)
"""
import inspect
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from media_manager.database import Database
from media_manager.fast_scan import fast_scan
from media_manager.manual_db import ManualDB
from media_manager.media_manager import MediaManager, scan_folder
import media_manager.web as web_mod
from media_manager.web import create_app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKED_STATES = {'tracked', 'partial', 'scanned', 'broken'}


def _png(path, color=(200, 30, 30)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (40, 30), color).save(path, format='PNG')


def _write(path, data=b''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)


def _build_tree(root):
    """The shared fixture tree (the already-tracked file is tracked by the caller)."""
    def j(*parts):
        return os.path.join(root, *parts)
    os.makedirs(j('.media'), exist_ok=True)
    _png(j('keep', 'orig.png'), (1, 2, 3))                     # tracked before the scan
    _png(j('trips', 'a.png'), (10, 0, 0))                      # untracked
    _png(j('trips', 'rome', 'b.png'), (20, 0, 0))              # untracked, nested
    _png(j('trips', 'rome', 'deep', 'c.png'), (30, 0, 0))      # untracked, deeper
    shutil.copyfile(j('keep', 'orig.png'), j('trips', 'copy.png'))   # duplicate content
    _png(j('trips', 'skip_me.png'), (40, 0, 0))                # .mediaignore'd
    _write(j('trips', 'notes.txt'), b'not media')
    _write(j('trips', 'rome', '.noface'))                      # folder marker
    _png(j('other', 'z.png'), (50, 0, 0))                      # outside the subtree
    _write(j('.mediaignore'), b'skip_*\n')
    return j


def _db_rows(db):
    return sorted((r['path'], r['checksum'], r['size'], r['modified_time']) for r in db.conn.execute(
        'SELECT fp.path, f.checksum, f.size, fp.modified_time FROM file_paths fp '
        'JOIN files f ON f.id = fp.file_id'))


def _noface(db, path):
    return db.conn.execute('SELECT f.noface FROM files f JOIN file_paths fp ON fp.file_id = f.id '
                           'WHERE fp.path = ?', (path,)).fetchone()[0]


def _wait_track(client):
    deadline = time.time() + 60
    while time.time() < deadline:
        st = client.get('/api/fm/track/status').json()
        if not st['running']:
            return st
        time.sleep(0.05)
    raise AssertionError('track job did not finish')


def check_cli_signatures():
    """New fast_scan/scan_folder arguments are keyword-only with CLI defaults; start_scan
    (what media.py calls) is unchanged."""
    params = inspect.signature(fast_scan).parameters
    positional = [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == ['root_path', 'db', 'data_root', 'recursive', 'max_workers',
                          'dup_report_path', 'reindex', 'manual'], positional
    assert params['progress'].kind is params['progress'].KEYWORD_ONLY and params['progress'].default is None
    assert params['use_pv'].kind is params['use_pv'].KEYWORD_ONLY and params['use_pv'].default is True
    assert list(inspect.signature(MediaManager.start_scan).parameters) == ['self', 'path', 'recursive', 'reindex']
    sp = inspect.signature(scan_folder).parameters
    assert all(sp[k].kind is sp[k].KEYWORD_ONLY for k in ('progress', 'use_pv', 'dup_report_path'))
    print('OK  signatures: fast_scan/start_scan positional args unchanged, extras keyword-only')


def _fake_pv(bin_dir, marker):
    """A `pv` that records it ran and passes stdin through (pv -l's data behaviour)."""
    path = os.path.join(bin_dir, 'pv')
    with open(path, 'w') as fh:
        fh.write(f'#!/bin/sh\necho ran >> "{marker}"\nexec cat\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def check_cli_unchanged():
    """`media add trips` as a real CLI run: exact output, duplicates.txt in the cwd, DB
    rows, markers, pv used. Then the web-style call (callback, use_pv=False) on an
    identical tree gives the same DB rows and duplicates.txt lines."""
    cli_root = tempfile.mkdtemp()
    _build_tree(cli_root)
    m = MediaManager(os.path.join(cli_root, '.media', 'media.db'))
    m.start_scan('keep')
    m.close()
    bin_dir = tempfile.mkdtemp()
    marker = os.path.join(bin_dir, 'pv-ran')
    _fake_pv(bin_dir, marker)
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''),
               PATH=bin_dir + os.pathsep + os.environ.get('PATH', ''))
    proc = subprocess.run([sys.executable, '-c', 'import sys; from media_manager.media import main; '
                           'sys.argv = ["media", "add", "trips"]; sys.exit(main())'],
                          cwd=cli_root, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == 'Scan done, 3 files\n', repr(proc.stdout)
    dup_file = os.path.join(cli_root, 'duplicates.txt')
    assert ('fast_scan: 1 duplicate file(s) found and NOT tracked — see ' + dup_file) in proc.stderr, proc.stderr
    assert 'WARNING: 1 duplicate file(s) found and NOT tracked — see duplicates.txt' in proc.stderr
    with open(dup_file) as fh:
        cli_dups = fh.read()
    assert cli_dups == 'trips/copy.png\tkeep/orig.png\n', repr(cli_dups)
    assert os.path.exists(marker), 'CLI path no longer pipes find through pv'
    cli_db = Database(os.path.join(cli_root, '.media', 'media.db'))
    cli_rows = _db_rows(cli_db)
    assert [r[0] for r in cli_rows] == ['keep/orig.png', 'trips/a.png', 'trips/rome/b.png',
                                        'trips/rome/deep/c.png'], cli_rows
    assert _noface(cli_db, 'trips/rome/b.png') == 1 and _noface(cli_db, 'trips/a.png') == 0
    print('OK  CLI `media add`: same output, duplicates.txt in cwd, pv used, markers applied')

    # Web-style call on an identical tree: same DB rows, pv never spawned.
    web_root = tempfile.mkdtemp()
    _build_tree(web_root)
    os.remove(marker)
    db = Database(os.path.join(web_root, '.media', 'media.db'))
    manual = ManualDB(os.path.join(web_root, '.media', 'manual.db'))
    scan_folder(db, manual, web_root, 'keep')
    old_path = os.environ['PATH']
    os.environ['PATH'] = bin_dir + os.pathsep + old_path
    try:
        events = []
        res = scan_folder(db, manual, web_root, 'trips', progress=events.append, use_pv=False,
                          dup_report_path=os.path.join(web_root, 'duplicates.txt'))
    finally:
        os.environ['PATH'] = old_path
    assert res == (3, 1, 0), res
    assert not os.path.exists(marker), 'web path must not spawn pv'
    # Same files, checksums, sizes and mtimes as the CLI run (mtimes differ per tree,
    # so compare everything but those, and mtimes against the files on disk).
    web_rows = _db_rows(db)
    assert [r[:3] for r in web_rows] == [r[:3] for r in cli_rows], (web_rows, cli_rows)
    for path, _c, _s, mtime in web_rows:
        assert abs(mtime - os.path.getmtime(os.path.join(web_root, path))) < 1e-3
    with open(os.path.join(web_root, 'duplicates.txt')) as fh:
        assert fh.read() == cli_dups
    phases = [e['phase'] for e in events if 'phase' in e]
    assert phases == ['listing', 'hashing', 'saving', 'markers'], phases
    last = {}
    for e in events:
        last.update(e)
    assert (last['added'], last['already'], last['duplicates'], last['ignored']) == (3, 0, 1, 1), last
    print('OK  web-style scan_folder: same DB rows + duplicates.txt, no pv, phases reported')


def check_track_job():
    root = tempfile.mkdtemp()
    j = _build_tree(root)
    app = create_app(root)
    db, _errors, manual = app.state.dbs
    # The already-tracked file, tracked the CLI way.
    scan_folder(db, manual, root, 'keep')
    client = TestClient(app)

    st = client.get('/api/fm/track/status').json()
    assert st['running'] is False and st['phase'] is None and st['path'] is None, st
    for key in ('running', 'path', 'phase', 'done', 'total', 'added', 'already', 'duplicates',
                'ignored', 'error', 'started_at', 'finished_at'):
        assert key in st, key

    # Jail + missing folder.
    for bad in ('../', '..', '.media', '.media/thumbs', '/etc', 'trips/../../x'):
        r = client.post('/api/fm/track/start', json={'path': bad})
        assert r.status_code == 400, (bad, r.status_code, r.text)
    assert client.post('/api/fm/track/start', json={'path': 'no/such'}).status_code == 404
    print('OK  track: jail (../, .media, absolute) 400, missing folder 404')

    r = client.post('/api/fm/track/start', json={'path': 'trips'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['started'] is True and body['path'] == 'trips', body
    st = _wait_track(client)
    assert st['error'] is None and st['phase'] == 'done', st

    # Only the trips/ subtree (other/z.png stays untracked).
    paths = {r[0] for r in _db_rows(db)}
    assert paths == {'keep/orig.png', 'trips/a.png', 'trips/rome/b.png', 'trips/rome/deep/c.png'}, paths
    assert 'trips/copy.png' not in paths and 'trips/skip_me.png' not in paths, paths
    # The duplicate goes to the path-level trash, not duplicates.txt; nothing deleted.
    assert not os.path.exists(j('duplicates.txt'))
    assert os.path.isfile(j('trips/copy.png'))
    dupes = client.get('/api/trash/duplicates').json()
    assert dupes['total'] == 1, dupes
    item = dupes['items'][0]
    assert (item['path'], item['kept_path']) == ('trips/copy.png', 'keep/orig.png'), item
    assert item['file_id'] == db.find_file_id_by_checksum(item['checksum'])
    assert not manual.is_trashed(item['checksum'])   # the kept original is NOT trashed
    assert _noface(db, 'trips/rome/deep/c.png') == 1 and _noface(db, 'trips/a.png') == 0
    print('OK  track: subtree tracked, duplicate -> Trash (not tracked), ignored, .noface applied')

    # Restore = track that copy as a second location of the same content.
    r = client.post('/api/trash/duplicates/restore', json={'path': 'trips/copy.png'})
    assert r.status_code == 200 and r.json()['file_id'] == item['file_id'], r.text
    assert 'trips/copy.png' in {r[0] for r in _db_rows(db)}
    assert client.get('/api/trash/duplicates').json()['total'] == 0
    assert client.post('/api/trash/duplicates/restore', json={'path': 'trips/copy.png'}).status_code == 404
    # Re-tracking doesn't re-trash it (it's a known path now).
    client.post('/api/fm/track/start', json={'path': 'trips'})
    st = _wait_track(client)
    assert st['duplicates'] == 0 and st['already'] == 4, st
    print('OK  restore tracks the copy as a second location; re-track leaves it alone')

    # A copy of content whose tracked copy is flagged damaged is relinked, not trashed,
    # and the damaged flag is cleared for a re-check. Same when the tracked copy is gone.
    orig_id = db.find_file_id_by_checksum(item['checksum'])
    db.mark_broken(orig_id)
    with open(j('keep/orig.png'), 'rb') as fh:
        data = fh.read()
    _write(j('trips/again/copy2.png'), data)
    client.post('/api/fm/track/start', json={'path': 'trips'})
    st = _wait_track(client)
    assert st['relinked'] == 1 and st['duplicates'] == 0, st
    assert 'trips/again/copy2.png' in {r[0] for r in _db_rows(db)}
    assert db.get_file_by_id(orig_id)['broken'] is None
    for p in ('keep/orig.png', 'trips/copy.png', 'trips/again/copy2.png'):
        os.remove(j(p))                       # every tracked copy gone from disk
    _write(j('other/back.png'), data)
    client.post('/api/fm/track/start', json={'path': 'other'})
    st = _wait_track(client)
    assert st['relinked'] == 1 and st['duplicates'] == 0, st
    assert 'other/back.png' in {r[0] for r in _db_rows(db)}
    print('OK  damaged or vanished tracked copy -> new copy relinked, damaged flag cleared')


def check_track_job_fields():
    """A clean run (no race with a second start) for exact counts and listing states."""
    root = tempfile.mkdtemp()
    j = _build_tree(root)
    app = create_app(root)
    db, _errors, manual = app.state.dbs
    scan_folder(db, manual, root, 'keep')
    client = TestClient(app)
    # Let the startup inventory walk finish so rollups are populated.
    deadline = time.time() + 30
    while client.get('/api/fm/inventory/status').json()['running'] and time.time() < deadline:
        time.sleep(0.05)
    before = client.get('/api/fm/list').json()
    trips = {f['name']: f for f in before['folders']}['trips']['counts']
    assert trips['untracked'] == 4 and trips['ignored'] == 1, trips   # a, b, c, copy

    r = client.post('/api/fm/track/start', json={'path': 'trips'})
    assert r.status_code == 200 and r.json()['started'] is True, r.text
    st = _wait_track(client)
    assert st['error'] is None, st
    assert st['phase'] == 'done' and st['path'] == 'trips', st
    assert (st['added'], st['already'], st['duplicates'], st['ignored']) == (3, 0, 1, 1), st
    assert st['total'] == 4 and st['done'] == 4, st          # 4 hashed candidates saved
    assert st['started_at'] and st['finished_at'] and st['finished_at'] >= st['started_at']

    # Only the subtree got tracked.
    paths = {r[0] for r in _db_rows(db)}
    assert paths == {'keep/orig.png', 'trips/a.png', 'trips/rome/b.png', 'trips/rome/deep/c.png'}, paths

    # fm_listing now shows them tracked; the duplicate is 'trashed', ignored stays ignored.
    files = {f['name']: f for f in client.get('/api/fm/list', params={'path': 'trips'}).json()['files']}
    assert files['a.png']['state'] in TRACKED_STATES and files['a.png']['file_id'], files['a.png']
    assert files['copy.png']['state'] == 'trashed', files['copy.png']
    assert files['skip_me.png']['state'] == 'ignored'
    deep = {f['name']: f for f in client.get('/api/fm/list', params={'path': 'trips/rome/deep'}).json()['files']}
    assert deep['c.png']['state'] in TRACKED_STATES
    other = {f['name']: f for f in client.get('/api/fm/list', params={'path': 'other'}).json()['files']}
    assert other['z.png']['state'] == 'untracked'
    root_fm = client.get('/api/fm/list').json()
    folders = {f['name']: f for f in root_fm['folders']}
    t = folders['trips']['counts']
    assert t['untracked'] == 0 and t['trashed'] == 1 and t['ignored'] == 1 \
        and sum(t[k] for k in TRACKED_STATES) == 3, t
    assert folders['other']['counts']['untracked'] == 1
    print('OK  track: status fields (phase done, added 3, duplicates 1, ignored 1), listing states')

    # 409: while a job is running (held on a gate), a second start is refused.
    gate = threading.Event()
    real = web_mod.scan_folder

    def slow_scan(*a, **kw):
        gate.wait(10)
        return real(*a, **kw)
    web_mod.scan_folder = slow_scan
    try:
        r = client.post('/api/fm/track/start', json={'path': 'other'})
        assert r.status_code == 200 and r.json()['running'] is True, r.text
        r2 = client.post('/api/fm/track/start', json={'path': 'trips'})
        assert r2.status_code == 409 and "'other'" in r2.json()['detail'], r2.text
        assert client.get('/api/fm/track/status').json()['running'] is True
    finally:
        gate.set()
        web_mod.scan_folder = real
    st = _wait_track(client)
    assert st['error'] is None and st['added'] == 1 and st['path'] == 'other', st
    print('OK  track: 409 while running, detail names the running folder')

    # Root ('' = whole library) works and is idempotent now.
    r = client.post('/api/fm/track/start', json={'path': ''})
    assert r.status_code == 200, r.text
    st = _wait_track(client)
    assert st['error'] is None and st['path'] == '' and st['added'] == 0 and st['already'] == 5, st

    page = client.get('/files/trips')
    assert page.status_code == 200 and 'id="fm-track-btn"' in page.text and 'id="fm-ml-btn"' in page.text
    assert 'data-untracked="0"' in page.text, 'trashed duplicate not counted as untracked'
    assert client.get('/files').status_code == 200
    print('OK  track: root scan, /files renders both buttons with the untracked count')


def run():
    check_cli_signatures()
    check_cli_unchanged()
    check_track_job()
    check_track_job_fields()
    print('ALL ASSERTIONS PASSED')


def test_fm_track():
    run()


if __name__ == '__main__':
    run()
