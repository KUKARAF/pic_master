"""The /files file manager: GET /api/fm/list, the disk inventory job, GET /api/fm/preview.

A temp library with tracked files (inserted via db.upsert_file_path), untracked media,
a non-media file, a .mediaignore'd file, captured_frames/ (a hidden tracked still + an
untracked image), a tracked file deleted from disk, a folder that only exists in the DB
and nested folders. Checks per-file states, folder rollups before the inventory has
ever run (tracked rows only) and after it (untracked/ignored/missing too), show_other,
the preview endpoint + its jail, and that the /files page still renders.

Run: python tests/test_file_manager_api.py   (or python -m pytest tests/test_file_manager_api.py)
"""
import io
import os
import tempfile
import time

from fastapi.testclient import TestClient
from PIL import Image

from media_manager import scan_status
from media_manager.web import create_app

TRACKED_STATES = {'tracked', 'partial', 'scanned', 'broken'}


def _png(path, color=(200, 30, 30)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (64, 48), color).save(path, format='PNG')


def _write(path, data=b'x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)


def _wait_inventory(client):
    deadline = time.time() + 30
    while time.time() < deadline:
        st = client.get('/api/fm/inventory/status').json()
        if not st['running']:
            return st
        time.sleep(0.05)
    raise AssertionError('inventory walk did not finish')


def _by_name(items):
    return {i['name']: i for i in items}


def _tracked_sum(counts):
    return sum(counts[k] for k in TRACKED_STATES)


def run():
    root = tempfile.mkdtemp()

    def j(*parts):
        return os.path.join(root, *parts)

    os.makedirs(j('.media'))
    _png(j('trips', 'rome', 'a.png'))                       # tracked
    _png(j('trips', 'rome', 'b.png'), (0, 90, 200))         # untracked
    _png(j('trips', 'rome', 'skip_me.png'))                 # .mediaignore'd
    _write(j('trips', 'rome', 'notes.txt'), b'hello')        # other
    _png(j('trips', 'milan', 'deep', 'c.png'))              # tracked, nested
    _png(j('trips', 'milan', 'd.png'))                      # untracked, nested
    _png(j('captured_frames', 'still.png'))                 # tracked, hidden
    _png(j('captured_frames', 'loose.png'))                 # untracked in a system dir
    _write(j('top.mp4'), b'video-bytes')                     # tracked video
    _write(j('.mediaignore'), b'skip_*\n')
    outside = tempfile.mkdtemp()
    _png(os.path.join(outside, 'secret.png'))
    os.symlink(outside, j('escape'))                         # symlink out of the library

    app = create_app(root)
    db, _errors, manual = app.state.dbs
    client = TestClient(app)

    old = int(time.time()) - 100
    ids = {}
    for path, checksum in [('trips/rome/a.png', 'ca'), ('trips/rome/gone.png', 'cg'),
                           ('trips/milan/deep/c.png', 'cc'), ('captured_frames/still.png', 'cs'),
                           ('top.mp4', 'cv'), ('old/lost.png', 'cl')]:
        ids[path] = db.upsert_file_path(path, checksum, size=10, modified_time=old, seen_at=old)
    db.set_file_hidden(ids['captured_frames/still.png'])   # commits
    db.conn.commit()

    # Startup auto-walks a never-inventoried library; let it finish, then wipe it to
    # see the "never ran" fallback first.
    _wait_inventory(client)
    db.conn.execute('DELETE FROM disk_files')
    db.conn.execute('DELETE FROM disk_inventory_meta')
    db.conn.commit()

    # --- root before any inventory: rollups from tracked rows only ---
    r = client.get('/api/fm/list')
    assert r.status_code == 200, r.text
    fm = r.json()
    assert fm['path'] == '' and fm['parent'] is None
    assert fm['inventory']['refreshed_at'] is None and fm['inventory']['stale'] is True
    assert [s['key'] for s in fm['stages']['image']] == [k for k, _l, _r in scan_status.STAGES['image']]
    folders = _by_name(fm['folders'])
    # .media is never listed; the escape/ symlink isn't followed (like the scan's find).
    assert set(folders) == {'trips', 'captured_frames', 'old'}, folders.keys()
    t = folders['trips']['counts']
    assert _tracked_sum(t) == 3 and t['untracked'] == 0 and t['missing'] == 0, t
    assert folders['old']['counts']['missing'] == 1 and folders['old']['total'] == 1
    assert folders['captured_frames']['system'] is True and folders['trips']['system'] is False
    assert _tracked_sum(folders['captured_frames']['counts']) == 1
    files = _by_name(fm['files'])
    assert set(files) == {'top.mp4'}, files.keys()          # .mediaignore is 'other'
    assert files['top.mp4']['kind'] == 'video' and files['top.mp4']['state'] in TRACKED_STATES
    assert fm['counts']['other'] == 1
    print('OK  root before inventory: tracked-only rollups, DB-only folder missing, system flag')

    # --- run the inventory walk via the job endpoints ---
    r = client.post('/api/fm/inventory/start')
    assert r.status_code == 200 and r.json()['started'] in (True, False), r.text
    st = _wait_inventory(client)
    assert st['error'] is None and st['refreshed_at'] is not None, st
    # disk_files: media extensions only, .media never, escape/ symlink not followed.
    paths = {p for (p,) in db.conn.execute('SELECT path FROM disk_files')}
    assert paths == {'trips/rome/a.png', 'trips/rome/b.png', 'trips/rome/skip_me.png',
                     'trips/milan/deep/c.png', 'trips/milan/d.png', 'captured_frames/still.png',
                     'captured_frames/loose.png', 'top.mp4'}, paths

    fm = client.get('/api/fm/list').json()
    assert fm['inventory']['refreshed_at'] == st['refreshed_at'] and fm['inventory']['stale'] is False
    folders = _by_name(fm['folders'])
    t = folders['trips']['counts']
    assert t['untracked'] == 2 and t['ignored'] == 1 and t['missing'] == 1 and _tracked_sum(t) == 2, t
    assert folders['trips']['total'] == 6
    cf = folders['captured_frames']['counts']
    assert _tracked_sum(cf) == 1 and cf['ignored'] == 1, cf
    assert folders['old']['counts']['missing'] == 1
    print('OK  root after inventory: untracked / ignored / missing rollups')

    # Nested rollups one level down.
    fm = client.get('/api/fm/list', params={'path': 'trips'}).json()
    assert fm['parent'] == ''
    folders = _by_name(fm['folders'])
    assert folders['milan']['counts']['untracked'] == 1 and _tracked_sum(folders['milan']['counts']) == 1
    assert folders['rome']['counts']['missing'] == 1 and folders['rome']['counts']['ignored'] == 1

    # --- one folder's own files ---
    fm = client.get('/api/fm/list', params={'path': 'trips/rome'}).json()
    assert fm['path'] == 'trips/rome' and fm['parent'] == 'trips'
    files = _by_name(fm['files'])
    assert list(files) == ['a.png', 'b.png', 'gone.png', 'skip_me.png'], list(files)
    a = files['a.png']
    expected = scan_status.status_for_files(db, manual, [(a['file_id'], a['path'])])[a['file_id']]
    assert a['file_id'] == ids['trips/rome/a.png'] and a['state'] == expected['state']
    assert a['stages'] == expected['stages'] and a['missing'] == expected['missing']
    assert a['card']['id'] == a['file_id'] and a['kind'] == 'image'
    assert files['b.png']['state'] == 'untracked' and files['b.png']['file_id'] is None
    assert files['b.png']['card'] is None and files['b.png']['stages'] is None
    assert files['b.png']['size'] == os.path.getsize(j('trips', 'rome', 'b.png'))
    assert files['gone.png']['state'] == 'missing' and files['gone.png']['file_id'] == ids['trips/rome/gone.png']
    assert files['gone.png']['card']['id'] == ids['trips/rome/gone.png']   # still a grid card
    assert files['skip_me.png']['state'] == 'ignored'
    c = fm['counts']
    assert (c['untracked'], c['missing'], c['ignored'], c['other']) == (1, 1, 1, 1), c

    fm = client.get('/api/fm/list', params={'path': 'trips/rome', 'show_other': 1}).json()
    notes = _by_name(fm['files'])['notes.txt']
    assert notes['state'] == 'other' and notes['kind'] == 'other'

    # Hidden tracked still: listed with its state, but no grid card; loose image ignored.
    files = _by_name(client.get('/api/fm/list', params={'path': 'captured_frames'}).json()['files'])
    assert files['still.png']['state'] in TRACKED_STATES and files['still.png']['card'] is None
    assert files['loose.png']['state'] == 'ignored'

    # DB-only folder lists its missing file.
    files = _by_name(client.get('/api/fm/list', params={'path': 'old'}).json()['files'])
    assert files['lost.png']['state'] == 'missing'
    print('OK  folder files: tracked (scan_status) / untracked / missing / ignored / other')

    # A live listing re-syncs that folder's inventory rows (a file deleted after the walk).
    os.remove(j('trips', 'milan', 'd.png'))
    client.get('/api/fm/list', params={'path': 'trips/milan'})
    folders = _by_name(client.get('/api/fm/list', params={'path': 'trips'}).json()['folders'])
    assert folders['milan']['counts']['untracked'] == 0, folders['milan']

    # An upload lands in the inventory immediately (counted tracked, not missing).
    # (uploads always land in uploads/<bucket>/, whatever folder you're viewing)
    r = client.post('/api/files/upload',
                    files=[('files', ('up.mp4', b'uploaded', 'application/octet-stream'))])
    assert r.status_code == 200 and r.json()['saved'], r.text
    folders = _by_name(client.get('/api/fm/list', params={'path': ''}).json()['folders'])
    assert _tracked_sum(folders['uploads']['counts']) == 1 and folders['uploads']['counts']['missing'] == 0
    print('OK  live listing re-syncs inventory rows; uploads are counted at once')

    # --- jail on list ---
    for bad in ('../', '..', '.media', '.media/thumbs', '/etc', 'escape', 'trips/../../x'):
        assert client.get('/api/fm/list', params={'path': bad}).status_code == 400, bad
    assert client.get('/api/fm/list', params={'path': 'no/such'}).status_code == 404

    # --- preview ---
    r = client.get('/api/fm/preview', params={'path': 'trips/rome/b.png'})
    assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg', r.text
    assert Image.open(io.BytesIO(r.content)).size[0] == 400
    shard_files = [n for _d, _s, ns in os.walk(j('.media', 'fm_thumbs')) for n in ns]
    assert len(shard_files) == 1, shard_files
    assert client.get('/api/fm/preview', params={'path': 'trips/rome/b.png'}).content == r.content
    # Editing the file replaces (not adds to) its cached preview.
    _png(j('trips', 'rome', 'b.png'), (10, 200, 10))
    os.utime(j('trips', 'rome', 'b.png'), (time.time() + 5, time.time() + 5))
    assert client.get('/api/fm/preview', params={'path': 'trips/rome/b.png'}).status_code == 200
    assert len([n for _d, _s, ns in os.walk(j('.media', 'fm_thumbs')) for n in ns]) == 1
    for bad in ('../x.png', '.media/x.png', '/etc/x.png', 'escape/secret.png'):
        assert client.get('/api/fm/preview', params={'path': bad}).status_code == 400, bad
    assert client.get('/api/fm/preview', params={'path': 'trips/rome/gone.png'}).status_code == 404
    assert client.get('/api/fm/preview', params={'path': 'top.mp4'}).status_code == 400
    assert client.get('/api/fm/preview', params={'path': 'trips/rome/notes.txt'}).status_code == 400
    assert db.get_file_by_path('trips/rome/gone.png')['broken'] is None
    print('OK  preview: 400px JPEG, cached + replaced on edit; jail, missing, video refused')

    # --- the /files page still renders with the old context + fm ---
    page = client.get('/files/trips/rome')
    assert page.status_code == 200, page.text[:500]
    assert '/photo/%d' % ids['trips/rome/a.png'] in page.text
    assert client.get('/files').status_code == 200
    assert client.get('/files/trips/rome?hide_with_set=1&hide_with_face=1').status_code == 200
    print('OK  /files page renders')
    print('ALL ASSERTIONS PASSED')


def test_file_manager_api():
    run()


if __name__ == '__main__':
    run()
