"""HTTP layer of the /bulk smart job queue (web.py): validation, enqueue / list /
cancel / clear, in-process starter + real cooperative cancel of a cheap real job.

No real ML job is ever started: the queue thread is never launched
(bulk_queue_autostart False; tick() is driven by hand) and the heavy starters are stubbed
or the job is the CPU-trivial metadata one.
"""
import os
import tempfile
import threading
import time

import pytest
import xxhash
from fastapi.testclient import TestClient
from PIL import Image

import media_manager.web as web_mod
from media_manager import job_queue
from media_manager.web import create_app


@pytest.fixture
def env():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    app = create_app(root)
    app.state.bulk_queue_autostart = False
    client = TestClient(app)
    yield root, app, client
    app.state.bulk_queue.stop()


def _body(path='/api/metadata/start', **kw):
    d = {'key': path, 'label': 'Job', 'start_url': path,
         'status_url': path.replace('/start', '/status')}
    d.update(kw)
    return d


def test_enqueue_list_cancel_queued_clear(env):
    _root, app, client = env
    q = app.state.bulk_queue
    assert client.get('/api/bulk-queue').json()['items'] == []
    r = client.post('/api/bulk-queue', json=_body('/api/siglip-dino/start?include_trashed=1'))
    assert r.status_code == 200, r.text
    item = r.json()
    assert item['state'] == 'queued' and item['id'] and item['cancellable'] is True
    assert item['status_url'] == '/api/siglip-dino/status'
    snap = client.get('/api/bulk-queue').json()
    assert [i['id'] for i in snap['items']] == [item['id']]
    assert 'resources' in snap and 'limits' in snap
    # cancel the queued item: nothing was started
    d = client.delete(f"/api/bulk-queue/{item['id']}")
    assert d.status_code == 200 and d.json()['ok'] is True
    assert d.json()['item']['state'] == 'cancelled'
    assert client.delete('/api/bulk-queue/nope').status_code == 404
    assert client.post('/api/bulk-queue/clear-finished').json()['ok'] is True
    assert client.get('/api/bulk-queue').json()['items'] == []
    assert q._items == []


def test_imdb_status_path_is_derived(env):
    _root, _app, client = env
    r = client.post('/api/bulk-queue', json=_body('/api/imdb/rebuild', status_url='/api/imdb/status'))
    assert r.status_code == 200 and r.json()['status_url'] == '/api/imdb/status'


@pytest.mark.parametrize('start', [
    '/api/files', '/api/fm/ml/start', '/api/broken/recheck/start', '/api/tags/x/train',
    'http://evil.example/api/metadata/start', '//evil.example/api/metadata/start',
    'api/metadata/start', '/api/metadata/start/../../etc', '',
    '/api/siglip-dino/start?include_trashed=maybe',
    '/api/scan-video-faces/start?threshold=abc',
])
def test_rejects_non_catalog_or_bad_query(env, start):
    _root, app, client = env
    r = client.post('/api/bulk-queue', json=_body(start, key='k'))
    assert r.status_code == 400, (start, r.status_code, r.text)
    assert app.state.bulk_queue._items == []


def test_rejects_bad_status_url_and_missing_fields(env):
    _root, _app, client = env
    assert client.post('/api/bulk-queue',
                       json=_body(status_url='/api/files')).status_code == 400
    assert client.post('/api/bulk-queue', json={'key': 'k'}).status_code == 422
    assert client.post('/api/bulk-queue', json=_body(label='  ')).status_code == 400
    assert client.post('/api/bulk-queue', json=_body(key='')).status_code == 400


def test_query_params_pass_through(env):
    _root, _app, client = env
    for u in ('/api/siglip-dino/start?include_trashed=true',
              '/api/scan-video-faces/start?threshold=0.7&capture_unmatched=false&include_trashed=1',
              '/api/match-faces/start?threshold=0.6&try_rotations=1',
              '/api/near-dup/compute/start'):
        assert client.post('/api/bulk-queue', json=_body(u, key=u)).status_code == 200, u
    kw = web_mod  # noqa: F841 (parse is a closure; exercised through the starter below)


def test_starter_runs_route_inprocess_and_stub_gpu_job_never_starts(env, monkeypatch):
    """The real starter resolves the POST route, passes the coerced query, and an
    'already running'-style refusal is surfaced as an error item (not an exception)."""
    _root, app, client = env
    q = app.state.bulk_queue
    seen = []

    def fake_start(include_trashed: bool = False):
        seen.append(include_trashed)
        return {'started': False, 'message': 'stubbed'}

    for r in app.router.routes:
        if getattr(r, 'path', None) == '/api/siglip-dino/start' and 'POST' in r.methods:
            monkeypatch.setattr(r, 'endpoint', fake_start)
    item = client.post('/api/bulk-queue', json=_body('/api/siglip-dino/start?include_trashed=1')).json()
    q.tick()
    assert seen == [True]
    it = client.get('/api/bulk-queue').json()['items'][0]
    assert it['state'] == 'error' and 'stubbed' in it['message'], it
    assert item['id'] == it['id']


def test_nothing_to_do_is_done_not_error(env):
    _root, app, client = env
    # empty library: metadata has no candidates -> real route, trivially cheap
    client.post('/api/bulk-queue', json=_body('/api/metadata/start'))
    app.state.bulk_queue.tick()
    for _ in range(100):
        it = client.get('/api/bulk-queue').json()['items'][0]
        if it['state'] != 'running':
            break
        app.state.bulk_queue.tick()
        time.sleep(0.05)
    assert it['state'] == 'done', it


def test_cancel_flag_plumbing():
    job = {'cancel_requested': False}
    stop = web_mod._cancel_check(job)
    assert stop() is False
    job['cancel_requested'] = True
    assert stop() is True


def test_running_cheap_job_is_cancelled_cooperatively(env, monkeypatch):
    root, app, client = env
    q = app.state.bulk_queue
    db, _e, _m = app.state.dbs
    os.makedirs(os.path.join(root, 'p'))
    for i in range(4):
        pth = os.path.join(root, 'p', f'{i}.jpg')
        Image.new('RGB', (8, 8), (i * 30, 0, 0)).save(pth)
        data = open(pth, 'rb').read()
        db.upsert_file_path(f'p/{i}.jpg', xxhash.xxh64(data).hexdigest(), size=len(data), modified_time=1)
    db.conn.commit()   # the starter closes this thread's connection (tick() runs on it here)

    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_exif(path):
        calls.append(path)
        entered.set()
        release.wait(10)
        return (None, None, None)

    monkeypatch.setattr(web_mod.exif_reader, 'extract_exif_metadata', slow_exif)
    item = client.post('/api/bulk-queue', json=_body('/api/metadata/start')).json()
    q.tick()                                   # real starter -> real metadata job
    assert entered.wait(5)
    running = client.get('/api/bulk-queue').json()['items'][0]
    assert running['state'] == 'running' and running['cancellable'] is True
    d = client.delete(f"/api/bulk-queue/{item['id']}")
    assert d.status_code == 200 and d.json()['item']['cancelling'] is True
    release.set()                              # file in hand completes; the rest must not run
    for _ in range(200):
        st = client.get('/api/metadata/status').json()
        if not st['running']:
            break
        time.sleep(0.05)
    assert not st['running'] and len(calls) == 1 and st['done'] == 1 < st['total'], (st, calls)
    q.tick()
    final = client.get('/api/bulk-queue').json()['items'][0]
    assert final['state'] == 'cancelled', final
    # restart resets the flag: the job runs to completion
    monkeypatch.setattr(web_mod.exif_reader, 'extract_exif_metadata', lambda p: (None, None, None))
    assert client.post('/api/metadata/start').json()['started'] is True
    for _ in range(200):
        st = client.get('/api/metadata/status').json()
        if not st['running']:
            break
        time.sleep(0.05)
    assert st['done'] == st['total'] == 3, st


def test_non_cancellable_running_job_reports_false(env):
    _root, app, client = env
    q = app.state.bulk_queue
    item = job_queue.QueueItem(id='x', key='k', label='l', start_url='/api/near-dup/compute/start',
                               status_url='/api/near-dup/compute/status',
                               weight=job_queue.JobWeight(), state='running')
    q._items.append(item)
    snap = client.get('/api/bulk-queue').json()['items'][0]
    assert snap['cancellable'] is False
    assert client.delete('/api/bulk-queue/x').status_code == 409


def web_mod_status(path):
    return '/api/imdb/status' if path == '/api/imdb/rebuild' else path.replace('/start', '/status')


@pytest.mark.parametrize('path', [
    '/api/link-faces-to-bodies/start', '/api/metadata/start', '/api/imdb/rebuild',
    '/api/near-dup/compute/start', '/api/fetch-cities/start', '/api/faces/renormalize/start',
    '/api/broken/scan/start', '/api/match-cities/start'])
def test_undeclared_params_are_dropped_not_rejected(env, path):
    """bulk.html appends include_trashed=1 to every start URL; functions that don't
    declare it must still queue (param dropped), while bad values for declared ones 400."""
    _root, _app, client = env
    r = client.post('/api/bulk-queue', json=_body(path + '?include_trashed=1&bogus=x', key=path,
                                  status_url=web_mod_status(path)))
    assert r.status_code == 200, (path, r.text)


def test_dropped_param_not_passed_to_function(env, monkeypatch):
    _root, app, client = env
    seen = []

    def fake_start():
        seen.append('called')
        return {'started': True, 'total': 1}
    for r in app.router.routes:
        if getattr(r, 'path', None) == '/api/metadata/start' and 'POST' in r.methods:
            monkeypatch.setattr(r, 'endpoint', fake_start)
    client.post('/api/bulk-queue', json=_body('/api/metadata/start?include_trashed=1'))
    app.state.bulk_queue.tick()
    assert seen == ['called']


def test_finished_item_delete_removes_it(env, monkeypatch):
    _root, app, client = env
    item = client.post('/api/bulk-queue', json=_body('/api/siglip-dino/start')).json()
    client.delete(f"/api/bulk-queue/{item['id']}")      # cancel queued -> finished
    snap = client.get('/api/bulk-queue').json()['items'][0]
    assert snap['state'] == 'cancelled' and snap['cancellable'] is True
    d = client.delete(f"/api/bulk-queue/{item['id']}")
    assert d.status_code == 200 and d.json()['ok'] is True
    assert client.get('/api/bulk-queue').json()['items'] == []


def test_queued_start_defers_when_job_already_running(env, monkeypatch):
    _root, app, client = env
    for r in app.router.routes:
        if getattr(r, 'path', None) == '/api/metadata/start' and 'POST' in r.methods:
            monkeypatch.setattr(r, 'endpoint',
                                lambda: {'started': False, 'message': 'Metadata extraction already running.'})
    client.post('/api/bulk-queue', json=_body('/api/metadata/start'))
    app.state.bulk_queue.tick()
    it = client.get('/api/bulk-queue').json()['items'][0]
    assert it['state'] == 'queued' and it['blocked_reason'] == 'already running (started manually)', it


def test_start_endpoints_are_serialized_per_job(env):
    """Concurrent starts of the same job never overlap their candidate scan, so the
    check-and-claim (`if running` ... `running=True`) is atomic."""
    _root, app, _client = env
    db, _e, _m = app.state.dbs
    route = next(r for r in app.router.routes
                 if getattr(r, 'path', None) == '/api/metadata/start' and 'POST' in r.methods)
    assert getattr(route.endpoint, '_bulk_serialized', False)
    assert route.dependant.call is route.endpoint      # what FastAPI really invokes
    active, peak, lk = [0], [0], threading.Lock()

    def slow_scan():
        with lk:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.15)
        with lk:
            active[0] -= 1
        return []
    db.get_files_without_metadata = slow_scan
    res = []
    ts = [threading.Thread(target=lambda: res.append(route.endpoint())) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] == 1, peak
