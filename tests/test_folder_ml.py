"""The file manager's "Run missing ML here" job (folder_ml.py + /api/fm/ml/*) and the
/bulk jobs it shares per-file loops with.

Every model is faked (no real model loads). Covers:

  * folder_ml.plan: run order, required-only default (tiles only when named),
    broken files never planned, unknown stage → ValueError;
  * run_plan: stage order, prepare() failure → 'skipped' + run continues, claim()
    message → 'skipped', an aborting stage → 'error' and the run stops;
  * /api/fm/ml/*: folder scoping (files outside untouched), only-missing (a file
    already done for a stage isn't reprocessed), preview counts == scan_status,
    cooperative cancel between files, 409 while running, /bulk refusal both ways,
    object-index infra abort writes no sentinel, GPU guard / worker-down → 'skipped';
  * /bulk metadata, CLIP index, face detection and phash still work end-to-end
    through the refactored per-file functions.

Run: python tests/test_folder_ml.py   (or python -m pytest tests/test_folder_ml.py)
"""
import os
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import folder_ml, scan_status
from media_manager import web as webmod
from media_manager.web import create_app
from media_manager.worker_client import WorkerError

IN_A = ('a/1.png', 'a/2.png', 'a/sub/3.png')
OUTSIDE = ('b/4.png', 'ab/5.png')   # 'ab' shares the prefix 'a' — must stay out of scope


# --- fixtures / fakes ----------------------------------------------------------------

def _lib(tmp, paths=IN_A + OUTSIDE):
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    db, _errors, manual = app.state.dbs
    ids = {}
    for i, rel in enumerate(paths):
        p = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.new('RGB', (64, 48), (20 * i, 90, 150)).save(p, format='PNG')
        ids[rel] = db.upsert_file_path(rel, f'cs{i:038d}', size=os.path.getsize(p))
    db.conn.commit()
    client = TestClient(app)
    _wait(client, '/api/fm/inventory/status')   # startup auto-walk; keep it out of the way
    return db, manual, client, ids


def _wait(client, url, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(url).json()
        if not st['running']:
            return st
        time.sleep(0.03)
    raise AssertionError(f'{url} never finished')


def _stages(st):
    return {s['key']: s for s in st['stages']}


def _status(db, manual, ids):
    return scan_status.status_for_files(db, manual, [(fid, rel) for rel, fid in ids.items()])


class FakeIndexer:
    model_name = 'fake-clip'

    def __init__(self, exc=None):
        self.calls = []
        self.exc = exc

    def model_id(self):
        return 'fake-clip'

    def embed_images(self, paths):
        if self.exc is not None:
            raise self.exc
        self.calls.extend(paths)
        return [np.ones(4, np.float32) for _ in paths], []

    def embed_pil_images(self, images):
        return [np.ones(4, np.float32) for _ in images]


class FakeFaceDetector:
    _model_name = 'fake'

    def __init__(self):
        self.calls = []

    def model_id(self, name):
        return f'fake-faces-{name}'

    def detect_faces(self, paths):
        self.calls.extend(paths)
        return [(p, [{'bbox': [1, 2, 30, 40], 'embedding': np.ones(8, np.float32),
                      'det_score': 0.9}], None) for p in paths]


class FakeYOLO:
    def __init__(self):
        self.calls = []
        self.vocab = None

    def set_vocab(self, vocab):
        self.vocab = list(vocab)

    def detect_images(self, paths):
        self.calls.extend(paths)
        return [(p, [('cat', 0.9, 1.0, 2.0, 3.0, 4.0)], None) for p in paths]


def _fakes(indexer=None, faces=None, yolo=None):
    """Patch every model getter the image stages use (except objects/region/pattern)."""
    indexer = indexer or FakeIndexer()
    faces = faces or FakeFaceDetector()
    yolo = yolo or FakeYOLO()
    return (patch.object(webmod, '_get_clip_indexer', lambda: indexer),
            patch.object(webmod, '_get_face_detector', lambda: faces),
            patch.object(webmod, '_get_object_detector', lambda: yolo))


# --- 1. planning / stage loop (no app) -------------------------------------------------

def _st(kind, done=(), state='partial'):
    keys = [k for k, _l, _r in scan_status.STAGES[kind]]
    return {'kind': kind, 'state': state, 'stages': {k: k in done for k in keys}, 'missing': []}


def test_plan_order_and_defaults():
    files = [(1, 'x.png'), (2, 'y.png'), (3, 'v.mp4'), (4, 'bad.png')]
    statuses = {1: _st('image', done={'exif', 'clip'}), 2: _st('image'),
                3: _st('video', done={'stills', 'clip'}), 4: _st('image', state='broken')}
    plan = folder_ml.plan(files, statuses)
    keys = [k for k, _f in plan]
    assert keys == ['exif', 'clip', 'tags', 'faces', 'bodies', 'objects', 'region', 'pattern',
                    'phash', 'video_faces'], keys            # no tiles, no stills (all done)
    got = dict(plan)
    assert got['exif'] == [(2, 'y.png')] and got['clip'] == [(2, 'y.png')]
    assert got['faces'] == [(1, 'x.png'), (2, 'y.png')]     # broken file 4 never planned
    assert got['video_faces'] == [(3, 'v.mp4')]
    # Explicit stages: run exactly those (tiles included), in STAGE_ORDER.
    plan = folder_ml.plan(files, statuses, ['tiles', 'exif', 'stills'])
    assert [k for k, _f in plan] == ['exif', 'tiles', 'stills']
    assert dict(plan)['stills'] == []
    try:
        folder_ml.plan(files, statuses, ['exif', 'nope'])
    except ValueError as exc:
        assert 'nope' in str(exc)
    else:
        raise AssertionError('unknown stage accepted')


def test_run_plan_order_skip_and_abort():
    order = []

    def proc(key):
        def _p(files, _ctx, progress, stop):
            order.append(key)
            for i, _f in enumerate(files, 1):
                progress(i, 0)
            return len(files), 0
        return _p

    def boom():
        raise RuntimeError('no model here')

    def abort(files, _ctx, progress, stop):
        order.append('objects')
        raise RuntimeError('Aborted: 5 in a row')

    f = [(1, 'x.png')]
    plan = [(k, f) for k in ('exif', 'clip', 'tags', 'faces', 'objects', 'region')]
    runners = {'exif': folder_ml.Runner(proc('exif')),
               'clip': folder_ml.Runner(proc('clip'), boom),          # unavailable
               'tags': folder_ml.Runner(proc('tags')),                # claimed by /bulk
               'faces': folder_ml.Runner(proc('faces')),
               'objects': folder_ml.Runner(abort),
               'region': folder_ml.Runner(proc('region'))}
    job = folder_ml.FolderMLJob()
    assert job.try_begin('a', plan) and not job.try_begin('a', plan)
    seen_current = []

    def claim(key):
        seen_current.append(job.state['current_stage'])
        return 'Skipped: busy' if key == 'tags' else None

    invalidations = []
    err = folder_ml.run_plan(job, plan, runners, claim=claim,
                             invalidate=lambda: invalidations.append(1))
    job.finish(err)
    st = job.status()
    by = _stages(st)
    assert order == ['exif', 'faces', 'objects'], order
    assert seen_current == ['exif', 'clip', 'tags', 'faces', 'objects']   # set before claim
    assert by['exif']['status'] == 'done' and by['exif']['done'] == 1
    assert by['clip']['status'] == 'skipped' and 'no model here' in by['clip']['message']
    assert by['tags']['status'] == 'skipped' and by['tags']['message'] == 'Skipped: busy'
    assert by['objects']['status'] == 'error' and 'Aborted' in by['objects']['message']
    assert by['region']['status'] == 'pending'                            # run stopped
    assert st['error'] and 'Aborted' in st['error'] and not st['running']
    assert st['current_stage'] is None and len(invalidations) == 3   # stages that ran


# --- 2. the folder job over the web API --------------------------------------------------

def test_folder_scope_only_missing_and_preview():
    with tempfile.TemporaryDirectory() as tmp:
        db, manual, client, ids = _lib(tmp)
        # a/2.png already has a CLIP embedding and faces → not reprocessed for those.
        db.insert_embedding(ids['a/2.png'], np.ones(4, np.float32).tobytes(), 'old')
        db.insert_faces(ids['a/2.png'], [], 'old')

        pv = client.get('/api/fm/ml/preview', params={'path': 'a'}).json()
        in_a = {rel: fid for rel, fid in ids.items() if rel.startswith('a/')}
        statuses = _status(db, manual, in_a)
        assert pv['path'] == 'a' and pv['files'] == 3
        assert [s['key'] for s in pv['stages']] == list(folder_ml.STAGE_ORDER)
        for s in pv['stages']:
            want = sum(1 for st in statuses.values() if s['key'] in st['stages']
                       and not st['stages'][s['key']])
            assert s['missing'] == want, (s, want)
            assert s['required'] == (s['key'] != 'tiles')
        assert _stages(pv)['clip']['missing'] == 2 and _stages(pv)['faces']['missing'] == 2

        indexer, faces, yolo = FakeIndexer(), FakeFaceDetector(), FakeYOLO()
        p1, p2, p3 = _fakes(indexer, faces, yolo)
        with p1, p2, p3:
            r = client.post('/api/fm/ml/start', json={
                'path': 'a', 'stages': ['phash', 'exif', 'clip', 'tags', 'faces']})
            assert r.status_code == 200 and r.json()['started'], r.text
            st = _wait(client, '/api/fm/ml/status')
        assert st['error'] is None and not st['cancelled'], st
        assert [s['key'] for s in st['stages']] == ['exif', 'clip', 'tags', 'faces', 'phash']
        assert all(s['status'] == 'done' for s in st['stages']), st['stages']
        by = _stages(st)
        assert by['clip']['total'] == by['clip']['done'] == 2 and by['exif']['total'] == 3
        # only-missing: a/2.png never went through CLIP / face detection again.
        embedded = sorted(os.path.relpath(p, tmp) for p in indexer.calls)
        assert embedded == ['a/1.png', 'a/sub/3.png'], embedded
        assert sorted(os.path.relpath(p, tmp) for p in faces.calls) == ['a/1.png', 'a/sub/3.png']
        assert sorted(os.path.relpath(p, tmp) for p in yolo.calls) == sorted(in_a)
        assert yolo.vocab, 'vocab not set before detection'
        after = _status(db, manual, ids)
        for rel, fid in ids.items():
            stg = after[fid]['stages']
            for key in ('exif', 'clip', 'tags', 'faces', 'phash'):
                assert stg[key] == rel.startswith('a/'), (rel, key, stg)   # outside untouched
        assert after[ids['a/1.png']]['stages']['region'] is False          # not requested
        # Nothing missing any more → a default run plans none of those stages.
        pv2 = _stages(client.get('/api/fm/ml/preview', params={'path': 'a'}).json())
        assert pv2['clip']['missing'] == 0 and pv2['phash']['missing'] == 0
        # Errors: jail, unknown stage, missing folder.
        assert client.post('/api/fm/ml/start', json={'path': '../x'}).status_code == 400
        assert client.post('/api/fm/ml/start', json={'path': 'a', 'stages': ['x']}).status_code == 400
        assert client.get('/api/fm/ml/preview', params={'path': 'nope'}).status_code == 404


def test_cancel_stops_between_files():
    with tempfile.TemporaryDirectory() as tmp:
        db, manual, client, ids = _lib(tmp)
        started, go, seen = threading.Event(), threading.Event(), []

        def slow_exif(path):
            seen.append(path)
            started.set()
            assert go.wait(10)
            return None, None, None

        with patch.object(webmod.exif_reader, 'extract_exif_metadata', slow_exif):
            r = client.post('/api/fm/ml/start', json={'path': 'a', 'stages': ['exif', 'phash']})
            assert r.status_code == 200, r.text
            assert started.wait(10)
            c = client.post('/api/fm/ml/cancel').json()
            assert c['cancel_requested'] is True and c['running'] is True
            go.set()
            st = _wait(client, '/api/fm/ml/status')
        by = _stages(st)
        assert len(seen) == 1, seen                      # stopped after the file in hand
        assert st['cancelled'] is True and st['error'] is None
        assert by['exif']['status'] == 'cancelled' and by['exif']['done'] == 1
        assert by['phash']['status'] == 'cancelled' and by['phash']['done'] == 0
        done = [rel for rel, s in ((r, _status(db, manual, ids)[f]) for r, f in ids.items())
                if s['stages']['exif']]
        assert len(done) == 1 and done[0].startswith('a/'), done
        assert client.post('/api/fm/ml/cancel').json()['cancel_requested'] is False


def test_concurrency_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        _db, _manual, client, _ids = _lib(tmp)
        started, go = threading.Event(), threading.Event()

        def slow_exif(path):
            started.set()
            assert go.wait(10)
            return None, None, None

        with patch.object(webmod.exif_reader, 'extract_exif_metadata', slow_exif):
            # Folder job in its exif stage: a 2nd folder job → 409; /bulk metadata refuses.
            assert client.post('/api/fm/ml/start', json={'path': 'a', 'stages': ['exif']}).status_code == 200
            assert started.wait(10)
            r = client.post('/api/fm/ml/start', json={'path': 'b'})
            assert r.status_code == 409 and "'a'" in r.json()['detail'], r.text
            b = client.post('/api/metadata/start').json()
            assert b['started'] is False and 'file manager' in b['message'], b
            assert client.get('/api/metadata/status').json()['running'] is False
            # A different stage's /bulk job is unaffected.
            assert client.post('/api/phash/start').json()['started'] is True
            go.set()
            _wait(client, '/api/fm/ml/status')
            _wait(client, '/api/phash/status')

            # Vice versa: /bulk metadata running → the folder job's exif stage is skipped.
            started.clear(); go.clear()
            os.makedirs(os.path.join(tmp, 'c'))
            Image.new('RGB', (8, 8)).save(os.path.join(tmp, 'c', 'n.png'))
            _db.upsert_file_path('c/n.png', 'c' * 40, size=1)
            _db.conn.commit()
            assert client.post('/api/metadata/start').json()['started'] is True
            assert started.wait(10)
            assert client.post('/api/fm/ml/start', json={'path': 'c', 'stages': ['exif', 'phash']}).status_code == 200
            deadline = time.time() + 10
            while _stages(client.get('/api/fm/ml/status').json())['exif']['status'] in ('pending', 'running'):
                assert time.time() < deadline
                time.sleep(0.03)
            by = _stages(client.get('/api/fm/ml/status').json())
            assert by['exif']['status'] == 'skipped' and '/bulk metadata' in by['exif']['message']
            go.set()
            _wait(client, '/api/metadata/status')
            _wait(client, '/api/fm/ml/status')


class RaisingEncoder:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def model_id(self):
        return 'fake-dino'

    def device_label(self):
        return 'fake'

    def embed_regions(self, im):
        self.calls += 1
        raise self.exc


def test_object_infra_abort_does_not_stamp():
    with tempfile.TemporaryDirectory() as tmp:
        paths = tuple(f'o/{i}.png' for i in range(7))
        db, manual, client, ids = _lib(tmp, paths)
        enc = RaisingEncoder(RuntimeError('encoder blew up'))
        with patch.object(webmod, '_get_object_encoder', lambda _root: enc), \
                patch.object(webmod.compute, 'torch_device', lambda: 'cuda'), \
                patch.dict(os.environ, {'MEDIA_OBJECT_DECODE_BATCH': '1'}):
            client.post('/api/fm/ml/start', json={'path': 'o', 'stages': ['objects', 'phash']})
            st = _wait(client, '/api/fm/ml/status')
        by = _stages(st)
        assert enc.calls == 5, enc.calls
        assert by['objects']['status'] == 'error' and 'Aborted' in by['objects']['message']
        assert by['phash']['status'] == 'pending', by['phash']   # the run stopped
        assert st['error'] and 'Aborted' in st['error']
        n = db.conn.execute('SELECT COUNT(*) FROM object_region_embeddings').fetchone()[0]
        assert n == 0, 'infra failure stamped a sentinel'
        assert not any(s['stages']['objects'] for s in _status(db, manual, ids).values())


def test_object_gpu_fatal_stops_at_once():
    with tempfile.TemporaryDirectory() as tmp:
        paths = tuple(f'o/{i}.png' for i in range(7))
        db, manual, client, ids = _lib(tmp, paths)
        enc = RaisingEncoder(RuntimeError('CUDA error: device lost'))
        with patch.object(webmod, '_get_object_encoder', lambda _root: enc), \
                patch.object(webmod.compute, 'torch_device', lambda: 'cuda'), \
                patch.dict(os.environ, {'MEDIA_OBJECT_DECODE_BATCH': '1'}):
            client.post('/api/fm/ml/start', json={'path': 'o', 'stages': ['objects', 'phash']})
            st = _wait(client, '/api/fm/ml/status')
        by = _stages(st)
        assert enc.calls == 1, enc.calls                         # no point trying the rest
        assert by['objects']['status'] == 'error'
        assert by['objects']['message'].startswith('The GPU ran out of memory or was lost'), by['objects']
        assert by['phash']['status'] == 'pending', by['phash']
        assert db.conn.execute('SELECT COUNT(*) FROM object_region_embeddings').fetchone()[0] == 0


def test_unavailable_backends_skip_and_continue():
    with tempfile.TemporaryDirectory() as tmp:
        db, manual, client, ids = _lib(tmp)
        down = FakeIndexer(exc=WorkerError('worker unreachable'))
        p1, p2, p3 = _fakes(indexer=down)
        env = {k: v for k, v in os.environ.items() if k != 'MEDIA_DEVICE'}
        with p1, p2, p3, patch.object(webmod.compute, 'torch_device', lambda: 'cpu'), \
                patch.dict(os.environ, env, clear=True):
            client.post('/api/fm/ml/start', json={'path': 'a', 'stages': ['clip', 'objects', 'phash']})
            st = _wait(client, '/api/fm/ml/status')
        by = _stages(st)
        assert by['clip']['status'] == 'skipped' and 'worker unreachable' in by['clip']['message']
        assert by['objects']['status'] == 'skipped' and 'GPU' in by['objects']['message']
        assert by['phash']['status'] == 'done' and by['phash']['done'] == 3
        assert st['error'] is None


# --- 3. /bulk jobs through the refactored functions ------------------------------------

def test_bulk_jobs_still_work():
    with tempfile.TemporaryDirectory() as tmp:
        db, manual, client, ids = _lib(tmp)
        indexer, faces, yolo = FakeIndexer(), FakeFaceDetector(), FakeYOLO()
        p1, p2, p3 = _fakes(indexer, faces, yolo)
        with p1, p2, p3:
            for start, status in (('/api/metadata/start', '/api/metadata/status'),
                                  ('/api/index/start', '/api/index/status'),
                                  ('/api/detect-faces/start', '/api/detect-faces/status'),
                                  ('/api/phash/start', '/api/phash/status')):
                r = client.post(start).json()
                assert r['started'] is True and r['total'] == 5, (start, r)
                st = _wait(client, status)
                assert st['error'] is None and st['done'] == st['total'] == 5, (status, st)
                assert st['pending'] == 0, (status, st)
        assert len(indexer.calls) == 5 and len(faces.calls) == 5
        assert client.get('/api/detect-faces/status').json()['matched'] == 0
        n = db.conn.execute('SELECT COUNT(*) FROM faces WHERE bbox != ?', ('[]',)).fetchone()[0]
        assert n == 5
        for st in _status(db, manual, ids).values():
            assert all(st['stages'][k] for k in ('exif', 'clip', 'faces', 'phash')), st


if __name__ == '__main__':
    for t in (test_plan_order_and_defaults,
              test_run_plan_order_skip_and_abort,
              test_folder_scope_only_missing_and_preview,
              test_cancel_stops_between_files,
              test_concurrency_refusals,
              test_object_infra_abort_does_not_stamp, test_object_gpu_fatal_stops_at_once,
              test_unavailable_backends_skip_and_continue,
              test_bulk_jobs_still_work):
        t()
        print('ok:', t.__name__)
    print('\nFOLDER-ML TESTS PASSED')
