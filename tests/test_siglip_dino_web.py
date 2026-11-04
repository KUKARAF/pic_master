"""Web layer for the SigLIP 2 + DINOv2 pipeline: bulk job (resume, failure isolation),
speed test math/no-writes, and search ranking -- all with fake embedders (no GPU, no models)."""
import os
import time

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from media_manager import web

DIMS = {'siglip': 6, 'dino': 4}


def _unit(seed, dim):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _vec_for(path, dim, salt):
    return _unit(abs(hash((os.path.basename(path), salt))) % (2 ** 31), dim)


class FakeEmb:
    def __init__(self, which):
        self.which = which
        self.batch_size = 4
        self.workers = 1

    def model_id(self):
        return f'fake-{self.which}/v1'

    def embed_text(self, text):
        return _unit(7, DIMS['siglip'])


@pytest.fixture
def env(tmp_path, monkeypatch):
    os.makedirs(tmp_path / '.media')
    calls = {'chunks': [], 'bad': set(), 'raise_after': None}

    def fake_shared(embs, paths):
        calls['chunks'].append(list(paths))
        if calls['raise_after'] is not None and len(calls['chunks']) > calls['raise_after']:
            raise RuntimeError('boom: no GPU')
        sig, dino, failed = [], [], []
        for p in paths:
            if os.path.basename(p) in calls['bad']:
                failed.append((p, 'cannot decode'))
                continue
            sig.append(_vec_for(p, DIMS['siglip'], 's'))
            dino.append(_vec_for(p, DIMS['dino'], 'd'))
        return [np.array(sig), np.array(dino)], failed

    monkeypatch.setattr(web, '_worker_client', None)
    monkeypatch.setattr(web, '_get_siglip_embedder', lambda: FakeEmb('siglip'))
    monkeypatch.setattr(web, '_get_dino_embedder', lambda: FakeEmb('dino'))
    monkeypatch.setattr(web, '_local_embed_shared', fake_shared)
    # local static ids are used by the status probe; patch the embedders-free path
    import types, sys
    fake_mod = types.ModuleType('media_manager.embedders')
    fake_mod.SigLIP2Embedder = type('S', (), {'model_id': staticmethod(lambda: 'fake-siglip/v1')})
    fake_mod.DINOv2Embedder = type('D', (), {'model_id': staticmethod(lambda: 'fake-dino/v1')})
    monkeypatch.setitem(sys.modules, 'media_manager.embedders', fake_mod)

    app = web.create_app(str(tmp_path))
    db, errors, manual = app.state.dbs
    ids = {}
    for i in range(10):
        name = f'img{i}.jpg'
        Image.new('RGB', (8, 8), (i * 20, 0, 0)).save(tmp_path / name)
        ids[name] = db.upsert_file_path(name, f'sum{i}', size=1, modified_time=1)
    db.conn.commit()
    return app, TestClient(app), db, errors, manual, ids, calls, tmp_path


def _wait(client, url='/api/siglip-dino/status'):
    for _ in range(200):
        d = client.get(url).json()
        if not d['running']:
            return d
        time.sleep(0.02)
    raise AssertionError('job did not finish')


def test_job_indexes_all_and_resumes(env):
    app, c, db, errors, manual, ids, calls, _ = env
    assert c.post('/api/siglip-dino/start').json()['total'] == 10
    st = _wait(c)
    assert st['error'] is None and st['done'] == 10 and st['failed'] == 0
    assert db.count_siglip_indexed() == 10 and db.count_dino_indexed() == 10
    # nothing left -> idempotent
    assert c.post('/api/siglip-dino/start').json()['total'] == 0
    # drop one dino row -> only that file is re-embedded, and only its dino row is rewritten
    db.conn.execute('DELETE FROM embeddings_dino WHERE file_id=?', (ids['img3.jpg'],))
    db.conn.commit()
    calls['chunks'].clear()
    assert c.post('/api/siglip-dino/start').json()['total'] == 1
    _wait(c)
    assert [os.path.basename(p) for ch in calls['chunks'] for p in ch] == ['img3.jpg']
    assert db.count_dino_indexed() == 10


def test_job_resumes_after_fatal_error(env, monkeypatch):
    app, c, db, errors, manual, ids, calls, _ = env
    monkeypatch.setattr(web, '_get_siglip_embedder', lambda: FakeEmb('siglip'))
    monkeypatch.setattr(web, '_get_dino_embedder', lambda: FakeEmb('dino'))
    # tiny chunk size via many files is impractical; fail the very first call instead
    calls['raise_after'] = 0
    c.post('/api/siglip-dino/start')
    st = _wait(c)
    assert 'boom: no GPU' in st['error'] and db.count_siglip_indexed() == 0
    calls['raise_after'] = None
    assert c.post('/api/siglip-dino/start').json()['total'] == 10
    assert _wait(c)['error'] is None and db.count_siglip_indexed() == 10


def test_failure_isolation_and_missing_file(env):
    app, c, db, errors, manual, ids, calls, tmp = env
    calls['bad'].add('img2.jpg')
    os.remove(tmp / 'img5.jpg')
    c.post('/api/siglip-dino/start')
    st = _wait(c)
    assert st['error'] is None and st['done'] == 10 and st['failed'] == 2
    assert db.count_siglip_indexed() == 8 and db.count_dino_indexed() == 8
    msgs = [r[1] for r in errors.conn.execute('SELECT path, message FROM errors').fetchall()]
    assert any('cannot decode' in m for m in msgs) and any('not on disk' in m for m in msgs)


def test_trashed_skipped_unless_included(env):
    app, c, db, errors, manual, ids, calls, _ = env
    manual.add_to_trash('sum1')
    assert c.post('/api/siglip-dino/start').json()['total'] == 9
    _wait(c)
    assert c.post('/api/siglip-dino/start?include_trashed=1').json()['total'] == 1


def test_no_gpu_runtimeerror_surfaces(env, monkeypatch):
    app, _c, *_ = env
    c = TestClient(app, raise_server_exceptions=False)
    def boom():
        raise RuntimeError('SigLIP2Embedder: no GPU available')
    monkeypatch.setattr(web, '_get_siglip_embedder', boom)
    r = c.post('/api/siglip-dino/start')
    assert r.status_code == 500 and 'no GPU available' in r.json()['detail']
    r = c.post('/api/embed-speedtest', json={'sample': 4})
    assert r.status_code == 500 and 'no GPU available' in r.json()['detail']


def test_speedtest_math_no_writes(env):
    app, c, db, errors, manual, ids, calls, _ = env
    # index 4 files first so remaining = 6
    for name in ['img0.jpg', 'img1.jpg', 'img2.jpg', 'img3.jpg']:
        db.insert_siglip_embedding(ids[name], _unit(1, 6).tobytes(), 'fake-siglip/v1')
        db.insert_dino_embedding(ids[name], _unit(1, 4).tobytes(), 'fake-dino/v1')
    calls['chunks'].clear()
    r = c.post('/api/embed-speedtest', json={'sample': 5}).json()
    assert r['library_total'] == 10 and r['remaining'] == 6 and r['sample'] == 5
    assert len(calls['chunks']) == 2  # warm-up + timed
    assert r['gpu_images_per_sec'] is None and r['breakdown']['gpu_seconds'] is None
    assert r['images_per_sec'] > 0
    assert r['est_seconds_total'] == pytest.approx(10 / r['images_per_sec'], rel=0.05, abs=0.2)
    assert r['est_seconds_remaining'] == pytest.approx(6 / r['images_per_sec'], rel=0.05, abs=0.2)
    assert db.count_siglip_indexed() == 4 and db.count_dino_indexed() == 4  # no DB writes


def test_speedtest_remote_gpu_breakdown(env, monkeypatch):
    app, c, *_ = env
    monkeypatch.setattr(web, '_embed_pair_chunk',
                        lambda s, d, paths: ([(np.zeros(6), np.zeros(4))] * len(paths), [None] * len(paths),
                                             0.5, ('rs/x', 'rd/y')))
    r = c.post('/api/embed-speedtest', json={'sample': 4}).json()
    assert r['gpu_images_per_sec'] == pytest.approx(4 / 0.5, rel=0.01)
    assert r['models'] == {'siglip': 'rs/x', 'dino': 'rd/y'}
    assert r['breakdown']['gpu_seconds'] == 0.5


def _put(db, fid, space, vec):
    getattr(db, f'insert_{space}_embedding')(fid, np.asarray(vec, np.float32).tobytes(), f'fake-{space}/v1')


def test_search_ranking_and_frame_collapse(env):
    app, c, db, errors, manual, ids, calls, tmp = env
    q = _unit(7, 6)
    near = q * 0.95 + _unit(1, 6) * 0.05
    far = -q
    _put(db, ids['img0.jpg'], 'siglip', far)
    _put(db, ids['img1.jpg'], 'siglip', q)           # best
    _put(db, ids['img2.jpg'], 'siglip', near)        # second
    # a hidden captured still of a video: scores highest overall but must surface as the video
    Image.new('RGB', (8, 8)).save(tmp / 'vid.mp4.jpg')
    still = db.upsert_file_path('vid.mp4.jpg', 'sumstill', size=1, modified_time=1)
    db.set_file_hidden(still, True)
    video = db.upsert_file_path('vid.mp4', 'sumvid', size=1, modified_time=1)
    manual.add_frame_capture('sumstill', 'sumvid', 1000)
    _put(db, still, 'siglip', q * 0.99 + _unit(2, 6) * 0.01)
    # unrelated hidden non-still file must be dropped
    hid = db.upsert_file_path('hid.jpg', 'sumhid', size=1, modified_time=1)
    db.set_file_hidden(hid, True)
    _put(db, hid, 'siglip', q)
    res = c.get('/api/search/siglip-text?q=cat&limit=10').json()['results']
    got = [r['file_id'] for r in res]
    assert got[0] == ids['img1.jpg'] or got[0] == video
    assert set(got[:3]) == {ids['img1.jpg'], video, ids['img2.jpg']}
    assert still not in got and hid not in got
    assert got[-1] == ids['img0.jpg']
    assert all(set(r) == {'file_id', 'score'} for r in res)


def test_image_similarity_endpoints(env):
    app, c, db, errors, manual, ids, calls, _ = env
    a, b, d = ids['img0.jpg'], ids['img1.jpg'], ids['img2.jpg']
    for space in ('dino', 'siglip'):
        dim = DIMS[space]
        _put(db, a, space, _unit(10, dim))
        _put(db, b, space, _unit(10, dim) * 0.9 + _unit(11, dim) * 0.1)
        _put(db, d, space, -_unit(10, dim))
        r = c.get(f'/api/files/{a}/similar-{space}?limit=5').json()
        assert [x['file_id'] for x in r['results']] == [b, d]  # self excluded, best first
        assert r['results'][0]['score'] > r['results'][1]['score']
        r = c.get(f'/api/files/{ids["img4.jpg"]}/similar-{space}').json()
        assert r['no_embedding'] is True and r['results'] == []


def test_dim_mismatch_is_clear_409(env):
    app, c, db, errors, manual, ids, calls, _ = env
    _put(db, ids['img0.jpg'], 'siglip', _unit(1, 3))
    r = c.get('/api/search/siglip-text?q=x')
    assert r.status_code == 409 and 'dim' in r.json()['detail']


def test_embed_stats(env):
    app, c, db, *_ = env
    d = c.get('/api/embed-stats').json()
    assert set(d['spaces']) == {'clip', 'siglip', 'dino'} and d['library_images'] == 10
