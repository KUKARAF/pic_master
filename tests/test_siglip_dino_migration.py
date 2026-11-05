"""CLIP -> SigLIP 2 + DINOv2 migration of the similarity / search / background-job paths.

Fake embedders only (no GPU, no model download). Covers:
  * /api/files/{id}/similar defaults to SigLIP 2 (stills folded into their video, trashed /
    hidden dropped), ?model=dino works, ?model=clip is the legacy path, unknown model is 400;
  * /api/index/start (the old "Index (CLIP)" bulk endpoint) now runs the SigLIP 2 + DINOv2 job
    and never touches the CLIP indexer;
  * upload / captured-still ingest hooks (_ensure_file_embedding) write SigLIP 2 + DINOv2 rows
    and never CLIP rows;
  * "Likely sets" (suggested-sets) rank in SigLIP 2 by default, CLIP only when asked;
  * category suggestions default to SigLIP 2 (no temperature threshold), honor negatives and the
    frame rule, and keep a legacy CLIP mode;
  * /similar page offers the CLIP button only while CLIP data exists;
  * a video with only captured stills is queryable through them;
  * the worker's --preload loads SigLIP 2 + DINOv2 (and no longer CLIP).
"""
import io
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import web

DIMS = {'siglip': 8, 'dino': 6, 'clip': 8}


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmb:
    batch_size = 4
    workers = 1

    def __init__(self, which):
        self.which = which

    def model_id(self):
        from media_manager.embedders import DINOv2Embedder, SigLIP2Embedder
        return (SigLIP2Embedder if self.which == 'siglip' else DINOv2Embedder).model_id()


class ExplodingClip:
    """Any CLIP use fails the test: no background job / ingest hook may compute CLIP."""

    def __getattr__(self, name):
        raise AssertionError(f'CLIP indexer used ({name})')


@pytest.fixture
def env(tmp_path, monkeypatch):
    os.makedirs(tmp_path / '.media')
    shared_calls = []

    def fake_shared(embs, paths, stats=None):
        shared_calls.extend(paths)
        n = len(paths)
        return [np.tile(_vec(DIMS['siglip'], e0=1), (n, 1)), np.tile(_vec(DIMS['dino'], e0=1), (n, 1))], []

    monkeypatch.setattr(web, '_worker_client', None)
    monkeypatch.setattr(web, '_get_siglip_embedder', lambda: FakeEmb('siglip'))
    monkeypatch.setattr(web, '_get_dino_embedder', lambda: FakeEmb('dino'))
    monkeypatch.setattr(web, '_local_embed_shared', fake_shared)
    monkeypatch.setattr(web, '_get_clip_indexer', lambda: ExplodingClip())
    app = web.create_app(str(tmp_path))
    app.state.test_root = str(tmp_path)
    db, _errors, manual = app.state.dbs
    ids, cs = {}, {}

    def add(name, vecs=None, ext='.jpg'):
        c = (name * 12)[:40].ljust(40, 'x')
        p = tmp_path / (name + ext)
        Image.new('RGB', (8, 8), (len(name) * 20, 0, 0)).save(p, format='PNG' if ext == '.png' else 'JPEG')
        fid = db.upsert_file_path(name + ext, c, size=os.path.getsize(p), modified_time=1)
        for space, v in (vecs or {}).items():
            b = np.asarray(v, np.float32).tobytes()
            if space == 'clip':
                db.insert_embedding(fid, b, 'clip-test')
            else:
                getattr(db, f'insert_{space}_embedding')(fid, b, f'{space}-test')
        ids[name], cs[name] = fid, c
        return fid

    def both(**c):
        return {s: _vec(DIMS[s], **c) for s in ('siglip', 'dino')}

    add('q', both(e0=1))
    add('near', both(e0=1, e1=0.2))
    add('mid', both(e0=1, e1=1))
    add('far', both(e3=1))
    add('hid', both(e0=1, e1=0.01))
    db.set_file_hidden(ids['hid'])
    add('trash', both(e0=1, e1=0.05))
    manual.add_to_trash(cs['trash'])
    add('cliponly', {'clip': _vec(8, e0=1)})
    db.conn.commit()
    return app, db, manual, ids, cs, TestClient(app), shared_calls


def _fids(res):
    return [r['file_id'] for r in res['results']]


def test_similar_api_defaults_to_siglip_and_folds_stills(env):
    app, db, manual, ids, cs, c, _ = env
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/s1.jpg', 's' * 40, size=1)
    db.insert_siglip_embedding(still, _vec(8, e0=1, e1=0.05).tobytes(), 'siglip-test')
    db.insert_dino_embedding(still, _vec(6, e0=1, e1=0.05).tobytes(), 'dino-test')
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    res = c.get(f'/api/files/{ids["q"]}/similar?limit=10').json()
    got = _fids(res)
    assert got[0] == vid or got[0] == ids['near']              # near-identical vectors first
    assert vid in got and still not in got                     # still folded into its video
    assert ids['hid'] not in got and ids['trash'] not in got   # hidden / trashed dropped
    assert ids['cliponly'] not in got and ids['q'] not in got  # CLIP-only data is invisible
    assert got.index(ids['near']) < got.index(ids['mid']) < got.index(ids['far'])
    assert c.get(f'/api/files/{ids["q"]}/similar-siglip').json()['results'] == res['results'][:50]
    dino = c.get(f'/api/files/{ids["q"]}/similar?model=dino').json()
    assert ids['near'] in _fids(dino) and 'cards' in dino
    # legacy CLIP path still answers from the CLIP table; unknown model is a 400
    db.insert_embedding(ids['q'], _vec(8, e0=1).tobytes(), 'clip-test')
    clip = c.get(f'/api/files/{ids["q"]}/similar?model=clip').json()
    assert _fids(clip) == [ids['cliponly']]
    assert c.get(f'/api/files/{ids["q"]}/similar?model=bogus').status_code == 400


def test_video_with_only_stills_is_queryable_through_them(env):
    app, db, manual, ids, cs, c, _ = env
    vid = db.upsert_file_path('only_stills.mp4', 'w' * 40, size=1)
    still = db.upsert_file_path('captured_frames/s2.jpg', 'u' * 40, size=1)
    db.insert_siglip_embedding(still, _vec(8, e0=1).tobytes(), 'siglip-test')
    db.set_file_hidden(still)
    manual.add_frame_capture('u' * 40, 'w' * 40, 5)
    db.conn.commit()
    res = c.get(f'/api/files/{vid}/similar-siglip').json()
    assert not res.get('no_embedding') and ids['q'] in _fids(res) and vid not in _fids(res)
    assert c.get(f'/api/files/{vid}/similar-dino').json()['no_embedding'] is True


def test_index_alias_runs_siglip_dino_job_not_clip(env):
    app, db, manual, ids, cs, c, shared = env
    Image.new('RGB', (8, 8), (50, 0, 0)).save(os.path.join(app.state.test_root, 'fresh.png'))
    fresh = db.upsert_file_path('fresh.png', 'f' * 40, size=1, modified_time=1)
    db.conn.commit()
    r = c.post('/api/index/start').json()
    assert r['started'] is True, r
    for _ in range(300):
        st = c.get('/api/index/status').json()
        if not st['running']:
            break
        time.sleep(0.02)
    assert st['error'] is None and 'img_per_sec' in st, st           # the pair job's status shape
    assert db.get_space_embedding('siglip', fresh) is not None
    assert db.get_space_embedding('dino', fresh) is not None
    assert db.get_embedding(fresh) is None                           # CLIP indexer never ran


def test_upload_hook_writes_siglip_dino_not_clip(env):
    app, db, manual, ids, cs, c, shared = env
    buf = io.BytesIO()
    Image.new('RGB', (16, 16), (1, 2, 3)).save(buf, format='PNG')
    r = c.post('/api/files/upload', files=[('file', ('new.png', buf.getvalue(), 'image/png'))])
    assert r.status_code == 200, r.text
    fid = r.json()['saved'][0]['id']
    for _ in range(200):
        if db.get_space_embedding('dino', fid) is not None:
            break
        time.sleep(0.02)
    assert db.get_space_embedding('siglip', fid) is not None
    assert db.get_space_embedding('dino', fid) is not None
    assert db.get_embedding(fid) is None                      # nothing CLIP
    assert any(p.endswith('new.png') for p in shared)


def test_embed_now_uses_new_models(env):
    app, db, manual, ids, cs, c, shared = env
    Image.new('RGB', (8, 8), (9, 9, 9)).save(os.path.join(app.state.test_root, 'fresh.png'))
    fresh = db.upsert_file_path('fresh.png', 'e' * 40, size=1, modified_time=1)
    db.conn.commit()
    assert c.post(f'/api/files/{fresh}/embed').json() == {'embedded': True}
    assert db.get_space_embedding('siglip', fresh) is not None and db.get_embedding(fresh) is None
    assert db.get_space_embedding('dino', fresh) is not None


def test_ensure_embedding_is_best_effort_and_loud(env, monkeypatch):
    """No GPU / worker down: the ingest hook must not raise (uploads still succeed) but logs."""
    app, db, manual, ids, cs, c, shared = env

    def no_gpu():
        raise RuntimeError('no GPU available')
    monkeypatch.setattr(web, '_get_siglip_embedder', no_gpu)
    buf = io.BytesIO()
    Image.new('RGB', (16, 16), (7, 7, 7)).save(buf, format='PNG')
    r = c.post('/api/files/upload', files=[('file', ('gpuless.png', buf.getvalue(), 'image/png'))])
    assert r.status_code == 200
    fid = r.json()['saved'][0]['id']
    errs = app.state.dbs[1]
    for _ in range(200):
        msgs = [m for _p, m in errs.conn.execute('SELECT path, message FROM errors').fetchall()]
        if any('no GPU available' in m for m in msgs):
            break
        time.sleep(0.02)
    assert any('ensure-embedding error' in m and 'no GPU available' in m for m in msgs), msgs
    assert db.get_space_embedding('siglip', fid) is None


def test_suggested_sets_rank_in_siglip_by_default(env):
    app, db, manual, ids, cs, c, _ = env
    sid = manual.create_set('Near', 'st')
    manual.assign_file_to_set(cs['near'], sid)
    far_sid = manual.create_set('Far', 'st')
    manual.assign_file_to_set(cs['far'], far_sid)
    db.conn.commit()
    res = c.get(f'/api/files/{ids["q"]}/suggested-sets?ids={sid},{far_sid}').json()['results']
    assert [r['id'] for r in res] == [sid, far_sid] and res[0]['score'] > res[1]['score']
    # default (no ids): no absolute CLIP threshold -> the far set is still listed after the near one
    res = c.get(f'/api/files/{ids["q"]}/suggested-sets').json()['results']
    assert [r['id'] for r in res][:1] == [sid]


def test_category_suggestions_default_siglip_with_negatives(env):
    app, db, manual, ids, cs, c, _ = env
    cat = manual.create_category('Cat', 0.99)           # temperature would reject everything in CLIP
    manual.add_file_category(cs['q'], cat)
    db.conn.commit()
    r = c.post(f'/api/categories/{cat}/suggestions/next?count=10', json={'exclude': []}).json()['cards']
    got = [x['file_id'] for x in r]
    assert ids['q'] not in got and ids['cliponly'] not in got
    assert ids['hid'] not in got                             # hidden dropped (trashed are kept, marked on the card)
    assert got.index(ids['near']) < got.index(ids['mid']) < got.index(ids['far'])   # top-K pool, no temperature threshold
    # negatives: rejecting 'near' removes it and steers away from its direction
    manual.exclude_file_from_category(cs['near'], cat)
    got2 = [x['file_id'] for x in
            c.post(f'/api/categories/{cat}/suggestions/next?count=10', json={'exclude': []}).json()['cards']]
    assert ids['near'] not in got2
    # legacy clip: temperature threshold applies and only CLIP data is read
    db.insert_embedding(ids['q'], _vec(8, e0=1).tobytes(), 'clip-test')
    got3 = c.post(f'/api/categories/{cat}/suggestions/next?model=clip', json={'exclude': []}).json()['cards']
    assert [x['file_id'] for x in got3] == [ids['cliponly']]  # CLIP data only; its cosine 1.0 clears 0.99
    assert c.post(f'/api/categories/{cat}/suggestions/next?model=bogus', json={'exclude': []}).status_code == 400


def test_similar_page_hides_clip_button_without_clip_data(env):
    app, db, manual, ids, cs, c, _ = env
    db.conn.execute('DELETE FROM embeddings')
    db.conn.commit()
    page = c.get(f'/similar/{ids["q"]}').text
    assert 'data-model="siglip"' in page and 'aria-current' in page
    assert 'data-model="clip"' not in page
    db.insert_embedding(ids['q'], _vec(8, e0=1).tobytes(), 'clip-test')
    page = c.get(f'/similar/{ids["q"]}').text
    assert 'data-model="clip"' in page and 'CLIP (legacy)' in page


def test_worker_preload_loads_siglip_and_dino_not_clip(monkeypatch):
    import signal
    import sys
    import types
    from media_manager import worker_server as ws
    loaded = []
    for name in ('get_face_detector', 'get_siglip_embedder', 'get_dino_embedder', 'get_object_detector'):
        monkeypatch.setattr(ws.models, name, (lambda n=name: loaded.append(n)))
    monkeypatch.setattr(ws.models, 'get_clip_indexer', lambda: loaded.append('get_clip_indexer'))
    monkeypatch.setattr(ws, 'build_app', lambda: object())
    fake_uvicorn = types.ModuleType('uvicorn')
    fake_uvicorn.run = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'uvicorn', fake_uvicorn)
    prev = signal.getsignal(signal.SIGTERM)
    try:
        ws.run(preload=True)
    finally:
        signal.signal(signal.SIGTERM, prev)
    for _ in range(200):
        if len(loaded) >= 4:
            break
        time.sleep(0.02)
    assert loaded == ['get_face_detector', 'get_siglip_embedder', 'get_dino_embedder', 'get_object_detector'], loaded
