"""Location 'combined' model (DINOv2 + SigLIP 2 z-score fusion) and the location text search.
All fakes: tiny hand-built vectors, stub text embedder; no model, no GPU."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from media_manager import web
from media_manager.similarity import fuse_zscores
from media_manager.web import create_app

D = {'siglip': 10, 'dino': 8}


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


def _put(db, space, fid, v):
    getattr(db, f'insert_{space}_embedding')(fid, np.asarray(v, np.float32).tobytes(), f'{space}-t')


@pytest.fixture
def env(tmp_path, monkeypatch):
    os.makedirs(tmp_path / '.media')
    app = create_app(str(tmp_path))
    db, _e, manual = app.state.dbs
    ids = {}

    def add(name, sig=None, dino=None):
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = db.upsert_file_path(name + '.jpg', cs, size=10)
        if sig is not None:
            _put(db, 'siglip', fid, _vec(10, **sig))
        if dino is not None:
            _put(db, 'dino', fid, _vec(8, **dino))
        ids[name] = (fid, cs)

    add('m1', {'e0': 1}, {'e0': 1})
    add('m2', {'e0': 1, 'e1': .01}, {'e0': 1})
    add('rej', {'e1': 1}, {'e1': 1})
    add('both', {'e0': 1, 'e2': .1}, {'e0': 1, 'e2': .1})        # close in both spaces
    add('dino_only_close', {'e5': 1}, {'e0': 1, 'e3': .05})       # same layout, different scene
    add('sig_only_close', {'e0': 1, 'e4': .05}, {'e6': 1})        # same scene, different layout
    add('nodino', {'e0': 1, 'e7': .2}, None)                      # SigLIP only
    add('far', {'e8': 1}, {'e7': 1})
    add('placed', {'e0': 1, 'e9': .1}, {'e0': 1, 'e4': .1})
    add('xneg', {'e0': 1, 'e1': .4}, {'e0': 1, 'e1': .4})        # leans toward rejected
    db.conn.commit()
    loc = manual.create_location('L')
    for n in ('m1', 'm2'):
        manual.add_file_location(ids[n][1], loc)
    manual.add_file_location(ids['placed'][1], manual.create_location('Other'))
    manual.exclude_file_from_location(ids['rej'][1], loc)
    db.conn.commit()
    return app, db, manual, ids, loc, TestClient(app), monkeypatch


def _post(client, loc, **q):
    qs = '&'.join(f'{k}={v}' for k, v in q.items())
    r = client.post(f'/api/locations/{loc}/similar-files?limit=100&{qs}', json={'exclude': []})
    assert r.status_code == 200, r.text
    return r.json()


def _names(data, ids):
    rev = {v[0]: k for k, v in ids.items()}
    return [rev[c['id']] for c in data['cards']]


def test_fuse_zscores_math():
    ids = np.array([1, 2, 3], np.int64)
    a = (ids, ['a', 'b', 'c'], np.array([0.9, 0.5, 0.1], np.float32), 0.5)
    b = (np.array([2, 3, 4], np.int64), ['b', 'c', 'd'], np.array([0.2, 0.8, 0.4], np.float32), 0.5)
    u, cs, rank, disp = fuse_zscores([a, b])
    assert list(u) == [1, 2, 3, 4] and cs == ['a', 'b', 'c', 'd']
    # file 1 only in space a: scored by a alone (its z), file 3 by the mean of both
    za = (a[2] - a[2].mean()) / a[2].std()
    assert rank[0] == pytest.approx(za[0], abs=1e-5)
    assert disp[0] == pytest.approx(0.9, abs=1e-5)
    assert disp[2] == pytest.approx(0.5 * 0.1 + 0.5 * 0.8, abs=1e-5)
    assert fuse_zscores([])[2].shape == (0,)


def test_default_combined_ranking(env):
    _app, _db, _m, ids, loc, client, _mp = env
    d = _post(client, loc, model='combined')
    names = _names(d, ids)
    assert d['model'] == 'combined' and d['threshold_applied'] is False and d['background_applied'] is False
    assert d['coverage']['members_total'] == 2 and d['coverage']['per_space'] == {'dino': 2, 'siglip': 2}
    assert not {'m1', 'm2', 'rej', 'placed'} & set(names)           # members/rejected/avoid_existing
    assert names[0] == 'both'                                         # best in both spaces
    assert names.index('both') < names.index('far')
    assert 'nodino' in names                                          # single-space file still ranked
    # unlike either single space, both partial matches outrank the unrelated photo
    assert names.index('dino_only_close') < names.index('far')
    assert names.index('sig_only_close') < names.index('far')
    assert 'placed' in _names(_post(client, loc, avoid_existing='false'), ids)


def test_negatives_and_pagination(env):
    _app, _db, _m, ids, loc, client, _mp = env
    base_d = _post(client, loc)
    neg_d = _post(client, loc, honor_negatives='true')
    base, neg = _names(base_d, ids), _names(neg_d, ids)
    assert 'rej' not in base and 'rej' not in neg
    # The Rocchio negative lowers the score of the candidate leaning toward the rejected
    # photo (the z-score fusion rank can stay put when everything else drops too).
    score = lambda d, n: next(c['score'] for c in d['cards'] if c['id'] == ids[n][0])
    assert score(neg_d, 'xneg') < score(base_d, 'xneg') - 0.1
    assert neg.index('xneg') >= base.index('xneg')
    r = client.post(f'/api/locations/{loc}/similar-files?limit=2&offset=1', json={'exclude': []})
    assert _names(r.json(), ids) == base[1:3]
    r = client.post(f'/api/locations/{loc}/similar-files?limit=2',
                    json={'exclude': [str(ids[base[0]][0])]})
    assert _names(r.json(), ids) == base[1:3]


def test_one_space_missing_falls_back(env):
    _app, db, manual, ids, loc, client, _mp = env
    loc2 = manual.create_location('SigOnly')
    manual.add_file_location(ids['nodino'][1], loc2)               # member has no DINO embedding
    d = _post(client, loc2)
    assert d['coverage']['per_space'] == {'dino': 0, 'siglip': 1}
    assert d['model_missing'] is False and d['cards']
    loc3 = manual.create_location('Nothing')
    cs = 'z' * 40
    db.upsert_file_path('plain.jpg', cs, size=1)
    db.conn.commit()
    manual.add_file_location(cs, loc3)
    d3 = _post(client, loc3)
    assert d3['model_missing'] is True and d3['cards'] == []


def test_frame_rule_for_combined(env):
    _app, db, manual, ids, loc, client, _mp = env
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    _put(db, 'siglip', still, _vec(10, e0=1, e1=.02))
    _put(db, 'dino', still, _vec(8, e0=1, e1=.02))
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    got = [c['id'] for c in _post(client, loc)['cards']]
    assert vid in got and still not in got and got.count(vid) == 1


def test_unknown_model_400(env):
    _app, _db, _m, _ids, loc, client, _mp = env
    assert client.post(f'/api/locations/{loc}/similar-files?model=nope', json={'exclude': []}).status_code == 400


class _Emb:
    def embed_text(self, q):
        return _vec(10, e0=1)


def test_text_search(env):
    _app, db, manual, ids, loc, client, mp = env
    mp.setattr(web, '_get_siglip_embedder', lambda: _Emb())
    r = client.get(f'/api/locations/{loc}/text-search?q=beach at sunset&limit=50')
    assert r.status_code == 200, r.text
    names = _names(r.json(), ids)
    assert names and not {'m1', 'm2', 'rej', 'placed'} & set(names)
    assert 'far' not in names[:3]
    assert 'placed' in _names(client.get(f'/api/locations/{loc}/text-search?q=x&avoid_existing=false&limit=50').json(), ids)
    page = client.get(f'/api/locations/{loc}/text-search?q=x&limit=2&offset=1').json()
    assert _names(page, ids) == names[1:3] and page['has_more'] is True
    assert client.get(f'/api/locations/{loc}/text-search?q=%20').status_code == 400
    assert client.get('/api/locations/9999/text-search?q=a').status_code == 404


def test_text_search_embedder_down_and_frame_rule(env):
    _app, db, manual, ids, loc, client, mp = env

    def boom():
        raise RuntimeError('worker down')
    mp.setattr(web, '_get_siglip_embedder', boom)
    assert client.get(f'/api/locations/{loc}/text-search?q=a').status_code == 502
    mp.setattr(web, '_get_siglip_embedder', lambda: _Emb())
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    _put(db, 'siglip', still, _vec(10, e0=1))
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    got = [c['id'] for c in client.get(f'/api/locations/{loc}/text-search?q=a&limit=50').json()['cards']]
    assert still not in got and vid in got


def test_location_page_renders(env):
    _app, _db, _m, _ids, loc, client, _mp = env
    html = client.get(f'/locations/{loc}').text
    assert 'value="combined"' in html and 'loc-text-form' in html
