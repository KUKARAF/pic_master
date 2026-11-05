"""Swipe-card suggestions for a location under a chosen embedding model (clip | siglip | dino).

All fakes: hand-built low-dimensional vectors written straight into the embedding tables
(no model, no GPU). CLIP keeps its absolute threshold (and the optional background-tile
ranking); siglip/dino take the top-K pool in their own space and ignore `background`."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from media_manager.web import create_app

DIMS = {'clip': 8, 'siglip': 10, 'dino': 8}


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


def _put(db, space, fid, vec):
    b = np.asarray(vec, np.float32).tobytes()
    if space == 'clip':
        db.insert_embedding(fid, b, 'clip-test')
    else:
        getattr(db, f'insert_{space}_embedding')(fid, b, f'{space}-test')


@pytest.fixture
def env(tmp_path):
    os.makedirs(tmp_path / '.media')
    app = create_app(str(tmp_path))
    db, _errors, manual = app.state.dbs
    ids = {}

    def add(name, vecs, ext='.jpg'):
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = db.upsert_file_path(name + ext, cs, size=10)
        for space, v in vecs.items():
            _put(db, space, fid, v)
        ids[name] = (fid, cs)
        return fid

    def same(**c):
        return {s: _vec(d, **c) for s, d in DIMS.items()}

    add('m1', same(e0=1))
    add('m2', same(e0=1, e1=0.01))
    add('rej', same(e1=1))
    add('x', same(e0=1, e1=0.3))                 # leans toward the rejected direction
    add('y', same(e0=1, e2=0.4))
    add('far', same(e3=1))
    add('noemb', {})
    add('clip_only', {'clip': _vec(8, e0=1, e4=0.05)})
    add('hid', same(e0=1, e5=0.05))
    db.set_file_hidden(ids['hid'][0])
    add('placed', same(e0=1, e6=0.1))            # already has another location
    add('settled', same(e0=1, e7=0.1))           # member of a set linked to this location
    db.conn.commit()

    loc = manual.create_location('L')
    for n in ('m1', 'm2'):
        manual.add_file_location(ids[n][1], loc)
    other = manual.create_location('Other')
    manual.add_file_location(ids['placed'][1], other)
    manual.exclude_file_from_location(ids['rej'][1], loc)
    sid = manual.create_set('S', 'st')
    manual.assign_file_to_set(ids['settled'][1], sid)
    manual.add_set_location(sid, loc)
    db.conn.commit()
    return app, db, manual, ids, loc, TestClient(app)


def _get(client, loc, model=None, **q):
    qs = ''.join(f'&{k}={v}' for k, v in q.items())
    if model:
        qs += f'&model={model}'
    r = client.post(f'/api/locations/{loc}/similar-files?limit=500{qs}', json={'exclude': []})
    assert r.status_code == 200, r.text
    return r.json()


def _order(data, ids):
    rev = {v[0]: k for k, v in ids.items()}
    return [rev[c['id']] for c in data['cards']]


def test_default_is_place_and_clip_still_selectable(env):
    _app, _db, _m, ids, loc, client = env
    default = _get(client, loc)
    comb = _get(client, loc, 'combined')
    # no photo here has person data and no member has GPS/sets, so 'place' == 'combined' order
    assert _order(default, ids) == _order(comb, ids)
    assert default['model'] == 'place' and comb['model'] == 'combined' and default['threshold_applied'] is False
    assert default['background_applied'] is False and default['cards'] == default['results']
    clip = _get(client, loc, 'clip', background='false')
    order = _order(clip, ids)
    assert 'clip_only' in order and 'far' not in order            # legacy: threshold applied
    assert not {'rej', 'm1', 'settled', 'placed'} & set(order)
    assert clip['model'] == 'clip' and clip['threshold_applied'] is True and clip['mode'] == 'whole'


def test_unknown_model_400(env):
    *_, loc, client = env
    r = client.post(f'/api/locations/{loc}/similar-files?model=bogus', json={'exclude': []})
    assert r.status_code == 400


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_model_routing_and_no_threshold(env, model):
    _app, _db, _m, ids, loc, client = env
    data = _get(client, loc, model, honor_negatives='false')   # pure-similarity order
    order = _order(data, ids)
    assert data['model'] == model and data['threshold_applied'] is False
    assert data['background_applied'] is False and data['model_missing'] is False
    assert data['coverage'] == {'members_embedded': 2, 'members_total': 2}
    assert 'clip_only' not in order and 'noemb' not in order
    assert 'far' in order
    assert order.index('x') < order.index('y') < order.index('far')
    # members, rejected, hidden, set-settled and already-placed are all excluded
    assert not {'m1', 'm2', 'rej', 'hid', 'settled', 'placed'} & set(order)


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_toggles(env, model):
    _app, _db, _m, ids, loc, client = env
    assert 'placed' not in _order(_get(client, loc, model), ids)
    assert 'placed' in _order(_get(client, loc, model, avoid_existing='false'), ids)
    # settled (set linked to the location) stays hidden even without avoid_existing
    assert 'settled' not in _order(_get(client, loc, model, avoid_existing='false'), ids)
    base = _order(_get(client, loc, model, honor_negatives='false'), ids)
    neg = _order(_get(client, loc, model, honor_negatives='true'), ids)
    assert base.index('x') < base.index('y')
    assert neg.index('y') < neg.index('x')
    assert _order(_get(client, loc, model), ids) == neg      # default (omitted) == on
    assert 'rej' not in base and 'rej' not in neg
    r = client.post(f'/api/locations/{loc}/similar-files?limit=2&offset=1&model={model}', json={'exclude': []})
    assert _order(r.json(), ids) == neg[1:3]
    r = client.post(f'/api/locations/{loc}/similar-files?limit=2&model={model}',
                    json={'exclude': [str(ids[neg[0]][0])]})
    assert _order(r.json(), ids) == neg[1:3]


def test_negatives_change_clip_too(env):
    _app, _db, _m, ids, loc, client = env
    assert 'x' in _order(_get(client, loc, 'clip', background='false', honor_negatives='false'), ids)
    neg = _order(_get(client, loc, 'clip', background='false', honor_negatives='true'), ids)
    assert 'x' not in neg and 'y' in neg


@pytest.mark.parametrize('model', ['clip', 'siglip', 'dino'])
def test_background_param_ignored(env, model):
    _app, _db, _m, ids, loc, client = env
    on = _get(client, loc, model, background='true')
    off = _get(client, loc, model, background='false')
    assert on['mode'] == 'whole' and on['background_applied'] is False
    assert _order(on, ids) == _order(off, ids)


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_model_missing_signal(env, model):
    _app, _db, manual, ids, loc, client = env
    loc2 = manual.create_location('Clip-only')
    manual.add_file_location(ids['clip_only'][1], loc2)
    data = _get(client, loc2, model)
    assert data['model_missing'] is True and data['cards'] == []
    assert data['coverage'] == {'members_embedded': 0, 'members_total': 1}
    assert _get(client, loc2, 'clip', background='false')['model_missing'] is False
    manual.add_file_location(ids['noemb'][1], loc)
    d = _get(client, loc, model)
    assert d['model_missing'] is False
    assert d['coverage'] == {'members_embedded': 2, 'members_total': 3}


def test_frame_rule_maps_stills_to_video(env):
    _app, db, manual, ids, loc, client = env
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    for sp in ('siglip', 'dino'):
        _put(db, sp, still, _vec(DIMS[sp], e0=1, e1=0.05))
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    for model in ('siglip', 'dino'):
        got = [c['id'] for c in _get(client, loc, model)['cards']]
        assert vid in got and still not in got
        assert got.count(vid) == 1


def test_topk_cap(env):
    _app, db, _m, _ids, loc, client = env
    for i in range(450):
        fid = db.upsert_file_path(f'bulk{i}.jpg', f'b{i:038d}', size=1)
        _put(db, 'dino', fid, _vec(8, e0=1, e1=0.5 + i / 1000))
        _put(db, 'clip', fid, _vec(8, e3=1, e4=1 + i))
    db.conn.commit()
    assert len(_get(client, loc, 'dino')['cards']) == 400
    assert len(_get(client, loc, 'clip', background='false')['cards']) < 400
