"""Swipe-card suggestions for a set under a chosen embedding model (clip | siglip | dino).

All fakes: hand-built low-dimensional vectors written straight into the embedding tables
(no model, no GPU). CLIP keeps its absolute threshold; siglip/dino take the top-K pool."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from media_manager.web import create_app

DIMS = {'clip': 8, 'siglip': 10, 'dino': 8}


def _vec(dim, **c):
    """unit vector with components given as e0=..., e1=..."""
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
    # near the centroid; x leans toward the rejected direction (e1), y toward e2
    add('x', same(e0=1, e1=0.3))
    add('y', same(e0=1, e2=0.4))
    add('far', same(e3=1))                      # orthogonal: below any sane threshold
    add('noemb', {})                            # no embedding in any space
    add('clip_only', {'clip': _vec(8, e0=1, e4=0.05)})
    add('hid', same(e0=1, e5=0.05))
    db.set_file_hidden(ids['hid'][0])
    add('inset', same(e0=1, e6=0.1))            # already in another set
    db.conn.commit()

    sid = manual.create_set('S', 'st')
    for n in ('m1', 'm2'):
        manual.assign_file_to_set(ids[n][1], sid)
    other = manual.create_set('O', 'st')
    manual.assign_file_to_set(ids['inset'][1], other)
    manual.exclude_file_from_set(ids['rej'][1], sid)
    db.conn.commit()
    return app, db, manual, ids, sid, TestClient(app)


def _get(client, sid, model=None, **q):
    qs = ''.join(f'&{k}={v}' for k, v in q.items())
    if model:
        qs += f'&model={model}'
    r = client.post(f'/api/sets/{sid}/similar-files?limit=500{qs}', json={'exclude': []})
    assert r.status_code == 200, r.text
    return r.json()


def _order(data, ids):
    rev = {v[0]: k for k, v in ids.items()}
    return [rev[c['id']] for c in data['cards']]


def test_default_is_siglip_and_clip_still_selectable(env):
    _app, _db, _m, ids, sid, client = env
    default = _get(client, sid)
    siglip = _get(client, sid, 'siglip')
    assert _order(default, ids) == _order(siglip, ids)
    assert default['model'] == 'siglip' and default['threshold_applied'] is False
    assert default['cards'] == default['results']
    # Legacy CLIP stays selectable and behaves as before (absolute threshold, CLIP data only).
    clip = _get(client, sid, 'clip')
    order = _order(clip, ids)
    assert 'clip_only' in order and 'far' not in order     # threshold applied
    assert 'rej' not in order and 'm1' not in order and 'hid' in order  # CLIP path as before
    assert clip['model'] == 'clip' and clip['threshold_applied'] is True


def test_unknown_model_400(env):
    *_, sid, client = env
    r = client.post(f'/api/sets/{sid}/similar-files?model=bogus', json={'exclude': []})
    assert r.status_code == 400


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_model_routing_and_no_threshold(env, model):
    _app, _db, _m, ids, sid, client = env
    data = _get(client, sid, model, honor_negatives='false')   # pure-similarity order
    order = _order(data, ids)
    assert data['model'] == model and data['threshold_applied'] is False
    assert data['model_missing'] is False
    assert data['coverage'] == {'members_embedded': 2, 'members_total': 2}
    assert 'clip_only' not in order and 'noemb' not in order   # no embedding in this space
    assert 'far' in order                                       # no absolute threshold
    assert order.index('x') < order.index('y') < order.index('far')
    # hard exclusions: members, rejected, hidden
    assert not {'m1', 'm2', 'rej', 'hid'} & set(order)
    assert all(isinstance(c['score'], float) for c in data['cards'])


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_toggles(env, model):
    _app, db, manual, ids, sid, client = env
    # avoid_existing
    assert 'inset' not in _order(_get(client, sid, model), ids)
    assert 'inset' in _order(_get(client, sid, model, avoid_existing='false'), ids)
    # honor_negatives: rejected photo's own space pulls x (leans to rej) below y
    base = _order(_get(client, sid, model, honor_negatives='false'), ids)
    neg = _order(_get(client, sid, model, honor_negatives='true'), ids)
    assert base.index('x') < base.index('y')
    assert neg.index('y') < neg.index('x')
    # default (field omitted) == on
    assert _order(_get(client, sid, model), ids) == neg
    # offset paging + exclude_ids
    r = client.post(f'/api/sets/{sid}/similar-files?limit=2&offset=1&model={model}', json={'exclude': []})
    assert _order(r.json(), ids) == neg[1:3]
    r = client.post(f'/api/sets/{sid}/similar-files?limit=2&model={model}',
                    json={'exclude': [str(ids[neg[0]][0])]})
    assert _order(r.json(), ids) == neg[1:3]


def test_negatives_change_clip_too(env):
    _app, _db, _m, ids, sid, client = env
    assert 'x' in _order(_get(client, sid, 'clip', honor_negatives='false'), ids)
    neg = _order(_get(client, sid, 'clip', honor_negatives='true'), ids)
    assert 'x' not in neg and 'y' in neg   # x falls under the CLIP threshold once steered away


def test_faces_filter_shared(env):
    _app, _db, manual, ids, sid, client = env
    # a confirmed person on a member, a different person on x
    manual.assign_identity_to_photo(ids['m1'][1], 'Alice')
    manual.assign_identity_to_photo(ids['x'][1], 'Bob')
    for model in ('clip', 'siglip', 'dino'):
        off = _order(_get(client, sid, model, exclude_non_matching_faces='false',
                          honor_negatives='false'), ids)
        on = _order(_get(client, sid, model, exclude_non_matching_faces='true',
                         honor_negatives='false'), ids)
        assert 'x' in off and 'x' not in on and 'y' in on
        # omitted field == on
        assert _order(_get(client, sid, model, honor_negatives='false'), ids) == on


@pytest.mark.parametrize('model', ['siglip', 'dino'])
def test_model_missing_signal(env, model):
    app, db, manual, ids, sid, client = env
    s2 = manual.create_set('Empty-in-model', 'st')
    manual.assign_file_to_set(ids['clip_only'][1], s2)   # member only has CLIP
    data = _get(client, s2, model)
    assert data['model_missing'] is True and data['cards'] == []
    assert data['coverage'] == {'members_embedded': 0, 'members_total': 1}
    assert _get(client, s2, 'clip')['model_missing'] is False
    # partial coverage: skipped, not missing
    manual.assign_file_to_set(ids['noemb'][1], sid)
    d = _get(client, sid, model)
    assert d['model_missing'] is False
    assert d['coverage'] == {'members_embedded': 2, 'members_total': 3}


def test_frame_rule_maps_stills_to_video(env):
    app, db, manual, ids, sid, client = env
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    for sp in ('siglip', 'dino'):
        _put(db, sp, still, _vec(DIMS[sp], e0=1, e1=0.05))   # only the still is embedded
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    for model in ('siglip', 'dino'):
        got = [c['id'] for c in _get(client, sid, model)['cards']]
        assert vid in got and still not in got
        assert got.count(vid) == 1


def test_topk_cap(env):
    app, db, manual, ids, sid, client = env
    for i in range(450):
        fid = db.upsert_file_path(f'bulk{i}.jpg', f'b{i:038d}', size=1)
        v = _vec(8, e0=1, e1=0.5 + i / 1000)
        _put(db, 'dino', fid, v)
        _put(db, 'clip', fid, _vec(8, e3=1, e4=1 + i))   # all far in CLIP space
    db.conn.commit()
    assert len(_get(client, sid, 'dino')['cards']) == 400
    assert len(_get(client, sid, 'clip')['cards']) < 400  # threshold, not top-K
