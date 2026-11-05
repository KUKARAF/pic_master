"""'place' model stage 2: lazy person-masked re-embedding (location_mask.py), the worker ops
embed_masked_paths / embed_masked_batch, cache, budget and old-worker fallbacks. Fakes only."""
import os
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import location_mask as lm
from media_manager import location_place as lp
from media_manager import web, worker_client, worker_protocol, worker_server
from media_manager.web import create_app


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


# ------------------------------------------------------------------ pixels

def _scene(path, box=None, size=(200, 100)):
    im = Image.new('RGB', size, (0, 200, 0))                         # green "place"
    if box:
        x1, y1, x2, y2 = box
        im.paste((255, 0, 0), (int(x1 * size[0]), int(y1 * size[1]), int(x2 * size[0]), int(y2 * size[1])))
    im.save(path)
    return path


def test_mask_fills_person_with_mean_background(tmp_path):
    p = _scene(str(tmp_path / 's.png'), (0.0, 0.0, 0.5, 1.0))        # red "person" on the left half
    img, cov = lm.decode_masked(p, [[0.0, 0.0, 0.5, 1.0]])
    a = np.asarray(img)
    assert cov == pytest.approx(0.5, abs=0.01)
    assert (a[:, :90] == (0, 200, 0)).all() and (a[:, 110:] == (0, 200, 0)).all()   # red gone, green kept
    img2, cov2 = lm.mask_pil(Image.new('RGB', (10, 10), (1, 2, 3)), [[0, 0, 1, 1]])
    assert cov2 == 1.0 and np.asarray(img2)[0, 0].tolist() == [1, 2, 3]           # nothing left: unchanged
    big = Image.new('RGB', (2000, 1000))
    big.save(tmp_path / 'big.jpg')
    assert lm.decode_masked(str(tmp_path / 'big.jpg'), [])[0].size[0] <= lm.MASK_DECODE_LONG


# ------------------------------------------------------------------ plans

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'REBUILD_MIN_S', 0.0)
    os.makedirs(tmp_path / '.media')
    app = create_app(str(tmp_path))
    db, _e, manual = app.state.dbs
    ids = {}

    def add(name, v=None, scan=None, dims=(1000, 1000)):
        """scan: None = never scanned | 'clean' | float = one person box covering that fraction."""
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = db.upsert_file_path(f'{name}.jpg', cs, size=10)
        if v is not None:
            db.insert_siglip_embedding(fid, _vec(10, **v).tobytes(), 'sig-t')
            db.insert_dino_embedding(fid, _vec(8, **v).tobytes(), 'dino-t')
        db.conn.execute('UPDATE files SET width = ?, height = ? WHERE id = ?', (*dims, fid))
        if scan == 'clean':
            db.insert_detections(fid, [], 'y')
        elif scan is not None:
            side = int(1000 * scan ** 0.5)
            db.insert_detections(fid, [('person', 0.9, 0, 0, side, side)], 'y')
        ids[name] = (fid, cs)
        return fid
    return app, db, manual, ids, add, TestClient(app), monkeypatch


def test_plan_statuses(env):
    _a, db, _m, ids, add, _c, _mp = env
    add('un', {'e0': 1}, None)
    add('cl', {'e0': 1}, 'clean')
    add('tiny', {'e0': 1}, 0.01)
    add('mk', {'e0': 1}, 0.3)
    add('cov', {'e0': 1}, 0.8)
    db.conn.commit()
    pl = lm.make_plans(db, [v[0] for v in ids.values()])
    st = {n: pl[ids[n][0]].status for n in ids}
    assert st == {'un': 'unscanned', 'cl': 'clean', 'tiny': 'clean', 'mk': 'masked', 'cov': 'covered'}
    mk = pl[ids['mk'][0]]
    assert mk.hash and mk.coverage == pytest.approx(0.3, abs=0.05) and mk.boxes[0][2] == pytest.approx(0.5477, abs=0.01)
    # box hash changes when the person box changes -> cache miss by construction
    db.insert_detections(ids['mk'][0], [('person', 0.9, 0, 0, 400, 400)], 'y')
    assert lm.make_plans(db, [ids['mk'][0]])[ids['mk'][0]].hash != mk.hash


# ------------------------------------------------------------------ service: cache, budget, failure

def _svc(db, fn):
    return lm.MaskService(db, fn, chunk=4)


def _jobs(db, ids, names, tmp):
    plans = lm.make_plans(db, [ids[n][0] for n in names])
    return [lm.Job(ids[n][0], plans[ids[n][0]].hash, str(tmp / f'{n}.jpg'), plans[ids[n][0]].boxes, 0.3) for n in names]


def test_service_cache_hit_miss_and_budget(env, tmp_path):
    _a, db, _m, ids, add, _c, _mp = env
    for n in 'abcdef':
        add(n, {'e0': 1}, 0.3)
    db.conn.commit()
    calls = []
    gate = threading.Event()

    def embed(items):
        calls.append(len(items))
        gate.wait(5)
        return [(_vec(10, e1=1), _vec(8, e1=1)) for _ in items], [None] * len(items), ('s-m', 'd-m')
    svc = _svc(db, embed)
    jobs = _jobs(db, ids, 'abcdef', tmp_path)
    t0 = time.monotonic()
    assert svc.request(jobs, 0.2) == 6 and time.monotonic() - t0 < 2      # budget hit: still pending
    gate.set()                                                             # background keeps going
    assert svc.request(jobs, 5) == 0
    got = db.get_masked_embeddings([ids[n][0] for n in 'abcdef'], lm.RECIPE)
    assert len(got) == 6 and next(iter(got.values()))[2] == 's-m'
    assert sum(calls) == 6 and max(calls) <= 4                            # chunked, each job embedded once


def test_service_failure_and_backoff(env, tmp_path, monkeypatch):
    _a, db, _m, ids, add, _c, _mp = env
    for n in 'ab':
        add(n, {'e0': 1}, 0.3)
    db.conn.commit()

    def boom(items):
        raise lm.MaskUnavailable('worker_outdated', 'worker needs redeploy')
    svc = _svc(db, boom)
    assert svc.request(_jobs(db, ids, 'ab', tmp_path), 3) == 0            # queue cleared, no hang
    ok, kind, msg = svc.status()
    assert not ok and kind == 'worker_outdated' and 'redeploy' in msg
    svc.state['until'] = 0
    assert svc.status()[0] is True
    # per-item errors are remembered and not retried right away
    n = []

    def partial(items):
        n.append(1)
        return [None for _ in items], ['unreadable'] * len(items), None
    svc2 = _svc(db, partial)
    jobs = _jobs(db, ids, 'ab', tmp_path)
    assert svc2.request(jobs, 3) == 0 and svc2.request(jobs, 1) == 0 and len(n) == 1


def test_blend_math():
    s1 = {i: 10.0 - i for i in range(8)}
    masked = {i: float(i) for i in range(8)}                  # masked says the opposite order
    out = lm.blend(s1, masked, alpha=1.0)
    assert sorted(out, key=lambda k: -out[k]) == list(range(7, -1, -1))
    assert lm.blend(s1, masked, alpha=0.0) == s1
    few = lm.blend(s1, {0: 1.0, 1: 2.0})
    assert few == s1                                           # < MASK_MIN_SUBSET: untouched
    out = lm.blend(s1, {i: float(i) for i in range(6)})
    assert out[6] == s1[6] and out[7] == s1[7]                 # unscored keys keep stage 1


# ------------------------------------------------------------------ end to end through the API

@pytest.fixture
def place(env, tmp_path):
    """Location L with 3 members at e0; candidates: whole-image look-alikes whose similarity is
    due to a person (masked vector: elsewhere) and real place matches (masked vector: e0)."""
    app, db, manual, ids, add, client, mp = env
    for n in ('m1', 'm2', 'm3'):
        add(n, {'e0': 1}, 'clean')
    # "peopley": 30% person, stored vector on the place direction + a bit; masked vector unrelated
    for i in range(3):
        add(f'peopley{i}', {'e0': 1, 'e2': .02}, 0.3)
    for i in range(3):
        add(f'real{i}', {'e0': 1, 'e3': .15}, 0.3)               # slightly lower stage-1 score
    for i in range(3):
        add(f'clean{i}', {'e0': 1, 'e4': .1}, 'clean')
    add('unscanned', {'e0': 1, 'e5': .08}, None)
    add('covered', {'e0': 1, 'e6': .05}, 0.8)
    for i in range(6):
        add(f'far{i}', {f'e{6 + i % 2}': 1, 'e1': 1}, 'clean')
    db.conn.commit()
    loc = manual.create_location('L')
    for n in ('m1', 'm2', 'm3'):
        manual.add_file_location(ids[n][1], loc)
    db.conn.commit()
    calls = []

    def fake_chunk(items):
        calls.append([os.path.basename(p) for p, _b in items])
        pairs = []
        for p, _b in items:
            name = os.path.basename(p)[:-4]
            v = {'e0': 1} if name.startswith('real') or name.startswith('m') else {'e5': 1}
            pairs.append((_vec(10, **v), _vec(8, **v)))
        return pairs, [None] * len(items), ('sig-t', 'dino-t')
    mp.setattr(web, '_masked_embed_chunk', fake_chunk)
    return app, db, manual, ids, loc, client, calls, mp


def _q(client, loc, **kw):
    q = '&'.join(f'{k}={v}' for k, v in kw.items())
    r = client.post(f'/api/locations/{loc}/similar-files?limit=100&{q}', json={'exclude': []})
    assert r.status_code == 200, r.text
    d = r.json()
    rev = {}
    return d, [c['id'] for c in d['cards']]


def _name_of(ids):
    return {v[0]: k for k, v in ids.items()}


def test_stage2_demotes_people_matches_and_caches(place):
    _a, db, _m, ids, loc, client, calls, _mp = place
    nm = _name_of(ids)
    base, base_ids = _q(client, loc, mask='false')
    assert 'stage2' not in base
    order1 = [nm[i] for i in base_ids]
    assert order1.index('peopley0') < order1.index('real0')            # stage 1 prefers the peopley ones
    d, got = _q(client, loc)
    order2 = [nm[i] for i in got]
    s2 = d['stage2']
    assert s2['state'] == 'ok' and s2['masked'] == 6 and s2['clean'] >= 3 and s2['unscanned'] == 1
    assert s2['covered'] == 1 and s2['pending'] == 0 and s2['members_used'] == 3
    assert max(order2.index(f'real{i}') for i in range(3)) < min(order2.index(f'peopley{i}') for i in range(3))
    # members + 6 masked candidates were embedded once (members first), nothing for clean/unscanned/covered
    flat = [n for c in calls for n in c]
    assert sorted(flat) == sorted([f'{n}.jpg' for n in ('peopley0', 'peopley1', 'peopley2', 'real0', 'real1', 'real2')])
    # unscanned / covered keep their place in the stage-1 world and say so
    cards = {nm[c['id']]: c for c in d['cards']}
    assert cards['unscanned']['why']['stage2'] == 'unscanned' and cards['covered']['why']['stage2'] == 'covered'
    assert cards['real0']['why']['masked'] > cards['peopley0']['why']['masked']
    assert cards['real0']['why']['stage2'] == 'masked' and 'final' in cards['real0']['why']
    assert cards['clean0']['why']['stage2'] == 'clean'
    # second query: pure cache hit, no new worker call, same order
    n_before = len(calls)
    d2, got2 = _q(client, loc)
    assert len(calls) == n_before and got2 == got and d2['stage2']['pending'] == 0
    # a changed person box invalidates exactly that file's cache entry
    db.insert_detections(ids['real0'][0], [('person', 0.9, 0, 0, 400, 400)], 'y')
    _q(client, loc)
    assert [n for c in calls[n_before:] for n in c] == ['real0.jpg']


def test_stage2_budget_keeps_stage1_then_catches_up(place):
    _a, _db, _m, ids, loc, client, calls, mp = place
    nm = _name_of(ids)
    gate = threading.Event()
    inner = web._masked_embed_chunk

    def slow(items):
        gate.wait(10)
        return inner(items)
    mp.setattr(web, '_masked_embed_chunk', slow)
    t0 = time.monotonic()
    d, got = _q(client, loc, mask_wait='0.3')
    assert time.monotonic() - t0 < 5
    assert d['stage2']['state'] == 'partial' and d['stage2']['pending'] > 0
    order0 = [nm[i] for i in got]
    assert set(got) == set(_q(client, loc, mask='false')[1])          # same candidates, nothing lost
    assert order0.index('peopley0') < order0.index('real0')           # unmasked ones keep their stage-1 order
    gate.set()
    for _ in range(40):                                              # the background thread finishes
        d, got = _q(client, loc, mask_wait='1')
        if d['stage2']['state'] == 'ok':
            break
    assert d['stage2']['state'] == 'ok'
    order = [nm[i] for i in got]
    assert max(order.index(f'real{i}') for i in range(3)) < min(order.index(f'peopley{i}') for i in range(3))


def test_stage2_old_worker_degrades_with_notice(place):
    _a, _db, _m, ids, loc, client, calls, mp = place
    hits = []

    def old(items):
        hits.append(1)
        raise lm.MaskUnavailable('worker_outdated', 'embed_masked: worker needs redeploy')
    mp.setattr(web, '_masked_embed_chunk', old)
    base = _q(client, loc, mask='false')[1]
    d, got = _q(client, loc)
    assert got == base                                               # stage-1 result, never an error
    assert d['stage2']['state'] == 'worker_outdated' and 'redeploy' in d['stage2']['message']
    d, got = _q(client, loc)
    assert d['stage2']['state'] == 'worker_outdated' and len(hits) == 1   # backed off: no new attempt
    html = client.get(f'/locations/{loc}').text
    assert 'location-mask-toggle' in html and 'worker_outdated' in html


def test_stage2_unreachable_worker_and_disabled(place, monkeypatch):
    _a, _db, _m, ids, loc, client, calls, mp = place

    def down(items):
        raise worker_client.WorkerUnavailable('cannot reach worker')
    mp.setattr(web, '_masked_embed_chunk', down)
    d, _ = _q(client, loc)
    assert d['stage2']['state'] == 'worker_unavailable' and d['cards']
    # (a plain WorkerUnavailable from a stub is mapped to MaskUnavailable by the service thread)
    assert _q(client, loc, mask='false')[0].get('stage2') is None
    assert _q(client, loc, model='combined')[0].get('stage2') is None


def test_stage2_frame_rule_negatives_and_legacy_models(place):
    _a, db, manual, ids, loc, client, calls, mp = place
    manual.exclude_file_from_location(ids['real1'][1], loc)
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    db.insert_siglip_embedding(still, _vec(10, e0=1, e3=.1).tobytes(), 's')
    db.insert_dino_embedding(still, _vec(8, e0=1, e3=.1).tobytes(), 'd')
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    for neg in ('false', 'true'):
        d, got = _q(client, loc, honor_negatives=neg)
        assert ids['real1'][0] not in got and still not in got
        assert got.count(vid) == 1 and d['stage2']['state'] == 'ok'
        assert d['stage2']['skipped'] == 1                           # the video is not masked
    for model in ('combined', 'siglip', 'dino'):
        d, got = _q(client, loc, model=model)
        assert d['model'] == model and got and 'stage2' not in d


# ------------------------------------------------------------------ worker ops + client

class _Emb:
    def __init__(self, dim, mid):
        self.dim, self.mid = dim, mid

    def model_id(self):
        return self.mid


@pytest.fixture
def wroot(tmp_path, monkeypatch):
    m = worker_server.models
    monkeypatch.setattr(m, '_siglip', _Emb(3, 'sig/fake'))
    monkeypatch.setattr(m, '_dino', _Emb(3, 'dino/fake'))
    seen = []

    def fake_shared(embs, sources, stats=None, labels=None):
        outs = [[] for _ in embs]
        for s in sources:
            a = np.asarray(s).reshape(-1, 3).mean(axis=0).astype(np.float32)
            seen.append(np.asarray(s).copy())
            for o in outs:
                o.append(a)
        return [np.stack(o) if o else np.empty((0,), np.float32) for o in outs], []
    monkeypatch.setattr(worker_server, '_embed_shared', fake_shared)
    r = tmp_path / 'lib'
    (r / '.media').mkdir(parents=True)
    _scene(str(r / 'a.png'), (0.0, 0.0, 0.5, 1.0))
    _scene(str(r / 'b.png'), (0.0, 0.0, 1.0, 1.0))
    return r, seen


class _Client:
    def __init__(self, stale=False, bad_proto=False):
        self.stale, self.bad_proto, self.calls = stale, bad_proto, []

    def record(self, *a):
        pass

    def request(self, path, req, timeout=0, retries=0, concurrent=False):
        self.calls.append(path)
        if self.stale and path in (worker_protocol.PATH_EMBED_MASKED_PATHS, worker_protocol.PATH_EMBED_MASKED_BATCH):
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        out = worker_protocol.unpack(worker_server.HANDLERS[path](path, worker_protocol.pack(req), None, None, None, None))
        if self.bad_proto:
            out['masked_proto'] = 99
        return out


def test_worker_masked_paths_roundtrip_and_security(wroot, tmp_path):
    root, seen = wroot
    half = [[0.0, 0.0, 0.5, 1.0]]
    secret = tmp_path / 'secret.png'
    _scene(str(secret))
    items = [(str(root / 'a.png'), half), (str(root / 'b.png'), [[0.0, 0.0, 1.0, 1.0]]), (str(secret), half)]
    res = worker_client.embed_masked(_Client(), items, data_root=str(root))
    # the probe handler is real: the worker "sees" the root -> shared-disk mode for the two inside it
    assert res.mode == 'mixed' and res.siglip_model == 'sig/fake'
    assert res.errors[0] is None and res.siglip[0].shape == (3,)
    # the masked image the models saw has no red left (the person half was filled with the green mean)
    assert (seen[0][..., 0] == 0).all() and (seen[0][..., 1] == 200).all()
    assert res.errors[1] and 'covers' in res.errors[1]            # whole frame masked: refused, no vector
    assert res.coverage[1] == pytest.approx(1.0)
    assert res.errors[2] is None                                  # outside the root: uploaded instead
    # direct handler: traversal / absolute / symlink are rejected and never opened
    (root / 'evil.png').symlink_to(secret)
    req = {'data_root': str(root), 'paths': ['../secret.png', str(secret), 'evil.png', 'a.png'],
           'boxes': [half] * 4}
    out = worker_protocol.unpack(worker_server.handle_embed_masked_paths('p', worker_protocol.pack(req), 0, 0, 0, 0))
    assert [bool(r['error']) for r in out['results']] == [True, True, True, False]
    assert all(r['error'].startswith('rejected') for r in out['results'][:3])
    assert out['masked_proto'] == worker_protocol.MASKED_PROTOCOL_VERSION
    bad = worker_protocol.unpack(worker_server.handle_embed_masked_paths(
        'p', worker_protocol.pack({'data_root': str(tmp_path), 'paths': ['a.png'], 'boxes': [[]]}), 0, 0, 0, 0))
    assert bad['error'] and bad['results'] == []                  # no .media dir: refused


def test_worker_masked_upload_and_old_worker(wroot):
    root, _seen = wroot
    items = [(str(root / 'a.png'), [[0.0, 0.0, 0.5, 1.0]])]
    c = _Client()
    res = worker_client.embed_masked(c, items, data_root=None)    # no data_root: upload mode
    assert res.mode == 'upload' and res.errors == [None] and worker_protocol.PATH_EMBED_MASKED_BATCH in c.calls
    with pytest.raises(worker_client.WorkerError, match='needs redeploy'):
        worker_client.embed_masked(_Client(stale=True), items)
    with pytest.raises(worker_client.WorkerError, match='needs redeploy'):
        worker_client.embed_masked(_Client(bad_proto=True), items)


def test_web_maps_worker_errors(env, monkeypatch):
    _a, _db, _m, _ids, _add, _c, mp = env
    cfg = type('C', (), {'is_configured': lambda self: True})()
    mp.setattr(web, '_worker_client', cfg)
    mp.setattr(web, '_embed_data_root', None)

    def stale(client, items, **kw):
        raise worker_client.WorkerError('embed_masked: worker needs redeploy (does not know x)')
    mp.setattr(worker_client, 'embed_masked', stale)
    with pytest.raises(lm.MaskUnavailable) as ei:
        web._masked_embed_chunk([('/x.jpg', [])])
    assert ei.value.kind == 'worker_outdated'

    def down(client, items, **kw):
        raise worker_client.WorkerUnavailable('nope')
    mp.setattr(worker_client, 'embed_masked', down)
    with pytest.raises(lm.MaskUnavailable) as ei:
        web._masked_embed_chunk([('/x.jpg', [])])
    assert ei.value.kind == 'worker_unavailable'
