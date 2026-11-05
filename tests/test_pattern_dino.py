"""Find-by-pattern, DINOv2-L dense-cell algorithm (pattern_dino.py): geometry, projection/versioning,
the batched describe pipeline, query vectors, fusion, DB shape, the worker handler <-> client path
(incl. stale worker / algo mismatch) and the web job + search endpoint -- with a FAKE dense model
(deterministic patch mean/std features through a fixed random map) on the CPU. No GPU, no weights.
"""
import os
import time
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import pattern_dino as pd
from media_manager import pattern_deep, web, worker_client, worker_protocol, worker_server


class FakeDinoModel:
    """forward_features(x) -> {'x_norm_patchtokens': (B, gh*gw, 1024)}: tanh(random-map(patch mean, patch std))."""

    def __init__(self):
        self.P = torch.tensor(np.random.RandomState(0).randn(6, pd.TOKEN_DIM), dtype=torch.float32)
        self.shapes = []

    def forward_features(self, x):
        self.shapes.append(tuple(x.shape))
        m = F.avg_pool2d(x, pd.PATCH)
        s = (F.avg_pool2d(x * x, pd.PATCH) - m * m).clamp_min(0).sqrt()
        f = torch.cat([m, s], 1).permute(0, 2, 3, 1)             # (B, gh, gw, 6)
        t = torch.tanh(3 * f @ self.P)
        return {'x_norm_patchtokens': t.reshape(t.shape[0], -1, pd.TOKEN_DIM)}


def make_embedder():
    return types.SimpleNamespace(device='cpu', model=FakeDinoModel(), num_workers=2, batch_size=4,
                                 workers=1, model_id=lambda: 'fake-dino/v1')


def stripes(rng, period=8, size=256, col=(0.9, 0.6, 0.3)):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    v = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * xx / period))
    img = np.stack([v * c for c in col], -1) + rng.normal(0, 0.02, (size, size, 3))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def checker(rng, cell=8, size=256):
    yy, xx = np.mgrid[0:size, 0:size]
    v = ((xx // cell + yy // cell) % 2).astype(np.float32) * 0.8 + 0.1
    img = np.stack([v * 0.2, v * 0.5, v * 0.9], -1) + rng.normal(0, 0.02, (size, size, 3))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def textures():
    rng = np.random.default_rng(1)
    return {
        'query.png': stripes(rng), 'stripes_same.png': stripes(rng), 'checker.png': checker(rng),
        'noise.png': rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        'smooth.png': np.full((256, 256, 3), (40, 120, 200), np.uint8),
        'tiny.png': np.zeros((20, 20, 3), np.uint8),
    }


@pytest.fixture
def proj():
    """Install a small fitted projection (from the fake model's raw cells), uninstall afterwards."""
    emb = make_embedder()
    imgs = [Image.fromarray(a) for n, a in textures().items() if n != 'tiny.png']
    import io
    srcs = []
    for im in imgs:
        b = io.BytesIO()
        im.save(b, format='PNG')
        srcs.append(b.getvalue())
    raw, errs = pd.describe_files(emb, srcs, raw=True)
    assert all(e is None for e in errs)
    mu, W = pd.fit_projection(np.concatenate(raw), out_dim=16)
    pd.set_projection(mu, W)
    yield mu, W
    pd.clear_projection()


# ------------------------------------------------------------------------------ unit-level

def test_target_size_multiples_of_cell_and_long_side():
    assert pd.target_size(256, 256) == (448, 448)
    assert pd.target_size(200, 300) == (308, 448)
    assert pd.target_size(4000, 6000) == (308, 448)
    assert pd.target_size(20, 20) is None
    th, tw = pd.target_size(10, 400)               # extreme panorama stays valid
    assert th % pd.CELL_PX == 0 and tw % pd.CELL_PX == 0 and th >= pd.CELL_PX


def test_box_token_weights_overlap_fraction():
    w = pd.box_token_weights(4, 4, (7, 7, 21, 35))        # covers parts of tokens
    assert w.shape == (4, 4) and w.max() <= 1.0
    full = pd.box_token_weights(4, 4, (0, 0, 56, 56))
    assert np.allclose(full, 1.0)
    assert pd.box_token_weights(4, 4, (0, 0, 14, 14)).sum() == pytest.approx(1.0)
    assert pd.box_token_weights(4, 4, (100, 100, 120, 120)).sum() == 0


def test_fuse_scores_zscore_and_partial_coverage():
    a = (np.array([1, 2, 3]), np.array([0.9, 0.5, 0.1], np.float32), 1.0)
    b = (np.array([2, 3, 4]), np.array([0.2, 0.8, -1.0], np.float32), 1.0)   # file 4 has no descriptor
    fids, s = pd.fuse_scores([a, b])
    assert list(fids) == [1, 2, 3]                 # 4 is ignored (-1 sentinel)
    assert s[0] > s[2] or s[1] != s[0]
    # weight 0 component is ignored entirely
    f2, s2 = pd.fuse_scores([a, (b[0], b[1], 0.0)])
    assert list(f2) == [1, 2, 3] and s2[0] > s2[1] > s2[2]
    assert pd.fuse_scores([])[0].shape == (0,)


def test_projection_roundtrip_versions_the_algo(proj):
    tag = pd.algo_tag()
    assert pd.projection_ready() and pd.dim() == 16 and 'pca16i8' in tag
    assert pd.ALGO_TAG == tag and pd.DIM == 16
    raw = np.random.RandomState(3).randn(5, pd.TOKEN_DIM).astype(np.float32)
    v = pd.project(raw)
    assert v.shape == (5, 16) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    q = pd.quantize(v)
    assert q.dtype == np.int8 and q.min() >= -127
    # refit => different tag => stored rows become stale ("needs reindex")
    mu, W = proj
    pd.set_projection(mu, W * 1.0001)
    assert pd.algo_tag() != tag
    pd.clear_projection()
    assert pd.algo_tag().endswith('unfitted') and not pd.projection_ready()
    with pytest.raises(RuntimeError, match='projection missing'):
        pd.project(raw)


def test_describe_files_shapes_errors_and_batch_invariance(proj):
    emb = make_embedder()
    rng = np.random.default_rng(5)
    import io

    def png(arr):
        b = io.BytesIO()
        Image.fromarray(arr).save(b, format='PNG')
        return b.getvalue()
    wide = stripes(rng, size=256)[:200, :]            # 200 x 256 -> different grid shape
    srcs = [png(stripes(rng)), b'not an image', png(textures()['tiny.png']), png(wide), png(stripes(rng))]
    stats = {}
    res, errs = pd.describe_files(emb, srcs, stats=stats)
    assert res[0].shape == (256, 16) and res[0].dtype == np.int8     # 16 x 16 cells
    assert res[1] is None and errs[1]
    assert res[2].shape == (0, 16) and errs[2] is None               # too small: indexed, nothing describable
    assert res[3].shape[1] == 16 and res[3].shape[0] == (pd.target_size(200, 256)[0] // 28) * 16
    assert stats['gpu_seconds'] > 0 and stats['decode_seconds'] > 0
    # equal-shape images were batched into one forward; singles give identical bytes
    single, _ = pd.describe_files(make_embedder(), [srcs[0]])
    assert np.array_equal(single[0], res[0])
    assert any(s[0] > 1 for s in emb.model.shapes)


def test_query_vectors_modes_and_selection(proj):
    emb = make_embedder()
    rgb = textures()['query.png']
    raw = pd.query_raw(emb, rgb, [10, 10, 130, 130], rots=(0, 1))
    kinds = [(k, s, r) for k, s, r, _ in raw]
    assert ('mask', 0.0, 0) in kinds
    assert {(s, r) for k, s, r in kinds if k == 'crop'} == {(s, r) for s in pd.CROP_SCALES for r in (0, 1)}
    assert pd.select_query(raw, 'mask').shape == (1, pd.TOKEN_DIM)
    assert pd.select_query(raw, 'crop', rots=(0,)).shape[0] == len(pd.CROP_SCALES)
    assert pd.select_query(raw, 'both').shape[0] == 1 + 2 * len(pd.CROP_SCALES)
    with pytest.raises(ValueError):
        pd.select_query(raw, 'nope')
    q = pd.describe_query(emb, rgb, [10, 10, 130, 130], mode='both')
    assert q.shape[1] == 16 and np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    # a sliver is rejected; a sub-token box falls back to crops when only 'mask' was asked for
    assert pd.describe_query(emb, rgb, [10, 10, 12, 12]) is None
    assert pd.select_query([('crop', 1.0, 0, np.ones(pd.TOKEN_DIM, np.float32))], 'mask').shape[0] == 1


def test_ranking_same_pattern_first(proj):
    emb = make_embedder()
    imgs = textures()
    names = [n for n in imgs if n != 'query.png']
    import io
    srcs = []
    for n in names:
        b = io.BytesIO()
        Image.fromarray(imgs[n]).save(b, format='PNG')
        srcs.append(b.getvalue())
    cells, _ = pd.describe_files(emb, srcs)
    counts = np.array([len(c) for c in cells])
    mat = np.concatenate(cells)
    q = pd.describe_query(emb, imgs['query.png'], [10, 10, 130, 130])
    scores = pd.score_cells(mat, counts, q)
    order = [names[i] for i in np.argsort(-scores)]
    assert order[0] == 'stripes_same.png', order
    assert scores[names.index('tiny.png')] == -1.0


# --------------------------------------------------------------------------------------- DB

def test_db_pattern_tables_are_separate_and_whitelisted(tmp_path):
    os.makedirs(tmp_path / '.media')
    app = web.create_app(str(tmp_path))
    db = app.state.dbs[0]
    f1 = db.upsert_file_path('a.png', 'c1', size=1)
    f2 = db.upsert_file_path('b.png', 'c2', size=1)
    db.conn.commit()
    arr = np.arange(-8, 8, dtype=np.int8).reshape(2, 8)
    db.insert_pattern_windows_batch([(f1, arr), (f2, arr[:0])], 'algo-A', table='pattern_dino_cells')
    assert db.count_pattern_indexed('algo-A', table='pattern_dino_cells') == 2
    assert db.count_pattern_indexed('algo-A') == 0                      # the ResNet table is untouched
    assert db.count_pattern_stale('algo-B', table='pattern_dino_cells') == 2
    got = list(db.iter_pattern_windows('algo-A', 8, table='pattern_dino_cells'))
    assert len(got) == 1 and list(got[0][1]) == [2, 0] and got[0][2].shape == (2, 8)
    assert db.get_unpattern_indexed_files('algo-A', table='pattern_dino_cells') == []
    assert len(db.get_unpattern_indexed_files('algo-B', table='pattern_dino_cells')) == 2
    with pytest.raises(ValueError):
        db.count_pattern_indexed('x', table='files; DROP TABLE files')


# ------------------------------------------------------------------- worker handler <-> client

class FakeClient:
    """Routes request() straight into the real server handlers via real pack/unpack."""

    def __init__(self):
        self.calls = []
        self.stale = False

    def record(self, op, name):
        pass

    def request(self, path, req, timeout=0, retries=0, concurrent=False):
        if self.stale:
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        self.calls.append((path, req.get('op')))
        out = worker_server.HANDLERS[path](path, worker_protocol.pack(req), None, None, None, None)
        return worker_protocol.unpack(out)


@pytest.fixture
def worker(proj, monkeypatch):
    monkeypatch.setattr(worker_server.models, '_dino', make_embedder())
    return FakeClient()


def _png_bytes(arr):
    import io
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='PNG')
    return b.getvalue()


def test_worker_index_upload_and_per_item_errors(worker):
    imgs = textures()
    items = [(n, _png_bytes(a)) for n, a in imgs.items()] + [('bad.png', b'garbage')]
    res = worker_client.pattern_dino_index(worker, items, pd.algo_tag(), batch_size=4, workers=2)
    assert res.algo == pd.algo_tag() and res.mode == 'upload'
    for i, (n, _) in enumerate(items):
        if n == 'bad.png':
            assert res.cells[i] is None and res.errors[i]
        else:
            assert res.errors[i] is None and res.cells[i].dtype == np.int8
    assert res.cells[list(imgs).index('tiny.png')].shape == (0, 16)
    assert res.cells[0].shape == (256, 16) and res.gpu_seconds > 0
    assert len(worker.calls) >= 2                                   # batched, not one call per image


def test_worker_index_shared_disk_paths_match_upload(worker, tmp_path):
    os.makedirs(tmp_path / '.media')
    paths = []
    for n, a in textures().items():
        p = tmp_path / n
        Image.fromarray(a).save(p)
        paths.append(str(p))
    via_paths = worker_client.pattern_dino_index(worker, paths, pd.algo_tag(), data_root=str(tmp_path))
    assert via_paths.mode == 'paths'
    via_upload = worker_client.pattern_dino_index(worker, paths, pd.algo_tag())
    assert via_upload.mode == 'upload'
    for a, b in zip(via_paths.cells, via_upload.cells):
        assert np.array_equal(a, b)
    # a path outside the root is rejected per item, never read
    outside = tmp_path.parent / 'outside.png'
    Image.fromarray(textures()['smooth.png']).save(outside)
    r = worker_client.pattern_dino_index(worker, [str(outside), paths[0]], pd.algo_tag(), data_root=str(tmp_path))
    assert r.mode == 'mixed' and r.cells[0] is not None and r.cells[1] is not None


def test_worker_query_matches_local(worker):
    imgs = textures()
    box = [10, 10, 130, 130]
    remote = worker_client.pattern_dino_query(worker, _png_bytes(imgs['query.png']), box, pd.algo_tag(),
                                              mode='both', rots=(0,))
    local = pd.describe_query(make_embedder(), imgs['query.png'], box, mode='both', rots=(0,))
    assert remote.shape == local.shape and np.allclose(remote, local, atol=1e-5)
    assert worker_client.pattern_dino_query(worker, _png_bytes(imgs['query.png']), [10, 10, 12, 12],
                                            pd.algo_tag()) is None


def test_stale_worker_is_loud_not_silent(worker):
    worker.stale = True
    with pytest.raises(worker_client.WorkerError, match='needs redeploy'):
        worker_client.pattern_dino_index(worker, [('a.png', b'x')], pd.algo_tag())
    with pytest.raises(worker_client.WorkerError, match='needs redeploy'):
        worker_client.pattern_dino_query(worker, b'x', [0, 0, 50, 50], pd.algo_tag())
    worker.stale = False
    with pytest.raises(worker_client.WorkerError, match='algo'):    # different projection on the worker
        worker_client.pattern_dino_index(worker, [('a.png', _png_bytes(textures()['smooth.png']))], 'other-algo')


def test_worker_without_projection_reports_instead_of_guessing(monkeypatch):
    pd.clear_projection()
    monkeypatch.setattr(worker_server.models, '_dino', make_embedder())
    out = worker_protocol.unpack(worker_server.HANDLERS[worker_protocol.PATH_PATTERN_DINO](
        worker_protocol.PATH_PATTERN_DINO, worker_protocol.pack({'op': 'index', 'items': []}),
        None, None, None, None))
    assert 'projection missing' in out['error']
    info = worker_protocol.unpack(worker_server.HANDLERS[worker_protocol.PATH_PATTERN_DINO](
        worker_protocol.PATH_PATTERN_DINO, worker_protocol.pack({'op': 'info'}), None, None, None, None))
    assert info['ready'] is False and info['error'] is None


# ------------------------------------------------------------------------------------ web

def _wait_job(client, url, tries=300):
    for _ in range(tries):
        st = client.get(url).json()
        if not st['running']:
            return st
        time.sleep(0.1)
    raise AssertionError('job did not finish')


@pytest.fixture
def web_env(proj, tmp_path, monkeypatch):
    os.makedirs(tmp_path / '.media')
    monkeypatch.setattr(web, '_worker_client', None)
    emb = make_embedder()
    monkeypatch.setattr(web, '_get_dino_embedder', lambda: emb)
    # deterministic stand-in for the ResNet pattern index (only used by the fused test)
    def fake_windows(rgb):
        v = np.zeros(pattern_deep.DIM, np.float32)
        v[int(rgb.reshape(-1, 3).std() // 12) % pattern_deep.DIM] = 1.0
        return pattern_deep.quantize(np.stack([v] * 3))
    monkeypatch.setattr(pattern_deep, 'describe_image', fake_windows)
    monkeypatch.setattr(pattern_deep, 'describe_query',
                        lambda rgb, box: fake_windows(rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])])[:1].astype(np.float32) / 127.0)
    app = web.create_app(str(tmp_path))
    db = app.state.dbs[0]
    ids = {}
    for i, (n, a) in enumerate(textures().items()):
        Image.fromarray(a).save(tmp_path / n)
        ids[n] = db.upsert_file_path(n, 'cs%02d' % i + 'x' * 6, size=100)
    db.conn.commit()
    return app, TestClient(app), db, ids


def test_web_job_search_and_needs_reindex(web_env):
    app, client, db, ids = web_env
    algo = pd.algo_tag()
    box = {'bbox': [10, 10, 130, 130]}
    # no index yet -> clear 400s
    r = client.post(f"/api/files/{ids['query.png']}/pattern-search", json={**box, 'algo': 'dino'})
    assert r.status_code == 400 and 'DINOv2' in r.json()['detail']
    r = client.post(f"/api/files/{ids['query.png']}/pattern-search", json={**box, 'algo': 'bogus'})
    assert r.status_code == 400

    st0 = client.get('/api/pattern-dino-index/status').json()
    assert st0['projection_ready'] and st0['pending'] == len(ids) and st0['stale'] == 0
    assert client.post('/api/pattern-dino-index/start').json()['started']
    st = _wait_job(client, '/api/pattern-dino-index/status')
    assert st['error'] is None and st['pending'] == 0 and st['done'] == len(ids) and st['failed'] == 0, st
    assert st['transport_mode'] == 'local' and st['img_per_sec'] > 0 and st['eta_seconds'] is not None
    assert st['db_write_pct'] is not None and st['gpu_busy_pct'] is not None
    assert db.count_pattern_indexed(algo, table='pattern_dino_cells') == len(ids)
    assert db.count_pattern_indexed(algo) == 0                      # ResNet index untouched
    assert client.post('/api/pattern-dino-index/start').json()['started'] is False   # nothing left

    r = client.post(f"/api/files/{ids['query.png']}/pattern-search", json={**box, 'algo': 'dino'})
    assert r.status_code == 200, r.text
    body = r.json()
    got = [x['file_id'] for x in body['results']]
    assert body['algo'] == 'dino' and ids['query.png'] not in got and got[0] == ids['stripes_same.png'], got
    assert ids['tiny.png'] not in got
    # the default stays the ResNet algorithm and still demands ITS index
    r = client.post(f"/api/files/{ids['query.png']}/pattern-search", json=box)
    assert r.status_code == 400 and 'Index patterns' in r.json()['detail']

    # refit => new tag => everything pending again, old rows reported stale
    mu, W = pd._get_proj()['mu'], pd._get_proj()['W']
    pd.set_projection(mu, W * 1.001)
    st = client.get('/api/pattern-dino-index/status').json()
    assert st['algo'] != algo and st['pending'] == len(ids) and st['stale'] == len(ids)
    r = client.post(f"/api/files/{ids['query.png']}/pattern-search", json={**box, 'algo': 'dino'})
    assert r.status_code == 400


def test_web_fused_search_uses_both_indexes(web_env):
    app, client, db, ids = web_env
    assert client.post('/api/pattern-dino-index/start').json()['started']
    _wait_job(client, '/api/pattern-dino-index/status')
    assert client.post('/api/pattern-index/start').json()['started']
    _wait_job(client, '/api/pattern-index/status')
    algos = client.get('/api/pattern-search/algos').json()['algos']
    assert algos['resnet']['indexed'] == len(ids) and algos['dino']['indexed'] == len(ids)
    r = client.post(f"/api/files/{ids['query.png']}/pattern-search",
                    json={'bbox': [10, 10, 130, 130], 'algo': 'fused', 'dino_weight': 0.5})
    assert r.status_code == 200, r.text
    got = [x['file_id'] for x in r.json()['results']]
    assert r.json()['algo'] == 'fused' and got[0] == ids['stripes_same.png'] and ids['query.png'] not in got


def test_web_start_without_projection_says_how_to_fix(tmp_path, monkeypatch):
    pd.clear_projection()
    os.makedirs(tmp_path / '.media')
    monkeypatch.setattr(web, '_worker_client', None)
    client = TestClient(web.create_app(str(tmp_path)))
    d = client.post('/api/pattern-dino-index/start').json()
    assert d['started'] is False and 'bench_pattern_dino.py' in d['message']
    st = client.get('/api/pattern-dino-index/status').json()
    assert st['projection_ready'] is False and st['pending'] is None
