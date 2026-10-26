"""End-to-end find-by-pattern: index synthetic textures through the web API, query a region.

Needs torch + timm (ResNet18 weights from the HF cache) — local compute path, no worker.
Run: venv python + libGL on LD_LIBRARY_PATH, `python -m pytest tests/test_pattern_search.py`.
"""
import os
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import pattern_deep
from media_manager.web import create_app


def _stripes(rng, period, angle_deg, size=(256, 256)):
    yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    th = np.deg2rad(angle_deg)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * u / period))
    img = np.stack([v * 0.9, v * 0.6, v * 0.3], -1) + rng.normal(0, 0.02, size[::-1] + (3,))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _checker(rng, cell, size=(256, 256)):
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    v = ((xx // cell + yy // cell) % 2).astype(np.float32) * 0.8 + 0.1
    img = np.stack([v * 0.2, v * 0.5, v * 0.9], -1) + rng.normal(0, 0.02, size[::-1] + (3,))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def test_pattern_search_end_to_end():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'))
    imgs = {
        'query.png': _stripes(rng, 8, 0),
        'stripes_same.png': _stripes(rng, 8, 0),
        'checker.png': _checker(rng, 8),
        'noise.png': rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        'smooth.png': np.full((256, 256, 3), (40, 120, 200), np.uint8),
        'tiny.png': np.zeros((20, 20, 3), np.uint8),   # too small for one window
    }
    for name, arr in imgs.items():
        Image.fromarray(arr).save(os.path.join(tmp, name))
    app = create_app(tmp)
    db = app.state.dbs[0]
    fid = {}
    for i, name in enumerate(imgs):
        fid[name] = db.upsert_file_path(name, 'cs%02d' % i + 'x' * 6, size=100)
    db.conn.commit()
    client = TestClient(app)

    # no index yet -> clear 400
    r = client.post(f"/api/files/{fid['query.png']}/pattern-search", json={'bbox': [10, 10, 130, 130]})
    assert r.status_code == 400

    assert client.post('/api/pattern-index/start').json()['started']
    for _ in range(300):
        st = client.get('/api/pattern-index/status').json()
        if not st['running']:
            break
        time.sleep(0.2)
    assert st['error'] is None and st['pending'] == 0, st
    assert db.count_pattern_indexed(pattern_deep.ALGO_TAG) == len(imgs)
    # compact: <= ~300 windows x DIM bytes per image
    size = db.conn.execute('SELECT MAX(LENGTH(data)) FROM pattern_windows').fetchone()[0]
    assert size <= 300 * pattern_deep.DIM

    r = client.post(f"/api/files/{fid['query.png']}/pattern-search", json={'bbox': [10, 10, 130, 130]})
    assert r.status_code == 200, r.text
    ids = [x['file_id'] for x in r.json()['results']]
    assert fid['query.png'] not in ids
    assert ids[0] == fid['stripes_same.png'], ids   # same stripes beat checker / noise / smooth
    assert fid['tiny.png'] not in ids               # windowless images never rank

    # an algo bump marks everything pending again (stale descriptors ignored)
    assert db.count_pattern_indexed('other-algo') == 0
    assert len(db.get_unpattern_indexed_files('other-algo')) == len(imgs)


def test_worker_handler_matches_local():
    """The worker's deep branch returns exactly what the local fallback would compute."""
    import io

    from media_manager import worker_protocol, worker_server
    rng = np.random.default_rng(2)
    buf = io.BytesIO()
    Image.fromarray(_stripes(rng, 6, 30, (200, 160))).save(buf, 'PNG')
    img = buf.getvalue()

    def call(**kw):
        req = worker_protocol.pack({'name': 't', 'image': img, 'deep': True, **kw})
        return worker_protocol.unpack(worker_server.handle_pattern('', req, '', '', '', 0))

    idx = call()
    assert idx['algo'] == pattern_deep.ALGO_TAG and idx['error'] is None
    w = np.frombuffer(idx['windows'], np.int8).reshape(idx['n'], pattern_deep.DIM)
    rgb = pattern_deep.load_rgb(img)
    assert np.array_equal(w, pattern_deep.describe_image(rgb))
    q = call(box=[10, 10, 120, 120])
    qv = np.frombuffer(q['queries'], np.float32).reshape(q['n'], -1)
    assert np.allclose(qv, pattern_deep.describe_query(rgb, [10, 10, 120, 120]), atol=1e-5)
