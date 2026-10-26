"""End-to-end instance-level region search on synthetic images (DINOv2 + RootSIFT/MAGSAC).

An 'object' (random coloured polygons + noise) is pasted at different positions/scales onto
different backgrounds; querying a box around it must rank the other photos containing it above
photos with only look-alike background. Needs torch + the cached DINOv2 hub weights; local
compute path, no worker. Run: `python -m pytest tests/test_region_search.py`.
"""
import io
import os
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from media_manager import region_index, worker_protocol, worker_server
from media_manager.web import create_app


def _object(seed, size=140):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(60, 200, (size, size, 3), dtype=np.uint8)).filter(
        __import__('PIL.ImageFilter', fromlist=['x']).GaussianBlur(1.2))
    d = ImageDraw.Draw(im)
    for _ in range(14):
        pts = [tuple(rng.integers(0, size, 2)) for _ in range(rng.integers(3, 6))]
        d.polygon(pts, fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return im


def _scene(seed, obj=None, pos=(0, 0), scale=1.0, size=(480, 360)):
    rng = np.random.default_rng(1000 + seed)
    yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    base = np.stack([128 + 100 * np.sin(xx / (30 + seed * 7)), 128 + 90 * np.cos(yy / (25 + seed * 5)),
                     np.full_like(xx, 90 + 20 * seed)], -1) + rng.normal(0, 6, size[::-1] + (3,))
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if obj is not None:
        o = obj.resize((int(obj.width * scale), int(obj.height * scale)), Image.BICUBIC)
        im.paste(o, pos)
    return im


def test_region_search_instance_end_to_end():
    obj = _object(7)
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'))
    scenes = {
        'query.jpg': _scene(1, obj, (60, 50), 1.0),
        'same_a.jpg': _scene(2, obj, (250, 120), 0.9),
        'same_b.jpg': _scene(3, obj, (20, 200), 1.1),
        'other_obj.jpg': _scene(4, _object(99), (60, 50), 1.0),
        'bg1.jpg': _scene(5),
        'bg2.jpg': _scene(6),
    }
    for name, im in scenes.items():
        im.save(os.path.join(tmp, name), quality=95)
    app = create_app(tmp)
    db = app.state.dbs[0]
    fid = {n: db.upsert_file_path(n, 'cs%02d' % i + 'x' * 6, size=100) for i, n in enumerate(scenes)}
    db.conn.commit()
    client = TestClient(app)
    box = [55, 45, 205, 195]

    assert client.post('/api/region-index/start').json()['started']
    for _ in range(600):
        st = client.get('/api/region-index/status').json()
        if not st['running']:
            break
        time.sleep(0.5)
    assert st['error'] is None and st['pending'] == 0, st
    assert db.count_region_indexed(region_index.ALGO_TAG) == len(scenes)
    # storage per image: 35 int8 windows + <=300 SIFT features
    w = db.conn.execute('SELECT MAX(LENGTH(data)) FROM region_windows').fetchone()[0]
    assert w <= 40 * region_index.DIM
    assert db.get_region_sift(fid['same_a.jpg'])[1].shape[0] <= region_index.SIFT_NFEAT

    r = client.post(f"/api/files/{fid['query.jpg']}/region-search", json={'bbox': box})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['mode'] == 'instance'
    ids = [x['file_id'] for x in data['results']]
    assert fid['query.jpg'] not in ids
    assert set(ids[:2]) == {fid['same_a.jpg'], fid['same_b.jpg']}, data['results']
    assert max(x['inliers'] for x in data['results'][:2]) > region_index.MIN_INL

    # stale-algo rows are re-queued
    assert len(db.get_unregion_indexed_files('other-algo')) == len(scenes)


def test_worker_region_handler_matches_local():
    buf = io.BytesIO()
    _scene(1, _object(7), (60, 50)).save(buf, 'PNG')
    img = buf.getvalue()

    def call(**kw):
        req = worker_protocol.pack({'name': 't', 'image': img, **kw})
        return worker_protocol.unpack(worker_server.handle_region('', req, '', '', '', 0))

    idx = call()
    assert idx['algo'] == region_index.ALGO_TAG and idx['error'] is None
    pil = region_index.load_pil(img)
    w, pts, desc = region_index.describe_image(pil)
    assert np.array_equal(np.frombuffer(idx['windows'], np.int8).reshape(idx['n'], -1), w)
    assert np.array_equal(np.frombuffer(idx['desc'], np.uint8).reshape(idx['nkp'], 128), desc)
    q = call(box=[55, 45, 205, 195])
    qv = np.frombuffer(q['vecs'], np.float32).reshape(q['n'], -1)
    assert np.allclose(qv, region_index.describe_query(pil, [55, 45, 205, 195]), atol=1e-4)
