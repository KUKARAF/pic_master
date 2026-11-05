"""Region/pattern search must work in the EXIF-rotated frame the browser displays.

A 100x60 raw image (left half red, right half blue) tagged Orientation=6 displays as
60x100 with the red half on TOP. A box drawn on the displayed top half must therefore
crop red pixels (region search) / be described like the red patch (pattern search).

Run: venv python + libGL on LD_LIBRARY_PATH, `python -I -m pytest tests/test_exif_region_search.py`.
"""
import os
import tempfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import region_index
from media_manager.formats import open_upright_rgb
from media_manager.web import create_app


def _rotated_jpeg(path):
    raw = Image.new('RGB', (100, 60), (255, 0, 0))
    raw.paste((0, 0, 255), (50, 0, 100, 60))
    ex = Image.Exif()
    ex[0x0112] = 6   # display rotated 90 CW: raw left -> displayed top
    raw.save(path, quality=100, exif=ex)


def test_open_upright_rgb():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.jpg')
    _rotated_jpeg(p)
    im = open_upright_rgb(p)
    assert im.size == (60, 100)
    top = np.asarray(im.crop((0, 0, 60, 40))).reshape(-1, 3).mean(0)
    assert top[0] > 200 and top[2] < 60
    bottom = np.asarray(im.crop((0, 60, 60, 100))).reshape(-1, 3).mean(0)
    assert bottom[2] > 200 and bottom[0] < 60


def test_region_search_crops_displayed_frame(monkeypatch):
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'))
    _rotated_jpeg(os.path.join(tmp, 'r.jpg'))
    app = create_app(tmp)
    db = app.state.dbs[0]
    fid = db.upsert_file_path('r.jpg', 'cs00xxxxxx', size=100)
    db.conn.commit()
    # Pretend the region index exists (the search refuses to run without it).
    db.insert_region_index(fid, np.zeros((0, region_index.DIM), np.int8), np.zeros((0, 2), np.float16),
                           np.zeros((0, 128), np.uint8), region_index.ALGO_TAG)
    seen = []

    def fake_query(pil, box):
        seen.append(np.asarray(pil.crop(tuple(box))).reshape(-1, 3).mean(0))   # what the box covers
        return np.zeros((3, region_index.DIM), np.float32)
    monkeypatch.setattr(region_index, 'describe_query', fake_query)
    monkeypatch.setattr(region_index, 'query_sift', lambda pil, box: (np.zeros((0, 2)), np.zeros((0, 128))))
    r = TestClient(app).post(f'/api/files/{fid}/region-search', json={'bbox': [0, 0, 60, 40]})
    assert r.status_code == 200, r.text
    assert seen[0][0] > 200 and seen[0][2] < 60   # red, not blue / mixed


def test_region_search_without_index_says_build_it_first():
    """No CLIP/whole-image fallback any more: an empty region index is a 409 with the next step."""
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'))
    _rotated_jpeg(os.path.join(tmp, 'r.jpg'))
    app = create_app(tmp)
    db = app.state.dbs[0]
    fid = db.upsert_file_path('r.jpg', 'cs00xxxxxx', size=100)
    db.conn.commit()
    r = TestClient(app).post(f'/api/files/{fid}/region-search', json={'bbox': [0, 0, 60, 40]})
    assert r.status_code == 409
    assert 'Index region matching' in r.json()['detail']


def test_pattern_decode_is_upright():
    from media_manager import pattern_descriptor as pd
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.jpg')
    _rotated_jpeg(p)
    bgr = pd.decode_bgr(open(p, 'rb').read())
    assert bgr.shape[:2] == (100, 60)
    assert bgr[:40].reshape(-1, 3).mean(0)[2] > 200   # BGR: red channel on top
