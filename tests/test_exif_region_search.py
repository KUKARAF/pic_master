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

from media_manager import tile_index, web
from media_manager.formats import open_upright_rgb
from media_manager.web import create_app


def _rotated_jpeg(path):
    raw = Image.new('RGB', (100, 60), (255, 0, 0))
    raw.paste((0, 0, 255), (50, 0, 100, 60))
    ex = Image.Exif()
    ex[0x0112] = 6   # display rotated 90 CW: raw left -> displayed top
    raw.save(path, quality=100, exif=ex)


def test_open_upright_rgb_and_tiles():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.jpg')
    _rotated_jpeg(p)
    im = open_upright_rgb(p)
    assert im.size == (60, 100)
    top = np.asarray(im.crop((0, 0, 60, 40))).reshape(-1, 3).mean(0)
    assert top[0] > 200 and top[2] < 60
    bottom = np.asarray(im.crop((0, 60, 60, 100))).reshape(-1, 3).mean(0)
    assert bottom[2] > 200 and bottom[0] < 60

    class FakeClip:
        def __init__(self):
            self.crops = []

        def embed_pil_images(self, crops):
            self.crops = list(crops)
            return np.zeros((len(crops), 4), np.float32)

        def model_id(self):
            return 'fake'

    class FakeDb:
        def insert_tile_embeddings(self, *a):
            pass
    clip = FakeClip()
    tile_index.embed_tiles_for_file(FakeDb(), clip, 1, p)
    assert clip.crops[0].size == (60, 100)   # tiles laid out in the displayed frame


def test_region_search_crops_displayed_frame(monkeypatch):
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'))
    _rotated_jpeg(os.path.join(tmp, 'r.jpg'))
    app = create_app(tmp)
    db = app.state.dbs[0]
    fid = db.upsert_file_path('r.jpg', 'cs00xxxxxx', size=100)
    db.conn.commit()
    seen = []

    class FakeClip:
        def embed_pil_images(self, crops):
            seen.append(np.asarray(crops[0]).reshape(-1, 3).mean(0))
            return np.ones((1, 4), np.float32)
    monkeypatch.setattr(web, '_get_clip_indexer', lambda: FakeClip())
    r = TestClient(app).post(f'/api/files/{fid}/region-search', json={'bbox': [0, 0, 60, 40]})
    assert r.status_code == 200, r.text
    assert seen[0][0] > 200 and seen[0][2] < 60   # red, not blue / mixed


def test_pattern_decode_is_upright():
    from media_manager import pattern_descriptor as pd
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.jpg')
    _rotated_jpeg(p)
    bgr = pd.decode_bgr(open(p, 'rb').read())
    assert bgr.shape[:2] == (100, 60)
    assert bgr[:40].reshape(-1, 3).mean(0)[2] > 200   # BGR: red channel on top
