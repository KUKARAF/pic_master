"""Videos get a similarity (CLIP) embedding: the mean of a few sampled frames, stored in
the same embeddings table as photos, so /similar and set tinder include videos. Covered:
scan_status lists it as a required video stage, folder ML / bulk index / embed-now all
embed videos. A fake CLIP indexer (no model download) records the frames it was given.

Run: python tests/test_video_embedding.py   (or python -m pytest tests/test_video_embedding.py)
"""
import os
import time
import tempfile

import cv2
import numpy as np
import xxhash
from fastapi.testclient import TestClient

import media_manager.web as web_mod
from media_manager import scan_status
from media_manager.web import create_app


class FakeCLIP:
    def __init__(self):
        self.frames = []

    @staticmethod
    def model_id():
        return 'fake-clip'

    def embed_pil_images(self, images):
        self.frames.extend(images)
        out = []
        for im in images:
            v = np.zeros(8, np.float32)
            v[int(np.asarray(im).mean() // 32)] = 1.0     # brightness bucket
            out.append(v)
        return out

    def embed_images(self, paths):
        return np.ones((len(paths), 8), np.float32), []


def _video(path, n=50):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 20 + i * 3, np.uint8))
    w.release()


def run():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    os.makedirs(os.path.join(root, 'v'))
    paths = {}
    for name in ('a.avi', 'b.avi'):
        _video(os.path.join(root, 'v', name), 50 if name == 'a.avi' else 60)
    app = create_app(root)
    db, _errors, manual = app.state.dbs
    for name in ('a.avi', 'b.avi'):
        rel = 'v/' + name
        data = open(os.path.join(root, rel), 'rb').read()
        db.upsert_file_path(rel, xxhash.xxh64(data).hexdigest(), size=len(data), modified_time=1)
        paths[name] = db.find_file_id_by_checksum(xxhash.xxh64(data).hexdigest())
    db.conn.commit()
    fake = FakeCLIP()
    web_mod._get_clip_indexer = lambda: fake
    client = TestClient(app)

    st = scan_status.status_for_files(db, manual, [(paths['a.avi'], 'v/a.avi')])[paths['a.avi']]
    assert st['stages']['clip'] is False and 'Similarity embedding (CLIP)' in st['missing'], st
    prev = client.get('/api/fm/ml/preview', params={'path': 'v'}).json()
    clip = [s for s in prev['stages'] if s['key'] == 'clip'][0]
    assert clip['missing'] == 2, prev

    # Embed-now on one video: 5 sampled frames -> one unit-length vector.
    assert client.post('/api/files/%d/embed' % paths['a.avi']).json() == {'embedded': True}
    assert len(fake.frames) == 5, len(fake.frames)
    emb = np.frombuffer(db.get_embedding(paths['a.avi']), np.float32)
    assert abs(np.linalg.norm(emb) - 1) < 1e-5 and (emb > 0).sum() > 1, emb   # frames differ
    scan_status.invalidate()
    st = scan_status.status_for_files(db, manual, [(paths['a.avi'], 'v/a.avi')])[paths['a.avi']]
    assert st['stages']['clip'] is True, st

    # Folder ML "Run missing ML" with just the clip stage embeds the other video.
    r = client.post('/api/fm/ml/start', json={'path': 'v', 'stages': ['clip']})
    assert r.status_code == 200, r.text
    for _ in range(200):
        s = client.get('/api/fm/ml/status').json()
        if not s['running']:
            break

        time.sleep(0.05)
    assert db.get_embedding(paths['b.avi']) is not None, s

    # /similar on a video works (no "videos can't be embedded").
    page = client.get('/similar/%d' % paths['a.avi']).text
    assert "can't be embedded" not in page
    print('OK  video similarity embedding: required stage, embed-now, folder ML, /similar')
    print('ALL ASSERTIONS PASSED')


def test_video_embedding():
    run()


if __name__ == '__main__':
    run()
