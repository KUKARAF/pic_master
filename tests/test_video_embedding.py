"""Videos get the similarity embeddings (SigLIP 2 + DINOv2): one vector per model per video, the
L2-normalised mean of a few sampled frames, stored in the same embedding tables as photos at
the video's own file id -- so /similar and set/location suggestions include videos. Covered:
scan_status lists the stage as required for videos, embed-now / the /bulk job / folder ML all
embed videos (frames go through the SAME one-decode-pass path as photos; the temp frame JPEGs
are cleaned up), and /similar on a video works. Fake embedders (no GPU, no model download)
record the frames they were given.

Run: python tests/test_video_embedding.py   (or python -m pytest tests/test_video_embedding.py)
"""
import os
import time
import tempfile

import cv2
import numpy as np
import xxhash
from fastapi.testclient import TestClient
from PIL import Image

import media_manager.web as web_mod
from media_manager import scan_status
from media_manager.web import create_app

DIMS = {'siglip': 8, 'dino': 6}


class FakeEmb:
    batch_size = 4
    workers = 1

    def __init__(self, which):
        self.which = which

    def model_id(self):
        # The real static ids, so the bulk job's "needs" check lines up with the stored model.
        from media_manager.embedders import DINOv2Embedder, SigLIP2Embedder
        return (SigLIP2Embedder if self.which == 'siglip' else DINOv2Embedder).model_id()


class FakeShared:
    """Stand-in for the local one-decode-pass SigLIP+DINO: brightness bucket -> one-hot vector."""

    def __init__(self):
        self.paths = []

    def __call__(self, embs, paths, stats=None):
        self.paths.extend(paths)
        sig, dino = [], []
        for p in paths:
            b = int(np.asarray(Image.open(p).convert('RGB')).mean() // 32)
            s = np.zeros(DIMS['siglip'], np.float32)
            d = np.zeros(DIMS['dino'], np.float32)
            s[b % DIMS['siglip']] = 1.0
            d[b % DIMS['dino']] = 1.0
            sig.append(s)
            dino.append(d)
        return [np.array(sig), np.array(dino)], []


def _video(path, n=50):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 20 + i * 3, np.uint8))
    w.release()


def _rows(db, fid):
    out = {}
    for space in ('siglip', 'dino'):
        blob = db.get_space_embedding(space, fid)
        out[space] = None if blob is None else np.frombuffer(blob, np.float32)
    return out


def run(monkeypatch=None):
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
    fake = FakeShared()
    setter = monkeypatch.setattr if monkeypatch is not None else (lambda o, n, v: setattr(o, n, v))
    setter(web_mod, '_worker_client', None)
    setter(web_mod, '_get_siglip_embedder', lambda: FakeEmb('siglip'))
    setter(web_mod, '_get_dino_embedder', lambda: FakeEmb('dino'))
    setter(web_mod, '_local_embed_shared', fake)
    client = TestClient(app)

    st = scan_status.status_for_files(db, manual, [(paths['a.avi'], 'v/a.avi')])[paths['a.avi']]
    assert st['stages']['embed'] is False and 'Similarity embedding (SigLIP 2 + DINOv2)' in st['missing'], st
    prev = client.get('/api/fm/ml/preview', params={'path': 'v'}).json()
    stage = [s for s in prev['stages'] if s['key'] == 'embed'][0]
    assert stage['missing'] == 2, prev

    # Embed-now on one video: 5 sampled frames -> one unit-length vector per model.
    assert client.post('/api/files/%d/embed' % paths['a.avi']).json() == {'embedded': True}
    frames = list(fake.paths)
    assert len(frames) == 5 and all(p.endswith('.jpg') for p in frames), frames
    assert not any(os.path.exists(p) for p in frames), 'temp frame files must be cleaned up'
    got = _rows(db, paths['a.avi'])
    for space, vec in got.items():
        assert vec is not None and vec.shape == (DIMS[space],), (space, vec)
        assert abs(np.linalg.norm(vec) - 1) < 1e-5 and (vec > 0).sum() > 1, (space, vec)  # frames differ
    scan_status.invalidate()
    st = scan_status.status_for_files(db, manual, [(paths['a.avi'], 'v/a.avi')])[paths['a.avi']]
    assert st['stages']['embed'] is True, st

    # Folder ML "Run missing ML" with just the embed stage embeds the other video.
    r = client.post('/api/fm/ml/start', json={'path': 'v', 'stages': ['embed']})
    assert r.status_code == 200, r.text
    for _ in range(200):
        s = client.get('/api/fm/ml/status').json()
        if not s['running']:
            break
        time.sleep(0.05)
    assert s['error'] is None, s
    got_b = _rows(db, paths['b.avi'])
    assert got_b['siglip'] is not None and got_b['dino'] is not None, s

    # The /bulk job sees nothing left (videos are candidates like photos).
    assert client.post('/api/siglip-dino/start').json().get('total') == 0
    db.conn.execute('DELETE FROM embeddings_dino WHERE file_id=?', (paths['b.avi'],))
    db.conn.commit()
    assert client.post('/api/siglip-dino/start').json()['total'] == 1
    for _ in range(200):
        d = client.get('/api/siglip-dino/status').json()
        if not d['running']:
            break
        time.sleep(0.05)
    assert d['error'] is None and _rows(db, paths['b.avi'])['dino'] is not None, d

    # /similar on a video works in SigLIP 2 and lists the other video.
    page = client.get('/similar/%d' % paths['a.avi']).text
    assert "can't be embedded" not in page and 'score-badge' in page
    res = client.get('/api/files/%d/similar' % paths['a.avi']).json()
    assert [x['file_id'] for x in res['results']] == [paths['b.avi']], res
    print('OK  video similarity embedding: required stage, embed-now, folder ML, bulk job, /similar')
    print('ALL ASSERTIONS PASSED')


def test_video_embedding(monkeypatch):
    run(monkeypatch)


if __name__ == '__main__':
    run()
