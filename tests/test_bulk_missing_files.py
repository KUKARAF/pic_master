"""A tracked file that's gone from disk must not crash a bulk job.

Bulk face detection with the remote worker used to raise FileNotFoundError out of
RemoteFaceDetector.detect_faces (an unguarded open()), killing the whole job on the
first missing file. Now: the remote clients report an unreadable file as that file's
own error, the job logs it, never marks it damaged, and keeps going; a file with
another tracked copy on disk is processed from that copy.

Run: python tests/test_bulk_missing_files.py   (or python -m pytest tests/test_bulk_missing_files.py)
"""
import os
import tempfile
import time

import xxhash
from fastapi.testclient import TestClient
from PIL import Image

import media_manager.web as web_mod
from media_manager import worker_client
from media_manager.web import create_app


class FakeClient:
    def __init__(self):
        self.sent = []

    def record(self, op, name):
        self.sent.append(name)

    def request(self, path, payload, timeout=None):
        return {'faces': [], 'error': None}


def _wait(client):
    for _ in range(400):
        s = client.get('/api/detect-faces/status').json()
        if not s['running']:
            return s
        time.sleep(0.05)
    raise AssertionError('job did not finish')


def run():
    # Remote clients: a missing file is a per-file error, not an exception.
    det = worker_client.RemoteFaceDetector(FakeClient())
    res = det.detect_faces(['/nope/missing.jpg'])
    assert res[0][1] == [] and 'cannot read file' in res[0][2], res
    fake = FakeClient()
    det = worker_client.RemoteFaceDetector(fake)

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    os.makedirs(os.path.join(root, 'p'))
    names = ['a.jpg', 'gone.jpg', 'moved.jpg', 'c.jpg']
    for i, n in enumerate(names):
        Image.new('RGB', (32, 32), (i * 40, 0, 0)).save(os.path.join(root, 'p', n))
    app = create_app(root)
    db, _errors, _manual = app.state.dbs
    ids = {}
    for n in names:
        data = open(os.path.join(root, 'p', n), 'rb').read()
        cs = xxhash.xxh64(data).hexdigest()
        db.upsert_file_path('p/' + n, cs, size=len(data), modified_time=1)
        ids[n] = db.find_file_id_by_checksum(cs)
    # moved.jpg: the primary copy is deleted but a second tracked copy exists.
    data = open(os.path.join(root, 'p', 'moved.jpg'), 'rb').read()
    os.makedirs(os.path.join(root, 'q'))
    open(os.path.join(root, 'q', 'moved.jpg'), 'wb').write(data)
    db.upsert_file_path('q/moved.jpg', xxhash.xxh64(data).hexdigest(), size=len(data), modified_time=0)
    db.conn.execute("UPDATE file_paths SET last_seen_at = last_seen_at + 100 WHERE path = 'p/moved.jpg'")
    db.conn.commit()
    os.remove(os.path.join(root, 'p', 'moved.jpg'))
    os.remove(os.path.join(root, 'p', 'gone.jpg'))

    web_mod._get_face_detector = lambda: det
    client = TestClient(app)
    assert client.post('/api/detect-faces/start').json()['started'] is True
    s = _wait(client)
    assert s['error'] is None and s['done'] == 4, s
    assert sorted(fake.sent) == ['a.jpg', 'c.jpg', 'moved.jpg'], fake.sent   # moved via q/
    assert db.get_file_by_id(ids['gone.jpg'])['broken'] is None             # missing ≠ damaged
    print('OK  missing file skipped + logged (not damaged), other copy used, job finishes')

    print('ALL ASSERTIONS PASSED')


def test_bulk_missing_files():
    run()


if __name__ == '__main__':
    run()
