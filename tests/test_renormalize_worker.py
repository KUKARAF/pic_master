"""Throwaway: a stale/unreachable face worker must ABORT the re-normalization
backfill, not silently stamp the whole library as normalized.

Regression guard for the hazard: normalize_faces raising per group used to be
caught, logged, and the rows advanced to the current norm_version — so against a
worker too old to know the op, every face got marked repaired while nothing was,
and the job reported success with changed=0.
"""
import os, tempfile, time

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from media_manager import web as webmod, worker_client
from media_manager.web import create_app
from media_manager.database import FACE_NORM_VERSION


def norm_versions(db):
    return [r[0] for r in db.conn.execute(
        "SELECT norm_version FROM faces WHERE bbox != '[]'")]


def run():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    Image.new('RGB', (64, 64), (120, 120, 120)).save(os.path.join(tmp, 'a.jpg'))

    app = create_app(tmp)
    db, _errors, _manual = app.state.dbs
    fid = db.upsert_file_path('a.jpg', 'a' * 8, size=100)
    db.conn.commit()
    emb = np.ones(512, dtype=np.float32).tobytes()
    for _ in range(3):
        db.add_manual_face(fid, [1, 2, 40, 40], emb, 0.9)
    db.conn.execute("UPDATE faces SET norm_version = 0 WHERE bbox != '[]'")
    db.conn.commit()

    assert norm_versions(db) == [0, 0, 0], norm_versions(db)
    assert db.count_unnormalized_faces() == 3

    class StaleWorker:
        """What a worker deployed before this change looks like: it has no
        normalize_faces request handler, so the client raises."""
        def normalize_faces(self, img, faces):
            raise worker_client.WorkerError('no matching request handler: normalize_faces')

    webmod._get_face_detector = lambda: StaleWorker()

    client = TestClient(app)
    assert client.post('/api/faces/renormalize/start').status_code == 200
    for _ in range(100):
        st = client.get('/api/faces/renormalize/status').json()
        if not st['running']:
            break
        time.sleep(0.1)

    print('job status:', st)
    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        ok = ok and cond
        print(('  PASS  ' if cond else '  FAIL  ') + name + (('   [' + extra + ']') if extra else ''))

    check('job reports an error instead of succeeding', bool(st['error']), str(st['error']))
    check('error names the worker and tells you to redeploy',
          bool(st['error']) and 'worker' in st['error'].lower() and 'redeploy' in st['error'].lower())
    check('error says nothing was modified',
          bool(st['error']) and 'no rows were modified' in st['error'].lower())
    check('changed stays 0', st['changed'] == 0, str(st['changed']))
    check('NO face was stamped as normalized', norm_versions(db) == [0, 0, 0], str(norm_versions(db)))
    check('the work is still queued for a retry',
          db.count_unnormalized_faces() == 3, str(db.count_unnormalized_faces()))

    # And the healthy path still advances rows, so the abort is not just "never works".
    class GoodWorker:
        def normalize_faces(self, img, faces):
            out = []
            for f in faces:
                g = dict(f)
                g['embedding'] = np.full(512, 0.5, dtype=np.float32)
                g['angle'] = 180.0
                out.append(g)
            return out

    webmod._get_face_detector = lambda: GoodWorker()
    client.post('/api/faces/renormalize/start')
    for _ in range(100):
        st2 = client.get('/api/faces/renormalize/status').json()
        if not st2['running']:
            break
        time.sleep(0.1)
    check('healthy worker still completes', st2['error'] is None, str(st2['error']))
    check('healthy worker rewrites the rows', st2['changed'] == 3, str(st2['changed']))
    check('rows now stamped at the current version',
          norm_versions(db) == [FACE_NORM_VERSION] * 3, str(norm_versions(db)))

    print('\noverall:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())
