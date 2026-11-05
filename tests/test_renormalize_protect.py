"""Renormalize backfill protections: hand-set (trust 3) faces are never rewritten,
video / frame faces are never stamped as repaired, and the old-worker fallback
for the orientation gate is visible rather than silent."""
import os, tempfile, time

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from media_manager import web as webmod, worker_client, worker_protocol
from media_manager.web import create_app
from media_manager.database import FACE_NORM_VERSION


def _wait(client):
    for _ in range(200):
        st = client.get('/api/faces/renormalize/status').json()
        if not st['running']:
            return st
        time.sleep(0.05)
    raise AssertionError('job did not finish')


class FixingDetector:
    """Rewrites every face it is handed (embedding 0.5s, angle 180) and remembers
    how many faces it saw."""
    def __init__(self):
        self.seen = 0
    def normalize_faces(self, img, faces):
        self.seen += len(faces)
        out = []
        for f in faces:
            g = dict(f)
            g['embedding'] = np.full(512, 0.5, dtype=np.float32)
            g['angle'] = 180.0
            out.append(g)
        return out


def _setup():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    Image.new('RGB', (64, 64), (120, 120, 120)).save(os.path.join(tmp, 'a.jpg'))
    open(os.path.join(tmp, 'v.mp4'), 'wb').write(b'not really a video')
    app = create_app(tmp)
    db, _errors, manual = app.state.dbs
    fid = db.upsert_file_path('a.jpg', 'a' * 8, size=100)
    vid = db.upsert_file_path('v.mp4', 'b' * 8, size=100)
    db.conn.commit()
    return app, db, manual, fid, vid


def test_backfill_leaves_hand_set_and_video_faces_alone():
    app, db, manual, fid, vid = _setup()
    old = np.ones(512, dtype=np.float32).tobytes()
    plain = db.add_manual_face(fid, [1, 2, 40, 40], old, 0.9)
    hand = db.add_manual_face(fid, [2, 2, 41, 41], old, 0.9)
    auto_named = db.add_manual_face(fid, [3, 2, 42, 42], old, 0.9)
    frame = db.add_manual_face(fid, [4, 2, 43, 43], old, 0.9, frame_index=3)
    video = db.add_manual_face(vid, [5, 2, 44, 44], old, 0.9)
    db.conn.execute("UPDATE faces SET norm_version = 0 WHERE bbox != '[]'")
    db.conn.commit()

    # `hand` is a trust-3 promoted reference; `auto_named` is an auto-trust one.
    hand_manual = manual.promote_auto_face(hand, 'a' * 8, [2, 2, 41, 41], old, 'Alice', None, None, trust=3)
    auto_manual = manual.promote_auto_face(auto_named, 'a' * 8, [3, 2, 42, 42], old, 'Bob', None, None,
                                           auto_matched=True)
    assert manual.get_trusted_source_ids([plain, hand, auto_named]) == {hand}

    assert db.count_unnormalized_faces() == 3, db.count_unnormalized_faces()  # not frame, not video

    det = FixingDetector()
    webmod._get_face_detector = lambda: det
    client = TestClient(app)
    client.post('/api/faces/renormalize/start')
    st = _wait(client)
    assert st['error'] is None, st
    assert st['skipped_manual'] == 1, st
    assert st['changed'] == 2, st            # plain + auto-named only
    assert det.seen == 2, det.seen           # the hand-set face never reached the detector

    row = lambda i: db.conn.execute('SELECT embedding, angle, norm_version FROM faces WHERE id=?', (i,)).fetchone()
    assert bytes(row(hand)['embedding']) == old and row(hand)['angle'] == 0.0     # byte-identical
    assert row(hand)['norm_version'] == FACE_NORM_VERSION                          # stamped done
    assert bytes(manual.get_face(hand_manual)['embedding']) == old                 # manual row untouched
    assert bytes(row(plain)['embedding']) != old and row(plain)['angle'] == 180.0
    assert bytes(manual.get_face(auto_manual)['embedding']) != old                 # auto reference refreshed
    # frame + video faces: not rewritten, NOT stamped as repaired.
    for i in (frame, video):
        assert bytes(row(i)['embedding']) == old and row(i)['norm_version'] == 0, i


def test_old_worker_fallback_is_visible(capfd):
    class Client:
        def record(self, *a): pass
        def request(self, path, payload, **kw):
            return {'faces': [], 'error': None}      # no "orient": a pre-gate worker
    det = worker_client.RemoteFaceDetector.__new__(worker_client.RemoteFaceDetector)
    det.client = Client()
    assert det._note_orient({'faces': []}, 'embed_bbox') is False
    assert det.orient_gate is False
    err = capfd.readouterr().err
    assert 'predates the always-on orientation gate' in err
    det._note_orient({'faces': []}, 'embed_bbox')             # warned once only
    assert capfd.readouterr().err == ''
    assert det._note_orient({'orient': worker_protocol.ORIENT_GATE_VERSION}, 'embed_bbox') is True
    assert det.orient_gate is True
    # embed_bbox itself works against such a worker, and sends auto_orient on the wire.
    sent = {}
    class C2(Client):
        def request(self, path, payload, **kw):
            sent.update(payload)
            return {'bbox': None, 'embedding': np.ones(512, np.float32).tobytes(),
                    'det_score': 0.8, 'angle': 0.0, 'error': None}
    det.client = C2()
    det._orient_warned = False
    res = det.embed_bbox(np.zeros((50, 50, 3), np.uint8), [1, 1, 30, 30], auto_orient=False)
    assert sent['auto_orient'] is False and res['det_score'] == 0.8


if __name__ == '__main__':
    test_backfill_leaves_hand_set_and_video_faces_alone()
    print('backfill protections PASS')
