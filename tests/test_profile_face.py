"""Profile picture override (🖼️ on /person/{name}).

A user-picked face replaces the automatic per-person crop everywhere: the /faces
grid (newest face), identity pickers (oldest face) and name suggestions. Only the
person's own non-rejected faces can be picked; "automatic" drops the pick.

Run: python tests/test_profile_face.py   (or python -m pytest tests/test_profile_face.py)
"""
import os
import tempfile

import numpy as np
from fastapi.testclient import TestClient

from media_manager.web import create_app


def run():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    app = create_app(root)
    manual = app.state.dbs[2]
    client = TestClient(app)

    emb = np.ones(512, dtype=np.float32).tobytes()
    ids = []
    for i in range(3):
        fid = manual.add_manual_face('cs%d' % i, (0, 0, 10, 10), emb, 100, 100)
        manual.conn.execute('UPDATE faces SET identity = ? WHERE id = ?', ('Ann', fid))
        ids.append(fid)
    other = manual.add_manual_face('cs9', (0, 0, 10, 10), emb, 100, 100)
    manual.conn.execute('UPDATE faces SET identity = ? WHERE id = ?', ('Bob', other))
    manual.conn.commit()

    def reps():
        summary = {e['identity']: e['ref'] for e in manual.get_identity_summary()}
        return (summary['Ann'], manual.get_representative_face_ids()['Ann'],
                manual.get_identity_reference_faces()['Ann'])

    # Automatic: /faces grid shows the newest, pickers the oldest.
    assert reps() == ('manual:%d' % ids[2], ids[0], ids[0]), reps()

    url = '/api/person/Ann/profile-face'
    r = client.put(url, json={'ref': 'manual:%d' % ids[1]})
    assert r.status_code == 200, r.text
    assert reps() == ('manual:%d' % ids[1], ids[1], ids[1]), reps()
    assert client.get('/api/person/Ann/faces').json()['profile_ref'] == 'manual:%d' % ids[1]

    # Someone else's face, an auto ref, or garbage -> 400, pick unchanged.
    for bad in ('manual:%d' % other, 'auto:1', 'nope'):
        assert client.put(url, json={'ref': bad}).status_code == 400, bad
    assert manual.get_profile_face_id('Ann') == ids[1]

    # Re-picking replaces; a rejected pick falls back to automatic.
    client.put(url, json={'ref': 'manual:%d' % ids[0]})
    assert manual.get_profile_face_ids() == {'Ann': ids[0]}
    manual.reject_face(ids[0])
    assert manual.get_profile_face_id('Ann') is None
    client.put(url, json={'ref': 'manual:%d' % ids[2]})

    # "Use automatic".
    assert client.delete(url).status_code == 200
    assert manual.get_profile_face_id('Ann') is None
    assert reps()[1] == ids[1]

    print('OK  picked face overrides /faces grid, pickers and suggestions')
    print('OK  only own non-rejected faces; re-pick replaces; automatic clears')
    print('ALL ASSERTIONS PASSED')


def test_profile_face():
    run()


if __name__ == '__main__':
    run()
