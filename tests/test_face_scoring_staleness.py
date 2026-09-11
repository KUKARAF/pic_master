"""An empty review stream must never be silently reported as "nothing matches".

Every review query reads the materialized top-K in face_candidates, so a face that
has never been scored is invisible to them — a never-scored library and a fully
reviewed one produce byte-identical (empty) SQL results. Naming somebody and
immediately searching for them is exactly the never-scored case, so the stream has
to distinguish the two and get the scoring run going by itself; nothing else in the
app ever triggers one.

Run standalone: python tests/test_face_scoring_staleness.py
"""
import os
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

from media_manager.web import create_app


def _unit(*seed):
    """A deterministic unit vector; two faces built from the same seed match at 1.0."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(app):
    """One named person plus an unidentified face that is unmistakably them."""
    db, _errors, manual = app.state.dbs
    fid = db.upsert_file_path('a.jpg', 'a' * 8, size=100)
    db.conn.commit()
    same = _unit(1)

    face_id = db.add_manual_face(fid, [1, 2, 40, 40], same.tobytes(), 0.9)
    manual.conn.execute(
        '''INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source,
                              source_face_id, image_width, image_height, created_at)
           VALUES (?, 'Ross', 1,2,40,40, ?, 'manual', NULL, 100, 100, ?)''',
        ('a' * 8, same.tobytes(), int(time.time())))
    manual.conn.commit()
    return face_id


def _stream(client, name='Ross'):
    return client.post('/api/face-suggestions/next?count=10&identity=' + name,
                       json={'exclude': []}).json()


def _settle(client, limit=200):
    for _ in range(limit):
        st = client.get('/api/face-scoring/status').json()
        if not st['running'] and not st['pending']:
            return st
        time.sleep(0.05)
    raise AssertionError('scoring never settled: %r' % (st,))


def run():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    _seed(app)
    client = TestClient(app)

    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        ok = ok and cond
        print(('  PASS  ' if cond else '  FAIL  ') + name + (('   [%s]' % (extra,)) if extra else ''))

    # Nothing has ever been scored: the person's own face is there, but invisible.
    first = _stream(client)
    check('stale library yields no cards yet', first['cards'] == [])
    check('the response says WHY it is empty', first['scoring']['pending'] > 0,
          repr(first['scoring']))
    check('and starts a scoring run by itself',
          first['scoring']['running'] or first['scoring']['pending'] > 0,
          repr(first['scoring']))

    _settle(client)

    # Same request, no user action in between, now finds the match.
    second = _stream(client)
    check('the same query now returns the match', len(second['cards']) == 1,
          '%d cards' % len(second['cards']))
    if second['cards']:
        check('and it is the right person at a real score',
              second['cards'][0]['identity'] == 'Ross' and second['cards'][0]['score'] > 0.9,
              '%s %.3f' % (second['cards'][0]['identity'], second['cards'][0]['score']))
    check('nothing left pending', second['scoring']['pending'] == 0, repr(second['scoring']))

    # Genuine exhaustion must still read as empty — the fix must not invent a
    # permanent "still computing" state that never resolves.
    for card in second['cards']:
        client.post('/api/faces/%s/reject' % card['ref'])
    _settle(client)
    third = _stream(client)
    check('a genuinely exhausted pool reports empty and settled',
          third['cards'] == [] and not third['scoring']['pending']
          and not third['scoring']['running'], repr(third['scoring']))

    # The global review stream carries the same signal.
    review = client.post('/api/face-review/next?count=5', json={'exclude': []}).json()
    check('the global review stream reports scoring state too', 'scoring' in review,
          repr(sorted(review.keys())))

    print('\noverall:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())
