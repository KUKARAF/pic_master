"""The two face-review streams ask different questions and must not share a cost.

  /find-person/{name}   "which unknown faces are Alice?"  -> one person's vectors
                        against the unknown pool. Self-sufficient: it is computed
                        live, per request, and must never wait on (or trigger) the
                        library-wide pass.
  /find_all_faces       "for each unknown face, who is it?" -> every unknown face
                        against every known person. This one IS the precomputed
                        ranking, so an unscored library has an empty pool and the
                        stream has to say so and get a run going.

Collapsing the first onto the second is what made opening a person's search sit
behind a multi-minute job; this pins the split so it cannot come back.

Also pins the fix for the pool being permanently empty: media.db faces.identity only
ever holds the '__indexed__' sentinel, so already-decided faces can only be excluded
via faces.handled — without it the top-scoring slice is all already-promoted rows and
the global review returns nothing forever.

Run standalone: python tests/test_face_scoring_staleness.py
"""
import os
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

from media_manager.web import create_app


def _unit(seed):
    """Deterministic unit vector; two faces from the same seed match at 1.0."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(app):
    """Ross, named, plus an unidentified face that is unmistakably him and a second
    unidentified face that is nobody."""
    db, _errors, manual = app.state.dbs
    fid = db.upsert_file_path('a.jpg', 'a' * 8, size=100)
    db.upsert_file_path('b.jpg', 'b' * 8, size=100)
    db.conn.commit()
    ross = _unit(1)
    db.add_manual_face(fid, [1, 2, 40, 40], ross.tobytes(), 0.9)
    db.add_manual_face(fid, [50, 2, 90, 40], _unit(99).tobytes(), 0.9)
    manual.conn.execute(
        '''INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source,
                              source_face_id, image_width, image_height, created_at)
           VALUES (?, 'Ross', 1,2,40,40, ?, 'manual', NULL, 100, 100, ?)''',
        ('b' * 8, ross.tobytes(), int(time.time())))
    manual.conn.commit()


def _person_stream(client, name='Ross'):
    return client.post('/api/face-suggestions/next?count=10&identity=' + name,
                       json={'exclude': []}).json()


def _global_stream(client):
    return client.post('/api/face-review/next?count=10', json={'exclude': []}).json()


def _settle(client, limit=400):
    st = {}
    for _ in range(limit):
        st = client.get('/api/face-scoring/status').json()
        if not st['running'] and not st['pending']:
            return st
        time.sleep(0.05)
    raise AssertionError('suggestion pass never settled: %r' % (st,))


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

    # --- the person-scoped stream is live and self-sufficient ------------------
    # Nothing has been precomputed at this point. It must answer anyway.
    first = _person_stream(client)
    check('a never-scored library still answers "which faces are Ross?"',
          len(first['cards']) == 1, '%d cards' % len(first['cards']))
    if first['cards']:
        check('and finds the right face at a real score',
              first['cards'][0]['identity'] == 'Ross' and first['cards'][0]['score'] > 0.9,
              '%s %.3f' % (first['cards'][0]['identity'], first['cards'][0]['score']))
    # The regression this file exists for: the person branch paying for the
    # all-vs-all pass it does not need.
    check('and does NOT start the library-wide pass',
          not first['scoring']['running'], repr(first['scoring']))

    # --- the global stream is the one that needs the precompute ----------------
    glob = _global_stream(client)
    check('the global review is empty until the pass runs', glob['cards'] == [],
          '%d cards' % len(glob['cards']))
    check('and says why, rather than reporting "nothing matches"',
          glob['scoring']['pending'] > 0 or glob['scoring']['running'],
          repr(glob['scoring']))
    check('and gets the pass going by itself',
          glob['scoring']['running'] or glob['scoring']['pending'] > 0,
          repr(glob['scoring']))

    _settle(client)

    after = _global_stream(client)
    check('once it lands the global review has the face', len(after['cards']) == 1,
          '%d cards' % len(after['cards']))
    if after['cards']:
        names = [c['name'] for c in after['cards'][0]['candidates']]
        check('with a ranked candidate list to arrow through', 'Ross' in names, str(names))

    # --- handled: the permanently-empty bug --------------------------------------
    # Deciding a face must remove it from the pool. Before faces.handled existed this
    # filtering happened in Python AFTER the SQL LIMIT, so decided faces kept occupying
    # the top-scoring slice and the queue drained to nothing forever.
    for card in after['cards']:
        client.post('/api/faces/%s/reject' % card['ref'])
    drained = _global_stream(client)
    check('a decided face leaves the pool', drained['cards'] == [],
          '%d cards' % len(drained['cards']))
    check('an exhausted pool reads as settled, not as "still computing"',
          not drained['scoring']['running'] and not drained['scoring']['pending'],
          repr(drained['scoring']))

    print('\noverall:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())
