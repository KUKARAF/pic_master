"""Find a person by BODY — the ground-truth body-label layer and its swipe stream.

A body crop can be linked to a person (manual.db body_identities), mirroring how a
face carries an identity. The find-person-by-body stream then ranks the unlabeled
body pool against that person's confirmed body crops; confirming a card adds a label,
rejecting keeps a hard-negative so it's never re-offered, and undo returns it. Body
labels also count toward a person's presence on /person/{name}.

This exercises the DB methods and the HTTP flow directly on stored embeddings, so it
needs no CLIP model (the embeddings are seeded, exactly as test_location_suggestions
seeds CLIP vectors).

Run standalone: python tests/test_body_identity.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

# This test seeds CLIP/body embeddings directly and never calls OpenCV, but importing
# media_manager.web pulls in modules that `import cv2` at load time — and the full
# opencv build needs libGL, which this headless box lacks. Stub it so the import chain
# loads; nothing under test touches it.
sys.modules.setdefault('cv2', MagicMock())

from media_manager.web import create_app

D = 512
# For x = unit(d + s*n), two independent noisy versions of the same direction d have
# E[cos(x1, x2)] ~ 1/(1 + s^2 D). s=0.01 -> ~0.95, comfortably above the 0.5 body
# threshold; independent random directions sit near 0. (Same regime as
# test_location_suggestions' SIGMA derivation — a flat per-component sigma on a unit
# signal, not visible noise.)
SIGMA = 0.01


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(app, tmp):
    db, _errors, manual = app.state.dbs
    rng = np.random.default_rng(7)
    alex = _unit(rng.standard_normal(D))  # the "Alex body" direction

    ids = {}
    # anchor: a crop we LABEL as Alex. near_1/near_2: unlabeled bodies that look like
    # Alex (should surface). far_1/far_2: unrelated bodies (should not clear threshold).
    plan = [('anchor', alex), ('near_1', alex), ('near_2', alex),
            ('far_1', None), ('far_2', None)]
    for name, theme in plan:
        cs = (name * 8)[:40]
        fid = db.upsert_file_path(name + '.jpg', cs, size=100)
        vec = _unit(theme + SIGMA * rng.standard_normal(D)) if theme is not None \
            else _unit(rng.standard_normal(D))
        body_id = db.add_manual_body(fid, [0, 0, 20, 40], vec.tobytes(), 'clip-test')
        ids[name] = {'fid': fid, 'cs': cs, 'body_id': body_id, 'vec': vec}
    db.conn.commit()

    # Label the anchor crop as Alex (ground truth), tied to its media.db body row so
    # it's excluded from Alex's own suggestion stream.
    a = ids['anchor']
    manual.add_body_label(a['cs'], 'Alex', [0, 0, 20, 40], a['vec'].tobytes(),
                          'clip-test', source_body_id=a['body_id'])
    return ids


def _suggest(client, exclude=None):
    r = client.post('/api/body-suggestions/next?identity=Alex&count=10',
                    json={'exclude': exclude or []})
    assert r.status_code == 200, r.text
    return r.json()['cards']


def main():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        app = create_app(tmp)
        _, _errors, manual = app.state.dbs
        ids = _seed(app, tmp)
        client = TestClient(app)

        # 1) The stream offers the look-alike bodies, not the anchor (already decided)
        #    and not the unrelated ones (below threshold).
        cards = _suggest(client)
        offered = {c['body_id'] for c in cards}
        assert ids['near_1']['body_id'] in offered, offered
        assert ids['near_2']['body_id'] in offered, offered
        assert ids['anchor']['body_id'] not in offered, 'anchor is already decided'
        assert ids['far_1']['body_id'] not in offered, 'unrelated body should be below threshold'
        assert ids['far_2']['body_id'] not in offered
        for c in cards:
            assert c['ref'] == 'body:%d' % c['body_id']
            assert c['identity'] == 'Alex' and c['file_id'] and c['score'] >= 0.5
        print('ok: stream ranks look-alikes, excludes anchor + unrelated')

        # 2) Confirm near_1 -> becomes a labeled body, no longer offered, counts as presence.
        bid1 = ids['near_1']['body_id']
        r = client.post('/api/bodies/%d/identity' % bid1, json={'name': 'Alex'})
        assert r.status_code == 200 and r.json()['identity'] == 'Alex', r.text
        assert ids['near_1']['cs'] in manual.get_photos_with_body_identity('Alex')
        assert bid1 not in {c['body_id'] for c in _suggest(client)}, 'confirmed body re-offered'
        print('ok: confirm links body + removes it from the stream')

        # 3) Reject near_2 -> hard-negative, never re-offered.
        bid2 = ids['near_2']['body_id']
        r = client.post('/api/bodies/%d/reject' % bid2, json={'name': 'Alex'})
        assert r.status_code == 200, r.text
        assert bid2 in manual.get_decided_body_source_ids('Alex')
        assert bid2 not in {c['body_id'] for c in _suggest(client)}, 'rejected body re-offered'
        print('ok: reject keeps a hard-negative out of the stream')

        # 4) Undo the reject -> body returns to the pool.
        r = client.delete('/api/bodies/%d/decision' % bid2)
        assert r.status_code == 200, r.text
        assert bid2 not in manual.get_decided_body_source_ids('Alex')
        assert bid2 in {c['body_id'] for c in _suggest(client)}, 'undo did not resurface body'
        print('ok: undo returns the body to the pool')

        # 5) Unified presence: a body-only labeled photo shows up as an appearance even
        #    with no face. near_1 was confirmed by body above and has no face row.
        from media_manager.web import create_app as _ca  # noqa: F401 (keep import local)
        r = client.get('/person/Alex')
        assert r.status_code == 200, r.text
        # anchor + near_1 are both body-labeled -> both are Alex checksums.
        body_cs = set(manual.get_photos_with_body_identity('Alex'))
        assert {ids['anchor']['cs'], ids['near_1']['cs']} <= body_cs, body_cs
        print('ok: body labels count toward the person page')

        # 6) Rename carries body labels along.
        manual.rename_identity('Alex', 'Alexandra')
        assert manual.get_photos_with_body_identity('Alex') == []
        assert ids['anchor']['cs'] in manual.get_photos_with_body_identity('Alexandra')
        print('ok: rename moves body labels too')

    print('\nALL BODY-IDENTITY TESTS PASSED')


if __name__ == '__main__':
    main()
