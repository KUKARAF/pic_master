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

    print('\nBODY-IDENTITY TESTS PASSED')


def _poll(client, url, timeout=10.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(url).json()
        if not s['running']:
            return s
        time.sleep(0.02)
    raise AssertionError('job did not finish: ' + url)


def main_autolink():
    """Naming a face auto-links the body that contains it; human decisions win; the
    backfill links every already-named face."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        app = create_app(tmp)
        db, _errors, manual = app.state.dbs
        client = TestClient(app)
        rng = np.random.default_rng(11)

        def photo(name):
            cs = (name * 8)[:40]
            fid = db.upsert_file_path(name + '.jpg', cs, size=100)
            return fid, cs

        # Photo 1: a detected body box that fully contains a face box -> naming the
        # face must auto-link that body.
        fid1, cs1 = photo('p1')
        body1 = db.add_manual_body(fid1, [0, 0, 100, 200], _unit(rng.standard_normal(D)).tobytes(), 'clip-test')
        db.conn.commit()
        face1 = manual.add_manual_face(cs1, [10, 10, 30, 30], _unit(rng.standard_normal(D)).tobytes(), 100, 200)
        r = client.post('/api/faces/manual:%d/identity' % face1, json={'name': 'Bob'})
        assert r.status_code == 200, r.text
        lab = manual.get_body_label_by_source(body1)
        assert lab and lab['identity'] == 'Bob' and lab['source'] == 'face-link', lab
        assert cs1 in manual.get_photos_with_body_identity('Bob')
        print('ok: naming a face auto-links its containing body')

        # Photo 2: a body a human already decided (manual label for someone else) must
        # NOT be clobbered when a face there is named.
        fid2, cs2 = photo('p2')
        body2 = db.add_manual_body(fid2, [0, 0, 100, 200], _unit(rng.standard_normal(D)).tobytes(), 'clip-test')
        db.conn.commit()
        manual.add_body_label(cs2, 'Carol', [0, 0, 100, 200], _unit(rng.standard_normal(D)).tobytes(),
                              'clip-test', source_body_id=body2, source='manual')
        face2 = manual.add_manual_face(cs2, [10, 10, 30, 30], _unit(rng.standard_normal(D)).tobytes(), 100, 200)
        r = client.post('/api/faces/manual:%d/identity' % face2, json={'name': 'Bob'})
        assert r.status_code == 200, r.text
        lab2 = manual.get_body_label_by_source(body2)
        assert lab2 and lab2['identity'] == 'Carol' and lab2['source'] == 'manual', lab2
        print('ok: a human-decided body is never clobbered by the auto-linker')

        # Photo 3: a body box that does NOT contain the face -> no link.
        fid3, cs3 = photo('p3')
        body3 = db.add_manual_body(fid3, [0, 0, 5, 5], _unit(rng.standard_normal(D)).tobytes(), 'clip-test')
        db.conn.commit()
        face3 = manual.add_manual_face(cs3, [50, 50, 80, 80], _unit(rng.standard_normal(D)).tobytes(), 100, 200)
        r = client.post('/api/faces/manual:%d/identity' % face3, json={'name': 'Bob'})
        assert r.status_code == 200, r.text
        assert manual.get_body_label_by_source(body3) is None, 'non-containing body must not link'
        print('ok: a body that does not contain the face is left alone')

        # Backfill: unlink Bob's photo-1 body, then run the job — it must re-link it.
        manual.delete_body_decision_by_source(body1)
        assert manual.get_body_label_by_source(body1) is None
        r = client.post('/api/link-faces-to-bodies/start')
        assert r.status_code == 200 and r.json()['started'], r.text
        s = _poll(client, '/api/link-faces-to-bodies/status')
        assert s['error'] is None, s
        assert s['linked'] >= 1, s
        relab = manual.get_body_label_by_source(body1)
        assert relab and relab['identity'] == 'Bob' and relab['source'] == 'face-link', relab
        # The backfill must still respect the human decision on photo 2.
        assert manual.get_body_label_by_source(body2)['source'] == 'manual'
        print('ok: backfill re-links named faces and respects human decisions')

    print('\nFACE->BODY AUTO-LINK TESTS PASSED')


def main_suggest():
    """suggest-body-identity signal #1 (labeled face), the common cases that used to
    fail: the face sits ABOVE the drawn torso box, and the single-subject shortcut.
    Signals #2/#3 only run with the file on disk (decode/embed) so they're correctly
    skipped by these DB-only fixtures; their wiring is covered by the build route-check."""
    def suggest(client, fid, bbox):
        return client.post('/api/files/%d/suggest-body-identity' % fid, json={'bbox': bbox}).json()

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        app = create_app(tmp)
        db, _errors, manual = app.state.dbs
        client = TestClient(app)
        rng = np.random.default_rng(29)

        def photo(name):
            cs = (name * 20)[:40]
            fid = db.upsert_file_path(name + '.jpg', cs, size=100)
            db.conn.commit()
            return fid, cs

        def name_face(cs, fbox, name):
            fid = manual.add_manual_face(cs, fbox, _unit(rng.standard_normal(D)).tobytes(), 200, 300)
            client.post('/api/faces/manual:%d/identity' % fid, json={'name': name})

        # (1) Head ABOVE the torso box — the case that used to miss. Face [20,0,45,25]
        #     sits just above body [15,30,60,180]; must still suggest Dana.
        f1, cs1 = photo('p1')
        name_face(cs1, [20, 0, 45, 25], 'Dana')
        j = suggest(client, f1, [15, 30, 60, 180])
        assert j['name'] == 'Dana' and j['source'] == 'labeled-face', j
        print('ok: a face drawn ABOVE the body box is still matched (head-on-top)')

        # (2) Single-subject shortcut — the one named face on the photo is suggested even
        #     when the box misses it entirely.
        f2, cs2 = photo('p2')
        name_face(cs2, [10, 10, 30, 30], 'Eli')
        j = suggest(client, f2, [120, 150, 160, 290])
        assert j['name'] == 'Eli' and j['source'] == 'labeled-face', j
        print('ok: single named face on the photo -> suggested even if the box missed it')

        # (3) Ambiguous — two named faces, box matches NEITHER geometrically -> no guess.
        f3, cs3 = photo('p3')
        name_face(cs3, [10, 10, 30, 30], 'Finn')
        name_face(cs3, [160, 10, 185, 35], 'Gwen')
        j = suggest(client, f3, [80, 120, 110, 260])
        assert j['name'] is None, j
        print('ok: two faces, box matches neither -> no false guess')

        # (4) Ambiguous but one clearly belongs — box over Gwen's head -> Gwen.
        j = suggest(client, f3, [150, 40, 195, 260])
        assert j['name'] == 'Gwen', j
        print('ok: with two faces, the one the box belongs to wins')

    print('\nSUGGEST-IDENTITY TESTS PASSED')


def main_broom():
    """Cleanup broom: a bad anchor's victims are rankable, removable (back to unknown +
    barred from this person), excluded from that person's suggestion stream afterwards,
    and the removal is undoable. All on seeded face embeddings (no detector)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        app = create_app(tmp)
        db, _errors, manual = app.state.dbs
        client = TestClient(app)
        rng = np.random.default_rng(41)

        alex = _unit(rng.standard_normal(D))      # the real Alex direction
        junk = _unit(rng.standard_normal(D))      # the scrambled "trashcan" anchor

        def auto_face(name, vec):
            """Create a file + one auto-detected media.db face carrying this embedding."""
            cs = (name * 10)[:40]
            fid = db.upsert_file_path(name + '.jpg', cs, size=100)
            db.insert_faces(fid, [{'bbox': [0, 0, 20, 20], 'embedding': vec, 'det_score': 0.9}], 'test')
            db.conn.commit()
            return cs, fid

        # Good Alex anchors (two real faces) + the junk anchor, all confirmed as Alex.
        good1 = _unit(alex + SIGMA * rng.standard_normal(D))
        good2 = _unit(alex + SIGMA * rng.standard_normal(D))
        g1_cs, _ = auto_face('ag1', good1)
        g2_cs, _ = auto_face('ag2', good2)
        j_cs, _ = auto_face('ajunk', junk)
        gf1 = manual.add_manual_face(g1_cs, [0, 0, 20, 20], good1.tobytes(), 100, 100)
        manual.assign_identity(gf1, 'Alex')
        gf2 = manual.add_manual_face(g2_cs, [0, 0, 20, 20], good2.tobytes(), 100, 100)
        manual.assign_identity(gf2, 'Alex')
        # The suspect: a confirmed Alex face carrying the junk embedding.
        jf = manual.add_manual_face(j_cs, [0, 0, 20, 20], junk.tobytes(), 100, 100)
        manual.assign_identity(jf, 'Alex')
        # Give the suspect a source_face_id so remove/undo exercises the real path:
        # make a victim that is a PROMOTED auto face resembling the junk anchor.
        victim_vec = _unit(junk + SIGMA * rng.standard_normal(D))
        v_cs, v_fid = auto_face('avictim', victim_vec)
        v_auto_id = db.get_faces_for_file(v_fid)[0]['id']
        promoted = manual.promote_auto_face(v_auto_id, v_cs, [0, 0, 20, 20], victim_vec.tobytes(),
                                            'Alex', None, None)
        db.mark_faces_handled([v_auto_id])

        suspect_ref = 'manual:%d' % jf
        r = client.post('/api/person/Alex/broom-candidates?suspect=%s' % suspect_ref, json={'exclude': []})
        assert r.status_code == 200, r.text
        refs = {c['ref'] for c in r.json()['cards']}
        assert ('manual:%d' % promoted) in refs, refs          # victim surfaces
        assert ('manual:%d' % gf1) not in refs                 # a real Alex face does not
        assert ('manual:%d' % gf2) not in refs
        print('ok: broom surfaces the suspect\'s victims, not the real faces')

        # Remove the victim -> returned to unknown + barred from Alex.
        r = client.post('/api/faces/manual:%d/not-identity' % promoted, json={'name': 'Alex'})
        assert r.status_code == 200 and r.json()['returned_to_unknown'], r.text
        assert manual.is_face_negated(v_auto_id, 'Alex')
        assert v_cs not in {cs for _id, cs, _e in manual.get_faces_for_identity('Alex')}
        print('ok: removal returns the face to unknown AND bars it from this person')

        # The barred face must not come back via Alex's suggestion stream.
        r = client.post('/api/face-suggestions/next?identity=Alex&count=50', json={'exclude': []})
        assert v_auto_id not in {c.get('face_id') for c in r.json()['cards']}, 'negated face re-offered'
        print('ok: a barred face is excluded from the person\'s face-suggestion stream')

        # Undo -> negative dropped and the face is Alex again.
        r = client.delete('/api/faces/auto:%d/not-identity?name=Alex' % v_auto_id)
        assert r.status_code == 200, r.text
        assert not manual.is_face_negated(v_auto_id, 'Alex')
        assert v_cs in {cs for _id, cs, _e in manual.get_faces_for_identity('Alex')}
        print('ok: undo clears the negative and restores the assignment')

    print('\nCLEANUP-BROOM TESTS PASSED')


def main_match():
    """Auto-match job: base (model-free) promotion works, the per-identity negative
    blocks promotion, and the try_rotations flag is accepted and completes. The
    rotation RE-EMBED itself needs the detector + a real image, so with these DB-only
    fixtures the rotation branch no-ops (no disk files) — that path's logic is reviewed,
    not unit-run."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        app = create_app(tmp)
        db, _errors, manual = app.state.dbs
        client = TestClient(app)
        rng = np.random.default_rng(53)
        alex = _unit(rng.standard_normal(D))

        # A named Alex anchor so find_matching_identity has something to match against.
        acs = ('mref' * 10)[:40]
        db.upsert_file_path('mref.jpg', acs, size=100); db.conn.commit()
        af = manual.add_manual_face(acs, [0, 0, 20, 20], alex.tobytes(), 100, 100)
        manual.assign_identity(af, 'Alex')

        def auto(name, vec):
            cs = (name * 10)[:40]
            fid = db.upsert_file_path(name + '.jpg', cs, size=100)
            db.insert_faces(fid, [{'bbox': [0, 0, 20, 20], 'embedding': vec, 'det_score': 0.9}], 'test')
            db.conn.commit()
            return cs, db.get_faces_for_file(fid)[0]['id']

        hit_cs, hit_id = auto('mhit', _unit(alex + SIGMA * rng.standard_normal(D)))   # should match
        neg_cs, neg_id = auto('mneg', _unit(alex + SIGMA * rng.standard_normal(D)))   # matches, but barred
        manual.add_face_identity_negative(neg_id, 'Alex')

        r = client.post('/api/match-faces/start?try_rotations=1', json=None)
        assert r.status_code == 200 and r.json()['started'], r.text
        s = _poll(client, '/api/match-faces/status')
        assert s['error'] is None, s
        alex_cs = {cs for _id, cs, _e in manual.get_faces_for_identity('Alex')}
        assert hit_cs in alex_cs, 'base auto-match did not promote the matching face'
        assert neg_cs not in alex_cs, 'negated face was promoted despite the broom negative'
        assert 'rotated' in s
        print('ok: auto-match promotes matches, honours negatives, accepts try_rotations')

    print('\nAUTO-MATCH TESTS PASSED')


if __name__ == '__main__':
    main()
    main_autolink()
    main_suggest()
    main_broom()
    main_match()
    print('\nALL TESTS PASSED')
