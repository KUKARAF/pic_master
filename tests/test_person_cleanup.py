"""Anchor-based person cleanup (/person/{name}/cleanup).

Run: python -m pytest tests/test_person_cleanup.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.modules.setdefault('cv2', MagicMock())

from media_manager import person_cleanup  # noqa: E402
from media_manager.web import create_app  # noqa: E402

D = 512
NAME = 'Alex'
BASE = '/api/people/Alex/cleanup'


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _at(rng, ref, cos):
    """Unit vector with cosine `cos` to `ref`."""
    o = rng.standard_normal(D)
    o = _unit(o - o.dot(ref) * ref)
    return _unit(cos * ref + np.sqrt(1 - cos * cos) * o)


class World:
    def __init__(self, tmp):
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        self.app = create_app(tmp)
        self.db, _e, self.manual = self.app.state.dbs
        self.client = TestClient(self.app)
        self.rng = np.random.default_rng(5)
        self.alex = _unit(self.rng.standard_normal(D))
        self.n = 0

    def auto_face(self, vec, det=0.9, bbox=(0, 0, 100, 100), angle=0.0):
        self.n += 1
        cs = ('f%d' % self.n).ljust(40, 'x')
        fid = self.db.upsert_file_path('f%d.jpg' % self.n, cs, size=100)
        self.db.insert_faces(fid, [{'bbox': list(bbox), 'embedding': vec, 'det_score': det,
                                    'angle': angle}], 'test')
        self.db.conn.commit()
        src = self.db.get_faces_for_file(fid)[0]['id']
        return cs, src

    def promoted(self, vec, auto_matched, **kw):
        cs, src = self.auto_face(vec, **kw)
        mid = self.manual.promote_auto_face(src, cs, [0, 0, 100, 100], vec.tobytes(), NAME, None, None,
                                            auto_matched=auto_matched)
        self.db.mark_faces_handled([src])
        return mid, src

    def assigned(self, face_id):
        r = self.manual.get_face(face_id)
        return r is not None and r['identity'] == NAME and not r['rejected']

    def preview(self, **q):
        r = self.client.get(BASE + '/preview', params=q)
        assert r.status_code == 200, r.text
        return r.json()


@pytest.fixture
def w():
    with tempfile.TemporaryDirectory() as tmp:
        yield World(tmp)


def build(w):
    """Anchor + near/mid/far machine faces + protected far faces of every kind."""
    rng, alex = w.rng, w.alex
    f = {}
    f['anchor'], _ = w.promoted(_at(rng, alex, 0.97), auto_matched=False)   # human-confirmed anchor
    f['near'], _ = w.promoted(_at(rng, alex, 0.80), auto_matched=True)
    f['mid'], f['mid_src'] = w.promoted(_at(rng, alex, 0.35), auto_matched=True)
    f['far1'], f['far1_src'] = w.promoted(_at(rng, alex, 0.02), auto_matched=True)
    f['far2'], f['far2_src'] = w.promoted(_at(rng, alex, -0.05), auto_matched=True)
    # Far but set by a human, three ways.
    f['human_far'], f['human_far_src'] = w.promoted(_at(rng, alex, 0.0), auto_matched=False)
    cs, _ = w.auto_face(_at(rng, alex, 0.0))
    hv = _at(rng, alex, 0.0)
    f['drawn_far'] = w.manual.add_manual_face(cs, [0, 0, 50, 50], hv.tobytes(), 100, 100)
    w.manual.assign_identity(f['drawn_far'], NAME)
    f['ok_far'], f['ok_far_src'] = w.promoted(_at(rng, alex, 0.0), auto_matched=True)
    w.manual.conn.execute('UPDATE faces SET reviewed_ok = 1 WHERE id = ?', (f['ok_far'],))
    # Legacy: origin unknown.
    f['legacy_far'], f['legacy_far_src'] = w.promoted(_at(rng, alex, 0.0), auto_matched=True)
    w.manual.conn.execute('UPDATE faces SET origin = NULL WHERE id = ?', (f['legacy_far'],))
    w.manual.conn.commit()
    w.client.put(BASE + '/anchors', json={'refs': ['manual:%d' % f['anchor']]})
    return f


PROTECTED = ('human_far', 'drawn_far', 'ok_far')


def test_page_renders_and_old_routes_gone(w):
    r = w.client.get('/person/Alex/cleanup')
    assert r.status_code == 200 and 'Pick anchors' in r.text
    assert w.client.post('/api/person/Alex/broom-candidates?suspect=manual:1', json={'exclude': []}).status_code == 404
    assert w.client.get('/api/faces/cleanup-queue').status_code == 404
    assert w.client.post('/api/faces/manual:1/mark-reviewed').status_code == 404
    assert w.client.post('/api/faces/manual:1/not-identity', json={'name': 'Alex'}).status_code == 404
    assert w.client.delete('/api/faces/auto:1/not-identity?name=Alex').status_code == 404
    assert 'cleanup-low-confidence' not in w.client.get('/faces').text


def test_quality_ranking(w):
    rng = w.rng
    good, _ = w.promoted(_at(rng, w.alex, .9), True, det=0.99, bbox=(0, 0, 220, 220), angle=0)
    small, _ = w.promoted(_at(rng, w.alex, .9), True, det=0.99, bbox=(0, 0, 40, 40), angle=0)
    tilted, _ = w.promoted(_at(rng, w.alex, .9), True, det=0.99, bbox=(0, 0, 220, 220), angle=60)
    weak, _ = w.promoted(_at(rng, w.alex, .9), True, det=0.3, bbox=(0, 0, 220, 220), angle=0)
    # promote_auto_face stores bbox as given; sizes come from the manual row
    w.manual.conn.execute('UPDATE faces SET x2=220,y2=220 WHERE id IN (?,?,?)', (good, tilted, weak))
    w.manual.conn.execute('UPDATE faces SET x2=40,y2=40 WHERE id=?', (small,))
    w.manual.conn.commit()
    d = w.client.get(BASE + '/faces').json()
    order = [x['face_id'] for x in d['faces']]
    assert order[0] == good and order.index(good) < order.index(tilted) < order.index(small)
    assert order.index(good) < order.index(weak)
    assert d['total'] == 4
    assert person_cleanup.quality_score(0.99, 220, 220, 0) > person_cleanup.quality_score(0.99, 40, 40, 0)


def test_bands_and_threshold_movement(w):
    f = build(w)
    p = w.preview()
    c = p['counts']
    assert (c['keep'], c['uncertain'], c['remove']) == (1, 1, 2), c
    assert c['protected'] == 3 and c['legacy_skipped'] == 1
    refs = {x['ref'] for x in p['remove_sample']}
    assert refs == {'manual:%d' % f['far1'], 'manual:%d' % f['far2']}
    p2 = w.preview(keep=0.9, remove=0.5)   # mid and near now fall in/under bands
    assert (p2['counts']['keep'], p2['counts']['uncertain'], p2['counts']['remove']) == (0, 1, 3)
    p3 = w.preview(keep=0.3, remove=0.5)   # remove > keep is clamped to keep
    assert p3['remove'] == p3['keep'] == 0.3
    assert sum(b['count'] for b in p['histogram']) == c['candidates']
    p4 = w.preview(include_legacy=1)
    assert p4['counts']['remove'] == 3
    assert (p['defaults']['keep'], p['defaults']['remove']) == (0.45, 0.25)


def test_preview_does_not_mutate(w):
    f = build(w)
    snap = lambda: ([tuple(r) for r in w.manual.conn.execute('SELECT * FROM faces ORDER BY id')],
                    w.manual.conn.execute('SELECT COUNT(*) FROM face_identity_negatives').fetchone()[0],
                    w.manual.conn.execute('SELECT COUNT(*) FROM person_cleanup_batches').fetchone()[0])
    before = snap()
    w.preview(); w.preview(keep=0.9, remove=0.8)
    w.client.post(BASE + '/uncertain', json={'exclude': []})
    assert snap() == before


def test_apply_requires_confirm_anchor_and_matching_count(w):
    f = build(w)
    assert w.client.post(BASE + '/apply', json={}).status_code == 400            # no confirm
    assert w.client.post(BASE + '/apply', json={'confirm': True, 'expected_remove': 5}).status_code == 409
    w.client.put(BASE + '/anchors', json={'refs': []})
    assert w.client.post(BASE + '/apply', json={'confirm': True}).status_code == 400   # no anchor
    assert w.preview()['needs_anchor']
    assert w.assigned(f['far1'])


def test_apply_removes_only_far_auto_faces_and_writes_negatives(w):
    f = build(w)
    r = w.client.post(BASE + '/apply', json={'confirm': True, 'expected_remove': 2})
    assert r.status_code == 200 and r.json()['removed'] == 2, r.text
    for k in ('far1', 'far2'):
        assert w.manual.get_face(f[k]) is None                       # un-assigned: row gone, pool again
        assert w.manual.is_face_negated(f[k + '_src'], NAME)
        assert w.db.conn.execute('SELECT handled FROM faces WHERE id=?', (f[k + '_src'],)).fetchone()[0] == 0
    for k in ('anchor', 'near', 'mid', 'legacy_far') + PROTECTED:
        assert w.assigned(f[k]), k
    for k in ('human_far_src', 'ok_far_src', 'legacy_far_src', 'mid_src'):
        assert not w.manual.is_face_negated(f[k], NAME), k
    # re-run: nothing left to do (idempotent)
    r2 = w.client.post(BASE + '/apply', json={'confirm': True})
    assert r2.json()['removed'] == 0
    assert w.preview()['counts']['remove'] == 0


def test_apply_endpoint_rechecks_protection_server_side(w):
    """Even if a protected id is handed straight to the manual layer, it is refused."""
    f = build(w)
    res = w.manual.cleanup_remove_faces(NAME, [f['human_far'], f['drawn_far'], f['ok_far'], f['anchor'],
                                               f['legacy_far']], include_legacy=False)
    assert res['removed'] == [] and res['batch_id'] is None
    assert {s['reason'] for s in res['skipped']} == {'protected', 'anchor', 'legacy'}
    res = w.manual.cleanup_remove_faces(NAME, [f['human_far'], f['drawn_far'], f['ok_far']], include_legacy=True)
    assert res['removed'] == []        # opt-in for legacy never unlocks human-set faces
    res = w.manual.cleanup_keep_faces(NAME, [f['human_far'], f['drawn_far']], include_legacy=True)
    assert res['kept'] == []
    for k in PROTECTED:
        assert w.assigned(f[k])
    assert w.manual.get_face(f['human_far'])['origin'] == 'human'
    assert w.manual.get_face(f['human_far'])['reviewed_ok'] == 0   # state not rewritten


def test_undo_restores_exactly_and_never_touches_protected(w):
    f = build(w)
    before = {k: tuple(w.manual.get_face(f[k])) for k in ('far1', 'far2')}
    prot_before = {k: tuple(w.manual.get_face(f[k])) for k in PROTECTED}
    w.client.post(BASE + '/apply', json={'confirm': True})
    r = w.client.post(BASE + '/undo')
    assert r.status_code == 200 and r.json()['restored'] == 2
    for k in ('far1', 'far2'):
        assert tuple(w.manual.get_face(f[k])) == before[k]
        assert not w.manual.is_face_negated(f[k + '_src'], NAME)
        assert w.db.conn.execute('SELECT handled FROM faces WHERE id=?', (f[k + '_src'],)).fetchone()[0] == 1
    assert {k: tuple(w.manual.get_face(f[k])) for k in PROTECTED} == prot_before
    assert w.client.post(BASE + '/undo').status_code == 404       # already undone
    assert w.preview()['counts']['remove'] == 2                    # candidates again


def test_undo_skips_face_reassigned_elsewhere(w):
    f = build(w)
    w.client.post(BASE + '/apply', json={'confirm': True})
    w.manual.promote_auto_face(f['far1_src'], 'f4'.ljust(40, 'x'), [0, 0, 5, 5], b'\0' * 2048, 'Bob', None, None)
    r = w.client.post(BASE + '/undo').json()
    assert r['restored'] == 1 and r['skipped'][0]['reason'] == 'source_reassigned'


def test_uncertain_feed_and_swipe_decisions(w):
    f = build(w)
    # add a second uncertain face closer to the middle (0.35)
    m2, m2_src = w.promoted(_at(w.rng, w.alex, 0.30), auto_matched=True)
    cards = w.client.post(BASE + '/uncertain', json={'exclude': []}).json()['cards']
    refs = [c['ref'] for c in cards]
    assert set(refs) == {'manual:%d' % f['mid'], 'manual:%d' % m2}
    assert refs[0] == 'manual:%d' % f['mid']           # 0.35 is nearest the 0.35 midpoint
    assert all(c['anchor_ref'] == 'manual:%d' % f['anchor'] for c in cards)
    assert not (set(refs) & {'manual:%d' % f[k] for k in PROTECTED})
    # keep -> sticky, never asked again
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % f['mid'], 'action': 'keep'}).json()
    assert r['ok']
    row = w.manual.get_face(f['mid'])
    assert row['origin'] == 'human' and row['reviewed_ok'] == 1 and row['identity'] == NAME
    cards = w.client.post(BASE + '/uncertain', json={'exclude': []}).json()['cards']
    assert [c['ref'] for c in cards] == ['manual:%d' % m2]
    assert w.preview()['counts']['uncertain'] == 1
    # remove -> un-assigned + negative
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % m2, 'action': 'remove'}).json()
    assert r['ok'] and w.manual.get_face(m2) is None and w.manual.is_face_negated(m2_src, NAME)
    # undo remove, undo keep
    assert w.client.delete(BASE + '/decide/%d' % m2).status_code == 200
    assert w.assigned(m2) and not w.manual.is_face_negated(m2_src, NAME)
    assert w.client.delete(BASE + '/decide/%d' % f['mid']).status_code == 200
    row = w.manual.get_face(f['mid'])
    assert row['origin'] == 'auto' and row['reviewed_ok'] == 0
    assert w.client.delete(BASE + '/decide/%d' % f['mid']).status_code == 404


def test_swipe_never_touches_manually_set_faces(w):
    f = build(w)
    snap = {k: tuple(w.manual.get_face(f[k])) for k in PROTECTED}
    for k in PROTECTED:
        for action in ('keep', 'remove'):
            r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % f[k], 'action': action,
                                                      'include_legacy': True}).json()
            assert r['ok'] is False and r['skipped'][0]['reason'] == 'protected'
        assert w.client.delete(BASE + '/decide/%d' % f[k]).status_code == 404
    w.client.post(BASE + '/apply', json={'confirm': True})
    w.client.post(BASE + '/undo')
    assert {k: tuple(w.manual.get_face(f[k])) for k in PROTECTED} == snap
    for k in PROTECTED:
        assert 'manual:%d' % f[k] not in [c['ref'] for c in w.client.post(BASE + '/uncertain', json={'exclude': []}).json()['cards']]


def test_origin_recorded_by_promotion_paths(w):
    h, _ = w.promoted(_at(w.rng, w.alex, .9), auto_matched=False)
    a, a_src = w.promoted(_at(w.rng, w.alex, .9), auto_matched=True)
    assert w.manual.get_face(h)['origin'] == 'human' and w.manual.get_face(a)['origin'] == 'auto'
    # machine re-promotion never downgrades a human confirmation; a human upgrades a machine one
    cs, src = w.auto_face(_at(w.rng, w.alex, .9))
    w.manual.promote_auto_face(src, cs, [0, 0, 5, 5], b'\0' * 2048, NAME, None, None)
    w.manual.promote_auto_face(src, cs, [0, 0, 5, 5], b'\0' * 2048, NAME, None, None, auto_matched=True)
    assert w.manual.conn.execute('SELECT origin FROM faces WHERE source_face_id=?', (src,)).fetchone()[0] == 'human'
    w.manual.promote_auto_face(a_src, 'x' * 40, [0, 0, 5, 5], b'\0' * 2048, NAME, None, None)
    assert w.manual.get_face(a)['origin'] == 'human'


def test_degenerate_people(w):
    # nobody / one face / missing embeddings / stale anchors
    assert w.preview()['needs_anchor']
    assert w.client.get(BASE + '/faces').json()['total'] == 0
    one, _ = w.promoted(_at(w.rng, w.alex, .9), True)
    assert w.client.put(BASE + '/anchors', json={'refs': ['manual:%d' % one]}).status_code == 200
    p = w.preview()
    assert not p['needs_anchor'] and p['counts']['candidates'] == 0
    bad, _ = w.promoted(_at(w.rng, w.alex, .1), True)
    w.manual.conn.execute("UPDATE faces SET embedding = x'0102' WHERE id = ?", (bad,))
    w.manual.conn.commit()
    assert w.preview()['counts']['missing_embedding'] == 1
    other = w.manual.add_manual_face('z' * 40, [0, 0, 5, 5], w.alex.tobytes(), 10, 10)
    assert w.client.put(BASE + '/anchors', json={'refs': ['manual:%d' % other]}).status_code == 400
    assert w.client.put(BASE + '/anchors', json={'refs': ['auto:1']}).status_code == 400
    w.manual.reject_face(one)                      # anchor deleted
    p = w.preview()
    assert p['needs_anchor'] and p['anchors_stale'] == 0 or p['needs_anchor']


def test_multiple_anchors_use_centroid(w):
    a1, _ = w.promoted(_at(w.rng, w.alex, .95), False)
    a2, _ = w.promoted(_at(w.rng, w.alex, .95), False)
    x, _ = w.promoted(_at(w.rng, w.alex, .6), True)
    w.client.put(BASE + '/anchors', json={'refs': ['manual:%d' % a1, 'manual:%d' % a2]})
    p = w.preview()
    assert p['counts']['candidates'] == 1 and p['counts']['keep'] == 1
    assert len(p['anchor_refs']) == 2


def test_removed_faces_not_rematched_and_matcher_cache_invalidated(w):
    f = build(w)
    far_vec = np.frombuffer(w.manual.get_face(f['far1'])['embedding'], dtype=np.float32).copy()
    names, _m = w.manual.get_named_face_matrix()
    n_before = len(names)
    w.client.post(BASE + '/apply', json={'confirm': True})
    names, _m = w.manual.get_named_face_matrix()
    assert len(names) == n_before - 2          # trash no longer attracts new matches
    assert w.manual.is_face_negated(f['far1_src'], NAME)


def test_rename_carries_anchors_and_batches(w):
    f = build(w)
    w.client.post(BASE + '/apply', json={'confirm': True})
    w.manual.rename_identity(NAME, 'Alexa')
    assert w.manual.get_cleanup_anchor_ids('Alexa') == [f['anchor']]
    assert w.manual.get_last_cleanup_batch('Alexa') is not None
