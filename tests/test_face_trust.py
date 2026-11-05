"""Trust hierarchy for named faces (faces.trust): tier derivation per write path, the
one-time legacy backfill, trust-aware auto-matching and its escape hatch.

Run: python -m pytest tests/test_face_trust.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('cv2', MagicMock())

from test_person_cleanup import World, _at, _unit, NAME, BASE  # noqa: E402
from media_manager import person_cleanup as pc  # noqa: E402


@pytest.fixture
def w():
    with tempfile.TemporaryDirectory() as tmp:
        yield World(tmp)


def trust(w, fid):
    return w.manual.get_face(fid)['trust']


# ---------------------------------------------------------------- tier per write path

def test_trust_per_write_path(w):
    v = lambda c=0.9: _at(w.rng, w.alex, c)
    auto, _ = w.promoted(v(), auto_matched=True)
    assert trust(w, auto) == 1 and w.manual.get_face(auto)['origin'] == 'auto'
    by_hand, _ = w.promoted(v(), auto_matched=False)                 # promote_auto_face default = typed name
    assert trust(w, by_hand) == 3
    swiped, _ = w.promoted(v(), auto_matched=False, trust=2)
    assert trust(w, swiped) == 2 and w.manual.get_face(swiped)['origin'] == 'human'
    cs, _src = w.auto_face(v())
    drawn = w.manual.add_manual_face(cs, [0, 0, 5, 5], v().tobytes(), 10, 10)
    assert trust(w, drawn) == 0                                       # unnamed box: not a reference
    w.manual.assign_identity(drawn, NAME)
    assert trust(w, drawn) == 3
    # trust only goes up: an auto re-promotion / a swipe never lowers a hand-set face
    w.manual.promote_auto_face(w.manual.get_face(by_hand)['source_face_id'], 'x' * 40, [0, 0, 5, 5],
                               b'\0' * 2048, NAME, None, None, auto_matched=True)
    assert trust(w, by_hand) == 3
    w.manual.assign_identity(by_hand, NAME, trust=2)
    assert trust(w, by_hand) == 3
    # favorite / profile picture are hand marks
    fav, _ = w.promoted(v(), auto_matched=True)
    w.manual.bump_face_favorite(fav, 1)
    assert trust(w, fav) == 3
    prof, _ = w.promoted(v(), auto_matched=True)
    assert w.manual.set_profile_face(NAME, prof) and trust(w, prof) == 3
    # origin/trust stay consistent: human <=> trust>=2, auto <=> 1
    for r in w.manual.conn.execute("SELECT origin, trust FROM faces WHERE identity IS NOT NULL"):
        assert (r['origin'] == 'human') == (r['trust'] >= 2)


def test_face_trust_helper_fallback_derivation(w):
    mk = lambda **kw: {'origin': None, 'reviewed_ok': 0, 'favorite': 0, 'profile_at': None,
                       'bbox_source': 'auto', 'identity': 'A', 'trust': 0, **kw}

    class Row(dict):
        pass
    assert pc.face_trust(Row(mk())) == 1
    assert pc.face_trust(Row(mk(trust=2))) == 2
    assert pc.face_trust(Row(mk(reviewed_ok=1))) == 2
    assert pc.face_trust(Row(mk(favorite=1))) == 3
    assert pc.face_trust(Row(mk(origin='human'))) == 3
    assert pc.face_trust(Row(mk(bbox_source='manual'))) == 3
    assert pc.face_trust(Row(mk(identity=None))) == 0


def test_swipe_endpoint_vs_hand_naming_and_bulk_confirm(w):
    anchor, _ = w.promoted(_at(w.rng, w.alex, 0.95), auto_matched=False)
    _cs, s1 = w.auto_face(_at(w.rng, w.alex, 0.9))
    _cs, s2 = w.auto_face(_at(w.rng, w.alex, 0.9))
    r = w.client.post('/api/faces/auto:%d/identity' % s1, json={'name': NAME, 'via': 'swipe'}).json()
    assert trust(w, int(r['face_id'].split(':')[1])) == 2
    r = w.client.post('/api/faces/auto:%d/identity' % s2, json={'name': NAME}).json()
    assert trust(w, int(r['face_id'].split(':')[1])) == 3
    # a swipe on a face that is already hand-set (e.g. the split page) does not lower it
    w.client.post('/api/faces/manual:%d/identity' % anchor, json={'name': NAME, 'via': 'swipe'})
    assert trust(w, anchor) == 3
    # bulk confirm above threshold = semi-automatic
    _cs, s3 = w.auto_face(_at(w.rng, w.alex, 0.95))
    r = w.client.post('/api/identities/%s/confirm-above-threshold' % NAME, json={'threshold': 0.5})
    assert r.status_code == 200 and r.json()['confirmed'] >= 1, r.text
    row = w.manual.conn.execute('SELECT trust, origin FROM faces WHERE source_face_id = ?', (s3,)).fetchone()
    assert (row['trust'], row['origin']) == (2, 'human')


def test_cleanup_swipe_keep_is_tier2_not_3(w):
    a, _ = w.promoted(_at(w.rng, w.alex, 0.95), auto_matched=False)
    m, _ = w.promoted(_at(w.rng, w.alex, 0.35), auto_matched=True)
    assert w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % m, 'action': 'keep'}).json()['ok']
    assert trust(w, m) == 2
    assert w.preview()['anchor_refs'] == ['manual:%d' % a]            # still not an anchor


# ---------------------------------------------------------------- migration

def _legacy_rows(w):
    """Named rows as they looked before the trust column: trust 0, various marks."""
    ids = {}
    def row(key, **cols):
        fid, _ = w.promoted(_at(w.rng, w.alex, 0.5), auto_matched=True)
        sets = ', '.join('%s = ?' % c for c in cols)
        w.manual.conn.execute('UPDATE faces SET trust = 0, origin = NULL, %s WHERE id = ?' % sets if sets
                              else 'UPDATE faces SET trust = 0, origin = NULL WHERE id = ?', (*cols.values(), fid))
        ids[key] = fid
    row('plain')
    row('fav', favorite=2)
    row('profile', profile_at=5)
    row('reviewed', reviewed_ok=1)
    row('hand_box', bbox_source='manual')
    row('human', origin='human')
    row('auto', origin='auto')
    row('human_reviewed', origin='human', reviewed_ok=1)
    cs, _ = w.auto_face(_at(w.rng, w.alex, 0.5))
    ids['unnamed'] = w.manual.add_manual_face(cs, [0, 0, 5, 5], w.alex.tobytes(), 10, 10)
    w.manual.conn.commit()
    return ids


def test_backfill_migration(w):
    ids = _legacy_rows(w)
    m = w.manual
    # an older hand-picked cleanup anchor (no prev_trust) on a plain legacy row
    picked = ids['plain']
    m.conn.execute('INSERT INTO person_cleanup_anchors (identity, face_id, created_at) VALUES (?,?,1)', (NAME, picked))
    # a row that already has a trust must never be rewritten
    keep_id, _ = w.promoted(_at(w.rng, w.alex, 0.5), auto_matched=False, trust=2)
    m.conn.execute("DELETE FROM schema_flags WHERE name = 'faces_trust_v1'")
    m.conn.commit()
    m._migrate_face_trust(m.conn.cursor()); m.conn.commit()
    got = {k: (r['trust'], r['origin']) for k in ids for r in [m.get_face(ids[k])]}
    assert got['fav'] == (3, 'human') and got['profile'] == (3, 'human') and got['hand_box'] == (3, 'human')
    assert got['reviewed'] == (2, 'human') and got['human_reviewed'] == (2, 'human')
    assert got['human'] == (3, 'human')
    assert got['auto'] == (1, 'auto')
    assert got['plain'] == (3, 'human')                               # was a picked anchor -> promoted, remembered
    assert got['unnamed'] == (0, None)                                # unnamed rows untouched
    pr = m.conn.execute('SELECT prev_trust, prev_origin FROM person_cleanup_anchors WHERE face_id = ?', (picked,)).fetchone()
    assert (pr['prev_trust'], pr['prev_origin']) == (1, 'auto')
    assert trust(w, keep_id) == 2
    # the SQL mirrors person_cleanup.derive_trust
    legacy_origin = {'human': 'human', 'human_reviewed': 'human', 'auto': 'auto'}
    for k in ('fav', 'profile', 'reviewed', 'hand_box', 'human', 'auto', 'human_reviewed'):
        r = m.get_face(ids[k])
        assert r['trust'] == pc.derive_trust(legacy_origin.get(k), r['reviewed_ok'], r['favorite'],
                                             r['profile_at'], r['bbox_source'], NAME), k
    # idempotent: second run (flag present) changes nothing, even if we tamper
    snap = [tuple(r) for r in m.conn.execute('SELECT * FROM faces ORDER BY id')]
    m._migrate_face_trust(m.conn.cursor())
    assert snap == [tuple(r) for r in m.conn.execute('SELECT * FROM faces ORDER BY id')]
    m.conn.execute("UPDATE faces SET trust = 0 WHERE id = ?", (ids['auto'],))
    m.conn.commit()
    m._migrate_face_trust(m.conn.cursor())
    assert trust(w, ids['auto']) == 0                                 # flag respected: not run again


def test_migration_runs_at_db_open_and_cleans_legacy_as_auto():
    with tempfile.TemporaryDirectory() as tmp:
        w1 = World(tmp)
        ids = _legacy_rows(w1)
        w1.manual.conn.execute("DELETE FROM schema_flags WHERE name = 'faces_trust_v1'")
        w1.manual.conn.commit()
        from media_manager.manual_db import ManualDB
        path = w1.manual.db_path if hasattr(w1.manual, 'db_path') else None
        if path is None:
            pytest.skip('no db_path on ManualDB')
        m2 = ManualDB(path)
        assert m2.get_face(ids['plain'])['trust'] == 1 and m2.get_face(ids['plain'])['origin'] == 'auto'
        assert m2.get_face(ids['fav'])['trust'] == 3


# ---------------------------------------------------------------- matcher

def test_matcher_ignores_auto_when_trusted_exists_and_falls_back(w):
    m = w.manual
    # identity with only auto faces: they ARE the references (brand-new identity still matches)
    only_auto, _ = w.promoted(w.alex, auto_matched=True)
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME
    # a hand-set face on a different direction: from now on auto faces stop being references
    other = _unit(np.random.default_rng(9).standard_normal(512))
    hand, _ = w.promoted(other, auto_matched=False)
    assert m.find_matching_identity(w.alex.tobytes())[0] is None       # alex-like query no longer attracted
    assert m.find_matching_identity(other.tobytes())[0] == NAME
    # the cascade: an auto-matched face next to the trusted one does not attract further matches
    drift = _at(w.rng, other, 0.8)
    w.promoted(drift, auto_matched=True)
    far_from_hand = _at(w.rng, drift, 0.8)
    q = far_from_hand if float(far_from_hand.dot(other)) < 0.6 else None
    if q is not None:
        assert m.find_matching_identity(q.tobytes())[0] is None
    # tier 2 and 3 both count; weights rank (not threshold): 3 beats 2 on a near tie
    n_ver = m._face_ver
    sw, _ = w.promoted(w.alex, auto_matched=False, trust=2)
    assert m._face_ver > n_ver and m.find_matching_identity(w.alex.tobytes())[0] == NAME


def test_matcher_weights_rank_between_identities(w):
    m = w.manual
    x = _unit(w.rng.standard_normal(512))
    w.promoted(x, auto_matched=False, trust=2)                          # Alex: approved ref, cos 1.0 -> 0.7
    cs, src = w.auto_face(x)
    m.promote_auto_face(src, cs, [0, 0, 5, 5], x.tobytes(), 'Bob', None, None)      # Bob: hand-set, same cos
    name, score = m.find_matching_identity(x.tobytes())
    assert name == 'Bob' and score == pytest.approx(1.0, abs=1e-4)        # threshold uses the raw cosine


def test_matcher_escape_hatch_env(w, monkeypatch):
    m = w.manual
    w.promoted(_unit(np.random.default_rng(3).standard_normal(512)), auto_matched=False)   # trusted, unrelated
    w.promoted(w.alex, auto_matched=True)
    assert m.find_matching_identity(w.alex.tobytes())[0] is None
    monkeypatch.setenv('MEDIA_FACE_TRUST_MATCHING', '0')
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME        # old behaviour: every named face is a reference
    monkeypatch.setenv('MEDIA_FACE_TRUST_MATCHING', '1')
    assert m.find_matching_identity(w.alex.tobytes())[0] is None


def test_matcher_cache_rebuilds_on_writes(w):
    m = w.manual
    a, _ = w.promoted(w.alex, auto_matched=True)
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME
    other = _unit(np.random.default_rng(4).standard_normal(512))
    w.promoted(other, auto_matched=False)
    assert m.find_matching_identity(w.alex.tobytes())[0] is None        # cache saw the new trusted face
    m.assign_identity(a, NAME)                                          # the auto face is now hand-confirmed
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME


# ---------------------------------------------------------------- tier 3 never touched

def test_tier3_never_touched_by_cleanup_paths(w):
    rng = w.rng
    hand = [w.promoted(_at(rng, w.alex, c), auto_matched=False)[0] for c in (0.97, 0.96, 0.0, -0.1)]
    auto_far, _ = w.promoted(_at(rng, w.alex, 0.0), auto_matched=True)
    snap = {h: tuple(w.manual.get_face(h)) for h in hand}
    w.preview(include_approved=1)
    w.client.post(BASE + '/uncertain', params={'include_approved': 1}, json={'exclude': []})
    for h in hand:
        for a in ('keep', 'remove'):
            w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % h, 'action': a, 'include_approved': True})
    w.client.post(BASE + '/apply', json={'confirm': True})
    w.client.post(BASE + '/undo')
    w.client.put(BASE + '/anchors', json={'refs': []})
    assert {h: tuple(w.manual.get_face(h)) for h in hand} == snap
