"""Location 'place' model, stage 1: people-aware correction from stored data (person area,
same-identity damping, GPS/time/set co-signals). Fake vectors, no model, no GPU."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from media_manager import location_place as lp
from media_manager.web import create_app


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'REBUILD_MIN_S', 0.0)
    os.makedirs(tmp_path / '.media')
    app = create_app(str(tmp_path))
    db, _e, manual = app.state.dbs
    ids = {}

    def add(name, e=None, dims=(1000, 1000)):
        """e = dict of coords for BOTH spaces (same direction); None = unembedded."""
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = db.upsert_file_path(f'{name}.jpg', cs, size=10)
        if e is not None:
            db.insert_siglip_embedding(fid, _vec(10, **e).tobytes(), 'siglip-t')
            db.insert_dino_embedding(fid, _vec(8, **e).tobytes(), 'dino-t')
        if dims:
            db.conn.execute('UPDATE files SET width = ?, height = ? WHERE id = ?', (*dims, fid))
        ids[name] = (fid, cs)
        return fid

    same = {'e0': 1}
    for n in ('m1', 'm2', 'm3'):
        add(n, same)
    for n in ('far1', 'far2', 'far3', 'far4'):
        add(n, {f'e{4 + int(n[-1])}': 1} if int(n[-1]) < 4 else {'e7': 1, 'e1': 1})
    add('clean', same)
    add('crowded', same)
    add('unscanned', same)
    add('plain', {'e0': 1, 'e3': 1})          # cosine .71: well below the near-duplicate ramp
    add('annmid', {'e0': 1, 'e3': 1})
    add('anndup', same)
    add('gnone', {'e0': 1, 'e2': .05})        # no GPS
    add('gnear', {'e0': 1, 'e2': .05})
    add('gfar', {'e0': 1, 'e2': .05})
    db.conn.commit()
    loc = manual.create_location('L')
    for n in ('m1', 'm2', 'm3'):
        manual.add_file_location(ids[n][1], loc)
    db.conn.commit()
    return app, db, manual, ids, loc, TestClient(app), monkeypatch


def _get(client, loc, **q):
    qs = '&'.join(f'{k}={v}' for k, v in q.items())
    r = client.post(f'/api/locations/{loc}/similar-files?limit=100&{qs}', json={'exclude': []})
    assert r.status_code == 200, r.text
    return r.json()


def _by(d, ids):
    rev = {v[0]: k for k, v in ids.items()}
    return {rev[c['id']]: c for c in d['cards']}, [rev[c['id']] for c in d['cards']]


def test_union_fraction_and_regions():
    assert lp.union_fraction([[0, 0, 500, 1000]], 1000, 1000) == pytest.approx(0.5, abs=0.01)
    assert lp.union_fraction([[0, 0, 500, 1000], [0, 0, 250, 1000]], 1000, 1000) == pytest.approx(0.5, abs=0.01)
    assert lp.union_fraction([], 1000, 1000) == 0.0
    assert lp.union_fraction([[10, 10, 12, 12]], 1000, 1000) > 0          # tiny box still counts
    # a face expands to a body-ish box; a person box holding the face centre replaces the expansion
    reg, _ = lp.person_regions([[400, 400, 500, 500]], [], 1000, 1000)
    assert len(reg) == 1 and reg[0][3] > 500 + 100 * 3 and reg[0][0] < 400
    reg2, _ = lp.person_regions([[400, 400, 500, 500]], [[300, 300, 700, 900]], 1000, 1000)
    assert reg2 == [[300.0, 300.0, 700.0, 900.0]]
    # stored dims un-rotated (landscape), boxes upright (portrait): dims are swapped
    _r, wh = lp.person_regions([], [[0, 0, 100, 900]], 1000, 500)
    assert wh == (500, 1000)
    assert lp.person_penalty_curve(0.0) == 0 and lp.person_penalty_curve(0.9) == pytest.approx(1.0)
    assert 0 < lp.person_penalty_curve(0.25) < 1


def test_place_is_default_with_why_and_legacy_models(env):
    _app, _db, _m, ids, loc, client, _mp = env
    d = _get(client, loc)
    assert d['model'] == 'place' and d['threshold_applied'] is False
    assert d['cards'] and all('why' in c for c in d['cards'])
    w = d['cards'][0]['why']
    assert {'place', 'cos', 'scanned', 'person_penalty', 'people_penalty', 'geo', 'co', 'total'} <= set(w)
    assert 'person_scan_coverage' in d and 'people' in d and d['geo']['active'] is False
    for model in ('combined', 'siglip', 'dino'):
        r = _get(client, loc, model=model)
        assert r['model'] == model and r['cards'] and not any('why' in c for c in r['cards'])
    assert client.post(f'/api/locations/{loc}/similar-files?model=nope', json={'exclude': []}).status_code == 400
    names = _by(_get(client, loc), ids)[1]
    assert not {'m1', 'm2', 'm3'} & set(names)


def test_person_area_weight_and_unscanned_get_no_bonus(env):
    _app, db, _m, ids, loc, client, _mp = env
    db.insert_detections(ids['clean'][0], [], 'yolo')                                  # sentinel: scanned, no person
    db.insert_detections(ids['crowded'][0], [('person', 0.9, 100, 50, 900, 1000)], 'yolo')   # ~72% of the frame
    db.insert_detections(ids['m1'][0], [], 'yolo')
    d = _get(client, loc)
    cards, order = _by(d, ids)
    assert cards['clean']['why']['scanned'] and cards['clean']['why']['person_penalty'] == pytest.approx(lp.CLEAN_BONUS)
    assert cards['unscanned']['why']['scanned'] is False and cards['unscanned']['why']['person_penalty'] == 0
    assert cards['crowded']['why']['person_area'] > 0.6
    assert cards['crowded']['why']['person_penalty'] < -1.0
    assert order.index('clean') < order.index('unscanned') < order.index('crowded')
    cov = d['person_scan_coverage']
    assert cov['pool_total'] >= 10 and 0 < cov['pool_scanned'] < cov['pool_total']
    assert cov['unscanned_pct'] > 50 and cov['library_scanned'] == 3
    # raw similarity is untouched: only the order moved
    assert cards['clean']['score'] == cards['crowded']['score']


def test_face_boxes_count_and_cache_invalidation(env):
    _app, db, _m, ids, loc, client, _mp = env
    fid = ids['crowded'][0]
    big_face = {'bbox': [300, 100, 700, 500], 'embedding': np.zeros(512, np.float32), 'det_score': 0.9}
    db.insert_faces(fid, [big_face], 'arc')
    c1 = _by(_get(client, loc), ids)[0]['crowded']['why']
    assert c1['scanned'] and c1['person_area'] > 0.3 and c1['person_penalty'] < -0.5
    low = dict(big_face, det_score=0.2)                      # below FACE_MIN_SCORE: scanned, no area
    db.insert_faces(fid, [low], 'arc')
    c2 = _by(_get(client, loc), ids)[0]['crowded']['why']
    assert c2['scanned'] and c2['person_area'] == 0
    builds = lp.store_for(db).builds
    _get(client, loc)
    assert lp.store_for(db).builds == builds                 # cached while versions are unchanged
    db.insert_detections(fid, [('crowd', 0.8, 0, 0, 1000, 1000)], 'yolo')
    c3 = _by(_get(client, loc), ids)[0]['crowded']['why']
    assert c3['person_area'] > 0.9 and lp.store_for(db).builds == builds + 1


def test_same_people_damping_unless_near_duplicate(env):
    _app, _db, manual, ids, loc, client, _mp = env
    for n in ('m1', 'm2'):
        manual.assign_identity_to_photo(ids[n][1], 'Ann')
    for n in ('annmid', 'anndup'):
        manual.assign_identity_to_photo(ids[n][1], 'Ann')
    d = _get(client, loc)
    cards, order = _by(d, ids)
    assert 'Ann' in d['people']
    assert cards['annmid']['why']['people_penalty'] < -0.5 and cards['annmid']['why']['people'] == ['Ann']
    assert order.index('plain') < order.index('annmid')           # same place-ish score, but Ann is damped
    assert cards['anndup']['why']['people_penalty'] == pytest.approx(0, abs=0.01)   # near-duplicate: exempt
    assert 'people' in cards['anndup']['why']
    # a person who is in only one member photo (and < half of them) is not a location "identity"
    assert lp.identity_profile(['a', 'b', 'c', 'd'], {'a': {'X'}}) == {}
    assert lp.identity_profile(['a', 'b'], {'a': {'X'}}) == {'X': 0.5}


def test_geo_boost_and_veto_only_with_gps(env):
    _app, db, manual, ids, loc, client, _mp = env
    db.update_file_metadata(ids['gnear'][0], None, 48.8600, 2.3500)
    db.update_file_metadata(ids['gfar'][0], None, 40.7, -74.0)
    db.conn.commit()
    d = _get(client, loc)
    assert d['geo']['active'] is False                         # no point, no geotagged members: inert
    assert _by(d, ids)[0]['gnear']['why']['geo'] == 0 and _by(d, ids)[0]['gfar']['why']['geo'] == 0
    manual.set_location_coords(loc, 48.8566, 2.3522)
    d = _get(client, loc)
    cards, order = _by(d, ids)
    assert d['geo']['active'] and d['geo']['source'] == 'location'
    assert cards['gnear']['why']['geo'] > 0.5 and cards['gnear']['why']['geo_km'] < 1
    assert cards['gfar']['why']['geo'] <= -lp.GEO_VETO_PENALTY + 0.01
    assert cards['gnone']['why']['geo'] == 0 and 'geo_km' not in cards['gnone']['why']   # no GPS: untouched
    assert order.index('gnear') < order.index('gnone') < order.index('gfar')


def test_geo_from_geotagged_members_and_loose_cluster_no_veto(env):
    _app, db, manual, ids, loc, client, _mp = env
    for n, (la, lo) in zip(('m1', 'm2', 'm3'), ((48.85, 2.35), (48.86, 2.36), (48.84, 2.34))):
        db.update_file_metadata(ids[n][0], None, la, lo)
    db.update_file_metadata(ids['gfar'][0], None, 40.7, -74.0)
    db.conn.commit()
    d = _get(client, loc)
    assert d['geo']['active'] and d['geo']['source'] == 'members' and d['geo']['veto'] is True
    assert _by(d, ids)[0]['gfar']['why']['geo'] < -2
    # members spread over two continents: still boosts near them, but never vetoes
    db.update_file_metadata(ids['m3'][0], None, 35.0, 139.0)
    db.conn.commit()
    d = _get(client, loc)
    assert d['geo']['veto'] is False and _by(d, ids)[0]['gfar']['why']['geo'] >= 0


def test_time_set_and_directory_cosignals(env):
    _app, db, manual, ids, loc, client, _mp = env
    t0 = 1_700_000_000
    db.update_file_metadata(ids['m1'][0], t0, None, None)
    db.update_file_metadata(ids['clean'][0], t0 + 3600, None, None)
    db.update_file_metadata(ids['unscanned'][0], t0 + 3 * 86400, None, None)
    sid = manual.create_set('Shoot')
    for n in ('m1', 'm2', 'm3', 'plain'):
        manual.assign_file_to_set(ids[n][1], sid)
    db.conn.commit()
    cards, _o = _by(_get(client, loc), ids)
    assert cards['clean']['why']['co'] == pytest.approx(lp.TIME_BOOST, abs=0.01)
    assert cards['unscanned']['why']['co'] == 0
    assert cards['plain']['why']['co'] == pytest.approx(lp.SET_BOOST, abs=0.01)


def test_negatives_pagination_and_frame_rule_preserved(env):
    _app, db, manual, ids, loc, client, _mp = env
    manual.exclude_file_from_location(ids['clean'][1], loc)
    db.conn.commit()
    base = _by(_get(client, loc), ids)[1]
    assert 'clean' not in base
    neg = _by(_get(client, loc, honor_negatives='true'), ids)[1]
    assert 'clean' not in neg
    r = client.post(f'/api/locations/{loc}/similar-files?limit=2&offset=1', json={'exclude': []})
    assert _by(r.json(), ids)[1] == base[1:3]
    vid = db.upsert_file_path('clip.mp4', 'v' * 40, size=1)
    still = db.upsert_file_path('captured_frames/clip_1.jpg', 's' * 40, size=1)
    db.insert_siglip_embedding(still, _vec(10, e0=1).tobytes(), 's')
    db.insert_dino_embedding(still, _vec(8, e0=1).tobytes(), 'd')
    db.set_file_hidden(still)
    manual.add_frame_capture('s' * 40, 'v' * 40, 1000)
    db.conn.commit()
    got = _get(client, loc)
    ids_got = [c['id'] for c in got['cards']]
    assert vid in ids_got and still not in ids_got and ids_got.count(vid) == 1
    assert next(c for c in got['cards'] if c['id'] == vid).get('why')       # the video carries the still's why


def test_location_page_defaults_to_place(env):
    _app, _db, _m, _ids, loc, client, _mp = env
    html = client.get(f'/locations/{loc}').text
    assert 'value="place"' in html and 'value="combined"' in html and 'whyBadge' in html
