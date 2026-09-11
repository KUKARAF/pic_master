"""/locations/{id} suggestion stack: ranked by visual similarity, never by geography.

A location's suggestions are "photos that look like the ones already here" — the same
CLIP-centroid ranking the set stack uses. A location's own GPS coordinates must play
no part: two photos taken in the same room rank together because they look alike, and
a photo with no coordinates at all is just as eligible.

The `s` key ("this photo's SET belongs here") is the reason the candidate pool also
subtracts photos of any set already linked to this location: one keystroke settles a
whole shoot without writing a file_locations row per photo, and unlinking the set
puts them all back.

Run standalone: python tests/test_location_suggestions.py
"""
import os
import tempfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from media_manager.web import create_app

D = 512
WITHIN_COS = 0.95
# For x = unit(d + s*n), n ~ N(0, I_D): E[cos(x, d)] ~ 1/sqrt(1 + s^2 D). Deriving s
# from the cosine keeps the fixture in the regime the threshold was tuned for; a flat
# per-component sigma is noise of norm s*sqrt(D) against a unit signal, i.e. static.
SIGMA = float(np.sqrt((1.0 / WITHIN_COS ** 2 - 1.0) / D))


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(app, tmp):
    db, _errors, manual = app.state.dbs
    rng = np.random.default_rng(3)
    indoor, outdoor = _unit(rng.standard_normal(D)), _unit(rng.standard_normal(D))

    ids = {}
    plan = [('member_1', indoor), ('member_2', indoor),
            ('shoot_a', indoor), ('shoot_b', indoor),
            ('loose_a', indoor), ('loose_b', indoor),
            ('other_1', outdoor), ('other_2', outdoor)]
    for name, theme in plan:
        Image.new('RGB', (32, 32), (90, 120, 160)).save(os.path.join(tmp, name + '.jpg'))
        cs = (name * 8)[:40]
        fid = db.upsert_file_path(name + '.jpg', cs, size=100)
        db.insert_embedding(fid, _unit(theme + SIGMA * rng.standard_normal(D)).tobytes(), 'clip-test')
        ids[name] = (fid, cs)
    db.conn.commit()

    loc_id = manual.create_location('Studio Loft')
    for n in ('member_1', 'member_2'):
        manual.add_file_location(ids[n][1], loc_id)
    set_id = manual.create_set('Loft Shoot', 'TestStudio')
    for n in ('shoot_a', 'shoot_b'):
        manual.assign_file_to_set(ids[n][1], set_id)
    return ids, loc_id, set_id


def run():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    ids, loc_id, set_id = _seed(app, tmp)
    _db, _errors, manual = app.state.dbs
    client = TestClient(app)

    def suggest():
        r = client.post('/api/locations/%d/similar-files?limit=20' % loc_id, json={'exclude': []})
        assert r.status_code == 200, (r.status_code, r.text[:200])
        return sorted(c['filename'] for c in r.json()['cards'])

    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        ok = ok and cond
        print(('  PASS  ' if cond else '  FAIL  ') + name + (('   [%s]' % (extra,)) if extra else ''))

    first = suggest()
    check('ranks the look-alikes in',
          set(first) == {'shoot_a.jpg', 'shoot_b.jpg', 'loose_a.jpg', 'loose_b.jpg'}, str(first))
    check('leaves visually unrelated photos out',
          not any(n.startswith('other') for n in first), str(first))
    check('never re-offers a photo already at the location',
          not any(n.startswith('member') for n in first), str(first))

    # --- the `s` key -------------------------------------------------------------
    r = client.post('/api/locations/%d/adopt-set' % loc_id, json={'file_id': ids['shoot_a'][0]})
    check('adopt-set reports the set it placed',
          r.status_code == 200 and [s['name'] for s in r.json()['sets']] == ['Loft Shoot'],
          r.text[:120])
    check('the set is linked to the location',
          loc_id in [l['id'] for l in manual.get_locations_for_set(set_id)])
    check('and the swiped photo itself is at the location',
          ids['shoot_a'][1] in manual.get_checksums_for_location(loc_id))
    after = suggest()
    check('one keystroke settles the WHOLE set, not just the swiped photo',
          set(after) == {'loose_a.jpg', 'loose_b.jpg'}, str(after))

    client.delete('/api/locations/%d/adopt-set?file_id=%d' % (loc_id, ids['shoot_a'][0]))
    check('undoing the adopt puts the whole set back', set(suggest()) == set(first), str(suggest()))
    check('and unlinks the set', manual.get_locations_for_set(set_id) == [])

    # --- reject is remembered ------------------------------------------------------
    client.post('/api/files/%d/locations/%d/exclude' % (ids['loose_a'][0], loc_id))
    check('a rejected photo stops being suggested', 'loose_a.jpg' not in suggest(), str(suggest()))
    client.delete('/api/files/%d/locations/%d/exclude' % (ids['loose_a'][0], loc_id))
    check('and comes back when the reject is undone', 'loose_a.jpg' in suggest(), str(suggest()))

    # --- guards ---------------------------------------------------------------------
    r = client.post('/api/locations/%d/adopt-set' % loc_id, json={'file_id': ids['loose_a'][0]})
    check('adopt-set on a photo with no set is a 400, not a silent no-op',
          r.status_code == 400, '%s %s' % (r.status_code, r.text[:80]))
    empty_loc = manual.create_location('Never Used')
    r = client.post('/api/locations/%d/similar-files' % empty_loc, json={'exclude': []})
    check('a location with no photos has nothing to compare against, and says so quietly',
          r.status_code == 200 and r.json()['cards'] == [], r.text[:120])
    r = client.post('/api/locations/999999/similar-files', json={'exclude': []})
    check('an unknown location 404s', r.status_code == 404, str(r.status_code))

    print('\noverall:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())
