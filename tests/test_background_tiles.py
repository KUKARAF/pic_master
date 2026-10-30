"""Location suggestions by BACKGROUND tiles (background_tiles.py + the location API).

Whole-image CLIP ranks "photos taken here" by whoever is in them. With background=true
the ranking only uses tiles of the tile index that person boxes cover at most 25% of
(never a person photo's whole-image tile), so a different photo of the same PLACE beats
a photo of the same PERSON somewhere else. Falls back to whole images when the
location's photos have no tiles.

Run: python tests/test_background_tiles.py   (or python -m pytest tests/test_background_tiles.py)
"""
import os
import tempfile

import numpy as np
from fastapi.testclient import TestClient

from media_manager import background_tiles as bt
from media_manager.web import create_app

D = 8


def _v(*pairs):
    v = np.zeros(D, np.float32)
    for i, w in pairs:
        v[i] = w
    return v / np.linalg.norm(v)


P, A, B = _v((0, 1)), _v((1, 1)), _v((2, 1))       # person look, place A, place B
TILES = [(0, 0, 300, 200), (0, 0, 150, 200), (150, 0, 300, 200), (150, 0, 300, 100)]


def check_unit():
    cov = bt.person_coverage([(0, 0, 10, 10), (20, 20, 30, 30), (0, 0, 20, 10)], [(0, 0, 10, 10)])
    assert cov[0] == 1.0 and cov[1] == 0.0 and abs(cov[2] - 0.5) < 0.07, cov
    fids = np.array([1, 1, 1, 1, 2, 2])
    tidx = np.array([0, 1, 2, 3, 0, 1])
    boxes = np.array(TILES + TILES[:2], np.float32)
    mask = bt.background_mask(fids, tidx, boxes, {1: [[0, 0, 150, 200]]})
    assert mask.tolist() == [False, False, True, True, True, True], mask   # file 2: no people
    ranked = bt.rank_files([(np.array([7, 7, 8]), np.stack([A, B, B]))], np.stack([A]))
    assert [f for f, _s in ranked] == [7, 8] and abs(ranked[0][1] - 0.5) < 1e-6, ranked  # top-2 mean
    print('OK  coverage, background mask (whole tile dropped when people), top-2 ranking')


def check_api():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    app = create_app(root)
    db, _errors, manual = app.state.dbs
    client = TestClient(app)

    def add(name, whole, tiles, person=None):
        db.upsert_file_path(name, 'cs-' + name, size=1, modified_time=1)
        fid = db.find_file_id_by_checksum('cs-' + name)
        db.insert_embedding(fid, whole.tobytes(), 'm')
        if tiles is not None:
            db.insert_tile_embeddings(fid, [{'bbox': list(b), 'embedding': e} for b, e in zip(TILES, tiles)], 'm')
        db.insert_detections(fid, [('person', 0.9, *person)] if person else [], 'm')
        return fid

    mix = _v((0, 0.9), (1, 0.3))
    member = add('m.jpg', P, [mix, P, A, A], person=(0, 0, 150, 200))   # her, at café A
    same_person = add('x.jpg', P, [P, P, B, B], person=(0, 0, 150, 200))  # her, elsewhere
    same_place = add('y.jpg', A, [A, A, A, A])                          # café A, nobody
    loc = manual.create_location('Café A')
    manual.add_file_location('cs-m.jpg', loc)

    url = '/api/locations/%d/similar-files?threshold=0&avoid_existing=false' % loc
    whole = client.post(url + '&background=false', json={'exclude': []}).json()
    assert whole['mode'] == 'whole' and [c['id'] for c in whole['cards']][0] == same_person, whole
    bg = client.post(url, json={'exclude': []}).json()
    ids = [c['id'] for c in bg['cards']]
    assert bg['mode'] == 'background' and ids[0] == same_place, bg
    assert member not in ids
    # Cached ranking pages consistently (refill with the first card excluded).
    nxt = client.post(url + '&limit=1', json={'exclude': [str(same_place)]}).json()
    assert [c['id'] for c in nxt['cards']] == [same_person], nxt

    st = client.get('/api/locations/%d/background-status' % loc).json()
    assert st['members'] == 1 and st['tiled_members'] == 1 and st['tiled_files'] == 3, st

    # A location whose photos have no tiles falls back to whole-image matching.
    lonely = add('z.jpg', A, None)
    loc2 = manual.create_location('Elsewhere')
    manual.add_file_location('cs-z.jpg', loc2)
    r = client.post('/api/locations/%d/similar-files?threshold=0&avoid_existing=false' % loc2,
                    json={'exclude': []}).json()
    assert r['mode'] == 'whole' and r['cards'][0]['id'] == same_place, r
    assert lonely not in [c['id'] for c in r['cards']]
    print('OK  API: background ranks same place over same person; whole-image fallback; status')


def run():
    check_unit()
    check_api()
    print('ALL ASSERTIONS PASSED')


def test_background_tiles():
    run()


if __name__ == '__main__':
    run()
