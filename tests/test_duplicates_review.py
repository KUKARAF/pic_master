"""/duplicates (exact copies, our own xxh64) and /near-duplicates (czkawka groups) review.

Exact: one file tracked at several paths; each copy listed with resolution/length/
format/date/size/path, the oldest copy outside uploads/ pre-ticked; trashing a copy
untracks that path into the path-level Trash (file stays on disk); "keep all" hides the
group until another copy appears; at least one copy always stays.
Similar: members carry the same facts, the biggest resolution/length/size is pre-ticked;
trashing moves content to the Trash (optionally carrying tags to the kept copy) and the
kept ones are marked not-duplicates; nothing happens without a request per group.

Run: python tests/test_duplicates_review.py   (or python -m pytest tests/test_duplicates_review.py)
"""
import os
import tempfile

import cv2
import numpy as np
import xxhash
from fastapi.testclient import TestClient
from PIL import Image

from media_manager.web import create_app


def _png(root, rel, size, color=(200, 30, 30)):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', size, color).save(path)
    return path


def _track(db, root, rel):
    path = os.path.join(root, rel)
    cs = xxhash.xxh64(open(path, 'rb').read()).hexdigest()
    st = os.stat(path)
    db.upsert_file_path(rel, cs, size=st.st_size, modified_time=st.st_mtime)
    db.conn.commit()
    return db.find_file_id_by_checksum(cs), cs


def check_exact(client, db, manual, root):
    src = _png(root, 'uploads/ab/pic.png', (120, 80))
    data = open(src, 'rb').read()
    for rel, mtime in (('trips/pic.png', 1_600_000_000), ('backup/old/pic.png', 1_500_000_000)):
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        open(os.path.join(root, rel), 'wb').write(data)
        os.utime(os.path.join(root, rel), (mtime, mtime))
    fid, cs = _track(db, root, 'uploads/ab/pic.png')
    _track(db, root, 'trips/pic.png')
    _track(db, root, 'backup/old/pic.png')

    d = client.get('/api/duplicates/exact').json()
    assert d['total'] == 1 and len(d['groups']) == 1, d
    g = d['groups'][0]
    assert g['id'] == fid and (g['width'], g['height'], g['format']) == (120, 80, 'PNG'), g
    assert sorted(c['path'] for c in g['copies']) == ['backup/old/pic.png', 'trips/pic.png', 'uploads/ab/pic.png']
    assert g['keep'] == ['backup/old/pic.png'], g['keep']        # oldest, not under uploads/

    url = '/api/duplicates/exact/resolve'
    assert client.post(url, json={'file_id': fid, 'trash': [c['path'] for c in g['copies']]}).status_code == 400
    assert client.post(url, json={'file_id': fid, 'trash': ['nope.png']}).status_code == 400
    r = client.post(url, json={'file_id': fid, 'trash': ['uploads/ab/pic.png']})
    assert r.status_code == 200, r.text
    assert sorted(p for p, _ in db.get_paths_for_file(fid)) == ['backup/old/pic.png', 'trips/pic.png']
    assert os.path.isfile(os.path.join(root, 'uploads/ab/pic.png'))               # not deleted
    t = manual.get_trashed_path('uploads/ab/pic.png')
    assert t and t['checksum'] == cs and t['kept_path'] == 'backup/old/pic.png', t
    assert not manual.is_trashed(cs)                                              # content stays

    # Keep all remaining copies -> group hidden; a new copy brings it back.
    assert client.post(url, json={'file_id': fid, 'trash': []}).status_code == 200
    assert client.get('/api/duplicates/exact').json()['groups'] == []
    open(os.path.join(root, 'trips/pic2.png'), 'wb').write(data)
    _track(db, root, 'trips/pic2.png')
    assert len(client.get('/api/duplicates/exact').json()['groups']) == 1
    print('OK  exact: facts per copy, oldest non-upload kept, path-level trash, keep-all, never all')


def check_similar(client, db, manual, root):
    _png(root, 'sim/big.png', (200, 150))
    _png(root, 'sim/small.png', (100, 75), color=(201, 30, 30))
    big, big_cs = _track(db, root, 'sim/big.png')
    small, small_cs = _track(db, root, 'sim/small.png')
    manual.add_tag(small_cs, 'beach')

    vid = os.path.join(root, 'sim/clip.avi')
    w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    for _ in range(50):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    _png(root, 'sim/other.png', (64, 48), color=(0, 0, 0))
    v_id, _ = _track(db, root, 'sim/clip.avi')
    o_id, _ = _track(db, root, 'sim/other.png')

    g1 = db.insert_dup_group('review', 'review', None, 'test', [small, big])
    g2 = db.insert_dup_group('review', 'review', None, 'test', [v_id, o_id])

    groups = {g['group_id']: g for g in client.get('/api/duplicates/similar').json()['groups']}
    assert groups[g1]['keep'] == [big], groups[g1]
    m = {x['id']: x for x in groups[g1]['members']}
    assert (m[big]['width'], m[big]['height']) == (200, 150) and m[small]['tag_count'] == 1, m
    v = {x['id']: x for x in groups[g2]['members']}[v_id]
    assert v['is_video'] and v['format'] == 'AVI' and abs(v['duration'] - 2.0) < 0.2, v

    # A damaged copy is never pre-ticked over a healthy one, however big it is.
    db.mark_broken(big)
    g = {g['group_id']: g for g in client.get('/api/duplicates/similar').json()['groups']}[g1]
    assert g['keep'] == [small], g['keep']
    db.conn.execute('UPDATE files SET broken = NULL WHERE id = ?', (big,))
    db.conn.commit()

    url = '/api/duplicates/similar/resolve'
    assert client.post(url, json={'group_id': g1, 'trash': [big, small]}).status_code == 400
    assert client.post(url, json={'group_id': g1, 'trash': [999999]}).status_code == 400
    r = client.post(url, json={'group_id': g1, 'trash': [small], 'carry_labels': True})
    assert r.status_code == 200, r.text
    assert manual.is_trashed(small_cs) and not manual.is_trashed(big_cs)
    assert 'beach' in manual.list_tags_for_checksums([big_cs]).get(big_cs, []), 'tag carried over'

    # Keep all -> not-duplicates, nothing trashed, group gone.
    assert client.post(url, json={'group_id': g2, 'trash': []}).status_code == 200
    assert client.get('/api/duplicates/similar').json()['groups'] == []
    assert len(manual.get_not_duplicate_pairs()) == 1
    print('OK  similar: facts incl. video length, biggest kept, trash + carry labels, keep-all')


def run():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    app = create_app(root)
    db, _errors, manual = app.state.dbs
    client = TestClient(app)
    check_exact(client, db, manual, root)
    check_similar(client, db, manual, root)
    for page in ('/duplicates', '/near-duplicates'):
        r = client.get(page)
        assert r.status_code == 200 and 'dup-review.js' in r.text, page
    print('OK  /duplicates and /near-duplicates render')
    print('ALL ASSERTIONS PASSED')


def test_duplicates_review():
    run()


if __name__ == '__main__':
    run()
