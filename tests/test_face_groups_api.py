"""Face group review API (/api/face-groups/*): compute on a tiny seeded DB, next /
approve / reject flows, rejection persistence, trust-2 confirmation, live filtering."""
import os
import tempfile
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from media_manager import compute, face_clusters
from media_manager.web import create_app

D = 64


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def cpu_device(monkeypatch):
    """Test hook: there is no XPU/CUDA in CI, so route the (GPU-only) grouping to the
    torch-on-CPU device through its single choke point. Production never does this."""
    monkeypatch.setattr(face_clusters, 'resolve_device', lambda device=None: device or 'cpu')


@pytest.fixture
def env(cpu_device):
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    app = create_app(root)
    app.state.bulk_queue_autostart = False
    client = TestClient(app)
    db, _errors, manual = app.state.dbs
    rng = np.random.default_rng(4)
    centers = [_unit(rng.standard_normal(D)) for _ in range(3)]
    sizes = [6, 4, 2]            # third is below min_size 3
    faces = {}                   # cluster -> [face ids]
    n = 0
    for k, (c, sz) in enumerate(zip(centers, sizes)):
        for _ in range(sz):
            n += 1
            fid = db.upsert_file_path('p%d.jpg' % n, ('%08x' % n), size=10)
            db.conn.commit()
            emb = _unit(c + 0.05 * rng.standard_normal(D))
            db.add_manual_face(fid, [1, 1, 60, 60], emb.tobytes(), 0.9)
            row = db.conn.execute('SELECT id FROM faces WHERE file_id = ?', (fid,)).fetchone()
            faces.setdefault(k, []).append(row[0])
    db.conn.commit()
    yield app, client, faces, (db, manual)
    app.state.bulk_queue.stop()


def _compute(client, **params):
    q = '&'.join('%s=%s' % kv for kv in params.items())
    assert client.post('/api/face-groups/compute?' + q).json()['started'] is True
    for _ in range(400):
        st = client.get('/api/face-groups/status').json()
        if not st['running']:
            return st
        time.sleep(0.05)
    raise AssertionError('job never finished')


def test_compute_status_and_noop(env):
    _app, client, faces, _ = env
    assert client.get('/api/face-groups/status').json()['computed_at'] is None
    st = _compute(client, threshold=0.55, min_size=3)
    assert st['error'] is None
    assert st['groups'] == 2 and st['faces_in_groups'] == 10
    assert st['threshold'] == 0.55 and st['computed_at'] and st['stale'] is False
    # same threshold cached -> no-op unless forced
    assert client.post('/api/face-groups/compute?threshold=0.55&min_size=3').json()['started'] is False
    assert client.post('/api/face-groups/compute?threshold=0.55&min_size=3&force=1').json()['started'] is True
    for _ in range(200):
        if not client.get('/api/face-groups/status').json()['running']:
            break
        time.sleep(0.05)


def test_threshold_clamped(env):
    _app, client, _f, _ = env
    st = _compute(client, threshold=0.01, min_size=3)
    assert st['threshold'] == 0.3


def test_next_biggest_first_and_skip(env):
    _app, client, faces, _ = env
    _compute(client)
    r = client.get('/api/face-groups/next?threshold=0.55').json()
    g = r['group']
    assert g['size'] == 6 and not g['truncated'] and g['suggested'] is None
    assert {f['face_id'] for f in g['faces']} == set(faces[0])
    assert all(f['ref'] == 'auto:%d' % f['face_id'] and f['file_id'] for f in g['faces'])
    scores = [f['score'] for f in g['faces']]
    assert scores == sorted(scores, reverse=True)
    assert r['remaining_groups'] == 2 and r['remaining_faces'] == 10 and r['computing'] is False
    second = client.get('/api/face-groups/next?threshold=0.55&skip=1').json()['group']
    assert second['size'] == 4
    assert client.get('/api/face-groups/next?threshold=0.55&skip=2').json()['group'] is None
    assert client.get('/api/face-groups/next?threshold=0.9').json()['group'] is None  # nothing cached


def test_approve_confirms_trust2_and_removes_group(env):
    app, client, faces, (db, manual) = env
    _compute(client)
    g = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    keep = faces[0][:-1]             # user removed one face
    r = client.post('/api/face-groups/%d/approve' % g['id'], json={'name': 'Alice', 'face_ids': keep})
    assert r.status_code == 200, r.text
    assert r.json() == {'approved': 5, 'skipped': 0}
    rows = manual.conn.execute(
        "SELECT trust, origin, source_face_id FROM faces WHERE identity = 'Alice'").fetchall()
    assert len(rows) == 5 and {r_[0] for r_ in rows} == {2} and {r_[1] for r_ in rows} == {'human'}
    assert {r_[2] for r_ in rows} == set(keep)
    handled = {r_[0] for r_ in db.conn.execute('SELECT id FROM faces WHERE handled = 1')}
    assert set(keep) <= handled
    nxt = client.get('/api/face-groups/next?threshold=0.55').json()
    assert nxt['group']['size'] == 4                      # the removed face is not re-offered
    assert nxt['remaining_groups'] == 1
    # the removed face stays unnamed in the pool
    assert faces[0][-1] not in handled
    # the stale flag does not flip on our own writes
    assert client.get('/api/face-groups/status').json()['stale'] is False
    # approved id again / foreign id -> skipped, and the group is gone
    again = client.post('/api/face-groups/%d/approve' % g['id'], json={'name': 'Alice', 'face_ids': keep})
    assert again.status_code == 404


def test_approve_skips_foreign_and_named_faces(env):
    _app, client, faces, (_db, manual) = env
    _compute(client)
    g = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    # name one face elsewhere (the single-face path) before the group is approved
    assert client.post('/api/faces/auto:%d/identity' % faces[0][0], json={'name': 'Bob'}).status_code == 200
    r = client.post('/api/face-groups/%d/approve' % g['id'],
                    json={'name': 'Alice', 'face_ids': faces[0] + faces[1][:1]}).json()
    assert r == {'approved': 5, 'skipped': 2}
    assert manual.conn.execute("SELECT COUNT(*) FROM faces WHERE identity = 'Bob'").fetchone()[0] == 1


def test_approve_validation(env):
    _app, client, faces, _ = env
    _compute(client)
    g = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    url = '/api/face-groups/%d/approve' % g['id']
    assert client.post(url, json={'name': '  ', 'face_ids': faces[0]}).status_code == 400
    assert client.post(url, json={'name': '__indexed__', 'face_ids': faces[0]}).status_code == 400
    assert client.post(url, json={'name': 'x' * 300, 'face_ids': faces[0]}).status_code == 400
    assert client.post(url, json={'name': 'A', 'face_ids': [-1]}).status_code == 400
    assert client.post(url, json={'name': 'A', 'face_ids': 'nope'}).status_code == 422
    assert client.post('/api/face-groups/99999/approve', json={'name': 'A', 'face_ids': [1]}).status_code == 404
    assert client.post('/api/face-groups/99999/reject', json={'face_ids': []}).status_code == 404


def test_reject_persists_across_recompute(env):
    _app, client, faces, (_db, manual) = env
    _compute(client)
    g = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    r = client.post('/api/face-groups/%d/reject' % g['id'], json={'face_ids': []})
    assert r.status_code == 200 and r.json()['rejected'] == 6
    assert manual.get_face_group_rejected_ids() == set(faces[0])
    nxt = client.get('/api/face-groups/next?threshold=0.55').json()
    assert nxt['group']['size'] == 4 and nxt['remaining_groups'] == 1
    # faces stay unnamed
    assert manual.conn.execute('SELECT COUNT(*) FROM faces WHERE identity IS NOT NULL').fetchone()[0] == 0
    # a recompute (new group ids) keeps the rejected group hidden
    _compute(client, force=1)
    after = client.get('/api/face-groups/next?threshold=0.55').json()
    assert after['group']['size'] == 4 and after['remaining_groups'] == 1
    assert client.get('/api/face-groups/status').json()['groups'] == 1


def test_live_filtering_drops_small_groups_and_named_faces(env):
    _app, client, faces, (_db, _manual) = env
    _compute(client)
    # name 4 of the 6 faces of the biggest group via the single-face path
    for f in faces[0][:4]:
        client.post('/api/faces/auto:%d/identity' % f, json={'name': 'Cara'})
    r = client.get('/api/face-groups/next?threshold=0.55').json()
    assert r['group']['size'] == 4 and {f['face_id'] for f in r['group']['faces']}.isdisjoint(faces[0][:4])
    # name 2 more of the 4-group: it drops below min_size and disappears
    for f in faces[1][:2]:
        client.post('/api/faces/auto:%d/identity' % f, json={'name': 'Cara'})
    r = client.get('/api/face-groups/next?threshold=0.55').json()
    assert r['group'] is None and r['remaining_groups'] == 0
    assert client.get('/api/face-groups/status').json()['stale'] is True


def test_suggested_known_person(env):
    _app, client, faces, (db, manual) = env
    # name every face of cluster 1 except by hand-promote: use cluster 2's embeddings
    # (below min_size, so it is never itself a group) as the reference for 'Dana'
    _compute(client)
    for f in faces[2]:
        client.post('/api/faces/auto:%d/identity' % f, json={'name': 'Dana'})
    # a fresh group made of cluster-2-like faces: add 3 more near that centre
    ref = np.frombuffer(db.conn.execute('SELECT embedding FROM faces WHERE id = ?',
                                        (faces[2][0],)).fetchone()[0], dtype=np.float32)
    rng = np.random.default_rng(9)
    for i in range(4):
        fid = db.upsert_file_path('q%d.jpg' % i, 'ffff%04x' % i, size=10)
        db.conn.commit()
        db.add_manual_face(fid, [1, 1, 60, 60], _unit(ref + 0.03 * rng.standard_normal(D)).tobytes(), 0.9)
    db.conn.commit()
    _compute(client, force=1)
    groups = []
    for skip in range(3):
        g = client.get('/api/face-groups/next?threshold=0.55&include_approved=1&skip=%d' % skip).json()['group']
        if g:
            groups.append(g)
    sug = [g['suggested'] for g in groups if g['suggested']]
    assert sug and sug[0]['name'] == 'Dana' and sug[0]['score'] > 0.55


def test_cancel_flag_and_queue_registration(env):
    app, client, _f, _ = env
    from media_manager import job_queue
    assert '/api/face-groups/compute/start' in job_queue.JOB_WEIGHTS
    r = client.post('/api/bulk-queue', json={'key': 'fg', 'label': 'Face groups',
                                             'start_url': '/api/face-groups/compute/start?threshold=0.6',
                                             'status_url': '/api/face-groups/compute/status'})
    assert r.status_code == 200, r.text
    assert client.get('/api/face-groups/compute/status').status_code == 200


def test_compute_excludes_trashed_and_damaged(env):
    _app, client, faces, (db, manual) = env
    # damage the file of one face of the 4-group so it drops to 3 (still a group), and
    # another one -> 2 (gone)
    for f in faces[1][:2]:
        fid = db.conn.execute('SELECT file_id FROM faces WHERE id = ?', (f,)).fetchone()[0]
        db.mark_broken(fid)
    db.conn.commit()
    st = _compute(client)
    assert st['groups'] == 1 and st['faces_in_groups'] == 6


def test_compute_without_gpu_ends_in_clear_error(env, monkeypatch):
    _app, client, _f, _ = env
    monkeypatch.undo()                                   # drop the cpu hook: real resolution
    monkeypatch.setattr(compute, 'torch_device', lambda: 'cpu')
    st = _compute(client, threshold=0.55, min_size=3)
    assert 'needs a GPU' in (st['error'] or '') and 'MEDIA_DEVICE' in st['error']
    assert st['groups'] == 0 and st['computed_at'] is None and st['running'] is False
    assert st['device'] is None


def test_status_reports_device(env):
    _app, client, _f, _ = env
    st = _compute(client, threshold=0.55, min_size=3)
    assert st['error'] is None and st['device'] == 'cpu'     # the hook device, reported


def _near_dana_group(env_tuple):
    """Name cluster 2 'Dana', add a 4-face group right next to her, recompute."""
    _app, client, faces, (db, _manual) = env_tuple
    _compute(client)
    for f in faces[2]:
        client.post('/api/faces/auto:%d/identity' % f, json={'name': 'Dana'})
    ref = np.frombuffer(db.conn.execute('SELECT embedding FROM faces WHERE id = ?',
                                        (faces[2][0],)).fetchone()[0], dtype=np.float32)
    rng = np.random.default_rng(9)
    for i in range(4):
        fid = db.upsert_file_path('q%d.jpg' % i, 'ffff%04x' % i, size=10)
        db.conn.commit()
        db.add_manual_face(fid, [1, 1, 60, 60], _unit(ref + 0.03 * rng.standard_normal(D)).tobytes(), 0.9)
    db.conn.commit()
    _compute(client, force=1)


def _sizes(client, **extra):
    out, skip = [], 0
    while True:
        q = 'threshold=0.55&skip=%d' % skip + ''.join('&%s=%s' % kv for kv in extra.items())
        g = client.get('/api/face-groups/next?' + q).json()['group']
        if not g:
            return out
        out.append(g['size'])
        skip += 1


def test_near_named_groups_hidden_by_default(env):
    _app, client, faces, _ = env
    _near_dana_group(env)
    r = client.get('/api/face-groups/next?threshold=0.55').json()
    assert r['group']['size'] == 6 and r['group']['suggested'] is None
    assert r['remaining_groups'] == 2 and r['remaining_faces'] == 10
    assert r['near_named_hidden'] == 1
    assert _sizes(client) == [6, 4]                       # skip indexes the filtered sequence
    assert client.get('/api/face-groups/next?threshold=0.55&skip=2').json()['group'] is None
    full = client.get('/api/face-groups/next?threshold=0.55&include_approved=1').json()
    assert full['remaining_groups'] == 3 and full['remaining_faces'] == 14
    assert full['near_named_hidden'] == 0
    assert sorted(_sizes(client, include_approved=1)) == [4, 4, 6]
    # tail skips: the hidden group is shown last-or-between only when included
    assert _sizes(client, include_approved=0) == [6, 4]


def test_near_named_param_clamped_and_threshold(env):
    _app, client, _f, _ = env
    _near_dana_group(env)
    base = '/api/face-groups/next?threshold=0.55&near_named='
    assert client.get(base + '0.6').json()['near_named_hidden'] == 1
    # absurd values clamp to 0.9 / 0.2 instead of erroring
    assert client.get(base + '5').json() == client.get(base + '0.9').json()
    assert client.get(base + '-3').json() == client.get(base + '0.2').json()


def test_far_groups_shown_when_nobody_named(env):
    _app, client, _f, _ = env
    _compute(client)
    r = client.get('/api/face-groups/next?threshold=0.55').json()
    assert r['near_named_hidden'] == 0 and r['remaining_groups'] == 2
    assert _sizes(client) == _sizes(client, include_approved=1) == [6, 4]


def test_approving_person_hides_similar_group_and_invalidates_cache(env):
    _app, client, faces, _ = env
    _compute(client)
    assert client.get('/api/face-groups/next?threshold=0.55').json()['remaining_groups'] == 2
    # approve the 4-group as Eve (via the group API): the 6-group stays far from her
    g = client.get('/api/face-groups/next?threshold=0.55&skip=1').json()['group']
    assert g['size'] == 4
    # name the 6-group's faces partially elsewhere so a group stays; instead name 1 face
    # of cluster 0 as Frank: that face is a named reference right inside the 6-group
    r = client.post('/api/faces/auto:%d/identity' % faces[0][0], json={'name': 'Frank'})
    assert r.status_code == 200
    now = client.get('/api/face-groups/next?threshold=0.55').json()
    assert now['near_named_hidden'] == 1 and now['remaining_groups'] == 1
    assert now['group']['size'] == 4
    assert client.get('/api/face-groups/next?threshold=0.55&include_approved=1').json()['remaining_groups'] == 2


def test_default_name_is_next_free_unknown_n(env):
    """The name box is prefilled with 'Unknown N' (one past the highest existing one)."""
    _app, client, faces, (_db, _manual) = env
    _compute(client)
    g = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    assert g['default_name'] == 'Unknown 1'
    # approve the biggest group under 'Unknown 1' -> the next group's default moves on
    r = client.post('/api/face-groups/%d/approve' % g['id'],
                    json={'name': g['default_name'], 'face_ids': [f['face_id'] for f in g['faces']]})
    assert r.status_code == 200, r.text
    nxt = client.get('/api/face-groups/next?threshold=0.55').json()['group']
    assert nxt is not None and nxt['default_name'] == 'Unknown 2'


def test_group_served_whole_above_old_200_cap(env, monkeypatch):
    """A group is no longer cut at 200 faces: the cap is a 5000-face backstop."""
    import re as _re
    src = open(__import__('media_manager.web', fromlist=['x']).__file__).read()
    m = _re.search(r'FACE_GROUP_MAX_FACES = (\d+)', src)
    assert m and int(m.group(1)) >= 5000
