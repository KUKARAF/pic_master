"""Batch category auto-match (MediaManager.match_categories) in SigLIP 2 / DINOv2 / legacy CLIP.

Hand-built low-dimensional vectors written straight into the embedding tables (no model, no GPU)."""
import os

import numpy as np
import pytest

from media_manager.category_resolver import (
    calibrate_category_threshold, resolve_categories_for_file, resolve_categories_for_checksums)
from media_manager.media_manager import MediaManager

DIMS = {'clip': 8, 'siglip': 10, 'dino': 8}


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def mm(tmp_path):
    os.makedirs(tmp_path / '.media')
    m = MediaManager(str(tmp_path / '.media' / 'media.db'))
    ids = {}

    def add(name, **c):
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = m.db.upsert_file_path(name + '.jpg', cs, size=10)
        m.db.conn.commit()
        for space, d in DIMS.items():
            b = _vec(d, **c).tobytes()
            if space == 'clip':
                m.db.insert_embedding(fid, b, 'clip-test')
            else:
                getattr(m.db, f'insert_{space}_embedding')(fid, b, f'{space}-test')
        ids[name] = (fid, cs)

    # category "beach" examples around e0 (+ tiny jitter), background spread on e3..e7
    for i, j in enumerate((0.0, 0.3, 0.6, 0.3, 0.1)):
        add(f'ex{i}', e0=1, e1=j)
    add("near", e0=1, e1=0.5)          # should match
    add('far', e5=1)                   # should not
    add('rej', e0=1, e1=0.1)           # rejected for the category
    for k in range(3, 8):
        add(f'bg{k}', **{f'e{k}': 1})
    m.ids = ids
    cid = m.manual.create_category('beach')
    for i in range(5):
        m.manual.add_file_category(ids[f'ex{i}'][1], cid)
    m.manual.exclude_file_from_category(ids['rej'][1], cid)
    m.cid = cid
    yield m
    m.close()


def _stored(m):
    return {(m.db.get_file_by_id(f)['path'], n, mod)
            for f, n, mod in [(r[0], r[1], r[3]) for r in m.db.conn.execute(
                'SELECT file_id, category_name, score, model FROM file_category_matches')]}


@pytest.mark.parametrize('model,mid', [('siglip', 'siglip-centroid-v1'), ('dino', 'dino-centroid-v1'),
                                       ('clip', 'clip-centroid-v1')])
def test_match_in_each_space_stores_model_id(mm, model, mid):
    matched, skipped, considered = mm.match_categories(model=model)
    got = _stored(mm)
    assert ('near.jpg', 'beach', mid) in got
    assert not any(p in ('far.jpg', 'rej.jpg') or p.startswith('ex') for p, _n, _m in got)
    assert {m for _p, _n, m in got} == {mid}
    assert matched == len(got) and considered >= 6


def test_dry_run_writes_nothing_and_unknown_model(mm):
    mm.match_categories(dry_run=True)
    assert _stored(mm) == set()
    with pytest.raises(ValueError):
        mm.match_categories(model='nope')


def test_legacy_rows_flagged_and_superseded_on_rerun(mm):
    near = mm.ids['near'][0]
    far = mm.ids['far'][0]
    mm.db.set_file_category_match(near, 'beach', 0.9, model='clip-centroid-v1')
    mm.db.set_file_category_match(far, 'beach', 0.9, model='clip-centroid-v1')
    # legacy flagged in resolvers
    res = resolve_categories_for_file(mm.manual, mm.db, near, mm.ids['near'][1])
    assert res[0]['legacy'] is True
    batched = resolve_categories_for_checksums(mm.manual, mm.db, [(near, mm.ids['near'][1])])
    assert batched[mm.ids['near'][1]][0]['legacy'] is True
    assert mm.db.count_category_matches_by_model() == {'clip-centroid-v1': 2}
    mm.match_categories(model='siglip')
    got = _stored(mm)
    assert ('near.jpg', 'beach', 'siglip-centroid-v1') in got      # replaced by the new space
    assert not any(p == 'far.jpg' for p, _n, _m in got)            # stale legacy guess superseded
    res = resolve_categories_for_file(mm.manual, mm.db, near, mm.ids['near'][1])
    assert res[0]['legacy'] is False


def test_legacy_kept_for_files_without_new_space_embedding(mm):
    cs = 'z' * 40
    fid = mm.db.upsert_file_path('only_clip.jpg', cs, size=1)
    mm.db.conn.commit()
    mm.db.insert_embedding(fid, _vec(8, e0=1).tobytes(), 'clip-test')
    mm.db.set_file_category_match(fid, 'beach', 0.9, model='clip-centroid-v1')
    mm.match_categories(model='siglip')
    assert [tuple(r) for r in mm.db.get_file_category_matches(fid)] == [('beach', 0.9, 'clip-centroid-v1')]


def test_stills_never_auto_matched_and_negatives_steer(mm):
    parent = 'p' * 40
    pid = mm.db.upsert_file_path('vid.mp4', parent, size=1)
    sid_cs = mm.ids['near'][1]
    mm.manual.add_frame_capture(sid_cs, parent, 1000)
    mm.match_categories(model='siglip')
    assert not any(p == 'near.jpg' for p, _n, _m in _stored(mm))
    # Rocchio: rejecting 'bg-ish' neighbours pulls the centroid; the call must still succeed
    mm.manual.exclude_file_from_category(mm.ids['far'][1], mm.cid)
    mm.match_categories(model='siglip')


def test_calibrate_threshold_is_per_space_scale():
    rng = np.random.default_rng(0)
    for lo, hi in ((0.0, 0.2), (0.3, 0.5)):          # two spaces with different baseline cosines
        pool = rng.uniform(lo, hi, 500)
        members = rng.uniform(hi + 0.1, hi + 0.3, 20)
        thr = calibrate_category_threshold(members, pool)
        assert hi - 0.05 < thr <= float(np.median(members))
    assert calibrate_category_threshold([], [0.1, 0.2, 0.3]) > 0.3
    assert calibrate_category_threshold([0.9], []) <= 0.9
