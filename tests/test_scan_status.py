"""Pipeline-state engine (media_manager/scan_status.py) behind the file manager colouring.

Seeds a temp media.db + manual.db with rows for every stage (sentinels included) and
checks the per-stage flags, the broken/scanned/partial/tracked classification, that the
small (IN-lookup) and large (cached full done-set) paths agree, summarize(), the cache's
invalidate(), and that 'missing X' agrees with the pending query the bulk job for X uses.

Run: python -m pytest tests/test_scan_status.py   (or standalone: python tests/test_scan_status.py)
"""
import os
import random
import tempfile

import numpy as np

from media_manager import pattern_deep, region_index, scan_status
from media_manager.database import Database
from media_manager.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from media_manager.manual_db import ManualDB

IMAGE_KEYS = [k for k, _l, _r in scan_status.STAGES['image']]
LABEL = {k: lbl for kind in scan_status.STAGES.values() for k, lbl, _r in kind}


def _dbs():
    tmp = tempfile.mkdtemp()
    return Database(os.path.join(tmp, 'media.db')), ManualDB(os.path.join(tmp, 'manual.db'))


def _add(db, path):
    fid = db.upsert_file_path(path, 'cs-' + path, size=1)
    db.conn.commit()
    return fid


def _stage(db, fid, key, algo=None):
    """Write the row(s) the job for `key` writes when it finds NOTHING (the sentinel form)."""
    if key == 'exif':
        db.update_file_metadata(fid)
    elif key == 'clip':
        db.insert_embedding(fid, b'\0' * 8, 'm')
    elif key == 'tags':
        db.insert_detections(fid, [], 'm')                       # '__indexed__' sentinel
    elif key == 'faces':
        db.insert_faces(fid, [], 'm')                            # bbox '[]' sentinel
    elif key == 'bodies':
        db.insert_body_embeddings(fid, [], 'm')                  # bbox '[]' sentinel
    elif key == 'objects':
        db.insert_object_regions(fid, [((0, 0, 0, 0), b'')], 'm')  # empty-box sentinel
    elif key == 'region':
        db.insert_region_index(fid, np.zeros((0, region_index.DIM), np.int8),
                               np.zeros((0, 2), np.float16), np.zeros((0, 128), np.uint8),
                               algo or region_index.ALGO_TAG)
    elif key == 'pattern':
        db.insert_pattern_windows(fid, np.zeros((0, pattern_deep.DIM), np.int8),
                                  algo or pattern_deep.ALGO_TAG)
    elif key == 'phash':
        db.insert_phash(fid, b'\0' * 8, b'\0' * 8, 1, 1, 'a')
    elif key == 'tiles':
        db.conn.execute('INSERT INTO tile_embeddings (file_id, tile_index, embedding) VALUES (?,0,?)',
                        (fid, b'\0' * 8))
        db.conn.commit()
    else:
        raise ValueError(key)


def _captures(manual, video_cs, n):
    for i in range(n):
        manual.add_frame_capture(f'{video_cs}-still{i}', video_cs, i * 1000)


def _both_paths(db, manual, files):
    """Small-path and forced large-path results; asserts they're identical."""
    scan_status.invalidate()
    small = scan_status.status_for_files(db, manual, files, large_threshold=10 ** 9)
    large = scan_status.status_for_files(db, manual, files, large_threshold=0)
    assert small == large, (small, large)
    scan_status.invalidate()
    return small


def run_classification():
    db, manual = _dbs()
    f = {}
    for name in ('full.jpg', 'broken.jpg', 'noface.jpg', 'stale.jpg', 'bare.jpg',
                 'frameonly.jpg', 'hidden.jpg', 'tilesonly.jpg',
                 'v_done.mp4', 'v_two.mp4', 'v_half.mp4', 'notes.txt'):
        f[name] = _add(db, 'dir/' + name)
    required = [k for k, _l, r in scan_status.STAGES['image'] if r]
    for name in ('full.jpg', 'broken.jpg'):
        for key in required:
            _stage(db, f[name], key)
    db.mark_broken(f['broken.jpg'])
    db.mark_broken(f['notes.txt'])  # broken wins even for 'other'

    db.set_noface_for_checksums(['cs-dir/noface.jpg'])
    _stage(db, f['noface.jpg'], 'exif')

    _stage(db, f['stale.jpg'], 'region', algo='old-region-algo')
    _stage(db, f['stale.jpg'], 'pattern', algo='old-pattern-algo')
    _stage(db, f['stale.jpg'], 'clip')

    # Frame-specific rows only: the primary-frame jobs still consider these pending.
    fo = f['frameonly.jpg']
    db.insert_embedding(fo, b'\0' * 8, 'm', frame_index=5)
    db.insert_detections(fo, [], 'm', frame_index=3)
    db.add_manual_face(fo, [1, 2, 3, 4], b'', 0.9, frame_index=2)

    db.conn.execute('UPDATE files SET hidden = 1 WHERE id = ?', (f['hidden.jpg'],))
    db.conn.commit()
    _stage(db, f['tilesonly.jpg'], 'tiles')

    _captures(manual, 'cs-dir/v_done.mp4', 3)
    manual.mark_video_frame_scanned('cs-dir/v_done.mp4')
    _stage(db, f['v_done.mp4'], 'clip')   # videos need a similarity embedding too
    _captures(manual, 'cs-dir/v_two.mp4', 2)
    _captures(manual, 'cs-dir/v_half.mp4', 4)

    files = [(i, 'dir/' + n) for n, i in f.items()]
    st = _both_paths(db, manual, files)
    by = {n: st[i] for n, i in f.items()}

    full = by['full.jpg']
    assert full['kind'] == 'image' and full['state'] == 'scanned', full
    assert all(full['stages'][k] for k in required) and full['stages']['tiles'] is False
    assert full['missing'] == []

    assert by['broken.jpg']['state'] == 'broken' and by['broken.jpg']['missing'] == []
    assert by['notes.txt'] == {'kind': 'other', 'state': 'broken', 'stages': {}, 'missing': []}

    nf = by['noface.jpg']
    assert nf['stages']['faces'] and nf['stages']['exif'] and nf['state'] == 'partial', nf
    assert LABEL['faces'] not in nf['missing'] and LABEL['clip'] in nf['missing']

    stale = by['stale.jpg']['stages']
    assert stale['region'] is False and stale['pattern'] is False and stale['clip'] is True

    fos = by['frameonly.jpg']
    assert not any(fos['stages'].values()) and fos['state'] == 'tracked', fos

    hid = by['hidden.jpg']
    assert hid['stages']['region'] and hid['stages']['pattern'] and hid['state'] == 'partial', hid

    to = by['tilesonly.jpg']
    assert to['state'] == 'partial' and to['stages']['tiles'], to
    assert len(to['missing']) == len(required)

    assert by['bare.jpg']['state'] == 'tracked'
    assert len(by['bare.jpg']['missing']) == len(required)
    assert set(by['bare.jpg']['stages']) == set(IMAGE_KEYS)

    assert by['v_done.mp4']['kind'] == 'video' and by['v_done.mp4']['state'] == 'scanned'
    assert by['v_two.mp4']['state'] == 'tracked'
    assert by['v_two.mp4']['missing'] == [LABEL['clip'], LABEL['stills'], LABEL['video_faces']]
    assert by['v_half.mp4']['state'] == 'partial'
    assert by['v_half.mp4']['stages'] == {'clip': False, 'stills': True, 'video_faces': False}

    summ = scan_status.summarize(st)
    assert summ == {'broken': 2, 'scanned': 2, 'partial': 5, 'tracked': 3, 'total': 12}, summ
    assert scan_status.summarize(list(st.values())) == summ
    assert scan_status.summarize({}) == {'broken': 0, 'scanned': 0, 'partial': 0,
                                         'tracked': 0, 'total': 0}
    assert scan_status.status_for_files(db, manual, []) == {}

    # Cache: the large path serves a snapshot until invalidate().
    scan_status.invalidate()
    bare = [(f['bare.jpg'], 'dir/bare.jpg')]
    assert not scan_status.status_for_files(db, manual, bare, large_threshold=0)[f['bare.jpg']]['stages']['clip']
    _stage(db, f['bare.jpg'], 'clip')
    assert not scan_status.status_for_files(db, manual, bare, large_threshold=0)[f['bare.jpg']]['stages']['clip']
    scan_status.invalidate()
    assert scan_status.status_for_files(db, manual, bare, large_threshold=0)[f['bare.jpg']]['stages']['clip']
    scan_status.invalidate()
    print('OK  classification / sentinels / noface / algo / broken / video / summarize / cache')


def run_consistency():
    """'missing X' == the bulk job's pending list for X, on a randomly seeded library."""
    rng = random.Random(7)
    db, manual = _dbs()
    files = []
    for i in range(300):
        ext = rng.choice(['.jpg', '.png', '.jpg', '.mp4', '.txt'])
        path = f'lib/{i % 7}/f{i}{ext}'
        fid = _add(db, path)
        files.append((fid, path))
        if ext == '.mp4':
            _captures(manual, 'cs-' + path, rng.choice([0, 2, 3, 5]))
            if rng.random() < 0.5:
                manual.mark_video_frame_scanned('cs-' + path)
        else:
            for key in IMAGE_KEYS:
                if rng.random() < 0.5:
                    stale = key in ('region', 'pattern') and rng.random() < 0.3
                    _stage(db, fid, key, algo='stale' if stale else None)
            if rng.random() < 0.1:
                db.set_noface_for_checksums(['cs-' + path])
        if rng.random() < 0.1:
            db.conn.execute('UPDATE files SET hidden = 1 WHERE id = ?', (fid,))
        if rng.random() < 0.05:
            db.mark_broken(fid)
    db.conn.commit()
    st = _both_paths(db, manual, files)

    def missing(key, kind='image', healthy=False):
        return {fid for fid, s in st.items() if s['kind'] == kind and LABEL[key] in s['missing']
                and not (healthy and s['state'] == 'broken')}

    def images(rows):
        return {r[0] for r in rows if os.path.splitext(r[1])[1].lower() in IMAGE_EXTENSIONS}

    assert missing('clip') == images(db.get_unindexed_files())
    assert missing('exif') == images(db.get_files_without_metadata())
    assert missing('tags') == images(db.get_undetected_files())
    assert missing('faces') == images(db.get_unface_indexed_files())
    assert missing('bodies') == images(db.get_unbody_indexed_files())
    assert missing('objects') == images(db.get_unobject_indexed_files())
    assert missing('region') == images(db.get_unregion_indexed_files(region_index.ALGO_TAG))
    assert missing('pattern') == images(db.get_unpattern_indexed_files(pattern_deep.ALGO_TAG))
    assert missing('phash') == images(db.get_unphashed_files())
    untiled = images(db.get_untiled_files())
    assert {fid for fid, s in st.items() if s['kind'] == 'image' and not s['stages']['tiles']} == untiled

    # web._videos_needing_frames / _videos_needing_face_scan (healthy, non-hidden videos).
    counts = manual.get_capture_counts_by_parent()
    scanned = manual.get_frame_scanned_video_checksums()
    vids = db.get_video_files(list(VIDEO_EXTENSIONS))
    assert missing('stills', 'video', healthy=True) == {f for f, _p, cs in vids if counts.get(cs, 0) < 3}
    assert missing('video_faces', 'video', healthy=True) == {f for f, _p, cs in vids if cs not in scanned}
    print('OK  consistency with pending queries on', len(files), 'files:', scan_status.summarize(st))


def run():
    run_classification()
    run_consistency()
    print('ALL ASSERTIONS PASSED')


def test_scan_status():
    run()


if __name__ == '__main__':
    run()
