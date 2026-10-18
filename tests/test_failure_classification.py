"""Data failure vs our-side failure: only the former may mark/stamp/sentinel a file.

A file whose bytes are genuinely bad is marked (broken / sentinel / stamped done) so
jobs stop re-picking it. An infrastructure failure on OUR side (EMFILE, ENOSPC, a
dead GPU/estimator, a bug) says nothing about the file, so it must be logged and
skipped WITHOUT marking — otherwise a full disk or a crashed model permanently
flags healthy photos as damaged / done. Covers:

  * /thumb: a thumbs-dir save failure serves a placeholder but does NOT mark broken;
    a genuinely corrupt source still does.
  * broken_finder: EMFILE while opening doesn't mark; a corrupt file does.
  * age estimation: per-face 'error' results are never saved; all-error → HTTP 500;
    the bulk job aborts after a run of all-error photos.
  * object_index: RuntimeError from the encoder writes no sentinel and aborts after
    a run; ValueError still writes the sentinel.
  * face renormalize: unexpected errors don't stamp (abort after a run); data errors
    do; an unavailable PCN aborts before touching anything.

Run: python -m pytest tests/test_failure_classification.py   (or standalone)
"""
import errno
import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import broken_finder, face_detector, object_index
from media_manager import web as webmod
from media_manager.database import FACE_NORM_VERSION
from media_manager.web import create_app


def _app(tmp):
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    db, errors, manual = app.state.dbs
    return app, db, manual, TestClient(app)


def _add_file(tmp, db, name, data=None):
    """A file row whose bytes exist on disk: a real 64x48 JPEG unless `data`."""
    path = os.path.join(tmp, name)
    if data is None:
        Image.new('RGB', (64, 48), (90, 120, 150)).save(path, format='JPEG')
    else:
        with open(path, 'wb') as f:
            f.write(data)
    cs = (name.replace('.', '') * 40)[:40]
    fid = db.upsert_file_path(name, cs, size=os.path.getsize(path))
    db.conn.commit()
    return fid, cs


def _wait(client, url):
    for _ in range(200):
        st = client.get(url).json()
        if not st['running']:
            return st
        time.sleep(0.05)
    raise AssertionError(f'{url} never finished')


# --- 1. thumbnails ---------------------------------------------------------------

def test_thumb_save_failure_does_not_mark_broken():
    with tempfile.TemporaryDirectory() as tmp:
        _app_, db, _m, client = _app(tmp)
        fid, _cs = _add_file(tmp, db, 'good.jpg')

        def full_disk(img, dst, quality):
            raise OSError(errno.ENOSPC, 'No space left on device')

        with patch.object(webmod, '_save_jpeg_atomic', full_disk):
            r = client.get('/thumb/%d' % fid)
        assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg'
        assert db.get_file_by_id(fid)['broken'] is None, 'healthy source marked broken'

        # Disk recovered → the very next request generates the real thumbnail.
        r = client.get('/thumb/%d' % fid)
        assert r.status_code == 200
        assert os.path.isfile(os.path.join(tmp, '.media', 'thumbs', '%d.jpg' % fid))


def test_thumb_corrupt_source_is_marked_broken():
    with tempfile.TemporaryDirectory() as tmp:
        _app_, db, _m, client = _app(tmp)
        fid, _cs = _add_file(tmp, db, 'bad.jpg', data=b'definitely not a jpeg')
        assert client.get('/thumb/%d' % fid).status_code == 200
        assert db.get_file_by_id(fid)['broken'] is not None


def test_thumb_status_contract():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'a.jpg')
        Image.new('RGB', (40, 30)).save(src)
        dst = os.path.join(tmp, 'out', 'a.jpg')
        assert webmod._make_thumbnail(src, dst)[0] == webmod.THUMB_OK
        with open(src, 'wb') as f:
            f.write(b'junk')
        assert webmod._make_thumbnail(src, dst)[0] == webmod.THUMB_DAMAGED
        emfile = OSError(errno.EMFILE, 'Too many open files')
        with patch.object(webmod.PILImage, 'open', side_effect=emfile):
            assert webmod._make_thumbnail(src, dst)[0] == webmod.THUMB_SAVE_FAILED


# --- 2. broken_finder ------------------------------------------------------------

def _broken_db(rel):
    conn = sqlite3.connect(':memory:')
    conn.executescript('CREATE TABLE files (id INTEGER PRIMARY KEY, broken INTEGER);'
                       'CREATE TABLE file_paths (file_id INTEGER, path TEXT);'
                       'INSERT INTO files (id) VALUES (1);')
    conn.execute('INSERT INTO file_paths VALUES (1, ?)', (rel,))
    return conn


def test_broken_finder_emfile_does_not_mark():
    with tempfile.TemporaryDirectory() as tmp:
        Image.new('RGB', (20, 20)).save(os.path.join(tmp, 'ok.jpg'))
        conn = _broken_db('ok.jpg')
        logged = []

        class Log:
            def log(self, path, msg):
                logged.append((path, msg))

        emfile = OSError(errno.EMFILE, 'Too many open files')
        with patch.object(broken_finder.Image, 'open', side_effect=emfile):
            assert broken_finder.verify_image(os.path.join(tmp, 'ok.jpg'))[0] is None
            n = broken_finder.find_broken(tmp, conn, tmp, max_workers=1, error_log=Log())
        assert n == 0
        assert conn.execute('SELECT broken FROM files').fetchone()[0] is None
        assert logged and 'Too many open files' in logged[0][1], logged
        # MemoryError is ours too.
        with patch.object(broken_finder.Image, 'open', side_effect=MemoryError()):
            assert broken_finder.verify_image(os.path.join(tmp, 'ok.jpg'))[0] is None


def test_broken_finder_corrupt_file_is_marked():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'bad.jpg'), 'wb') as f:
            f.write(b'junk junk junk')
        conn = _broken_db('bad.jpg')
        assert broken_finder.find_broken(tmp, conn, tmp, max_workers=1) == 1
        assert conn.execute('SELECT broken FROM files').fetchone()[0] is not None


# --- 3. age estimation -----------------------------------------------------------

class ErroringEstimator:
    """Answers every face with the worker's per-face error shape, optionally
    letting the first `ok` refs through with a real estimate."""

    def __init__(self, ok=()):
        self.ok = set(ok)
        self.calls = 0

    def estimate(self, image_path, faces, bodies=None):
        self.calls += 1
        out = []
        for f in list(faces) + list(bodies or []):
            if f['ref'] in self.ok:
                out.append({'face_ref': f['ref'], 'age': 30.0, 'gender': 'male'})
            else:
                out.append({'face_ref': f['ref'], 'age': None, 'gender': None,
                            'error': 'XPU out of resources'})
        return out


def _named_face(manual, cs, name='Ann', bbox=(1, 1, 30, 30)):
    fid = manual.add_manual_face(cs, list(bbox), np.ones(512, np.float32).tobytes(), 64, 48)
    manual.conn.execute('UPDATE faces SET identity = ? WHERE id = ?', (name, fid))
    manual.conn.commit()
    return 'manual:%d' % fid


def _age_refs(manual, cs):
    return {r['face_ref']: r for r in manual.get_age_estimates_for_checksum(cs)}


def test_age_all_error_not_saved_and_500():
    with tempfile.TemporaryDirectory() as tmp:
        _app_, db, manual, client = _app(tmp)
        fid, cs = _add_file(tmp, db, 'p.jpg')
        _named_face(manual, cs)
        with patch.object(webmod, '_get_age_estimator', lambda: ErroringEstimator()):
            r = client.post('/api/files/%d/estimate-age' % fid)
        assert r.status_code == 500 and 'XPU out of resources' in r.text, r.text
        assert _age_refs(manual, cs) == {}
        assert cs in (manual.get_all_checksums_with_named_face()
                      - manual.get_checksums_with_age_estimate()), 'photo marked done'


def test_age_partial_error_saves_only_good():
    with tempfile.TemporaryDirectory() as tmp:
        _app_, db, manual, client = _app(tmp)
        fid, cs = _add_file(tmp, db, 'p.jpg')
        good = _named_face(manual, cs, 'Ann', (1, 1, 20, 20))
        bad = _named_face(manual, cs, 'Bob', (40, 1, 60, 20))
        with patch.object(webmod, '_get_age_estimator', lambda: ErroringEstimator(ok={good})):
            r = client.post('/api/files/%d/estimate-age' % fid)
        assert r.status_code == 200, r.text
        assert [x['face_ref'] for x in r.json()['results']] == [good]
        rows = _age_refs(manual, cs)
        assert good in rows and bad not in rows, rows


def test_age_bulk_job_aborts_on_consecutive_all_error():
    with tempfile.TemporaryDirectory() as tmp:
        _app_, db, manual, client = _app(tmp)
        checksums = []
        for i in range(webmod.MAX_CONSECUTIVE_FAILURES + 2):
            _fid, cs = _add_file(tmp, db, 'p%d.jpg' % i)
            _named_face(manual, cs)
            checksums.append(cs)
        est = ErroringEstimator()
        with patch.object(webmod, '_get_age_estimator', lambda: est):
            assert client.post('/api/estimate-age/start').status_code == 200
            st = _wait(client, '/api/estimate-age/status')
        assert st['error'] and 'Aborted' in st['error'], st
        assert est.calls == webmod.MAX_CONSECUTIVE_FAILURES, est.calls
        assert st['failed'] == webmod.MAX_CONSECUTIVE_FAILURES, st
        assert st['pending'] == len(checksums), 'errored photos were marked done'


# --- 4/5. face renormalize -------------------------------------------------------

def _norm_versions(db):
    return [r[0] for r in db.conn.execute(
        "SELECT norm_version FROM faces WHERE bbox != '[]' ORDER BY id")]


def _renorm_env(tmp, n_files):
    app, db, _manual, client = _app(tmp)
    for i in range(n_files):
        fid, _cs = _add_file(tmp, db, 'f%d.jpg' % i)
        db.add_manual_face(fid, [1, 2, 40, 40], np.ones(512, np.float32).tobytes(), 0.9)
    db.conn.execute("UPDATE faces SET norm_version = 0 WHERE bbox != '[]'")
    db.conn.commit()
    return db, client


def _run_renorm(client):
    assert client.post('/api/faces/renormalize/start').status_code == 200
    return _wait(client, '/api/faces/renormalize/status')


def test_renormalize_unexpected_error_does_not_stamp_and_aborts():
    with tempfile.TemporaryDirectory() as tmp:
        db, client = _renorm_env(tmp, webmod.MAX_CONSECUTIVE_FAILURES + 2)

        class Buggy:
            calls = 0

            def normalize_faces(self, img, faces):
                Buggy.calls += 1
                raise KeyError('bug')

        with patch.object(webmod, '_get_face_detector', lambda: Buggy()):
            st = _run_renorm(client)
        assert st['error'] and 'Aborted' in st['error'], st
        assert Buggy.calls == webmod.MAX_CONSECUTIVE_FAILURES, Buggy.calls
        assert set(_norm_versions(db)) == {0}, _norm_versions(db)


def test_renormalize_data_error_stamps():
    with tempfile.TemporaryDirectory() as tmp:
        db, client = _renorm_env(tmp, webmod.MAX_CONSECUTIVE_FAILURES + 2)

        class BadData:
            def normalize_faces(self, img, faces):
                raise ValueError('degenerate bbox')

        with patch.object(webmod, '_get_face_detector', lambda: BadData()):
            st = _run_renorm(client)
        assert st['error'] is None, st
        assert set(_norm_versions(db)) == {FACE_NORM_VERSION}, _norm_versions(db)


def test_renormalize_aborts_when_pcn_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        db, client = _renorm_env(tmp, 2)

        class NoPCN:
            def pcn_available(self):
                return False

            def normalize_faces(self, img, faces):
                raise AssertionError('must not be called')

        with patch.object(webmod, '_get_face_detector', lambda: NoPCN()):
            st = _run_renorm(client)
        assert st['error'] and 'PCN' in st['error'], st
        assert set(_norm_versions(db)) == {0}


def test_face_detector_normalize_refuses_without_pcn():
    d = face_detector.FaceDetector.__new__(face_detector.FaceDetector)
    d._pcn, d._pcn_failed = None, True
    assert d.pcn_available() is False
    try:
        d.normalize_faces(np.zeros((10, 10, 3), np.uint8), [])
    except face_detector.PCNUnavailable:
        pass
    else:
        raise AssertionError('normalize_faces ran without PCN')


# --- 6. object index -------------------------------------------------------------

class FakeObjDB:
    def __init__(self):
        self.inserted = {}

    def insert_object_regions(self, fid, rows, model):
        self.inserted[fid] = rows


class FakeEncoder:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def model_id(self):
        return 'fake'

    def embed_regions(self, im):
        self.calls += 1
        raise self.exc


def _obj_candidates(tmp, n):
    out = []
    for i in range(n):
        Image.new('RGB', (32, 32)).save(os.path.join(tmp, 'o%d.jpg' % i))
        out.append((i + 1, 'o%d.jpg' % i))
    return out


def test_object_index_runtime_error_no_sentinel_and_aborts():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _obj_candidates(tmp, object_index.MAX_CONSECUTIVE_EMBED_FAILURES + 3)
        db, enc = FakeObjDB(), FakeEncoder(RuntimeError('XPU lost'))
        with patch.dict(os.environ, {'MEDIA_OBJECT_DECODE_BATCH': '1'}):
            try:
                object_index.build_object_index(db, enc, tmp, image_exts={'.jpg'},
                                                candidates=cands)
            except RuntimeError as exc:
                assert 'Aborted' in str(exc), exc
            else:
                raise AssertionError('build did not abort')
        assert db.inserted == {}, 'sentinel written for an our-side failure'
        assert enc.calls == object_index.MAX_CONSECUTIVE_EMBED_FAILURES, enc.calls


def test_object_index_value_error_writes_sentinel():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _obj_candidates(tmp, 3)
        db = FakeObjDB()
        object_index.build_object_index(db, FakeEncoder(ValueError('degenerate')), tmp,
                                        image_exts={'.jpg'}, candidates=cands)
        assert sorted(db.inserted) == [1, 2, 3]
        assert all(rows[0][1] == b'' for rows in db.inserted.values())


if __name__ == '__main__':
    for t in (test_thumb_save_failure_does_not_mark_broken,
              test_thumb_corrupt_source_is_marked_broken,
              test_thumb_status_contract,
              test_broken_finder_emfile_does_not_mark,
              test_broken_finder_corrupt_file_is_marked,
              test_age_all_error_not_saved_and_500,
              test_age_partial_error_saves_only_good,
              test_age_bulk_job_aborts_on_consecutive_all_error,
              test_renormalize_unexpected_error_does_not_stamp_and_aborts,
              test_renormalize_data_error_stamps,
              test_renormalize_aborts_when_pcn_unavailable,
              test_face_detector_normalize_refuses_without_pcn,
              test_object_index_runtime_error_no_sentinel_and_aborts,
              test_object_index_value_error_writes_sentinel):
        t()
        print('ok:', t.__name__)
    print('\nFAILURE-CLASSIFICATION TESTS PASSED')
