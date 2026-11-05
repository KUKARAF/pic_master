"""Face pipeline: PCN device/GPU-fatal handling, shared-disk + concurrent worker detection,
protocol fallback, and batched DB writes. Stubs only -- no GPU, no real models.

Run: python -m pytest tests/test_face_pipeline_perf.py
"""
import os
import sys
import threading
import time

import numpy as np
import pytest
import xxhash
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

import media_manager.web as web_mod
from media_manager import compute, face_detector, worker_client, worker_protocol, worker_server
from media_manager.pcn import pcn_detect
from media_manager.web import create_app

OCL = ('[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running '
       'OpenVINO-EP-subgraph_1 node.\n[GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES')


# ------------------------------------------------------------------ PCN device selection

def test_pcn_providers_default_cpu_and_env_override(monkeypatch):
    monkeypatch.delenv('MEDIA_PCN_DEVICE', raising=False)
    monkeypatch.setattr(compute, 'onnx_providers',
                        lambda: (_ for _ in ()).throw(AssertionError('GPU provider list must not be consulted')))
    assert pcn_detect.pcn_providers() == ['CPUExecutionProvider']
    assert pcn_detect.pcn_device() == 'cpu'
    monkeypatch.setenv('MEDIA_PCN_DEVICE', 'gpu')
    monkeypatch.setattr(compute, 'onnx_providers', lambda: [('OpenVINOExecutionProvider', {}), 'CPUExecutionProvider'])
    assert pcn_detect.pcn_device() == 'gpu'
    assert pcn_detect.pcn_providers()[0][0] == 'OpenVINOExecutionProvider'
    monkeypatch.setenv('MEDIA_PCN_DEVICE', 'nonsense')
    assert pcn_detect.pcn_providers() == ['CPUExecutionProvider']


def test_pcn_detector_sessions_are_cpu_by_default(monkeypatch):
    monkeypatch.delenv('MEDIA_PCN_DEVICE', raising=False)
    monkeypatch.setattr(compute, 'onnx_providers',
                        lambda: (_ for _ in ()).throw(AssertionError('must not be consulted')))
    det = pcn_detect.PCNDetector()
    assert det.on_gpu is False
    assert all(n.get_providers() == ['CPUExecutionProvider'] for n in det.nets)


# ------------------------------------------------------------------ FaceDetector (stubbed models)

class _Face:
    def __init__(self, bbox=(10, 10, 30, 30), score=0.95):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = score
        self.normed_embedding = np.ones(4, dtype=np.float32) / 2
        self.kps = np.array([[0, 0], [1, 0], [0.5, 0.5], [0, 1], [1, 1]], dtype=np.float32)


class _App:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, 0

    def get(self, img):
        self.calls += 1
        if self.exc:
            raise self.exc
        return [_Face()]


class _PCN:
    def __init__(self, exc=None, on_gpu=False):
        self.exc, self.on_gpu, self.calls = exc, on_gpu, 0

    def detect(self, img):
        self.calls += 1
        if self.exc:
            raise self.exc
        return []


def _detector(app=None, pcn=None):
    det = face_detector.FaceDetector.__new__(face_detector.FaceDetector)
    det.det_thresh = 0.5
    det._model_name = 'buffalo_l'
    det.app = app or _App()
    det._pcn, det._pcn_failed = pcn, pcn is None
    return det


IMG = np.zeros((64, 64, 3), np.uint8)


def test_gpu_fatal_in_pcn_disables_pcn_without_raising(capsys):
    pcn = _PCN(RuntimeError(OCL))
    det = _detector(pcn=pcn)
    for _ in range(3):  # the job continues; PCN is asked exactly once, then parked
        faces = det.detect_image(IMG)
        assert len(faces) == 1 and faces[0]['angle'] == 0.0
    assert pcn.calls == 1
    assert det._pcn is None and det._pcn_failed
    err = capsys.readouterr().err
    assert err.count('DISABLED') == 1          # logged once, loudly
    assert 'Traceback' not in err               # no per-image traceback spam


def test_gpu_fatal_in_gpu_pcn_falls_back_to_cpu(monkeypatch):
    cpu = _PCN(on_gpu=False)
    made = []

    def factory(*a, **k):
        made.append(k)
        return cpu
    monkeypatch.setattr('media_manager.pcn.PCNDetector', factory)
    det = _detector(pcn=_PCN(RuntimeError(OCL), on_gpu=True))
    det.detect_image(IMG)
    det.detect_image(IMG)
    assert made == [{'device': 'cpu'}]
    assert det._pcn is cpu and cpu.calls == 2   # retried on CPU for this and the next image


def test_non_fatal_pcn_error_logs_one_line_and_continues(capsys):
    det = _detector(pcn=_PCN(ValueError('boom')))
    det._pcn_errors = 0
    for _ in range(3):
        assert len(det.detect_image(IMG)) == 1
    err = capsys.readouterr().err
    assert err.count('PCN correct/recover pass failed') == 3
    assert err.count('Traceback') == 1          # full traceback only the first time
    assert det._pcn is not None                  # not disabled for an ordinary error


def test_gpu_fatal_in_insightface_still_surfaces_as_job_fatal():
    det = _detector(app=_App(RuntimeError(OCL)), pcn=_PCN())
    _p, faces, error = _detect_with_file(det)[0]
    assert faces == [] and compute.is_gpu_fatal(error)
    with pytest.raises(compute.GPUFatalError):
        compute.raise_if_gpu_fatal(error)


def _detect_with_file(det, name='f.jpg', size=(8, 8)):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, name)
        Image.new('RGB', size).save(p)
        return det.detect_faces([p])


def test_gpu_lock_only_wraps_insightface_calls():
    lock = threading.Lock()
    held = []

    class App(_App):
        def get(self, img):
            held.append(lock.locked())
            return super().get(img)

    class P(_PCN):
        def detect(self, img):
            held.append(('pcn', lock.locked()))
            return super().detect(img)
    det = _detector(app=App(), pcn=P())
    det.detect_image(IMG, gpu_lock=lock)
    assert held == [True, ('pcn', False)]        # GPU call locked, CPU PCN not
    assert not lock.locked()


def test_capped_read_scales_boxes_back(tmp_path):
    p = tmp_path / 'wide.jpg'
    Image.new('RGB', (6400, 200), (90, 90, 90)).save(p)
    img, scale = face_detector._read_bgr_capped(str(p), 3200)
    assert max(img.shape[:2]) <= 3200 and abs(scale - 2.0) < 0.05
    img, scale = face_detector._read_bgr_capped(str(p), 0)
    assert img.shape[1] == 6400 and scale == 1.0

    det = _detector(pcn=None)
    assert det.detect_image(IMG, scale=2.0)[0]['bbox'] == [20.0, 20.0, 60.0, 60.0]


# ------------------------------------------------------------------ worker: paths + concurrency

class FakeFaces:
    """Stands in for FaceDetector on the worker: deterministic faces per file, scrambled timing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live = self.peak = 0
        self.seen = []
        self.gpu_lock_arg = None

    def detect_faces(self, paths, gpu_lock=None, timings=None):
        self.gpu_lock_arg = gpu_lock
        out = []
        for p in paths:
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
                self.seen.append(p)
            data = open(p, 'rb').read()
            time.sleep(0.02 * (data[0] % 4))
            with self.lock:
                self.live -= 1
            if data[0] % 7 == 0:
                out.append((p, [], 'cv2.imread returned None'))
                continue
            out.append((p, [{'bbox': [float(data[0]), 1.0, 2.0, 3.0],
                             'embedding': np.full(4, data[0], dtype=np.float32),
                             'det_score': 0.9, 'angle': float(data[0] % 3)}], None))
        return out


class LoopClient:
    def __init__(self, no_paths=False, no_probe=False, bad_proto=False):
        self.no_paths, self.no_probe, self.bad_proto = no_paths, no_probe, bad_proto
        self.calls, self.lock = [], threading.Lock()

    def record(self, op, name):
        pass

    def request(self, path, req, timeout=0, retries=0, concurrent=False):
        if (path == worker_protocol.PATH_DETECT_FACES_PATHS and self.no_paths) or \
           (path == worker_protocol.PATH_PROBE_DATA_ROOT and self.no_probe):
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        with self.lock:
            self.calls.append(path)
        out = worker_protocol.unpack(worker_server.HANDLERS[path](
            path, worker_protocol.pack(req), None, None, None, None))
        if self.bad_proto and path == worker_protocol.PATH_DETECT_FACES_PATHS:
            out['proto'] = 99
        return out


@pytest.fixture
def lib(tmp_path, monkeypatch):
    fake = FakeFaces()
    monkeypatch.setattr(worker_server.models, '_face_detector', fake)
    root = tmp_path / 'lib'
    (root / '.media').mkdir(parents=True)
    (root / 'a').mkdir()
    paths = []
    for i in range(20):
        p = root / 'a' / f'{i}.jpg'
        p.write_bytes(bytes([i + 1]) + b'img')
        paths.append(str(p))
    return root, paths, fake


def _norm(res):
    return [(p, [(f['bbox'], f['embedding'].tolist(), f['det_score'], f['angle']) for f in faces], err)
            for p, faces, err in res]


def test_path_mode_matches_upload_mode_and_keeps_order(lib):
    root, paths, fake = lib
    c1, c2 = LoopClient(), LoopClient()
    via_paths = worker_client.RemoteFaceDetector(c1, data_root=str(root)).detect_faces(paths)
    via_upload = worker_client.RemoteFaceDetector(c2, data_root=None).detect_faces(paths)
    assert _norm(via_paths) == _norm(via_upload)
    assert [r[0] for r in via_paths] == paths                      # input order
    assert c1.calls.count(worker_protocol.PATH_DETECT_FACES_PATHS) == 3   # 20 files / chunks of 8
    assert worker_protocol.PATH_DETECT_FACES not in c1.calls       # no bytes uploaded
    assert c2.calls.count(worker_protocol.PATH_DETECT_FACES) == 20
    assert any(r[2] for r in via_paths)                             # per-item errors preserved
    assert fake.peak >= 2                                           # requests really overlapped
    assert fake.gpu_lock_arg is compute.GPU_LOCK                    # lock is passed down, not held here


def test_outside_root_files_upload_and_traversal_is_rejected(lib, tmp_path):
    root, paths, fake = lib
    outside = tmp_path / 'elsewhere.jpg'
    outside.write_bytes(b'\x05zz')
    c = LoopClient()
    res = worker_client.RemoteFaceDetector(c, data_root=str(root)).detect_faces([paths[0], str(outside)])
    assert [r[0] for r in res] == [paths[0], str(outside)] and res[1][2] is None
    assert worker_protocol.PATH_DETECT_FACES in c.calls and worker_protocol.PATH_DETECT_FACES_PATHS in c.calls

    fake.seen.clear()
    reply = worker_protocol.unpack(worker_server.handle_detect_faces_paths(
        'p', worker_protocol.pack({'data_root': str(root), 'paths': ['../elsewhere.jpg', '/etc/passwd', 'a/1.jpg']}),
        None, None, None, None))
    errs = [r['error'] for r in reply['results']]
    assert errs[0].startswith('rejected') and errs[1].startswith('rejected') and errs[2] is None
    assert [os.path.basename(p) for p in fake.seen] == ['1.jpg']   # rejected files never opened
    bad = worker_protocol.unpack(worker_server.handle_detect_faces_paths(
        'p', worker_protocol.pack({'data_root': str(root.parent), 'paths': ['x']}), None, None, None, None))
    assert bad['error'] and bad['results'] == []                    # no .media -> whole request refused


@pytest.mark.parametrize('kw', [{'no_paths': True}, {'no_probe': True}])
def test_old_worker_falls_back_to_upload(lib, kw):
    root, paths, _fake = lib
    ref = worker_client.RemoteFaceDetector(LoopClient(), data_root=None).detect_faces(paths)
    c = LoopClient(**kw)
    got = worker_client.RemoteFaceDetector(c, data_root=str(root)).detect_faces(paths)
    assert _norm(got) == _norm(ref)
    assert c.calls.count(worker_protocol.PATH_DETECT_FACES) == 20


def test_protocol_mismatch_is_a_clear_error(lib):
    root, paths, _ = lib
    with pytest.raises(worker_client.WorkerError, match='needs redeploy'):
        worker_client.RemoteFaceDetector(LoopClient(bad_proto=True), data_root=str(root)).detect_faces(paths[:3])


def test_unreadable_upload_is_per_item_error():
    res = worker_client.RemoteFaceDetector(LoopClient()).detect_faces(['/nope/missing.jpg'])
    assert res[0][1] == [] and 'cannot read file' in res[0][2]


# ------------------------------------------------------------------ batched DB writes

class DetForDb:
    """Deterministic faces per file name; `batch_size` exercises multi-batch + read-ahead."""
    _model_name = 'fake'

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.threads = set()

    @staticmethod
    def model_id(_n):
        return 'fake'

    def detect_faces(self, paths):
        self.threads.add(threading.current_thread().name)
        out = []
        for p in paths:
            k = int(os.path.basename(p).split('.')[0])
            if k == 3:
                out.append((p, [], 'cv2.imread returned None'))
                continue
            faces = []
            for j in range(k % 3):  # 0, 1 or 2 faces
                v = np.zeros(8, np.float32)
                v[(k + j) % 8] = 1.0
                faces.append({'bbox': [float(j), 0.0, 10.0 + k, 10.0], 'embedding': v,
                              'det_score': 0.9 - 0.1 * j, 'angle': 0.0})
            out.append((p, faces, None))
        return out


def _make_app(tmp_path, tag, n=10):
    root = tmp_path / tag
    (root / '.media').mkdir(parents=True)
    for i in range(n):
        Image.new('RGB', (16, 16), (i * 20, i, 0)).save(root / f'{i}.jpg')
    app = create_app(str(root))
    db, _errs, manual = app.state.dbs
    for i in range(n):
        data = (root / f'{i}.jpg').read_bytes()
        db.upsert_file_path(f'{i}.jpg', xxhash.xxh64(data).hexdigest(), size=len(data), modified_time=1)
    db.conn.commit()
    # Auto-match: faces whose embedding peaks at slot 1 or 2 are "Alice"; one pair is negated.
    manual.find_matching_identity = lambda emb, threshold=None: (
        ('Alice', 0.99) if np.frombuffer(emb, np.float32)[[1, 2]].max() > 0.5 else (None, None))
    manual.is_face_negated = lambda fid, name: False
    return root, app, db, manual


def _snapshot(db, manual):
    rows = db.conn.execute(
        "SELECT f.path, fa.bbox, fa.det_score, fa.identity, fa.handled, length(fa.embedding) "
        "FROM faces fa JOIN files_with_path f ON f.id = fa.file_id ORDER BY f.path, fa.bbox").fetchall()
    promoted = sorted(tuple(r) for r in manual.conn.execute(
        "SELECT checksum, identity FROM faces").fetchall()) if hasattr(manual, 'conn') else []
    return [tuple(r) for r in rows], promoted


def _oracle(root, db, manual, detector, threshold=None):
    """The pre-optimisation per-file loop, verbatim, as the reference behaviour."""
    import json
    matched = 0
    for fid, rel in [(r['id'], r['path']) for r in db.conn.execute(
            'SELECT id, path FROM files_with_path ORDER BY id')]:
        _p, faces, error = detector.detect_faces([str(root / rel)])[0]
        if error:
            continue
        db.insert_faces(fid, faces, 'fake')
        row = db.get_file_by_id(fid)
        for face_row in db.get_faces_for_file(fid):
            emb = db.get_face_embedding(face_row['id'])
            if not emb:
                continue
            name, _s = manual.find_matching_identity(emb, threshold=threshold)
            if name is None or manual.is_face_negated(face_row['id'], name):
                continue
            manual.promote_auto_face(face_row['id'], row['checksum'], json.loads(face_row['bbox']),
                                     emb, name, None, None)
            db.mark_faces_handled([face_row['id']])
            matched += 1
    return matched


def _wait(client):
    for _ in range(600):
        s = client.get('/api/detect-faces/status').json()
        if not s['running']:
            return s
        time.sleep(0.05)
    raise AssertionError('job did not finish')


@pytest.mark.parametrize('batch_size', [1, 3, 8, 50])
def test_batched_write_equals_per_file_loop(tmp_path, monkeypatch, batch_size):
    root_a, app_a, db_a, man_a = _make_app(tmp_path, 'new')
    root_b, _app_b, db_b, man_b = _make_app(tmp_path, 'old')
    det = DetForDb(batch_size)
    monkeypatch.setattr(web_mod, '_get_face_detector', lambda: det)
    client = TestClient(app_a)
    assert client.post('/api/detect-faces/start').json()['started'] is True
    s = _wait(client)
    assert s['error'] is None and s['done'] == 10 and s['matched'] > 0, s
    expect_matched = _oracle(root_b, db_b, man_b, DetForDb(1))
    assert s['matched'] == expect_matched
    snap_a, snap_b = _snapshot(db_a, man_a), _snapshot(db_b, man_b)
    # indexed_at / ids differ run to run; the decisions (paths, boxes, identities, handled) must not.
    assert snap_a == snap_b
    assert any(r[4] for r in snap_a[0])      # some faces were marked handled
    if batch_size < 10:
        assert det.threads and not any(t == 'MainThread' for t in det.threads)   # read-ahead thread


def test_batched_gpu_fatal_keeps_earlier_files_written(tmp_path, monkeypatch):
    _root, app, db, manual = _make_app(tmp_path, 'fatal')

    class Fatal(DetForDb):
        def detect_faces(self, paths):
            out = super().detect_faces(paths)
            return [(p, [], OCL) if os.path.basename(p) == '5.jpg' else (p, f, e) for p, f, e in out]
    det = Fatal(4)
    monkeypatch.setattr(web_mod, '_get_face_detector', lambda: det)
    client = TestClient(app)
    client.post('/api/detect-faces/start')
    s = _wait(client)
    assert s['error'].startswith('The GPU ran out of memory or was lost'), s
    assert s['done'] == 5                                   # status counts 0-4; the fatal 5.jpg is not reported done
    paths = {r[0] for r in db.conn.execute(
        "SELECT f.path FROM faces fa JOIN files_with_path f ON f.id = fa.file_id").fetchall()}
    assert {f'{i}.jpg' for i in (0, 1, 2, 4)} <= paths      # files before the fatal one were written
    assert '5.jpg' not in paths and '6.jpg' not in paths
