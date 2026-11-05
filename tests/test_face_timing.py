"""Per-stage face timing: worker accounting, reply/back-compat, host aggregation, status
fields, PCN slot cap, env knobs. Stubs only -- no GPU, no real models.

Run: python -m pytest tests/test_face_timing.py
"""
import os
import sys
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))

import media_manager.web as web_mod
from media_manager import compute, face_detector, face_timing, worker_client, worker_protocol, worker_server
from media_manager.pcn import pcn_detect

from test_face_pipeline_perf import (IMG, LoopClient, _App, _Face, _PCN, _detector, _make_app,  # noqa: F401
                                     _wait, DetForDb)


class SlowApp(_App):
    def __init__(self, n=1, score=0.95, delay=0.0):
        super().__init__()
        self.n, self.score, self.delay = n, score, delay

    def get(self, img):
        self.calls += 1
        time.sleep(self.delay)
        return [_Face(score=self.score) for _ in range(self.n)]


def _write_img(path, w=32, h=32):
    from PIL import Image
    Image.new('RGB', (w, h), (10, 20, 30)).save(path)


# ------------------------------------------------------------------ worker-side accounting

def test_detect_faces_accounts_stages_and_calls(tmp_path):
    p = tmp_path / 'a.jpg'
    _write_img(p)
    app = SlowApp(delay=0.01)
    det = _detector(app=app)
    det.face_timing = face_timing.FaceTimings()
    lock = threading.Lock()
    t = {}
    res = det.detect_faces([str(p), str(tmp_path / 'x.txt')], gpu_lock=lock, timings=t)
    assert len(res[0][1]) == 1 and res[1][2] == 'unsupported extension'
    assert t['images'] == 1                       # unsupported file is not an image
    assert t['app_get_calls'] == 1 and t.get('rec_calls', 0) == 0
    assert t['scrfd_s'] >= 0.01 and t['decode_s'] > 0 and t['total_s'] >= t['scrfd_s'] + t['decode_s']
    assert det.face_timing.snapshot()['images'] == 1   # also accumulated on the detector


def test_rec_calls_counted_for_warped_and_backstop(tmp_path):
    # weak face (< ROTATION_TTA_TRIGGER) -> quarter-turn sweep: 1 upright + 3 rotated reads
    p = tmp_path / 'a.jpg'
    _write_img(p, 128, 128)
    app = SlowApp(score=0.6)
    det = _detector(app=app)
    t = {}
    det.detect_faces([str(p)], timings=t)
    assert t['app_get_calls'] == app.calls == 5 and t['rec_calls'] == 4
    assert t['rec_s'] > 0 and t['scrfd_s'] > 0


def test_lock_wait_is_separated_from_gpu_time(tmp_path):
    p = tmp_path / 'a.jpg'
    _write_img(p)
    det = _detector(app=SlowApp())
    lock = threading.Lock()
    lock.acquire()
    threading.Timer(0.15, lock.release).start()
    t = {}
    det.detect_faces([str(p)], gpu_lock=lock, timings=t)
    assert t['lock_wait_s'] >= 0.1 and t['scrfd_s'] < 0.1


def test_pcn_time_and_wait_recorded_and_slot_cap(monkeypatch):
    monkeypatch.setenv('MEDIA_PCN_CONCURRENCY', '1')
    monkeypatch.setattr(face_detector.FaceDetector, '_pcn_sem', None)
    peak = {'live': 0, 'max': 0}
    mu = threading.Lock()

    class P(_PCN):
        def detect(self, img):
            with mu:
                peak['live'] += 1
                peak['max'] = max(peak['max'], peak['live'])
            time.sleep(0.05)
            with mu:
                peak['live'] -= 1
            return []
    det = _detector(pcn=P())
    ts = [{} for _ in range(4)]
    th = [threading.Thread(target=det.detect_image, args=(IMG, 1.0, None, t)) for t in ts]
    [x.start() for x in th]
    [x.join() for x in th]
    assert peak['max'] == 1                                   # slot cap serialised the cascades
    assert all(t['pcn_s'] >= 0.05 for t in ts) and sum(t['pcn_wait_s'] for t in ts) >= 0.1
    monkeypatch.setattr(face_detector.FaceDetector, '_pcn_sem', None)


def test_pcn_concurrency_defaults(monkeypatch):
    monkeypatch.delenv('MEDIA_PCN_CONCURRENCY', raising=False)
    monkeypatch.setenv('MEDIA_PCN_THREADS', '4')
    monkeypatch.setattr(os, 'cpu_count', lambda: 32)
    assert pcn_detect.pcn_concurrency() == 8
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)
    assert pcn_detect.pcn_concurrency() == 2                  # never fully serial by default
    monkeypatch.setenv('MEDIA_PCN_CONCURRENCY', '0')
    assert pcn_detect.pcn_concurrency() == 0                  # 0 = unbounded


def test_detection_output_unchanged_by_timing():
    det = _detector()
    with_t = det.detect_image(IMG, timings={})
    without = det.detect_image(IMG)
    assert [(f['bbox'], f['det_score'], f['angle']) for f in with_t] == \
           [(f['bbox'], f['det_score'], f['angle']) for f in without]


# ------------------------------------------------------------------ pure helpers

def test_stage_percent_sums_to_100_and_summary_line():
    snap = {'images': 10, 'decode_s': 1, 'pcn_s': 4, 'pcn_wait_s': 1, 'scrfd_s': 1, 'rec_s': 1,
            'lock_wait_s': 1, 'app_get_calls': 25, 'rec_calls': 15, 'total_s': 10}
    pct = face_timing.stage_percent(snap)
    assert pct['pcn'] == 50.0 and pct['decode'] == 10.0 and pct['other'] == 10.0
    assert abs(sum(pct.values()) - 100.0) < 0.2
    line = face_timing.summary_line(snap, 5.0)
    assert '10 imgs' in line and '2.00 img/s' in line and '2.50 app.get/img' in line
    assert face_timing.stage_percent({}) is None and face_timing.summary_line({}) is None


def test_flight_tracker_under_fill():
    f = face_timing.FlightTracker(2)
    f.call(+1)
    f.request(+1)
    time.sleep(0.1)           # 1 in flight < cap
    f.request(+1)
    time.sleep(0.1)           # 2 in flight == cap
    f.request(-1)
    f.request(-1)
    f.call(-1)
    time.sleep(0.05)          # idle, no active call: not counted
    s = f.snapshot()
    assert 0.18 < s['active_s'] < 0.35 and 0.08 < s['under_s'] < 0.2
    assert 1.2 < s['flight_area'] / s['active_s'] < 1.8


def test_job_stats_math():
    snap = {'images': 4, 'scrfd_s': 1.0, 'rec_s': 1.0, 'app_get_calls': 10, 'total_s': 8.0}
    out = face_timing.job_stats(snap, {'active_s': 10, 'under_s': 3, 'flight_area': 35, 'cap': 4},
                                wall_s=10, db_s=1, done=40, total=100, ema=2.0)
    assert out['eta_seconds'] == 30.0 and out['img_per_sec'] == 2.0
    assert out['gpu_busy_pct'] == 20.0 and out['db_write_pct'] == 10.0
    assert out['app_get_per_img'] == 2.5 and out['flight_under_pct'] == 30.0 and out['inflight_avg'] == 3.5


# ------------------------------------------------------------------ worker reply + host aggregation

class TimedFaces:
    """Worker-side stand-in that fills `timings` like the real detector."""

    def detect_faces(self, paths, gpu_lock=None, timings=None):
        out = []
        for p in paths:
            out.append((p, [{'bbox': [0.0, 0.0, 1.0, 1.0], 'embedding': np.ones(4, np.float32),
                             'det_score': 0.9, 'angle': 0.0}], None))
            for k, v in (('images', 1), ('decode_s', 0.01), ('pcn_s', 0.02), ('scrfd_s', 0.01),
                         ('rec_s', 0.01), ('app_get_calls', 3), ('rec_calls', 2), ('total_s', 0.05)):
                timings[k] = timings.get(k, 0) + v
        return out


def _lib(tmp_path, n=20):
    root = tmp_path / 'lib'
    (root / '.media').mkdir(parents=True)
    paths = []
    for i in range(n):
        p = root / f'{i}.jpg'
        p.write_bytes(bytes([i + 1]) + b'img')
        paths.append(str(p))
    return root, paths


def test_worker_reply_has_timings_and_host_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_server.models, '_face_detector', TimedFaces())
    root, paths = _lib(tmp_path)
    for data_root in (str(root), None):                       # shared-disk and upload transports
        det = worker_client.RemoteFaceDetector(LoopClient(), data_root=data_root)
        res = det.detect_faces(paths)
        assert all(r[2] is None for r in res)
        snap = det.face_timing.snapshot()
        assert snap['images'] == 20 and snap['app_get_calls'] == 60
        assert det.transport_mode == ('shared-disk' if data_root else 'upload')
        fl = det.flight.snapshot()
        assert fl['cap'] == worker_client.FACE_INFLIGHT and fl['active_s'] > 0


def test_old_worker_without_timings_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_server.models, '_face_detector', TimedFaces())
    root, paths = _lib(tmp_path, 4)

    class Old(LoopClient):
        def request(self, *a, **k):
            out = super().request(*a, **k)
            out.pop('timings', None)
            return out
    det = worker_client.RemoteFaceDetector(Old(), data_root=str(root))
    assert len(det.detect_faces(paths)) == 4
    assert det.face_timing.snapshot()['images'] == 0


def test_worker_log_rolls_up_every_n_images(capsys):
    log = worker_server._FaceLog(every=10)
    t = {'images': 4, 'decode_s': 0.1, 'pcn_s': 0.2, 'app_get_calls': 8, 'total_s': 0.4}
    log.note(t, 4, 0)
    log.note(t, 4, 0)
    assert capsys.readouterr().out == ''                       # 8 < 10: nothing printed
    log.note(t, 4, 1)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and '12 imgs' in out[0] and '2.00 app.get/img' in out[0] and '1 errors' in out[0]
    log.note(t, 0, 0)
    assert capsys.readouterr().out == ''                       # window reset


def test_worker_handler_no_longer_logs_per_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(worker_server.models, '_face_detector', TimedFaces())
    monkeypatch.setattr(worker_server, '_face_log', worker_server._FaceLog(every=1000))
    root, paths = _lib(tmp_path, 3)
    worker_client.RemoteFaceDetector(LoopClient(), data_root=str(root)).detect_faces(paths)
    assert 'handled detect_faces' not in capsys.readouterr().out


# ------------------------------------------------------------------ env knobs

def test_env_knobs(monkeypatch):
    import importlib
    monkeypatch.setenv('MEDIA_FACE_INFLIGHT', '6')
    monkeypatch.setenv('MEDIA_FACE_CHUNK', '5')
    monkeypatch.delenv('MEDIA_FACE_BATCH', raising=False)
    monkeypatch.setenv('MEDIA_FACE_PATHS_CHUNK', '9')
    try:
        importlib.reload(worker_client)
        assert (worker_client.FACE_INFLIGHT, worker_client.FACE_PATHS_CHUNK) == (6, 5)
        assert worker_client.RemoteFaceDetector.batch_size == 6 * 5 * 3
    finally:
        monkeypatch.undo()
        importlib.reload(worker_client)
    assert worker_client.FACE_PATHS_CHUNK == 8 and worker_client.RemoteFaceDetector.batch_size == 96


# ------------------------------------------------------------------ web status

class TimedDet(DetForDb):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.face_timing = face_timing.FaceTimings()
        self.flight = face_timing.FlightTracker(4)
        self.transport_mode = 'shared-disk'

    def detect_faces(self, paths):
        self.flight.call(+1)
        self.flight.request(+1)
        time.sleep(0.02)
        out = super().detect_faces(paths)
        self.flight.request(-1)
        self.flight.call(-1)
        self.face_timing.add({'images': len(paths), 'decode_s': 0.01 * len(paths),
                              'pcn_s': 0.03 * len(paths), 'scrfd_s': 0.01 * len(paths),
                              'app_get_calls': 2 * len(paths), 'total_s': 0.06 * len(paths)})
        return out


def test_status_endpoint_exposes_face_diagnostics(tmp_path, monkeypatch):
    _root, app, _db, _man = _make_app(tmp_path, 'st')
    det = TimedDet(3)
    monkeypatch.setattr(web_mod, '_get_face_detector', lambda: det)
    client = TestClient(app)
    idle = client.get('/api/detect-faces/status').json()
    assert idle['stage_pct'] is None and idle['img_per_sec'] == 0.0 and 'transport_mode' in idle
    assert client.post('/api/detect-faces/start').json()['started']
    s = _wait(client)
    assert s['error'] is None and s['done'] == 10
    assert s['img_per_sec'] > 0 and s['eta_seconds'] == 0.0
    assert s['transport_mode'] == 'shared-disk'
    assert s['app_get_per_img'] == 2.0 and s['stage_pct']['pcn'] == 50.0
    assert s['db_write_pct'] is not None and s['gpu_busy_pct'] is not None
    assert s['flight_under_pct'] is not None and s['inflight_cap'] == 4


def test_status_without_diagnostics_still_works(tmp_path, monkeypatch):
    _root, app, _db, _man = _make_app(tmp_path, 'plain')
    monkeypatch.setattr(web_mod, '_get_face_detector', lambda: DetForDb(4))
    client = TestClient(app)
    client.post('/api/detect-faces/start')
    s = _wait(client)
    assert s['error'] is None and s['done'] == 10 and s['stage_pct'] is None and s['img_per_sec'] > 0
