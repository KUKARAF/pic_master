"""Lightweight wall-time accounting for the face-detection pipeline (torch/cv2-free).

The worker times every stage of every image (decode, PCN on CPU, the InsightFace full-image
SCRFD pass, the extra recognition/landmark ``app.get`` calls made by the PCN correct/recover
arms and the quarter-turn backstop, and time spent waiting for the GPU lock / a PCN slot) and
returns the per-request sums in its reply as an OPTIONAL ``timings`` field. The host merges
them per job and turns them into the ``/api/detect-faces/status`` diagnostics.

All durations are THREAD-seconds summed over images, so stage percentages are shares of the
time images spent in the worker (not of job wall time, which concurrency makes smaller).
"""
import threading
import time

TIMING_KEYS = ('images', 'decode_s', 'pcn_s', 'pcn_wait_s', 'scrfd_s', 'rec_s',
               'lock_wait_s', 'app_get_calls', 'rec_calls', 'total_s')


def tadd(timings, key, value):
    """timings[key] += value, tolerating a missing key and a None accumulator."""
    if timings is not None:
        timings[key] = timings.get(key, 0) + value


class FaceTimings:
    """Thread-safe running sums of the TIMING_KEYS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d = dict.fromkeys(TIMING_KEYS, 0)

    def add(self, other):
        if not other:
            return
        with self._lock:
            for k in TIMING_KEYS:
                v = other.get(k)
                if isinstance(v, (int, float)):
                    self._d[k] += v

    def reset(self):
        with self._lock:
            self._d = dict.fromkeys(TIMING_KEYS, 0)

    def snapshot(self):
        with self._lock:
            return dict(self._d)


class FlightTracker:
    """How full the host's request pipeline is. While any detect_faces() call is active,
    integrates the time spent with fewer than `cap` requests in flight (an under-filled
    pipeline starves the worker no matter how fast it is)."""

    def __init__(self, cap):
        self.cap = max(1, int(cap))
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._calls = 0
            self._inflight = 0
            self._t = time.perf_counter()
            self._active_s = 0.0
            self._under_s = 0.0
            self._flight_area = 0.0

    def _tick(self):  # caller holds the lock
        now = time.perf_counter()
        if self._calls > 0:
            dt = now - self._t
            self._active_s += dt
            self._flight_area += dt * self._inflight
            if self._inflight < self.cap:
                self._under_s += dt
        self._t = now

    def call(self, delta):
        with self._lock:
            self._tick()
            self._calls += delta

    def request(self, delta):
        with self._lock:
            self._tick()
            self._inflight += delta

    def snapshot(self):
        with self._lock:
            self._tick()
            return {'active_s': self._active_s, 'under_s': self._under_s,
                    'flight_area': self._flight_area, 'cap': self.cap}


def stage_percent(snap):
    """{'decode','pcn','scrfd','rec','gpu_wait','other'} -> % of the time images spent in the
    worker (None when nothing was timed). 'pcn' includes waiting for a PCN slot."""
    total = float(snap.get('total_s') or 0.0)
    if total <= 0:
        return None
    parts = {
        'decode': snap.get('decode_s', 0.0),
        'pcn': snap.get('pcn_s', 0.0) + snap.get('pcn_wait_s', 0.0),
        'scrfd': snap.get('scrfd_s', 0.0),
        'rec': snap.get('rec_s', 0.0),
        'gpu_wait': snap.get('lock_wait_s', 0.0),
    }
    out = {k: round(100.0 * v / total, 1) for k, v in parts.items()}
    out['other'] = round(max(0.0, 100.0 - sum(out.values())), 1)
    return out


def summary_line(snap, wall_s=None):
    """One human line for the worker log."""
    n = int(snap.get('images') or 0)
    if not n:
        return None
    pct = stage_percent(snap) or {}
    rate = f", {n / wall_s:.2f} img/s" if wall_s and wall_s > 0 else ''
    return (f"[worker] faces: {n} imgs{rate}, {snap.get('app_get_calls', 0) / n:.2f} app.get/img "
            f"({snap.get('rec_calls', 0) / n:.2f} rec), "
            + ' '.join(f"{k} {v:.0f}%" for k, v in pct.items())
            + f", {snap.get('total_s', 0.0) / n * 1000:.0f} ms/img")


def job_stats(snap, flight, wall_s, db_s, done, total, ema):
    """The status fields for the host job from the aggregated worker timings, the flight
    tracker snapshot (or None), job wall time and cumulative DB-write time."""
    wall = max(wall_s, 1e-9)
    n = int(snap.get('images') or 0)
    gpu_s = snap.get('scrfd_s', 0.0) + snap.get('rec_s', 0.0)
    out = {
        'img_per_sec': round(ema, 2) if ema else 0.0,
        'eta_seconds': round((total - done) / ema, 1) if ema and ema > 0 else None,
        'db_write_pct': round(100 * min(db_s / wall, 1.0), 1),
        'gpu_busy_pct': round(100 * min(gpu_s / wall, 1.0), 1) if gpu_s else None,
        'stage_pct': stage_percent(snap),
        'app_get_per_img': round(snap.get('app_get_calls', 0) / n, 2) if n else None,
        'ms_per_img': round(snap.get('total_s', 0.0) / n * 1000, 1) if n else None,
        'flight_under_pct': None, 'inflight_avg': None, 'inflight_cap': None,
    }
    if flight and flight.get('active_s', 0) > 0:
        out['flight_under_pct'] = round(100 * flight['under_s'] / flight['active_s'], 1)
        out['inflight_avg'] = round(flight['flight_area'] / flight['active_s'], 2)
        out['inflight_cap'] = flight['cap']
    return out


JOB_STAT_DEFAULTS = {
    'img_per_sec': 0.0, 'eta_seconds': None, 'transport_mode': None, 'gpu_busy_pct': None,
    'db_write_pct': None, 'stage_pct': None, 'app_get_per_img': None, 'ms_per_img': None,
    'flight_under_pct': None, 'inflight_avg': None, 'inflight_cap': None,
}
