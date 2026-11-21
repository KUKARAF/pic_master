"""Smart job queue for the /bulk page (pure core: no FastAPI / web.py imports).

Many bulk jobs can be queued. A scheduler decides, from the AVERAGE GPU load, CPU
load and available RAM over the last N minutes (default 10) plus a per-job resource
weight estimate, whether the next queued job may start concurrently with the ones
already running.

Pieces
  * ResourceSampler  - background sampler (/proc/stat, /proc/meminfo, pluggable GPU probe).
  * JOB_WEIGHTS      - catalog of start-URL path -> JobWeight (expected load + exclusive group).
  * JobQueue         - thread-safe scheduler; HTTP is delegated to injected callables so the
                       web layer owns the real POST/GET and the core is unit-testable.

Everything time-related is injectable (clock, sample functions) and the scheduler
exposes a synchronous `tick()`; the background threads are thin loops around
`sampler.sample()` / `queue.tick()`.
"""
from __future__ import annotations

import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------

def read_cpu_times() -> Optional[tuple[int, int]]:
    """(total_jiffies, idle_jiffies) from the aggregate 'cpu' line of /proc/stat."""
    try:
        with open('/proc/stat') as f:
            parts = f.readline().split()
        if parts[0] != 'cpu':
            return None
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        return sum(vals[:8]), idle
    except (OSError, ValueError, IndexError):
        return None


def read_meminfo() -> Optional[tuple[float, float]]:
    """(total_mb, available_mb) from /proc/meminfo (MemAvailable)."""
    try:
        kb = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, _, rest = line.partition(':')
                if k in ('MemTotal', 'MemAvailable'):
                    kb[k] = float(rest.split()[0])
                if len(kb) == 2:
                    break
        if len(kb) < 2:
            return None
        return kb['MemTotal'] / 1024.0, kb['MemAvailable'] / 1024.0
    except (OSError, ValueError, IndexError):
        return None


class NvidiaSmiProbe:
    """Default LOCAL GPU probe via nvidia-smi. A failure (binary missing, timeout, bad
    output) is cached for `retry_after_s` so a GPU-less host does not fork every sample."""

    def __init__(self, timeout_s: float = 2.0, retry_after_s: float = 120.0,
                 clock: Callable[[], float] = time.monotonic,
                 runner: Optional[Callable[[], str]] = None):
        self.timeout_s = timeout_s
        self.retry_after_s = retry_after_s
        self._clock = clock
        self._runner = runner or self._run
        self._failed_at: Optional[float] = None

    def _run(self) -> str:
        return subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=self.timeout_s, check=True).stdout

    def __call__(self) -> Optional[dict]:
        now = self._clock()
        if self._failed_at is not None and now - self._failed_at < self.retry_after_s:
            return None
        try:
            lines = [ln for ln in self._runner().strip().splitlines() if ln.strip()]
            rows = [[float(x) for x in ln.split(',')] for ln in lines]
            if not rows:
                raise ValueError('empty')
            # Several GPUs: report the busiest one's utilization, summed memory.
            self._failed_at = None
            return {'util_pct': max(r[0] for r in rows),
                    'vram_used_mb': sum(r[1] for r in rows),
                    'vram_total_mb': sum(r[2] for r in rows)}
        except Exception:
            self._failed_at = now
            return None


class ResourceSampler:
    """Samples CPU / RAM / GPU every `interval_s` into a rolling `window_s` window.

    GPU sources, tried in order each sample: `worker_probe` (the web layer fills this
    with a call to the remote ML worker's stats - the GPU is usually REMOTE), then
    `gpu_probe` (default: local nvidia-smi). Fallback when neither yields anything:
    running jobs that report `gpu_busy_pct` call `note_job_gpu(pct)`.
    """

    def __init__(self, window_s: float = 600.0, interval_s: float = 5.0,
                 gpu_probe: Optional[Callable[[], Optional[dict]]] = None,
                 worker_probe: Optional[Callable[[], Optional[dict]]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 cpu_reader: Callable[[], Optional[tuple[int, int]]] = read_cpu_times,
                 mem_reader: Callable[[], Optional[tuple[float, float]]] = read_meminfo):
        self.window_s = float(window_s)
        self.interval_s = float(interval_s)
        self.gpu_probe = gpu_probe if gpu_probe is not None else NvidiaSmiProbe()
        self.worker_probe = worker_probe  # slot, assigned later by the web layer
        self._clock = clock
        self._cpu_reader = cpu_reader
        self._mem_reader = mem_reader
        self._lock = threading.Lock()
        # (t, cpu_pct|None, ram_free_mb|None, ram_total_mb|None, gpu_pct|None, gpu_source|None)
        self._samples: deque = deque()
        self._job_gpu: deque = deque()  # (t, pct)
        self._prev_cpu: Optional[tuple[int, int]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- sampling -----------------------------------------------------------
    def _read_cpu_pct(self) -> Optional[float]:
        cur = self._cpu_reader()
        prev, self._prev_cpu = self._prev_cpu, cur
        if cur is None or prev is None:
            return None
        dt, di = cur[0] - prev[0], cur[1] - prev[1]
        if dt <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (dt - di) / dt))

    def _read_gpu(self) -> tuple[Optional[float], Optional[str]]:
        for src, probe in (('worker', self.worker_probe), ('local', self.gpu_probe)):
            if probe is None:
                continue
            try:
                r = probe()
            except Exception:
                r = None
            if r and r.get('util_pct') is not None:
                return float(r['util_pct']), src
        return None, None

    def sample(self) -> None:
        """Take one sample now (the daemon thread calls this every interval)."""
        now = self._clock()
        cpu = self._read_cpu_pct()
        mem = self._mem_reader()
        gpu, src = self._read_gpu()
        with self._lock:
            self._samples.append((now, cpu, mem[1] if mem else None,
                                  mem[0] if mem else None, gpu, src))
            self._trim(now)

    def note_job_gpu(self, pct: Optional[float]) -> None:
        """A running job reported `gpu_busy_pct`; kept as a fallback GPU signal."""
        if pct is None:
            return
        now = self._clock()
        with self._lock:
            self._job_gpu.append((now, float(pct)))
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        while self._job_gpu and self._job_gpu[0][0] < cutoff:
            self._job_gpu.popleft()

    @staticmethod
    def _avg(vals) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def averages(self) -> dict:
        now = self._clock()
        with self._lock:
            self._trim(now)
            s = list(self._samples)
            jg = [p for _, p in self._job_gpu]
        gpu = self._avg(x[4] for x in s)
        gpu_source = None
        if gpu is not None:
            srcs = [x[5] for x in s if x[5]]
            gpu_source = max(set(srcs), key=srcs.count)
        elif jg:
            gpu, gpu_source = self._avg(jg), 'job'
        r = lambda v: None if v is None else round(v, 1)  # noqa: E731
        return {'cpu_pct': r(self._avg(x[1] for x in s)),
                'gpu_pct': r(gpu),
                'ram_free_mb': r(self._avg(x[2] for x in s)),
                'ram_total_mb': r(self._avg(x[3] for x in s)),
                'samples': len(s), 'window_s': self.window_s, 'gpu_source': gpu_source}

    # -- thread -------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                try:
                    self.sample()
                except Exception:
                    pass
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=_loop, name='resource-sampler', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Job weight catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobWeight:
    """Expected load of ONE running job. gpu/cpu are %-points (0-100) of the whole
    resource (cpu: of all cores); ram_mb is absolute host RAM. `exclusive_group`: two
    jobs of the same group are never run together."""
    gpu: float = 0.0
    cpu: float = 0.0
    ram_mb: float = 0.0
    exclusive_group: Optional[str] = None


# Unknown start paths: assume a moderately heavy job so we err on the side of serial.
DEFAULT_WEIGHT = JobWeight(gpu=40, cpu=40, ram_mb=1500)

# Reasoning (estimates; the heavy compute runs on the REMOTE worker, the host mostly
# decodes images + writes the DB, so host cpu stays modest while gpu is the worker's):
#  * Torch embedding / detection jobs (siglip-dino, faces, body, region, pattern-dino,
#    video face scan) saturate the worker GPU: gpu 55-70. Two of them together would
#    thrash VRAM / the worker's request pipeline, and with the 90% limit only one fits
#    next to light jobs.
#  * 'yolo' group: body-index and the YOLO-World tag stage share ONE detector singleton
#    whose vocab is re-set per batch (web.py "no GPU/model locks between bulk jobs ...
#    'tags' also collides with the body index"), so body-index never runs alongside
#    another user of that detector. link-faces-to-bodies reads body rows -> same group.
#  * 'faces' group: detect-faces, scan-video-faces, estimate-age and faces/renormalize
#    all write face rows / use the face models and the worker GPU-lock; match-faces
#    reads them. Keeping them serial avoids reading half-written embeddings.
#  * 'czkawka' group: near-dup and broken scan are CPU + disk heavy external processes;
#    two czkawka runs would only fight for the disk.
#  * 'decode' group: capture-frames runs ffmpeg and writes still files that the video
#    face scan and indexers consume - serial with scan-video-faces.
#  * imdb/rebuild streams the whole SigLIP+face matrices to the worker over RNS: RAM and
#    network heavy, tiny GPU; its own group (the worker's in-memory index swap).
#  * metadata / cities / matching jobs are light (EXIF reads, HTTP, numpy dot products).
JOB_WEIGHTS: dict[str, JobWeight] = {
    '/api/siglip-dino/start':        JobWeight(gpu=70, cpu=30, ram_mb=2000, exclusive_group='embed'),
    '/api/pattern-dino-index/start': JobWeight(gpu=60, cpu=25, ram_mb=1500, exclusive_group='embed'),
    '/api/pattern-index/start':      JobWeight(gpu=15, cpu=45, ram_mb=1200),
    '/api/region-index/start':       JobWeight(gpu=55, cpu=30, ram_mb=1500, exclusive_group='embed'),
    '/api/detect-faces/start':       JobWeight(gpu=60, cpu=35, ram_mb=2000, exclusive_group='faces'),
    '/api/scan-video-faces/start':   JobWeight(gpu=55, cpu=40, ram_mb=2000, exclusive_group='faces'),
    '/api/estimate-age/start':       JobWeight(gpu=35, cpu=20, ram_mb=1000, exclusive_group='faces'),
    '/api/faces/renormalize/start':  JobWeight(gpu=0,  cpu=20, ram_mb=1500, exclusive_group='faces'),
    '/api/body-index/start':         JobWeight(gpu=55, cpu=30, ram_mb=2000, exclusive_group='yolo'),
    '/api/link-faces-to-bodies/start': JobWeight(gpu=0, cpu=15, ram_mb=500, exclusive_group='yolo'),
    '/api/match-faces/start':        JobWeight(gpu=0,  cpu=40, ram_mb=1500),
    '/api/capture-frames/start':     JobWeight(gpu=0,  cpu=55, ram_mb=800, exclusive_group='decode'),
    '/api/near-dup/compute/start':   JobWeight(gpu=0,  cpu=75, ram_mb=3000, exclusive_group='czkawka'),
    '/api/broken/scan/start':        JobWeight(gpu=0,  cpu=65, ram_mb=1500, exclusive_group='czkawka'),
    '/api/imdb/rebuild':             JobWeight(gpu=5,  cpu=20, ram_mb=3500, exclusive_group='imdb'),
    '/api/face-groups/compute/start': JobWeight(gpu=40, cpu=60, ram_mb=2500, exclusive_group='facegroups'),
    '/api/metadata/start':           JobWeight(gpu=0,  cpu=20, ram_mb=300),
    '/api/fetch-cities/start':       JobWeight(gpu=0,  cpu=5,  ram_mb=100),
    '/api/match-cities/start':       JobWeight(gpu=0,  cpu=10, ram_mb=200),
}


def weight_for(start_url: str, catalog: Optional[dict] = None) -> JobWeight:
    """Look up by URL path (query string / host stripped); unknown -> DEFAULT_WEIGHT."""
    path = start_url.split('?', 1)[0]
    if '://' in path:
        path = '/' + path.split('://', 1)[1].split('/', 1)[-1]
    return (catalog if catalog is not None else JOB_WEIGHTS).get(path, DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

QUEUED, STARTING, RUNNING, DONE, ERROR, CANCELLED = (
    'queued', 'starting', 'running', 'done', 'error', 'cancelled')
FINISHED_STATES = (DONE, ERROR, CANCELLED)
CANCELLING = 'cancelling'   # snapshot-only state: a running/starting item with the cancel flag set
STATUS_UNAVAILABLE = 'status unavailable'

DEFAULT_LIMITS = {
    'gpu_limit': 90.0,        # projected GPU % must stay <= this
    'cpu_limit': 85.0,        # projected CPU % must stay <= this
    'ram_floor_mb': 1500.0,   # available RAM after the candidate must stay >= this
    'max_parallel': 3,        # hard cap on concurrently running jobs
    'max_skips': 5,           # starvation guard: skipped this often -> becomes the head
    'max_finished': 50,       # finished items kept for display
    'status_fail_window_s': 300.0,  # status polls failing this long -> a running item errors
}


@dataclass
class QueueItem:
    id: str
    key: str
    label: str
    start_url: str
    status_url: str
    weight: JobWeight
    state: str = QUEUED
    enqueued_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: dict = field(default_factory=dict)   # last status snapshot
    message: str = ''
    blocked_reason: Optional[str] = None
    skips: int = 0
    cancelling: bool = False
    cancel_pending: bool = False       # cancel asked while STARTING; honored when the start returns
    cancel_inflight: bool = False      # a canceller call is in progress (outside the lock)
    status_failures: int = 0
    status_first_failure: Optional[float] = None
    next_poll_at: float = 0.0          # status-poll backoff


class JobQueue:
    """Thread-safe scheduler.

    Locking: `_lock` (RLock) guards the item list/fields only and is NEVER held while
    calling starter / status_fn / canceller (they may be slow: candidate scans, worker
    connections), so snapshot()/cancel()/enqueue() never block on a job call.
    `_tick_lock` serializes tick() so two ticks never overlap.
    """

    def __init__(self,
                 starter: Callable[[QueueItem], dict],
                 status_fn: Callable[[QueueItem], dict],
                 canceller: Optional[Callable[[QueueItem], bool]] = None,
                 sampler: Optional[ResourceSampler] = None,
                 clock: Callable[[], float] = time.time,
                 catalog: Optional[dict] = None,
                 tick_interval_s: float = 3.0,
                 **limits):
        self._starter, self._status_fn, self._canceller = starter, status_fn, canceller
        self.sampler = sampler or ResourceSampler()
        self._clock = clock
        self._catalog = catalog
        self.tick_interval_s = tick_interval_s
        self._lock = threading.RLock()   # state only; never held across starter/status/canceller
        self._tick_lock = threading.Lock()
        self.limits = dict(DEFAULT_LIMITS)
        self.set_limits(**limits)
        self._items: list[QueueItem] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API ---------------------------------------------------------
    def set_limits(self, **kw) -> dict:
        with self._lock:
            for k, v in kw.items():
                if k not in DEFAULT_LIMITS:
                    raise KeyError(f'unknown limit {k!r}')
                if v is not None:
                    self.limits[k] = type(DEFAULT_LIMITS[k])(v)
            return dict(self.limits)

    def enqueue(self, key_or_start_url: str, label: str,
                start_url: Optional[str] = None, status_url: str = '') -> QueueItem:
        """Queue a job. If `start_url` is omitted, `key_or_start_url` is the start URL."""
        start_url = start_url or key_or_start_url
        item = QueueItem(id=uuid.uuid4().hex[:8], key=key_or_start_url, label=label,
                         start_url=start_url, status_url=status_url,
                         weight=weight_for(start_url, self._catalog),
                         enqueued_at=self._clock())
        with self._lock:
            self._items.append(item)
        self.kick()
        return item

    def cancel(self, item_id: str) -> bool:
        with self._lock:
            it = self._find(item_id)
            if it is None:
                return False
            if it.state == QUEUED:
                self._finish(it, CANCELLED, 'cancelled before start')
                return True
            if it.state == STARTING:
                # The start call is in flight (outside the lock): honor the flag when it returns.
                if it.cancel_pending or self._canceller is None:
                    return False
                it.cancel_pending = True
                it.message = 'cancelling…'
                return True
            if (it.state != RUNNING or it.cancelling or it.cancel_inflight
                    or self._canceller is None):
                return False
            it.cancel_inflight = True
        ok, err = False, None
        try:
            ok = bool(self._canceller(it))
        except Exception as e:
            err = e
        with self._lock:
            it.cancel_inflight = False
            if err is not None:
                it.message = f'cancel failed: {err}'
                return False
            if not ok:
                it.message = 'cancel refused'
                return False
            if it.state == RUNNING:
                it.cancelling = True     # becomes CANCELLED once the status stops running
                it.message = 'cancelling…'
        self.kick()
        return True

    def remove(self, item_id: str) -> bool:
        """Remove ONE finished item (queued/running ones are cancelled instead)."""
        with self._lock:
            it = self._find(item_id)
            if it is None or it.state not in FINISHED_STATES:
                return False
            self._items = [i for i in self._items if i is not it]
            return True

    def clear_finished(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items = [i for i in self._items if i.state not in FINISHED_STATES]
            return n - len(self._items)

    def snapshot(self) -> dict:
        with self._lock:
            items = [self._item_json(i) for i in self._items]
            limits = dict(self.limits)
        return {'items': items, 'resources': self.sampler.averages(), 'limits': limits}

    def kick(self) -> None:
        self._wake.set()

    # -- tick ---------------------------------------------------------------
    def tick(self) -> None:
        """One synchronous scheduling pass: poll running jobs, then admit queued ones.
        Job calls happen outside `_lock`; `_tick_lock` keeps passes from overlapping."""
        with self._tick_lock:
            self._poll_running()
            self._dispatch()
            with self._lock:
                self._trim_finished()

    def _poll_running(self) -> None:
        with self._lock:
            running = [i for i in self._items if i.state == RUNNING]
        for it in running:
            now = self._clock()
            if it.next_poll_at > now:
                continue                      # backing off after failed polls
            try:
                st = self._status_fn(it) or {}
                err = None
            except Exception as e:
                st, err = {}, e
            with self._lock:
                if it.state != RUNNING:
                    continue
                if err is not None:
                    self._poll_failed(it, err)
                else:
                    self._apply_status(it, st)

    def _poll_failed(self, it: QueueItem, err: Exception) -> None:
        """Keep the slot (the job may well still run); mark it, back off, and only
        error after `status_fail_window_s` of continuous failure."""
        now = self._clock()
        it.status_failures += 1
        if it.status_first_failure is None:
            it.status_first_failure = now
        if now - it.status_first_failure >= self.limits['status_fail_window_s']:
            self._finish(it, ERROR, f'status poll failed for '
                                    f"{self.limits['status_fail_window_s']:.0f}s: {err}")
            return
        it.message = STATUS_UNAVAILABLE
        it.next_poll_at = now + min(60.0, self.tick_interval_s * 2 ** min(it.status_failures, 5))

    def _apply_status(self, it: QueueItem, st: dict) -> None:
        it.status_failures, it.status_first_failure, it.next_poll_at = 0, None, 0.0
        it.status = st
        if st.get('gpu_busy_pct') is not None:
            self.sampler.note_job_gpu(st['gpu_busy_pct'])
        if st.get('running'):
            it.message = 'cancelling…' if it.cancelling else self._progress_msg(st)
            return
        # Not running any more.
        # Cancel outcome (documented choice): an error the job reports WITHOUT its
        # 'cancelled' marker wins over 'cancelled'; otherwise a cancelling item that
        # stopped short (or is marked cancelled) is CANCELLED, and one that ran to
        # completion without the marker is DONE.
        done_all = bool(st.get('total')) and (st.get('done') or 0) >= st['total']
        if it.cancelling:
            if st.get('error') and not st.get('cancelled'):
                self._finish(it, ERROR, str(st['error']))
            elif st.get('cancelled') or not done_all:
                self._finish(it, CANCELLED, 'cancelled')
            else:
                self._finish(it, DONE, self._progress_msg(st))
        elif st.get('error'):
            self._finish(it, ERROR, str(st['error']))
        elif not st.get('total'):
            self._finish(it, DONE, 'nothing to do')
        else:
            self._finish(it, DONE, self._progress_msg(st))

    @staticmethod
    def _progress_msg(st: dict) -> str:
        msg = f"{st.get('done') or 0}/{st.get('total') or 0}"
        if st.get('device'):
            msg += f" · {st['device']}"
        return msg

    def _dispatch(self) -> None:
        with self._lock:
            queued = [i for i in self._items if i.state == QUEUED]
            if not queued:
                return
            # Starvation guard: a long-skipped item becomes the only candidate until it starts.
            starving = next((i for i in queued if i.skips >= self.limits['max_skips']), None)
            candidates = [starving] if starving else queued
        for it in candidates:
            with self._lock:
                if it.state != QUEUED:       # cancelled meanwhile
                    continue
                ok, reason = self._admit(it)
                it.blocked_reason = None if ok else reason
                if not ok:
                    continue
                # Claim: STARTING counts as running for admission / exclusive groups and is
                # visible to cancel(), so nothing can double-dispatch it.
                it.state, it.started_at, it.blocked_reason = STARTING, self._clock(), None
                it.message = 'starting…'
            self._start(it)                  # starter runs OUTSIDE the lock
            with self._lock:
                if it.state == QUEUED:       # deferred
                    continue
                # Everything queued ahead of `it` was passed over once more.
                for earlier in queued:
                    if earlier is it:
                        break
                    if earlier.state == QUEUED:
                        earlier.skips += 1

    def _start(self, it: QueueItem) -> None:
        resp, err = {}, None
        try:
            resp = self._starter(it) or {}
        except Exception as e:
            err = e
        do_cancel = False
        with self._lock:
            pending = it.cancel_pending
            it.cancel_pending = False
            if err is not None:
                self._finish(it, CANCELLED if pending else ERROR,
                             'cancelled' if pending else f'start failed: {err}')
                return
            if resp.get('started') is False:
                if resp.get('deferred') and not pending:
                    # Someone else (e.g. a manual Start click) owns the job: stay queued.
                    it.state, it.started_at = QUEUED, None
                    it.message = ''
                    it.blocked_reason = resp.get('message') or 'already running'
                    return
                self._finish(it, CANCELLED if pending else ERROR,
                             'cancelled' if pending else (resp.get('message') or 'refused to start'))
                return
            # started with total 0 may still be a long-running job whose total is unknown
            # yet; the first poll (running flag) decides between "nothing to do" and running.
            it.state, it.status, it.message = RUNNING, resp, 'started'
            do_cancel = pending
        if do_cancel:
            # Cancel was requested while the start was in flight: forward it now.
            with self._lock:
                it.cancel_pending = False
            it.cancel_inflight = True
            try:
                ok = bool(self._canceller(it)) if self._canceller else False
            except Exception:
                ok = False
            with self._lock:
                it.cancel_inflight = False
                if ok and it.state == RUNNING:
                    it.cancelling, it.message = True, 'cancelling…'
                elif not ok:
                    it.message = 'cancel refused'

    # -- admission ----------------------------------------------------------
    def _running(self) -> list[QueueItem]:
        return [i for i in self._items if i.state in (RUNNING, STARTING)]

    def _admit(self, cand: QueueItem) -> tuple[bool, str]:
        running = self._running()
        if not running:
            return True, ''          # never deadlock on a tight box
        lim, w = self.limits, cand.weight
        if len(running) >= lim['max_parallel']:
            return False, f"{len(running)} jobs already running (max {lim['max_parallel']})"
        if w.exclusive_group:
            for r in running:
                if r.weight.exclusive_group == w.exclusive_group:
                    return False, (f"'{r.label}' is running and shares "
                                   f"'{w.exclusive_group}' resources with this job")
        avg = self.sampler.averages()
        win_min = max(1, round(avg['window_s'] / 60))

        def check(name, measured, declared_sum, extra, limit):
            if extra <= 0:
                return None
            base = max(measured or 0.0, declared_sum)
            if base + extra > limit:
                m = f'{measured:.0f}% avg over {win_min} min' if measured is not None \
                    else f'{declared_sum:.0f}% from running jobs'
                return f'{name} {m} + this job ≈ {extra:.0f}% (limit {limit:.0f}%)'
            return None

        reason = (check('GPU', avg['gpu_pct'], sum(r.weight.gpu for r in running),
                        w.gpu, lim['gpu_limit'])
                  or check('CPU', avg['cpu_pct'], sum(r.weight.cpu for r in running),
                           w.cpu, lim['cpu_limit']))
        if reason:
            return False, reason
        if w.ram_mb > 0 and avg['ram_free_mb'] is not None:
            now = self._clock()
            # Running jobs' memory is only partly visible in the window average yet:
            # subtract the not-yet-observed share of each job's declared RAM.
            pending = sum(r.weight.ram_mb * max(0.0, 1 - (now - (r.started_at or now)) / avg['window_s'])
                          for r in running)
            free = avg['ram_free_mb'] - pending
            if free - w.ram_mb < lim['ram_floor_mb']:
                return False, (f"RAM {free:.0f} MB available (avg over {win_min} min) - "
                               f"this job ≈ {w.ram_mb:.0f} MB would leave < {lim['ram_floor_mb']:.0f} MB")
        return True, ''

    # -- helpers ------------------------------------------------------------
    def _find(self, item_id: str) -> Optional[QueueItem]:
        return next((i for i in self._items if i.id == item_id), None)

    def _finish(self, it: QueueItem, state: str, message: str) -> None:
        it.state, it.message, it.finished_at = state, message, self._clock()
        it.blocked_reason, it.cancelling, it.cancel_pending = None, False, False
        self.kick()

    def _trim_finished(self) -> None:
        fin = [i for i in self._items if i.state in FINISHED_STATES]
        extra = len(fin) - self.limits['max_finished']
        if extra > 0:
            drop = {id(i) for i in sorted(fin, key=lambda i: i.finished_at or 0)[:extra]}
            self._items = [i for i in self._items if id(i) not in drop]

    def _item_json(self, i: QueueItem) -> dict:
        w = i.weight
        active = i.state in (RUNNING, STARTING)
        cancelling = active and (i.cancelling or i.cancel_pending)
        if cancelling:
            state = CANCELLING
        elif i.state == STARTING:
            state = RUNNING
        else:
            state = i.state
        can = i.state == QUEUED or i.state in FINISHED_STATES or (
            active and not cancelling and not i.cancel_inflight and self._canceller is not None)
        return {'id': i.id, 'key': i.key, 'label': i.label, 'start_url': i.start_url,
                'status_url': i.status_url, 'state': state,
                'enqueued_at': i.enqueued_at, 'started_at': i.started_at,
                'finished_at': i.finished_at,
                'done': i.status.get('done'), 'total': i.status.get('total'),
                'message': i.message,
                'weights': {'gpu': w.gpu, 'cpu': w.cpu, 'ram_mb': w.ram_mb},
                'blocked_reason': i.blocked_reason,
                'cancelling': cancelling,
                'cancellable': can}

    # -- threads ------------------------------------------------------------
    def start(self) -> None:
        """Start the sampler and the tick loop (idempotent)."""
        self.sampler.start()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    pass
                self._wake.wait(self.tick_interval_s)
                self._wake.clear()

        self._thread = threading.Thread(target=_loop, name='job-queue', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self.sampler.stop()
