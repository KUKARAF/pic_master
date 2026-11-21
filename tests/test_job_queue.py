import pytest
"""Tests for media_manager.job_queue (fake clock / probes / starter; synchronous tick)."""
import time
from media_manager.job_queue import (DEFAULT_WEIGHT, JOB_WEIGHTS, JobQueue, JobWeight,
                                     NvidiaSmiProbe, ResourceSampler, weight_for)
from media_manager import job_queue
from media_manager.job_queue import IntelXpuProbe, LocalGpuProbe


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make_sampler(clock, gpu=None, cpu=None, ram=(16000.0, 12000.0), **kw):
    """Sampler whose readers return mutable values from a dict."""
    state = {'gpu': gpu, 'cpu': cpu, 'ram': ram, 'jiff': [0, 0]}

    def cpu_reader():
        if state['cpu'] is None:
            return None
        state['jiff'][0] += 100
        state['jiff'][1] += 100 - state['cpu']
        return tuple(state['jiff'])

    s = ResourceSampler(window_s=600, clock=clock, cpu_reader=cpu_reader,
                        mem_reader=lambda: state['ram'],
                        gpu_probe=lambda: ({'util_pct': state['gpu']} if state['gpu'] is not None else None),
                        **kw)
    return s, state


def feed(s, clock, n=5, step=5):
    for _ in range(n):
        clock.advance(step)
        s.sample()


class Harness:
    def __init__(self, catalog=None, gpu=None, cpu=None, ram=(16000.0, 12000.0), **limits):
        self.clock = Clock()
        self.sampler, self.state = make_sampler(self.clock, gpu, cpu, ram)
        self.started, self.cancelled = [], []
        self.status = {}          # item.key -> status dict
        self.start_resp = {}      # item.key -> start response
        self.q = JobQueue(self.starter, lambda it: dict(self.status.get(it.key, {'running': True, 'done': 0, 'total': 10})),
                          canceller=self.cancel, sampler=self.sampler, clock=self.clock,
                          catalog=catalog or {}, **limits)

    def starter(self, it):
        self.started.append(it.key)
        return self.start_resp.get(it.key, {'started': True, 'total': 10})

    def cancel(self, it):
        self.cancelled.append(it.key)
        return True

    def add(self, key, **w):
        self.q._catalog[key] = JobWeight(**w)
        return self.q.enqueue(key, key, key, '/s/' + key)

    def finish(self, key, **st):
        self.status[key] = dict({'running': False, 'done': 10, 'total': 10}, **st)

    def refresh(self, n=3):
        feed(self.sampler, self.clock, n)


# ---- sampler ---------------------------------------------------------------

def test_sampler_cpu_gpu_ram_average():
    c = Clock(); s, st = make_sampler(c, gpu=50, cpu=20)
    c.advance(5); s.sample()            # first sample: no cpu delta yet
    st['gpu'] = 70; feed(s, c, 1)
    a = s.averages()
    assert a['cpu_pct'] == 20.0 and a['gpu_pct'] == 60.0
    assert a['ram_free_mb'] == 12000.0 and a['ram_total_mb'] == 16000.0
    assert a['gpu_source'] == 'local' and a['samples'] == 2 and a['window_s'] == 600


def test_sampler_window_drops_old_samples():
    c = Clock(); s, st = make_sampler(c, gpu=100)
    feed(s, c, 3)
    st['gpu'] = 0
    c.advance(700); s.sample()
    assert s.averages()['gpu_pct'] == 0.0 and s.averages()['samples'] == 1


def test_sampler_unknown_everything():
    c = Clock(); s, _ = make_sampler(c, ram=None)
    s.mem_reader = None
    feed(s, c, 2)
    a = s.averages()
    assert a['gpu_pct'] is None and a['cpu_pct'] is None and a['ram_free_mb'] is None
    assert a['gpu_source'] is None


def test_sampler_job_gpu_fallback_and_probe_priority():
    c = Clock(); s, st = make_sampler(c)          # no GPU probe result
    feed(s, c, 2)
    assert s.averages()['gpu_pct'] is None
    s.note_job_gpu(40); s.note_job_gpu(60); s.note_job_gpu(None)
    a = s.averages()
    assert a['gpu_pct'] == 50.0 and a['gpu_source'] == 'job'
    st['gpu'] = 10; feed(s, c, 1)                 # a real probe wins
    assert s.averages()['gpu_pct'] == 10.0 and s.averages()['gpu_source'] == 'local'


def test_worker_probe_precedes_local_and_errors_are_swallowed():
    c = Clock(); s, _ = make_sampler(c, gpu=5)
    s.worker_probe = lambda: {'util_pct': 77, 'vram_used_mb': 1, 'vram_total_mb': 2}
    feed(s, c, 1)
    assert s.averages()['gpu_pct'] == 77.0 and s.averages()['gpu_source'] == 'worker'
    s.worker_probe = lambda: (_ for _ in ()).throw(RuntimeError('boom'))
    c.advance(700); feed(s, c, 1)
    assert s.averages()['gpu_pct'] == 5.0 and s.averages()['gpu_source'] == 'local'


def test_nvidia_smi_probe_parse_and_cached_failure():
    c = Clock(); calls = []

    def runner():
        calls.append(1)
        raise FileNotFoundError('nvidia-smi')
    p = NvidiaSmiProbe(clock=c, runner=runner, retry_after_s=100)
    assert p() is None and p() is None and len(calls) == 1   # cached
    c.advance(101); p()
    assert len(calls) == 2
    ok = NvidiaSmiProbe(runner=lambda: '35, 1000, 8000\n')
    assert ok() == {'util_pct': 35.0, 'vram_used_mb': 1000.0, 'vram_total_mb': 8000.0}
    assert NvidiaSmiProbe(runner=lambda: 'garbage')() is None


# ---- catalog ---------------------------------------------------------------

def test_catalog_lookup():
    assert weight_for('/api/siglip-dino/start').gpu >= 50
    assert weight_for('/api/siglip-dino/start?x=1') is JOB_WEIGHTS['/api/siglip-dino/start']
    assert weight_for('http://h:1/api/imdb/rebuild').ram_mb >= 3000
    assert weight_for('/api/nope/start') is DEFAULT_WEIGHT
    assert JOB_WEIGHTS['/api/body-index/start'].exclusive_group == \
        JOB_WEIGHTS['/api/link-faces-to-bodies/start'].exclusive_group


# ---- scheduler -------------------------------------------------------------

def test_idle_always_starts_even_when_box_is_loaded():
    h = Harness(gpu=99, cpu=99, ram=(16000.0, 100.0))
    h.refresh()
    h.add('big', gpu=95, cpu=95, ram_mb=9000)
    h.q.tick()
    assert h.started == ['big']


def test_gpu_projection_blocks_then_allows():
    h = Harness(gpu=20)
    h.refresh()
    h.add('a', gpu=50); b = h.add('b', gpu=50)       # 50+50 > 90
    h.q.tick()
    assert h.started == ['a'] and b.state == 'queued'
    assert 'GPU' in b.blocked_reason and 'avg over 10 min' in b.blocked_reason
    h.add('c', gpu=30)                               # 50+30 <= 90
    h.q.tick()
    assert 'c' in h.started


def test_measured_gpu_average_counts():
    h = Harness(gpu=80)                              # measured high, declared low
    h.refresh()
    h.add('a', gpu=5); b = h.add('b', gpu=20)        # max(80, 5)+20 > 90
    h.q.tick()
    assert h.started == ['a'] and b.state == 'queued'


def test_cpu_limit_and_light_jobs_bypass_gpu_check():
    h = Harness(gpu=95, cpu=10)
    h.refresh()
    h.add('a', cpu=70); h.add('cpuheavy', cpu=30)    # 70+30 > 85
    h.add('light', gpu=0, cpu=5)                     # gpu avg 95 but weight 0 -> ok
    h.q.tick()
    assert h.started == ['a', 'light']
    assert 'CPU' in h.q._find(h.q._items[1].id).blocked_reason


def test_ram_floor():
    h = Harness(ram=(16000.0, 3000.0))
    h.refresh()
    h.add('a', ram_mb=100); b = h.add('b', ram_mb=2000)   # 3000-2000 < 1500
    h.q.tick()
    assert h.started == ['a'] and 'RAM' in b.blocked_reason
    h.state['ram'] = (16000.0, 9000.0)
    h.clock.advance(700); h.refresh()                # old low samples age out
    h.q.tick()
    assert 'b' in h.started


def test_max_parallel_cap():
    h = Harness(max_parallel=2)
    for k in 'abc':
        h.add(k)
    h.q.tick()
    assert h.started == ['a', 'b']
    assert 'max 2' in h.q._items[2].blocked_reason


def test_exclusive_group():
    h = Harness()
    h.add('a', exclusive_group='yolo'); b = h.add('b', exclusive_group='yolo')
    h.add('c', exclusive_group='other')
    h.q.tick()
    assert h.started == ['a', 'c'] and 'yolo' in b.blocked_reason
    h.finish('a')
    h.q.tick()                                       # a done -> b admitted same tick
    assert h.started == ['a', 'c', 'b']


def test_no_head_of_line_blocking_and_starvation_guard():
    h = Harness(max_skips=2, gpu=0)
    h.add('run', gpu=60)
    h.q.tick()                                       # running
    heavy = h.add('heavy', gpu=60)                   # blocked (60+60>90)
    for i in range(2):
        h.add(f'l{i}', gpu=10)
    h.q.tick()
    assert h.started[1:] == ['l0', 'l1'] and heavy.skips == 2
    h.add('l2', gpu=10)
    h.q.tick()                                       # heavy starving: nobody else may start
    assert 'l2' not in h.started and heavy.state == 'queued'
    h.finish('run'); h.finish('l0'); h.finish('l1')
    h.q.tick()
    assert 'heavy' in h.started


def test_started_false_marks_error_with_message():
    h = Harness()
    h.start_resp['a'] = {'started': False, 'message': 'already running'}
    a = h.add('a'); h.add('b')
    h.q.tick()
    assert a.state == 'error' and a.message == 'already running'
    assert h.started == ['a', 'b']                   # next one proceeds in the same tick


def test_zero_total_is_nothing_to_do():
    h = Harness()
    h.start_resp['a'] = {'started': True, 'total': 0}
    h.status['a'] = {'running': False, 'done': 0, 'total': 0}
    a = h.add('a')
    h.q.tick(); h.q.tick()        # start, then the first poll decides
    assert a.state == 'done' and a.message == 'nothing to do'


def test_zero_total_but_running_keeps_running():
    h = Harness()
    h.start_resp['a'] = {'started': True, 'total': 0}
    h.status['a'] = {'running': True, 'done': 0, 'total': 0}
    a = h.add('a')
    h.q.tick()
    assert a.state == 'running'


def test_status_error_and_done_and_gpu_feedback():
    h = Harness()
    a, b = h.add('a'), h.add('b')
    h.q.tick()
    h.status['a'] = {'running': True, 'done': 3, 'total': 10, 'gpu_busy_pct': 42, 'device': 'cuda'}
    h.status['b'] = {'running': False, 'error': 'boom', 'total': 5}
    h.q.tick()
    assert a.state == 'running' and a.message == '3/10 · cuda'
    assert h.sampler.averages()['gpu_pct'] == 42.0
    assert b.state == 'error' and b.message == 'boom'
    h.finish('a')
    h.q.tick()
    assert a.state == 'done' and a.finished_at is not None


def test_starter_exception_is_error():
    h = Harness()
    def bad(it): raise RuntimeError('net down')
    h.q._starter = bad
    a = h.add('a')
    h.q.tick()
    assert a.state == 'error' and 'net down' in a.message


def test_status_failures_keep_slot_then_error_after_window():
    h = Harness(status_fail_window_s=300, max_parallel=1)
    a = h.add('a', exclusive_group='g'); b = h.add('b')
    h.q.tick()
    h.q._status_fn = lambda it: (_ for _ in ()).throw(OSError('x'))
    for _ in range(6):                      # many failed polls, well inside the window
        h.clock.advance(20); h.q.tick()
    assert a.state == 'running' and a.message == 'status unavailable'
    assert b.state == 'queued'              # slot still occupied
    # backoff: a poll right after a failure is skipped
    n = a.status_failures
    h.q.tick(); assert a.status_failures == n
    h.clock.advance(400); h.q.tick()        # past the window -> error, slot freed
    assert a.state == 'error' and 'x' in a.message
    h.q.tick(); assert b.state == 'running'


def test_status_recovers_after_failures():
    h = Harness(status_fail_window_s=300)
    a = h.add('a'); h.q.tick()
    good = h.q._status_fn
    h.q._status_fn = lambda it: (_ for _ in ()).throw(OSError('x'))
    h.clock.advance(10); h.q.tick()
    assert a.message == 'status unavailable'
    h.q._status_fn = good
    h.clock.advance(100); h.q.tick()
    assert a.state == 'running' and a.status_failures == 0 and a.message != 'status unavailable'


def test_cancel_queued():
    h = Harness(max_parallel=1)
    h.add('a'); b = h.add('b')
    h.q.tick()
    assert h.q.cancel(b.id) and b.state == 'cancelled'
    assert h.q.cancel(b.id) is False and h.cancelled == []
    h.finish('a'); h.q.tick()
    assert 'b' not in h.started


def test_cancel_running_then_confirm_when_status_stops():
    h = Harness()
    a = h.add('a'); h.q.tick()
    assert h.q.cancel(a.id) and h.cancelled == ['a']
    snap = h.q.snapshot()['items'][0]
    assert snap['state'] == 'cancelling' and snap['cancelling'] and not snap['cancellable']
    h.q.tick(); assert a.state == 'running'         # job still winding down
    h.finish('a', cancelled=True)
    h.q.tick()
    assert a.state == 'cancelled'
    assert h.q.cancel('nope') is False


def test_cancel_refused_or_no_canceller():
    h = Harness(); a = h.add('a'); h.q.tick()
    h.q._canceller = lambda it: False
    assert h.q.cancel(a.id) is False and not a.cancelling
    h.q._canceller = None
    assert h.q.cancel(a.id) is False
    assert h.q.snapshot()['items'][0]['cancellable'] is False


def test_clear_finished_and_cap():
    h = Harness(max_finished=3)
    for i in range(6):
        h.add(f'j{i}'); h.q.tick(); h.finish(f'j{i}'); h.q.tick()
        h.clock.advance(1)
    states = [i['state'] for i in h.q.snapshot()['items']]
    assert states == ['done'] * 3
    h.add('live'); h.q.tick()
    assert h.q.clear_finished() == 3
    assert [i['key'] for i in h.q.snapshot()['items']] == ['live']


def test_snapshot_shape_and_limits():
    h = Harness(gpu=10)
    h.refresh()
    a = h.add('a', gpu=50, cpu=10, ram_mb=7)
    h.q.tick()
    snap = h.q.snapshot()
    it = snap['items'][0]
    for k in ('id', 'key', 'label', 'start_url', 'status_url', 'state', 'enqueued_at', 'started_at',
              'finished_at', 'done', 'total', 'message', 'weights', 'blocked_reason', 'cancellable'):
        assert k in it
    assert it['weights'] == {'gpu': 50, 'cpu': 10, 'ram_mb': 7} and len(it['id']) == 8
    assert snap['resources']['gpu_pct'] == 10.0
    assert snap['limits']['gpu_limit'] == 90.0
    assert h.q.set_limits(gpu_limit=50)['gpu_limit'] == 50.0
    try:
        h.q.set_limits(bogus=1); assert False
    except KeyError:
        pass
    import json; json.dumps(snap)


def test_enqueue_by_start_url_uses_catalog_weight():
    q = JobQueue(lambda it: {}, lambda it: {}, sampler=ResourceSampler(gpu_probe=lambda: None))
    it = q.enqueue('/api/siglip-dino/start', 'Embed', '/api/siglip-dino/start', '/api/siglip-dino/status')
    assert it.weight is JOB_WEIGHTS['/api/siglip-dino/start']


# -- review-fix regressions ---------------------------------------------------

import threading


def test_cancelling_ending_in_error_is_error_and_normal_finish_is_done():
    h = Harness()
    a, b, c = h.add('a'), h.add('b'), h.add('c'); h.q.tick()
    for x in (a, b, c):
        assert h.q.cancel(x.id)
    h.finish('a', error='boom')                       # error without cancelled marker
    h.finish('b')                                     # ran to completion, no marker
    h.finish('c', done=3, total=10)                   # stopped short
    h.q.tick()
    assert (a.state, a.message) == ('error', 'boom')
    assert b.state == 'done' and c.state == 'cancelled'


def test_finished_items_are_cancellable_and_removable():
    h = Harness()
    a = h.add('a'); h.q.tick(); h.finish('a'); h.q.tick()
    assert h.q.snapshot()['items'][0]['cancellable'] is True
    assert h.q.remove(a.id) is True and h.q.snapshot()['items'] == []
    assert h.q.remove(a.id) is False


def test_slow_starter_does_not_block_snapshot_cancel_enqueue_and_no_double_dispatch():
    h = Harness()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow(it):
        calls.append(it.key); entered.set(); release.wait(5)
        return {'started': True, 'total': 10}
    h.q._starter = slow
    a = h.add('a')
    t = threading.Thread(target=h.q.tick); t.start()
    assert entered.wait(3)
    snap = h.q.snapshot()['items'][0]                 # must not block
    assert snap['state'] == 'running' and snap['message'] == 'starting…'
    h.add('b')                                        # enqueue does not block
    t2 = threading.Thread(target=h.q.tick); t2.start()  # overlapping tick waits on _tick_lock
    time.sleep(0.2)
    assert calls == ['a']
    assert h.q.cancel(a.id) is True                   # cancel of a starting item
    assert h.q.snapshot()['items'][0]['state'] == 'cancelling'
    release.set(); t.join(3); t2.join(3)
    assert h.cancelled == ['a'] and a.cancelling and a.state == 'running'
    assert calls.count('a') == 1


def test_slow_status_fn_does_not_block_snapshot():
    h = Harness()
    a = h.add('a'); h.q.tick()
    entered, release = threading.Event(), threading.Event()

    def slow(it):
        entered.set(); release.wait(5); return {'running': True, 'done': 1, 'total': 10}
    h.q._status_fn = slow
    t = threading.Thread(target=h.q.tick); t.start()
    assert entered.wait(3)
    t0 = time.time(); h.q.snapshot(); assert time.time() - t0 < 1
    assert h.q.cancel('nope') is False
    release.set(); t.join(3)
    assert a.message == '1/10'


def test_deferred_start_stays_queued_not_error():
    h = Harness()
    h.q._starter = lambda it: {'started': False, 'deferred': True,
                               'message': 'already running (started manually)'}
    a = h.add('a')
    h.q.tick()
    assert a.state == 'queued' and a.blocked_reason == 'already running (started manually)'
    h.q._starter = lambda it: {'started': True, 'total': 1}
    h.q.tick()
    assert a.state == 'running'


# ---- Intel GPU probe ---------------------------------------------------------

XPU_SMI_JSON = """{"device_id": 0, "memory_physical_size_byte": "33689878528",
 "device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 63.5},
  {"metrics_type": "XPUM_STATS_POWER", "value": 120.0},
  {"metrics_type": "XPUM_STATS_MEMORY_USED", "value": 4096.25}]}"""
XPU_SMI_FLAT = '{"GPU Utilization (%)": 12, "GPU Memory Used (MiB)": 512, "memory_physical_size_byte": 1073741824}'
IGT = """[
{"period": {"duration": 0.0}, "engines": {"Render/3D/0": {"busy": 0.0}}},
{"period": {"duration": 500.1}, "engines": {"Render/3D/0": {"busy": 20.5, "unit": "%"},
  "Compute/0": {"busy": 71.0, "unit": "%"}}}
]"""


def test_parse_xpu_smi_json_layouts_and_misses():
    r = job_queue.parse_xpu_smi_json(XPU_SMI_JSON)
    assert r['util_pct'] == 63.5 and r['vram_used_mb'] == 4096.25
    assert r['vram_total_mb'] == pytest.approx(33689878528 / 2 ** 20)
    f = job_queue.parse_xpu_smi_json(XPU_SMI_FLAT)
    assert f == {'util_pct': 12.0, 'vram_used_mb': 512.0, 'vram_total_mb': 1024.0}
    for bad in ('', 'nope', '{}', '[]', '{"device_level": [{"metrics_type": "X", "value": 1}]}', None):
        assert job_queue.parse_xpu_smi_json(bad) is None


def test_parse_intel_gpu_top_uses_last_sample_busiest_engine():
    assert job_queue.parse_intel_gpu_top(IGT) == {'util_pct': 71.0, 'vram_used_mb': None,
                                                  'vram_total_mb': None}
    assert job_queue.parse_intel_gpu_top('{"engines": {"a": {"busy": 5}}}\n{"engines": {"a": {"busy": 9}}}')['util_pct'] == 9.0
    for bad in ('', 'junk', '[{"x": 1}]', '{"engines": {}}'):
        assert job_queue.parse_intel_gpu_top(bad) is None


def test_intel_probe_xpu_smi_then_igt_fallback_and_cached_failure():
    c = Clock(); seen = []

    def runner(cmd):
        seen.append(cmd[0])
        if cmd[0] == 'xpu-smi':
            raise FileNotFoundError('xpu-smi')
        return IGT
    p = IntelXpuProbe(clock=c, runner=runner)
    r = p()
    assert r['util_pct'] == 71.0 and r['vendor'] == 'intel' and seen == ['xpu-smi', 'intel_gpu_top']

    seen.clear()
    ok = IntelXpuProbe(runner=lambda cmd: (seen.append(cmd) or XPU_SMI_JSON))
    r = ok()
    assert r['util_pct'] == 63.5 and r['vendor'] == 'intel'
    assert seen == [['xpu-smi', 'stats', '-d', '0', '-j']]        # igt not touched

    calls = []
    def dead(cmd):
        calls.append(cmd[0]); raise FileNotFoundError(cmd[0])
    p = IntelXpuProbe(clock=c, runner=dead, retry_after_s=100)
    assert p() is None and p() is None and len(calls) == 2         # both tried once, then cached
    c.advance(101); p()
    assert len(calls) == 4
    assert IntelXpuProbe(runner=lambda cmd: 'garbage')() is None


def test_local_probe_chain_and_default_sampler():
    nv = lambda: None
    intel = lambda: {'util_pct': 33.0, 'vram_used_mb': None, 'vram_total_mb': None, 'vendor': 'intel'}
    assert LocalGpuProbe([nv, intel])()['util_pct'] == 33.0
    assert LocalGpuProbe([lambda: {'util_pct': 1.0}, intel])()['util_pct'] == 1.0
    assert LocalGpuProbe([nv, nv])() is None
    assert [type(p) for p in ResourceSampler().gpu_probe.probes] == [NvidiaSmiProbe, IntelXpuProbe]


def test_face_groups_weight_is_gpu_heavy_and_shares_embed_group():
    w = weight_for('/api/face-groups/compute/start')
    assert w.gpu >= 80 and w.cpu <= 20 and w.ram_mb >= 2000 and w.exclusive_group == 'embed'
    assert w.exclusive_group == weight_for('/api/siglip-dino/start').exclusive_group
