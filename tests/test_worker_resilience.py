"""Worker-outage resilience: remote trainer retry/restart + web's worker supervisor."""
import json
import os

import pytest

from media_manager import media, remote_tag_trainer as rtt


def WorkerUnavailable(*a):
    # Resolve at call time: other tests reload media_manager.worker_client, and the
    # trainer imports the class lazily, so a module-level import could be a stale class.
    from media_manager.worker_client import WorkerUnavailable as W
    return W(*a)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(rtt, '_sleep', lambda s: None)
    monkeypatch.setattr(rtt, '_POLL_INTERVAL', 0)
    rtt._driver_lines.clear()
    rtt._worker_tail[0] = ''


class FakeClient:
    def __init__(self, script):
        self.script = list(script)  # items: Exception or dict
        self.calls = 0

    def train_status(self, job_id, want_artifact=False):
        self.calls += 1
        item = self.script.pop(0) if self.script else {'status': 'done'}
        if isinstance(item, Exception):
            raise item
        return item


def _status(d):
    return json.load(open(os.path.join(d, 'metadata.json')))


def test_poll_survives_outage_then_finishes(tmp_path, monkeypatch):
    done = []
    monkeypatch.setattr(rtt, '_finalize_done', lambda *a: done.append(a))
    c = FakeClient([WorkerUnavailable('refused')] * 4 + [{'status': 'running'}, {'status': 'done', 'metrics': {}}])
    rtt._poll_until_done(c, 'j', str(tmp_path), 1, 0, 'siglip_model', {})
    assert done
    log = open(tmp_path / 'train.log').read()
    assert 'worker not running' in log and 'reachable again' in log
    assert 'worker_wait' not in _status(str(tmp_path))


def test_poll_gives_up_after_window_with_clear_message(tmp_path, monkeypatch):
    monkeypatch.setenv('MEDIA_TRAIN_WORKER_RECONNECT_SECS', '0')
    c = FakeClient([WorkerUnavailable('refused')] * 3)
    with pytest.raises(RuntimeError, match='worker not running'):
        rtt._poll_until_done(c, 'j', str(tmp_path), 1, 0, 'siglip_model', {})


def test_unknown_job_raises_joblost(tmp_path):
    c = FakeClient([{'status': 'failed', 'error': 'unknown job'}])
    with pytest.raises(rtt.JobLost):
        rtt._poll_until_done(c, 'j', str(tmp_path), 1, 0, 'siglip_model', {})


def test_recovery_restarts_job_after_worker_restart(tmp_path, monkeypatch):
    runs = []

    def drive(*a, **k):
        runs.append(1)
        if len(runs) == 1:
            raise rtt.JobLost('restarted')

    monkeypatch.setattr(rtt, '_drive_clip', drive)
    monkeypatch.setattr(rtt, '_client', lambda root: FakeClient([WorkerUnavailable('x'), {'status': 'failed'}]))
    rtt.train_for_tag(str(tmp_path), 'cat', 'siglip_model')
    assert len(runs) == 2
    assert 'restarting upload' in open(rtt.classifier_dir(str(tmp_path), 'cat', 'siglip_model') + '/train.log').read()


def test_recovery_fails_actionably_when_worker_never_returns(tmp_path, monkeypatch):
    monkeypatch.setenv('MEDIA_TRAIN_WORKER_RECONNECT_SECS', '5')
    monkeypatch.setattr(rtt, '_drive_clip', lambda *a, **k: (_ for _ in ()).throw(WorkerUnavailable('down')))
    monkeypatch.setattr(rtt, '_client', lambda root: FakeClient([WorkerUnavailable('x')] * 100))
    with pytest.raises(RuntimeError, match='worker not running'):
        rtt.train_for_tag(str(tmp_path), 'cat', 'siglip_model')
    out = rtt.classifier_dir(str(tmp_path), 'cat', 'siglip_model')
    assert _status(out)['status'] == 'failed' and 'worker not running' in _status(out)['error']


# ---- supervisor ----

class StubProc:
    _n = 0

    def __init__(self, exit_code=None):
        StubProc._n += 1
        self.pid = 1000 + StubProc._n
        self.code = exit_code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


def _run_sync(sup, max_iters=50):
    """Drive the supervisor loop on this thread; stop it once spawns are exhausted."""
    sup._run()


def test_supervisor_respawns_crashed_worker():
    spawned = []

    def spawn():
        p = StubProc(exit_code=None)
        spawned.append(p)
        sup._stop.set()  # one respawn is enough for the test
        return p

    first = StubProc(exit_code=3)
    ticks = []

    def sleep(s):
        ticks.append(s)

    sup = media._WorkerSupervisor(spawn, first, sleep=sleep)
    sup._run()
    assert len(spawned) == 1 and sup.child is spawned[0]
    sup.stop()
    assert spawned[0].terminated


def test_supervisor_caps_crash_loop():
    sup = media._WorkerSupervisor(lambda: StubProc(exit_code=1), StubProc(exit_code=1),
                                  max_restarts=3, sleep=lambda s: None)
    sup._run()
    assert sup.gave_up and len(sup._restarts) == 3
