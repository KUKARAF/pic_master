"""Worker `stats` op (GPU/CPU/RAM snapshot) and the never-raising client fetch_stats."""
import subprocess

import pytest

from media_manager import worker_client, worker_protocol, worker_server


def test_parse_single_gpu():
    assert worker_server._parse_nvidia_smi("37, 2048, 24576\n") == {
        "util_pct": 37.0, "vram_used_mb": 2048.0, "vram_total_mb": 24576.0}


def test_parse_multi_gpu_picks_busiest_consistently():
    out = "10, 100, 8000\n90, 7000, 8000\n"
    assert worker_server._parse_nvidia_smi(out) == {
        "util_pct": 90.0, "vram_used_mb": 7000.0, "vram_total_mb": 8000.0}


@pytest.mark.parametrize("txt", ["", "garbage", "[N/A], [N/A], [N/A]\n", None])
def test_parse_bad_output_is_none(txt):
    assert worker_server._parse_nvidia_smi(txt) is None


def test_read_gpu_failure_paths(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(worker_server.subprocess, "run", missing)
    assert worker_server._read_gpu() is None

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 2)
    monkeypatch.setattr(worker_server.subprocess, "run", timeout)
    assert worker_server._read_gpu() is None

    monkeypatch.setattr(worker_server.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "err"))
    assert worker_server._read_gpu() is None

    monkeypatch.setattr(worker_server.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "5, 1, 2\n", ""))
    assert worker_server._read_gpu()["util_pct"] == 5.0


def _fake_run(outputs):
    """subprocess.run stand-in: outputs maps argv[0] -> stdout (absent -> FileNotFoundError)."""
    def run(cmd, *a, **k):
        if cmd[0] not in outputs:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, outputs[cmd[0]], "")
    return run


def test_read_gpu_nvidia_has_vendor(monkeypatch):
    monkeypatch.setattr(worker_server.subprocess, "run", _fake_run({"nvidia-smi": "5, 1, 2\n"}))
    assert worker_server._read_gpu() == {"util_pct": 5.0, "vram_used_mb": 1.0,
                                         "vram_total_mb": 2.0, "vendor": "nvidia"}


def test_read_gpu_intel_xpu_smi_when_no_nvidia(monkeypatch):
    monkeypatch.setattr(worker_server, "_intel_probe", None)
    out = ('{"memory_physical_size_byte": "34359738368", "device_level": ['
           '{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 88},'
           '{"metrics_type": "XPUM_STATS_MEMORY_USED", "value": 2048}]}')
    monkeypatch.setattr(worker_server.subprocess, "run", _fake_run({"xpu-smi": out}))
    assert worker_server._read_gpu() == {"util_pct": 88.0, "vram_used_mb": 2048.0,
                                         "vram_total_mb": 32768.0, "vendor": "intel"}


def test_read_gpu_intel_gpu_top_fallback_and_cached_failure(monkeypatch):
    monkeypatch.setattr(worker_server, "_intel_probe", None)
    igt = '[{"engines": {"Compute/0": {"busy": 0}}}, {"engines": {"Compute/0": {"busy": 42.0}}}]'
    monkeypatch.setattr(worker_server.subprocess, "run", _fake_run({"intel_gpu_top": igt}))
    g = worker_server._read_gpu()
    assert g["util_pct"] == 42.0 and g["vendor"] == "intel" and g["vram_total_mb"] is None
    # nothing installed at all: None, and the Intel probe caches the failure
    monkeypatch.setattr(worker_server, "_intel_probe", None)
    calls = []
    def none_run(cmd, *a, **k):
        calls.append(cmd[0]); raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(worker_server.subprocess, "run", none_run)
    assert worker_server._read_gpu() is None
    n = len(calls)
    assert worker_server._read_gpu() is None
    assert calls[n:] == ["nvidia-smi"]            # xpu-smi / intel_gpu_top not re-forked


def test_cpu_pct_delta(monkeypatch):
    monkeypatch.setattr(worker_server, "_cpu_prev", None)
    samples = iter([(100, 1000), (150, 1100), (150, 1100)])
    monkeypatch.setattr(worker_server, "_read_proc_stat", lambda: next(samples))
    assert worker_server._read_cpu_pct() is None          # first call
    assert worker_server._read_cpu_pct() == 50.0          # 50 busy / 100 total
    assert worker_server._read_cpu_pct() is None          # zero interval


def test_proc_stat_and_meminfo_parsing(monkeypatch, tmp_path):
    stat = tmp_path / "stat"
    stat.write_text("cpu  10 0 10 70 10 0 0 0 0 0\ncpu0 1 2 3 4\n")
    mem = tmp_path / "meminfo"
    mem.write_text("MemTotal:       16384 kB\nMemFree: 1 kB\nMemAvailable:    8192 kB\n")
    real_open = open

    def fake_open(path, *a, **k):
        return real_open({"/proc/stat": stat, "/proc/meminfo": mem}.get(str(path), path), *a, **k)
    monkeypatch.setattr("builtins.open", fake_open)
    assert worker_server._read_proc_stat() == (20, 100)   # idle=70+10 iowait
    assert worker_server._read_meminfo() == (16.0, 8.0)


def test_proc_readers_missing_files(monkeypatch):
    def boom(*a, **k):
        raise OSError("nope")
    monkeypatch.setattr("builtins.open", boom)
    assert worker_server._read_proc_stat() is None
    assert worker_server._read_meminfo() == (None, None)


def test_handler_shape_and_registration(monkeypatch):
    monkeypatch.setattr(worker_server, "_read_gpu", lambda: None)
    monkeypatch.setattr(worker_server, "_read_cpu_pct", lambda: 12.5)
    monkeypatch.setattr(worker_server, "_read_meminfo", lambda: (100.0, 40.0))
    assert worker_server.HANDLERS[worker_protocol.PATH_STATS] is worker_server.handle_stats
    r = worker_protocol.unpack(worker_server.handle_stats("stats", b"", None, None, None, None))
    assert r == {"ok": True, "gpu": None, "cpu_pct": 12.5, "ram_total_mb": 100.0,
                 "ram_available_mb": 40.0, "error": None}


class _FakeClient:
    def __init__(self, configured=True, resp=None, exc=None):
        self.configured, self.resp, self.exc, self.kw = configured, resp, exc, None

    def is_configured(self):
        return self.configured

    def request(self, path, req, **kw):
        self.kw = kw
        if self.exc:
            raise self.exc
        return self.resp


def test_fetch_stats_ok():
    c = _FakeClient(resp={"ok": True, "gpu": {"util_pct": 1.0}, "cpu_pct": 2.0,
                          "ram_total_mb": 3.0, "ram_available_mb": 4.0, "error": None})
    assert worker_client.fetch_stats(c) == {"gpu": {"util_pct": 1.0}, "cpu_pct": 2.0,
                                            "ram_total_mb": 3.0, "ram_available_mb": 4.0}
    assert c.kw["concurrent"] is True and c.kw["retries"] == 0 and c.kw["timeout"] == 3


@pytest.mark.parametrize("client", [
    None,
    _FakeClient(configured=False),
    _FakeClient(exc=worker_client.WorkerUnavailable("down")),
    _FakeClient(exc=worker_client.WorkerError("worker has no handler for 'stats' (HTTP 404)")),
    _FakeClient(exc=RuntimeError("boom")),
    _FakeClient(resp={"ok": False, "error": "x"}),
    _FakeClient(resp={"error": "unknown op"}),
    _FakeClient(resp=None),
])
def test_fetch_stats_none_paths(client):
    assert worker_client.fetch_stats(client) is None


def test_fetch_stats_old_worker_404_through_real_client(monkeypatch):
    c = worker_client.WorkerClient("/nonexistent")
    monkeypatch.setattr(c, "is_configured", lambda: True)
    monkeypatch.setattr(c, "address", lambda: "http://x:1")

    class R:
        status_code = 404
        text = "unknown op"
        content = b"unknown op"

    class S:
        def post(self, *a, **k):
            return R()
    c._session = S()
    assert worker_client.fetch_stats(c) is None


def test_ensure_session_does_not_take_lock_when_session_exists():
    """A held request lock (long in-flight request) must not block a concurrent caller
    once the session exists."""
    import threading
    c = worker_client.WorkerClient("/nonexistent")
    first = c._ensure_session()
    out = []
    with c._lock:                      # simulate a long non-concurrent request
        t = threading.Thread(target=lambda: out.append(c._ensure_session()))
        t.start()
        t.join(2)
        assert not t.is_alive(), "blocked on the request lock"
    assert out == [first]
