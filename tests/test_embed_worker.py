"""Remote SigLIP2/DINOv2 embedding: client <-> server handlers wired through fakes (no GPU, no models)."""
import threading
import numpy as np
import pytest

from media_manager import worker_client, worker_protocol, worker_server


class FakeEmb:
    def __init__(self, dim, mid):
        self.dim, self.mid = dim, mid

    def model_id(self):
        return self.mid

    def embed_text(self, text):
        v = np.zeros(self.dim, np.float32)
        v[len(text) % self.dim] = 1.0
        return v


def fake_embed_shared(embs, sources, stats=None):
    ok, failed = [], []
    for s in sources:
        if open(s, "rb").read().startswith(b"BAD"):
            failed.append((s, "cannot decode"))
        else:
            ok.append(s)
    outs = []
    for e in embs:
        rows = []
        for s in ok:
            v = np.zeros(e.dim, np.float32)
            v[open(s, "rb").read()[0] % e.dim] = 1.0
            rows.append(v)
        outs.append(np.stack(rows) if rows else np.empty((0,), np.float32))
    return outs, failed


class FakeClient:
    """Routes request() straight into the real server handlers via real pack/unpack."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.stale = False

    def record(self, op, name):
        pass

    def request(self, path, req, timeout=0, retries=0, concurrent=False):
        if self.stale:
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        with self.lock:
            self.calls.append((path, len(req.get("items", []))))
        out = worker_server.HANDLERS[path](path, worker_protocol.pack(req), None, None, None, None)
        return worker_protocol.unpack(out)


@pytest.fixture
def client(monkeypatch):
    m = worker_server.models
    monkeypatch.setattr(m, "_siglip", FakeEmb(8, "sig/fake"))
    monkeypatch.setattr(m, "_dino", FakeEmb(4, "dino/fake"))
    monkeypatch.setattr(worker_server, "_embed_shared", fake_embed_shared)
    return FakeClient()


def test_roundtrip_order_and_batching(client):
    items = [(f"{i}.jpg", bytes([i]) + b"x") for i in range(7)]
    res = worker_client.embed_both(client, items, batch_size=3, workers=2)
    assert [c[1] for c in client.calls] == [3, 3, 1] or sorted(c[1] for c in client.calls) == [1, 3, 3]
    assert res.n_batches == 3 and res.siglip_model == "sig/fake" and res.dino_model == "dino/fake"
    for i in range(7):
        assert res.siglip[i].shape == (8,) and res.siglip[i].dtype == np.float32
        assert res.siglip[i][i % 8] == 1.0 and res.dino[i][i % 4] == 1.0
    assert res.gpu_seconds >= 0 and res.errors == [None] * 7


def test_per_item_error_isolated(client):
    items = [("a.jpg", b"\x01a"), ("b.jpg", b"BADdata"), ("c.jpg", b"\x02c")]
    res = worker_client.embed_both(client, items, batch_size=16)
    assert res.errors[1] == "cannot decode" and res.siglip[1] is None and res.dino[1] is None
    assert res.siglip[0][1] == 1.0 and res.siglip[2][2] == 1.0


def test_unreadable_path_isolated(client, tmp_path):
    good = tmp_path / "g.jpg"
    good.write_bytes(b"\x03z")
    res = worker_client.embed_both(client, [str(tmp_path / "missing.jpg"), str(good)])
    assert "read failed" in res.errors[0] and res.siglip[1] is not None


def test_proxies(client, tmp_path):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    p1.write_bytes(b"\x01a")
    p2.write_bytes(b"BAD")
    sig = worker_client.RemoteSigLIPEmbedder(client)
    dino = worker_client.RemoteDINOEmbedder(client)
    emb, failed = sig.embed_images([str(p1), str(p2)])
    assert emb.shape == (1, 8) and failed == [(str(p2), "cannot decode")]
    assert sig.model_id() == "sig/fake"
    emb, failed = dino.embed_images([str(p1)])
    assert emb.shape == (1, 4) and dino.model_id() == "dino/fake"
    assert sig.embed_text("abc").shape == (8,)
    fresh = worker_client.RemoteDINOEmbedder(client)
    assert fresh.model_id() == "dino/fake"  # probe path
    from PIL import Image
    assert worker_client.RemoteSigLIPEmbedder(client).embed_pil_images(
        [Image.new("RGB", (4, 4))]).shape == (1, 8)


def test_whole_batch_failure_raises(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("oom")
    monkeypatch.setattr(worker_server, "_embed_shared", boom)
    with pytest.raises(worker_client.WorkerError, match="oom"):
        worker_client.embed_both(client, [("a.jpg", b"x")])


def test_stale_worker_message(client):
    client.stale = True
    with pytest.raises(worker_client.WorkerError, match="needs redeploy"):
        worker_client.embed_both(client, [("a.jpg", b"x")])
    with pytest.raises(worker_client.WorkerError, match="needs redeploy"):
        worker_client.RemoteSigLIPEmbedder(client).embed_text("x")
