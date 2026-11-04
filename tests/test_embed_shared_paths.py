"""Shared-disk embed mode: worker reads <data_root>/<rel> itself. Fakes only (no GPU, no models)."""
import os
import numpy as np
import pytest

from media_manager import worker_client, worker_protocol, worker_server
from test_embed_worker import FakeEmb, fake_embed_shared, FakeClient


class PathsClient(FakeClient):
    def __init__(self, no_paths=False, no_probe=False):
        super().__init__()
        self.no_paths, self.no_probe = no_paths, no_probe

    def request(self, path, req, timeout=0, retries=0, concurrent=False):
        if self.stale:
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        gone = (path == worker_protocol.PATH_EMBED_BOTH_PATHS and self.no_paths) or \
               (path == worker_protocol.PATH_PROBE_DATA_ROOT and self.no_probe)
        if gone:
            raise worker_client.WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        with self.lock:
            self.calls.append((path, len(req.get("items", req.get("paths", [])))))
        out = worker_server.HANDLERS[path](path, worker_protocol.pack(req), None, None, None, None)
        return worker_protocol.unpack(out)

    def paths_of(self, p):
        return [c for c in self.calls if c[0] == p]


@pytest.fixture
def root(tmp_path, monkeypatch):
    m = worker_server.models
    monkeypatch.setattr(m, "_siglip", FakeEmb(8, "sig/fake"))
    monkeypatch.setattr(m, "_dino", FakeEmb(4, "dino/fake"))
    monkeypatch.setattr(worker_server, "_embed_shared", fake_embed_shared)
    r = tmp_path / "lib"
    (r / ".media").mkdir(parents=True)
    (r / "sub").mkdir()
    for i in range(5):
        (r / "sub" / f"{i}.jpg").write_bytes(bytes([i + 1]) + b"x")
    return r


def _call(handler, req):
    return worker_protocol.unpack(handler("p", worker_protocol.pack(req), None, None, None, None))


def _paths(root, rels):
    return _call(worker_server.handle_embed_both_paths, {"data_root": str(root), "paths": rels})


# ---------------- security ----------------

def test_traversal_absolute_and_symlink_rejected(root, tmp_path):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"\x09s")
    (root / "evil.jpg").symlink_to(secret)                      # file symlink out
    (root / "evildir").symlink_to(tmp_path)                     # dir symlink out
    (root / "ok_link.jpg").symlink_to(root / "sub" / "0.jpg")   # symlink staying inside is fine
    rels = ["../secret.jpg", "sub/../../secret.jpg", str(secret), "/etc/passwd", "evil.jpg",
            "evildir/secret.jpg", "", ".", "sub", "sub/\0x.jpg", "sub/0.jpg", "ok_link.jpg",
            "sub/missing.jpg"]
    res = _paths(root, rels)["results"]
    assert [r["error"] is None for r in res] == [False] * 10 + [True, True, False]
    assert all(r["error"].startswith("rejected") for r in res if r["error"])
    assert res[0]["name"] == "../secret.jpg"


def test_rejected_files_never_read(root, tmp_path, monkeypatch):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"\x09s")
    seen = []
    monkeypatch.setattr(worker_server, "_embed_shared",
                        lambda e, src, stats=None: (seen.extend(src), fake_embed_shared(e, src))[1])
    _paths(root, ["../secret.jpg", str(secret), "sub/1.jpg"])
    assert seen == [os.path.realpath(root / "sub" / "1.jpg")]


def test_refused_without_media_dir_or_relative_root(root, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "a.jpg").write_bytes(b"\x01")
    r = _call(worker_server.handle_embed_both_paths, {"data_root": str(bare), "paths": ["a.jpg"]})
    assert r["error"] and r["results"] == []
    (bare / ".media").write_text("a file, not a dir")
    assert _call(worker_server.handle_embed_both_paths,
                 {"data_root": str(bare), "paths": ["a.jpg"]})["error"]
    assert _call(worker_server.handle_embed_both_paths,
                 {"data_root": "lib", "paths": ["a.jpg"]})["error"]
    assert _call(worker_server.handle_embed_both_paths, {"paths": ["a.jpg"]})["error"]


def test_probe(root, tmp_path):
    ok = _call(worker_server.handle_probe_data_root, {"data_root": str(root)})
    assert ok["visible"] is True
    for bad in (str(tmp_path / "nope"), str(tmp_path), "rel", None):
        r = _call(worker_server.handle_probe_data_root, {"data_root": bad})
        assert r["visible"] is False and r["reason"]


# ---------------- worker roundtrip / client ----------------

def test_worker_reply_schema_and_dedupe(root):
    r = _paths(root, ["sub/0.jpg", "ok", "sub/0.jpg"])
    assert r["error"] is None and r["proto"] == worker_protocol.EMBED_PROTOCOL_VERSION
    assert r["siglip_model"] == "sig/fake" and r["dino_model"] == "dino/fake"
    assert "gpu_seconds" in r
    res = r["results"]
    assert res[0]["siglip"] == res[2]["siglip"] and res[0]["siglip"] is not None
    assert res[1]["error"]


def test_client_path_mode_no_bytes(root):
    c = PathsClient()
    items = [str(root / "sub" / f"{i}.jpg") for i in range(5)]
    res = worker_client.embed_both(c, items, batch_size=2, workers=2, data_root=str(root))
    assert res.mode == "paths" and res.errors == [None] * 5
    assert not c.paths_of(worker_protocol.PATH_EMBED_BOTH_BATCH)
    assert c.paths_of(worker_protocol.PATH_EMBED_BOTH_PATHS) == [(worker_protocol.PATH_EMBED_BOTH_PATHS, 5)]  # bs bumped to 64
    for i in range(5):
        assert res.siglip[i].argmax() == (i + 1) % 8 and res.dino[i].argmax() == (i + 1) % 4
    assert res.siglip_model == "sig/fake"
    # probe is cached
    worker_client.embed_both(c, items[:1], data_root=str(root))
    assert len(c.paths_of(worker_protocol.PATH_PROBE_DATA_ROOT)) == 1


def test_matches_upload_results(root):
    items = [str(root / "sub" / f"{i}.jpg") for i in range(5)]
    a = worker_client.embed_both(PathsClient(), items, data_root=str(root))
    b = worker_client.embed_both(PathsClient(), items)
    assert b.mode == "upload"
    for i in range(5):
        assert np.array_equal(a.siglip[i], b.siglip[i]) and np.array_equal(a.dino[i], b.dino[i])


def test_fallback_when_not_visible(root, tmp_path):
    other = tmp_path / "other"
    other.mkdir()  # no .media -> probe says not visible
    items = [str(root / "sub" / "0.jpg")]
    c = PathsClient()
    res = worker_client.embed_both(c, items, data_root=str(other))
    assert res.mode == "upload" and res.errors == [None]
    assert not c.paths_of(worker_protocol.PATH_EMBED_BOTH_PATHS)


def test_fallback_old_worker_no_probe(root):
    c = PathsClient(no_probe=True, no_paths=True)
    res = worker_client.embed_both(c, [str(root / "sub" / "0.jpg")], data_root=str(root))
    assert res.mode == "upload" and res.errors == [None]


def test_fallback_old_worker_probe_ok_but_paths_404(root):
    c = PathsClient(no_paths=True)
    items = [str(root / "sub" / f"{i}.jpg") for i in range(3)]
    res = worker_client.embed_both(c, items, data_root=str(root))
    assert res.errors == [None] * 3 and res.siglip[2] is not None
    assert c.paths_of(worker_protocol.PATH_EMBED_BOTH_BATCH)


def test_base_path_still_demands_redeploy(root):
    c = PathsClient()
    c.stale = True
    with pytest.raises(worker_client.WorkerError, match="needs redeploy"):
        worker_client.embed_both(c, [("a.jpg", b"x")], data_root=str(root))


def test_mixed_in_and_out_of_root(root, tmp_path):
    out_file = tmp_path / "outside.jpg"
    out_file.write_bytes(b"\x07z")
    (root / "esc.jpg").symlink_to(out_file)  # in-root name, resolves outside -> upload
    items = [str(root / "sub" / "0.jpg"), str(out_file), str(root / "esc.jpg"),
             ("t.jpg", b"\x05q"), str(root / "sub" / "1.jpg")]
    c = PathsClient()
    res = worker_client.embed_both(c, items, data_root=str(root))
    assert res.mode == "mixed" and res.errors == [None] * 5
    assert c.paths_of(worker_protocol.PATH_EMBED_BOTH_PATHS)[0][1] == 2
    assert c.paths_of(worker_protocol.PATH_EMBED_BOTH_BATCH)[0][1] == 3
    assert [int(res.siglip[i].argmax()) for i in range(5)] == [1, 7, 7, 5, 2]


def test_item_errors_in_path_mode(root):
    (root / "sub" / "bad.jpg").write_bytes(b"BAD")
    items = [str(root / "sub" / "bad.jpg"), str(root / "sub" / "0.jpg"), str(root / "sub" / "gone.jpg")]
    res = worker_client.embed_both(PathsClient(), items, data_root=str(root))
    assert res.errors[0] == "cannot decode" and res.errors[1] is None and res.errors[2]
