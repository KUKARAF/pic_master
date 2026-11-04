"""Embedding pipeline throughput work: JPEG draft decode, GPU lock vs decode overlap, timing
instrumentation round trip, batched DB writes. Fakes only -- no models, no GPU."""
import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

from media_manager import compute, worker_client, worker_protocol, worker_server
from media_manager import embedders
from media_manager.database import Database


# ---- draft decode ---------------------------------------------------------------

@pytest.fixture
def big_jpeg(tmp_path):
    img = Image.new('RGB', (2400, 1200), (200, 30, 30))
    img.paste((20, 20, 220), (0, 0, 1200, 1200))  # left half blue
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display -> portrait
    p = tmp_path / 'big.jpg'
    img.save(p, 'JPEG', exif=exif)
    return str(p)


def test_draft_decodes_smaller_but_keeps_orientation(big_jpeg):
    full = embedders._decode_rgb(big_jpeg)
    small = embedders._decode_rgb(big_jpeg, 448)
    assert full.size == (1200, 2400)  # exif_transpose made it portrait
    assert small.size[0] < full.size[0] and min(small.size) >= 448
    assert small.size[0] < small.size[1]  # still portrait: orientation preserved
    # same content: blue was the left half -> after 90 CW it is the TOP half
    w, h = small.size
    assert small.getpixel((w // 2, h // 4))[2] > 150 and small.getpixel((w // 2, 3 * h // 4))[0] > 150


def test_draft_disabled_by_env(big_jpeg, monkeypatch):
    monkeypatch.setenv('MEDIA_EMBED_DRAFT', '0')
    assert embedders._decode_rgb(big_jpeg, 448).size == (1200, 2400)


def test_draft_target_from_embedder_sizes():
    class E:
        def __init__(self, s):
            self.input_size = s
    assert embedders._draft_target([E(224), E(384)]) == 2 * 384
    assert embedders._draft_target([object()]) == 2 * embedders.DINO_INPUT


def test_draft_skipped_for_png_and_pil(tmp_path):
    p = tmp_path / 'a.png'
    Image.new('RGB', (1000, 900)).save(p)
    assert embedders._decode_rgb(str(p), 100).size == (1000, 900)
    assert embedders._decode_rgb(Image.new('RGB', (1000, 900)), 100).size == (1000, 900)


# ---- gpu lock -------------------------------------------------------------------

class SleepEmb:
    batch_size, num_workers, input_size = 2, 3, 224
    state = None

    def __init__(self, st, dim=4):
        self.st, self.dim = st, dim

    def _prep(self, img):
        with self.st['m']:
            self.st['prep_active'] += 1
            if self.st['run_active']:
                self.st['overlap'] += 1
        time.sleep(0.02)
        with self.st['m']:
            self.st['prep_active'] -= 1
        return np.zeros(self.dim, np.float32)

    def _run(self, tensors):
        with self.st['m']:
            self.st['run_active'] += 1
            self.st['max_run'] = max(self.st['max_run'], self.st['run_active'])
            if self.st['prep_active']:
                self.st['overlap'] += 1
        time.sleep(0.03)
        with self.st['m']:
            self.st['run_active'] -= 1
        return np.zeros((len(tensors), self.dim), np.float32)


def test_gpu_lock_excludes_forwards_but_overlaps_decode(tmp_path):
    st = dict(m=threading.Lock(), prep_active=0, run_active=0, max_run=0, overlap=0)
    imgs = []
    for i in range(16):
        p = tmp_path / f'{i}.png'
        Image.new('RGB', (8, 8)).save(p)
        imgs.append(str(p))
    all_stats = [{}, {}, {}]
    errs = []

    def go(stats):
        try:
            (o,), failed = embedders.embed_shared([SleepEmb(st)], imgs, stats=stats)
            assert o.shape == (16, 4) and not failed
        except Exception as e:  # noqa
            errs.append(e)

    ts = [threading.Thread(target=go, args=(s,)) for s in all_stats]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert st['max_run'] == 1            # forwards never concurrent
    assert st['overlap'] > 0             # decode ran while a forward was in flight
    tot = {k: sum(s[k] for s in all_stats) for k in ('decode_seconds', 'gpu_seconds', 'lock_wait_seconds')}
    assert tot['decode_seconds'] > 0 and tot['gpu_seconds'] > 0 and tot['lock_wait_seconds'] > 0


def test_models_lock_still_excludes_embed_forwards():
    """Non-embed handlers (`with models.lock`) hold the GPU lock, so an embed forward waits."""
    got = []
    with worker_server.models.lock:
        assert compute.GPU_LOCK.locked()
        t = threading.Thread(target=lambda: (compute.GPU_LOCK.acquire(), got.append(1),
                                             compute.GPU_LOCK.release()))
        t.start()
        time.sleep(0.1)
        assert not got
    t.join(2)
    assert got == [1] and not compute.GPU_LOCK.locked()


# ---- instrumentation round trip ----------------------------------------------------

def test_timing_fields_roundtrip(monkeypatch):
    from test_embed_worker import FakeEmb, FakeClient
    m = worker_server.models
    monkeypatch.setattr(m, '_siglip', FakeEmb(8, 'sig/fake'))
    monkeypatch.setattr(m, '_dino', FakeEmb(4, 'dino/fake'))

    def fake(embs, sources, stats=None):
        stats.update(decode_seconds=0.4, gpu_seconds=0.2, lock_wait_seconds=0.05)
        return [np.zeros((len(sources), e.dim), np.float32) for e in embs], []

    monkeypatch.setattr(worker_server, '_embed_shared', fake)
    items = [(f'{i}.jpg', bytes([i]) + b'x') for i in range(6)]
    res = worker_client.embed_both(FakeClient(), items, batch_size=3, workers=2)
    assert res.n_batches == 2 and res.workers == 2
    assert res.decode_seconds == pytest.approx(0.8) and res.gpu_seconds == pytest.approx(0.4)
    assert res.lock_wait_seconds == pytest.approx(0.1) and res.worker_wall_seconds > 0


def test_speedtest_breakdown_verdicts():
    from media_manager import web
    b = web._speedtest_breakdown(10.0, 'paths', 9.0, {'decode_seconds': 5.0, 'lock_wait_seconds': 1.0})
    assert b['gpu_busy_pct'] == 90.0 and b['verdict'].startswith('GPU-bound')
    b = web._speedtest_breakdown(10.0, 'paths', 3.0, {'decode_seconds': 20.0})
    assert b['verdict'].startswith('decode-bound') and b['decode_threads_avg'] == 2.0
    b = web._speedtest_breakdown(10.0, 'paths', 3.0, {'decode_seconds': 1.0})
    assert b['verdict'].startswith('neither')
    assert web._speedtest_breakdown(1.0, 'local', None, None)['gpu_busy_pct'] is None
    # summed-over-overlap seconds can never push busy above 100%
    assert web._speedtest_breakdown(1.0, 'paths', 1.7, {'decode_seconds': 0.1})['gpu_busy_pct'] == 100.0


# ---- batched DB writes ----------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / 'media.db'))
    ids = [d.upsert_file_path(f'f{i}.jpg', f'sum{i}', size=1, modified_time=1) for i in range(5)]
    d.conn.commit()
    return d, ids


def _blob(i, dim):
    return np.full(dim, i, np.float32).tobytes()


def test_pair_batch_insert_one_commit_and_version(db):
    d, ids = db
    v_s, v_d = d._siglip_ver, d._dino_ver
    commits = []
    real = d.conn
    d.insert_pair_embeddings_batch([(i, _blob(i, 6), 'm/s') for i in ids],
                                   [(i, _blob(i, 4), 'm/d') for i in ids[:3]])
    assert d.count_siglip_indexed() == 5 and d.count_dino_indexed() == 3
    assert d._siglip_ver == v_s + 1 and d._dino_ver == v_d + 1  # bumped once per space
    d.insert_siglip_embeddings_batch([(ids[0], _blob(9, 6), 'm/s2')])  # upsert
    assert d.count_siglip_indexed() == 5
    row = d.conn.execute('SELECT model FROM embeddings_siglip WHERE file_id=?', (ids[0],)).fetchone()
    assert row[0] == 'm/s2'
    d.insert_pair_embeddings_batch([], [])  # no-op


def test_pair_batch_rolls_back_on_error_and_single_row_still_works(db):
    d, ids = db
    v = d._siglip_ver
    with pytest.raises(Exception):
        d.insert_pair_embeddings_batch([(ids[0], _blob(1, 6), 'm'), (ids[1], object(), 'm')])
    assert d.count_siglip_indexed() == 0 and d._siglip_ver == v
    d.insert_siglip_embedding(ids[2], _blob(2, 6), 'm')  # legacy API intact
    d.insert_dino_embedding(ids[2], _blob(2, 4), 'm')
    assert d.count_siglip_indexed() == 1 and d.count_dino_indexed() == 1
