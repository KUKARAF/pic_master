"""Body index on DINOv2-L / SigLIP 2 (no CLIP): model tag, stale old-space rows, label refresh."""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from media_manager import body_index, scan_status
from media_manager.database import Database
from media_manager.manual_db import ManualDB

OLD = 'ViT-B-32/openai'


class FakeEmbedder:
    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = 0

    def embed_pil_images(self, images):
        self.calls += 1
        out = np.zeros((len(images), self.dim), np.float32)
        out[:, 0] = 1.0
        return out


@pytest.fixture
def dbs(monkeypatch):
    monkeypatch.delenv('MEDIA_BODY_MODEL', raising=False)
    monkeypatch.delenv('MEDIA_DINO_MODEL', raising=False)
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
    db = Database(os.path.join(tmp, 'media.db'))
    manual = ManualDB(os.path.join(tmp, 'manual.db'))
    return db, manual, tmp


def _file(db, path):
    fid = db.upsert_file_path(path, 'cs-' + path, size=1)
    db.conn.commit()
    return fid


def test_model_tag_env(monkeypatch):
    monkeypatch.delenv('MEDIA_BODY_MODEL', raising=False)
    assert body_index.model_tag() == 'dino:large:1024' and body_index.model_dim() == 1024
    monkeypatch.setenv('MEDIA_BODY_MODEL', 'siglip')
    assert body_index.model_tag() == 'siglip:ViT-SO400M-14-SigLIP2:1152'
    assert body_index.threshold() == body_index.BODY_THRESHOLDS['siglip']
    monkeypatch.setenv('MEDIA_BODY_MODEL', 'clip')
    with pytest.raises(ValueError):
        body_index.model_tag()


def test_dim_tables_match_embedders():
    embedders = pytest.importorskip('media_manager.embedders')
    assert body_index._DINO_DIMS == {k: v[1] for k, v in embedders.DINO_VARIANTS.items()}
    assert body_index._SIGLIP_DIMS == embedders.SIGLIP_DIM


def test_stale_rows_are_missing_and_reembedded_in_place(dbs):
    db, _manual, _ = dbs
    tag = body_index.model_tag()
    f_old, f_new, f_none, f_empty = (_file(db, p) for p in ('old.jpg', 'new.jpg', 'none.jpg', 'empty.jpg'))
    old = np.ones(512, np.float32)
    db.insert_body_embeddings(f_old, [{'bbox': [0, 0, 50, 80], 'embedding': old}], OLD)
    db.insert_body_embeddings(f_new, [{'bbox': [0, 0, 50, 80], 'embedding': np.ones(1024, np.float32)}], tag)
    db.insert_body_embeddings(f_empty, [], OLD)          # no-person sentinel: model independent
    pending = {r[0] for r in db.get_unbody_indexed_files(tag)}
    assert pending == {f_old, f_none}
    assert db.body_file_state(f_old, tag) == 'stale' and db.body_file_state(f_none, tag) == 'none'
    assert db.body_file_state(f_new, tag) == 'fresh' and db.body_file_state(f_empty, tag) == 'fresh'

    # matrices / per-file reads never mix spaces
    ids, fids, _b, mat = db.get_body_embeddings_matrix(tag)
    assert list(fids) == [f_new] and mat.shape == (1, 1024)
    assert db.get_body_embeddings_for_file(f_old, tag) == []

    # re-embed keeps the row id (manual.db decisions reference it) and flips the tag
    before = db.get_body_embeddings_for_file(f_old)[0][0]
    db.insert_body_embeddings(f_old, [{'bbox': [0, 0, 50, 80], 'embedding': np.ones(1024, np.float32)}], tag)
    rows = db.get_body_embeddings_for_file(f_old, tag)
    assert [r[0] for r in rows] == [before]
    assert {r[0] for r in db.get_unbody_indexed_files(tag)} == {f_none}
    assert db.get_body_embeddings_matrix(tag)[3].shape == (2, 1024)


def test_scan_status_counts_stale_bodies_as_missing(dbs):
    db, manual, _ = dbs
    tag = body_index.model_tag()
    f = _file(db, 'a.jpg')
    db.insert_body_embeddings(f, [{'bbox': [0, 0, 9, 9], 'embedding': np.ones(512, np.float32)}], OLD)
    scan_status.invalidate()
    sets = scan_status._load_full(db, manual)['sets']['bodies']
    assert f not in sets
    db.insert_body_embeddings(f, [{'bbox': [0, 0, 9, 9], 'embedding': np.ones(1024, np.float32)}], tag)
    assert f in scan_status._load_full(db, manual)['sets']['bodies']


def test_embed_bodies_for_file_tags_and_checks_dim(dbs):
    db, _m, tmp = dbs
    path = os.path.join(tmp, 'p.jpg')
    Image.new('RGB', (200, 300), 'red').save(path)
    fid = _file(db, 'p.jpg')
    n = body_index.embed_bodies_for_file(db, FakeEmbedder(1024), fid, path, person_boxes=[[0, 0, 100, 200]])
    assert n == 1 and db.body_file_state(fid, body_index.model_tag()) == 'fresh'
    with pytest.raises(RuntimeError, match='expected'):
        body_index.embed_bodies_for_file(db, FakeEmbedder(768), fid, path, person_boxes=[[0, 0, 100, 200]])


def test_refresh_stale_labels(dbs):
    db, manual, tmp = dbs
    tag = body_index.model_tag()
    Image.new('RGB', (200, 300), 'blue').save(os.path.join(tmp, 'q.jpg'))
    _file(db, 'q.jpg')
    manual.add_body_label('cs-q.jpg', 'Ann', [0, 0, 100, 200], np.ones(512, np.float32).tobytes(), OLD)
    assert manual.get_body_embeddings_for_identity('Ann', tag) == []       # stale anchor invisible
    assert len(manual.get_body_embeddings_for_identity('Ann')) == 1
    emb = FakeEmbedder()
    assert body_index.refresh_stale_labels(db, manual, emb, tmp, identity='Ann') == 1
    anchors = manual.get_body_embeddings_for_identity('Ann', tag)
    assert len(anchors) == 1 and len(np.frombuffer(anchors[0], np.float32)) == 1024
    assert body_index.refresh_stale_labels(db, manual, emb, tmp) == 0       # idempotent


def test_bodies_stage_runner_does_not_use_clip(dbs, monkeypatch):
    import inspect
    from media_manager import web
    src = inspect.getsource(web)
    start = src.index('def _folder_ml_runners')
    assert "'bodies': R(_bodies, lambda: (_get_body_embedder()" in src[start:]
    assert '_get_clip_indexer' not in inspect.getsource(body_index)
