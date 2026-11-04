"""SigLIP 2 / DINOv2 per-image embedding tables (embeddings_siglip / embeddings_dino)."""
import numpy as np
import pytest

from media_manager.database import Database

SPACES = [
    ('siglip', 'embeddings_siglip'),
    ('dino', 'embeddings_dino'),
]


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / 'media.db'))
    yield d
    d.close()


def _add(db, name):
    return db.upsert_file_path(name, 'sum-' + name, size=1, modified_time=1)


def _vec(seed, dim=8):
    return np.random.RandomState(seed).rand(dim).astype(np.float32).tobytes()


def _m(db, space, name):
    return getattr(db, name.format(space))


@pytest.mark.parametrize('space,table', SPACES)
def test_insert_fetch_and_stats(db, space, table):
    a, b = _add(db, 'a.jpg'), _add(db, 'b.jpg')
    ins = _m(db, space, 'insert_{}_embedding')
    ins(a, _vec(1), 'm1')
    ins(b, _vec(2), 'm1')
    ins(a, _vec(3), 'm1', frame_index=5)  # non-primary frame
    rows = _m(db, space, 'get_all_{}_embeddings')()
    assert sorted(r[0] for r in rows) == sorted([a, b])
    assert {r[1] for r in rows} == {'a.jpg', 'b.jpg'}
    assert _m(db, space, 'count_{}_indexed')() == 2
    chunks = list(_m(db, space, 'iter_{}_embeddings')(chunk=1))
    assert [c[0] for c in chunks] == [[min(a, b)], [max(a, b)]]
    assert len(chunks[0][1]) == 32
    st = db.embedding_stats()[space]
    assert st == {'rows': 2, 'dim': 8, 'bytes': 2 * 8 * 4}
    # upsert replaces
    ins(a, _vec(9), 'm2')
    assert _m(db, space, 'count_{}_indexed')() == 2
    assert db.conn.execute(f'SELECT model FROM {table} WHERE file_id=? AND frame_index=0',
                           (a,)).fetchone()[0] == 'm2'


@pytest.mark.parametrize('space,table', SPACES)
def test_needing_detects_missing_and_model_change(db, space, table):
    a, b = _add(db, 'a.jpg'), _add(db, 'b.jpg')
    need = _m(db, space, 'get_files_needing_{}')
    ins = _m(db, space, 'insert_{}_embedding')
    assert {r[0] for r in need('m1')} == {a, b}
    ins(a, _vec(1), 'm1')
    assert {r[0] for r in need('m1')} == {b}
    assert {r[0] for r in need('m2')} == {a, b}  # model changed -> re-embed
    ins(a, _vec(1), 'm1', frame_index=3)  # frame-only row does not count as primary
    ins(b, _vec(1), 'm1', frame_index=3)
    assert {r[0] for r in need('m1')} == {b}
    assert len(need('m2', limit=1)) == 1


@pytest.mark.parametrize('space,table', SPACES)
def test_matrix_cache_invalidation(db, space, table):
    a, b = _add(db, 'a.jpg'), _add(db, 'b.jpg')
    ins = _m(db, space, 'insert_{}_embedding')
    mat = _m(db, space, 'get_{}_embeddings_matrix')
    ids, sums, m = mat()
    assert m.shape[0] == 0
    ins(a, _vec(1), 'm1')
    ids, sums, m = mat()
    assert list(ids) == [a] and sums == ['sum-a.jpg'] and m.shape == (1, 8)
    assert mat() is mat()  # cached
    ins(b, _vec(2), 'm1')
    assert mat()[2].shape == (2, 8)
    # does not disturb the CLIP cache version
    assert db._emb_ver == 0


@pytest.mark.parametrize('space,table', SPACES)
def test_deletion_cleanup(db, space, table):
    a, b, c = _add(db, 'x/a.jpg'), _add(db, 'x/b.jpg'), _add(db, 'y/c.jpg')
    ins = _m(db, space, 'insert_{}_embedding')
    mat = _m(db, space, 'get_{}_embeddings_matrix')
    for f in (a, b, c):
        ins(f, _vec(f), 'm1')
    assert mat()[2].shape[0] == 3
    db.remove_paths_under('x')
    assert [r[0] for r in db.conn.execute(f'SELECT file_id FROM {table}')] == [c]
    assert mat()[2].shape[0] == 1
    db.delete_file_completely(c)
    assert db.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
    assert mat()[2].shape[0] == 0


@pytest.mark.parametrize('space,table', SPACES)
def test_clear_primary_ml_data(db, space, table):
    a = _add(db, 'a.jpg')
    ins = _m(db, space, 'insert_{}_embedding')
    ins(a, _vec(1), 'm1')
    ins(a, _vec(2), 'm1', frame_index=4)
    mat = _m(db, space, 'get_{}_embeddings_matrix')
    assert mat()[2].shape[0] == 1
    db.clear_primary_ml_data(a)
    rows = db.conn.execute(f'SELECT frame_index FROM {table}').fetchall()
    assert [tuple(r) for r in rows] == [(4,)]
    assert mat()[2].shape[0] == 0
