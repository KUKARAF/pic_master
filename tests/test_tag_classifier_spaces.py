"""Tag classifiers / tag suggestions on the SigLIP 2 + DINOv2 embedding spaces (CLIP was removed).

No GPU, no models: embeddings are tiny fake vectors inserted straight into the space tables,
training runs the (torch, CPU-fine) logistic fit in-process, and worker/subprocess edges are stubbed."""
import io
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import embedding_tag_classifier as ct
from media_manager import remote_tag_trainer, web, worker_server

DIM = {'siglip': 6, 'dino': 4}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _vec(dim, positive, seed):
    """Positives cluster around +e0, negatives around -e0 (plus noise)."""
    rng = np.random.RandomState(seed)
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0 if positive else -1.0
    return _unit(base + 0.15 * rng.randn(dim).astype(np.float32))


@pytest.fixture
def env(tmp_path, monkeypatch):
    os.makedirs(tmp_path / '.media')
    monkeypatch.setattr(web, '_worker_client', None)
    app = web.create_app(str(tmp_path))
    db, errors, manual = app.state.dbs
    ids, sums = {}, {}
    # 12 files: 0-5 "cat" positives, 6-9 negatives, 10-11 undecided candidates (one cat-like, one not)
    for i in range(12):
        name = f'img{i}.jpg'
        Image.new('RGB', (8, 8), (i * 20, 0, 0)).save(tmp_path / name)
        sums[name] = f'sum{i}'
        ids[name] = db.upsert_file_path(name, f'sum{i}', size=1, modified_time=1)
    db.conn.commit()
    pos = [i < 6 or i == 10 for i in range(12)]
    db.insert_pair_embeddings_batch(
        siglip_rows=[(ids[f'img{i}.jpg'], _vec(DIM['siglip'], pos[i], i).tobytes(), 'fake-siglip/v1') for i in range(12)],
        dino_rows=[(ids[f'img{i}.jpg'], _vec(DIM['dino'], pos[i], 100 + i).tobytes(), 'fake-dino/v1') for i in range(12)])
    for i in range(6):
        manual.add_tag(sums[f'img{i}.jpg'], 'cat', 'positive')
    for i in range(6, 10):
        manual.add_tag(sums[f'img{i}.jpg'], 'cat', 'negative')
    manual.add_tag('sum-ghost', 'cat', 'negative')  # labeled but never embedded
    return app, TestClient(app), db, manual, ids, tmp_path


# ---- pure classifier module ---------------------------------------------------------------

def test_spaces_and_kinds():
    assert ct.DEFAULT_SPACE == 'siglip'
    assert ct.space_kind('siglip') == 'siglip_model' and ct.space_kind('dino') == 'dino_model'
    with pytest.raises(ValueError):
        ct.space_kind('clip')   # the CLIP space is gone
    with pytest.raises(ValueError):
        ct.space_kind('nope')
    assert ct.CLASSIFIER_KINDS == ('siglip_model', 'dino_model', 'yolo_model')


def test_training_with_too_few_embedded_examples_fails_loudly(env):
    _app, _c, _db, manual, ids, root = env
    # 6 pos / 4 embedded neg (the ghost negative has no embedding) -> clear failure + failed status
    with pytest.raises(ValueError, match='Need at least'):
        ct.train_for_tag(str(root), 'cat', 'siglip')
    st = ct.classifier_dir(str(root), 'cat', 'siglip_model')
    assert 'failed' in open(os.path.join(st, 'metadata.json')).read()


def _train_ok(env):
    _app, _c, db, manual, ids, root = env
    # make negatives >= 5: img11 is a (non-cat) embedded negative
    manual.add_tag('sum11', 'cat', 'negative')
    ct.train_for_tag(str(root), 'cat', 'siglip')
    return root


def test_siglip_training_writes_stamped_artifact_and_scores(env):
    root = _train_ok(env)
    kdir = ct.classifier_dir(str(root), 'cat', 'siglip_model')
    data = np.load(os.path.join(kdir, 'weights.npz'))
    assert str(data['space']) == 'siglip' and int(data['dim']) == DIM['siglip']
    assert str(data['embedding_model']) == ct.current_embedding_model('siglip')
    assert ct.weights_state(str(root), 'cat', 'siglip') == 'ok'
    assert ct.weights_state(str(root), 'cat', 'dino') == 'missing'
    # no clip artifact was created
    assert not os.path.exists(ct.classifier_dir(str(root), 'cat', 'clip_model'))
    s_pos = ct.score_all(str(root), 'cat', [('p', _vec(6, True, 1).tobytes()), ('n', _vec(6, False, 2).tobytes())],
                         space='siglip')
    assert s_pos['p'] > 0.5 > s_pos['n']


def test_dimension_mismatch_and_model_change_need_retrain(env, monkeypatch):
    root = _train_ok(env)
    with pytest.raises(ct.NeedsRetrain, match='dim'):
        ct.score_all(str(root), 'cat', [('x', np.ones(512, dtype=np.float32).tobytes())], space='siglip')
    with pytest.raises(ct.NeedsRetrain):
        ct.score_matrix(str(root), 'cat', np.ones((2, 512), dtype=np.float32), 'siglip')
    monkeypatch.setenv('MEDIA_SIGLIP_MODEL', 'some-other-model')
    assert ct.weights_state(str(root), 'cat', 'siglip') == 'needs_retrain'
    with pytest.raises(ct.NeedsRetrain, match='retrain'):
        ct.score_all(str(root), 'cat', [], space='siglip')


def test_legacy_unstamped_clip_artifact_needs_retrain(tmp_path):
    # exactly what the pre-migration CLIP trainer wrote: no space stamp, 512-d, CLIP model id
    sdir = ct.classifier_dir(str(tmp_path), 'dog', 'siglip_model')
    os.makedirs(sdir)
    np.savez(os.path.join(sdir, 'weights.npz'), w=np.ones(512, np.float32), b=np.float32(0),
             embedding_model='ViT-B-32/openai')
    assert ct.weights_state(str(tmp_path), 'dog', 'siglip') == 'needs_retrain'
    with pytest.raises(ct.NeedsRetrain):
        ct.score_matrix(str(tmp_path), 'dog', np.ones((1, 512), np.float32), 'siglip')
    # a leftover clip_model/ dir is simply never read
    assert ct.LEGACY_CLIP_DIR_KIND not in ct.CLASSIFIER_KINDS


# ---- web layer ----------------------------------------------------------------------------

def test_train_clip_route_retired(env):
    _app, c, *_ = env
    r = c.post('/api/tags/cat/train-clip')
    assert r.status_code == 410 and 'train-siglip' in r.json()['detail']


def test_train_siglip_refuses_when_not_embedded_and_names_the_fix(env):
    _app, c, db, manual, ids, root = env
    db.conn.execute('DELETE FROM embeddings_siglip')
    db.conn.commit()
    db._siglip_ver += 1
    r = c.post('/api/tags/cat/train-siglip')
    assert r.status_code == 400 and 'Index SigLIP 2' in r.json()['detail']


def test_train_siglip_launches_with_space(env, monkeypatch):
    _app, c, db, manual, ids, root = env
    manual.add_tag('sum11', 'cat', 'negative')
    launched = []
    monkeypatch.setattr(web.subprocess, 'Popen', lambda argv, **kw: launched.append(argv))
    monkeypatch.setenv('MEDIA_ALLOW_LOCAL_TRAINING', '1')
    r = c.post('/api/tags/cat/train-siglip')
    assert r.status_code == 200 and r.json()['space'] == 'siglip'
    assert launched[-1][-2:] == ['--space', 'siglip'] and 'media_manager.embedding_tag_classifier' in launched[-1]
    st = c.get('/api/tags/cat/train-status?kind=siglip').json()
    assert st['status'] == 'running'
    # worker configured -> remote driver with the embedding-space kind
    monkeypatch.delenv('MEDIA_ALLOW_LOCAL_TRAINING')
    monkeypatch.setattr(web, '_worker_client', type('W', (), {'is_configured': lambda self: True})())
    assert c.post('/api/tags/cat/train-dino').status_code == 200
    assert launched[-1][-2:] == ['--kind', 'dino_model'] and 'media_manager.remote_tag_trainer' in launched[-1]


def test_unknown_kind_rejected(env):
    _app, c, *_ = env
    assert c.get('/api/tags/cat/train-status?kind=bogus').status_code == 400


def test_by_model_ranks_in_space_and_default_is_siglip(env):
    root = _train_ok(env)
    _app, c, *_ = env
    r = c.post('/api/tags/cat/suggestions/next-by-model', json={'exclude': []})
    assert r.status_code == 200
    cards = r.json()['cards']
    # img10 (undecided, cat-like) must outrank nothing else decided: decided files never come back
    assert [x['file_id'] for x in cards][:1] == [env[4]['img10.jpg']]
    assert all('bbox' not in x for x in cards)
    # untrained space -> empty, not an error
    assert c.post('/api/tags/cat/suggestions/next-by-model?model=dino', json={'exclude': []}).json()['cards'] == []
    # the removed clip model is reported as "needs retrain" (409), never a crash or a silent empty list
    r = c.post('/api/tags/cat/suggestions/next-by-model?model=clip', json={'exclude': []})
    assert r.status_code == 409 and 'retrain' in r.json()['detail'].lower()
    assert c.post('/api/tags/cat/suggestions/next-by-model?model=zzz', json={'exclude': []}).status_code == 400


def test_by_model_needs_retrain_is_409_and_status_says_so(env, monkeypatch):
    _train_ok(env)
    _app, c, *_ = env
    monkeypatch.setenv('MEDIA_SIGLIP_MODEL', 'swapped-model')
    r = c.post('/api/tags/cat/suggestions/next-by-model', json={'exclude': []})
    assert r.status_code == 409 and 'retrain' in r.json()['detail'].lower()
    st = c.get('/api/tags/cat/train-status?kind=siglip').json()
    assert st['status'] == 'needs_retrain'


def test_by_model_dimension_mismatch_in_index_is_409(env):
    root = _train_ok(env)
    _app, c, db, manual, ids, _root = env
    # simulate the index being rebuilt with a different-dimension model while the stamp still matches
    db.conn.execute('DELETE FROM embeddings_siglip')
    db.conn.commit()
    db.insert_siglip_embeddings_batch([(ids[f'img{i}.jpg'], np.ones(9, np.float32).tobytes(), 'm') for i in range(12)])
    r = c.post('/api/tags/cat/suggestions/next-by-model', json={'exclude': []})
    assert r.status_code == 409


def test_centroid_stream_uses_siglip_by_default_without_bbox(env):
    _app, c, db, manual, ids, root = env
    cards = c.post('/api/tags/cat/suggestions/next', json={'exclude': []}).json()['cards']
    got = [x['file_id'] for x in cards]
    decided = {ids[f'img{i}.jpg'] for i in range(10)}
    assert got and not (set(got) & decided)
    assert got[0] == ids['img10.jpg']          # the cat-like undecided one ranks first
    assert all('bbox' not in x for x in cards)
    # dino works too; the removed clip model is a 400
    assert c.post('/api/tags/cat/suggestions/next?model=dino', json={'exclude': []}).json()['cards']
    assert c.post('/api/tags/cat/suggestions/next?model=clip', json={'exclude': []}).status_code == 400
    # exclude list honoured
    again = c.post('/api/tags/cat/suggestions/next', json={'exclude': [str(ids['img10.jpg'])]}).json()['cards']
    assert ids['img10.jpg'] not in [x['file_id'] for x in again]


def test_tags_pages_label_siglip_not_clip(env):
    _app, c, *_ = env
    html = c.get('/tags').text
    assert 'SigLIP 2' in html and 'DINOv2' in html
    assert 'Train CLIP' not in html
    page = c.get('/search?tag=cat').text
    assert 'Review with SigLIP 2' in page and 'Review with CLIP' not in page


def test_legacy_clip_artifact_is_noted_not_crashing(env):
    _app, c, db, manual, ids, root = env
    kdir = ct.classifier_dir(str(root), 'cat', ct.LEGACY_CLIP_DIR_KIND)
    os.makedirs(kdir)
    np.savez(os.path.join(kdir, 'weights.npz'), w=np.ones(512, np.float32), b=np.float32(0),
             embedding_model='ViT-B-32/openai')
    ct.write_status(kdir, status='done', cv_accuracy=0.9)
    page = c.get('/search?tag=cat').text
    assert 'old CLIP classifier' in page and 'Retrain it with SigLIP 2' in page
    # every route that used to take kind/model=clip answers "needs retrain" (409), not a crash
    r = c.get('/api/tags/cat/train-status?kind=clip')
    assert r.status_code == 409 and 'retrain' in r.json()['detail'].lower()
    r = c.post('/api/tags/cat/suggestions/next-by-model?model=clip', json={'exclude': []})
    assert r.status_code == 409 and 'retrain' in r.json()['detail'].lower()


# ---- remote trainer / worker --------------------------------------------------------------

class _FakeClient:
    def __init__(self, blob):
        self.blob = blob

    def train_status(self, job_id, want_artifact=False):
        return {'artifact': self.blob}


def _npz(dim):
    buf = io.BytesIO()
    np.savez(buf, w=np.ones(dim, np.float32), b=np.float32(0.1), embedding_model='ViT-B-32/openai')
    return buf.getvalue()


def test_remote_finalize_validates_dim_and_restamps(tmp_path):
    out = str(tmp_path)
    remote_tag_trainer._finalize_done(_FakeClient(_npz(6)), 'j', out, 'siglip_model', {'cv_accuracy': 0.8},
                                      {'expected_dim': 6})
    data = np.load(os.path.join(out, 'weights.npz'))
    assert str(data['space']) == 'siglip' and str(data['embedding_model']) == ct.current_embedding_model('siglip')
    # a stale worker that trained in the wrong space (512-d vs the uploaded 6-d) is rejected loudly
    with pytest.raises(RuntimeError, match='redeploy the worker'):
        remote_tag_trainer._finalize_done(_FakeClient(_npz(512)), 'j', out, 'siglip_model', {}, {'expected_dim': 6})


def test_worker_job_trains_in_the_requested_space_and_embeds_regions_with_it(tmp_path, monkeypatch):
    calls = []

    class FakeEmb:
        def embed_pil_images(self, imgs):
            calls.append(len(imgs))
            return np.stack([_vec(6, True, 900 + i) for i in range(len(imgs))])

    monkeypatch.setattr(worker_server, 'TAG_MODELS_DIR', str(tmp_path))
    monkeypatch.setattr(worker_server.models, 'get_siglip_embedder', lambda: FakeEmb())
    monkeypatch.setattr(worker_server.models, 'get_dino_embedder', lambda: pytest.fail('wrong space'))
    assert not hasattr(worker_server.models, 'get_clip_indexer')
    job = worker_server.training_jobs.create('siglip_model', 'cat', 'cat', None)
    for i in range(5):
        job.whole_examples.append((_vec(6, False, i).tobytes(), 0.0))
        job.whole_examples.append((_vec(6, True, 50 + i).tobytes(), 1.0))
    buf = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buf, format='JPEG')
    job.region_examples.append((buf.getvalue(), 1.0))
    worker_server._run_embedding_job(job)
    assert job.status == 'done', job.error
    assert calls == [1]
    data = np.load(io.BytesIO(job.artifact_bytes))
    assert str(data['space']) == 'siglip' and data['w'].shape == (6,)
    assert os.path.isdir(tmp_path / 'cat' / 'siglip_model')


def test_worker_rejects_unknown_kind_so_stale_worker_cannot_train_wrong_space():
    # (the inverse: a NEW host talking to an OLD worker gets "unknown training kind" from train_create)
    from media_manager import worker_protocol
    resp = worker_protocol.unpack(worker_server.handle_train_create(
        '', worker_protocol.pack({'kind': 'bogus_model', 'tag': 't', 'slug': 't'}), 0, 0, None, 0))
    assert resp['job_id'] is None and 'unknown training kind' in resp['error']
    gone = worker_protocol.unpack(worker_server.handle_train_create(
        '', worker_protocol.pack({'kind': 'clip_model', 'tag': 't', 'slug': 't'}), 0, 0, None, 0))
    assert gone['job_id'] is None and 'unknown training kind' in gone['error']   # CLIP training removed
    ok = worker_protocol.unpack(worker_server.handle_train_create(
        '', worker_protocol.pack({'kind': 'dino_model', 'tag': 't', 'slug': 't'}), 0, 0, None, 0))
    assert ok['job_id'] and ok['error'] is None
