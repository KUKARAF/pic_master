"""Age/gender from a named BODY alone (MiVOLO body-only mode).

A photo whose person was labelled with "Label person" (a hand-drawn, named body in
manual.db body_identities) but has no face used to get 'No faces detected' from the
🎂 Age / gender button. Now each such face-less body is sent to the estimator as a
'body:{id}' ref and its result saved like a face's. A body that contains a face is
skipped (the face's estimate already uses it), as are rejected and auto 'face-link'
bodies; deleting/re-deciding a body label drops its age row.

The MiVOLO subprocess is replaced by a fake estimator, so no .age-venv is needed.

Run: python -m pytest tests/test_body_age.py   (or standalone: python tests/test_body_age.py)
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

# Same headless-import stub as test_body_identity: importing media_manager.web pulls
# in modules that `import cv2`; nothing under test touches it.
sys.modules.setdefault('cv2', MagicMock())

import media_manager.web as web  # noqa: E402  # hook-allow: after cv2 stub
from media_manager.web import create_app  # noqa: E402  # hook-allow: after cv2 stub

D = 512


class FakeEstimator:
    """Records every call; answers each face/body with a fixed age/gender."""

    def __init__(self):
        self.calls = []

    def estimate(self, image_path, faces, bodies=None):
        bodies = bodies or []
        self.calls.append({'faces': [f['ref'] for f in faces],
                           'bodies': [b['ref'] for b in bodies]})
        return ([{'face_ref': f['ref'], 'age': 30.0, 'gender': 'male'} for f in faces]
                + [{'face_ref': b['ref'], 'age': 7.5, 'gender': 'female'} for b in bodies])


def _emb(rng):
    v = rng.standard_normal(D)
    return (v / np.linalg.norm(v)).astype(np.float32).tobytes()


class _Env:
    def __init__(self, tmp):
        os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
        self.tmp = tmp
        self.app = create_app(tmp)
        self.db, _errors, self.manual = self.app.state.dbs
        self.client = TestClient(self.app)
        self.rng = np.random.default_rng(3)

    def photo(self, name):
        """A file row whose path exists on disk (estimate-age 404s otherwise); the
        bytes don't matter — the fake estimator never opens it."""
        with open(os.path.join(self.tmp, name + '.jpg'), 'wb') as f:
            f.write(b'not really a jpeg')
        cs = (name * 40)[:40]
        fid = self.db.upsert_file_path(name + '.jpg', cs, size=17)
        self.db.conn.commit()
        return fid, cs

    def label_body(self, cs, fid, identity, bbox, **kw):
        """Label person: a media.db body crop + its manual.db ground-truth label.
        Returns (source_body_id, body_identities.id)."""
        emb = _emb(self.rng)
        src = self.db.add_manual_body(fid, bbox, emb, 'clip-test')
        self.db.conn.commit()
        bid = self.manual.add_body_label(cs, identity, bbox, emb, 'clip-test',
                                         source_body_id=src, **kw)
        return src, bid


def _with_env(fn):
    def run():
        fake = FakeEstimator()
        orig = web._get_age_estimator
        web._get_age_estimator = lambda: fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(_Env(tmp), fake)
        finally:
            web._get_age_estimator = orig
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


def _age_rows(manual, cs):
    return {r['face_ref']: r for r in manual.get_age_estimates_for_checksum(cs)}


@_with_env
def test_body_only_photo_is_estimated(env, fake):
    """A named body with no face is sent as 'body:{id}', saved, and shown on the page."""
    fid, cs = env.photo('bodyonly')
    _src, bid = env.label_body(cs, fid, 'Dana', [0, 0, 100, 200])

    r = env.client.post('/api/files/%d/estimate-age' % fid)
    assert r.status_code == 200, r.text
    ref = 'body:%d' % bid
    assert fake.calls == [{'faces': [], 'bodies': [ref]}], fake.calls
    results = r.json()['results']
    assert results == [{'face_ref': ref, 'age': 7.5, 'gender': 'female', 'identity': 'Dana'}], results
    row = _age_rows(env.manual, cs)[ref]
    assert row['age'] == 7.5 and row['gender'] == 'female'

    # Photo page context carries the chip data.
    page = env.client.get('/photo/%d' % fid)
    assert page.status_code == 200, page.text[:500]
    assert page.context['body_ages'] == [
        {'ref': ref, 'identity': 'Dana', 'age': 7.5, 'gender': 'female'}]

    # A manual override is accepted for a body ref on this photo.
    r = env.client.post('/api/files/%d/age-gender' % fid,
                        json={'face_ref': ref, 'age': 9, 'gender': 'female'})
    assert r.status_code == 200, r.text


@_with_env
def test_body_containing_a_face_is_skipped(env, fake):
    """The face's estimate already uses its body, so that body isn't sent again — but
    a second, face-less body on the same photo still is."""
    fid, cs = env.photo('withface')
    _src, inside = env.label_body(cs, fid, 'Eve', [0, 0, 100, 200])
    _src, alone = env.label_body(cs, fid, 'Finn', [300, 0, 400, 200])
    face_id = env.manual.add_manual_face(cs, [20, 10, 50, 40], _emb(env.rng), 500, 300)

    r = env.client.post('/api/files/%d/estimate-age' % fid)
    assert r.status_code == 200, r.text
    assert fake.calls == [{'faces': ['manual:%d' % face_id], 'bodies': ['body:%d' % alone]}], fake.calls
    rows = _age_rows(env.manual, cs)
    assert 'body:%d' % inside not in rows
    assert set(rows) == {'manual:%d' % face_id, 'body:%d' % alone}
    page = env.client.get('/photo/%d' % fid)
    assert [b['ref'] for b in page.context['body_ages']] == ['body:%d' % alone]


@_with_env
def test_rejected_and_face_link_bodies_are_skipped(env, fake):
    """Rejected (hard-negative) and auto face-link bodies aren't estimable — with
    nothing else on the photo the endpoint says so and never calls the model."""
    fid, cs = env.photo('skipped')
    env.label_body(cs, fid, 'Gus', [0, 0, 100, 200], rejected=1)
    env.label_body(cs, fid, 'Hal', [200, 0, 300, 200], source='face-link')
    assert env.manual.get_bodies_for_age(cs) == []

    r = env.client.post('/api/files/%d/estimate-age' % fid)
    assert r.status_code == 200, r.text
    assert r.json() == {'results': [], 'message': 'No faces or labelled bodies on this photo yet.'}
    assert fake.calls == []


@_with_env
def test_deleting_or_redeciding_a_body_drops_its_age(env, fake):
    """Undo (delete decision) and re-deciding the same media.db body (a fresh row id)
    both remove the old 'body:{id}' age row instead of orphaning it."""
    fid, cs = env.photo('deleted')
    src, bid = env.label_body(cs, fid, 'Ivy', [0, 0, 100, 200])
    assert env.client.post('/api/files/%d/estimate-age' % fid).status_code == 200
    assert 'body:%d' % bid in _age_rows(env.manual, cs)

    # Re-decide (relabel) the same body: replaced row, old estimate gone.
    new_bid = env.manual.add_body_label(cs, 'Ivy B', [0, 0, 100, 200], _emb(env.rng),
                                        'clip-test', source_body_id=src)
    assert new_bid != bid
    assert 'body:%d' % bid not in _age_rows(env.manual, cs)

    assert env.client.post('/api/files/%d/estimate-age' % fid).status_code == 200
    assert 'body:%d' % new_bid in _age_rows(env.manual, cs)

    # Undo the decision: body row and its estimate both go.
    assert env.manual.delete_body_decision_by_source(src) == 1
    assert _age_rows(env.manual, cs) == {}
    assert env.manual.get_bodies_for_age(cs) == []


if __name__ == '__main__':
    for t in (test_body_only_photo_is_estimated, test_body_containing_a_face_is_skipped,
              test_rejected_and_face_link_bodies_are_skipped,
              test_deleting_or_redeciding_a_body_drops_its_age):
        t()
        print('ok:', t.__name__)
    print('\nBODY-AGE TESTS PASSED')
