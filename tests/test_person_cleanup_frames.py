"""Person cleanup sees faces captured from video frames (stills + per-frame faces).

Run: python -m pytest tests/test_person_cleanup_frames.py
"""
import tempfile

import pytest

from test_person_cleanup import BASE, NAME, World, _at, build  # noqa: F401


@pytest.fixture
def w():
    with tempfile.TemporaryDirectory() as tmp:
        yield World(tmp)


def video(w, n):
    cs = ('vid%d' % n).ljust(40, 'v')
    fid = w.db.upsert_file_path('clip%d.mp4' % n, cs, size=10)
    return cs, fid


def still_face(w, vcs, ms, vec, auto_matched=True, trust=None):
    """A captured still (own file row + frame_captures link) with a named face."""
    mid, src = w.promoted(vec, auto_matched=auto_matched, trust=trust)
    still_cs = w.manual.get_face(mid)['checksum']
    w.manual.add_frame_capture(still_cs, vcs, ms)
    return mid, src


def indexed_face(w, vcs, vfid, idx, vec, auto_matched=True):
    """A face found on frame `idx` of the file itself (faces.frame_index)."""
    src = w.db.add_manual_face(vfid, [0, 0, 100, 100], vec.tobytes(), 0.9, frame_index=idx)
    mid = w.manual.promote_auto_face(src, vcs, [0, 0, 100, 100], vec.tobytes(), NAME, None, None,
                                     frame_index=idx, auto_matched=auto_matched)
    w.db.mark_faces_handled([src])
    return mid, src


def test_frame_faces_scored_banded_labelled_and_linked_to_video(w):
    f = build(w)
    vcs, vfid = video(w, 1)
    near, _ = still_face(w, vcs, 12000, _at(w.rng, w.alex, 0.80))
    mid, _ = still_face(w, vcs, 13000, _at(w.rng, w.alex, 0.35))
    far, _ = still_face(w, vcs, 14000, _at(w.rng, w.alex, 0.0))
    p = w.preview()
    assert p['counts']['frame_faces'] == 3 and p['counts']['frame_remove'] == 1
    sample = {s['ref']: s for s in p['remove_sample']}
    s = sample['manual:%d' % far]
    assert s['frame']['label'] == 'video frame 00:14' and s['file_id'] == vfid   # video, not the still
    assert p['remove_groups'] == [{'video_file_id': vfid, 'video_name': 'clip1.mp4', 'count': 1}]
    cards = w.client.post(BASE + '/uncertain', json={'exclude': []}).json()['cards']
    c = next(c for c in cards if c['ref'] == 'manual:%d' % mid)
    assert c['file_id'] == vfid and c['frame']['label'] == 'video frame 00:13'
    # the still never surfaces as a standalone file in the card
    assert c['filename'] == 'clip1.mp4'
    faces = w.client.get(BASE + '/faces', params={'limit': 500}).json()['faces']
    fr = next(x for x in faces if x['face_id'] == near)
    assert fr['frame']['video_file_id'] == vfid and fr['file_id'] == vfid


def test_apply_removes_frame_faces_with_negative_and_undo_restores(w):
    f = build(w)
    vcs, vfid = video(w, 1)
    far, src = still_face(w, vcs, 5000, _at(w.rng, w.alex, 0.0))
    far2, src2 = indexed_face(w, vcs, vfid, 3, _at(w.rng, w.alex, -0.05))
    n = w.preview()['counts']['remove']
    r = w.client.post(BASE + '/apply', json={'keep': .45, 'remove': .25, 'confirm': True, 'expected_remove': n})
    assert r.status_code == 200, r.text
    assert w.manual.get_face(far) is None and w.manual.get_face(far2) is None
    negs = {r_[0] for r_ in w.manual.conn.execute('SELECT source_face_id FROM face_identity_negatives')}
    assert {src, src2} <= negs and src != src2    # two frames of one video do not collide
    assert w.client.post(BASE + '/undo').status_code == 200
    assert w.assigned(far) and w.assigned(far2)
    negs = {r_[0] for r_ in w.manual.conn.execute('SELECT source_face_id FROM face_identity_negatives')}
    assert not ({src, src2} & negs)


def test_trust3_frame_faces_untouched_and_can_be_anchors(w):
    f = build(w)
    vcs, vfid = video(w, 1)
    hand, _ = still_face(w, vcs, 1000, _at(w.rng, w.alex, 0.0), auto_matched=False)   # far, set by hand
    n = w.preview()['counts']['remove']
    w.client.post(BASE + '/apply', json={'keep': .45, 'remove': .25, 'confirm': True, 'expected_remove': n})
    assert w.assigned(hand)
    p = w.preview()
    assert 'manual:%d' % hand not in [s['ref'] for s in p['remove_sample']]
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % hand, 'action': 'remove'})
    assert r.json()['ok'] is False and w.assigned(hand)
    faces = w.client.get(BASE + '/faces', params={'limit': 500}).json()['faces']
    assert next(x for x in faces if x['face_id'] == hand)['is_anchor']


def test_group_decision_applies_to_all_frames_of_one_video_only_when_asked(w):
    f = build(w)
    v1, vf1 = video(w, 1)
    v2, vf2 = video(w, 2)
    a = [still_face(w, v1, 1000 * i, _at(w.rng, w.alex, 0.35))[0] for i in range(1, 4)]
    other, _ = still_face(w, v2, 1000, _at(w.rng, w.alex, 0.35))
    hand, _ = still_face(w, v1, 9000, _at(w.rng, w.alex, 0.35), auto_matched=False)
    cards = w.client.post(BASE + '/uncertain', json={'exclude': []}).json()['cards']
    assert next(c for c in cards if c['face_id'] == a[0])['group_count'] == 3
    # default: per-face
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % a[0], 'action': 'remove'}).json()
    assert r['applied'] == 1 and not w.assigned(a[0]) and w.assigned(a[1]) and w.assigned(a[2])
    # explicit group
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % a[1], 'action': 'remove', 'group': True}).json()
    assert r['applied'] == 2
    assert not w.assigned(a[1]) and not w.assigned(a[2])
    assert w.assigned(other) and w.assigned(hand)       # other video / hand-set untouched
    # undo of a group decision restores the whole batch
    assert w.client.delete(BASE + '/decide/%d' % a[1]).status_code == 200
    assert w.assigned(a[1]) and w.assigned(a[2])


def test_group_keep_approves_frames(w):
    f = build(w)
    v1, _ = video(w, 1)
    a = [still_face(w, v1, 1000 * i, _at(w.rng, w.alex, 0.35))[0] for i in range(1, 3)]
    r = w.client.post(BASE + '/decide', json={'ref': 'manual:%d' % a[0], 'action': 'keep', 'group': True}).json()
    assert r['applied'] == 2
    assert all(w.manual.get_face(i)['trust'] == 2 for i in a)


def test_normal_faces_have_no_frame_info(w):
    f = build(w)
    p = w.preview()
    assert p['counts']['frame_faces'] == 0 and p['remove_groups'] == []
    assert all(s['frame'] is None for s in p['remove_sample'])


def test_face_crop_uses_the_right_frame_of_an_animated_file(tmp_path):
    from PIL import Image
    from media_manager.web import _make_face_crop
    a = Image.new('RGB', (64, 64), (255, 0, 0))
    b = Image.new('RGB', (64, 64), (0, 0, 255))
    src = str(tmp_path / 'anim.gif')
    a.save(src, save_all=True, append_images=[b], duration=100, loop=0)
    out0, out1 = str(tmp_path / 'c0.jpg'), str(tmp_path / 'c1.jpg')
    assert _make_face_crop(src, '[0,0,32,32]', out0)
    assert _make_face_crop(src, '[0,0,32,32]', out1, frame_index=1)
    assert Image.open(out0).getpixel((50, 50))[0] > 200
    assert Image.open(out1).getpixel((50, 50))[2] > 200
    assert not _make_face_crop(src, '[0,0,32,32]', str(tmp_path / 'c2.jpg'), frame_index=7)
