""".avi / .mpg / .mpeg are tracked videos: uploadable, thumbnailed (cv2 decodes them),
listed on the Videos tab, served with a video MIME type, and /photo sends them straight
to the H.264 transcode (data-needs-transcode) since no browser plays them inline.

Run: python tests/test_legacy_video_formats.py   (or python -m pytest tests/test_legacy_video_formats.py)
"""
import os
import tempfile

import cv2
import numpy as np
from fastapi.testclient import TestClient

from media_manager.web import create_app


def _make_video(path, fourcc, color):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 25, (64, 48))
    assert w.isOpened(), (path, fourcc)
    for _ in range(10):
        w.write(np.full((48, 64, 3), color, np.uint8))
    w.release()
    return open(path, 'rb').read()


def run():
    src = tempfile.mkdtemp()
    colors = {'old.avi': (0, 0, 255), 'tape.mpg': (0, 255, 0), 'TAPE2.MPEG': (255, 0, 0)}  # BGR
    clips = {
        'old.avi': _make_video(os.path.join(src, 'old.avi'), 'MJPG', colors['old.avi']),
        'tape.mpg': _make_video(os.path.join(src, 'tape.mpg'), 'PIM1', colors['tape.mpg']),
        'TAPE2.MPEG': _make_video(os.path.join(src, 'TAPE2.MPEG'), 'PIM1', colors['TAPE2.MPEG']),
    }
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    client = TestClient(create_app(root))

    r = client.post('/api/files/upload',
                    files=[('files', (n, d, 'application/octet-stream')) for n, d in clips.items()])
    assert r.status_code == 200, r.text
    saved = {f['name']: f['id'] for f in r.json()['saved']}
    assert set(saved) == set(clips), r.json()

    mimes = {'old.avi': 'video/x-msvideo', 'tape.mpg': 'video/mpeg', 'TAPE2.MPEG': 'video/mpeg'}
    for name, fid in saved.items():
        assert client.get('/image/%d' % fid).headers['content-type'] == mimes[name], name
        thumb = client.get('/thumb/%d' % fid)
        img = cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR)
        assert thumb.status_code == 200 and img is not None, name
        center = img[img.shape[0] // 2, img.shape[1] // 2].astype(int)
        assert np.abs(center - colors[name]).max() < 60, (name, center)  # a real frame, not the gray placeholder
        page = client.get('/photo/%d' % fid).text
        assert 'data-needs-transcode="1"' in page, name

    ids = set(client.get('/api/videos?limit=50').json()['ids'])
    assert set(saved.values()) <= ids, ids

    print('OK  .avi/.mpg/.mpeg upload, thumbnail, MIME type, Videos tab')
    print('OK  /photo goes straight to the H.264 transcode for them')
    print('ALL ASSERTIONS PASSED')


def test_legacy_video_formats():
    run()


if __name__ == '__main__':
    run()
