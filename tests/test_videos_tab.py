"""Smoke test for the Videos nav tab (/videos page + /api/videos pages).

Seeds a temp media repo with 7 videos and 3 photos, then hides one video and trashes
another, and asserts:
  - /videos and /api/videos list ONLY videos (no photos), minus hidden + trashed;
  - pages are full (the exclusions are applied before LIMIT/OFFSET) and, walked by
    offset, cover every eligible video exactly once with no overlap;
  - sort=filename (the Python-ranked path) paginates the same way;
  - the Videos nav link is marked active only on /videos;
  - an empty library renders the empty state.

Run with the venv python + libGL on LD_LIBRARY_PATH, e.g.:
  /path/to/.venv/bin/python -m pytest tests/test_videos_tab.py
It also runs standalone as `python tests/test_videos_tab.py`.
"""
import os
import re
import tempfile

from fastapi.testclient import TestClient

from media_manager.web import create_app


def _new_app():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    app.state.test_root = tmp   # where /download looks for the seeded files
    return app


def _seed(app):
    db, _errors, manual = app.state.dbs
    fid = {}
    names = ['v%d.mp4' % i for i in range(5)] + ['v5.mov', 'v6.webm'] + ['p0.jpg', 'p1.png', 'p2.jpg']
    for i, name in enumerate(names):
        fid[name] = db.upsert_file_path(name, 'cs%02d' % i + 'x' * 6, size=100)
    db.conn.commit()
    db.set_file_hidden(fid['v5.mov'])        # hidden (e.g. derived) -> excluded
    manual.add_to_trash('cs01' + 'x' * 6)    # v1.mp4 trashed -> excluded
    return fid


def _ids_in(html):
    return [int(m) for m in re.findall(r'<div class="card" data-file-id="(\d+)"', html)]


def _walk(client, page, **params):
    """Every id the API yields walking offset by `page`, plus the per-page sizes."""
    ids, sizes, offset = [], [], 0
    while True:
        r = client.get('/api/videos', params=dict(offset=offset, limit=page, **params))
        assert r.status_code == 200, r.status_code
        data = r.json()
        assert _ids_in(data['html']) == data['ids'], (data['html'], data['ids'])
        if not data['count']:
            return ids, sizes, data['total']
        ids += data['ids']
        sizes.append(data['count'])
        offset += data['count']


def run():
    app = _new_app()
    fid = _seed(app)
    client = TestClient(app)
    expected = {fid[n] for n in ('v0.mp4', 'v2.mp4', 'v3.mp4', 'v4.mp4', 'v6.webm')}

    # The page: only eligible videos, Videos tab active, queue label set.
    r = client.get('/videos')
    assert r.status_code == 200, r.status_code
    assert set(_ids_in(r.text)) == expected, _ids_in(r.text)
    assert 'data-queue-source="Videos"' in r.text
    assert 'class="nav-link active" href="/videos"' in r.text
    assert 'class="video-badge"' in r.text   # hover frame rotation keys on this
    assert 'class="card-select-checkbox"' in r.text

    # Other pages don't mark the Videos tab active.
    assert 'class="nav-link active"' not in client.get('/files').text

    # API, page size 2 -> pages 2,2,1: full pages, every video once, no photos.
    for params in ({}, {'sort': 'added', 'order': 'asc'}, {'sort': 'filename', 'order': 'asc'}):
        ids, sizes, total = _walk(client, 2, **params)
        assert total == 5, total
        assert sizes == [2, 2, 1], (params, sizes)
        assert len(ids) == len(set(ids)) and set(ids) == expected, (params, ids)

    # Filename sort really orders by name.
    ids, _sizes, _total = _walk(client, 2, sort='filename', order='asc')
    assert ids == [fid[n] for n in ('v0.mp4', 'v2.mp4', 'v3.mp4', 'v4.mp4', 'v6.webm')], ids

    # ⬇ Download: the original bytes as an attachment under the file's own name,
    # and the button on the video's own page.
    vid = fid['v0.mp4']
    abs_path = os.path.join(app.state.test_root, 'v0.mp4')
    with open(abs_path, 'wb') as f:
        f.write(b'fake-mp4-bytes')
    d = client.get('/download/%d' % vid)
    assert d.status_code == 200, d.status_code
    assert d.content == b'fake-mp4-bytes'
    assert d.headers['content-disposition'].startswith('attachment'), d.headers
    assert 'v0.mp4' in d.headers['content-disposition'], d.headers
    page = client.get('/photo/%d' % vid)
    assert page.status_code == 200, page.status_code
    assert 'href="/download/%d"' % vid in page.text
    assert client.get('/download/999999').status_code == 404

    # Empty library -> empty state, no grid.
    empty = TestClient(_new_app()).get('/videos')
    assert empty.status_code == 200 and 'No videos yet' in empty.text
    assert 'id="videos-grid"' not in empty.text

    print('OK  /videos ids:', sorted(expected))
    print('OK  /api/videos pagination (added desc/asc, filename)')
    print('OK  nav active + empty state')
    print('OK  /download attachment + button on the video page')
    print('ALL ASSERTIONS PASSED')


def test_videos_tab():
    run()


if __name__ == '__main__':
    run()
