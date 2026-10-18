"""The /files page's ⬆ Upload (POST /api/files/upload).

Files land in the chosen folder under their own name (a clash gets ' (1)' — never an
overwrite), are tracked with the same full-file xxh64 checksum a scan computes, and
show up in that folder's listing. Folders outside the library or inside .media are
refused; non-media files are skipped and reported. Videos only, so the background
indexing never loads CLIP.

Run: python tests/test_files_upload.py   (or python -m pytest tests/test_files_upload.py)
"""
import os
import tempfile

import xxhash
from fastapi.testclient import TestClient

from media_manager.web import create_app


def _upload(client, folder, *named_bytes):
    return client.post('/api/files/upload', data={'folder': folder},
                       files=[('files', (name, data, 'application/octet-stream'))
                              for name, data in named_bytes])


def run():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    os.makedirs(os.path.join(root, 'trips', 'rome'))
    app = create_app(root)
    db = app.state.dbs[0]
    client = TestClient(app)

    # Upload into a subfolder: saved on disk, tracked with the scan's checksum.
    r = _upload(client, 'trips/rome', ('clip.mp4', b'first-video'), ('notes.txt', b'hi'))
    assert r.status_code == 200, r.text
    d = r.json()
    assert [f['name'] for f in d['saved']] == ['clip.mp4'], d
    assert d['skipped'][0]['name'] == 'notes.txt', d
    path = os.path.join(root, 'trips', 'rome', 'clip.mp4')
    assert open(path, 'rb').read() == b'first-video'
    row = db.get_file_by_path('trips/rome/clip.mp4')
    assert row is not None and row['checksum'] == xxhash.xxh64(b'first-video').hexdigest()
    assert not any(n.endswith('.upload') for n in os.listdir(os.path.dirname(path)))

    # Same name again: never overwrites, gets ' (1)'; same bytes elsewhere = duplicate.
    r = _upload(client, 'trips/rome', ('clip.mp4', b'second-video'))
    assert r.json()['saved'][0]['name'] == 'clip (1).mp4', r.json()
    assert open(path, 'rb').read() == b'first-video'
    r = _upload(client, '', ('copy.mp4', b'first-video'))
    assert r.json()['saved'][0]['duplicate'] is True, r.json()
    assert os.path.isfile(os.path.join(root, 'copy.mp4'))

    # The listing for that folder shows the uploads.
    page = client.get('/files/trips/rome').text
    for name in ('clip.mp4', 'clip (1).mp4'):
        fid = db.get_file_by_path('trips/rome/' + name)['id']
        assert '/photo/%d' % fid in page, name

    # Refused targets.
    assert _upload(client, '../escape', ('x.mp4', b'x')).status_code == 400
    assert _upload(client, '.media', ('x.mp4', b'x')).status_code == 400
    assert _upload(client, 'no/such/dir', ('x.mp4', b'x')).status_code == 404
    assert not os.path.exists(os.path.join(os.path.dirname(root), 'escape'))

    # A path-ish filename is reduced to its basename (no writing outside the folder).
    r = _upload(client, 'trips', ('../../evil.mp4', b'evil'))
    assert r.json()['saved'][0]['name'] == 'evil.mp4', r.json()
    assert os.path.isfile(os.path.join(root, 'trips', 'evil.mp4'))

    print('OK  upload into folder, checksum, skip non-media')
    print('OK  name clash -> " (1)", duplicate flag, listing shows uploads')
    print('OK  refuses ../, .media, missing folder; basename-only filenames')
    print('ALL ASSERTIONS PASSED')


def test_files_upload():
    run()


if __name__ == '__main__':
    run()
