"""⬆ Upload (POST /api/files/upload) — the Files tab button and plain curl.

Every upload lands in uploads/<first 2 hex of its xxh64>/ (256 buckets, so no one
folder grows huge), tracked with the same full-file checksum a scan computes. Content
already in the library — or repeated within one request — is rejected, never saved.
Accepts the multipart field as `file` (curl -F file=@x) or `files` (the web UI).
Videos only, so the background indexing never loads CLIP.

Run: python tests/test_files_upload.py   (or python -m pytest tests/test_files_upload.py)
"""
import os
import tempfile

import xxhash
from fastapi.testclient import TestClient

from media_manager.web import create_app


def _upload(client, *named_bytes, field='file'):
    return client.post('/api/files/upload',
                       files=[(field, (name, data, 'application/octet-stream'))
                              for name, data in named_bytes])


def run():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'), exist_ok=True)
    app = create_app(root)
    db = app.state.dbs[0]
    client = TestClient(app)

    # curl-style single upload: saved under uploads/<bucket>/, tracked, scan checksum.
    r = _upload(client, ('clip.mp4', b'first-video'))
    assert r.status_code == 200, r.text
    saved = r.json()['saved']
    cs = xxhash.xxh64(b'first-video').hexdigest()
    assert saved[0]['path'] == 'uploads/%s/clip.mp4' % cs[:2], saved
    path = os.path.join(root, saved[0]['path'])
    assert open(path, 'rb').read() == b'first-video'
    row = db.get_file_by_path(saved[0]['path'])
    assert row is not None and row['checksum'] == cs

    # Same content again (any name) -> rejected with 409, nothing new on disk/in db.
    r = _upload(client, ('renamed.mp4', b'first-video'))
    assert r.status_code == 409, r.status_code
    d = r.json()
    assert d['saved'] == [] and d['duplicates'][0]['existing_path'] == saved[0]['path'], d
    assert not os.path.exists(os.path.join(os.path.dirname(path), 'renamed.mp4'))

    # Content already tracked from a scan elsewhere in the library -> also rejected.
    db.upsert_file_path('trips/old.mp4', xxhash.xxh64(b'scanned-video').hexdigest(), size=13)
    db.conn.commit()
    assert _upload(client, ('copy.mp4', b'scanned-video')).status_code == 409

    # Batch (web UI field name): new + in-batch repeat + already present + non-media.
    r = _upload(client, ('a.mp4', b'aaa'), ('a-again.mp4', b'aaa'), ('dup.mp4', b'first-video'),
                ('notes.txt', b'hi'), field='files')
    assert r.status_code == 200, r.text
    d = r.json()
    assert [f['name'] for f in d['saved']] == ['a.mp4'], d
    assert sorted(f['name'] for f in d['duplicates']) == ['a-again.mp4', 'dup.mp4'], d
    assert d['skipped'][0]['name'] == 'notes.txt', d

    # Same name, different content in the same bucket never overwrites.
    other = b'x' * 7
    while xxhash.xxh64(other).hexdigest()[:2] != cs[:2]:
        other += b'x'
    r = _upload(client, ('clip.mp4', other))
    assert r.json()['saved'][0]['name'] == 'clip (1).mp4', r.json()
    assert open(path, 'rb').read() == b'first-video'

    # Nothing usable -> 400; no files at all -> 400. A path-ish name keeps only its basename.
    assert _upload(client, ('notes.txt', b'hi')).status_code == 400
    assert client.post('/api/files/upload').status_code in (400, 422)
    r = _upload(client, ('../../evil.mp4', b'evil'))
    assert r.json()['saved'][0]['path'].startswith('uploads/') and r.json()['saved'][0]['name'] == 'evil.mp4'

    # A '"' arrives as %22 (browser/curl form encoding) and is stored as '"'.
    r = _upload(client, ('say %22hi%22.mp4', b'quoted'))
    assert r.json()['saved'][0]['name'] == 'say "hi".mp4', r.json()

    # No half-written temp files left behind; uploads show up in the Files tab.
    assert os.listdir(os.path.join(root, 'uploads', '.incoming')) == []
    page = client.get('/files/uploads/' + cs[:2]).text
    assert '/photo/%d' % saved[0]['id'] in page

    print('OK  curl-style upload -> uploads/<bucket>/, scan checksum')
    print('OK  duplicates rejected (409): repeat, in-batch, already scanned elsewhere')
    print('OK  no overwrite, non-media skipped, 400 when nothing usable, no temp leftovers')
    print('ALL ASSERTIONS PASSED')


def test_files_upload():
    run()


if __name__ == '__main__':
    run()
