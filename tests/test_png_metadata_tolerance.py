"""PNGs that image viewers open must not be flagged damaged.

Pillow refuses a whole PNG for problems viewers ignore: a compressed text chunk over
its MAX_TEXT_CHUNK (big AI-workflow / XMP metadata) or a bad CRC on a metadata chunk.
formats.open_rgb retries without those chunks, so thumbnails, verify_image and CLIP
treat them as healthy; genuinely broken pixel data is still damaged. The /trash
"Re-check damaged files" job clears earlier false flags.

Run: python tests/test_png_metadata_tolerance.py   (or python -m pytest tests/test_png_metadata_tolerance.py)
"""
import io
import os
import struct
import tempfile
import time
import zlib

import xxhash
from fastapi.testclient import TestClient
from PIL import Image

from media_manager import broken_finder
from media_manager.formats import open_rgb, png_without_bad_metadata
from media_manager.web import _make_thumbnail, create_app


def _chunk(ctype, data, crc=None):
    c = zlib.crc32(ctype + data) & 0xFFFFFFFF if crc is None else crc
    return struct.pack('>I', len(data)) + ctype + data + struct.pack('>I', c)


def _png(color=(200, 50, 50)):
    b = io.BytesIO()
    Image.new('RGB', (64, 48), color).save(b, 'PNG')
    return b.getvalue()


def _insert(png, chunk):
    i = png.index(b'IDAT') - 4
    return png[:i] + chunk + png[i:]


def cases():
    png = _png()
    big_ztxt = _insert(png, _chunk(b'zTXt', b'workflow\0\0' + zlib.compress(b'x' * 3_000_000)))
    bad_crc = _insert(png, _chunk(b'tEXt', b'Comment\0hi', crc=0))
    i = png.index(b'IDAT') + 4
    broken_pixels = png[:i] + b'\xff' * 40 + png[i + 40:]      # garbage inside IDAT
    return big_ztxt, bad_crc, broken_pixels


def run():
    big_ztxt, bad_crc, broken_pixels = cases()
    d = tempfile.mkdtemp()
    paths = {}
    for name, data in (('big.png', big_ztxt), ('crc.png', bad_crc), ('bad.png', broken_pixels)):
        paths[name] = os.path.join(d, name)
        open(paths[name], 'wb').write(data)

    assert png_without_bad_metadata(b'not a png') is None
    for name in ('big.png', 'crc.png'):
        assert open_rgb(paths[name]).getpixel((5, 5)) == (200, 50, 50), name
        assert broken_finder.verify_image(paths[name])[0] is True, name
        assert _make_thumbnail(paths[name], os.path.join(d, 't.jpg'))[0] == 'ok', name
    assert broken_finder.verify_image(paths['bad.png'])[0] is False
    assert _make_thumbnail(paths['bad.png'], os.path.join(d, 't.jpg'))[0] == 'damaged'
    print('OK  big zTXt / bad-CRC metadata PNGs decode; broken pixel data is still damaged')

    # Re-check clears an old false flag, keeps the real one.
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    app = create_app(root)
    db = app.state.dbs[0]
    ids = {}
    for name in ('crc.png', 'bad.png'):
        data = open(paths[name], 'rb').read()
        open(os.path.join(root, name), 'wb').write(data)
        cs = xxhash.xxh64(data).hexdigest()
        db.upsert_file_path(name, cs, size=len(data), modified_time=1)
        ids[name] = db.find_file_id_by_checksum(cs)
        db.mark_broken(ids[name])
    db.conn.commit()
    client = TestClient(app)
    assert client.post('/api/broken/recheck/start').json()['started'] is True
    for _ in range(200):
        s = client.get('/api/broken/recheck/status').json()
        if not s['running']:
            break
        time.sleep(0.05)
    assert s['cleared'] == 1 and s['total'] == 2 and s['error'] is None, s
    assert db.get_file_by_id(ids['crc.png'])['broken'] is None
    assert db.get_file_by_id(ids['bad.png'])['broken'] is not None
    assert 'damaged-recheck' in client.get('/trash').text
    print('OK  re-check clears the false flag, keeps the genuinely broken file')
    print('ALL ASSERTIONS PASSED')


def test_png_metadata_tolerance():
    run()


if __name__ == '__main__':
    run()
