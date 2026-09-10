"""Smoke test for the /files listing filters (hide_with_face / hide_with_set).

Seeds a temp media repo with four root-level files:
  a.jpg — has a media.db (auto-detected) face
  b.jpg — belongs to a set
  c.jpg — plain: no face, no set
  d.jpg — has a manual.db named face (no media.db row)

and asserts the two independent toggles remove exactly the right cards.

Run with the venv python + libGL on LD_LIBRARY_PATH, e.g.:
  LD_LIBRARY_PATH=/nix/store/.../lib:$LD_LIBRARY_PATH \
    /path/to/.venv/bin/python -m pytest tests/test_files_filters.py
It also runs standalone as `python tests/test_files_filters.py`.
"""
import os
import tempfile
import time

from fastapi.testclient import TestClient

from media_manager.web import create_app


def _seed(app):
    db, _errors, manual = app.state.dbs

    # Four files sitting directly in the library root.
    fid = {}
    for name in ('a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'):
        cs = name.split('.')[0] * 8  # distinct fake checksum, e.g. 'aaaaaaaa'
        fid[name] = db.upsert_file_path(name, cs, size=100)
    db.conn.commit()

    # a.jpg -> a real (non-sentinel) media.db face (identity NULL == detected face).
    db.add_manual_face(fid['a.jpg'], [1, 2, 3, 4], b'', 0.9)

    # b.jpg -> membership in a set.
    set_id = manual.create_set('shoot', studio=None)
    manual.assign_file_to_set('bbbbbbbb', set_id)

    # d.jpg -> a NAMED manual.db face with no media.db counterpart.
    manual.conn.execute(
        '''INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source,
                              source_face_id, image_width, image_height, created_at)
           VALUES (?, 'Alice', 1,2,3,4, ?, 'manual', NULL, 10, 10, ?)''',
        ('dddddddd', b'', int(time.time())))
    manual.conn.commit()

    return fid


def _shown(client, fid, **params):
    r = client.get('/files', params=params)
    assert r.status_code == 200, r.status_code
    html = r.text
    return {name for name, i in fid.items() if f'data-file-id="{i}"' in html}


def run():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    fid = _seed(app)
    client = TestClient(app)

    # Off by default -> everything shows.
    base = _shown(client, fid)
    assert base == {'a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'}, base

    # hide_with_face=1 -> a (media.db face) and d (manual named face) drop; b, c stay.
    face = _shown(client, fid, hide_with_face=1)
    assert face == {'b.jpg', 'c.jpg'}, face

    # hide_with_set=1 -> only b (set member) drops.
    hset = _shown(client, fid, hide_with_set=1)
    assert hset == {'a.jpg', 'c.jpg', 'd.jpg'}, hset

    # Both on -> only the plain file c survives.
    both = _shown(client, fid, hide_with_face=1, hide_with_set=1)
    assert both == {'c.jpg'}, both

    # The plain file c is present in every combination.
    assert 'c.jpg' in base and 'c.jpg' in face and 'c.jpg' in hset and 'c.jpg' in both

    print('OK  off:', sorted(base))
    print('OK  hide_with_face:', sorted(face))
    print('OK  hide_with_set:', sorted(hset))
    print('OK  both:', sorted(both))
    print('ALL ASSERTIONS PASSED')


def test_files_filters():
    run()


if __name__ == '__main__':
    run()
