"""'Remove auto tags' on /photo must remove ONLY the detector's unruled-on guesses.

The two kinds of chip in that list come from different databases, which is what makes
the guarantee structural rather than a filter someone can get wrong later:

  media.db  detections   auto-detected candidates (the gray chips)  <- removable
  manual.db file_tags    anything a human confirmed, typed, rejected
                         or drew a region around                    <- never touched

Also pins the two things that are easy to break: the '__indexed__' sentinel must
survive (or `media index` simply regenerates everything that was just dismissed), and
frame-scoped detections must survive (they belong to individual video frames reviewed
elsewhere, so clearing them from the photo page would delete things off-screen).

Run standalone: python tests/test_clear_auto_tags.py
"""
import os
import tempfile
import time

from fastapi.testclient import TestClient

from media_manager.web import create_app


def _detect(db, file_id, class_name, conf, frame_index=None):
    db.conn.execute(
        'INSERT INTO detections (file_id, class_name, confidence, x1, y1, x2, y2, '
        'model, indexed_at, frame_index) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (file_id, class_name, conf, 1.0, 2.0, 30.0, 40.0, 'yolo-test', int(time.time()),
         frame_index))
    db.conn.commit()


def run():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, '.media'), exist_ok=True)
    app = create_app(tmp)
    db, _errors, manual = app.state.dbs

    cs = 'a' * 8
    file_id = db.upsert_file_path('a.jpg', cs, size=100)
    db.conn.commit()

    # Auto: three primary-frame guesses plus one that belongs to a video frame.
    for cls, conf in (('dog', 0.9), ('car', 0.8), ('tree', 0.7)):
        _detect(db, file_id, cls, conf)
    _detect(db, file_id, 'boat', 0.6, frame_index=12)

    # Human: a typed tag, a confirmed one, a rejection, and a region the user drew.
    manual.add_tag(cs, 'holiday')
    manual.add_tag(cs, 'car')                      # same label, but CONFIRMED by a human
    manual.add_tag(cs, 'bicycle', polarity='negative')
    manual.add_spatial_tag(cs, 'monstera plant', 10, 10, 50, 50, 100, 100)

    client = TestClient(app)
    r = client.delete('/api/files/%d/detections' % file_id)
    assert r.status_code == 200, (r.status_code, r.text[:200])
    data = r.json()

    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        ok = ok and cond
        print(('  PASS  ' if cond else '  FAIL  ') + name + (('   [%s]' % (extra,)) if extra else ''))

    check('reports how many auto tags went', data['removed'] == 3, repr(data['removed']))
    check('no auto candidates left on the photo', data['detected_classes'] == [],
          repr(data['detected_classes']))
    check('primary-frame detections are gone',
          db.get_detected_classes(file_id) == [], repr(db.get_detected_classes(file_id)))

    labels = {(t['label'], t['polarity'], t['located']) for t in data['tags']}
    check('a hand-typed tag survives', ('holiday', 'positive', False) in labels, str(sorted(labels)))
    check('a CONFIRMED tag survives even though the detector also guessed it',
          ('car', 'positive', False) in labels, str(sorted(labels)))
    check('a rejection survives', ('bicycle', 'negative', False) in labels, str(sorted(labels)))
    check('a user-drawn REGION tag survives', ('monstera plant', 'positive', True) in labels,
          str(sorted(labels)))

    # The sentinel: without it the next `media index` re-detects the file and every
    # dismissed guess comes straight back.
    sentinel = db.conn.execute(
        "SELECT COUNT(*) FROM detections WHERE file_id = ? AND frame_index IS NULL "
        "AND class_name = '__indexed__'", (file_id,)).fetchone()[0]
    check('the file still counts as detection-indexed', sentinel == 1, repr(sentinel))
    check('and so is not re-queued for detection',
          file_id not in [r[0] for r in db.get_undetected_files()])

    frame_rows = db.conn.execute(
        'SELECT class_name FROM detections WHERE file_id = ? AND frame_index IS NOT NULL',
        (file_id,)).fetchall()
    check('frame-scoped detections are left alone', [r[0] for r in frame_rows] == ['boat'],
          str([r[0] for r in frame_rows]))

    # Idempotent: pressing it again on a clean photo is a no-op, not a second sentinel.
    again = client.delete('/api/files/%d/detections' % file_id).json()
    sentinel2 = db.conn.execute(
        "SELECT COUNT(*) FROM detections WHERE file_id = ? AND class_name = '__indexed__'",
        (file_id,)).fetchone()[0]
    check('a second press removes nothing', again['removed'] == 0, repr(again['removed']))
    check('and does not duplicate the sentinel', sentinel2 == 1, repr(sentinel2))

    print('\noverall:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())
