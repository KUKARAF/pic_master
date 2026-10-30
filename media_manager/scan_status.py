"""How far each tracked file has got through the ML pipeline — the engine behind the
file manager's per-file / per-folder state colouring.

Every stage's "done" test mirrors the pending query its bulk job uses (database.py
get_un*_files + the web.py wrappers around them), so a file the file manager shows as
missing stage X is exactly a file the /bulk page counts as pending for X:

  exif        files.metadata_checked_at IS NOT NULL         (get_files_without_metadata)
  clip        embeddings row with frame_index = 0            (get_unindexed_files)
  tags        detections row with frame_index IS NULL; the  (get_undetected_files)
              '__indexed__' sentinel counts
  faces       files.noface = 1, or a faces row with         (get_unface_indexed_files)
              frame_index IS NULL (sentinel bbox '[]' counts)
  bodies      body_embeddings row, frame_index IS NULL       (get_unbody_indexed_files)
              (sentinel bbox '[]' counts)
  objects     any object_region_embeddings row (sentinel)    (get_unobject_indexed_files)
  region      files.hidden = 1, or region_windows row with   (get_unregion_indexed_files)
              algo = region_index.ALGO_TAG
  pattern     files.hidden = 1, or pattern_windows row with  (get_unpattern_indexed_files)
              algo = pattern_deep.ALGO_TAG
  phash       any phashes row                                (get_unphashed_files)
  tiles       any tile_embeddings row (optional stage)       (get_untiled_files)
  stills      files.hidden = 1, or >= 3 frame_captures rows  (web _videos_needing_frames)
              whose parent_checksum is the video's checksum
  video_faces files.hidden = 1, or video_frame_scans has     (web _videos_needing_face_scan)
              the video's checksum

Image stages apply to formats.IMAGE_EXTENSIONS, video stages to VIDEO_EXTENSIONS;
anything else is kind 'other' with no stages.

Two lookup strategies, identical results: small inputs query each table with
`file_id IN (...)` (chunked); large inputs load every table's full done-set once and
keep it in a short-TTL in-process cache (see invalidate()). No per-file queries.
"""
import os
import threading
import time

from . import pattern_deep, region_index
from .database import ThreadLocalDB
from .formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

STAGES = {
    'image': [  # (key, label, required)
        ('exif', 'EXIF', True),
        ('clip', 'Similarity embedding (CLIP)', True),
        ('tags', 'Auto-tags', True),
        ('faces', 'Faces', True),
        ('bodies', 'Bodies', True),
        ('objects', 'Objects', True),
        ('region', 'Region index', True),
        ('pattern', 'Pattern index', True),
        ('phash', 'pHash', True),
        ('tiles', 'Tiles (location / region matching)', False),
    ],
    'video': [
        # One CLIP vector per video (mean of a few sampled frames), in the same
        # embeddings table as photos — what /similar and set tinder search over.
        ('clip', 'Similarity embedding (CLIP)', True),
        ('stills', 'Captured stills (≥3)', True),
        ('video_faces', 'Video face scan', True),
    ],
}

STATES = ('broken', 'scanned', 'partial', 'tracked')

# Above this many ids, load whole done-sets (cached) instead of IN (...) lookups.
LARGE_THRESHOLD = 2000
# Seconds a full done-set snapshot stays fresh; invalidate() busts it immediately.
CACHE_TTL = 30.0
# Matches web._videos_needing_frames (a video with fewer stills is re-captured).
MIN_STILLS = 3

# (table, extra WHERE condition) — the done-set for each table-backed stage. The
# conditions are the LEFT JOIN conditions of the matching get_un*_files query.
_TABLE_STAGES = {
    'clip': ('embeddings', 'frame_index = 0'),
    'tags': ('detections', 'frame_index IS NULL'),
    'faces': ('faces', 'frame_index IS NULL'),
    'bodies': ('body_embeddings', 'frame_index IS NULL'),
    'objects': ('object_region_embeddings', None),
    'region': ('region_windows', 'algo = ?'),
    'pattern': ('pattern_windows', 'algo = ?'),
    'phash': ('phashes', None),
    'tiles': ('tile_embeddings', None),
}

_FILE_COLS = 'id, checksum, broken, noface, hidden, metadata_checked_at'

_cache = {}
_cache_lock = threading.Lock()
_build_lock = threading.Lock()


def invalidate():
    """Drop every cached full done-set snapshot (call after a job writes stage rows)."""
    with _cache_lock:
        _cache.clear()


def _algo_params(key):
    if key == 'region':
        return (region_index.ALGO_TAG,)
    if key == 'pattern':
        return (pattern_deep.ALGO_TAG,)
    return ()


def _stage_sql(key, with_in=None):
    table, cond = _TABLE_STAGES[key]
    conds = [cond] if cond else []
    if with_in is not None:
        conds.append(f'file_id IN ({",".join("?" * with_in)})')
    where = f' WHERE {" AND ".join(conds)}' if conds else ''
    return f'SELECT DISTINCT file_id FROM {table}{where}'


def _kind(path):
    ext = os.path.splitext(path or '')[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    return 'other'


def _load_full(db, manual):
    """Every lookup structure for the whole library, one query per table."""
    conn = db.conn
    files = {r[0]: tuple(r[1:]) for r in conn.execute(f'SELECT {_FILE_COLS} FROM files')}
    sets = {key: {r[0] for r in conn.execute(_stage_sql(key), _algo_params(key))}
            for key in _TABLE_STAGES}
    mconn = manual.conn
    captures = {r[0]: r[1] for r in mconn.execute(
        'SELECT parent_checksum, COUNT(*) FROM frame_captures GROUP BY parent_checksum')}
    vscanned = {r[0] for r in mconn.execute('SELECT checksum FROM video_frame_scans')}
    return {'files': files, 'sets': sets, 'captures': captures, 'vscanned': vscanned}


def _load_subset(db, manual, ids, image_ids, video_ids):
    """Same structures as _load_full, restricted to the given ids (chunked IN lookups)."""
    conn = db.conn
    files = {}
    for chunk in ThreadLocalDB._chunked(ids):
        sql = f'SELECT {_FILE_COLS} FROM files WHERE id IN ({",".join("?" * len(chunk))})'
        for r in conn.execute(sql, chunk):
            files[r[0]] = tuple(r[1:])
    sets = {key: set() for key in _TABLE_STAGES}
    for key in _TABLE_STAGES:
        for chunk in ThreadLocalDB._chunked(ids if key == 'clip' else image_ids):
            sql = _stage_sql(key, with_in=len(chunk))
            sets[key].update(r[0] for r in conn.execute(sql, _algo_params(key) + tuple(chunk)))
    checksums = [files[i][0] for i in video_ids if i in files]
    captures, vscanned = {}, set()
    mconn = manual.conn
    for chunk in ThreadLocalDB._chunked(checksums):
        ph = ','.join('?' * len(chunk))
        for r in mconn.execute('SELECT parent_checksum, COUNT(*) FROM frame_captures '
                               f'WHERE parent_checksum IN ({ph}) GROUP BY parent_checksum', chunk):
            captures[r[0]] = r[1]
        vscanned.update(r[0] for r in mconn.execute(
            f'SELECT checksum FROM video_frame_scans WHERE checksum IN ({ph})', chunk))
    return {'files': files, 'sets': sets, 'captures': captures, 'vscanned': vscanned}


def _cached_full(db, manual):
    key = (getattr(db, 'db_path', id(db)), getattr(manual, 'db_path', id(manual)),
           region_index.ALGO_TAG, pattern_deep.ALGO_TAG)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    # One builder at a time: concurrent requests for a big folder wait for the first
    # load instead of each scanning every table.
    with _build_lock:
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None and hit[0] > time.monotonic():
            return hit[1]
        data = _load_full(db, manual)
        with _cache_lock:
            _cache[key] = (time.monotonic() + CACHE_TTL, data)
        return data


def _status(fid, kind, row, data):
    """Status dict for one file given its files row (or None) and the lookups."""
    checksum, broken, noface, hidden, meta_at = row if row is not None else (None,) * 5
    sets = data['sets']
    stages = {}
    if kind == 'image':
        stages['exif'] = meta_at is not None
        stages['clip'] = fid in sets['clip']
        stages['tags'] = fid in sets['tags']
        stages['faces'] = bool(noface) or fid in sets['faces']
        stages['bodies'] = fid in sets['bodies']
        stages['objects'] = fid in sets['objects']
        stages['region'] = bool(hidden) or fid in sets['region']
        stages['pattern'] = bool(hidden) or fid in sets['pattern']
        stages['phash'] = fid in sets['phash']
        stages['tiles'] = fid in sets['tiles']
    elif kind == 'video':
        stages['clip'] = fid in sets['clip']
        stages['stills'] = bool(hidden) or data['captures'].get(checksum, 0) >= MIN_STILLS
        stages['video_faces'] = bool(hidden) or checksum in data['vscanned']
    missing = [label for key, label, req in STAGES.get(kind, ()) if req and not stages[key]]
    if broken is not None:
        state = 'broken'
    elif kind == 'other' or not any(stages.values()):
        state = 'tracked'
    elif not missing:
        state = 'scanned'
    else:
        state = 'partial'
    return {'kind': kind, 'state': state, 'stages': stages, 'missing': missing}


def status_for_files(db, manual, files, large_threshold=None):
    """{file_id: {'kind', 'state', 'stages', 'missing'}} for tracked (file_id, path) pairs.

    state: 'broken' (files.broken set; wins over everything), 'scanned' (every required
    stage done), 'tracked' (no stage done at all, optional ones included; always for kind
    'other'), else 'partial'. `large_threshold` overrides LARGE_THRESHOLD (tests)."""
    kinds = {}
    for fid, path in files:
        kinds[fid] = _kind(path)
    if not kinds:
        return {}
    threshold = LARGE_THRESHOLD if large_threshold is None else large_threshold
    if len(kinds) > threshold:
        data = _cached_full(db, manual)
    else:
        image_ids = [i for i, k in kinds.items() if k == 'image']
        video_ids = [i for i, k in kinds.items() if k == 'video']
        data = _load_subset(db, manual, list(kinds), image_ids, video_ids)
    rows = data['files']
    out = {}
    for fid, kind in kinds.items():
        out[fid] = _status(fid, kind, rows.get(fid), data)
    return out


def summarize(statuses):
    """{'total': n, 'broken': n, 'scanned': n, 'partial': n, 'tracked': n} for a folder
    rollup. Accepts status_for_files' dict or any iterable of its status dicts."""
    values = statuses.values() if isinstance(statuses, dict) else statuses
    counts = dict.fromkeys(STATES, 0)
    total = 0
    for st in values:
        counts[st['state']] += 1
        total += 1
    counts['total'] = total
    return counts
