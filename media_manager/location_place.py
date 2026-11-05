"""People-aware "place" scoring for location suggestions (stage 1: stored data only).

The default location ranking ('combined' = DINOv2 + SigLIP 2 z-score fusion) happily matches
the same PEOPLE in a different place. The 'place' model keeps that fused score as the base
and corrects it with signals that are already in the databases (no re-scan, no GPU):

  (a) person area   - fraction of the frame covered by the union of face boxes (expanded to a
                      body-ish box) and person-like detection boxes (incl. crowd / group of
                      people). Candidates dominated by people are down-weighted smoothly.
                      Photos never scanned for persons (no detections row / sentinel and no
                      faces row) are NOT presumed person-free: they get no clean bonus and are
                      counted in `person_scan_coverage`.
  (b) same people   - identities present in the location's own photos (named faces + whole-photo
                      assignments) damp candidates that show them, unless the candidate is a
                      place-wise near-duplicate (very high raw similarity).
  (c) co-signals    - GPS (boost near the location's point / geotagged members, veto candidates
                      geotagged far away - never penalising photos without GPS), taken_at within
                      a few hours of a member (same shoot), shared set / directory with members.

All corrections are in "fused z-score units" (the base rank is a weighted mean of per-space
z-scores over the whole library, so 1.0 is one standard deviation of cosine). They only touch
the top PLACE_RERANK_POOL candidates of the base ranking; everything below keeps its base rank.
Every constant below is a plain module attribute: tune them against scripts/eval_location_place.py.
"""
import json
import math
import threading
import time
from collections import Counter, defaultdict

import numpy as np

# ---- (a) person area -------------------------------------------------------------------
PERSON_BOX_MIN_CONF = 0.3          # detection confidence floor (as get_person_boxes_by_file)
FACE_MIN_SCORE = 0.5               # faces.det_score floor
EXTRA_PERSON_CLASSES = ('crowd', 'group of people')   # not in body_index.PERSON_LIKE_CLASSES
PERSON_GRID = 24                   # raster resolution of the union-area estimate
# face box -> body-ish box, in multiples of the face box width / height. A person-like
# detection box that already contains the face centre replaces this expansion.
FACE_BODY_SIDE = 0.8               # each side
FACE_BODY_UP = 0.4                 # above the face
FACE_BODY_DOWN = 3.5               # below the face (waist-up portrait)
PERSON_AREA_FREE = 0.05            # coverage below this costs nothing
PERSON_AREA_FULL = 0.45            # coverage at/above this costs the full penalty
PERSON_AREA_PENALTY = 1.5          # z units at full coverage (smoothstep in between)
CLEAN_BONUS = 0.25                 # z bonus for photos scanned and (almost) person-free;
                                   # unscanned photos never get it
REBUILD_MIN_S = 30.0               # min seconds between rebuilds while scans keep writing

# ---- (b) same-people damping -------------------------------------------------------------
PEOPLE_LAMBDA = 1.2                # z units at prevalence 1.0
PEOPLE_MIN_COUNT = 2               # an identity counts if in >= this many member photos...
PEOPLE_MIN_PREV = 0.5              # ...or in >= this fraction of them (tiny locations)
NEAR_DUP_COS_LO = 0.78             # raw fused cosine: damping fades out between LO and HI
NEAR_DUP_COS_HI = 0.90             # (place-wise near duplicates keep their people)

# ---- (c) geo / time / set co-signals -----------------------------------------------------
GEO_MIN_MEMBERS = 3                # geotagged members needed when the location has no point
GEO_MAX_REFS = 64                  # member points used as references (evenly sampled)
GEO_BOOST = 0.8                    # at <= GEO_NEAR_KM, tapering to 0 at GEO_RADIUS_KM
GEO_NEAR_KM = 2.0
GEO_RADIUS_KM = 25.0
GEO_VETO_KM = 100.0                # geotagged this far from every reference: veto...
GEO_VETO_PENALTY = 3.0             # ...by this many z (soft, so a wrong EXIF is not fatal)
GEO_TIGHT_KM = 30.0                # members must sit this close to their median to allow a
                                   # veto when the location has no explicit point
TIME_WINDOW_S = 6 * 3600           # same shoot
TIME_BOOST = 0.35
SET_BOOST = 0.4                    # shares a set with >= SET_FULL_MEMBERS members: full boost
DIR_BOOST = 0.3                    # same directory, likewise
CO_FULL_MEMBERS = 3

PLACE_RERANK_POOL = 3000           # how many of the base ranking's best are corrected


# ================================================================================ geometry

def union_fraction(boxes, w, h, grid=PERSON_GRID):
    """Fraction (0..1) of a w x h frame covered by the union of pixel boxes, estimated on a
    grid x grid raster with fractional cell coverage (so tiny boxes still count)."""
    if not boxes or w <= 0 or h <= 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, grid + 1)
    lo, hi = edges[:-1], edges[1:]
    acc = np.zeros((grid, grid), np.float32)
    for x1, y1, x2, y2 in boxes:
        nx1, nx2 = max(0.0, x1 / w), min(1.0, x2 / w)
        ny1, ny2 = max(0.0, y1 / h), min(1.0, y2 / h)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        xc = np.clip(np.minimum(nx2, hi) - np.maximum(nx1, lo), 0, None) * grid
        yc = np.clip(np.minimum(ny2, hi) - np.maximum(ny1, lo), 0, None) * grid
        np.maximum(acc, np.outer(yc, xc), out=acc)
    return float(acc.mean())


def _fits(boxes, w, h):
    return all(b[2] <= w * 1.05 and b[3] <= h * 1.05 for b in boxes)


def person_regions(face_boxes, person_boxes, w, h):
    """(regions, (w, h)): pixel boxes that contain people - every person-like box plus each
    face expanded to a body-ish box (unless a person box already holds the face centre) -
    clipped to the frame. (w, h) may come back swapped when the stored dimensions are the
    un-rotated ones and the boxes are upright (boxes are upright pixel coordinates)."""
    allb = list(face_boxes) + list(person_boxes)
    if allb and w > 0 and h > 0 and not _fits(allb, w, h) and _fits(allb, h, w):
        w, h = h, w
    out = [list(map(float, b)) for b in person_boxes]
    for x1, y1, x2, y2 in face_boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in person_boxes):
            continue
        fw, fh = x2 - x1, y2 - y1
        out.append([x1 - FACE_BODY_SIDE * fw, y1 - FACE_BODY_UP * fh,
                    x2 + FACE_BODY_SIDE * fw, y2 + FACE_BODY_DOWN * fh])
    if w > 0 and h > 0:
        out = [[max(0.0, a), max(0.0, b), min(float(w), c), min(float(h), d)] for a, b, c, d in out]
        out = [b for b in out if b[2] > b[0] and b[3] > b[1]]
    return out, (w, h)


def person_penalty_curve(area):
    """0..1 smoothstep of person coverage between PERSON_AREA_FREE and PERSON_AREA_FULL."""
    x = np.clip((np.asarray(area, np.float32) - PERSON_AREA_FREE) / (PERSON_AREA_FULL - PERSON_AREA_FREE), 0, 1)
    return x * x * (3 - 2 * x)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (numpy broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================================ person store

class PersonStore:
    """Resident per-file arrays indexed by file id: `area` (float16 person coverage),
    `scanned` (bool: a detections row / sentinel or a faces row exists AND the dimensions
    needed to use any boxes are known). Rebuilt when db._face_ver / db._det_ver change, but
    at most every REBUILD_MIN_S so a running scan does not trigger a rebuild per request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.built_at = 0.0
        self.area = np.zeros(1, np.float16)
        self.scanned = np.zeros(1, bool)
        self.total = 0
        self.builds = 0

    def get(self, db):
        key = (getattr(db, '_face_ver', 0), getattr(db, '_det_ver', 0))
        with self.lock:
            if self.key == key:
                return self
            if self.key is not None and time.time() - self.built_at < REBUILD_MIN_S:
                return self
            self._build(db)
            self.key = key
            self.built_at = time.time()
            self.builds += 1
            return self

    def _build(self, db):
        from media_manager import body_index  # hook-allow: lazy -- PIL/compute imports
        conn = db.conn
        max_id = conn.execute('SELECT COALESCE(MAX(id), 0) FROM files').fetchone()[0]
        n = max_id + 1
        dims = np.zeros((n, 2), np.int32)
        for fid, w, h in conn.execute('SELECT id, width, height FROM files WHERE width > 0 AND height > 0'):
            dims[fid] = (w, h)
        for fid, w, h in conn.execute('SELECT file_id, width, height FROM phashes '
                                      'WHERE width > 0 AND height > 0'):
            if fid < n and dims[fid, 0] == 0:
                dims[fid] = (w, h)
        classes = tuple(body_index.PERSON_LIKE_CLASSES) + EXTRA_PERSON_CLASSES
        q = ','.join('?' * len(classes))
        persons, faces = defaultdict(list), defaultdict(list)
        for fid, x1, y1, x2, y2 in conn.execute(
                f'SELECT file_id, x1, y1, x2, y2 FROM detections WHERE class_name IN ({q}) '
                'AND confidence >= ? AND frame_index IS NULL AND x1 IS NOT NULL',
                (*classes, PERSON_BOX_MIN_CONF)):
            persons[fid].append([x1, y1, x2, y2])
        for fid, bbox in conn.execute(
                "SELECT file_id, bbox FROM faces WHERE frame_index IS NULL AND det_score >= ? "
                "AND (identity IS NULL OR identity != '__indexed__')", (FACE_MIN_SCORE,)):
            try:
                b = json.loads(bbox)
            except (TypeError, ValueError):
                continue
            if isinstance(b, list) and len(b) == 4:
                faces[fid].append([float(v) for v in b])
        scanned = np.zeros(n, bool)
        for (fid,) in conn.execute('SELECT DISTINCT file_id FROM detections WHERE frame_index IS NULL'):
            scanned[fid] = True
        for (fid,) in conn.execute('SELECT DISTINCT file_id FROM faces'):
            scanned[fid] = True
        area = np.zeros(n, np.float16)
        for fid in set(persons) | set(faces):
            w, h = int(dims[fid, 0]), int(dims[fid, 1])
            if w <= 0:
                scanned[fid] = False      # boxes we cannot normalise: unknown, not "clean"
                continue
            regions, (w2, h2) = person_regions(faces.get(fid, ()), persons.get(fid, ()), w, h)
            area[fid] = union_fraction(regions, w2, h2)
        self.area, self.scanned = area, scanned
        self.total = int(conn.execute('SELECT COUNT(*) FROM files').fetchone()[0])

    def lookup(self, file_ids):
        """(area float32[k], scanned bool[k]) for an array of file ids (unknown ids: 0, False)."""
        ids = np.asarray(file_ids, dtype=np.int64)
        ok = (ids >= 0) & (ids < len(self.area))
        area = np.zeros(len(ids), np.float32)
        sc = np.zeros(len(ids), bool)
        area[ok] = self.area[ids[ok]]
        sc[ok] = self.scanned[ids[ok]]
        return area, sc


def store_for(db):
    s = getattr(db, '_place_store', None)
    if s is None:
        s = db._place_store = PersonStore()
    return s


# ================================================================================ signals

def identity_profile(member_checksums, names_by_checksum):
    """{identity: prevalence} for identities present in enough of the location's photos.
    `names_by_checksum` = {checksum: iterable of names} (named faces + whole-photo assignments)."""
    members = list(member_checksums)
    if not members:
        return {}
    counts = Counter()
    for cs in members:
        counts.update(set(names_by_checksum.get(cs, ())))
    n = len(members)
    return {name: c / n for name, c in counts.items() if c >= PEOPLE_MIN_COUNT or c / n >= PEOPLE_MIN_PREV}


def near_dup_factor(cos):
    """0..1: how much of the same-people damping is waived because the candidate is already
    a place-wise near-duplicate (raw fused cosine ramp NEAR_DUP_COS_LO -> HI)."""
    x = np.clip((np.asarray(cos, np.float32) - NEAR_DUP_COS_LO) / (NEAR_DUP_COS_HI - NEAR_DUP_COS_LO), 0, 1)
    return x * x * (3 - 2 * x)


def _names(db_manual, checksums):
    out = {}
    for cs, names in db_manual.get_identities_for_checksums(list(checksums)).items():
        out.setdefault(cs, set()).update(names)
    for cs, names in db_manual.get_photo_assignments_for_checksums(list(checksums)).items():
        out.setdefault(cs, set()).update(names)
    return out


def _file_rows(db, ids):
    rows = {}
    ids = [int(i) for i in ids]
    for s in range(0, len(ids), 500):
        for r in db.get_files_by_ids(ids[s:s + 500]):
            rows[int(r['id'])] = r
    return rows


def _valid_gps(lat, lon):
    return lat is not None and lon is not None and not (lat == 0 and lon == 0)


def geo_adjust(cand_lat, cand_lon, ref_lat, ref_lon, allow_veto):
    """(boost float32[k], veto float32[k], km float32[k] or nan) for candidates (NaN lat =
    no GPS: never boosted nor penalised) against reference points."""
    k = len(cand_lat)
    boost, veto, km = np.zeros(k, np.float32), np.zeros(k, np.float32), np.full(k, np.nan, np.float32)
    has = ~np.isnan(cand_lat)
    if not has.any() or len(ref_lat) == 0:
        return boost, veto, km
    d = haversine_km(cand_lat[has][:, None], cand_lon[has][:, None],
                     np.asarray(ref_lat)[None, :], np.asarray(ref_lon)[None, :]).min(axis=1)
    km[has] = d
    t = np.clip((GEO_RADIUS_KM - d) / max(GEO_RADIUS_KM - GEO_NEAR_KM, 1e-6), 0, 1)
    boost[has] = GEO_BOOST * t
    if allow_veto:
        veto[has] = np.where(d > GEO_VETO_KM, GEO_VETO_PENALTY, 0.0)
    return boost, veto, km


def apply_place_correction(db, manual, rank, display, fids, cks, member_ids, member_checksums,
                           location=None, pool=None):
    """Correct the base fused `rank` for people and add the geo/time/set co-signals.

    rank/display/fids/cks are the aligned outputs of the fused scoring over the whole library;
    member_ids/member_checksums describe the location's photos; `location` is its manual.db row
    (for an explicit GPS point). Returns (new_rank, why, info): new_rank is a copy of `rank`
    with the top `pool` entries corrected, why = {array index: breakdown dict} for those, info =
    {'person_scan_coverage': {...}, 'geo': {...}, 'people': [...]} for the response."""
    pool = PLACE_RERANK_POOL if pool is None else pool
    n = rank.shape[0]
    info = {}
    new_rank = np.array(rank, dtype=np.float32, copy=True)
    store = store_for(db).get(db)
    if n == 0:
        info['person_scan_coverage'] = _coverage(store, np.zeros(0, bool), 0)
        return new_rank, {}, info
    m = min(pool, n)
    idx = np.argpartition(-rank, m - 1)[:m] if m < n else np.arange(n)
    idx = idx[np.argsort(-rank[idx], kind='stable')]
    cand_ids = np.asarray(fids)[idx]
    area, scanned = store.lookup(cand_ids)
    info['person_scan_coverage'] = _coverage(store, scanned, len(idx))

    # (a) person area ---------------------------------------------------------------------
    pen_area = PERSON_AREA_PENALTY * person_penalty_curve(area) * scanned
    bonus = np.where(scanned & (area < PERSON_AREA_FREE), CLEAN_BONUS, 0.0).astype(np.float32)

    # (b) same people ---------------------------------------------------------------------
    cand_cs = [cks[i] for i in idx]
    prof = identity_profile(member_checksums, _names(manual, member_checksums)) if member_checksums else {}
    info['people'] = sorted(prof, key=lambda p: -prof[p])[:8]
    pen_people = np.zeros(len(idx), np.float32)
    shared = [()] * len(idx)
    if prof:
        cand_names = _names(manual, cand_cs)
        damp = 1.0 - near_dup_factor(np.asarray(display)[idx])
        for j, cs in enumerate(cand_cs):
            common = [nm for nm in cand_names.get(cs, ()) if nm in prof]
            if common:
                shared[j] = tuple(sorted(common))
                pen_people[j] = PEOPLE_LAMBDA * max(prof[nm] for nm in common) * float(damp[j])

    # (c) geo / time / set / directory ----------------------------------------------------
    rows = _file_rows(db, set(int(i) for i in cand_ids) | set(int(i) for i in member_ids))
    mem_pts = [(rows[i]['gps_lat'], rows[i]['gps_lon']) for i in member_ids if i in rows
               and _valid_gps(rows[i]['gps_lat'], rows[i]['gps_lon'])]
    loc_pt = None
    if location is not None and _valid_gps(location['gps_lat'], location['gps_lon']):
        loc_pt = (float(location['gps_lat']), float(location['gps_lon']))
    geo_boost = np.zeros(len(idx), np.float32)
    geo_veto = np.zeros(len(idx), np.float32)
    geo_km = np.full(len(idx), np.nan, np.float32)
    geo_info = {'active': False}
    if loc_pt is not None or len(mem_pts) >= GEO_MIN_MEMBERS:
        pts = np.array(mem_pts, np.float64).reshape(-1, 2)
        if len(pts) > GEO_MAX_REFS:
            pts = pts[np.linspace(0, len(pts) - 1, GEO_MAX_REFS).astype(int)]
        tight = True
        if len(pts):
            med = np.median(pts, axis=0)
            tight = bool(haversine_km(pts[:, 0], pts[:, 1], med[0], med[1]).max() <= GEO_TIGHT_KM)
        refs = pts.tolist() + ([list(loc_pt)] if loc_pt else [])
        ref = np.array(refs, np.float64)
        c_lat = np.array([rows[int(i)]['gps_lat'] if int(i) in rows and _valid_gps(rows[int(i)]['gps_lat'],
                          rows[int(i)]['gps_lon']) else np.nan for i in cand_ids], np.float64)
        c_lon = np.array([rows[int(i)]['gps_lon'] if int(i) in rows and not np.isnan(c_lat[j]) else np.nan
                          for j, i in enumerate(cand_ids)], np.float64)
        allow_veto = loc_pt is not None or tight
        geo_boost, geo_veto, geo_km = geo_adjust(c_lat, c_lon, ref[:, 0], ref[:, 1], allow_veto)
        geo_info = {'active': True, 'source': 'location' if loc_pt else 'members',
                    'refs': int(len(ref)), 'veto': bool(allow_veto)}
    info['geo'] = geo_info

    time_boost = np.zeros(len(idx), np.float32)
    mem_t = np.sort(np.array([rows[i]['taken_at'] for i in member_ids
                              if i in rows and rows[i]['taken_at']], np.float64))
    if len(mem_t):
        ct = np.array([rows[int(i)]['taken_at'] if int(i) in rows and rows[int(i)]['taken_at'] else np.nan
                       for i in cand_ids], np.float64)
        ok = ~np.isnan(ct)
        pos = np.searchsorted(mem_t, ct[ok])
        left = np.abs(ct[ok] - mem_t[np.clip(pos - 1, 0, len(mem_t) - 1)])
        right = np.abs(ct[ok] - mem_t[np.clip(pos, 0, len(mem_t) - 1)])
        time_boost[ok] = np.where(np.minimum(left, right) <= TIME_WINDOW_S, TIME_BOOST, 0.0)

    set_boost = np.zeros(len(idx), np.float32)
    dir_boost = np.zeros(len(idx), np.float32)
    if member_checksums:
        mem_sets = Counter(s['id'] for sl in manual.get_sets_for_checksums(list(member_checksums)).values()
                           for s in sl)
        if mem_sets:
            cand_sets = manual.get_sets_for_checksums(cand_cs)
            for j, cs in enumerate(cand_cs):
                best = max((mem_sets.get(s['id'], 0) for s in cand_sets.get(cs, ())), default=0)
                set_boost[j] = SET_BOOST * min(1.0, best / CO_FULL_MEMBERS)
        mem_dirs = Counter(_dirname(rows[i]['path']) for i in member_ids if i in rows)
        mem_dirs.pop('', None)
        if mem_dirs:
            for j, fid in enumerate(cand_ids):
                r = rows.get(int(fid))
                if r is not None:
                    dir_boost[j] = DIR_BOOST * min(1.0, mem_dirs.get(_dirname(r['path']), 0) / CO_FULL_MEMBERS)

    geo_net = geo_boost - geo_veto
    delta = bonus - pen_area - pen_people + geo_net + time_boost + set_boost + dir_boost
    new_rank[idx] = rank[idx] + delta.astype(np.float32)

    why = {}
    for j, i in enumerate(idx):
        w = {'place': round(float(rank[i]), 2), 'cos': round(float(display[i]), 3),
             'scanned': bool(scanned[j]),
             'person_area': round(float(area[j]), 2) if scanned[j] else None,
             'person_penalty': round(float(-pen_area[j] + bonus[j]), 2),
             'people_penalty': round(float(-pen_people[j]), 2),
             'geo': round(float(geo_net[j]), 2),
             'co': round(float(time_boost[j] + set_boost[j] + dir_boost[j]), 2),
             'total': round(float(delta[j]), 2)}
        if shared[j]:
            w['people'] = list(shared[j])
        if not math.isnan(float(geo_km[j])):
            w['geo_km'] = round(float(geo_km[j]), 1)
        why[int(i)] = w
    return new_rank, why, info


def _dirname(path):
    p = (path or '').replace('\\', '/')
    return p.rsplit('/', 1)[0] if '/' in p else ''


def _coverage(store, scanned_pool, pool_n):
    lib_scanned = int(store.scanned.sum())
    pool_scanned = int(np.count_nonzero(scanned_pool))
    return {'library_scanned': lib_scanned, 'library_total': int(store.total),
            'pool_scanned': pool_scanned, 'pool_total': int(pool_n),
            'unscanned_pct': round(100.0 * (1 - pool_scanned / pool_n), 1) if pool_n else 0.0}


def file_regions(db, file_ids):
    """{file_id: {'scanned': bool, 'regions': [[x1,y1,x2,y2] px], 'wh': (w, h)}} for the given
    files, from the same tables and rules as PersonStore (person-like + crowd boxes, faces
    expanded to body-ish boxes). `scanned` False = never scanned for persons (no detections row
    or faces row) or boxes whose frame size is unknown: callers must not treat it as person-free."""
    from media_manager import body_index  # hook-allow: lazy -- PIL/compute imports
    classes = tuple(body_index.PERSON_LIKE_CLASSES) + EXTRA_PERSON_CLASSES
    cq = ','.join('?' * len(classes))
    ids = [int(i) for i in file_ids]
    out = {}
    conn = db.conn
    for s in range(0, len(ids), 400):
        chunk = ids[s:s + 400]
        q = ','.join('?' * len(chunk))
        dims = {}
        for fid, w, h in conn.execute(f'SELECT id, width, height FROM files WHERE id IN ({q}) '
                                      'AND width > 0 AND height > 0', chunk):
            dims[fid] = (w, h)
        for fid, w, h in conn.execute(f'SELECT file_id, width, height FROM phashes WHERE file_id IN ({q}) '
                                      'AND width > 0 AND height > 0', chunk):
            dims.setdefault(fid, (w, h))
        persons, faces = defaultdict(list), defaultdict(list)
        scanned = set()
        for fid, x1, y1, x2, y2 in conn.execute(
                f'SELECT file_id, x1, y1, x2, y2 FROM detections WHERE file_id IN ({q}) '
                f'AND class_name IN ({cq}) AND confidence >= ? AND frame_index IS NULL AND x1 IS NOT NULL',
                (*chunk, *classes, PERSON_BOX_MIN_CONF)):
            persons[fid].append([x1, y1, x2, y2])
        for (fid,) in conn.execute(f'SELECT DISTINCT file_id FROM detections WHERE file_id IN ({q}) '
                                   'AND frame_index IS NULL', chunk):
            scanned.add(fid)
        for (fid,) in conn.execute(f'SELECT DISTINCT file_id FROM faces WHERE file_id IN ({q})', chunk):
            scanned.add(fid)
        for fid, bbox in conn.execute(
                f"SELECT file_id, bbox FROM faces WHERE file_id IN ({q}) AND frame_index IS NULL "
                "AND det_score >= ? AND (identity IS NULL OR identity != '__indexed__')",
                (*chunk, FACE_MIN_SCORE)):
            try:
                b = json.loads(bbox)
            except (TypeError, ValueError):
                continue
            if isinstance(b, list) and len(b) == 4:
                faces[fid].append([float(v) for v in b])
        for fid in chunk:
            wh = dims.get(fid)
            has_boxes = bool(persons.get(fid) or faces.get(fid))
            if fid not in scanned or (has_boxes and wh is None):
                out[fid] = {'scanned': False, 'regions': [], 'wh': wh or (0, 0)}
                continue
            if not has_boxes:
                out[fid] = {'scanned': True, 'regions': [], 'wh': wh or (0, 0)}
                continue
            regions, wh2 = person_regions(faces.get(fid, ()), persons.get(fid, ()), *wh)
            out[fid] = {'scanned': True, 'regions': regions, 'wh': wh2}
    return out


# ============================================================================ offline use

def fused_rank_numpy(db, member_ids, negative_ids=None, weights=None):
    """Numpy twin of web._fused_space_scores for offline use (eval script): (fids, cks, rank,
    display, coverage). `weights` {space: w}; default dino 0.6 + siglip 0.4."""
    from media_manager.similarity import adjusted_centroid, fuse_zscores, mean_normalized_centroid
    weights = weights or {'dino': 0.6, 'siglip': 0.4}
    parts, cov = [], {}
    for space, w in weights.items():
        ids, cs, mat = getattr(db, f'get_{space}_embeddings_matrix')()
        if mat.shape[0] == 0:
            cov[space] = 0
            continue

        def vecs(sel):
            idx = np.flatnonzero(np.isin(ids, np.asarray(list(sel), dtype=np.int64)))
            return [np.ascontiguousarray(mat[i], dtype=np.float32).tobytes() for i in idx]
        pos = vecs(member_ids)
        cov[space] = len(pos)
        neg = vecs(negative_ids) if negative_ids else []
        c = adjusted_centroid(pos, neg) if neg else mean_normalized_centroid(pos)
        if c is None:
            continue
        parts.append((ids, cs, mat.dot(np.asarray(c, np.float32)), w))
    f, c, r, d = fuse_zscores(parts)
    return f, c, r, d, cov
