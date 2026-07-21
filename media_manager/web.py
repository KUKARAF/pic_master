"""
FastAPI web server for media_manager — gallery UI with CLIP search, tags, and similar-image discovery.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from media_manager import frames
from media_manager.category_resolver import (
    get_category_counts,
    get_resolved_checksums_for_category,
    resolve_categories_for_checksums,
    resolve_category_for_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THUMB_SIZE = (400, 400)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

_HERE = Path(__file__).parent

# Request body models. These must live at module level, not inside create_app():
# web.py uses `from __future__ import annotations`, which turns every annotation
# into a string; FastAPI/Pydantic resolve those forward refs against the module's
# globals when building the OpenAPI schema, and classes defined inside create_app()
# aren't visible there — the route silently stops accepting the JSON body.
class TagBody(BaseModel):
    tag: str
    polarity: str = 'positive'

class SetBody(BaseModel):
    name: Optional[str] = None
    studio: Optional[str] = None
    set_id: Optional[int] = None

class IdentityBody(BaseModel):
    name: Optional[str] = None

class ManualFaceBody(BaseModel):
    bbox: List[float]

class SpatialTagBody(BaseModel):
    label: str
    bbox: List[float]

class FavoriteBody(BaseModel):
    favorite: bool

class TagLabelBody(BaseModel):
    label: str

class RenameIdentityBody(BaseModel):
    name: str

class UnassignIdentityBody(BaseModel):
    file_ids: List[int]

class AssignIdentitySetBody(BaseModel):
    set_id: int

class TitleBody(BaseModel):
    title: str

class CategoryBody(BaseModel):
    name: Optional[str] = None
    temperature: Optional[float] = None
    category_id: Optional[int] = None

class SearchChip(BaseModel):
    type: str
    value: str

class SearchPaletteBody(BaseModel):
    q: str = ''
    chips: List[SearchChip] = []

class SwipeExcludeBody(BaseModel):
    """Shared body for every swipe stack's buffer-refill endpoint. Sent as a
    POST body rather than a query-string `exclude` param because the client's
    exclude set is the full list of every card ever seen this session, which
    a long swiping session can grow past what a GET request line can hold
    (previously ~64KB / ~15000 refs) — a POST body has no such constraint at
    this scale, so it's just sent in full every time instead of being
    truncated. Truncating it (an earlier fix) caused a worse bug: the server
    would happily resurface already-seen top-ranked candidates that fell
    outside the truncated set, which the client silently re-discards as
    already-known, so the buffer would appear permanently stuck at 0 new
    cards even though plenty of fresh (or even the same, still-valid)
    candidates existed — exactly what a page reload (which resets the
    client's known-set to empty) would then reveal."""
    exclude: List[str] = []


def _gray_placeholder() -> bytes:
    """Return a gray 400×400 JPEG as bytes (used when thumbnail generation fails)."""
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (400, 400), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return buf.getvalue()


def _save_jpeg_atomic(img, dst_path: str, quality: int):
    """Save via a unique temp file + os.replace so dst_path only ever holds a
    complete JPEG. Concurrent requests for a not-yet-cached thumb/crop each
    generate it in parallel, and a plain in-place save lets one request stream
    a half-written file ('Response content shorter than Content-Length')."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f'{dst_path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        img.save(tmp_path, format='JPEG', quality=quality)
        os.replace(tmp_path, dst_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _make_thumbnail(src_path: str, dst_path: str) -> tuple:
    """Resize src to 400 px wide, save as JPEG at dst_path.
    Returns (success, message). message is a non-fatal decoder warning (e.g.
    corrupt-but-recoverable JPEG data) when success=True, or the exception
    text when success=False, or None when there's nothing to report."""
    from PIL import Image as PILImage
    import warnings
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with PILImage.open(src_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                if w == 0:
                    return False, 'image has zero width'
                new_w = 400
                new_h = max(1, int(h * new_w / w))
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                _save_jpeg_atomic(img, dst_path, quality=80)
        message = '; '.join(str(w.message) for w in caught) if caught else None
        return True, message
    except Exception as exc:
        return False, str(exc)


_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        from media_manager.face_detector import FaceDetector
        _face_detector = FaceDetector()
    return _face_detector


_object_detector = None


def _get_object_detector():
    global _object_detector
    if _object_detector is None:
        from media_manager.detector import YOLOWorldDetector
        _object_detector = YOLOWorldDetector()
    return _object_detector


_clip_indexer = None


def _get_clip_indexer():
    global _clip_indexer
    if _clip_indexer is None:
        from media_manager.indexer import CLIPIndexer
        _clip_indexer = CLIPIndexer()
    return _clip_indexer


_age_estimator = None


def _get_age_estimator():
    """Experimental — see age_estimator.py. Runs in a separate, isolated venv (MiVOLO
    pins an old timm/ultralytics that conflict with this app's own detector/indexer),
    so this lazy singleton is a thin subprocess client, not a loaded model."""
    global _age_estimator
    if _age_estimator is None:
        from media_manager.age_estimator import AgeGenderEstimator
        _age_estimator = AgeGenderEstimator()
    return _age_estimator


def _make_body_crop(src_path: str, bbox_json: str, dst_path: str, height: int = 260) -> bool:
    """Crop a person from src_path using bbox JSON, save JPEG to dst_path. Unlike
    _make_face_crop this preserves aspect ratio — body boxes are tall, and squashing
    them square makes outfits (the thing find-by-body compares) unrecognizable."""
    import json
    from PIL import Image as PILImage
    try:
        bbox = json.loads(bbox_json)
        if not bbox or len(bbox) < 4:
            return False
        x1, y1, x2, y2 = [int(v) for v in bbox]
        with PILImage.open(src_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return False
            body = img.crop((x1, y1, x2, y2))
            new_w = max(1, int(body.width * height / body.height))
            body = body.resize((new_w, height), PILImage.LANCZOS)
            _save_jpeg_atomic(body, dst_path, quality=85)
            return True
    except Exception:
        return False


def _make_face_crop(src_path: str, bbox_json: str, dst_path: str, size: int = 200) -> bool:
    """Crop a face from src_path using bbox JSON, save JPEG to dst_path. Returns True on success."""
    import json
    from PIL import Image as PILImage
    try:
        bbox = json.loads(bbox_json)
        if not bbox or len(bbox) < 4:
            return False
        x1, y1, x2, y2 = [int(v) for v in bbox]
        with PILImage.open(src_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return False
            face = img.crop((x1, y1, x2, y2))
            face = face.resize((size, size), PILImage.LANCZOS)
            _save_jpeg_atomic(face, dst_path, quality=85)
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(data_root: str) -> FastAPI:
    """Create and return the FastAPI application rooted at data_root."""
    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')

    if not os.path.isdir(media_dir):
        raise RuntimeError(
            f"No media repository found at '{data_root}'. "
            "Run 'media init' inside that directory first."
        )

    db_path = os.path.join(media_dir, 'media.db')
    thumbs_dir = os.path.join(media_dir, 'thumbs')
    os.makedirs(thumbs_dir, exist_ok=True)

    # Import here so the module can be imported without heavy deps loaded
    from media_manager.database import Database
    from media_manager.error_log import ErrorLog
    from media_manager.manual_db import ManualDB

    db = Database(db_path)
    errors = ErrorLog(os.path.join(media_dir, 'error.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    app = FastAPI(title='media gallery')

    # Static files and templates
    static_dir = _HERE / 'static'
    templates_dir = _HERE / 'templates'

    app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')
    templates = Jinja2Templates(directory=str(templates_dir))

    # ------------------------------------------------------------------
    # Internal helpers bound to this app instance
    # ------------------------------------------------------------------

    def _file_or_404(file_id: int):
        row = db.get_file_by_id(file_id)
        if row is None:
            raise HTTPException(status_code=404, detail='File not found')
        return row

    def _live_abs_path(file_id: int, primary_rel_path: str):
        """Resolve a content record to a path that actually exists on disk. Tries the
        primary (most-recently-seen) path first, then falls back to any other known
        path for the same content — e.g. the primary path was one of several
        duplicate copies and that particular copy got deleted, but the content is
        still available at another tracked location. Returns None if nothing's live."""
        primary_abs = os.path.join(data_root, primary_rel_path)
        if os.path.isfile(primary_abs):
            return primary_abs
        for other_path, _ in db.get_paths_for_file(file_id):
            if other_path == primary_rel_path:
                continue
            other_abs = os.path.join(data_root, other_path)
            if os.path.isfile(other_abs):
                return other_abs
        return None

    def _combined_faces_for_file(file_id, checksum):
        """Merge manual.db's rows for this file (hand-drawn + promoted/named, keyed by
        checksum) with media.db's auto-detected rows that haven't been named yet
        (keyed by file_id, since those live entirely inside this one media.db)."""
        import json as _json
        manual_rows = manual.get_faces_for_file(checksum)
        promoted = {r['source_face_id'] for r in manual_rows if r['source_face_id']}
        out = [{'ref': f"manual:{r['id']}", 'bbox': [r['x1'], r['y1'], r['x2'], r['y2']],
                'identity': r['identity'], 'frame_index': r['frame_index'],
                'favorite': bool(r['favorite'])} for r in manual_rows]
        for r in db.get_faces_for_file(file_id):
            if r['id'] in promoted:
                continue
            out.append({'ref': f"auto:{r['id']}", 'bbox': _json.loads(r['bbox']),
                        'identity': None, 'frame_index': r['frame_index'], 'favorite': False})
        return out

    def _parse_face_ref(face_id: str):
        if ':' not in face_id:
            raise HTTPException(status_code=400, detail='face id must be "auto:<id>" or "manual:<id>"')
        kind, _, raw = face_id.partition(':')
        if kind not in ('auto', 'manual'):
            raise HTTPException(status_code=400, detail='unknown face id prefix')
        try:
            return kind, int(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail='invalid face id')

    def _row_to_dict(row, tags=None):
        return {
            'id': row['id'],
            'path': row['path'],
            'size': row['size'],
            'modified_time': row['modified_time'],
            'checksum': row['checksum'],
            'tags': tags if tags is not None else [],
            'favorite': manual.is_file_favorite(row['checksum']),
            'title': manual.get_file_title(row['checksum']),
        }

    def _enrich_rows(rows, scores=None):
        """Add tags + favorite state + title + sets + recognized people to a list of
        (id, path, has_embedding, checksum) rows — the single shared card-building
        path for every photo-grid view (gallery, search results, set detail), so
        those views don't each duplicate the same batch of lookups.

        `scores` is an optional {file_id: score} map for search results ranked by
        similarity/relevance; when given, each card gets a 'score' key."""
        checksums = [row[3] for row in rows]
        tag_map = manual.list_tags_for_checksums(checksums)
        favorite_checksums = manual.get_favorite_checksums(checksums)
        title_map = manual.get_titles_for_checksums(checksums)
        sets_map = manual.get_sets_for_checksums(checksums)
        identities_map = manual.get_identities_for_checksums(checksums)
        category_map = resolve_categories_for_checksums(manual, db, [(row[0], row[3]) for row in rows])
        result = []
        for row in rows:
            file_id = row[0]
            path = row[1]
            has_embedding = row[2] if len(row) > 2 else 0
            checksum = row[3]
            card = {
                'id': file_id,
                'path': path,
                'checksum': checksum,
                'filename': os.path.basename(path),
                'has_embedding': bool(has_embedding),
                'tags': tag_map.get(checksum, []),
                'favorite': checksum in favorite_checksums,
                'title': title_map.get(checksum),
                'sets': sets_map.get(checksum, []),
                'people': identities_map.get(checksum, []),
                'category': category_map.get(checksum),
            }
            if scores is not None and file_id in scores:
                card['score'] = scores[file_id]
            result.append(card)
        return result

    def _category_checksums_by_name():
        """{category_name: set(checksums)} for every category, resolved once —
        manual assignments plus unsuppressed auto-matches, mirroring
        get_resolved_checksums_for_category's own manual-wins-over-auto logic but
        computed in a single pass over the whole auto-match table instead of one
        full-table scan per category. get_resolved_checksums_for_category does its
        own fresh db.get_all_file_category_matches() scan every call — fine for a
        single lookup, but the search palette calls this once per category
        *candidate* while ranking suggestions, so re-scanning per category turned
        one keystroke into O(categories) full-library scans."""
        by_name = {}
        for row in manual.list_categories():
            by_name[row['name']] = set(manual.get_example_checksums_for_category(row['id'], limit=1000))
        all_manual_checksums = set().union(*by_name.values()) if by_name else set()
        overridden = manual.get_all_category_override_checksums()
        for _file_id, checksum, name, _score in db.get_all_file_category_matches():
            if checksum in all_manual_checksums or checksum in overridden:
                continue
            by_name.setdefault(name, set()).add(checksum)
        return by_name

    def _tag_checksums_by_label():
        """{label: set(checksums)} for every positive tag, one query total. Same
        reasoning as _category_checksums_by_name: get_files_by_tag is a cheap,
        indexed single-tag query, but calling it once per tag *candidate* while
        ranking suggestions still means one query per tag on every keystroke —
        for a library with hundreds of tags that's hundreds of round trips just
        to render a dropdown."""
        cur = manual.conn.cursor()
        cur.execute("SELECT checksum, label FROM tags WHERE polarity = 'positive'")
        by_label = {}
        for checksum, label in cur.fetchall():
            by_label.setdefault(label, set()).add(checksum)
        return by_label

    def _set_checksums_by_id():
        """{set_id: set(checksums)} for every set, one query total — same
        reasoning as _tag_checksums_by_label, for get_files_by_set."""
        cur = manual.conn.cursor()
        cur.execute('SELECT checksum, set_id FROM file_sets')
        by_id = {}
        for checksum, set_id in cur.fetchall():
            by_id.setdefault(set_id, set()).add(checksum)
        return by_id

    def _identity_checksums_by_name(set_map):
        """{identity: set(checksums)} — own (named, non-rejected) faces, plus
        whole-photo assignments, plus sets linked to that identity (expanded via
        the already-computed set_map) — a handful of queries total instead of
        ~2-4 queries per identity *candidate*. get_files_by_face_identity in
        particular does a LOWER(identity) LIKE '%...%' scan with no usable index;
        calling that once per identity on every keystroke was the single biggest
        cost in ranking suggestions once the category scan was fixed."""
        cur = manual.conn.cursor()
        by_name = {}
        cur.execute('SELECT DISTINCT checksum, identity FROM faces WHERE identity IS NOT NULL AND rejected = 0')
        for checksum, identity in cur.fetchall():
            by_name.setdefault(identity, set()).add(checksum)
        cur.execute('SELECT checksum, identity FROM identity_photo_assignments')
        for checksum, identity in cur.fetchall():
            by_name.setdefault(identity, set()).add(checksum)
        cur.execute('SELECT set_id, identity FROM identity_set_assignments')
        for set_id, identity in cur.fetchall():
            by_name.setdefault(identity, set()).update(set_map.get(set_id, ()))
        return by_name

    def _checksums_for_chip(chip, category_map=None, tag_map=None, set_map=None, identity_map=None):
        """Resolve one search-palette/multi-filter chip ({'type', 'value'}) to the
        set of checksums it matches. 'face' replicates the union search_page's
        person branch already does inline (own faces + whole-photo assignments +
        photos in sets linked to this identity) so the palette and /search share one
        definition instead of drifting apart. The `*_map` args, when given (the
        search palette precomputes all four once per request), avoid re-resolving
        from scratch per call — see _category_checksums_by_name and friends above;
        omitted (the /search f= path's case — a handful of chips, not a per-
        keystroke ranking loop) falls back to the plain per-chip queries."""
        t, v = chip['type'], chip['value']
        if t == 'category':
            if category_map is not None:
                return set(category_map.get(v, ()))
            cat = manual.find_category(v)
            if cat is None:
                return set()
            return set(get_resolved_checksums_for_category(manual, db, cat['id'], cat['name'], limit=500))
        if t == 'tag':
            if tag_map is not None:
                return set(tag_map.get(v, ()))
            return set(manual.get_files_by_tag(v, limit=1000))
        if t == 'face':
            if identity_map is not None:
                return set(identity_map.get(v, ()))
            checksums = set(manual.get_files_by_face_identity(v, limit=1000))
            checksums |= set(manual.get_photos_assigned_to_identity(v, limit=1000))
            for s in manual.get_sets_linked_to_identity(v):
                checksums |= set(manual.get_files_by_set(s['id'], limit=1000))
            return checksums
        if t == 'set':
            if set_map is not None:
                return set(set_map.get(int(v), ()))
            return set(manual.get_files_by_set(int(v), limit=1000))
        if t == 'file':
            return {row[2] for row in db.search_by_path_substring(v, limit=200)}
        return set()

    def _intersect_chips(chips, category_map=None, tag_map=None, set_map=None, identity_map=None):
        """None means 'no restriction' (caller decides what that means); an empty
        chip list always means None, never an empty set, so a chip-less query isn't
        mistaken for 'matches nothing'."""
        if not chips:
            return None
        sets = [
            _checksums_for_chip(c, category_map=category_map, tag_map=tag_map,
                                 set_map=set_map, identity_map=identity_map)
            for c in chips
        ]
        return set.intersection(*sets) if sets else set()

    def _attach_file_meta(cards, file_id_field='file_id'):
        """Merge filename/tags/sets/category/people onto swipe-suggestion cards that
        weren't already built via _enrich_rows (faces/categories/tags suggestions —
        sets suggestions already go through _enrich_rows and don't need this).
        Reuses the exact same batched lookups as _enrich_rows so this doesn't
        become a second, divergent way of assembling the same information."""
        file_rows = {r['id']: r for r in db.get_files_by_ids([c[file_id_field] for c in cards])}
        checksums = [r['checksum'] for r in file_rows.values()]
        tag_map = manual.list_tags_for_checksums(checksums)
        sets_map = manual.get_sets_for_checksums(checksums)
        identities_map = manual.get_identities_for_checksums(checksums)
        category_map = resolve_categories_for_checksums(
            manual, db, [(r['id'], r['checksum']) for r in file_rows.values()]
        )
        for card in cards:
            row = file_rows.get(card[file_id_field])
            if row is None:
                continue
            checksum = row['checksum']
            card['filename'] = os.path.basename(row['path'])
            card['tags'] = tag_map.get(checksum, [])
            card['sets'] = sets_map.get(checksum, [])
            card['people'] = identities_map.get(checksum, [])
            card['category'] = category_map.get(checksum)
        return cards

    def _chunked(items, size=500):
        """SQLite has a hard cap on bound parameters per query (varies by build,
        often as low as 999) — any `WHERE x IN (...)` lookup built from an
        *unbounded* candidate pool (e.g. every file clearing a similarity
        threshold across a whole library, not just a fixed-size page) needs to
        be split into chunks like this rather than bound in one query, or it
        raises 'too many SQL variables' once the library is large enough."""
        items = list(items)
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def _deprioritize_files_with_named_face(candidates, file_id_index):
        """Stable-sort (score order preserved within each group) so candidates
        whose photo already has at least one identified person come after ones
        from photos with no identified face yet — a UI toggle (on by default)
        for the face-suggestion stream, matching _find_similar_files_for_set's
        avoid_existing for sets: surface still-unidentified photos first
        without hard-excluding ones that already have someone named."""
        if not candidates:
            return candidates
        file_ids = list({c[file_id_index] for c in candidates})
        file_rows = {}
        for chunk in _chunked(file_ids):
            for r in db.get_files_by_ids(chunk):
                file_rows[r['id']] = r
        named_checksums = manual.get_all_checksums_with_named_face()

        def has_named_face(c):
            row = file_rows.get(c[file_id_index])
            return bool(row is not None and row['checksum'] in named_checksums)

        candidates.sort(key=has_named_face)
        return candidates

    def _sort_cards_by_age(cards, order):
        """Sort already-enriched cards by their average estimated age (see
        manual.get_average_ages_for_checksums). Cards with no age estimate at all
        keep their existing relative order and always sort to the end, regardless of
        direction — there's no meaningful age to rank them by."""
        age_map = manual.get_average_ages_for_checksums([c['checksum'] for c in cards])
        with_age = [c for c in cards if c['checksum'] in age_map]
        without_age = [c for c in cards if c['checksum'] not in age_map]
        with_age.sort(key=lambda c: age_map[c['checksum']], reverse=(order != 'asc'))
        return with_age + without_age

    def _sort_file_rows(file_rows, sort, order):
        """Sort raw files_with_path rows (has first_seen/modified_time/checksum) by
        the 'added'/'modified' timestamp columns — used by the checksum-driven views
        (search by tag/person, set detail) which don't paginate via SQL. 'age' isn't
        handled here since it needs manual.db and enriched cards; see _sort_cards_by_age."""
        reverse = (order != 'asc')
        if sort == 'modified':
            return sorted(file_rows, key=lambda r: r['modified_time'] or 0, reverse=reverse)
        return sorted(file_rows, key=lambda r: r['first_seen'], reverse=reverse)

    # ------------------------------------------------------------------
    # Thumbnail / image serving
    # ------------------------------------------------------------------

    @app.get('/thumb/{file_id}')
    def serve_thumb(file_id: int):
        row = db.get_file_by_id(file_id)
        if row is None:
            return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)

        rel_path = row['path']
        ext = os.path.splitext(rel_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return Response(content=_gray_placeholder(), media_type='image/jpeg')

        thumb_path = os.path.join(thumbs_dir, f'{file_id}.jpg')

        if not os.path.isfile(thumb_path):
            abs_path = _live_abs_path(file_id, rel_path)
            if abs_path is None:
                return Response(content=_gray_placeholder(), media_type='image/jpeg')
            success, message = _make_thumbnail(abs_path, thumb_path)
            if message:
                errors.log(rel_path, message)
            if not success:
                return Response(content=_gray_placeholder(), media_type='image/jpeg')

        return FileResponse(thumb_path, media_type='image/jpeg')

    @app.get('/image/{file_id}')
    def serve_image(file_id: int):
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        return FileResponse(abs_path)

    @app.get('/api/files/{file_id}/neighbors')
    def api_file_neighbors(file_id: int):
        _file_or_404(file_id)
        prev_id, next_id = db.get_neighbor_ids(file_id)
        return {'prev': prev_id, 'next': next_id}

    # ------------------------------------------------------------------
    # HTML pages
    # ------------------------------------------------------------------

    @app.get('/', response_class=HTMLResponse)
    def gallery_page(request: Request, page: int = 1, favorite: bool = False, sort: str = 'added', order: str = 'desc'):
        limit = 60
        all_tags = manual.list_all_tags()
        if favorite or sort == 'age':
            # Favorites are typically a small subset, and "age" needs manual.db data
            # SQL here can't join against — both filter/sort in Python rather than
            # adding those paths to the paginated SQL query.
            sql_sort = sort if sort != 'age' else 'added'
            all_rows = db.list_files_with_embedding_flag(limit=1_000_000, offset=0, sort=sql_sort, order=order)
            all_files = _enrich_rows(all_rows)
            if favorite:
                all_files = [f for f in all_files if f['favorite']]
            if sort == 'age':
                all_files = _sort_cards_by_age(all_files, order)
            total = len(all_files)
            offset = (page - 1) * limit
            files = all_files[offset:offset + limit]
        else:
            offset = (page - 1) * limit
            rows = db.list_files_with_embedding_flag(limit=limit, offset=offset, sort=sort, order=order)
            files = _enrich_rows(rows)
            total = db.count_files()
        return templates.TemplateResponse(request, 'gallery.html', {
            'files': files,
            'page': page,
            'total': total,
            'limit': limit,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'favorite_only': favorite,
            'sort': sort,
            'order': order,
        })

    @app.get('/photo/{file_id}', response_class=HTMLResponse)
    def photo_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        checksum = row['checksum']
        tag_rows = manual.get_tags(checksum)
        whole_tags = [
            {'id': t['id'], 'label': t['label'], 'polarity': t['polarity'], 'favorite': bool(t['favorite'])}
            for t in tag_rows if t['x1'] is None
        ]
        spatial_tags = [{'id': t['id'], 'label': t['label']} for t in tag_rows if t['x1'] is not None]
        all_tags = manual.list_all_tags()
        negated = manual.get_negated_labels(checksum)
        detected_classes = [c for c in db.get_detected_classes(file_id) if c not in negated]
        faces = _combined_faces_for_file(file_id, checksum)
        file_info = dict(row)
        file_info['filename'] = os.path.basename(file_info['path'])
        file_info['tags'] = whole_tags
        ext = os.path.splitext(file_info['path'])[1].lower()
        file_info['is_image'] = ext in IMAGE_EXTENSIONS
        frame_count = 1
        if file_info['is_image']:
            abs_path = _live_abs_path(file_id, file_info['path'])
            if abs_path is not None:
                try:
                    frame_count = frames.get_frame_count(abs_path)
                except Exception:
                    frame_count = 1
        file_info['frame_count'] = frame_count
        file_info['favorite'] = manual.is_file_favorite(checksum)
        file_info['title'] = manual.get_file_title(checksum)
        current_sets = [
            {'id': s['id'], 'name': s['name'], 'studio': s['studio'], 'favorite': bool(s['favorite'])}
            for s in manual.get_sets_for_file(checksum)
        ]
        file_info['category'] = resolve_category_for_file(manual, db, file_id, checksum)
        all_categories = _all_categories_for_nav()
        # Age/gender estimates (experimental — see age_estimator.py) are shown inline
        # next to each face's name, not as a separate section, so merge them directly
        # into the same `faces` list the template already iterates over.
        age_by_ref = {r['face_ref']: r for r in manual.get_age_estimates_for_checksum(checksum)}
        for face in faces:
            est = age_by_ref.get(face['ref'])
            face['age'] = est['age'] if est else None
            face['gender'] = est['gender'] if est else None
        return templates.TemplateResponse(request, 'photo.html', {
            'file': file_info,
            'detected_classes': detected_classes,
            'faces': faces,
            'spatial_tags': spatial_tags,
            'all_tags': all_tags,
            'all_categories': all_categories,
            'current_sets': current_sets,
        })

    def _unsorted_candidate_photos_for_identity(exclude_checksums, limit=60, overfetch=300):
        """Plain, unranked browse list for the person page's manual "assign to an
        individual photo" picker: recent files that belong to no set at all and
        aren't already linked to this identity by any means (a real face or a
        prior whole-photo assignment). Deliberately not ML-ranked — this path is
        for photos face detection/suggestion missed entirely, so there's nothing
        principled to rank by; `overfetch` just needs to be generous enough that
        filtering out excluded checksums still leaves `limit` candidates in a
        typical personal library."""
        rows = db.list_files_with_embedding_flag(limit=overfetch, sort='added', order='desc')
        candidates = [row for row in rows if row[3] not in exclude_checksums][:limit]
        return _enrich_rows(candidates)

    @app.get('/search', response_class=HTMLResponse)
    def search_page(request: Request, q: str = '', tag: str = '', face_id: str = '', person: str = '',
                     face_ref: str = '', category: str = '', sort: str = 'added', order: str = 'desc',
                     f: List[str] = Query([])):
        files = []
        message = ''
        all_tags = manual.list_all_tags()
        all_categories = _all_categories_for_nav()
        similar_unknown_faces = []
        similar_faces = []
        available_sets = []
        unsorted_candidate_photos = []
        queue_label = 'Search'

        if f:
            # Multi-facet AND search from the search palette — each entry is
            # "type:value" (built by the palette's chip list); resolved via the same
            # per-chip checksum resolver the palette itself uses for live counts, so
            # this and /api/search-palette never drift apart.
            chips = []
            for part in f:
                chip_type, _, chip_value = part.partition(':')
                if chip_type and chip_value:
                    chips.append({'type': chip_type, 'value': chip_value})
            checksums = _intersect_chips(chips) or set()
            if not checksums:
                message = 'No files match this combination of filters.'
            else:
                file_rows = _sort_file_rows(db.get_files_by_checksums(list(checksums)), sort, order)
                rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
                files = _enrich_rows(rows)
                if sort == 'age':
                    files = _sort_cards_by_age(files, order)
            queue_label = 'Search: ' + ' + '.join(f'{c["type"]}:{c["value"]}' for c in chips)

        elif face_ref:
            # "Find similar faces" from a specific face chip — reuses the same
            # grid+expand-slider UI as searching by person name, but seeded from one
            # face's embedding instead of a name, so it works on faces with no name yet.
            similar_faces = _find_similar_faces_to_ref(face_ref, SUGGEST_THRESHOLD)
            return templates.TemplateResponse(request, 'search.html', {
                'files': files,
                'q': q,
                'tag': tag,
                'face_id': face_id,
                'person': person,
                'face_ref': face_ref,
                'category': category,
                'query_face_file_id': _get_face_file_id(face_ref),
                'similar_faces': similar_faces,
                'message': message,
                'all_tags': all_tags,
                'all_categories': all_categories,
                'similar_unknown_faces': similar_unknown_faces,
            })

        name_query = person or face_id
        linked_set_ids = set()
        if f:
            pass
        elif name_query:
            checksums = set(manual.get_files_by_face_identity(name_query, limit=200))
            if person:
                # Union in whole-photo manual assignments (no face crop involved)
                # alongside the regular per-face identification — either one
                # makes a file "this person's", and both need to show up here.
                checksums |= set(manual.get_photos_assigned_to_identity(person, limit=200))
                # Sets linked to this identity (manual.link_identity_to_set) aren't
                # stamped onto individual photos — expand their *current* membership
                # here instead, so a photo added to a linked set later shows up too
                # without any extra row ever being written.
                linked_sets = manual.get_sets_linked_to_identity(person)
                linked_set_ids = {s['id'] for s in linked_sets}
                for s in linked_sets:
                    checksums |= set(manual.get_files_by_set(s['id'], limit=1000))
            if not checksums:
                message = f'No files found with person "{name_query}" — run <code>media faces</code> to detect faces first.'
            else:
                file_rows = _sort_file_rows(db.get_files_by_checksums(list(checksums)), sort, order)
                rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
                files = _enrich_rows(rows)
                if sort == 'age':
                    files = _sort_cards_by_age(files, order)

            if person:
                # Manual-only tools (no ML suggestion involved): link this identity
                # to a whole set, or assign it to one unsorted photo at a time — for
                # photos face detection missed the person in entirely.
                available_sets = manual.list_sets()
                already_linked = checksums | manual.get_all_set_member_checksums()
                unsorted_candidate_photos = _unsorted_candidate_photos_for_identity(already_linked)
            queue_label = f'Person: {name_query}'

        elif tag:
            checksums = manual.get_files_by_tag(tag, limit=200)
            file_rows = _sort_file_rows(db.get_files_by_checksums(checksums), sort, order)
            rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
            files = _enrich_rows(rows)
            if sort == 'age':
                files = _sort_cards_by_age(files, order)
            queue_label = f'Tag: {tag}'

        elif category:
            queue_label = f'Category: {category}'
            cat_row = manual.find_category(category)
            if cat_row is None:
                message = f'No category named "{category}".'
            else:
                checksums = get_resolved_checksums_for_category(manual, db, cat_row['id'], cat_row['name'], limit=500)
                if not checksums:
                    message = f'No files resolved to category "{category}" yet.'
                else:
                    file_rows = _sort_file_rows(db.get_files_by_checksums(checksums), sort, order)
                    rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
                    files = _enrich_rows(rows)
                    if sort == 'age':
                        files = _sort_cards_by_age(files, order)

        elif q:
            queue_label = f'Search: {q}'
            # YOLO-World object detection search, plus a filename/path substring
            # search — both kinds of matches are merged into one results grid.
            try:
                from media_manager.media_manager import _parse_query
                tokens = _parse_query(q)

                class_rows = db.search_by_classes(tokens, limit=50) if tokens else []
                path_rows = db.search_by_path_substring(q, limit=50)

                if not tokens and not path_rows:
                    message = 'No searchable terms found in query.'
                elif not class_rows and not path_rows:
                    message = 'No matches found — run <code>media index</code> to detect objects first.'
                else:
                    scores = {file_id: round(score, 4) for file_id, _path, score, _checksum in class_rows}
                    seen_ids = set()
                    rows = []
                    for file_id, path, _score, checksum in class_rows:
                        seen_ids.add(file_id)
                        rows.append((file_id, path, True, checksum))
                    for file_id, path, checksum in path_rows:
                        if file_id in seen_ids:
                            continue
                        seen_ids.add(file_id)
                        rows.append((file_id, path, False, checksum))
                    files = _enrich_rows(rows, scores=scores)
            except Exception as exc:
                message = f'Search error: {exc}'

        return templates.TemplateResponse(request, 'search.html', {
            'files': files,
            'q': q,
            'tag': tag,
            'face_id': face_id,
            'person': person,
            'face_ref': '',
            'category': category,
            'query_face_file_id': None,
            'message': message,
            'all_tags': all_tags,
            'all_categories': all_categories,
            'similar_unknown_faces': similar_unknown_faces,
            'available_sets': available_sets,
            'linked_set_ids': linked_set_ids,
            'unsorted_candidate_photos': unsorted_candidate_photos,
            'sort': sort,
            'order': order,
            'queue_label': queue_label,
        })

    @app.get('/similar/{file_id}', response_class=HTMLResponse)
    def similar_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        filename = os.path.basename(row['path'])
        files = []
        message = ''
        all_tags = manual.list_all_tags()

        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            message = (
                f'"{filename}" has no embedding yet — '
                'run <code>media embed</code> to build CLIP embeddings.'
            )
        else:
            try:
                import numpy as np
                query_emb = np.frombuffer(emb_bytes, dtype=np.float32)

                all_embs = db.get_all_embeddings()
                matrix = np.stack([
                    np.frombuffer(r[2], dtype=np.float32) for r in all_embs
                ])
                scores = matrix.dot(query_emb).tolist()
                ranked = sorted(
                    zip([r[0] for r in all_embs], [r[1] for r in all_embs], scores, [r[3] for r in all_embs]),
                    key=lambda x: x[2],
                    reverse=True,
                )
                # skip self
                ranked = [(fid, path, score, cs) for fid, path, score, cs in ranked if fid != file_id][:20]

                rows = [(fid, path, True, cs) for fid, path, _score, cs in ranked]
                scores = {fid: round(score, 4) for fid, _path, score, _cs in ranked}
                files = _enrich_rows(rows, scores=scores)
            except Exception as exc:
                message = f'Similarity search error: {exc}'

        return templates.TemplateResponse(request, 'similar.html', {
            'source_file': dict(row),
            'source_filename': filename,
            'source_id': file_id,
            'files': files,
            'message': message,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
        })

    # ------------------------------------------------------------------
    # Find-by-body: person re-ID by outfit/build (see body_index.py).
    # Web-only by design — the body index is built and queried from here,
    # never by a CLI command.
    # ------------------------------------------------------------------

    body_index_job = {'running': False, 'done': 0, 'total': 0, 'error': None}

    def _ensure_own_bodies(file_id: int, row) -> list:
        """Return this photo's non-sentinel body rows, embedding them on demand the
        first time find-by-body is opened for a not-yet-indexed photo. A photo with
        no stored person detections gets a lazy one-image YOLO-World pass (vocab
        pinned to 'person'; results are NOT written to the detections table — that
        would destructively replace the photo's real tag detections)."""
        from media_manager import body_index

        own = db.get_body_embeddings_for_file(file_id)
        if own:
            return own
        cursor = db.conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) FROM body_embeddings WHERE file_id = ? AND frame_index IS NULL',
            (file_id,)
        )
        if cursor.fetchone()[0] > 0:
            return []  # already processed: sentinel row says "no people here"

        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        try:
            body_index.embed_bodies_for_file(
                db, _get_clip_indexer(), file_id, abs_path, detector=_get_object_detector())
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f'Person detection failed: {exc}')
        return db.get_body_embeddings_for_file(file_id)

    def _body_search(query_row, exclude_file_id: int, limit: int = 40, threshold: float = 0.5):
        """Cosine-rank all body crops against one query crop's embedding. Dedupes to
        the best crop per file (like search_by_face_image) and enriches each result
        with tags/sets/recognized people, same as every other card grid."""
        import numpy as np

        query_emb = np.frombuffer(query_row[2], dtype=np.float32)
        rows = db.get_all_body_embeddings()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        scores = matrix.dot(query_emb)

        best_per_file = {}
        for i in np.argsort(-scores):
            score = float(scores[int(i)])
            if score < threshold or len(best_per_file) >= limit:
                break
            r = rows[int(i)]
            if r[1] == exclude_file_id or r[1] in best_per_file:
                continue
            best_per_file[r[1]] = (r, score)

        body_ids = {}
        rows = []
        scores = {}
        for r, score in best_per_file.values():
            body_id, fid, path, _bbox_json, _emb = r
            file_row = db.get_file_by_id(fid)
            if file_row is None:
                continue
            body_ids[fid] = body_id
            rows.append((fid, path, False, file_row['checksum']))
            scores[fid] = round(score, 4)

        results = _enrich_rows(rows, scores=scores)
        for card in results:
            card['body_id'] = body_ids[card['id']]
        results.sort(key=lambda item: item['score'], reverse=True)
        return results

    @app.get('/body-similar/{file_id}', response_class=HTMLResponse)
    def body_similar_page(request: Request, file_id: int, body_id: Optional[int] = None):
        row = _file_or_404(file_id)
        filename = os.path.basename(row['path'])
        message = ''
        results = []
        selected_id = None

        try:
            own = _ensure_own_bodies(file_id, row)
        except HTTPException as exc:
            own = []
            message = str(exc.detail)

        crops = [{'id': r[0]} for r in own]
        if not own and not message:
            message = 'No people detected in this photo.'
        elif own:
            chosen = None
            if body_id is not None:
                chosen = next((r for r in own if r[0] == body_id), None)
                if chosen is None:
                    message = 'Unknown person crop for this photo.'
            elif len(own) == 1:
                chosen = own[0]
            if chosen is not None:
                selected_id = chosen[0]
                results = _body_search(chosen, file_id)

        return templates.TemplateResponse(request, 'body_similar.html', {
            'source_id': file_id,
            'source_filename': filename,
            'crops': crops,
            'selected_id': selected_id,
            'results': results,
            'message': message,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/body-crop/{body_id}')
    def serve_body_crop(body_id: int):
        rec = db.get_body_embedding(body_id)
        if rec is None:
            return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)
        b_file_id, bbox_json, _emb = rec
        file_row = db.get_file_by_id(b_file_id)
        if file_row is None:
            return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)
        crop_path = os.path.join(thumbs_dir, f'body_{body_id}.jpg')
        if not os.path.isfile(crop_path):
            src_path = _live_abs_path(b_file_id, file_row['path'])
            if src_path is None or not _make_body_crop(src_path, bbox_json, crop_path):
                return Response(content=_gray_placeholder(), media_type='image/jpeg')
        return FileResponse(crop_path, media_type='image/jpeg')

    @app.post('/api/body-index/start')
    def api_body_index_start():
        import threading
        from media_manager import body_index

        if body_index_job['running']:
            return {'started': False, 'message': 'Body index build already running.'}
        body_index_job.update(
            running=True, done=0, total=len(db.get_unbody_indexed_files()), error=None)

        def _run():
            try:
                def _progress(done, total):
                    body_index_job['done'] = done
                    body_index_job['total'] = total
                body_index.build_body_index(
                    db, errors, _get_clip_indexer(), _get_object_detector(), data_root,
                    on_progress=_progress)
            except Exception as exc:
                body_index_job['error'] = str(exc)
            finally:
                body_index_job['running'] = False

        threading.Thread(target=_run, daemon=True).start()
        return {'started': True, 'total': body_index_job['total']}

    @app.get('/api/body-index/status')
    def api_body_index_status():
        return {
            'running': body_index_job['running'],
            'done': body_index_job['done'],
            'total': body_index_job['total'],
            'error': body_index_job['error'],
            'pending': len(db.get_unbody_indexed_files()),
        }

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------

    @app.get('/api/files')
    def api_list_files(limit: int = 100, offset: int = 0):
        rows = db.list_files_with_embedding_flag(limit=limit, offset=offset)
        tag_map = manual.list_tags_for_checksums([row[3] for row in rows])
        result = []
        for row in rows:
            file_id, path, has_emb, checksum = row[0], row[1], row[2], row[3]
            result.append({
                'id': file_id,
                'path': path,
                'has_embedding': bool(has_emb),
                'tags': tag_map.get(checksum, []),
            })
        return result

    @app.get('/api/files/{file_id}')
    def api_get_file(file_id: int):
        row = _file_or_404(file_id)
        tags = manual.get_whole_image_tag_labels(row['checksum'])
        info = _row_to_dict(row, tags)
        detected = db.get_detected_classes(file_id)
        info['detected_classes'] = detected
        return info

    def _whole_tags(checksum):
        return [
            {'id': t['id'], 'label': t['label'], 'polarity': t['polarity'], 'favorite': bool(t['favorite'])}
            for t in manual.get_tags(checksum) if t['x1'] is None
        ]

    @app.post('/api/files/{file_id}/tags')
    def api_add_tag(file_id: int, body: TagBody):
        row = _file_or_404(file_id)
        checksum = row['checksum']
        tag = body.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail='Tag must not be empty')
        polarity = 'negative' if body.polarity == 'negative' else 'positive'
        existing = [t for t in _whole_tags(checksum) if t['label'] == tag and t['polarity'] == polarity]
        if not existing:
            manual.add_tag(checksum, tag, polarity=polarity)
            if polarity == 'negative':
                # A human rejected this YOLO detection — remove it at the source so it
                # doesn't come back next time detections are viewed or re-indexed.
                db.remove_detection(file_id, tag)
        return {'tags': _whole_tags(checksum)}

    @app.delete('/api/files/{file_id}/tags/{tag_id}')
    def api_remove_tag(file_id: int, tag_id: int):
        row = _file_or_404(file_id)
        manual.remove_tag(tag_id)
        return {'tags': _whole_tags(row['checksum'])}

    @app.delete('/api/files/{file_id}/tags-by-label')
    def api_remove_tag_by_label(file_id: int, label: str, polarity: str = 'positive'):
        """Undo a confirm or reject made via the tag-suggestion swipe stream —
        removes the specific tag row by (label, polarity) instead of by numeric
        id, since the swipe stream never learns the id the POST endpoint
        assigned (it doesn't read the response body)."""
        row = _file_or_404(file_id)
        manual.remove_tag_by_label(row['checksum'], label, polarity)
        return {'tags': _whole_tags(row['checksum'])}

    @app.patch('/api/files/{file_id}/tags/{tag_id}')
    def api_update_tag(file_id: int, tag_id: int, body: TagLabelBody):
        row = _file_or_404(file_id)
        label = body.label.strip()
        if not label:
            raise HTTPException(status_code=400, detail='Label must not be empty')
        manual.update_tag_label(tag_id, label)
        return {'tags': _whole_tags(row['checksum'])}

    @app.post('/api/tags/{tag_id}/favorite')
    def api_set_tag_favorite(tag_id: int, body: FavoriteBody):
        manual.set_tag_favorite(tag_id, body.favorite)
        return {'id': tag_id, 'favorite': body.favorite}

    @app.post('/api/files/{file_id}/tags/region')
    def api_add_spatial_tag(file_id: int, body: SpatialTagBody):
        row = _file_or_404(file_id)
        label = body.label.strip()
        if not label:
            raise HTTPException(status_code=400, detail='Label must not be empty')
        if len(body.bbox) != 4:
            raise HTTPException(status_code=400, detail='bbox must be [x1,y1,x2,y2]')
        x1, y1, x2, y2 = body.bbox
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(status_code=400, detail='Invalid bbox')

        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        from PIL import Image as PILImage
        with PILImage.open(abs_path) as img:
            width, height = img.size

        tag_id = manual.add_spatial_tag(row['checksum'], label, x1, y1, x2, y2, width, height)
        return {'id': tag_id, 'label': label, 'bbox': [x1, y1, x2, y2]}

    @app.post('/api/files/{file_id}/reindex')
    def api_reindex_tags(file_id: int):
        row = _file_or_404(file_id)
        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        from media_manager.detector import YOLOWorldDetector, load_vocab_from_file, merge_vocab
        vocab_path = os.path.join(media_dir, 'search_terms.txt')
        base_vocab = load_vocab_from_file(vocab_path)
        manual_labels = manual.get_all_positive_labels()
        negated = manual.get_negated_labels(row['checksum'])
        vocab = merge_vocab(base_vocab, manual_labels, exclude=negated)

        detector = _get_object_detector()
        detector.set_vocab(vocab)
        results = detector.detect_images([abs_path])
        _, detections, error = results[0]
        if error:
            raise HTTPException(status_code=500, detail=f'Detection failed: {error}')

        db.insert_detections(file_id, detections, YOLOWorldDetector.model_id())
        detected_classes = [c for c in db.get_detected_classes(file_id) if c not in negated]
        return {'detected_classes': detected_classes}

    @app.get('/api/tags')
    def api_list_tags():
        rows = manual.list_all_tags()
        return [{'tag': r[0], 'count': r[1]} for r in rows]

    @app.get('/tags', response_class=HTMLResponse)
    def tags_page(request: Request):
        all_tags = manual.list_all_tags()
        tags = [{'label': r[0], 'count': r[1]} for r in all_tags]
        return templates.TemplateResponse(request, 'tags.html', {
            'tags': tags,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
        })

    @app.put('/api/tags/{label}')
    def api_rename_tag_label(label: str, body: TagLabelBody):
        new_label = body.label.strip()
        if not new_label:
            raise HTTPException(status_code=400, detail='Label must not be empty')
        manual.rename_tag_label(label, new_label)
        return {'label': new_label}

    @app.delete('/api/tags/{label}')
    def api_delete_tag_label(label: str):
        manual.delete_tag_label(label)
        return {'ok': True}

    # CLIP tag-relevance (single object/concept label) sits closer to the
    # face-identity similarity regime than the same-scene-photo regime that
    # justified SET_SUGGEST_THRESHOLD=0.75, so this borrows SUGGEST_THRESHOLD's
    # value rather than inventing a new one without data.
    TAG_SUGGEST_THRESHOLD = 0.45

    def _next_tag_suggestions(label, count, exclude_ids, bias_file_id=None, bias=None):
        """Rank not-yet-decided files against the centroid of files already
        positively tagged `label`. Returns up to `count` dicts, best score first:
        [{'ref': str, 'file_id': int, 'label': str, 'score': float}, ...].

        A file is excluded if it already has a tag row (either polarity) for this
        exact label — manual.get_decided_checksums_for_tag covers both a past
        confirm and a past reject, so neither is ever re-suggested.

        Bias is keyed on the specific just-confirmed/rejected file_id (the label
        itself is constant for the whole stream, so biasing on it would be a
        no-op). Unlike the categorical identity-match bias faces use, this blends
        a continuous file-to-file similarity signal into the file-to-centroid
        score, since there's no discrete identity dimension to bias tags by."""
        from media_manager.similarity import mean_normalized_centroid, rank_by_similarity

        member_checksums = manual.get_files_by_tag(label, limit=1000)
        member_ids = [r['id'] for r in db.get_files_by_checksums(member_checksums)]
        member_embeddings = db.get_embeddings_for_files(member_ids)
        centroid = mean_normalized_centroid([e for _fid, e in member_embeddings])
        if centroid is None:
            return []

        decided = manual.get_decided_checksums_for_tag(label)
        candidates = [
            row for row in db.get_all_embeddings()
            if row[3] not in decided and row[0] not in exclude_ids
        ]
        ranked = rank_by_similarity(centroid, candidates, embedding_index=2)
        qualifying = [(row, score) for row, score in ranked if score >= TAG_SUGGEST_THRESHOLD]

        if bias_file_id is not None and bias in ('confirm', 'reject'):
            import numpy as np

            bias_rows = db.get_embeddings_for_files([bias_file_id])
            if bias_rows:
                bias_vec = np.frombuffer(bias_rows[0][1], dtype=np.float32)
                sign = 1.0 if bias == 'confirm' else -1.0

                def blended(item):
                    row, score = item
                    sim_to_bias = float(np.frombuffer(row[2], dtype=np.float32).dot(bias_vec))
                    return score + sign * sim_to_bias

                qualifying.sort(key=blended, reverse=True)

        return _attach_file_meta([
            {'ref': str(row[0]), 'file_id': row[0], 'label': label, 'score': round(float(score), 3)}
            for row, score in qualifying[:count]
        ])

    @app.post('/api/tags/{label}/suggestions/next')
    def api_next_tag_suggestions(label: str, body: SwipeExcludeBody, count: int = 10,
                                  bias_file_id: str = '', bias: str = ''):
        exclude_ids = {int(x) for x in body.exclude if x.isdigit()}
        bias_id = int(bias_file_id) if bias_file_id.strip().isdigit() else None
        cards = _next_tag_suggestions(label, count, exclude_ids, bias_id, bias or None)
        return {'cards': cards}

    @app.get('/api/vocab')
    def api_vocab():
        """Every label worth suggesting while typing a tag or a labeled region — the same
        merged vocabulary (search_terms.txt + manual.db's confirmed tags) that feeds
        YOLO-World's detector, so autocomplete only ever offers real, usable class names."""
        from media_manager.detector import load_vocab_from_file, merge_vocab
        vocab_path = os.path.join(media_dir, 'search_terms.txt')
        base_vocab = load_vocab_from_file(vocab_path)
        vocab = merge_vocab(base_vocab, manual.get_all_positive_labels())
        return {'vocab': vocab}

    @app.post('/api/search-palette')
    def api_search_palette(body: SearchPaletteBody):
        """Resolves the current chip combination (+ an optional free-text filename
        fallback) to a result grid. Suggestion *ranking* — matching what you're
        typing against known tags/categories/faces/sets — no longer happens here
        at all: the frontend preloads /api/tags, /api/categories, /api/identities,
        /api/sets once and fuzzy-matches them client-side (same fuzzyScore used by
        the entity picker), so that part is instant and needs no round trip. This
        endpoint only has to do real checksum-set work, which is inherently a
        server-side/DB job — but only once per call now, not once per candidate."""
        q = body.q.strip().lower()
        chips = [{'type': c.type, 'value': c.value} for c in body.chips]
        # Computed once per request — see each _*_checksums_by_* helper's
        # docstring for why resolving per-chip from scratch (fine for the rare
        # /search f= page load) doesn't scale to a call made on every keystroke.
        category_map = _category_checksums_by_name()
        tag_map = _tag_checksums_by_label()
        set_map = _set_checksums_by_id()
        identity_map = _identity_checksums_by_name(set_map)

        media = []
        sets_result = []
        people_result = []
        total_count = 0
        checksums = _intersect_chips(chips, category_map=category_map, tag_map=tag_map,
                                      set_map=set_map, identity_map=identity_map)
        if checksums is not None or q:
            if checksums is None:
                # q alone (no chips yet) — filename substring is the only sensible
                # "just typed text" match; there's no free-text object-detection
                # search here (that's what the plain /search q= path is for).
                checksums = {row[2] for row in db.search_by_path_substring(q, limit=200)}
            total_count = len(checksums)
            if checksums:
                rows = db.get_files_by_checksums(list(checksums)[:16])
                enriched_rows = [(r['id'], r['path'], False, r['checksum']) for r in rows]
                media = _enrich_rows(enriched_rows)

                # Which sets/people are actually represented in the current result
                # set — shown as their own tiles alongside individual photos, so a
                # result reads as "these sets / these people / these photos", not
                # just an undifferentiated photo grid.
                sets_map = {}
                identities_map = {}
                for chunk in _chunked(checksums):
                    sets_map.update(manual.get_sets_for_checksums(chunk))
                    identities_map.update(manual.get_identities_for_checksums(chunk))

                set_agg = {}
                for cs, sets_here in sets_map.items():
                    for s in sets_here:
                        agg = set_agg.setdefault(s['id'], {'id': s['id'], 'name': s['name'], 'count': 0, 'checksum': cs})
                        agg['count'] += 1
                sets_result = sorted(set_agg.values(), key=lambda s: s['count'], reverse=True)[:6]
                for s in sets_result:
                    thumb_rows = db.get_files_by_checksums([s.pop('checksum')])
                    s['thumb_file_id'] = thumb_rows[0]['id'] if thumb_rows else None

                people_agg = {}
                for cs, names in identities_map.items():
                    for name in names:
                        people_agg[name] = people_agg.get(name, 0) + 1
                face_ids = manual.get_representative_face_ids()
                people_result = sorted(
                    [{'name': n, 'count': c, 'face_id': face_ids.get(n)} for n, c in people_agg.items()],
                    key=lambda p: p['count'], reverse=True
                )[:6]

        return {'media': media, 'sets': sets_result, 'people': people_result, 'total_count': total_count}

    @app.get('/duplicates', response_class=HTMLResponse)
    def duplicates_page(request: Request):
        """Content seen at more than one path — the point of content-addressable
        identity: surfaces real duplicates for manual review instead of silently
        tracking (and letting a user separately tag/sort) each copy. Deliberately no
        delete action here — that's destructive on real files and stays human-driven."""
        groups = [
            {'file_id': file_id, 'paths': [p for p, _ in db.get_paths_for_file(file_id)]}
            for file_id, _path_count in db.find_duplicates(limit=200)
        ]
        return templates.TemplateResponse(request, 'duplicates.html', {'groups': groups})

    @app.get('/sets', response_class=HTMLResponse)
    def sets_page(request: Request, favorite: bool = False, studio: str = ''):
        all_tags = manual.list_all_tags()
        studio_filter = studio or None
        sets = []
        for r in manual.list_sets(favorite_only=favorite, studio=studio_filter):
            thumb_id = None
            first_checksums = manual.get_files_by_set(r['id'], limit=1)
            if first_checksums:
                thumb_rows = db.get_files_by_checksums(first_checksums)
                if thumb_rows:
                    thumb_id = thumb_rows[0]['id']
            sets.append({
                'id': r['id'], 'name': r['name'], 'studio': r['studio'],
                'image_count': r['image_count'], 'favorite': bool(r['favorite']),
                'thumb_id': thumb_id,
            })
        return templates.TemplateResponse(request, 'sets.html', {
            'sets': sets,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'favorite_only': favorite,
            'studio_filter': studio_filter,
        })

    @app.get('/studios', response_class=HTMLResponse)
    def studios_page(request: Request):
        all_tags = manual.list_all_tags()
        studios = [
            {'name': r['studio'], 'set_count': r['set_count'], 'image_count': r['image_count']}
            for r in manual.list_studios()
        ]
        return templates.TemplateResponse(request, 'studios.html', {
            'studios': studios,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/api/studios')
    def api_list_studios():
        return [
            {'name': r['studio'], 'set_count': r['set_count'], 'image_count': r['image_count']}
            for r in manual.list_studios()
        ]

    # Cosine similarity above this, an image outside the set is suggested as a
    # possible member. CLIP image-image similarity runs hotter than face similarity
    # (same-scene photos commonly score 0.7-0.9), hence the higher default.
    SET_SUGGEST_THRESHOLD = 0.75

    def _find_similar_files_for_set(set_id, threshold, limit=12, offset=0, exclude_ids=None, avoid_existing=True):
        """Images not yet in the set whose CLIP embedding is close to the set's
        centroid (mean of its members' embeddings) — "images that might belong
        here". `offset` skips that many already-threshold-passing candidates
        before taking `limit`, for infinite-scroll pagination; the full ranking
        is recomputed each call (as before offset existed), which is fine at
        personal-library scale.

        `exclude_ids`, when given (a set of file ids), filters the
        threshold-passing candidates by id instead and `offset` is ignored — the
        swipe-buffer caller (set-detail's swipe stack) has no stable page
        position since confirming a file removes it from the candidate pool on
        the very next call, so it just wants "up to `limit` more I haven't
        already been shown", not a page number.

        `avoid_existing` (default True — a UI toggle, on by default): files
        already belonging to ANY set are stable-sorted after ones in no set at
        all, so browsing/swiping surfaces still-unsorted photos first without
        hard-excluding files that happen to already be organized somewhere —
        applied across the whole ranked pool before slicing to `limit`, not
        just within whatever page/buffer size was requested, so it can
        actually pull in fresh candidates rather than just reordering a
        handful already selected."""
        from media_manager.similarity import mean_normalized_centroid, rank_by_similarity

        member_checksums = manual.get_files_by_set(set_id, limit=1000)
        member_ids = [r['id'] for r in db.get_files_by_checksums(member_checksums)]
        member_embeddings = db.get_embeddings_for_files(member_ids)
        centroid = mean_normalized_centroid([e for _fid, e in member_embeddings])
        if centroid is None:
            return []

        member_checksum_set = set(member_checksums)
        excluded_checksum_set = manual.get_excluded_checksums_for_set(set_id)
        candidates = [
            row for row in db.get_all_embeddings()
            if row[3] not in member_checksum_set and row[3] not in excluded_checksum_set
        ]
        if not candidates:
            return []
        ranked = rank_by_similarity(centroid, candidates, embedding_index=2)
        passing = [((fid, path, cs), score) for (fid, path, _emb, cs), score in ranked if score >= threshold]
        if avoid_existing and passing:
            member_checksums_anywhere = manual.get_all_set_member_checksums()
            passing.sort(key=lambda item: item[0][2] in member_checksums_anywhere)
        if exclude_ids:
            passing = [item for item in passing if item[0][0] not in exclude_ids]
            page = passing[:limit]
        else:
            page = passing[offset:offset + limit]

        rows = [(fid, path, True, cs) for (fid, path, cs), _score in page]
        scores_map = {fid: round(score, 3) for (fid, _path, _cs), score in page}
        cards = _enrich_rows(rows, scores=scores_map)
        for card in cards:
            card['ref'] = str(card['id'])
        return cards

    def _find_best_sets_for_file(file_id, checksum, threshold=SET_SUGGEST_THRESHOLD, limit=3):
        """The reverse of _find_similar_files_for_set: given one photo, rank every
        set it's not already in by how close the photo's embedding is to that set's
        centroid — "which set does this probably belong to". Sets are cheap and
        unpaginated everywhere else in the app (personal-library scale), so unlike
        the set-detail version this runs synchronously over all of them per call."""
        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            return []
        import numpy as np
        query_emb = np.frombuffer(emb_bytes, dtype=np.float32)

        member_of = {s['id'] for s in manual.get_sets_for_checksums([checksum]).get(checksum, [])}
        candidates = [s for s in manual.list_sets() if s['id'] not in member_of]
        if not candidates:
            return []

        scored = []
        for set_row in candidates:
            member_checksums = manual.get_files_by_set(set_row['id'], limit=1000)
            member_ids = [r['id'] for r in db.get_files_by_checksums(member_checksums)]
            member_embeddings = db.get_embeddings_for_files(member_ids)
            if not member_embeddings:
                continue
            vecs = np.stack([np.frombuffer(e, dtype=np.float32) for _, e in member_embeddings])
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue
            centroid = centroid / norm
            score = float(centroid.dot(query_emb))
            if score >= threshold:
                scored.append((set_row, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {'id': s['id'], 'name': s['name'], 'studio': s['studio'], 'score': round(score, 3)}
            for s, score in scored[:limit]
        ]

    @app.get('/api/files/{file_id}/suggested-sets')
    def api_suggested_sets_for_file(file_id: int):
        row = _file_or_404(file_id)
        return {'results': _find_best_sets_for_file(file_id, row['checksum'])}

    def _people_present_in_set(set_id, member_checksums):
        """Everyone confirmed to appear in this set, for the "People present"
        strip at the top of the set page: name, a representative face crop (if
        any face for them exists anywhere), and their age/gender estimate for
        that specific face (if one's been run) — else the file_id of a photo of
        theirs, so the template can offer "Estimate age" against the existing
        whole-photo endpoint instead. Order comes straight from
        manual.get_people_present_in_set (manually assigned people first, then
        detected-face-only people, each name once) — not re-sorted here. Each
        entry also carries 'set_linked': True only for an explicit
        identity_set_assignments link, the one case the set page's "remove"
        action (DELETE /api/identities/{name}/assign-set/{set_id}) applies to."""
        people = manual.get_people_present_in_set(set_id, member_checksums)
        result = []
        for identity, (face_id, set_linked) in people.items():
            entry = {'identity': identity, 'face_ref': None, 'age': None, 'gender': None, 'file_id': None,
                     'set_linked': set_linked}
            if face_id is not None:
                face_ref = f'manual:{face_id}'
                entry['face_ref'] = face_ref
                face_row = manual.get_face(face_id)
                if face_row is not None:
                    file_row = db.get_file_by_checksum(face_row['checksum'])
                    if file_row is not None:
                        entry['file_id'] = file_row['id']
                estimate = manual.get_age_estimate_for_face_ref(face_ref)
                if estimate is not None and estimate['age'] is not None:
                    entry['age'] = estimate['age']
                    entry['gender'] = estimate['gender']
            result.append(entry)
        return result

    @app.get('/sets/{set_id}', response_class=HTMLResponse)
    def set_detail_page(request: Request, set_id: int, sort: str = 'added', order: str = 'desc'):
        set_row = manual.get_set(set_id)
        if set_row is None:
            raise HTTPException(status_code=404, detail='Set not found')
        all_tags = manual.list_all_tags()
        checksums = manual.get_files_by_set(set_id, limit=500)
        file_rows = _sort_file_rows(db.get_files_by_checksums(checksums), sort, order)
        rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
        files = _enrich_rows(rows)
        if sort == 'age':
            files = _sort_cards_by_age(files, order)
        people_present = _people_present_in_set(set_id, checksums)
        return templates.TemplateResponse(request, 'set_detail.html', {
            'set': dict(set_row),
            'files': files,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'set_suggest_threshold': SET_SUGGEST_THRESHOLD,
            'people_present': people_present,
            'sort': sort,
            'order': order,
        })

    @app.post('/api/sets/{set_id}/similar-files')
    def api_similar_files_for_set(set_id: int, body: SwipeExcludeBody, threshold: float = SET_SUGGEST_THRESHOLD,
                                   limit: int = 12, offset: int = 0, avoid_existing: bool = True):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        threshold = max(0.0, min(1.0, threshold))
        offset = max(0, offset)
        exclude_ids = {int(r) for r in body.exclude if r.isdigit()} or None
        results = _find_similar_files_for_set(set_id, threshold, limit=limit, offset=offset, exclude_ids=exclude_ids,
                                                avoid_existing=avoid_existing)
        # 'cards' is the same list under the key swipe-core.js's fetchMoreUrl
        # response contract expects; 'results' stays for the pre-existing grid
        # (offset-based load-more + threshold-expand) callers, unchanged.
        return {'results': results, 'cards': results}

    @app.get('/api/sets')
    def api_list_sets(favorite: bool = False):
        return [
            {'id': r['id'], 'name': r['name'], 'studio': r['studio'],
             'image_count': r['image_count'], 'favorite': bool(r['favorite'])}
            for r in manual.list_sets(favorite_only=favorite)
        ]

    @app.post('/api/sets')
    def api_create_set(body: SetBody):
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Set name must not be empty')
        studio = body.studio.strip() if body.studio else None
        set_id = manual.create_set(name, studio)
        return {'id': set_id, 'name': name, 'studio': studio}

    @app.put('/api/sets/{set_id}')
    def api_rename_set(set_id: int, body: SetBody):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Set name must not be empty')
        studio = body.studio.strip() if body.studio else None
        manual.rename_set(set_id, name, studio)
        return {'id': set_id, 'name': name, 'studio': studio}

    @app.delete('/api/sets/{set_id}')
    def api_delete_set(set_id: int):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        manual.delete_set(set_id)
        return {'ok': True}

    @app.post('/api/sets/{set_id}/favorite')
    def api_set_favorite_set(set_id: int, body: FavoriteBody):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        manual.set_set_favorite(set_id, body.favorite)
        return {'id': set_id, 'favorite': body.favorite}

    # ------------------------------------------------------------------
    # Categories (single-value, ML-assisted classification)
    # ------------------------------------------------------------------

    def _all_categories_for_nav():
        counts = get_category_counts(manual, db)
        return sorted(
            ({'name': name, 'count': count} for name, count in counts.items()),
            key=lambda c: c['name']
        )

    @app.get('/categories', response_class=HTMLResponse)
    def categories_page(request: Request):
        all_tags = manual.list_all_tags()
        all_categories = _all_categories_for_nav()
        categories = []
        for r in manual.list_categories():
            thumb_id = None
            example_checksums = manual.get_example_checksums_for_category(r['id'], limit=1)
            if example_checksums:
                thumb_rows = db.get_files_by_checksums(example_checksums)
                if thumb_rows:
                    thumb_id = thumb_rows[0]['id']
            categories.append({
                'id': r['id'], 'name': r['name'], 'temperature': r['temperature'],
                'image_count': r['image_count'], 'thumb_id': thumb_id,
            })
        return templates.TemplateResponse(request, 'categories.html', {
            'categories': categories,
            'all_tags': all_tags,
            'all_categories': all_categories,
        })

    @app.get('/api/categories')
    def api_list_categories():
        return [
            {'id': r['id'], 'name': r['name'], 'temperature': r['temperature'], 'image_count': r['image_count']}
            for r in manual.list_categories()
        ]

    @app.post('/api/categories')
    def api_create_category(body: CategoryBody):
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Category name must not be empty')
        temperature = body.temperature if body.temperature is not None else 0.75
        category_id = manual.create_category(name, temperature)
        cat = manual.get_category(category_id)
        return {'id': cat['id'], 'name': cat['name'], 'temperature': cat['temperature']}

    @app.put('/api/categories/{category_id}')
    def api_update_category(category_id: int, body: CategoryBody):
        if manual.get_category(category_id) is None:
            raise HTTPException(status_code=404, detail='Category not found')
        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail='Category name must not be empty')
            manual.rename_category(category_id, name)
        if body.temperature is not None:
            manual.set_category_temperature(category_id, body.temperature)
        row = manual.get_category(category_id)
        return {'id': row['id'], 'name': row['name'], 'temperature': row['temperature']}

    @app.delete('/api/categories/{category_id}')
    def api_delete_category(category_id: int):
        if manual.get_category(category_id) is None:
            raise HTTPException(status_code=404, detail='Category not found')
        manual.delete_category(category_id)
        return {'ok': True}

    @app.post('/api/files/{file_id}/category')
    def api_set_file_category(file_id: int, body: CategoryBody):
        row = _file_or_404(file_id)
        if body.category_id is None:
            raise HTTPException(status_code=400, detail='category_id is required')
        cat = manual.get_category(body.category_id)
        if cat is None:
            raise HTTPException(status_code=404, detail='Category not found')
        manual.set_file_category(row['checksum'], body.category_id)
        return {'category': {'name': cat['name'], 'source': 'manual', 'score': None}}

    @app.delete('/api/files/{file_id}/category')
    def api_clear_file_category(file_id: int):
        """Explicit clear — records an explicit 'no category' override, not a
        revert to whatever the ML auto-match would otherwise suggest."""
        row = _file_or_404(file_id)
        manual.set_file_category(row['checksum'], None)
        return {'category': None}

    @app.delete('/api/files/{file_id}/category/decision')
    def api_undo_file_category_decision(file_id: int):
        """Undo a confirm or reject made via the category-suggestion swipe
        stream — hard-deletes the override row entirely (unlike the plain
        DELETE above, which upserts an explicit 'no category' row), restoring
        'no manual decision was ever made'. Safe because the suggestion stream
        never offers a file that already had an override."""
        row = _file_or_404(file_id)
        manual.clear_category_override(row['checksum'])
        return {'ok': True}

    def _next_category_suggestions(category_id, count, exclude_refs):
        """Rank currently-unresolved-or-differently-resolved files against this ONE
        category's centroid (mirrors media_manager.py's MediaManager.match_categories
        exactly — same centroid-vs-temperature-threshold math — but interactive/
        one-at-a-time instead of batch-write). Returns up to `count` dicts, best
        score first: [{'ref': str, 'file_id': int, 'score': float}, ...].

        A file is excluded from candidacy if it has ANY manual category override —
        assigned to this category, assigned to a DIFFERENT category, or an explicit
        "no category" decision — manual decisions must never be re-suggested,
        matching match_categories' own skip rule exactly.

        A file that already has an ML auto-match to a DIFFERENT category IS still
        offered here: auto-matches are provisional machine guesses, not human
        decisions, and confirming here always writes a manual override that
        outranks any auto-match per category_resolver.py's precedence rules."""
        from media_manager.similarity import mean_normalized_centroid, rank_by_similarity
        from media_manager.swipe_support import bias_reorder

        cat = manual.get_category(category_id)
        if cat is None:
            return []

        example_checksums = manual.get_example_checksums_for_category(category_id)
        example_ids = [r['id'] for r in db.get_files_by_checksums(example_checksums)]
        centroid = mean_normalized_centroid(
            [e for _fid, e in db.get_embeddings_for_files(example_ids)]
        )
        if centroid is None:
            return []

        all_candidates = db.get_all_embeddings()  # (file_id, path, embedding, checksum)
        overridden_checksums = manual.get_all_category_override_checksums()

        filtered = [
            c for c in all_candidates
            if c[3] not in overridden_checksums and f"file:{c[0]}" not in exclude_refs
        ]
        ranked = rank_by_similarity(centroid, filtered, embedding_index=2)
        threshold = cat['temperature']
        qualifying = [(c, score) for c, score in ranked if score >= threshold]

        # Wrapped in the shared bias-reorder helper for structural consistency with
        # every other feature's swipe endpoint, even though bias is inert here: a
        # single-category detail page has only one possible "key" for every
        # candidate, so confirm/reject bias can never produce a different
        # partition than plain score-desc order.
        ordered = bias_reorder(qualifying, key_fn=lambda pair: category_id,
                                bias_key=None, bias_action=None)

        return _attach_file_meta([
            {'ref': f"file:{c[0]}", 'file_id': c[0], 'score': round(float(score), 3)}
            for c, score in ordered[:count]
        ])

    @app.get('/categories/{category_id}', response_class=HTMLResponse)
    def category_detail_page(request: Request, category_id: int, sort: str = 'added', order: str = 'desc'):
        cat = manual.get_category(category_id)
        if cat is None:
            raise HTTPException(status_code=404, detail='Category not found')
        all_tags = manual.list_all_tags()
        all_categories = _all_categories_for_nav()
        example_checksums = manual.get_example_checksums_for_category(category_id, limit=500)
        file_rows = _sort_file_rows(db.get_files_by_checksums(example_checksums), sort, order)
        rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
        files = _enrich_rows(rows)
        if sort == 'age':
            files = _sort_cards_by_age(files, order)
        return templates.TemplateResponse(request, 'category_detail.html', {
            'category': dict(cat),
            'files': files,
            'all_tags': all_tags,
            'all_categories': all_categories,
            'sort': sort,
            'order': order,
        })

    @app.post('/api/categories/{category_id}/suggestions/next')
    def api_next_category_suggestions(category_id: int, body: SwipeExcludeBody, count: int = 10):
        if manual.get_category(category_id) is None:
            raise HTTPException(status_code=404, detail='Category not found')
        exclude_refs = set(body.exclude)
        cards = _next_category_suggestions(category_id, count, exclude_refs)
        return {'cards': cards}

    @app.post('/api/files/{file_id}/sets')
    def api_assign_set(file_id: int, body: SetBody):
        """Adds a set membership — a file can belong to any number of sets, so this
        never replaces an existing one."""
        row = _file_or_404(file_id)
        checksum = row['checksum']
        if body.set_id is not None:
            set_row = manual.get_set(body.set_id)
            if set_row is None:
                raise HTTPException(status_code=404, detail='Set not found')
            manual.assign_file_to_set(checksum, set_row['id'])
            return {'id': set_row['id'], 'name': set_row['name'], 'studio': set_row['studio']}

        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Set name must not be empty')
        studio = body.studio.strip() if body.studio else None
        set_id = manual.create_set(name, studio)
        manual.assign_file_to_set(checksum, set_id)
        return {'id': set_id, 'name': name, 'studio': studio}

    @app.delete('/api/files/{file_id}/sets/{set_id}')
    def api_remove_set(file_id: int, set_id: int):
        row = _file_or_404(file_id)
        manual.remove_file_from_set(row['checksum'], set_id)
        return {'ok': True}

    @app.post('/api/files/{file_id}/sets/{set_id}/exclude')
    def api_exclude_from_set(file_id: int, set_id: int):
        """The suggestion stream's reject action: records a permanent 'this file
        is NOT in this set' decision so it's never suggested for this set again."""
        row = _file_or_404(file_id)
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        manual.exclude_file_from_set(row['checksum'], set_id)
        return {'ok': True}

    @app.delete('/api/files/{file_id}/sets/{set_id}/exclude')
    def api_remove_set_exclusion(file_id: int, set_id: int):
        """Undoes a prior exclude (the reject-swipe's Ctrl+Z)."""
        row = _file_or_404(file_id)
        manual.remove_set_exclusion(row['checksum'], set_id)
        return {'ok': True}

    @app.post('/api/files/{file_id}/favorite')
    def api_set_file_favorite(file_id: int, body: FavoriteBody):
        row = _file_or_404(file_id)
        manual.set_file_favorite(row['checksum'], body.favorite)
        return {'id': file_id, 'favorite': body.favorite}

    @app.patch('/api/files/{file_id}/title')
    def api_set_file_title(file_id: int, body: TitleBody):
        row = _file_or_404(file_id)
        manual.set_file_title(row['checksum'], body.title)
        return {'id': file_id, 'title': manual.get_file_title(row['checksum'])}

    @app.post('/api/files/{file_id}/estimate-age')
    def api_estimate_age(file_id: int):
        """Experimental — see age_estimator.py. Runs MiVOLO (in its own isolated venv,
        via subprocess) on every face already known for this photo."""
        from media_manager.age_estimator import MODEL_ID

        row = _file_or_404(file_id)
        checksum = row['checksum']
        faces = _combined_faces_for_file(file_id, checksum)
        if not faces:
            return {'results': [], 'message': 'No faces detected in this photo yet.'}

        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        try:
            results = _get_age_estimator().estimate(abs_path, faces)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        manual.save_age_estimates(checksum, results, MODEL_ID)
        return {'results': results}

    # Cosine similarity above this, an unidentified face is suggested as a match
    # for an already-known person (embeddings are pre-normalized by InsightFace).
    SUGGEST_THRESHOLD = 0.45

    def _suggest_names(unidentified_rows):
        """Given (face_id, file_id, path, bbox, embedding) rows, return
        {face_id: {'name': str, 'score': float}} for rows whose embedding is
        close to an already-named face — lets a repeat appearance of a known
        person be confirmed with one click instead of retyping their name."""
        named = manual.get_named_face_embeddings()
        if not named:
            return {}
        import numpy as np
        names = [n for n, _ in named]
        matrix = np.stack([np.frombuffer(e, dtype=np.float32) for _, e in named])
        suggestions = {}
        for row in unidentified_rows:
            face_id, emb_bytes = row[0], row[4]
            if not emb_bytes:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            scores = matrix.dot(vec)
            best_idx = int(scores.argmax())
            if scores[best_idx] >= SUGGEST_THRESHOLD:
                suggestions[face_id] = {'name': names[best_idx], 'score': round(float(scores[best_idx]), 3)}
        return suggestions

    def _unpromoted_auto_faces(limit=200):
        promoted = manual.get_promoted_source_ids()
        return [r for r in db.get_unidentified_faces(limit=None) if r[0] not in promoted][:limit]

    def _get_face_embedding(face_id):
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            return row['embedding'] if row is not None else None
        cursor = db.conn.cursor()
        cursor.execute('SELECT embedding FROM faces WHERE id = ?', (raw_id,))
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def _get_face_file_id(face_id):
        """The id of the file this face belongs to — used to jump back to that photo
        after acting on a face (e.g. assigning a name found via similar-faces search)."""
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                return None
            file_row = db.get_file_by_checksum(row['checksum'])
            return file_row['id'] if file_row is not None else None
        cursor = db.conn.cursor()
        cursor.execute('SELECT file_id FROM faces WHERE id = ?', (raw_id,))
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def _find_similar_faces_to_ref(face_id, threshold, limit=20):
        """Every other face — named, unnamed-but-manual, or still-unnamed-auto — whose
        embedding is within `threshold` cosine similarity of this one face. Unlike
        _find_similar_unknown_faces (which needs an existing person name to search
        from), this works from a single face with no name at all, so you can spot a
        repeat appearance of the same person before deciding what to call them."""
        query_emb = _get_face_embedding(face_id)
        if query_emb is None:
            return []
        import numpy as np
        query_vec = np.frombuffer(query_emb, dtype=np.float32)

        candidates = []  # (ref, file_id, path, identity, emb_bytes)
        for row in _unpromoted_auto_faces(limit=None):
            face_id_, file_id_, path_, _bbox, emb_bytes = row
            ref = f"auto:{face_id_}"
            if ref == face_id or not emb_bytes:
                continue
            candidates.append((ref, file_id_, path_, None, emb_bytes))
        for row in manual.get_all_faces_with_embedding():
            manual_id, checksum, identity, emb_bytes = row
            ref = f"manual:{manual_id}"
            if ref == face_id or not emb_bytes:
                continue
            file_row = db.get_file_by_checksum(checksum)
            if file_row is None:
                continue
            candidates.append((ref, file_row['id'], file_row['path'], identity, emb_bytes))

        scored = []
        for ref, file_id_, path_, identity, emb_bytes in candidates:
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(query_vec.dot(vec))
            if score >= threshold:
                scored.append((score, ref, file_id_, path_, identity))
        scored.sort(reverse=True)
        return [
            {
                'ref': ref, 'file_id': file_id_, 'path': path_,
                'filename': os.path.basename(path_), 'identity': identity,
                'score': round(score, 3),
            }
            for score, ref, file_id_, path_, identity in scored[:limit]
        ]

    def _find_similar_unknown_faces(name, threshold, limit=20):
        """Unidentified auto-detected faces whose embedding is within `threshold` cosine
        similarity of any of `name`'s confirmed faces — powers both the initial /search?person=
        render and the "expand similar face search" slider when nothing is found at the default."""
        ref_embeddings = manual.get_embeddings_for_identity(name)
        unidentified = _unpromoted_auto_faces(limit=None)
        results = []
        if not ref_embeddings or not unidentified:
            return results
        import numpy as np
        ref_matrix = np.stack([np.frombuffer(e, dtype=np.float32) for e in ref_embeddings])
        scored = []
        for row in unidentified:
            face_id_, file_id_, path_, bbox_, emb_bytes = row
            if not emb_bytes:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(ref_matrix.dot(vec).max())
            if score >= threshold:
                scored.append((score, face_id_, file_id_, path_))
        scored.sort(reverse=True)
        for score, face_id_, file_id_, path_ in scored[:limit]:
            results.append({
                'ref': f"auto:{face_id_}",
                'file_id': file_id_,
                'path': path_,
                'filename': os.path.basename(path_),
                'score': round(score, 3),
            })
        return results

    @app.get('/faces', response_class=HTMLResponse)
    def faces_page(request: Request, favorite: bool = False):
        all_tags = manual.list_all_tags()
        identities = manual.get_all_identities()
        favorite_faces = []
        if favorite:
            for row in manual.get_favorite_faces():
                file_row = db.get_file_by_checksum(row['checksum'])
                if file_row is None:
                    continue
                favorite_faces.append({
                    'ref': f"manual:{row['id']}",
                    'file_id': file_row['id'],
                    'identity': row['identity'],
                })
        faces = []
        return templates.TemplateResponse(request, 'faces.html', {
            'faces': faces,
            'identities': identities,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'favorite_only': favorite,
            'favorite_faces': favorite_faces,
        })

    @app.get('/face-crop/{face_id}')
    def serve_face_crop(face_id: str):
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)
            bbox_json = json.dumps([row['x1'], row['y1'], row['x2'], row['y2']])
            file_row = db.get_file_by_checksum(row['checksum'])
        else:
            cursor = db.conn.cursor()
            cursor.execute('SELECT file_id, bbox FROM faces WHERE id = ?', (raw_id,))
            row = cursor.fetchone()
            if row is None:
                return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)
            file_id, bbox_json = row[0], row[1]
            file_row = db.get_file_by_id(file_id)

        if file_row is None:
            return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)

        src_path = os.path.join(data_root, file_row['path'])
        crop_path = os.path.join(thumbs_dir, f'face_{kind}_{raw_id}.jpg')
        if not os.path.isfile(crop_path):
            if not _make_face_crop(src_path, bbox_json, crop_path):
                return Response(content=_gray_placeholder(), media_type='image/jpeg')
        return FileResponse(crop_path, media_type='image/jpeg')

    @app.get('/api/faces')
    def api_list_faces():
        unidentified = _unpromoted_auto_faces(limit=200)
        identities = manual.get_all_identities()
        suggestions = _suggest_names(unidentified)
        return {
            'faces': [
                {
                    'id': f"auto:{r[0]}", 'file_id': r[1], 'path': r[2], 'bbox': r[3],
                    'suggested_name': suggestions.get(r[0], {}).get('name'),
                    'suggested_score': suggestions.get(r[0], {}).get('score'),
                }
                for r in unidentified
            ],
            'identities': [{'name': r[0], 'count': r[1]} for r in identities],
        }

    @app.get('/api/identities')
    def api_list_identities():
        face_ids = manual.get_representative_face_ids()
        return [
            {'name': r[0], 'count': r[1], 'face_id': face_ids.get(r[0])}
            for r in manual.get_all_identities()
        ]

    @app.put('/api/identities/{name}')
    def api_rename_identity(name: str, body: RenameIdentityBody):
        """Renames a person everywhere — every face tagged with this identity, not
        just one — since a name typo/correction should apply to the whole person."""
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail='Name must not be empty')
        manual.rename_identity(name, new_name)
        return {'name': new_name}

    @app.post('/api/identities/{name}/unassign')
    def api_unassign_identity(name: str, body: UnassignIdentityBody):
        """Bulk 'make unknown again' — clears this identity from every named
        face on each given photo (not rejected, just back to unidentified),
        for the person-search page's select-many + shift-click-range bulk
        action. A photo not actually carrying this identity is silently
        skipped rather than erroring, so a mixed selection doesn't need to be
        filtered client-side first."""
        unassigned = 0
        for file_id in body.file_ids:
            row = db.get_file_by_id(file_id)
            if row is None:
                continue
            unassigned += manual.unassign_identity_for_checksum(row['checksum'], name)
        return {'unassigned': unassigned}

    @app.post('/api/identities/{name}/assign-set')
    def api_link_identity_to_set(name: str, body: AssignIdentitySetBody):
        """'This person appears in this set' — one link row, not one per member
        photo. Which photos that implies is resolved dynamically wherever "files
        for this person" gets built (see manual.get_sets_linked_to_identity's
        callers), so a photo added to this set later counts too, with nothing
        further to write here."""
        set_row = manual.get_set(body.set_id)
        if set_row is None:
            raise HTTPException(status_code=404, detail='Set not found')
        manual.link_identity_to_set(body.set_id, name)
        return {'linked': True}

    @app.delete('/api/identities/{name}/assign-set/{set_id}')
    def api_unlink_identity_from_set(name: str, set_id: int):
        manual.unlink_identity_from_set(set_id, name)
        return {'linked': False}

    @app.post('/api/files/{file_id}/identity-assignments')
    def api_assign_identity_to_file(file_id: int, body: IdentityBody):
        """Manual, whole-photo 'this person is in this picture' — for a photo
        where no individual face crop exists to name (missed detection, face
        turned away, etc.). See manual_db.assign_identity_to_photo. Unlike the
        face-confirm endpoint below, a name is still required here — there's no
        face match driving this, so nothing to anchor an auto-generated
        placeholder to."""
        row = _file_or_404(file_id)
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Name must not be empty')
        manual.assign_identity_to_photo(row['checksum'], name)
        return {'ok': True}

    @app.delete('/api/files/{file_id}/identity-assignments/{name}')
    def api_remove_identity_assignment(file_id: int, name: str):
        row = _file_or_404(file_id)
        manual.remove_identity_photo_assignment(row['checksum'], name)
        return {'ok': True}

    @app.get('/api/identities/{name}/similar-faces')
    def api_similar_unknown_faces(name: str, threshold: float = SUGGEST_THRESHOLD, limit: int = 20):
        threshold = max(0.0, min(1.0, threshold))
        return {'results': _find_similar_unknown_faces(name, threshold, limit=limit)}

    @app.get('/api/faces/{face_id}/suggestions')
    def api_face_suggestions(face_id: str):
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            emb_bytes = row['embedding'] if row is not None else None
        else:
            cursor = db.conn.cursor()
            cursor.execute('SELECT embedding FROM faces WHERE id = ?', (raw_id,))
            row = cursor.fetchone()
            emb_bytes = row[0] if row is not None else None
        if emb_bytes is None:
            return {'suggestions': []}
        named = manual.get_named_face_embeddings()
        if not named:
            return {'suggestions': []}
        import numpy as np
        query = np.frombuffer(emb_bytes, dtype=np.float32)
        names = [n for n, _ in named]
        matrix = np.stack([np.frombuffer(e, dtype=np.float32) for _, e in named])
        scores = matrix.dot(query)
        ranked = sorted(zip(names, scores.tolist()), key=lambda x: x[1], reverse=True)
        seen = set()
        out = []
        for name, score in ranked:
            if name in seen or score < 0.3:
                continue
            seen.add(name)
            out.append({'name': name, 'score': round(float(score), 3)})
            if len(out) >= 5:
                break
        return {'suggestions': out}

    # Cosine similarity above this, another face is considered a likely match for
    # "find similar faces" — same threshold as the named-person suggestion search,
    # since it's the same embedding space and comparable false-positive tolerance.
    FACE_SIMILAR_THRESHOLD = 0.45

    @app.get('/api/faces/{face_id}/similar')
    def api_similar_faces(face_id: str, threshold: float = FACE_SIMILAR_THRESHOLD, limit: int = 20):
        """Find other faces that look like this one — named or not. Unlike
        suggestions (which only ever proposes an existing person's name), this works
        even when the face itself has no name yet, so you can spot every other photo
        of the same person before deciding what to call them."""
        threshold = max(0.0, min(1.0, threshold))
        return {'results': _find_similar_faces_to_ref(face_id, threshold, limit=limit)}

    # ------------------------------------------------------------------
    # Tinder-style face-suggestion card stack (swipe to confirm/reject)
    # ------------------------------------------------------------------

    def _next_face_suggestions(count, exclude_refs, bias_identity=None, bias=None, identity_filter=None, avoid_existing=True):
        """Rank unpromoted auto-detected faces against every known identity (same
        embedding math as _suggest_names) and return up to `count` candidates that
        clear SUGGEST_THRESHOLD, best score first. `exclude_refs` filters out faces
        already sitting in the client's buffer (not yet swiped) so a refill never
        hands back a duplicate card. `bias_identity`/`bias` optionally reorder the
        result: after a same-face confirm we want more of that same person surfaced
        next; after a reject we want that identity pushed to the back rather than
        immediately re-suggested — a simple sort tweak, not a new ranking model.

        `identity_filter`, when set, scopes the whole comparison to one identity's
        own confirmed embeddings (same accessor _find_similar_unknown_faces uses)
        instead of arguing over every named person — every returned card then
        already carries that one identity, which makes bias_identity/bias
        naturally inert here (the reorder partition never has anything to
        partition against); that's expected, not a bug, so callers in this mode
        simply don't bother passing bias params.

        `avoid_existing` (default True — a UI toggle, on by default): faces
        whose photo already has at least one identified person are
        deprioritized (see _deprioritize_files_with_named_face) rather than
        excluded, so already-covered photos don't crowd out fresh ones but
        can still surface if nothing else clears the threshold."""
        if identity_filter is not None:
            ref_embeddings = manual.get_embeddings_for_identity(identity_filter)
            if not ref_embeddings:
                return []
            import numpy as np
            matrix = np.stack([np.frombuffer(e, dtype=np.float32) for e in ref_embeddings])

            candidates = []  # (score, ref, file_id, identity)
            for face_id_, file_id_, _path, _bbox, emb_bytes in _unpromoted_auto_faces(limit=None):
                ref = f"auto:{face_id_}"
                if ref in exclude_refs or not emb_bytes:
                    continue
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                score = float(matrix.dot(vec).max())
                if score >= SUGGEST_THRESHOLD:
                    candidates.append((score, ref, file_id_, identity_filter))

            from media_manager.swipe_support import bias_reorder
            candidates.sort(key=lambda c: -c[0])
            if avoid_existing:
                candidates = _deprioritize_files_with_named_face(candidates, 2)
            candidates = bias_reorder(candidates, key_fn=lambda c: c[3], bias_key=bias_identity, bias_action=bias)
            return _attach_file_meta([
                {'ref': ref, 'file_id': file_id_, 'identity': identity, 'score': round(score, 3)}
                for score, ref, file_id_, identity in candidates[:count]
            ])

        named = manual.get_named_face_embeddings()
        if not named:
            return []
        import numpy as np
        names = [n for n, _ in named]
        matrix = np.stack([np.frombuffer(e, dtype=np.float32) for _, e in named])

        candidates = []  # (score, ref, file_id, identity)
        for face_id_, file_id_, _path, _bbox, emb_bytes in _unpromoted_auto_faces(limit=None):
            ref = f"auto:{face_id_}"
            if ref in exclude_refs or not emb_bytes:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            scores = matrix.dot(vec)
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            if best_score >= SUGGEST_THRESHOLD:
                candidates.append((best_score, ref, file_id_, names[best_idx]))

        from media_manager.swipe_support import bias_reorder
        candidates.sort(key=lambda c: -c[0])
        if avoid_existing:
            candidates = _deprioritize_files_with_named_face(candidates, 2)
        candidates = bias_reorder(candidates, key_fn=lambda c: c[3], bias_key=bias_identity, bias_action=bias)
        return _attach_file_meta([
            {'ref': ref, 'file_id': file_id_, 'identity': identity, 'score': round(score, 3)}
            for score, ref, file_id_, identity in candidates[:count]
        ])

    @app.get('/swipe')
    def swipe_page_redirect():
        """/faces absorbed the face-swipe stack — redirect old bookmarks/links
        rather than 404ing them."""
        return RedirectResponse(url='/faces')

    @app.post('/api/face-suggestions/next')
    def api_next_face_suggestions(body: SwipeExcludeBody, count: int = 10, bias_identity: str = '', bias: str = '',
                                   identity: str = '', avoid_existing: bool = True):
        """Feeds the swipe-card stack's background buffer. `body.exclude` is the
        full list of face refs the client already holds (queued or just-decided)
        so a refill never repeats a card still in flight client-side. `identity`
        (distinct from `bias_identity`) scopes the stream to one person's own
        confirmed faces, for the person-search page's swipe stack."""
        exclude_refs = set(body.exclude)
        cards = _next_face_suggestions(count, exclude_refs, bias_identity or None, bias or None,
                                        identity_filter=identity or None, avoid_existing=avoid_existing)
        return {'cards': cards}

    @app.post('/api/faces/{face_id}/identity')
    def api_assign_identity(face_id: str, body: IdentityBody):
        # A blank/omitted name confirms "this is a distinct person" without
        # requiring one to be typed — see manual.generate_placeholder_identity_name.
        # A placeholder behaves exactly like a real name everywhere else (rename,
        # suggestion matching, /search?person=, ...), it's just waiting to be
        # renamed, or not.
        name = (body.name or '').strip() or manual.generate_placeholder_identity_name()
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                raise HTTPException(status_code=404, detail='Face not found')
            manual.assign_identity(raw_id, name)
            return {'face_id': face_id, 'identity': name}

        cursor = db.conn.cursor()
        cursor.execute('SELECT file_id, bbox, embedding FROM faces WHERE id = ?', (raw_id,))
        src = cursor.fetchone()
        if src is None:
            raise HTTPException(status_code=404, detail='Face not found')
        src_file = db.get_file_by_id(src['file_id'])
        if src_file is None:
            raise HTTPException(status_code=404, detail='File not found')
        bbox = json.loads(src['bbox'])
        new_id = manual.promote_auto_face(raw_id, src_file['checksum'], bbox, src['embedding'], name, None, None)
        return {'face_id': f"manual:{new_id}", 'identity': name}

    @app.post('/api/faces/{face_id}/reject')
    def api_reject_face(face_id: str):
        """'This is not a face' / 'remove this face'. Kept as a row in manual.db
        (rejected=1) rather than deleted, whether it started out auto-detected or
        hand-drawn — a confirmed rejection is itself useful training signal, and it
        keeps a rejected auto-detection from ever reappearing as 'unidentified'."""
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                raise HTTPException(status_code=404, detail='Face not found')
            manual.reject_face(raw_id)
            return {'ok': True}

        cursor = db.conn.cursor()
        cursor.execute('SELECT file_id, bbox, embedding FROM faces WHERE id = ?', (raw_id,))
        src = cursor.fetchone()
        if src is None:
            raise HTTPException(status_code=404, detail='Face not found')
        src_file = db.get_file_by_id(src['file_id'])
        if src_file is None:
            raise HTTPException(status_code=404, detail='File not found')
        bbox = json.loads(src['bbox'])
        manual.reject_auto_face(raw_id, src_file['checksum'], bbox, src['embedding'], None, None)
        return {'ok': True}

    @app.delete('/api/faces/{face_id}/decision')
    def api_undo_face_decision(face_id: str):
        """Undo a confirm or reject made via the face-suggestion swipe stream —
        hard-deletes the manual.db row so the face returns to fully unhandled,
        rather than reject_face's usual kept-as-hard-negative row. Only
        meaningful for auto-sourced refs (the swipe stream never offers
        already-manual faces)."""
        kind, raw_id = _parse_face_ref(face_id)
        if kind != 'auto':
            raise HTTPException(status_code=400, detail='Only auto-detected faces have a decision to undo')
        manual.delete_face_decision_by_source(raw_id)
        return {'ok': True}

    @app.post('/api/faces/{face_id}/favorite')
    def api_set_face_favorite(face_id: str, body: FavoriteBody):
        """Only named/manual faces can be favorited — an auto-detected face has no
        manual.db row to attach the flag to until it's named."""
        kind, raw_id = _parse_face_ref(face_id)
        if kind != 'manual':
            raise HTTPException(status_code=400, detail='Only named faces can be favorited')
        row = manual.get_face(raw_id)
        if row is None:
            raise HTTPException(status_code=404, detail='Face not found')
        manual.set_face_favorite(raw_id, body.favorite)
        return {'face_id': face_id, 'favorite': body.favorite}

    @app.post('/api/files/{file_id}/faces')
    def api_add_manual_face(file_id: int, body: ManualFaceBody):
        row = _file_or_404(file_id)
        if len(body.bbox) != 4:
            raise HTTPException(status_code=400, detail='bbox must be [x1,y1,x2,y2]')
        x1, y1, x2, y2 = body.bbox
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
            raise HTTPException(status_code=400, detail='Invalid or too-small bbox')

        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        import cv2
        img = cv2.imread(abs_path)
        if img is None:
            raise HTTPException(status_code=400, detail='Could not read image')
        img_height, img_width = img.shape[:2]

        result = _get_face_detector().embed_bbox(img, [x1, y1, x2, y2])
        final_bbox = result['bbox'] if result['bbox'] is not None else [x1, y1, x2, y2]
        face_id = manual.add_manual_face(row['checksum'], final_bbox, result['embedding'].tobytes(), img_width, img_height)
        return {'id': f"manual:{face_id}", 'bbox': final_bbox, 'det_score': result['det_score']}

    @app.post('/api/files/{file_id}/detect-faces')
    def api_detect_faces(file_id: int):
        """Run InsightFace on just this one image (the GUI equivalent of the batch
        `media faces` CLI command), then auto-promote any high-confidence match against
        an existing identity — see manual_db.AUTO_MATCH_THRESHOLD."""
        row = _file_or_404(file_id)
        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        detector = _get_face_detector()
        results = detector.detect_faces([abs_path])
        _, faces, error = results[0]
        if error:
            raise HTTPException(status_code=500, detail=f'Face detection failed: {error}')

        db.insert_faces(file_id, faces, detector.model_id(detector._model_name))

        auto_matched = []
        for face_row in db.get_faces_for_file(file_id):
            face_db_id = face_row['id']
            emb_bytes = db.get_face_embedding(face_db_id)
            if not emb_bytes:
                continue
            name, score = manual.find_matching_identity(emb_bytes)
            if name is None:
                continue
            bbox = json.loads(face_row['bbox'])
            manual.promote_auto_face(face_db_id, row['checksum'], bbox, emb_bytes, name, None, None)
            auto_matched.append({'name': name, 'score': round(score, 3)})

        return {'faces_found': len(faces), 'auto_matched': auto_matched}

    # Safety cap on a single "scan all frames" job — a pathologically long GIF
    # shouldn't be able to run forever. This is a deliberate, manual, single-image
    # action (see frames.py / the plan doc), not a batch job, so there's no need to
    # support arbitrarily large animations here. Raised from an initial 300 after
    # hitting a real 692-frame file — still a backstop against a truly pathological
    # (e.g. tens-of-thousands-of-frames) file, not a limit meant to bite on normal
    # library content.
    MAX_SCAN_FRAMES = 3000

    # In-memory progress for running/finished scan-all-frames jobs, keyed by file_id.
    # Ephemeral by design (lost on server restart) — this is UI progress feedback,
    # not data; the actual results are already durably written to the DB per frame
    # as the job runs, same as any other write in this app.
    _frame_scan_jobs = {}

    def _run_frame_scan_job(file_id, checksum, abs_path, frame_count):
        import tempfile
        job = _frame_scan_jobs[file_id]
        face_detector = _get_face_detector()
        object_detector = _get_object_detector()
        clip_indexer = _get_clip_indexer()
        model_objects = object_detector.model_id()
        model_clip = clip_indexer.model_id()

        try:
            for idx in range(1, frame_count):  # frame 0 handled by the normal pipeline
                img = frames.extract_frame(abs_path, idx)
                if img is None:
                    job['frames_processed'] += 1
                    continue
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img.save(tmp, format='JPEG', quality=90)
                    tmp_path = tmp.name
                try:
                    _, frame_faces, face_error = face_detector.detect_faces([tmp_path])[0]
                    if not face_error:
                        for f in frame_faces:
                            emb_bytes = f['embedding'].tobytes()
                            face_db_id = db.add_manual_face(file_id, f['bbox'], emb_bytes, f['det_score'], frame_index=idx)
                            job['faces_found'] += 1
                            name, score = manual.find_matching_identity(emb_bytes)
                            if name is not None:
                                manual.promote_auto_face(
                                    face_db_id, checksum, f['bbox'], emb_bytes, name, None, None, frame_index=idx
                                )
                                job['auto_matched'].append({'name': name, 'score': round(score, 3), 'frame_index': idx})

                    _, frame_detections, det_error = object_detector.detect_images([tmp_path])[0]
                    if not det_error:
                        db.insert_detections(file_id, frame_detections, model_objects, frame_index=idx)
                        job['objects_found'] += len(frame_detections)

                    embeddings, embed_failed = clip_indexer.embed_images([tmp_path])
                    if not embed_failed and len(embeddings) > 0:
                        db.insert_embedding(file_id, embeddings[0].tobytes(), model_clip, frame_index=idx)
                finally:
                    os.unlink(tmp_path)
                    job['frames_processed'] += 1
        except Exception as exc:
            job['error'] = str(exc)
        finally:
            job['done'] = True

    @app.post('/api/files/{file_id}/scan-all-frames')
    def api_scan_all_frames(file_id: int):
        """Kick off face + object detection + CLIP embedding across every frame of
        an animated file (frame 0 is already covered by the normal single-frame
        pipeline), running in a background thread so the request returns
        immediately — poll GET .../scan-all-frames/progress for status. Deliberately
        manual/opt-in and scoped to one image — see the plan doc for why automatic
        multi-frame indexing isn't done for the whole library."""
        import threading
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        existing = _frame_scan_jobs.get(file_id)
        if existing is not None and not existing['done']:
            raise HTTPException(status_code=409, detail='A scan is already running for this file')

        try:
            frame_count = frames.get_frame_count(abs_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Could not read file: {exc}')

        if frame_count <= 1:
            raise HTTPException(status_code=400, detail='This file has only one frame — nothing to scan')
        if frame_count > MAX_SCAN_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=f'{frame_count} frames exceeds the {MAX_SCAN_FRAMES}-frame safety cap',
            )

        _frame_scan_jobs[file_id] = {
            'frame_count': frame_count,
            'frames_processed': 0,
            'faces_found': 0,
            'objects_found': 0,
            'auto_matched': [],
            'done': False,
            'error': None,
        }
        thread = threading.Thread(
            target=_run_frame_scan_job,
            args=(file_id, row['checksum'], abs_path, frame_count),
            daemon=True,
        )
        thread.start()
        return {'started': True, 'frame_count': frame_count}

    @app.get('/api/files/{file_id}/scan-all-frames/progress')
    def api_scan_all_frames_progress(file_id: int):
        job = _frame_scan_jobs.get(file_id)
        if job is None:
            raise HTTPException(status_code=404, detail='No scan job found for this file')
        return job

    @app.post('/api/search/face')
    async def api_search_by_face(file: UploadFile = File(...)):
        import tempfile
        from media_manager.media_manager import MediaManager
        contents = await file.read()
        suffix = os.path.splitext(file.filename or '.jpg')[1] or '.jpg'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            detector = _get_face_detector()
            results = detector.detect_faces([tmp_path])
            _, faces, error = results[0]
            if error or not faces:
                return {'results': [], 'message': 'No face detected in uploaded image.'}

            import numpy as np
            query_face = max(faces, key=lambda f: f['det_score'])
            query_emb = query_face['embedding']

            promoted_ids = manual.get_promoted_source_ids()
            combined = []  # (ref, file_id, path, embedding_bytes, identity)

            for fid, file_id, path, emb in db.get_all_face_embeddings():
                if fid in promoted_ids:
                    continue
                combined.append((f"auto:{fid}", file_id, path, emb, None))

            manual_checksums = [r[1] for r in manual.get_all_faces_with_embedding()]
            file_rows = {r['checksum']: (r['id'], r['path']) for r in db.get_files_by_checksums(manual_checksums)}
            for mid, checksum, identity, emb in manual.get_all_faces_with_embedding():
                resolved = file_rows.get(checksum)
                if resolved is None:
                    continue
                file_id, path = resolved
                combined.append((f"manual:{mid}", file_id, path, emb, identity))

            if not combined:
                return {'results': [], 'message': 'No faces indexed yet — run <code>media faces</code> first.'}

            matrix = np.stack([np.frombuffer(c[3], dtype=np.float32) for c in combined])
            scores = matrix.dot(query_emb).tolist()

            ranked = sorted(
                zip(combined, scores),
                key=lambda x: x[1], reverse=True,
            )

            seen = {}
            results_out = []
            for (ref, file_id, fpath, _emb, identity), score in ranked:
                if score < 0.3:
                    break
                if file_id not in seen:
                    seen[file_id] = True
                    results_out.append({
                        'file_id': file_id,
                        'path': fpath,
                        'filename': os.path.basename(fpath),
                        'score': round(score, 4),
                        'face_id': ref,
                        'identity': identity,
                    })
                if len(results_out) >= 50:
                    break
            return {'results': results_out}
        finally:
            os.unlink(tmp_path)

    @app.get('/api/errors')
    def api_list_errors(unread_only: bool = True, limit: int = 50):
        rows = errors.list_errors(unread_only=unread_only, limit=limit)
        return {
            'unread_count': errors.count_unread(),
            'items': [{'id': r['id'], 'path': r['path'], 'message': r['message'],
                       'read': bool(r['read']), 'created_at': r['created_at']} for r in rows],
        }

    @app.post('/api/errors/{error_id}/read')
    def api_mark_error_read(error_id: int):
        errors.mark_read(error_id)
        return {'ok': True}

    @app.post('/api/errors/read-all')
    def api_mark_all_errors_read():
        errors.mark_all_read()
        return {'ok': True}

    @app.on_event('shutdown')
    def _close_databases():
        db.close()
        errors.close()
        manual.close()

    return app
