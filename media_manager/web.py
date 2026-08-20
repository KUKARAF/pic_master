"""
FastAPI web server for media_manager — gallery UI with CLIP search, tags, and similar-image discovery.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import quote
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from media_manager import frames, worker_client, worker_config
from media_manager.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, VIDEO_MIME_TYPES
from media_manager.category_resolver import (
    get_category_counts,
    get_resolved_checksums_for_category,
    resolve_categories_for_checksums,
    resolve_categories_for_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THUMB_SIZE = (400, 400)

# Applied to /thumb, /image, /face-crop, /body-crop's successful responses only
# (never the gray-placeholder fallbacks) — each of those is served from a path
# that's content-addressed by file_id/face_id/body_id and never mutated in
# place once written (a re-processed/edited photo gets a new checksum and thus
# a new file_id, never reuses the old id's cached thumbnail/crop), so it's safe
# to tell the browser to never re-request it.
IMMUTABLE_CACHE_HEADERS = {'Cache-Control': 'public, max-age=31536000, immutable'}

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
    confirm_merge: bool = False
    confirm_studio_merge: bool = False

class AddFolderBody(BaseModel):
    path: str

class ReorderBody(BaseModel):
    file_ids: List[int]

class IdentityBody(BaseModel):
    name: Optional[str] = None

class ConfirmAboveThresholdBody(BaseModel):
    threshold: float

class ManualFaceBody(BaseModel):
    bbox: List[float]

class SpatialTagBody(BaseModel):
    label: str
    bbox: List[float]
    polarity: str = 'positive'

class FavoriteBody(BaseModel):
    # Favorites are a counter now: the client sends a signed delta (+1 on
    # left-click, -1 on right-click) and the endpoint returns the new count.
    delta: int = 1

class TagLabelBody(BaseModel):
    label: str

class RenameIdentityBody(BaseModel):
    name: str
    confirm_merge: bool = False

class StudioRenameBody(BaseModel):
    name: str

class IdentityAliasBody(BaseModel):
    alias: str

class SetAliasBody(BaseModel):
    alias: str

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

class LocationBody(BaseModel):
    # Mirrors CategoryBody. gps_lat/gps_lon are optional and may be explicitly
    # null to CLEAR a location's stored coordinates on PUT — see api_update_location.
    name: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

class LocationAssignBody(BaseModel):
    location_id: int

class CaptureFrameIndexBody(BaseModel):
    frame_index: int

class SearchChip(BaseModel):
    type: str
    value: str

class SearchPaletteBody(BaseModel):
    q: str = ''
    chips: List[SearchChip] = []
    sort: str = ''      # '', 'added', 'modified', 'filename', 'age', 'favorites'
    order: str = 'desc'

class FileMetaBody(BaseModel):
    """Body for POST /api/files/meta — a list of file ids the client wants
    display metadata (name + set names) for. Used by the Space photo-grid
    overlay, which only carries file ids and fetches captions in one batch."""
    ids: List[int] = []

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

class RegionSearchBody(BaseModel):
    bbox: list[float]


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


def _make_video_thumbnail(src_path: str, dst_path: str) -> tuple:
    """Poster-frame thumbnail for a video: grab a frame with OpenCV (the same
    decoder broken_finder.verify_video uses), then reuse the image thumbnail's
    resize-to-400px + atomic-JPEG-save tail. Seeks ~1s in so the poster isn't a
    black/leading frame, falling back to frame 0. Returns (success, message) with
    the same shape as _make_thumbnail so serve_thumb treats them identically."""
    import cv2
    from PIL import Image as PILImage
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return False, 'could not open video'
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Seeking past a very short clip's end leaves nothing to read —
                # rewind and take the first frame instead.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                return False, 'could not read a video frame'
        finally:
            cap.release()
        img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = img.size
        if w == 0:
            return False, 'video frame has zero width'
        new_w = 400
        new_h = max(1, int(h * new_w / w))
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        _save_jpeg_atomic(img, dst_path, quality=80)
        return True, None
    except Exception as exc:
        return False, str(exc)


# Remote-worker offload. `_worker_client` is set by create_app() to the
# process-global WorkerClient for this data_root; it stays None on code paths that
# never build an app (getters then behave exactly as before — local only).
#
# Choice is by is_configured() (cheap: worker enabled + address, NO network probe),
# NOT is_available(): when a worker is CONFIGURED the web process ALWAYS uses the
# remote proxy and NEVER builds a local model — even if the worker is momentarily
# down (the proxy retries and then raises, which the caller surfaces). Building a
# local model here is exactly what OOMs a memory-constrained web process. Only with
# no worker configured does a getter build the real local model.
_worker_client = None

_face_detector_local = None
_face_detector_remote = None


def _get_face_detector():
    global _face_detector_local, _face_detector_remote
    if _worker_client is not None and _worker_client.is_configured():
        if _face_detector_remote is None:
            _face_detector_remote = worker_client.RemoteFaceDetector(_worker_client)
        return _face_detector_remote
    if _face_detector_local is None:
        from media_manager.face_detector import FaceDetector
        _face_detector_local = FaceDetector()
    return _face_detector_local


_object_detector_local = None
_object_detector_remote = None


def _get_object_detector():
    # Local YOLOWorldDetector() defaults its vocab to detector.DEFAULT_VOCAB; build
    # the remote proxy with the SAME default so it is never left with vocab=None.
    # Callers that need a different vocab (reindex, body_index person-only pass) call
    # .set_vocab(...) after getting it — the proxy stores + forwards that, same as
    # the local model.
    global _object_detector_local, _object_detector_remote
    if _worker_client is not None and _worker_client.is_configured():
        if _object_detector_remote is None:
            from media_manager.detector import DEFAULT_VOCAB
            _object_detector_remote = worker_client.RemoteYOLODetector(
                _worker_client, vocab=list(DEFAULT_VOCAB))
        return _object_detector_remote
    if _object_detector_local is None:
        from media_manager.detector import YOLOWorldDetector
        _object_detector_local = YOLOWorldDetector()
    return _object_detector_local


_clip_indexer_local = None
_clip_indexer_remote = None


def _get_clip_indexer():
    global _clip_indexer_local, _clip_indexer_remote
    if _worker_client is not None and _worker_client.is_configured():
        if _clip_indexer_remote is None:
            _clip_indexer_remote = worker_client.RemoteCLIPIndexer(_worker_client)
        return _clip_indexer_remote
    if _clip_indexer_local is None:
        from media_manager.indexer import CLIPIndexer
        _clip_indexer_local = CLIPIndexer()
    return _clip_indexer_local


_age_estimator = None
_age_estimator_remote = None


def _get_age_estimator():
    """Experimental — see age_estimator.py. Runs in a separate, isolated venv (MiVOLO
    pins an old timm/ultralytics that conflict with this app's own detector/indexer),
    so this lazy singleton is a thin subprocess client, not a loaded model. When a
    worker is configured, offload it too (the worker runs MiVOLO in ITS .age-venv) —
    same is_configured() gating as the other model getters, worker-only with no local
    fallback so the low-RAM host never runs MiVOLO."""
    global _age_estimator, _age_estimator_remote
    if _worker_client is not None and _worker_client.is_configured():
        if _age_estimator_remote is None:
            _age_estimator_remote = worker_client.RemoteAgeEstimator(_worker_client)
        return _age_estimator_remote
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


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points
    (standard haversine formula). Used by the /search distance sort to rank
    photos by how close their EXIF GPS is to a target point."""
    import math
    r = 6371.0  # Earth mean radius, km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(data_root: str) -> FastAPI:
    """Create and return the FastAPI application rooted at data_root."""
    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')

    # Wire the process-global worker client for this data_root so the model getters
    # can offload to a remote worker when one is configured + reachable. Cheap: no
    # RNS init and no network until is_available() is first called (and that pays
    # nothing when no worker is configured).
    global _worker_client
    _worker_client = worker_client.get_client(data_root)

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

    # Every route on this app is hit by fetch()-based JS that always does
    # `await res.json()` on the response, success or failure (see static/app.js).
    # Without this handler, an unhandled exception (e.g. a face-detection model
    # failing to load, a corrupt onnxruntime cache, ...) falls through to
    # Starlette's default plain-text "Internal Server Error" body, which isn't
    # JSON — the frontend's .json() call then throws its own unrelated
    # "JSON.parse: unexpected character..." error, masking the real failure.
    # This does not swallow anything: the exception is still logged (with full
    # traceback) and the response is still a 500 — just JSON instead of text/plain,
    # so callers see the actual error message.
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'detail': f'{exc.__class__.__name__}: {exc}' if str(exc) else exc.__class__.__name__},
        )

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
                'favorite': r['favorite']} for r in manual_rows]
        for r in db.get_faces_for_file(file_id):
            if r['id'] in promoted:
                continue
            out.append({'ref': f"auto:{r['id']}", 'bbox': _json.loads(r['bbox']),
                        'identity': None, 'frame_index': r['frame_index'], 'favorite': 0})
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
            'favorite': manual.get_file_favorite_count(row['checksum']),
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
        favorite_counts = manual.get_favorite_counts(checksums)
        title_map = manual.get_titles_for_checksums(checksums)
        sets_map = manual.get_sets_for_checksums(checksums)
        _attach_set_people(sets_map.values())
        identities_map = _people_with_ages(manual.get_identities_for_checksums(checksums))
        category_map = resolve_categories_for_checksums(manual, db, [(row[0], row[3]) for row in rows])
        result = []
        for row in rows:
            file_id = row[0]
            path = row[1]
            has_embedding = row[2] if len(row) > 2 else 0
            checksum = row[3]
            card_sets = sets_map.get(checksum, [])
            card = {
                'id': file_id,
                'path': path,
                'checksum': checksum,
                'filename': os.path.basename(path),
                'is_video': os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS,
                'has_embedding': bool(has_embedding),
                'tags': tag_map.get(checksum, []),
                'favorite': favorite_counts.get(checksum, 0),
                'title': title_map.get(checksum),
                'sets': card_sets,
                'people': _people_not_in_sets(identities_map.get(checksum, []), card_sets),
                'categories': category_map.get(checksum, []),
            }
            if scores is not None and file_id in scores:
                card['score'] = scores[file_id]
            result.append(card)
        return result

    def _people_not_in_sets(people, card_sets):
        """Drop anyone from a card's own 'recognized in this photo' people list
        who's already shown via one of the card's set chips (chip-set already
        renders every one of a set's people with an '@' prefix — see
        set_meta_line/_attach_set_people's get_all_people_for_sets). Without
        this, someone linked to (or detected in) a set this photo belongs to
        shows up twice on the same card: once as '@Name' inside the chip-set,
        once again as a bare 'Name' chip-person — same person, and often a
        different age since the two used to come from different queries."""
        if not card_sets:
            return people
        set_people_names = {p['name'] for s in card_sets for p in s.get('people', [])}
        return [p for p in people if p['name'] not in set_people_names]

    def _people_with_ages(identities_map):
        """Reshape a {checksum: [name, ...]} map (manual.get_identities_for_checksums'
        return shape) into {checksum: [{'name','age','gender'}, ...]} — one batched
        get_ages_for_identities_by_checksum call across every checksum involved.
        Age/gender here are THIS SPECIFIC PHOTO's own estimate for that person
        (same semantics as photo_page's own get_age_estimates_for_checksum), not a
        global "most recently estimated anywhere" value — so a card's people chip
        always agrees with what you'd see if you clicked into that exact photo.
        None when that identity has no estimate on this specific photo yet."""
        per_checksum_ages = manual.get_ages_for_identities_by_checksum(list(identities_map.keys()))
        return {
            checksum: [
                {
                    'name': name,
                    'age': per_checksum_ages.get(checksum, {}).get(name, {}).get('age'),
                    'gender': per_checksum_ages.get(checksum, {}).get(name, {}).get('gender'),
                }
                for name in names
            ]
            for checksum, names in identities_map.items()
        }

    def _attach_set_people(sets_lists):
        """Given an iterable of set-dict lists (e.g. a sets_map's values()), attaches
        a 'people' key to each set dict in place — one {'name', 'age', 'gender'}
        entry per identity actually present in that set: linked via
        link_identity_to_set, OR recognized via a real face or whole-photo
        assignment on one of the set's own photos (see get_all_people_for_sets) —
        a set's roster always shows everyone in it, not just whoever happened to
        be explicitly linked, matching the same "everyone" list the set's own
        People-present strip already used. Age/gender are that identity's average
        across only THIS set's own member photos (see
        get_average_ages_for_identities_in_sets) — not a global "most recently
        estimated anywhere" value — so a set's roster always reflects the people
        actually in that set, not some unrelated photo of the same person
        elsewhere in the library. None when that identity has no estimate within
        this set yet. Batches the set->identity lookup and the per-set age
        lookup across every set involved instead of querying per set (mirrors
        the batching _enrich_rows already does for tags/sets/people)."""
        all_sets = [s for lst in sets_lists for s in lst]
        set_ids = {s['id'] for s in all_sets}
        people_map = manual.get_all_people_for_sets(set_ids)
        ages_map = manual.get_average_ages_for_identities_in_sets(set_ids)
        for s in all_sets:
            s['people'] = [
                {'name': name, 'age': ages_map.get((s['id'], name), {}).get('age'), 'gender': ages_map.get((s['id'], name), {}).get('gender')}
                for name in people_map.get(s['id'], [])
            ]

    def _attach_set_people_names_only(sets_lists):
        """Cheaper sibling of _attach_set_people: same roster (get_all_people_for_sets),
        but skips get_average_ages_for_identities_in_sets entirely — every
        person gets age/gender None. For callers that only need to know WHO's
        in a set (e.g. the set-search picker's personAware "type a name to
        narrow the list" filter), not their age, this avoids the per-set
        face/estimate join that's real query cost with nothing to show for it
        in a text-only picker."""
        all_sets = [s for lst in sets_lists for s in lst]
        set_ids = {s['id'] for s in all_sets}
        people_map = manual.get_all_people_for_sets(set_ids)
        for s in all_sets:
            s['people'] = [{'name': name, 'age': None, 'gender': None} for name in people_map.get(s['id'], [])]

    def _attach_set_aliases(sets_lists):
        """Given an iterable of set-dict lists, attaches an 'aliases' key to each
        set dict in place — same batching shape as _attach_set_people, one query
        total across every set involved rather than one per set."""
        all_sets = [s for lst in sets_lists for s in lst]
        set_ids = {s['id'] for s in all_sets}
        aliases_map = manual.get_aliases_for_sets(set_ids)
        for s in all_sets:
            s['aliases'] = aliases_map.get(s['id'], [])

    def _favorites_highlight(limit=6):
        """Random mix of up to `limit` items drawn from favorited photos, favorited
        faces (people), and favorited sets — reshuffled on every call. Homepage
        'Favorites' teaser section. Each favorite kind is sampled with its own
        bounded/cheap query rather than the gallery's 'fetch everything then
        filter' pattern (which doesn't scale with library size)."""
        items = []

        photo_checksums = manual.get_random_favorite_checksums(limit=limit)
        if photo_checksums:
            rows = db.get_files_by_checksums(photo_checksums)
            if rows:
                cards = _enrich_rows([(r['id'], r['path'], 0, r['checksum']) for r in rows])
                items += [{'kind': 'photo', 'card': c} for c in cards]

        favorite_faces = manual.get_favorite_faces()
        if favorite_faces:
            sampled = random.sample(favorite_faces, min(limit, len(favorite_faces)))
            for row in sampled:
                file_row = db.get_file_by_checksum(row['checksum'])
                if file_row is None:
                    continue
                items.append({'kind': 'face', 'face': {
                    'ref': f"manual:{row['id']}",
                    'file_id': file_row['id'],
                    'identity': row['identity'],
                }})

        favorite_sets = manual.list_sets(favorite_only=True)
        if favorite_sets:
            sampled = random.sample(favorite_sets, min(limit, len(favorite_sets)))
            set_dicts = []
            for r in sampled:
                thumb_id = None
                first_checksums = manual.get_files_by_set(r['id'], limit=1)
                if first_checksums:
                    thumb_rows = db.get_files_by_checksums(first_checksums)
                    if thumb_rows:
                        thumb_id = thumb_rows[0]['id']
                set_dicts.append({
                    'id': r['id'], 'name': r['name'], 'studio': r['studio'],
                    'image_count': r['image_count'], 'favorite': True, 'thumb_id': thumb_id,
                })
            _attach_set_people([set_dicts])
            items += [{'kind': 'set', 'set': s} for s in set_dicts]

        random.shuffle(items)
        return items[:limit]

    def _needs_love_highlight(limit=6, candidate_cap=200, batch_size=50):
        """Up to `limit` photos with no manual tags, no category (manual or
        auto-matched), no set membership, AND at least one still-unidentified
        face on that specific photo — checked via cheap per-candidate queries
        against a random sample, not a full-library scan (mirrors the per-file
        'unidentified face' check _unpromoted_auto_faces does library-wide).
        Stops once `limit` qualify or `candidate_cap` candidates have been
        examined, whichever comes first — can legitimately return fewer than
        `limit` items, including zero, on a well-curated or barely-started
        library."""
        set_member_checksums = manual.get_all_set_member_checksums()
        promoted_ids = manual.get_promoted_source_ids()

        found = []
        examined = 0
        seen_ids = set()
        while examined < candidate_cap and len(found) < limit:
            batch = db.get_random_files(limit=batch_size)
            if not batch:
                break
            new_in_batch = 0
            for row in batch:
                file_id, path, checksum = row[0], row[1], row[2]
                if file_id in seen_ids:
                    continue
                seen_ids.add(file_id)
                new_in_batch += 1
                examined += 1
                if examined > candidate_cap:
                    break
                if checksum in set_member_checksums:
                    continue
                if manual.list_tags_for_checksums([checksum]).get(checksum):
                    continue
                if resolve_categories_for_checksums(manual, db, [(file_id, checksum)]).get(checksum):
                    continue
                unidentified = [f for f in db.get_faces_for_file(file_id) if f[0] not in promoted_ids]
                if not unidentified:
                    continue
                found.append((file_id, path, checksum))
                if len(found) >= limit:
                    break
            if new_in_batch == 0:
                # RANDOM() sampling has cycled back over an already-fully-seen
                # pool (the library has fewer distinct files than candidate_cap)
                # — stop instead of spinning forever re-fetching the same rows.
                break

        if not found:
            return []
        rows = [(fid, path, 0, checksum) for fid, path, checksum in found]
        return _enrich_rows(rows)

    def _category_checksums_by_name():
        """{category_name: set(checksums)} for every category, resolved once —
        manual assignments plus unsuppressed auto-matches, mirroring
        get_resolved_checksums_for_category's own manual-wins-over-auto logic but
        computed in a single pass over the whole auto-match table instead of one
        full-table scan per category. get_resolved_checksums_for_category does its
        own fresh db.get_all_file_category_matches() scan every call — fine for a
        single lookup, but the search palette calls this once per category
        *candidate* while ranking suggestions, so re-scanning per category turned
        one keystroke into O(categories) full-library scans. The manual-assignment
        and exclusion lookups below are similarly batched (get_all_category_checksums/
        get_all_category_exclusions, one query each for the whole library) instead
        of one query per category — that residual N+1 survived the first fix above."""
        checksums_by_category_id = manual.get_all_category_checksums()
        exclusions_by_category_id = manual.get_all_category_exclusions()
        by_name = {}
        excluded_by_category = {}
        for row in manual.list_categories():
            by_name[row['name']] = set(checksums_by_category_id.get(row['id'], set()))
            excluded_by_category[row['name']] = exclusions_by_category_id.get(row['id'], set())
        for _file_id, checksum, name, _score in db.get_all_file_category_matches():
            if checksum in by_name.get(name, ()):
                continue
            if checksum in excluded_by_category.get(name, ()):
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
        cur.execute("SELECT checksum, label FROM file_tags_with_label WHERE polarity = 'positive'")
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
        if t == 'location':
            # value 'any' == "has any location metadata": EXIF-geotagged files
            # unioned with files that carry at least one manual location
            # assignment; otherwise a specific location id's assigned files.
            if v == 'any':
                return db.get_geotagged_checksums() | manual.get_checksums_with_any_location()
            return set(manual.get_checksums_for_location(int(v)))
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
        _attach_set_people(sets_map.values())
        identities_map = _people_with_ages(manual.get_identities_for_checksums(checksums))
        category_map = resolve_categories_for_checksums(
            manual, db, [(r['id'], r['checksum']) for r in file_rows.values()]
        )
        for card in cards:
            row = file_rows.get(card[file_id_field])
            if row is None:
                continue
            checksum = row['checksum']
            card_sets = sets_map.get(checksum, [])
            card['filename'] = os.path.basename(row['path'])
            card['tags'] = tag_map.get(checksum, [])
            card['sets'] = card_sets
            card['people'] = _people_not_in_sets(identities_map.get(checksum, []), card_sets)
            card['categories'] = category_map.get(checksum, [])
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

    def _sort_file_rows(file_rows, sort, order, target_lat=None, target_lon=None):
        """Sort raw files_with_path rows (has first_seen/modified_time/checksum) by
        the 'added'/'modified' timestamp columns — used by the checksum-driven views
        (search by tag/person, set detail) which don't paginate via SQL. 'age' isn't
        handled here since it needs manual.db and enriched cards; see _sort_cards_by_age.

        'distance' ranks by haversine distance from (target_lat, target_lon) —
        always nearest-first (ascending distance) regardless of `order`; rows with
        no EXIF GPS (gps_lat is None) always sort LAST (mirrors the 'age'-with-no-
        estimate 'sort to end' pattern). If a distance sort is requested without a
        target point, it falls back to the default 'added' sort."""
        reverse = (order != 'asc')
        if sort == 'distance':
            if target_lat is None or target_lon is None:
                return sorted(file_rows, key=lambda r: r['first_seen'], reverse=True)
            with_gps = [r for r in file_rows if r['gps_lat'] is not None and r['gps_lon'] is not None]
            without_gps = [r for r in file_rows if r['gps_lat'] is None or r['gps_lon'] is None]
            with_gps.sort(key=lambda r: _haversine(r['gps_lat'], r['gps_lon'], target_lat, target_lon))
            return with_gps + without_gps
        if sort == 'filename':
            return sorted(file_rows, key=lambda r: os.path.basename(r['path']).lower(), reverse=reverse)
        if sort == 'modified':
            return sorted(file_rows, key=lambda r: r['modified_time'] or 0, reverse=reverse)
        if sort == 'favorites':
            counts = manual.get_favorite_counts([r['checksum'] for r in file_rows])
            return sorted(file_rows, key=lambda r: counts.get(r['checksum'], 0), reverse=reverse)
        return sorted(file_rows, key=lambda r: r['first_seen'], reverse=reverse)

    def _ordered_rows(checksums, sort, order, target_lat=None, target_lon=None):
        """Resolve a checksum set to files_with_path rows ordered by sort/order.
        Shared by the search palette (preview + open-as-queue). 'age' ranks by the
        checksum age map at the row level (no enrichment needed); everything else
        goes through _sort_file_rows. Rows with no age sort last."""
        if not checksums:
            return []
        rows = db.get_files_by_checksums(list(checksums))
        if sort == 'age':
            age_map = manual.get_average_ages_for_checksums([r['checksum'] for r in rows])
            with_age = [r for r in rows if r['checksum'] in age_map]
            without_age = [r for r in rows if r['checksum'] not in age_map]
            with_age.sort(key=lambda r: age_map[r['checksum']], reverse=(order != 'asc'))
            return with_age + without_age
        return _sort_file_rows(rows, sort or 'added', order, target_lat=target_lat, target_lon=target_lon)

    def _palette_media(checksums, sort, order, cap=16):
        """Top `cap` of `checksums` as enriched cards, ordered by sort/order."""
        rows = _ordered_rows(checksums, sort, order)[:cap]
        return _enrich_rows([(r['id'], r['path'], False, r['checksum']) for r in rows])

    def _search_resolve(q, chips, category_map, tag_map, set_map, identity_map):
        """Core resolver shared by the palette preview and open-as-queue endpoints.
        Returns (chip_cs, tag_cs, file_cs): the chip AND-intersection (None if no
        chips), and — when there's free text — the checksums whose tag names match
        and whose path matches, each narrowed by the chips when present."""
        chip_cs = _intersect_chips(chips, category_map=category_map, tag_map=tag_map,
                                   set_map=set_map, identity_map=identity_map)
        tag_cs = file_cs = None
        if q:
            ql = q.lower()
            tag_cs = set()
            for label, cksums in tag_map.items():
                if ql in label.lower():
                    tag_cs |= cksums
            file_cs = {row[2] for row in db.search_by_path_substring(q, limit=100_000)}
            if chip_cs is not None:
                tag_cs &= chip_cs
                file_cs &= chip_cs
        return chip_cs, tag_cs, file_cs

    def _effective_checksums(chip_cs, tag_cs, file_cs):
        """The set a committed search resolves to: chip intersection when chips are
        present, else the union of the free-text tag + file matches."""
        if chip_cs is not None:
            return chip_cs
        if tag_cs is not None or file_cs is not None:
            return (tag_cs or set()) | (file_cs or set())
        return set()

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
        is_video = ext in VIDEO_EXTENSIONS
        if ext not in IMAGE_EXTENSIONS and not is_video:
            return Response(content=_gray_placeholder(), media_type='image/jpeg')

        # Videos get a poster frame cached to the same {file_id}.jpg path as image
        # thumbnails (still served as image/jpeg), so the gallery's <img> works unchanged.
        thumb_path = os.path.join(thumbs_dir, f'{file_id}.jpg')

        if not os.path.isfile(thumb_path):
            abs_path = _live_abs_path(file_id, rel_path)
            if abs_path is None:
                return Response(content=_gray_placeholder(), media_type='image/jpeg')
            make_thumb = _make_video_thumbnail if is_video else _make_thumbnail
            success, message = make_thumb(abs_path, thumb_path)
            if message:
                errors.log(rel_path, message)
            if not success:
                return Response(content=_gray_placeholder(), media_type='image/jpeg')

        return FileResponse(thumb_path, media_type='image/jpeg', headers=IMMUTABLE_CACHE_HEADERS)

    @app.get('/image/{file_id}')
    def serve_image(file_id: int):
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        ext = os.path.splitext(abs_path)[1].lower()
        media_type = VIDEO_MIME_TYPES.get(ext)   # None for images → FileResponse guesses
        return FileResponse(abs_path, media_type=media_type, headers=IMMUTABLE_CACHE_HEADERS)

    @app.get('/api/files/{file_id}/neighbors')
    def api_file_neighbors(file_id: int):
        _file_or_404(file_id)
        prev_id, next_id = db.get_neighbor_ids(file_id)
        return {'prev': prev_id, 'next': next_id}

    # ------------------------------------------------------------------
    # HTML pages
    # ------------------------------------------------------------------

    def _cards_for_checksums(checksums):
        """Enrich a small, specific set of checksums into cards — never the whole
        library. Order follows get_files_by_checksums (arbitrary); callers re-sort."""
        checksums = list(checksums)
        if not checksums:
            return []
        rows = db.get_files_by_checksums(checksums)
        return _enrich_rows([(r['id'], r['path'], False, r['checksum']) for r in rows])

    @app.get('/', response_class=HTMLResponse)
    def gallery_page(request: Request):
        """A fast, curated landing page — three small sections, each enriching only a
        few dozen cards (never the whole library): favorites (most-favorited first),
        a random 'needs attention' sample (no face / no set / no manual tag), and a
        random sample. Browsing everything lives on Search / Files now."""
        # 1. Favorites, most-favorited first.
        fav = manual.get_top_favorite_checksums(limit=60)
        fav_rank = {cs: i for i, (cs, _c) in enumerate(fav)}
        favorites = _cards_for_checksums(fav_rank.keys())
        favorites.sort(key=lambda c: fav_rank.get(c['checksum'], 1 << 30))

        # 2. Needs attention: no face (SQL) ∩ no set ∩ no manual tag (Python).
        set_ex = manual.get_all_set_member_checksums()
        tag_ex = manual.get_checksums_with_manual_tags()
        needs = []
        for r in db.get_random_faceless_files(limit=200):
            cs = r['checksum']
            if cs in set_ex or cs in tag_ex:
                continue
            needs.append(cs)
            if len(needs) >= 6:
                break
        needs_attention = _cards_for_checksums(needs)

        # 3. Random.
        random_cards = _cards_for_checksums(db.get_random_file_checksums(limit=6))

        return templates.TemplateResponse(request, 'gallery.html', {
            'favorites': favorites,
            'needs_attention': needs_attention,
            'random_cards': random_cards,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
            'homepage_stats': _homepage_stats(),
        })

    @app.get('/files', response_class=HTMLResponse)
    @app.get('/files/{subpath:path}', response_class=HTMLResponse)
    def files_page(request: Request, subpath: str = ''):
        """Browse the tracked library as a folder tree. `subpath` (via the {path}
        converter, so it may contain slashes) is the current folder relative to
        data_root; '' is the root. Shows immediate subfolders + the media directly
        in this folder, and a "Use this folder" button that adds the whole subtree
        to a set."""
        folder = subpath.strip('/')
        subfolders, file_rows = db.browse_folder(folder)
        rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
        files = _enrich_rows(rows)
        return templates.TemplateResponse(request, 'files.html', {
            'folder': folder,
            'subfolders': subfolders,
            'files': files,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/photo/{file_id}', response_class=HTMLResponse)
    def photo_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        checksum = row['checksum']
        tag_rows = manual.get_tags(checksum)
        whole_tags = [
            {'id': t['id'], 'label': t['label'], 'polarity': t['polarity'], 'favorite': t['favorite']}
            for t in tag_rows if t['x1'] is None
        ]
        spatial_tags = [
            {'id': t['id'], 'label': t['label'], 'polarity': t['polarity']}
            for t in tag_rows if t['x1'] is not None
        ]
        all_tags = manual.list_all_tags()
        negated = manual.get_negated_labels(checksum)
        detected_classes = [c for c in db.get_detected_classes(file_id) if c not in negated]
        detection_bboxes = db.get_detection_bboxes(file_id)
        faces = _combined_faces_for_file(file_id, checksum)
        whole_photo_identities = manual.get_identities_assigned_to_photo(checksum)
        file_info = dict(row)
        file_info['filename'] = os.path.basename(file_info['path'])
        file_info['tags'] = whole_tags
        ext = os.path.splitext(file_info['path'])[1].lower()
        file_info['is_image'] = ext in IMAGE_EXTENSIONS
        file_info['is_video'] = ext in VIDEO_EXTENSIONS
        frame_count = 1
        if file_info['is_image']:
            abs_path = _live_abs_path(file_id, file_info['path'])
            if abs_path is not None:
                try:
                    frame_count = frames.get_frame_count(abs_path)
                except Exception:
                    frame_count = 1
        file_info['frame_count'] = frame_count
        # Frame capture applies to videos and multi-frame (animated) images.
        file_info['can_capture'] = file_info['is_video'] or (file_info['is_image'] and frame_count > 1)
        file_info['favorite'] = manual.get_file_favorite_count(checksum)
        file_info['title'] = manual.get_file_title(checksum)
        current_sets = [
            {'id': s['id'], 'name': s['name'], 'studio': s['studio'], 'favorite': s['favorite']}
            for s in manual.get_sets_for_file(checksum)
        ]
        _attach_set_people([current_sets])
        file_info['categories'] = resolve_categories_for_file(manual, db, file_id, checksum)
        file_info['locations'] = manual.get_locations_for_checksum(checksum)
        all_categories = _all_categories_for_nav()
        # Age/gender estimates (experimental — see age_estimator.py) are shown inline
        # next to each face's name, not as a separate section, so merge them directly
        # into the same `faces` list the template already iterates over.
        age_by_ref = {r['face_ref']: r for r in manual.get_age_estimates_for_checksum(checksum)}
        for face in faces:
            est = age_by_ref.get(face['ref'])
            face['age'] = est['age'] if est else None
            face['gender'] = est['gender'] if est else None
        # Manually-captured stills of this video (for the Frames pod), and — when this
        # file is itself a captured still — a link back to its source video.
        captured_frames = []
        if file_info['can_capture']:
            for cap in manual.get_frame_captures_for(checksum):
                rows = db.get_files_by_checksums([cap['child_checksum']])
                if rows:
                    captured_frames.append({'id': rows[0]['id'], 'time_ms': cap['time_ms']})
        parent_capture = None
        pc = manual.get_parent_capture(checksum)
        if pc:
            prows = db.get_files_by_checksums([pc['parent_checksum']])
            if prows:
                parent_capture = {'id': prows[0]['id'], 'time_ms': pc['time_ms'],
                                  'filename': os.path.basename(prows[0]['path'])}
        return templates.TemplateResponse(request, 'photo.html', {
            'file': file_info,
            'detected_classes': detected_classes,
            'detection_bboxes': detection_bboxes,
            'faces': faces,
            'whole_photo_identities': whole_photo_identities,
            'spatial_tags': spatial_tags,
            'all_tags': all_tags,
            'all_categories': all_categories,
            'current_sets': current_sets,
            'captured_frames': captured_frames,
            'parent_capture': parent_capture,
        })

    def _identity_instances(name):
        """Every individual APPEARANCE of this identity, not just every unique
        photo: one entry per real detected face (faces table, identity=name,
        rejected=0), so a photo where this person has two separate faces in it
        (e.g. a before/after collage) yields TWO entries, each carrying its own
        face_ref and therefore its own independently correct age estimate (see
        get_ages_for_face_refs / _apply_instance_ages) — rather than being
        silently collapsed into one card with an averaged or arbitrary age.
        Plus one entry per whole-photo manual assignment (no bbox, so no
        "instance" concept — always at most one per checksum) that isn't
        already covered by a face instance for this identity on that same
        checksum, to avoid double-counting the same photo from both sources.

        Deliberately does NOT expand into every member of a set linked to this
        identity — being linked to a set (see get_sets_linked_to_identity/
        link_identity_to_set) is a fact about the SET, not a claim that this
        person is recognizable in every one of that set's photos, so it must
        not fan out into fake per-photo matches on /person/{name}'s photo grid
        or count. Sets linked this way are still shown on the page as their
        own single roster entry (see person_page's linked_sets/sets_by_id) —
        just not exploded into individual "photos of this person" cards.

        Shared by /person/{name} and its /api/person/{name}/photos pagination
        endpoint (so it runs on every page load and every scroll-triggered
        page fetch)."""
        instances = []
        seen_face_checksums = set()
        for face_id, checksum, _embedding in manual.get_faces_for_identity(name):
            instances.append({'checksum': checksum, 'face_ref': f'manual:{face_id}'})
            seen_face_checksums.add(checksum)
        for checksum in manual.get_photos_assigned_to_identity(name, limit=1000):
            if checksum not in seen_face_checksums:
                instances.append({'checksum': checksum, 'face_ref': None})
        linked_sets = manual.get_sets_linked_to_identity(name)
        return instances, linked_sets

    def _apply_instance_ages(cards, instances, name, ages_by_ref):
        """Overrides each card's own entry for `name` in its people list with
        THIS SPECIFIC face instance's own age/gender (see _identity_instances
        and get_ages_for_face_refs) — _enrich_rows' generic per-checksum people
        list can't distinguish between two of this identity's own faces on the
        same photo, so without this override a photo with e.g. a before/after
        pair of this person would show the same (averaged, or arbitrarily
        picked) age on both cards instead of each face's own real estimate.

        Also re-adds `name`'s entry if it's missing entirely: _enrich_rows'
        _people_not_in_sets step strips a person from a card's own people chip
        when they're already shown via one of the card's set chips (avoiding a
        duplicate on generic gallery/set pages) — but /person/{name}'s whole
        point is showing THIS identity's own per-photo appearance, so it must
        always be present here even when a set chip elsewhere on the card also
        happens to mention them.

        `cards` and `instances` must be the same length and order (the cards
        were built from rows constructed directly from these instances)."""
        for card, inst in zip(cards, instances):
            card['face_ref'] = inst['face_ref']
            est = ages_by_ref.get(inst['face_ref']) if inst['face_ref'] else None
            age = est['age'] if est else None
            gender = est['gender'] if est else None
            matched = False
            for p in card['people']:
                if p['name'].lower() == name.lower():
                    p['age'] = age
                    p['gender'] = gender
                    matched = True
            if not matched:
                card['people'].append({'name': name, 'age': age, 'gender': gender})

    @app.get('/search', response_class=HTMLResponse)
    def search_page(request: Request, q: str = '', tag: str = '', start: str = '',
                     face_ref: str = '', category: str = '', sort: str = 'added', order: str = 'desc',
                     lat: float = None, lon: float = None,
                     f: List[str] = Query([])):
        # Distance sort needs a target point; without one, fall back to the default.
        if sort == 'distance' and (lat is None or lon is None):
            sort = 'added'
        files = []
        message = ''
        all_tags = manual.list_all_tags()
        all_categories = _all_categories_for_nav()
        similar_unknown_faces = []
        similar_faces = []
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
                file_rows = _sort_file_rows(db.get_files_by_checksums(list(checksums)), sort, order,
                                            target_lat=lat, target_lon=lon)
                rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
                files = _enrich_rows(rows)
                if sort == 'age':
                    files = _sort_cards_by_age(files, order)
            queue_label = 'Search: ' + ' + '.join(f'{c["type"]}:{c["value"]}' for c in chips)

        if f:
            pass
        elif tag:
            checksums = manual.get_files_by_tag(tag, limit=200)
            file_rows = _sort_file_rows(db.get_files_by_checksums(checksums), sort, order,
                                        target_lat=lat, target_lon=lon)
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
                    file_rows = _sort_file_rows(db.get_files_by_checksums(checksums), sort, order,
                                                target_lat=lat, target_lon=lon)
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
            'start': start,
            'category': category,
            'message': message,
            'all_tags': all_tags,
            'all_categories': all_categories,
            'similar_unknown_faces': similar_unknown_faces,
            'sort': sort,
            'order': order,
            'queue_label': queue_label,
            **(_classifier_ui_state(tag) if tag else {}),
        })

    @app.get('/person/{name}', response_class=HTMLResponse)
    def person_page(request: Request, name: str):
        """Per-person profile page: common tags across their photos, sets they
        appear in, and (via person.html's own JS + /api/person/{name}/photos)
        a lazily-paginated grid of their photos. Replaces the old /search?person=
        branch, which dumped an unpaginated grid mixed with generic
        bulk-select/add-to-set controls and manual set/photo assignment tools
        that belong on the set-detail and photo-detail pages instead."""
        canonical = manual.resolve_identity_name(name)
        if canonical != name:
            return RedirectResponse(url=f'/person/{quote(canonical)}')
        instances, linked_sets = _identity_instances(name)
        checksums = {inst['checksum'] for inst in instances}
        message = ''
        tags = []
        sets = []
        if not checksums:
            message = f'No files found with person "{name}" — run <code>media faces</code> to detect faces first.'
        else:
            tag_map = manual.list_tags_for_checksums(checksums)
            tag_counts = {}
            for labels in tag_map.values():
                for label in labels:
                    tag_counts[label] = tag_counts.get(label, 0) + 1
            tags = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))

            # Sets: everything directly linked to this identity, plus any set
            # that actually contains one of their photos (covers sets a photo
            # was added to individually, without the identity ever being
            # explicitly linked to that set).
            sets_by_id = {s['id']: {'id': s['id'], 'name': s['name'], 'studio': s['studio']} for s in linked_sets}
            for lst in manual.get_sets_for_checksums(list(checksums)).values():
                for s in lst:
                    sets_by_id.setdefault(s['id'], s)
            sets = sorted(sets_by_id.values(), key=lambda s: s['name'].lower())
            _attach_set_people([sets])

        return templates.TemplateResponse(request, 'person.html', {
            'name': name,
            'message': message,
            'tags': tags,
            'sets': sets,
            'aliases': manual.get_aliases_for_identity(name),
            'total': len(instances),
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    def _person_photos_collapsed(name, instances, file_rows, ages_by_ref, offset, limit, sort, order):
        """Collapse-to-set view: photos in a set become one set card (positioned by
        this-person's-slice averages — avg age of the person in the set, avg first_seen
        of the person's photos in it); set-less photos stay as photo cards. All mixed
        into one grid, sorted together by the page's active sort, then paged."""
        checksums = list(file_rows.keys())
        sets_by_cs = manual.get_sets_for_checksums(checksums)  # {cs: [{id,name,studio}]}

        # Group the person's in-set photos by set; keep set-less ones as photo items.
        set_group = {}   # set_id -> {'name','studio','checksums': set()}
        photo_insts = []
        for inst in instances:
            here = sets_by_cs.get(inst['checksum'])
            if here:
                for s in here:
                    g = set_group.setdefault(s['id'], {'name': s['name'], 'studio': s['studio'], 'checksums': set()})
                    g['checksums'].add(inst['checksum'])
            else:
                photo_insts.append(inst)

        # Per-(set, this person) average age — for positioning the set card under 'age'.
        age_in_set = manual.get_average_ages_for_identities_in_sets(list(set_group.keys())) if set_group else {}

        items = []
        for inst in photo_insts:
            r = file_rows[inst['checksum']]
            est = ages_by_ref.get(inst['face_ref']) if inst['face_ref'] else None
            items.append({'kind': 'photo', 'inst': inst, 'added': r['first_seen'],
                          'modified': r['modified_time'] or 0, 'age': est['age'] if est else None})
        for sid, g in set_group.items():
            members = list(g['checksums'])
            added_vals = [file_rows[cs]['first_seen'] for cs in members if cs in file_rows]
            avg_added = sum(added_vals) / len(added_vals) if added_vals else 0
            est = age_in_set.get((sid, name))
            items.append({'kind': 'set', 'set_id': sid, 'name': g['name'], 'studio': g['studio'],
                          'members': members, 'added': avg_added, 'modified': avg_added,
                          'age': est.get('age') if est else None})

        if sort == 'age':
            with_age = [it for it in items if it['age'] is not None]
            without_age = [it for it in items if it['age'] is None]
            with_age.sort(key=lambda it: it['age'], reverse=(order != 'asc'))
            items = with_age + without_age
        else:
            key = 'modified' if sort == 'modified' else 'added'
            items.sort(key=lambda it: it[key], reverse=(order != 'asc'))

        total = len(items)
        page = items[offset:offset + limit]

        # Enrich the page's photo items (batched, order-preserving); build set cards.
        photo_page = [it for it in page if it['kind'] == 'photo']
        rows = [(file_rows[it['inst']['checksum']]['id'], file_rows[it['inst']['checksum']]['path'],
                 False, it['inst']['checksum']) for it in photo_page]
        enriched = _enrich_rows(rows)
        _apply_instance_ages(enriched, [it['inst'] for it in photo_page], name, ages_by_ref)
        estimated = manual.get_estimated_checksums([c['checksum'] for c in enriched])
        for c in enriched:
            c['kind'] = 'photo'
            c['has_age_estimate'] = c['checksum'] in estimated

        set_cards = {}
        set_dicts = []
        for idx, it in enumerate(page):
            if it['kind'] != 'set':
                continue
            thumb_cs = it['members'][0] if it['members'] else None
            sd = {'kind': 'set', 'id': it['set_id'], 'name': it['name'], 'studio': it['studio'],
                  'image_count': len(it['members']),
                  'thumb_id': file_rows[thumb_cs]['id'] if thumb_cs and thumb_cs in file_rows else None}
            set_cards[idx] = sd
            set_dicts.append(sd)
        _attach_set_people([set_dicts])  # adds 'people' roster (with per-set ages)

        cards = []
        ei = iter(enriched)
        for idx, it in enumerate(page):
            cards.append(set_cards[idx] if it['kind'] == 'set' else next(ei))
        return {'cards': cards, 'total': total, 'offset': offset, 'limit': limit}

    @app.get('/api/person/{name}/photos')
    def api_person_photos(name: str, offset: int = 0, limit: int = 20, sort: str = 'added',
                          order: str = 'desc', collapse: bool = False):
        """Paginated card feed backing person.html's infinite-scroll photo list.
        Built from per-face-instance appearances (see _identity_instances), not
        unique photos — a photo where this person has two separate detected
        faces (e.g. a before/after collage) yields two cards, one per face,
        each showing that face's own independently correct age estimate (see
        _apply_instance_ages) rather than a card averaging (or, worse, mixing
        in unrelated people sharing the same photo — the bug behind ages that
        looked like random noise when sorting, since the old implementation
        sorted by manual.get_average_ages_for_checksums, a photo-wide average
        across EVERY face on that checksum regardless of identity).
        'age' needs every instance's age resolved before sorting/slicing since
        the ordering spans the whole list, not just one page; other sorts only
        need the page actually being returned enriched."""
        instances, _ = _identity_instances(name)
        if not instances:
            return {'cards': [], 'total': 0, 'offset': offset, 'limit': limit}

        checksums = list({inst['checksum'] for inst in instances})
        file_rows = {r['checksum']: r for r in db.get_files_by_checksums(checksums)}
        instances = [inst for inst in instances if inst['checksum'] in file_rows]

        face_refs = {inst['face_ref'] for inst in instances if inst['face_ref']}
        ages_by_ref = manual.get_ages_for_face_refs(face_refs)

        if collapse:
            return _person_photos_collapsed(name, instances, file_rows, ages_by_ref, offset, limit, sort, order)

        if sort == 'age':
            def age_key(inst):
                est = ages_by_ref.get(inst['face_ref']) if inst['face_ref'] else None
                return est['age'] if est else None
            with_age = [i for i in instances if age_key(i) is not None]
            without_age = [i for i in instances if age_key(i) is None]
            with_age.sort(key=age_key, reverse=(order != 'asc'))
            instances = with_age + without_age
        else:
            def timestamp_key(inst):
                r = file_rows[inst['checksum']]
                return (r['modified_time'] or 0) if sort == 'modified' else r['first_seen']
            instances.sort(key=timestamp_key, reverse=(order != 'asc'))

        page_instances = instances[offset:offset + limit]
        rows = [
            (file_rows[inst['checksum']]['id'], file_rows[inst['checksum']]['path'], False, inst['checksum'])
            for inst in page_instances
        ]
        page_cards = _enrich_rows(rows)
        _apply_instance_ages(page_cards, page_instances, name, ages_by_ref)

        # Lets person.html's "estimate all" button skip photos already estimated on a
        # prior (possibly interrupted) run instead of redoing them — see
        # get_estimated_checksums.
        estimated = manual.get_estimated_checksums([c['checksum'] for c in page_cards])
        for c in page_cards:
            c['has_age_estimate'] = c['checksum'] in estimated
        return {'cards': page_cards, 'total': len(instances), 'offset': offset, 'limit': limit}

    @app.get('/find-person/{name}', response_class=HTMLResponse)
    def find_person_page(request: Request, name: str):
        """Person-scoped face-suggestion swipe stream — 'is this <name>?' cards
        ranked by embedding similarity to their confirmed faces, same
        /api/face-suggestions/next stream as /faces but scoped via the
        `identity` param (see _next_face_suggestions' identity_filter). Split
        out from /person/{name} (which is a static profile: tags/sets/photos)
        so that page stays fast and this one can own the swipe/undo/threshold
        UI without cluttering the profile. Reachable via the 🔍 icon next to
        the name on /person/{name}."""
        canonical = manual.resolve_identity_name(name)
        if canonical != name:
            return RedirectResponse(url=f'/find-person/{quote(canonical)}')
        return templates.TemplateResponse(request, 'find_person.html', {
            'name': name,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/person/{name}/split', response_class=HTMLResponse)
    def split_person_page(request: Request, name: str, with_: str = Query('', alias='with')):
        """'Split' review: walks every real face currently assigned to `name`
        and lets the user confirm it's really them (↑, no-op) or move it to
        `with_` instead (↓, single reassign call) — see _next_split_candidates.
        Fixes the case where a batch of one person's photos got incorrectly
        assigned to another. Reachable via the ✂️ icon on /person/{name}."""
        canonical = manual.resolve_identity_name(name)
        if canonical != name:
            return RedirectResponse(url=f'/person/{quote(canonical)}/split?with={quote(with_)}')
        other = manual.resolve_identity_name(with_)
        known_identities = {n for n, _count in manual.get_all_identities()}
        error = None
        if not other:
            error = 'Pick a person to split against.'
        elif other.lower() == name.lower():
            error = f'Pick someone other than "{name}" to split against.'
        elif other not in known_identities:
            error = f'"{other}" isn\'t a known person yet.'
        return templates.TemplateResponse(request, 'split_person.html', {
            'name': name,
            'other': other,
            'error': error,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    def _next_split_candidates(name, other, exclude_refs, count):
        """Ranks `name`'s currently-assigned real faces by resemblance to
        `other`'s confirmed embeddings, best match first — the faces most
        likely to actually be `other` surface first, directly serving the
        "a batch of Joe's photos got tagged as Steve" case. Falls back to
        input order if `other` has no confirmed faces yet (nothing to rank
        against). Mirrors _next_face_suggestions' identity_filter branch."""
        other_embeddings = manual.get_embeddings_for_identity(other)
        faces = manual.get_faces_for_identity(name)
        import numpy as np
        matrix = (np.stack([np.frombuffer(e, dtype=np.float32) for e in other_embeddings])
                  if other_embeddings else None)
        candidates = []  # (score, ref, file_id)
        for face_id, checksum, emb_bytes in faces:
            ref = f'manual:{face_id}'
            if ref in exclude_refs:
                continue
            file_row = db.get_file_by_checksum(checksum)
            if file_row is None:
                continue
            if matrix is not None and emb_bytes:
                score = float(matrix.dot(np.frombuffer(emb_bytes, dtype=np.float32)).max())
            else:
                score = 0.0
            candidates.append((score, ref, file_row['id']))
        candidates.sort(key=lambda c: -c[0])
        return _attach_file_meta([
            {'ref': ref, 'file_id': file_id, 'score': round(score, 3)}
            for score, ref, file_id in candidates[:count]
        ])

    @app.post('/api/person/{name}/split-candidates')
    def api_split_candidates(name: str, body: SwipeExcludeBody, with_: str = Query(..., alias='with'), count: int = 10):
        exclude_refs = set(body.exclude)
        return {'cards': _next_split_candidates(name, with_, exclude_refs, count)}

    @app.get('/similar/{file_id}', response_class=HTMLResponse)
    def similar_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        filename = os.path.basename(row['path'])
        files = []
        message = ''
        no_embedding = False
        source_is_video = os.path.splitext(row['path'])[1].lower() in VIDEO_EXTENSIONS
        all_tags = manual.list_all_tags()

        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            # The template shows an "embed now" button instead of a message here.
            no_embedding = True
        else:
            try:
                import numpy as np
                query_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                dim = query_emb.shape[0]
                all_embs = db.get_all_embeddings()  # (file_id, path, embedding, checksum)
                if all_embs:
                    # One contiguous (N,D) matrix + a single BLAS matmul (GIL released),
                    # then a C-level top-K select via argpartition — NOT a Python
                    # sorted(zip(...)) over the whole embeddings table plus four full
                    # list comprehensions, which held the GIL long enough to stall the
                    # entire server on a large library.
                    matrix = np.frombuffer(
                        b''.join(r[2] for r in all_embs), dtype=np.float32
                    ).reshape(len(all_embs), dim)
                    scores = matrix @ query_emb
                    k = min(len(all_embs), 21)  # 20 results + room to drop self
                    top = np.argpartition(scores, -k)[-k:]
                    top = top[np.argsort(scores[top])[::-1]]
                    ranked = []
                    for i in top:
                        r = all_embs[int(i)]
                        if r[0] == file_id:
                            continue
                        ranked.append((r[0], r[1], float(scores[i]), r[3]))
                        if len(ranked) >= 20:
                            break
                    rows = [(fid, path, True, cs) for fid, path, _score, cs in ranked]
                    scoremap = {fid: round(score, 4) for fid, _path, score, _cs in ranked}
                    files = _enrich_rows(rows, scores=scoremap)
            except Exception as exc:
                message = f'Similarity search error: {exc}'

        return templates.TemplateResponse(request, 'similar.html', {
            'source_file': dict(row),
            'source_filename': filename,
            'source_id': file_id,
            'files': files,
            'message': message,
            'no_embedding': no_embedding,
            'source_is_video': source_is_video,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/api/files/{file_id}/similar')
    def api_similar_files(file_id: int, limit: int = 50):
        """Whole-image CLIP similarity as JSON: file_ids ranked most-similar first.
        Lets the photo page's "distance" quick-action open the matches straight
        into the browsable watch-queue (same UX as find-by-face) instead of the
        /similar/{id} results page. Uses the cached embeddings matrix + a
        GIL-free argpartition top-K."""
        _file_or_404(file_id)
        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            return {'results': [], 'message': 'This photo has no embedding yet.'}
        import numpy as np
        from media_manager.similarity import top_k_indices
        query_emb = np.frombuffer(emb_bytes, dtype=np.float32)
        file_ids, _checksums, matrix = db.get_embeddings_matrix()
        if matrix.shape[0] == 0:
            return {'results': []}
        scores = matrix.dot(query_emb)
        k = min(matrix.shape[0], max(1, limit) + 1)  # +1 to absorb self-match
        results = []
        for i in top_k_indices(scores, k):
            fid = int(file_ids[i])
            if fid == file_id:
                continue
            results.append({'file_id': fid, 'score': round(float(scores[i]), 4)})
            if len(results) >= limit:
                break
        return {'results': results}

    @app.post('/api/files/{file_id}/region-search')
    def api_region_search(file_id: int, body: RegionSearchBody, limit: int = 50):
        """Semantic 'search by region': CLIP-embed the drawn crop and rank the library's
        whole-image embeddings by cosine similarity. Opens matches in the watch-queue."""
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None or not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        x1, y1, x2, y2 = [float(v) for v in body.bbox]
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise HTTPException(status_code=400, detail='Region too small.')
        from PIL import Image as PILImage
        with PILImage.open(abs_path) as im:
            crop = im.convert('RGB').crop((x1, y1, x2, y2))
            crop.load()
        embs = _get_clip_indexer().embed_pil_images([crop])
        if len(embs) == 0:
            raise HTTPException(status_code=500, detail='Could not embed the region.')
        import numpy as np
        from media_manager.similarity import top_k_indices
        query = np.asarray(embs[0], dtype=np.float32)

        # Prefer the tile index when it's been built: rank each photo by its
        # BEST-matching tile, which localizes small/off-center regions (a wall in
        # a corner) that a whole-image embedding would miss. Scanned in RAM-bounded
        # chunks (see db.iter_tile_embeddings) rather than one giant resident matrix.
        if db.count_tiled_files() > 0:
            best = {}  # file_id -> best tile cosine
            for file_ids, matrix in db.iter_tile_embeddings():
                if matrix.shape[0] == 0:
                    continue
                scores = matrix.dot(query)
                for idx in range(scores.shape[0]):
                    fid = int(file_ids[idx])
                    if fid == file_id:
                        continue
                    s = float(scores[idx])
                    if s > best.get(fid, -1.0):
                        best[fid] = s
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:limit]
            return {'mode': 'tiles',
                    'results': [{'file_id': fid, 'score': round(s, 4)} for fid, s in ranked]}

        # Whole-image fallback (no tile index yet): rank the cached whole-image matrix.
        file_ids, _checksums, matrix = db.get_embeddings_matrix()
        if matrix.shape[0] == 0:
            return {'mode': 'whole', 'results': []}
        scores = matrix.dot(query)
        results = []
        for i in top_k_indices(scores, min(matrix.shape[0], max(1, limit) + 1)):
            fid = int(file_ids[i])
            if fid == file_id:
                continue
            results.append({'file_id': fid, 'score': round(float(scores[i]), 4)})
            if len(results) >= limit:
                break
        return {'mode': 'whole', 'results': results}

    # ------------------------------------------------------------------
    # Find-by-body: person re-ID by outfit/build (see body_index.py).
    # Web-only by design — the body index is built and queried from here,
    # never by a CLI command.
    # ------------------------------------------------------------------

    body_index_job = {'running': False, 'done': 0, 'total': 0, 'error': None}
    # Two more library-wide bulk jobs surfaced from the navbar's ⚡ menu, same
    # {running,done,total,error} shape as body_index_job (see the ⚡ dropdown in
    # base.html + runBulkAction in app.js). index_job = CLIP-embed every
    # not-yet-indexed photo; match_faces_job = match unnamed auto-detected faces
    # to known people (no model — pure embedding math, so it also carries a
    # 'matched' count).
    index_job = {'running': False, 'done': 0, 'total': 0, 'error': None}
    match_faces_job = {'running': False, 'done': 0, 'total': 0, 'matched': 0, 'error': None}
    # Tile (region) index: per-image CLIP crops of an overlapping grid, so region
    # search can localize small/off-center things (see tile_index.py + the region
    # search endpoint). Same job shape; surfaced in the ⚡ menu as "Index regions".
    tile_index_job = {'running': False, 'done': 0, 'total': 0, 'error': None}
    # Metadata (EXIF capture-time + GPS) extraction: same {running,done,total,error}
    # job shape; surfaced in the ⚡ menu as "Extract locations". Mirrors
    # MediaManager.extract_metadata but with progress (see api_metadata_start).
    metadata_job = {'running': False, 'done': 0, 'total': 0, 'error': None}

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
        # Cached, write-invalidated [N, D] body matrix — no per-request full
        # re-load + np.stack of every body embedding.
        body_ids_arr, file_ids_arr, _bboxes, matrix = db.get_body_embeddings_matrix()
        if matrix.shape[0] == 0:
            return []
        scores = matrix.dot(query_emb)

        best_per_file = {}
        for i in np.argsort(-scores):
            i = int(i)
            score = float(scores[i])
            if score < threshold or len(best_per_file) >= limit:
                break
            fid = int(file_ids_arr[i])
            if fid == exclude_file_id or fid in best_per_file:
                continue
            best_per_file[fid] = (int(body_ids_arr[i]), score)

        body_ids = {}
        rows = []
        scores = {}
        for fid, (body_id, score) in best_per_file.items():
            file_row = db.get_file_by_id(fid)
            if file_row is None:
                continue
            body_ids[fid] = body_id
            rows.append((fid, file_row['path'], False, file_row['checksum']))
            scores[fid] = round(score, 4)

        results = _enrich_rows(rows, scores=scores)
        for card in results:
            card['body_id'] = body_ids[card['id']]
        results.sort(key=lambda item: item['score'], reverse=True)
        return results

    @app.get('/body-similar/{file_id}', response_class=HTMLResponse)
    def body_similar_page(request: Request, file_id: int, body_id: Optional[int] = None):
        """Page shell only — deliberately does NOT call _ensure_own_bodies/_body_search
        here. The first visit to this page for a not-yet-indexed photo needs a full
        YOLO-World + CLIP pass (see _ensure_own_bodies), which is too slow to run
        inline in a page request; the template's JS fetches
        /api/files/{file_id}/body-similar after render instead, showing a loading
        state meanwhile (mirrors person.html's lazily-loaded photo grid)."""
        row = _file_or_404(file_id)
        return templates.TemplateResponse(request, 'body_similar.html', {
            'source_id': file_id,
            'source_filename': os.path.basename(row['path']),
            'body_id': body_id,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/api/files/{file_id}/body-similar')
    def api_body_similar(file_id: int, body_id: Optional[int] = None):
        """The actual find-by-body work (see body_similar_page's docstring for why
        it's not inline in the page route): ensures this photo's body crops are
        embedded (on-demand, first call only — see _ensure_own_bodies), then ranks
        every other photo's body crops against the chosen one."""
        row = _file_or_404(file_id)
        message = ''
        results = []
        selected_id = None

        try:
            own = _ensure_own_bodies(file_id, row)
        except HTTPException as exc:
            own = []
            message = str(exc.detail)

        crops = [{'id': r[0]} for r in own]
        no_people = False
        if not own and not message:
            message = 'No people detected in this photo.'
            no_people = True
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

        return {
            'crops': crops,
            'selected_id': selected_id,
            'results': results,
            'message': message,
            'no_people': no_people,
        }

    @app.post('/api/files/{file_id}/body-reindex')
    def api_body_reindex(file_id: int):
        """Force a fresh person-only YOLO pass for ONE photo and (re)embed its body
        crops, overwriting any prior 'no people' result. Backs the find-by-body
        page's "Reindex now" button: the normal path reuses the general object index
        and only counts detections labeled exactly 'person', so a person YOLO-World
        labeled as child/crowd/group/selfie/portrait is missed — this re-runs
        detection with the vocab pinned to 'person' to recover it."""
        from media_manager import body_index
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        detector = _get_object_detector()
        detector.set_vocab(['person'])
        try:
            _, dets, err = detector.detect_images([abs_path])[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'Person detection failed: {exc}')
        if err:
            raise HTTPException(status_code=500, detail=f'Person detection failed: {err}')
        person_boxes = [[x1, y1, x2, y2]
                        for _cls, conf, x1, y1, x2, y2 in dets
                        if conf >= body_index.MIN_PERSON_CONFIDENCE]
        count = body_index.embed_bodies_for_file(
            db, _get_clip_indexer(), file_id, abs_path, person_boxes=person_boxes)
        return {'bodies': count}

    @app.post('/api/files/{file_id}/bodies')
    def api_add_manual_body(file_id: int, body: ManualFaceBody):
        """Add one hand-drawn body box as a searchable body embedding (mirrors
        api_add_manual_face, but crops+CLIP-embeds instead of face recognition).
        Backs the photo view's "Label person" region tool."""
        from media_manager import body_index
        if len(body.bbox) != 4:
            raise HTTPException(status_code=400, detail='bbox must be [x1, y1, x2, y2]')
        x1, y1, x2, y2 = body.bbox
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
            raise HTTPException(status_code=400, detail='Region too small')
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        pairs = body_index.crop_bodies(abs_path, [[int(x1), int(y1), int(x2), int(y2)]])
        if not pairs:
            raise HTTPException(status_code=400, detail='Region too small or out of bounds')
        indexer = _get_clip_indexer()
        embs = indexer.embed_pil_images([crop for _, crop in pairs])
        used_bbox = pairs[0][0]
        body_id = db.add_manual_body(
            file_id, used_bbox, embs[0].astype('float32').tobytes(), indexer.model_name)
        return {'id': body_id, 'bbox': used_bbox}

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
        return FileResponse(crop_path, media_type='image/jpeg', headers=IMMUTABLE_CACHE_HEADERS)

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

    @app.post('/api/index/start')
    def api_index_start():
        """Bulk-embed every photo that has no CLIP embedding yet — the library-wide
        equivalent of the /similar page's per-file "embed now". Mirrors
        MediaManager.embed_files but with progress + the web CLIP accessor (so it
        offloads to the worker when configured)."""
        import threading
        from media_manager.indexer import SUPPORTED_EXTENSIONS

        if index_job['running']:
            return {'started': False, 'message': 'Indexing already running.'}
        candidates = [
            (fid, os.path.join(data_root, rel_path))
            for fid, rel_path in db.get_unindexed_files(limit=None)
            if os.path.splitext(rel_path)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        index_job.update(running=True, done=0, total=len(candidates), error=None)

        def _run():
            try:
                indexer = _get_clip_indexer()
                model_id = indexer.model_id()
                done = 0
                for start in range(0, len(candidates), 32):
                    batch = candidates[start:start + 32]
                    paths = [p for _fid, p in batch]
                    embeddings, failed = indexer.embed_images(paths)
                    failed_messages = dict(failed)
                    emb_iter = iter(embeddings)
                    for fid, fpath in batch:
                        if fpath in failed_messages:
                            errors.log(fpath, failed_messages[fpath])
                        else:
                            db.insert_embedding(fid, next(emb_iter).tobytes(), model_id)
                        done += 1
                        index_job['done'] = done
            except Exception as exc:
                index_job['error'] = str(exc)
            finally:
                index_job['running'] = False

        threading.Thread(target=_run, daemon=True).start()
        return {'started': True, 'total': index_job['total']}

    @app.get('/api/index/status')
    def api_index_status():
        return {
            'running': index_job['running'],
            'done': index_job['done'],
            'total': index_job['total'],
            'error': index_job['error'],
            'pending': len(db.get_unindexed_files()),
        }

    @app.post('/api/metadata/start')
    def api_metadata_start():
        """Bulk-extract EXIF capture time + GPS for every file whose metadata has
        never been checked. Mirrors MediaManager.extract_metadata but with progress:
        non-image files are still stamped checked (update_file_metadata(fid, None,
        None, None)) so they don't resurface on every future run; image files get
        their real EXIF read via exif_reader.extract_exif_metadata."""
        import threading
        from media_manager import exif_reader

        if metadata_job['running']:
            return {'started': False, 'message': 'Metadata extraction already running.'}
        candidates = db.get_files_without_metadata()  # [(id, path), ...]
        metadata_job.update(running=True, done=0, total=len(candidates), error=None)

        def _run():
            try:
                done = 0
                for fid, rel_path in candidates:
                    ext = os.path.splitext(rel_path)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        # Not an image — nothing to extract, but stamp it checked.
                        db.update_file_metadata(fid, None, None, None)
                    else:
                        taken_at, gps_lat, gps_lon = exif_reader.extract_exif_metadata(
                            os.path.join(data_root, rel_path))
                        db.update_file_metadata(fid, taken_at, gps_lat, gps_lon)
                    done += 1
                    metadata_job['done'] = done
            except Exception as exc:
                metadata_job['error'] = str(exc)
            finally:
                metadata_job['running'] = False

        threading.Thread(target=_run, daemon=True).start()
        return {'started': True, 'total': metadata_job['total']}

    @app.get('/api/metadata/status')
    def api_metadata_status():
        return {
            'running': metadata_job['running'],
            'done': metadata_job['done'],
            'total': metadata_job['total'],
            'error': metadata_job['error'],
            'pending': len(db.get_files_without_metadata()),
        }

    @app.post('/api/match-faces/start')
    def api_match_faces_start():
        """Match every not-yet-named auto-detected face against known identities and
        promote the confident hits — the library-wide version of the auto-match that
        api_detect_faces already does per photo. Needs no model: find_matching_identity
        is a dot-product against the cached named-face matrix, so this is fast and
        never touches the worker."""
        import threading

        if match_faces_job['running']:
            return {'started': False, 'message': 'Face matching already running.'}
        pool = _unpromoted_auto_faces(limit=None)
        match_faces_job.update(running=True, done=0, total=len(pool), matched=0, error=None)

        def _run():
            try:
                # One file_id -> checksum lookup per hit is fine (hits are the rare
                # case); get_file_by_id is a single indexed row read.
                done = 0
                matched = 0
                for face_id, file_id, _path, bbox, emb_bytes in pool:
                    done += 1
                    match_faces_job['done'] = done
                    if not emb_bytes:
                        continue
                    name, _score = manual.find_matching_identity(emb_bytes)
                    if name is None:
                        continue
                    file_row = db.get_file_by_id(file_id)
                    if file_row is None:
                        continue
                    manual.promote_auto_face(
                        face_id, file_row['checksum'], json.loads(bbox), emb_bytes,
                        name, None, None)
                    matched += 1
                    match_faces_job['matched'] = matched
            except Exception as exc:
                match_faces_job['error'] = str(exc)
            finally:
                match_faces_job['running'] = False

        threading.Thread(target=_run, daemon=True).start()
        return {'started': True, 'total': match_faces_job['total']}

    @app.get('/api/match-faces/status')
    def api_match_faces_status():
        return {
            'running': match_faces_job['running'],
            'done': match_faces_job['done'],
            'total': match_faces_job['total'],
            'matched': match_faces_job['matched'],
            'error': match_faces_job['error'],
        }

    @app.post('/api/tile-index/start')
    def api_tile_index_start():
        """Build the per-image CLIP tile index that powers precise region search —
        an overlapping grid of crops embedded per photo. Same background-job shape
        as the body index; CLIP goes through _get_clip_indexer() so it offloads to
        the worker when configured."""
        import threading
        from media_manager import tile_index

        if tile_index_job['running']:
            return {'started': False, 'message': 'Region indexing already running.'}
        tile_index_job.update(
            running=True, done=0, total=len(db.get_untiled_files()), error=None)

        def _run():
            try:
                def _progress(done, total):
                    tile_index_job['done'] = done
                    tile_index_job['total'] = total
                tile_index.build_tile_index(
                    db, errors, _get_clip_indexer(), data_root, on_progress=_progress)
            except Exception as exc:
                tile_index_job['error'] = str(exc)
            finally:
                tile_index_job['running'] = False

        threading.Thread(target=_run, daemon=True).start()
        return {'started': True, 'total': tile_index_job['total']}

    @app.get('/api/tile-index/status')
    def api_tile_index_status():
        return {
            'running': tile_index_job['running'],
            'done': tile_index_job['done'],
            'total': tile_index_job['total'],
            'error': tile_index_job['error'],
            'pending': len(db.get_untiled_files()),
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
            {'id': t['id'], 'label': t['label'], 'polarity': t['polarity'], 'favorite': t['favorite']}
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
        count = manual.bump_tag_favorite(tag_id, body.delta)
        return {'id': tag_id, 'count': count}

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
        polarity = 'negative' if body.polarity == 'negative' else 'positive'

        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')

        from PIL import Image as PILImage
        with PILImage.open(abs_path) as img:
            width, height = img.size

        tag_id = manual.add_spatial_tag(row['checksum'], label, x1, y1, x2, y2, width, height, polarity=polarity)
        return {'id': tag_id, 'label': label, 'bbox': [x1, y1, x2, y2], 'polarity': polarity}

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
        return {'detected_classes': detected_classes, 'detection_bboxes': db.get_detection_bboxes(file_id)}

    @app.post('/api/files/{file_id}/embed')
    def api_embed_file(file_id: int):
        """Build + store the CLIP embedding for a single file on demand — powers
        the "embed now" button on the /similar page when a file has no embedding
        yet (instead of telling the user to run `media embed`). Synchronous, same
        shape as api_reindex_tags; the lazy CLIP singleton is usually already warm
        from other endpoints (find-by-body / text search)."""
        row = _file_or_404(file_id)
        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='File not found on disk')
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail='Only images can be embedded (CLIP is image-only).')

        indexer = _get_clip_indexer()
        embeddings, failed = indexer.embed_images([abs_path])
        if len(embeddings) == 0:
            detail = failed[0][1] if failed else 'no embedding produced'
            raise HTTPException(status_code=500, detail=f'Embedding failed: {detail}')
        db.insert_embedding(file_id, embeddings[0].tobytes(), indexer.model_id())
        return {'embedded': True}

    @app.get('/api/tags')
    def api_list_tags():
        rows = manual.list_all_tags()
        return [{'tag': r[0], 'count': r[1]} for r in rows]

    @app.get('/tags', response_class=HTMLResponse)
    def tags_page(request: Request):
        all_tags = manual.list_all_tags()
        tags = [
            {'label': r[0], 'count': r[1], **_classifier_ui_state(r[0])}
            for r in all_tags
        ]
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

    # Real per-image YOLO inference (unlike CLIP's cheap embedding dot-product)
    # — how many undecided candidates one "find more via YOLO" fetch scans per
    # call. Bounded so a real library doesn't turn one fetch into hundreds of
    # sequential/batched model calls (see _next_tag_suggestions_by_model).
    YOLO_CANDIDATE_SAMPLE_SIZE = 60
    # When the fine-tuned checkpoint lives on the worker, each detection is a
    # separate RNS round-trip shipping an image (not one batched local predict),
    # so scan far fewer per fetch — repeated refills still cover the library.
    WORKER_YOLO_SAMPLE_SIZE = 8

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

        # Whole-image centroid ranking has no inherent "where", so the swipe
        # fullview (Right arrow) had nothing to highlight for this path. If the
        # tile index is built, localize each match to its best-matching sub-tile
        # vs the centroid and attach a normalized bbox — same shape the YOLO path
        # emits, so swipe-core highlights it identically.
        cards = []
        for row, score in qualifying[:count]:
            card = {'ref': str(row[0]), 'file_id': row[0], 'label': label,
                    'score': round(float(score), 3)}
            bbox = _best_tile_bbox(row[0], centroid)
            if bbox is not None:
                card['bbox'] = bbox
            cards.append(card)
        return _attach_file_meta(cards)

    def _best_tile_bbox(file_id, centroid):
        """Normalized [x1,y1,x2,y2] of the sub-tile of `file_id` most similar to
        `centroid` — "where in this photo the tag matches", for the swipe fullview
        highlight. The whole-image tile is skipped (it can't localize). None when
        the file has no tile index yet (▦ Index regions not run)."""
        tiles = db.get_tiles_for_file(file_id)
        if not tiles:
            return None
        import numpy as np
        w = max((t[2] for t in tiles), default=0) or 1.0
        h = max((t[3] for t in tiles), default=0) or 1.0
        best, best_score = None, -2.0
        for x1, y1, x2, y2, emb in tiles:
            if (x2 - x1) * (y2 - y1) >= 0.9 * w * h:
                continue  # the whole-image tile — highlighting it tells you nothing
            s = float(np.frombuffer(emb, dtype=np.float32).dot(centroid))
            if s > best_score:
                best_score, best = s, (x1, y1, x2, y2)
        if best is None:
            return None
        x1, y1, x2, y2 = best
        return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]

    @app.post('/api/tags/{label}/suggestions/next')
    def api_next_tag_suggestions(label: str, body: SwipeExcludeBody, count: int = 10,
                                  bias_file_id: str = '', bias: str = ''):
        exclude_ids = {int(x) for x in body.exclude if x.isdigit()}
        bias_id = int(bias_file_id) if bias_file_id.strip().isdigit() else None
        cards = _next_tag_suggestions(label, count, exclude_ids, bias_id, bias or None)
        return {'cards': cards}

    # ------------------------------------------------------------------
    # Per-tag classifier training (experimental) — a CLIP linear classifier
    # and a fine-tuned YOLO-World checkpoint, each trained from this tag's
    # own confirmed/rejected examples (see clip_tag_classifier.py /
    # yolo_tag_classifier.py). Both always run as a real subprocess, never
    # in-process — training can take anywhere from seconds (CLIP) to minutes
    # (YOLO), and this app already hit real GIL-contention problems earlier
    # from heavy in-process computation (see the /sets/{id} "opening 2 tabs
    # makes everything else unresponsive" incident that led to --workers).
    # metadata.json (see clip_tag_classifier.write_status) is the single
    # source of truth for status — no separate DB table to keep in sync.
    # ------------------------------------------------------------------

    def _read_classifier_status(label, kind):
        """kind is 'clip_model' or 'yolo_model'. Returns None if training was
        never started for this tag/kind. A 'running' status whose pid is no
        longer alive (crash, OOM-kill, `kill -9`) is reported back as 'failed'
        here — WITHOUT rewriting the file — so the UI doesn't show
        "training…" forever after something goes wrong outside the normal
        try/except in the training script itself."""
        from media_manager.clip_tag_classifier import classifier_dir

        path = os.path.join(classifier_dir(data_root, label, kind), 'metadata.json')
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                status = json.load(f)
        except (OSError, ValueError):
            return None
        if status.get('status') == 'running':
            pid = status.get('pid')
            # pid is None only in the brief window between the launching
            # request writing this placeholder and the child process getting
            # far enough into Python startup to overwrite it with its real
            # pid (see _launch_training_subprocess) — not itself a sign of
            # failure. A pid that's SET but no longer alive is the real
            # "died unexpectedly" signal.
            if pid and not os.path.exists(f'/proc/{pid}'):
                return {**status, 'status': 'failed', 'error': 'Training process died unexpectedly.'}
        return status

    def _classifier_button_state(kind, label, status, enough, hint):
        """Everything a template needs to render one train button, fully
        precomputed server-side — no client-side logic decides what the
        button says or whether it's clickable, only a trivial click-to-fetch
        handler is left to JS (see tags.html/search.html), so a broken/blocked
        script can never leave the button silently missing or wrong the way
        the previous JS-rendered version did.
        `href` set means "render as a link to the Find-more page"; `href` None
        means "render as a button", disabled per `disabled`. `title` is the
        tooltip — either the failure error or the "why disabled" hint.
        `cancel_url` is only set while genuinely running (the one state a
        stuck/long job can actually be stopped from) — killing the subprocess
        is currently the only way to stop it, see api_cancel_training.
        `log_url` is set whenever a train.log could plausibly exist (anything
        past "never started"), so progress/failure detail beyond the coarse
        status is always one click away."""
        noun = 'CLIP' if kind == 'clip' else 'YOLO'
        log_url = f'/api/tags/{quote(label)}/train-log?kind={kind}' if status else None
        if status and status.get('status') == 'done':
            metric = status.get('cv_accuracy') if kind == 'clip' else status.get('held_out_map')
            metric_text = f'{round(metric * 100)}%' if isinstance(metric, (int, float)) else 'done'
            return {
                'kind': kind, 'label': label, 'disabled': False,
                'text': f'{noun}: {metric_text} — Find more',
                'href': f'/search?tag={quote(label)}&start={kind}', 'title': '',
                'cancel_url': None, 'log_url': log_url,
            }
        if status and status.get('status') == 'running':
            elapsed_min = max(0, round((time.time() - status.get('started_at', time.time())) / 60))
            progress = ''
            if status.get('current_epoch') and status.get('total_epochs'):
                progress = f' — epoch {status["current_epoch"]}/{status["total_epochs"]}'
            return {
                'kind': kind, 'label': label, 'disabled': True, 'href': None,
                'text': f'{noun}: training…{progress} ({elapsed_min}m)',
                'title': 'Cancel to stop, or check the log for detail.',
                'cancel_url': f'/api/tags/{quote(label)}/train-cancel?kind={kind}', 'log_url': log_url,
            }
        if status and status.get('status') == 'failed':
            return {
                'kind': kind, 'label': label, 'disabled': False, 'href': None,
                'text': f'{noun}: failed — retry', 'title': status.get('error', ''),
                'cancel_url': None, 'log_url': log_url,
            }
        return {
            'kind': kind, 'label': label, 'disabled': not enough, 'href': None,
            'text': f'Train {noun}', 'title': '' if enough else hint,
            'cancel_url': None, 'log_url': None,
        }

    def _classifier_ui_state(label):
        """Everything tags.html/search.html need to render this tag's train
        buttons — one precomputed dict per model kind (see
        _classifier_button_state)."""
        from media_manager.clip_tag_classifier import MIN_EXAMPLES_PER_CLASS
        from media_manager.yolo_tag_classifier import MIN_BOXES

        polarities = manual.get_whole_image_tag_polarities(label)
        n_positive = sum(1 for p in polarities.values() if p == 'positive')
        n_negative = sum(1 for p in polarities.values() if p == 'negative')
        # Region crops (see get_all_spatial_tags_for_label/clip_tag_classifier.py's
        # _add_region_examples) count toward CLIP's threshold too — a hard-negative
        # region (e.g. a dog boxed "not monstera plant") is just as valid a
        # training example as a whole-image rejection.
        for region_row in manual.get_all_spatial_tags_for_label(label):
            if region_row[-1] == 'positive':
                n_positive += 1
            else:
                n_negative += 1
        n_boxes = len(manual.get_spatial_tags_for_label(label))
        clip_status = _read_classifier_status(label, 'clip_model')
        yolo_status = _read_classifier_status(label, 'yolo_model')
        return {
            'clip_btn': _classifier_button_state(
                'clip', label, clip_status, n_positive >= MIN_EXAMPLES_PER_CLASS and n_negative >= MIN_EXAMPLES_PER_CLASS,
                f'Need {MIN_EXAMPLES_PER_CLASS}+ confirmed and {MIN_EXAMPLES_PER_CLASS}+ rejected examples, '
                f'whole-image or region (have {n_positive} confirmed, {n_negative} rejected).'
            ),
            'yolo_btn': _classifier_button_state(
                'yolo', label, yolo_status, n_boxes >= MIN_BOXES,
                f'Need {MIN_BOXES}+ labeled bounding boxes (have {n_boxes}).'
            ),
            'clip_status': clip_status,
            'yolo_status': yolo_status,
        }

    def _launch_training_subprocess(module_name, label, kind):
        """Writes the initial 'running' status itself, synchronously, before
        returning — the child process's own train_for_tag also writes this
        (with its real pid, overwriting this placeholder within moments), but
        relying on that alone leaves a real race: two rapid clicks on "Train"
        could both pass the "already running?" check in api_train_*_classifier
        before either child has actually gotten far enough into Python
        startup/imports to write anything."""
        from media_manager.clip_tag_classifier import classifier_dir, write_status

        # When a worker is configured, offload the heavy training: launch the
        # remote_tag_trainer driver (ships the data to the worker, polls it)
        # instead of the local trainer, passing --kind so one driver handles
        # both. The driver honours the same metadata.json/train.log/pid contract,
        # so the status/log/cancel routes below are unchanged.
        if _worker_client is not None and _worker_client.is_configured():
            argv = [sys.executable, '-m', 'media_manager.remote_tag_trainer',
                    '--data-root', data_root, '--tag', label, '--kind', kind]
        elif os.environ.get('MEDIA_ALLOW_LOCAL_TRAINING') == '1':
            # Explicit opt-in for a standalone/beefy host with no worker.
            argv = [sys.executable, '-m', module_name,
                    '--data-root', data_root, '--tag', label]
        else:
            # No worker AND no opt-in: refuse. Local training (YOLO-World fine-tune
            # especially) is the heaviest job in the app and will exhaust memory on
            # a low-RAM host — this is the exact click that OOM'd a prod box. Fail
            # loud instead of silently launching it.
            raise HTTPException(
                status_code=400,
                detail='No ML worker is configured, so training would run locally and can '
                       'exhaust this host\'s memory (YOLO-World fine-tuning is the heaviest '
                       'job in the app). Connect a worker with `media worker-connect` to '
                       'offload it, or set MEDIA_ALLOW_LOCAL_TRAINING=1 to train locally anyway.')

        out_dir = classifier_dir(data_root, label, kind)
        os.makedirs(out_dir, exist_ok=True)
        write_status(out_dir, status='running', pid=None, started_at=int(time.time()))
        subprocess.Popen(argv, start_new_session=True)

    @app.post('/api/tags/{label}/train-clip')
    def api_train_clip_classifier(label: str):
        from media_manager.clip_tag_classifier import MIN_EXAMPLES_PER_CLASS

        existing = _read_classifier_status(label, 'clip_model')
        if existing and existing.get('status') == 'running':
            raise HTTPException(status_code=409, detail='CLIP training is already running for this tag.')

        polarities = manual.get_whole_image_tag_polarities(label)
        n_positive = sum(1 for p in polarities.values() if p == 'positive')
        n_negative = sum(1 for p in polarities.values() if p == 'negative')
        if n_positive < MIN_EXAMPLES_PER_CLASS or n_negative < MIN_EXAMPLES_PER_CLASS:
            raise HTTPException(
                status_code=400,
                detail=f'Need at least {MIN_EXAMPLES_PER_CLASS} confirmed and {MIN_EXAMPLES_PER_CLASS} '
                       f'rejected examples; have {n_positive} confirmed, {n_negative} rejected.'
            )
        _launch_training_subprocess('media_manager.clip_tag_classifier', label, 'clip_model')
        return {'status': 'running'}

    @app.post('/api/tags/{label}/train-yolo')
    def api_train_yolo_classifier(label: str):
        from media_manager.yolo_tag_classifier import MIN_BOXES

        existing = _read_classifier_status(label, 'yolo_model')
        if existing and existing.get('status') == 'running':
            raise HTTPException(status_code=409, detail='YOLO training is already running for this tag.')

        n_boxes = len(manual.get_spatial_tags_for_label(label))
        if n_boxes < MIN_BOXES:
            raise HTTPException(
                status_code=400,
                detail=f'Need at least {MIN_BOXES} labeled bounding boxes; have {n_boxes}.'
            )
        _launch_training_subprocess('media_manager.yolo_tag_classifier', label, 'yolo_model')
        return {'status': 'running'}

    @app.get('/api/tags/{label}/train-status')
    def api_train_status(label: str, kind: str = 'clip'):
        status = _read_classifier_status(label, 'clip_model' if kind == 'clip' else 'yolo_model')
        return status or {'status': 'not_trained'}

    @app.post('/api/tags/{label}/train-cancel')
    def api_cancel_training(label: str, kind: str = 'clip'):
        """Stops a running training subprocess — the only way to do this today
        is to actually kill the OS process (there's no cooperative "please
        stop soon" flag either script checks). SIGTERM first (lets torch/
        ultralytics unwind and release any file handles cleanly); the status
        is marked 'failed' here directly rather than waiting for the next
        status read's dead-pid check, so the UI reflects the cancellation
        immediately instead of still saying "running" until someone happens
        to reload."""
        from media_manager.clip_tag_classifier import classifier_dir

        model_kind = 'clip_model' if kind == 'clip' else 'yolo_model'
        status = _read_classifier_status(label, model_kind)
        if not status or status.get('status') != 'running':
            raise HTTPException(status_code=400, detail='No training is currently running for this tag.')
        pid = status.get('pid')
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone — fine, we're marking it failed either way
        out_dir = classifier_dir(data_root, label, model_kind)
        from media_manager.clip_tag_classifier import write_status
        write_status(out_dir, status='failed', error='Cancelled by user.', failed_at=int(time.time()))
        return {'status': 'failed'}

    @app.get('/api/tags/{label}/train-log')
    def api_train_log(label: str, kind: str = 'clip'):
        """Raw train.log content (the training subprocess's own stdout/stderr,
        see clip_tag_classifier.py's/yolo_tag_classifier.py's main()) — the
        only place per-run detail lives beyond the coarse running/done/failed
        status, e.g. ultralytics' own per-epoch loss table for a YOLO run."""
        from media_manager.clip_tag_classifier import classifier_dir

        model_kind = 'clip_model' if kind == 'clip' else 'yolo_model'
        log_path = os.path.join(classifier_dir(data_root, label, model_kind), 'train.log')
        if not os.path.isfile(log_path):
            return PlainTextResponse('No log yet.')
        with open(log_path, errors='replace') as f:
            return PlainTextResponse(f.read())

    def _next_tag_suggestions_by_model(label, model_kind, count, exclude_ids):
        """Candidate ranking for the "find more" swipe view — same shape as
        _next_tag_suggestions (same decided/exclude filtering, same
        _attach_file_meta response), but scored by a trained classifier
        instead of centroid-cosine similarity."""
        decided = manual.get_decided_checksums_for_tag(label)

        if model_kind == 'yolo':
            from media_manager.clip_tag_classifier import classifier_dir, slugify

            # With a worker configured the checkpoint lives on the worker (kept
            # there so the heavy YOLO predict never runs on this host); gate on
            # the tag being trained (metadata status=done) instead of a local
            # best.pt. Otherwise the checkpoint is local as before.
            use_worker = _worker_client is not None and _worker_client.is_configured()
            if use_worker:
                st = _read_classifier_status(label, 'yolo_model')
                if not st or st.get('status') != 'done':
                    return []
            else:
                checkpoint = os.path.join(classifier_dir(data_root, label, 'yolo_model'), 'best.pt')
                if not os.path.isfile(checkpoint):
                    return []

            candidates = [
                row for row in db.get_all_embeddings()
                if row[3] not in decided and row[0] not in exclude_ids
            ]
            if not candidates:
                return []

            # Real model inference per image (not a cheap embedding dot-product
            # like the CLIP path) — scanning the WHOLE undecided-photo pool
            # synchronously on every single fetch was the actual cause of the
            # swipe stream getting stuck on "searching" forever on a real
            # library. Bounded random sample keeps each fetch fast; repeated
            # fetches (the swipe stack's own refill, each excluding what it's
            # already seen) progressively cover more of the library over a
            # session. The worker path samples fewer, since each detection is a
            # separate RNS round-trip shipping an image rather than one batched
            # local predict() call.
            sample_size = WORKER_YOLO_SAMPLE_SIZE if use_worker else YOLO_CANDIDATE_SAMPLE_SIZE
            sample = random.sample(candidates, min(len(candidates), sample_size))

            path_by_file_id = {}
            for file_id, path, _emb, _checksum in sample:
                abs_path = os.path.join(data_root, path)
                if os.path.isfile(abs_path):
                    path_by_file_id[file_id] = abs_path
            if not path_by_file_id:
                return []

            scored = []
            if use_worker:
                # Offload each detection to the worker's fine-tuned checkpoint.
                # Ship a downscaled JPEG (same 640px cap training used). tag_detect
                # returns [(class, conf, x1,y1,x2,y2)] with NORMALIZED coords.
                from media_manager.remote_tag_trainer import _downscale_jpeg
                from media_manager.worker_client import WorkerError
                from PIL import Image as PILImage
                slug = slugify(label)
                for file_id, abs_path in path_by_file_id.items():
                    try:
                        with PILImage.open(abs_path) as im:
                            image_bytes = _downscale_jpeg(im.convert('RGB'))
                        dets = _worker_client.tag_detect(slug, image_bytes,
                                                         name=os.path.basename(abs_path))
                    except WorkerError as exc:
                        # e.g. checkpoint missing on the worker (wiped/replaced) —
                        # loud, not silent, then stop scanning this fetch.
                        print(f'[tag classifier] worker tag_detect failed for tag "{label}": {exc}')
                        break
                    except Exception:
                        print(f'[tag classifier] worker tag_detect error for tag "{label}":')
                        traceback.print_exc()
                        break
                    if dets:
                        _cls, conf, x1, y1, x2, y2 = max(dets, key=lambda d: d[1])
                        scored.append((file_id, float(conf),
                                       [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]))
            else:
                try:
                    from ultralytics import YOLOWorld
                    model = YOLOWorld(checkpoint)
                    # Batched in one call (not one predict() per image) — batching
                    # amortizes real per-call model overhead.
                    results_list = model.predict(list(path_by_file_id.values()), verbose=False)
                except Exception:
                    # Loud, not silent — a bad/incompatible checkpoint or a
                    # transient inference failure should show up in the server
                    # console immediately, not hang the request or look empty.
                    print(f'[tag classifier] YOLO inference failed for tag "{label}" using {checkpoint}:')
                    traceback.print_exc()
                    return []
                for file_id, results in zip(path_by_file_id.keys(), results_list):
                    confs = results.boxes.conf if results.boxes is not None else []
                    if len(confs):
                        best_idx = int(confs.argmax())
                        # Normalized (0..1) xyxy — resolution-independent.
                        bbox = [round(float(v), 4) for v in results.boxes.xyxyn[best_idx]]
                        scored.append((file_id, float(confs[best_idx]), bbox))

            scored.sort(key=lambda x: -x[1])
            return _attach_file_meta([
                {'ref': str(fid), 'file_id': fid, 'label': label, 'score': round(score, 3), 'bbox': bbox}
                for fid, score, bbox in scored[:count]
            ])

        from media_manager.clip_tag_classifier import score_all

        all_embeddings = [
            row for row in db.get_all_embeddings()
            if row[3] not in decided and row[0] not in exclude_ids
        ]
        candidates = [(row[3], row[2]) for row in all_embeddings]
        checksum_to_file_id = {row[3]: row[0] for row in all_embeddings}
        scores = score_all(data_root, label, candidates)
        ranked = sorted(
            ((checksum_to_file_id[c], s) for c, s in scores.items() if c in checksum_to_file_id),
            key=lambda x: -x[1],
        )
        return _attach_file_meta([
            {'ref': str(fid), 'file_id': fid, 'label': label, 'score': round(score, 3)}
            for fid, score in ranked[:count]
        ])

    @app.post('/api/tags/{label}/suggestions/next-by-model')
    def api_next_tag_suggestions_by_model(label: str, body: SwipeExcludeBody, model: str = 'clip', count: int = 10):
        exclude_ids = {int(x) for x in body.exclude if x.isdigit()}
        cards = _next_tag_suggestions_by_model(label, model, count, exclude_ids)
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
        q = body.q.strip()
        chips = [{'type': c.type, 'value': c.value} for c in body.chips]
        sort, order = body.sort, body.order
        # Computed once per request — see each _*_checksums_by_* helper's
        # docstring for why resolving per-chip from scratch (fine for the rare
        # /search f= page load) doesn't scale to a call made on every keystroke.
        category_map = _category_checksums_by_name()
        tag_map = _tag_checksums_by_label()
        set_map = _set_checksums_by_id()
        identity_map = _identity_checksums_by_name(set_map)

        chip_cs, tag_cs, file_cs = _search_resolve(q, chips, category_map, tag_map, set_map, identity_map)
        effective = _effective_checksums(chip_cs, tag_cs, file_cs)

        # Applied-filter photos (only when chips are active — free text results go
        # in the tag/file sections instead). Free text yields its own two sections.
        media = _palette_media(chip_cs, sort, order) if chip_cs else []
        tag_matches = _palette_media(tag_cs, sort, order) if tag_cs else []
        file_matches = _palette_media(file_cs, sort, order) if file_cs else []

        # Which sets/people are represented in the effective result set — shown as
        # their own tiles so a result reads as "these sets / these people / these
        # photos", not one undifferentiated grid.
        sets_result = []
        people_result = []
        if effective:
            sets_map = {}
            identities_map = {}
            for chunk in _chunked(effective):
                sets_map.update(manual.get_sets_for_checksums(chunk))
                identities_map.update(manual.get_identities_for_checksums(chunk))

            set_agg = {}
            for cs, sets_here in sets_map.items():
                for s in sets_here:
                    agg = set_agg.setdefault(s['id'], {'id': s['id'], 'name': s['name'], 'studio': s['studio'], 'count': 0, 'checksum': cs})
                    agg['count'] += 1
            sets_result = sorted(set_agg.values(), key=lambda s: s['count'], reverse=True)[:6]
            for s in sets_result:
                thumb_rows = db.get_files_by_checksums([s.pop('checksum')])
                s['thumb_file_id'] = thumb_rows[0]['id'] if thumb_rows else None
            _attach_set_people([sets_result])

            people_agg = {}
            for cs, names in identities_map.items():
                for name in names:
                    people_agg[name] = people_agg.get(name, 0) + 1
            face_ids = manual.get_representative_face_ids()
            people_result = sorted(
                [{'name': n, 'count': c, 'face_id': face_ids.get(n)} for n, c in people_agg.items()],
                key=lambda p: p['count'], reverse=True
            )[:6]

        return {'media': media, 'tag_matches': tag_matches, 'file_matches': file_matches,
                'sets': sets_result, 'people': people_result, 'total_count': len(effective)}

    @app.post('/api/search-ids')
    def api_search_ids(body: SearchPaletteBody):
        """Full ordered list of file ids for the current search (chips + free text),
        for the 'open results as a queue' action — the frontend writes these into the
        photo viewer's watch queue and opens the first. Same resolver/sort as the
        palette preview, just uncapped and ids-only."""
        q = body.q.strip()
        chips = [{'type': c.type, 'value': c.value} for c in body.chips]
        category_map = _category_checksums_by_name()
        tag_map = _tag_checksums_by_label()
        set_map = _set_checksums_by_id()
        identity_map = _identity_checksums_by_name(set_map)
        chip_cs, tag_cs, file_cs = _search_resolve(q, chips, category_map, tag_map, set_map, identity_map)
        effective = _effective_checksums(chip_cs, tag_cs, file_cs)
        rows = _ordered_rows(effective, body.sort, body.order)
        return {'ids': [r['id'] for r in rows]}

    @app.post('/api/files/meta')
    def api_files_meta(body: FileMetaBody):
        """Lightweight per-file display metadata (name + set names) for a list of
        file ids — the Space photo-grid overlay only carries ids and fetches
        captions in one batch. Reuses the same batched accessors as _enrich_rows
        (titles + sets keyed by checksum); no tags/people/categories/thumbnails.
        `name` is the file's title if it has one, else its basename. Returns only
        the ids that still resolve to a file (order not guaranteed — the client
        keys by id)."""
        if not body.ids:
            return []
        rows = db.get_files_by_ids(body.ids)
        checksums = [r['checksum'] for r in rows]
        title_map = manual.get_titles_for_checksums(checksums)
        sets_map = manual.get_sets_for_checksums(checksums)
        result = []
        for r in rows:
            checksum = r['checksum']
            result.append({
                'id': r['id'],
                'name': title_map.get(checksum) or os.path.basename(r['path']),
                'sets': [{'id': s['id'], 'name': s['name']}
                         for s in sets_map.get(checksum, [])],
            })
        return result

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
    def sets_page(request: Request, favorite: bool = False, studio: str = '',
                  f: List[str] = Query([])):
        all_tags = manual.list_all_tags()
        studio_filter = studio or None

        # Chips from the search palette (currently just 'face', e.g. "joe" ->
        # a confirmed person chip) narrow the set list down to sets that
        # actually contain a photo matching every chip — same chip format and
        # resolver as /search's f= handling, so this and the palette never
        # drift apart. None means "no restriction" (all sets), not "match
        # nothing", to match _intersect_chips' own convention.
        chips = []
        for part in f:
            chip_type, _, chip_value = part.partition(':')
            if chip_type and chip_value:
                chips.append({'type': chip_type, 'value': chip_value})

        # Computed once regardless of chips — reused below both to resolve a
        # 'face'/'set' chip filter (avoiding _checksums_for_chip's unindexed
        # LOWER(identity) LIKE '%...%' scan for 'face') and to pick each set's
        # thumbnail without a get_files_by_set + get_files_by_checksums round
        # trip per set (this page is unpaginated — that used to be 2 queries
        # times every set in the library, on every view).
        set_checksums_map = _set_checksums_by_id()
        identity_map = None
        if any(c['type'] == 'face' for c in chips):
            identity_map = _identity_checksums_by_name(set_checksums_map)
        matching_checksums = _intersect_chips(chips, set_map=set_checksums_map, identity_map=identity_map)
        matching_set_ids = None
        if matching_checksums is not None:
            matching_set_ids = set()
            if matching_checksums:
                for sets_here in manual.get_sets_for_checksums(list(matching_checksums)).values():
                    matching_set_ids.update(s['id'] for s in sets_here)

        filtered_rows = [
            r for r in manual.list_sets(favorite_only=favorite, studio=studio_filter)
            if matching_set_ids is None or r['id'] in matching_set_ids
        ]
        representative_checksum_by_set = {}
        for r in filtered_rows:
            checksums_here = set_checksums_map.get(r['id'])
            if checksums_here:
                representative_checksum_by_set[r['id']] = next(iter(checksums_here))
        thumb_rows_by_checksum = {}
        all_repr_checksums = list(representative_checksum_by_set.values())
        if all_repr_checksums:
            thumb_rows_by_checksum = {row['checksum']: row for row in db.get_files_by_checksums(all_repr_checksums)}

        sets = []
        for r in filtered_rows:
            checksum = representative_checksum_by_set.get(r['id'])
            thumb_row = thumb_rows_by_checksum.get(checksum) if checksum else None
            sets.append({
                'id': r['id'], 'name': r['name'], 'studio': r['studio'],
                'image_count': r['image_count'], 'favorite': r['favorite'],
                'thumb_id': thumb_row['id'] if thumb_row else None,
            })
        _attach_set_people([sets])
        return templates.TemplateResponse(request, 'sets.html', {
            'sets': sets,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'favorite_only': favorite,
            'studio_filter': studio_filter,
            'chips': chips,
        })

    @app.get('/studios', response_class=HTMLResponse)
    def studios_page(request: Request):
        all_tags = manual.list_all_tags()
        studios = [
            {'id': r['id'], 'name': r['studio'], 'set_count': r['set_count'],
             'image_count': r['image_count'], 'locations': manual.get_locations_for_studio(r['id'])}
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

    @app.put('/api/studios/{studio}')
    def api_rename_studio(studio: str, body: StudioRenameBody):
        """Renames a studio everywhere it's used. Since studios are just a free-text
        column on `sets` (no separate table/id), renaming one studio to another
        that already exists IS the merge — every set lands on the same string,
        exactly like fixing a typo would."""
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail='Studio name must not be empty')
        manual.rename_studio(studio, new_name)
        return {'name': new_name}

    # Cosine similarity above this, an image outside the set is suggested as a
    # possible member. CLIP image-image similarity runs hotter than face similarity
    # (same-scene photos commonly score 0.7-0.9), hence the higher default.
    SET_SUGGEST_THRESHOLD = 0.75

    def _find_similar_files_for_set(set_id, threshold, limit=12, offset=0, exclude_ids=None, avoid_existing=True,
                                     honor_negatives=False, exclude_non_matching_faces=False):
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
        handful already selected.

        `honor_negatives` (default False — a UI toggle, off by default):
        rejected photos (file_set_exclusions) are always hard-excluded from the
        candidate pool regardless of this flag — what it controls is whether
        they *also* pull the ranking centroid away from their own embeddings
        (Rocchio-style relevance feedback, see similarity.adjusted_centroid),
        so photos that merely resemble something already rejected rank lower
        too, not just literal re-suggestions of the same rejected photo.

        `exclude_non_matching_faces` (default False — a UI toggle, off by
        default): drops any threshold-passing candidate that has a recognized
        face belonging to someone who isn't already confirmed in this set (see
        get_people_present_in_set — manually assigned people plus anyone with a
        named face on one of the set's own photos). A candidate with no
        recognized face at all is never dropped by this — there's nothing to
        judge a mismatch against. If the set has no confirmed people yet
        either, this is a no-op (same reason)."""
        from media_manager.similarity import mean_normalized_centroid, adjusted_centroid, rank_by_similarity

        member_checksums = manual.get_files_by_set(set_id, limit=1000)
        member_ids = [r['id'] for r in db.get_files_by_checksums(member_checksums)]
        member_embeddings = [e for _fid, e in db.get_embeddings_for_files(member_ids)]

        member_checksum_set = set(member_checksums)
        excluded_checksum_set = manual.get_excluded_checksums_for_set(set_id)

        if honor_negatives and excluded_checksum_set:
            excluded_ids = [r['id'] for r in db.get_files_by_checksums(list(excluded_checksum_set))]
            excluded_embeddings = [e for _fid, e in db.get_embeddings_for_files(excluded_ids)]
            centroid = adjusted_centroid(member_embeddings, excluded_embeddings)
        else:
            centroid = mean_normalized_centroid(member_embeddings)
        if centroid is None:
            return []

        candidates = [
            row for row in db.get_all_embeddings()
            if row[3] not in member_checksum_set and row[3] not in excluded_checksum_set
        ]
        if not candidates:
            return []
        ranked = rank_by_similarity(centroid, candidates, embedding_index=2)
        passing = [((fid, path, cs), score) for (fid, path, _emb, cs), score in ranked if score >= threshold]
        if exclude_non_matching_faces and passing:
            set_people = set(manual.get_people_present_in_set(set_id, member_checksums).keys())
            if set_people:
                identities_map = manual.get_identities_for_checksums([item[0][2] for item in passing])
                passing = [
                    item for item in passing
                    if not (set(identities_map.get(item[0][2], [])) - set_people)
                ]
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

    def _find_best_sets_for_file(file_id, checksum, threshold=SET_SUGGEST_THRESHOLD, limit=3, set_ids=None):
        """The reverse of _find_similar_files_for_set: given one photo, rank every
        set it's not already in by how close the photo's embedding is to that set's
        centroid — "which set does this probably belong to". Sets are cheap and
        unpaginated everywhere else in the app (personal-library scale), so unlike
        the set-detail version this runs synchronously over all of them per call.

        set_ids, when given, restricts the candidate pool to just those ids (still
        excluding sets the file already belongs to) and returns a score for every
        one of them, regardless of `threshold`/`limit` — used by the set-detail
        page's quick-access modal, which wants a rank ordering of a small,
        caller-chosen set of sets, not "does this clear the bar to suggest it at
        all". Cheaper than the default path too: only the requested sets' centroids
        get computed, not every set in the library."""
        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            return []
        import numpy as np
        query_emb = np.frombuffer(emb_bytes, dtype=np.float32)

        member_of = {s['id'] for s in manual.get_sets_for_checksums([checksum]).get(checksum, [])}
        if set_ids is not None:
            candidates = [s for s in manual.list_sets() if s['id'] in set_ids and s['id'] not in member_of]
        else:
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
            if set_ids is not None or score >= threshold:
                # Representative thumbnail = a member photo we already resolved
                # to a file id above (no extra query).
                thumb_id = member_ids[0] if member_ids else None
                scored.append((set_row, score, thumb_id))

        scored.sort(key=lambda item: item[1], reverse=True)
        page = scored if set_ids is not None else scored[:limit]
        results = [
            {'id': s['id'], 'name': s['name'], 'studio': s['studio'],
             'score': round(score, 3), 'thumb_id': thumb_id}
            for s, score, thumb_id in page
        ]
        _attach_set_people([results])
        return results

    @app.get('/api/files/{file_id}/suggested-sets')
    def api_suggested_sets_for_file(file_id: int, ids: str = '', limit: int = 3):
        row = _file_or_404(file_id)
        set_ids = None
        if ids:
            set_ids = {int(s) for s in ids.split(',') if s.strip().isdigit()}
        return {'results': _find_best_sets_for_file(file_id, row['checksum'],
                                                    limit=max(1, min(limit, 24)), set_ids=set_ids)}

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

    def _folder_suggestions_for_set(file_rows, limit=8):
        """Candidate folder paths to prime the "+ Add folder" modal with: the parent
        directory of every photo already in this set, deduplicated and ranked by how
        many of the set's current photos live there (most first) — if 3 of 5 photos
        already in a set live under 'vacation/john/2023/', that's almost certainly
        the folder the user wants to bulk-add the rest of. Paths are stored relative
        to data_root with '/' separators (see upsert_file_path); a file directly at
        the root (no '/') has no parent folder and is skipped."""
        counts = {}
        order = []
        for r in file_rows:
            path = r['path'] if 'path' in r.keys() else None
            if not path or '/' not in path:
                continue
            folder = path.rsplit('/', 1)[0]
            if folder not in counts:
                counts[folder] = 0
                order.append(folder)
            counts[folder] += 1
        ranked = sorted(order, key=lambda f: (-counts[f], f))
        return [{'path': f, 'count': counts[f]} for f in ranked[:limit]]

    @app.get('/sets/{set_id}', response_class=HTMLResponse)
    def set_detail_page(request: Request, set_id: int, sort: str = 'added', order: str = 'desc'):
        set_row = manual.get_set(set_id)
        if set_row is None:
            raise HTTPException(status_code=404, detail='Set not found')
        all_tags = manual.list_all_tags()
        manual_mode = (sort == 'manual')
        checksums = manual.get_files_by_set(set_id, limit=500, manual_order=manual_mode)
        file_rows = db.get_files_by_checksums(checksums)
        if manual_mode:
            # get_files_by_checksums returns rows unordered; keep the manual order.
            by_ck = {r['checksum']: r for r in file_rows}
            file_rows = [by_ck[c] for c in checksums if c in by_ck]
        else:
            file_rows = _sort_file_rows(file_rows, sort, order)
        rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
        files = _enrich_rows(rows)
        # Every card here already belongs to this set — showing this set's own
        # roster again on each card (chip-set + set_meta_line) is pure redundant
        # context and, since a set's people and a photo's own recognized people
        # commonly overlap, reads as the same person listed twice on one card.
        # Cards still show any OTHER set they also belong to.
        for f in files:
            f['sets'] = [s for s in f['sets'] if s['id'] != set_id]
        if sort == 'age':
            files = _sort_cards_by_age(files, order)
        people_present = _people_present_in_set(set_id, checksums)
        folder_suggestions = _folder_suggestions_for_set(file_rows)
        set_dict = dict(set_row)
        _attach_set_people([[set_dict]])
        set_dict['aliases'] = manual.get_aliases_for_set(set_id)
        return templates.TemplateResponse(request, 'set_detail.html', {
            'set': set_dict,
            'files': files,
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'set_suggest_threshold': SET_SUGGEST_THRESHOLD,
            'people_present': people_present,
            'folder_suggestions': folder_suggestions,
            'sort': sort,
            'order': order,
        })

    def _files_under_folder(folder_path):
        """Resolve a folder path (as sent by the add-folder endpoints / the /files
        "Use this folder" button) to its tracked files. An empty path is the library
        root — the /files root button uses this to add the entire library — so it
        maps to every tracked file rather than being rejected. Any non-empty path
        goes through the recursive, indexed find_files_under_folder."""
        folder_path = (folder_path or '').strip().strip('/')
        if not folder_path:
            return db.list_files(limit=100_000_000)
        return db.find_files_under_folder(folder_path)

    @app.get('/api/sets/{set_id}/add-folder/preview')
    def api_preview_add_folder_to_set(set_id: int, path: str):
        """Read-only counterpart to add-folder — lets the "add folder" modal show
        what it's about to do (a review carousel, when there's more than a
        handful of new photos) before actually committing any membership."""
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        matches = _files_under_folder(path)
        existing = set(manual.get_files_by_set(set_id, limit=100000))
        new_files = [
            {'id': r['id'], 'filename': os.path.basename(r['path']), 'path': r['path']}
            for r in matches if r['checksum'] not in existing
        ]
        return {
            'matched': len(matches),
            'already_present': len(matches) - len(new_files),
            'new_count': len(new_files),
            'files': new_files,
        }

    @app.post('/api/sets/{set_id}/add-folder')
    def api_add_folder_to_set(set_id: int, body: AddFolderBody):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        matches = _files_under_folder(body.path)
        if not matches:
            return {'matched': 0, 'added': 0, 'already_present': 0}
        existing = {c for c in manual.get_files_by_set(set_id, limit=100000)}
        added = 0
        already_present = 0
        for row in matches:
            checksum = row['checksum']
            if checksum in existing:
                already_present += 1
                continue
            manual.assign_file_to_set(checksum, set_id)
            existing.add(checksum)
            added += 1
        return {'matched': len(matches), 'added': added, 'already_present': already_present}

    @app.get('/api/sets/{set_id}/file-ids')
    def api_set_file_ids(set_id: int):
        """Current member file ids of a set. Used by the photo viewer's
        "Add queue to set" grid to grey out (but still show) queue items that
        are already in the target set. Membership is checksum-keyed in manual.db,
        so resolve those to live file ids via media.db."""
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        checksums = manual.get_files_by_set(set_id, limit=100_000_000)
        rows = db.get_files_by_checksums(checksums)
        return {'file_ids': [r['id'] for r in rows]}

    @app.post('/api/sets/{set_id}/similar-files')
    def api_similar_files_for_set(set_id: int, body: SwipeExcludeBody, threshold: float = SET_SUGGEST_THRESHOLD,
                                   limit: int = 12, offset: int = 0, avoid_existing: bool = True,
                                   honor_negatives: bool = False, exclude_non_matching_faces: bool = False):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        threshold = max(0.0, min(1.0, threshold))
        offset = max(0, offset)
        exclude_ids = {int(r) for r in body.exclude if r.isdigit()} or None
        results = _find_similar_files_for_set(set_id, threshold, limit=limit, offset=offset, exclude_ids=exclude_ids,
                                                avoid_existing=avoid_existing, honor_negatives=honor_negatives,
                                                exclude_non_matching_faces=exclude_non_matching_faces)
        # 'cards' is the same list under the key swipe-core.js's fetchMoreUrl
        # response contract expects; 'results' stays for the pre-existing grid
        # (offset-based load-more + threshold-expand) callers, unchanged.
        return {'results': results, 'cards': results}

    @app.get('/api/sets')
    def api_list_sets(favorite: bool = False, light: bool = False):
        """`light=True` uses _attach_set_people_names_only instead of
        _attach_set_people — the set-search picker (openEntitySearchModal's
        'set' type) needs to know WHO's linked to each set (its personAware
        "type a name to narrow the list" filter depends on it) but never
        displays their age, so skip the per-set face/estimate join
        (get_average_ages_for_identities_in_sets) that computes it — real
        query cost with nothing to show for it in a text-only picker. The
        full (non-light) list — /sets/'s own page, favorites, etc. — still
        gets ages attached as before."""
        sets = [
            {'id': r['id'], 'name': r['name'], 'studio': r['studio'],
             'image_count': r['image_count'], 'favorite': r['favorite']}
            for r in manual.list_sets(favorite_only=favorite)
        ]
        # Representative thumbnail per set (so the set picker can show one) — one
        # whole-library checksum map + a single batched row lookup, not per-set.
        set_checksums_map = _set_checksums_by_id()
        repr_by_id = {}
        for s in sets:
            cks = set_checksums_map.get(s['id'])
            if cks:
                repr_by_id[s['id']] = next(iter(cks))
        thumb_rows = ({row['checksum']: row for row in db.get_files_by_checksums(list(repr_by_id.values()))}
                      if repr_by_id else {})
        for s in sets:
            cs = repr_by_id.get(s['id'])
            row = thumb_rows.get(cs) if cs else None
            s['thumb_id'] = row['id'] if row else None
        if light:
            _attach_set_people_names_only([sets])
        else:
            _attach_set_people([sets])
        _attach_set_aliases([sets])
        return sets

    @app.post('/api/sets')
    def api_create_set(body: SetBody):
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Set name must not be empty')
        studio = body.studio.strip() if body.studio else None

        if studio and not body.confirm_studio_merge:
            variant = manual.find_studio_variant(studio)
            if variant is not None:
                return JSONResponse(status_code=409, content={
                    'conflict': 'studio',
                    'existing_studio': {
                        'studio': variant['studio'], 'set_count': variant['set_count'],
                        'image_count': variant['image_count'],
                    },
                })

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

        # Renaming onto an already-existing name is very likely the same set
        # under a slightly different name (or a straight-up duplicate) rather
        # than a deliberate second set sharing a name — ask before silently
        # creating a name collision, unless the client already confirmed the
        # merge (confirm_merge, set after the user accepts the prompt below).
        # An existing *alias* of another set counts the same way — aliases are
        # meant to be an equally valid way to find that set, so renaming onto
        # one is just as much a collision as renaming onto its literal name.
        collisions = manual.find_sets_by_name(name, exclude_id=set_id)
        if not collisions:
            alias_set_id = manual.find_set_id_by_alias(name)
            if alias_set_id is not None and alias_set_id != set_id:
                alias_set = manual.get_set(alias_set_id)
                if alias_set is not None:
                    collisions = [alias_set]
        if collisions and not body.confirm_merge:
            existing = collisions[0]
            return JSONResponse(status_code=409, content={
                'conflict': 'name',
                'existing_set': {
                    'id': existing['id'], 'name': existing['name'], 'studio': existing['studio'],
                    'image_count': len(manual.get_files_by_set(existing['id'], limit=100000)),
                },
            })
        if collisions and body.confirm_merge:
            dest_id = manual.merge_sets(source_id=set_id, dest_id=collisions[0]['id'])
            # dest_id already has the name we're renaming to — no separate
            # rename call needed, and set_id itself no longer exists.
            merged = manual.get_set(dest_id)
            return {'id': dest_id, 'name': merged['name'], 'studio': merged['studio'], 'merged': True, 'merged_from': set_id}

        # Only reached with no name collision (a merge above already returned) —
        # here it's worth also checking whether the studio being saved matches
        # an existing one under a different exact spelling, same reasoning as
        # the name check but nothing to merge in the DB (studio's just a
        # free-text column), so this only offers to reuse the existing spelling.
        if studio and not body.confirm_studio_merge:
            variant = manual.find_studio_variant(studio, exclude_set_id=set_id)
            if variant is not None:
                return JSONResponse(status_code=409, content={
                    'conflict': 'studio',
                    'existing_studio': {
                        'studio': variant['studio'], 'set_count': variant['set_count'],
                        'image_count': variant['image_count'],
                    },
                })

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
        count = manual.bump_set_favorite(set_id, body.delta)
        return {'id': set_id, 'count': count}

    @app.get('/api/sets/{set_id}/aliases')
    def api_list_set_aliases(set_id: int):
        return {'aliases': manual.get_aliases_for_set(set_id)}

    @app.post('/api/sets/{set_id}/aliases')
    def api_add_set_alias(set_id: int, body: SetAliasBody):
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        alias = (body.alias or '').strip()
        if not alias:
            raise HTTPException(status_code=400, detail='Alias must not be empty')
        manual.add_set_alias(set_id, alias)
        return {'aliases': manual.get_aliases_for_set(set_id)}

    @app.delete('/api/sets/{set_id}/aliases/{alias}')
    def api_remove_set_alias(set_id: int, alias: str):
        manual.remove_set_alias(alias)
        return {'aliases': manual.get_aliases_for_set(set_id)}

    # ------------------------------------------------------------------
    # Categories (single-value, ML-assisted classification)
    # ------------------------------------------------------------------

    def _all_categories_for_nav():
        counts = get_category_counts(manual, db)
        return sorted(
            ({'name': name, 'count': count} for name, count in counts.items()),
            key=lambda c: c['name']
        )

    def _homepage_stats():
        """The 8 homepage stat tiles. 'known_people'/'unknown_faces' distinguish
        distinct-identity count from raw unidentified-face-instance count (see
        faces_page's own 'known people' == distinct-identity precedent).
        'total_videos' is a reserved placeholder — video tracking isn't
        implemented yet, so it's always None (rendered as '—').

        'photos_without_set' is total files minus the *count* of set-member
        checksums, not a resolved file-row count — get_all_set_member_checksums()
        exists specifically so callers never need to feed its (potentially huge)
        result into an IN (...) query (see its own docstring: that's the exact
        pattern that already caused a 'too many SQL variables' crash once
        before). Treating checksum count as a stand-in for file-row count is
        the same approximation already accepted for 'total_photos' above."""
        total_files = db.count_files()
        return {
            # Cheap counts only — no whole-library row scans/joins (this runs on the
            # home page): distinct-identity/set-member/unidentified-face COUNTs instead
            # of materializing + len()-ing full result sets.
            'known_people': manual.count_identities(),
            'unknown_faces': db.count_unidentified_faces(),
            'num_sets': manual.count_sets(),
            'photos_without_set': max(0, total_files - manual.count_set_member_checksums()),
            'total_photos': total_files,
            'total_videos': None,
            'total_manual_tags': len(manual.list_all_tags()),
            'total_auto_tags': db.count_detected_classes(),
        }

    @app.get('/categories', response_class=HTMLResponse)
    def categories_page(request: Request):
        all_tags = manual.list_all_tags()
        all_categories = _all_categories_for_nav()
        # One query for the whole library instead of a get_example_checksums_for_
        # category + get_files_by_checksums round trip per category (this page is
        # unpaginated — that used to be 2 queries times every category, per view).
        category_checksums_map = manual.get_all_category_checksums()
        representative_checksum_by_category = {
            cat_id: next(iter(checksums)) for cat_id, checksums in category_checksums_map.items() if checksums
        }
        all_repr_checksums = list(representative_checksum_by_category.values())
        thumb_rows_by_checksum = {}
        if all_repr_checksums:
            thumb_rows_by_checksum = {row['checksum']: row for row in db.get_files_by_checksums(all_repr_checksums)}

        categories = []
        for r in manual.list_categories():
            checksum = representative_checksum_by_category.get(r['id'])
            thumb_row = thumb_rows_by_checksum.get(checksum) if checksum else None
            categories.append({
                'id': r['id'], 'name': r['name'], 'temperature': r['temperature'],
                'image_count': r['image_count'], 'thumb_id': thumb_row['id'] if thumb_row else None,
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

    @app.post('/api/files/{file_id}/categories')
    def api_add_file_category(file_id: int, body: CategoryBody):
        """Adds a category — a file can belong to any number of categories, so
        this never replaces an existing one (mirrors api_assign_set)."""
        row = _file_or_404(file_id)
        if body.category_id is None:
            raise HTTPException(status_code=400, detail='category_id is required')
        cat = manual.get_category(body.category_id)
        if cat is None:
            raise HTTPException(status_code=404, detail='Category not found')
        manual.add_file_category(row['checksum'], body.category_id)
        return {'category': {'id': cat['id'], 'name': cat['name'], 'source': 'manual', 'score': None}}

    @app.delete('/api/files/{file_id}/categories/{category_id}')
    def api_remove_file_category(file_id: int, category_id: int):
        """Removes just this one category from the file — any other categories
        it has are untouched."""
        row = _file_or_404(file_id)
        manual.remove_file_category(row['checksum'], category_id)
        return {'ok': True}

    @app.post('/api/files/{file_id}/categories/{category_id}/exclude')
    def api_exclude_file_category(file_id: int, category_id: int):
        """Category-suggestion swipe stream's reject action — 'confirmed NOT in
        this category', permanent, scoped to just this one category (mirrors
        exclude_file_from_set)."""
        row = _file_or_404(file_id)
        manual.exclude_file_from_category(row['checksum'], category_id)
        return {'ok': True}

    @app.delete('/api/files/{file_id}/categories/{category_id}/exclude')
    def api_undo_exclude_file_category(file_id: int, category_id: int):
        """Undo a reject made via the category-suggestion swipe stream (Ctrl+Z)."""
        row = _file_or_404(file_id)
        manual.remove_category_exclusion(row['checksum'], category_id)
        return {'ok': True}

    def _next_category_suggestions(category_id, count, exclude_refs):
        """Rank currently-unresolved-or-differently-resolved files against this ONE
        category's centroid (mirrors media_manager.py's MediaManager.match_categories
        exactly — same centroid-vs-temperature-threshold math — but interactive/
        one-at-a-time instead of batch-write). Returns up to `count` dicts, best
        score first: [{'ref': str, 'file_id': int, 'score': float}, ...].

        A file is excluded from candidacy if it already manually has THIS category,
        or has been explicitly rejected for THIS category before — categories are
        multi-valued now, so having some OTHER manual category doesn't exclude a
        file from also being suggested this one.

        A file that already has an ML auto-match to a DIFFERENT category IS still
        offered here: auto-matches are provisional machine guesses, not human
        decisions, and confirming here always writes a manual assignment that
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
        already_has_this = set(example_checksums)
        rejected_for_this = manual.get_excluded_checksums_for_category(category_id)

        filtered = [
            c for c in all_candidates
            if c[3] not in already_has_this and c[3] not in rejected_for_this
            and f"file:{c[0]}" not in exclude_refs
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

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------
    # Endpoint contract (for the templates/app.js agent):
    #   GET/POST  /api/locations
    #   PUT/DELETE /api/locations/{id}
    #   POST /api/files/{id}/locations         body {location_id}
    #   DELETE /api/files/{id}/locations/{loc}
    #   POST /api/sets/{id}/locations          body {location_id}
    #   DELETE /api/sets/{id}/locations/{loc}
    #   POST /api/studios/{id}/locations       body {location_id}   (id = studios.id integer)
    #   DELETE /api/studios/{id}/locations/{loc}
    #   POST /api/metadata/start ; GET /api/metadata/status
    #   search chip type "location" (value = location id, or "any" for "has any location metadata")
    #   sort=distance&lat=<f>&lon=<f>  (nearest-first; rows without GPS sort last)

    @app.get('/locations', response_class=HTMLResponse)
    def locations_page(request: Request):
        locations = [
            {'id': r['id'], 'name': r['name'], 'gps_lat': r['gps_lat'], 'gps_lon': r['gps_lon'],
             'file_count': r['file_count']}
            for r in manual.list_locations()
        ]
        return templates.TemplateResponse(request, 'locations.html', {
            'locations': locations,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    # NOTE: this static route must precede /locations/{location_id} so "map"
    # isn't parsed as an int location id.
    @app.get('/locations/map', response_class=HTMLResponse)
    def locations_map_page(request: Request):
        return templates.TemplateResponse(request, 'locations_map.html', {
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
        })

    @app.get('/api/locations/map-points')
    def api_locations_map_points():
        """Every non-hidden EXIF-geotagged photo as [file_id, lat, lon] — plotted
        on the offline locations map (see locations_map.html)."""
        return {'points': [[r[0], r[1], r[2]] for r in db.get_geotagged_points()]}

    @app.get('/locations/{location_id}', response_class=HTMLResponse)
    def location_detail_page(request: Request, location_id: int, sort: str = 'added', order: str = 'desc',
                             lat: float = None, lon: float = None):
        loc = manual.get_location(location_id)
        if loc is None:
            raise HTTPException(status_code=404, detail='Location not found')
        if sort == 'distance' and (lat is None or lon is None):
            sort = 'added'
        checksums = manual.get_checksums_for_location(location_id)
        file_rows = _sort_file_rows(db.get_files_by_checksums(list(checksums)), sort, order,
                                    target_lat=lat, target_lon=lon)
        rows = [(r['id'], r['path'], False, r['checksum']) for r in file_rows]
        files = _enrich_rows(rows)
        if sort == 'age':
            files = _sort_cards_by_age(files, order)
        return templates.TemplateResponse(request, 'location_detail.html', {
            'location': {'id': loc['id'], 'name': loc['name'],
                         'gps_lat': loc['gps_lat'], 'gps_lon': loc['gps_lon']},
            'files': files,
            'all_tags': manual.list_all_tags(),
            'all_categories': _all_categories_for_nav(),
            'sort': sort,
            'order': order,
        })

    @app.get('/api/locations')
    def api_list_locations():
        return [
            {'id': r['id'], 'name': r['name'], 'gps_lat': r['gps_lat'], 'gps_lon': r['gps_lon'],
             'file_count': r['file_count']}
            for r in manual.list_locations()
        ]

    @app.post('/api/locations')
    def api_create_location(body: LocationBody):
        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Location name must not be empty')
        location_id = manual.create_location(name, body.gps_lat, body.gps_lon)
        loc = manual.get_location(location_id)
        return {'id': loc['id'], 'name': loc['name'], 'gps_lat': loc['gps_lat'], 'gps_lon': loc['gps_lon']}

    @app.put('/api/locations/{location_id}')
    def api_update_location(location_id: int, body: LocationBody):
        if manual.get_location(location_id) is None:
            raise HTTPException(status_code=404, detail='Location not found')
        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail='Location name must not be empty')
            manual.rename_location(location_id, name)
        # Coordinates: a PUT that carries either coord key sets them (either may be
        # null to clear). A name-only PUT (both keys omitted -> both None) leaves
        # coords untouched, so renaming never silently wipes a location's pin.
        fields = body.model_fields_set if hasattr(body, 'model_fields_set') else set()
        if 'gps_lat' in fields or 'gps_lon' in fields:
            manual.set_location_coords(location_id, body.gps_lat, body.gps_lon)
        return {'ok': True}

    @app.delete('/api/locations/{location_id}')
    def api_delete_location(location_id: int):
        if manual.get_location(location_id) is None:
            raise HTTPException(status_code=404, detail='Location not found')
        manual.delete_location(location_id)
        return {'ok': True}

    @app.post('/api/files/{file_id}/locations')
    def api_add_file_location(file_id: int, body: LocationAssignBody):
        """Adds a location to a file — a file can carry any number of locations, so
        this never replaces an existing one (mirrors api_add_file_category)."""
        row = _file_or_404(file_id)
        manual.add_file_location(row['checksum'], body.location_id)
        return {'ok': True}

    @app.delete('/api/files/{file_id}/locations/{location_id}')
    def api_remove_file_location(file_id: int, location_id: int):
        row = _file_or_404(file_id)
        manual.remove_file_location(row['checksum'], location_id)
        return {'ok': True}

    @app.post('/api/sets/{set_id}/locations')
    def api_add_set_location(set_id: int, body: LocationAssignBody):
        manual.add_set_location(set_id, body.location_id)
        return {'ok': True}

    @app.delete('/api/sets/{set_id}/locations/{location_id}')
    def api_remove_set_location(set_id: int, location_id: int):
        manual.remove_set_location(set_id, location_id)
        return {'ok': True}

    @app.post('/api/studios/{studio_id}/locations')
    def api_add_studio_location(studio_id: int, body: LocationAssignBody):
        """studio_id is the integer studios.id (studio_locations FKs studios(id)) —
        not the free-text studio name the /api/studios/{studio} rename route keys on."""
        manual.add_studio_location(studio_id, body.location_id)
        return {'ok': True}

    @app.delete('/api/studios/{studio_id}/locations/{location_id}')
    def api_remove_studio_location(studio_id: int, location_id: int):
        manual.remove_studio_location(studio_id, location_id)
        return {'ok': True}

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
            return {'id': set_row['id'], 'name': set_row['name'], 'studio': set_row['studio'],
                    'people': manual.get_identities_for_sets([set_row['id']]).get(set_row['id'], [])}

        name = (body.name or '').strip()
        if not name:
            raise HTTPException(status_code=400, detail='Set name must not be empty')
        studio = body.studio.strip() if body.studio else None
        set_id = manual.create_set(name, studio)
        manual.assign_file_to_set(checksum, set_id)
        return {'id': set_id, 'name': name, 'studio': studio, 'people': []}

    @app.delete('/api/files/{file_id}/sets/{set_id}')
    def api_remove_set(file_id: int, set_id: int):
        row = _file_or_404(file_id)
        manual.remove_file_from_set(row['checksum'], set_id)
        return {'ok': True}

    @app.post('/api/sets/{set_id}/reorder')
    def api_reorder_set(set_id: int, body: ReorderBody):
        """Persist the manual order of a set from the full ordered list of file ids
        the set-detail page sends (its cards' data-file-id, in DOM order). Resolves
        each id to its checksum and writes dense positions via set_file_positions."""
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        checksums = []
        for fid in body.file_ids:
            row = db.get_file_by_id(fid)
            if row is not None:
                checksums.append(row['checksum'])
        manual.set_file_positions(set_id, checksums)
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
        count = manual.bump_file_favorite(row['checksum'], body.delta)
        return {'id': file_id, 'count': count}

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

        if not candidates:
            return []
        # One BLAS matmul instead of a Python loop over N faces: pack all embeddings
        # into a contiguous (N, D) matrix and score with `mat @ query_vec`. numpy's
        # matmul runs in C and RELEASES the GIL, so a large library's scan no longer
        # pins the interpreter and stalls every other request (the old per-face
        # Python `.dot()` loop held the GIL for the whole scan).
        dim = query_vec.shape[0]
        mat = np.frombuffer(b''.join(c[4] for c in candidates), dtype=np.float32).reshape(len(candidates), dim)
        scores = mat @ query_vec
        scored = [
            (float(scores[i]), candidates[i][0], candidates[i][1], candidates[i][2], candidates[i][3])
            for i in range(len(candidates)) if scores[i] >= threshold
        ]
        scored.sort(reverse=True)
        return [
            {
                'ref': ref, 'file_id': file_id_, 'path': path_,
                'filename': os.path.basename(path_), 'identity': identity,
                'score': round(score, 3),
            }
            for score, ref, file_id_, path_, identity in scored[:limit]
        ]

    def _find_similar_unknown_faces(name, threshold, limit=20, ceiling=None, ascending=False):
        """Unidentified auto-detected faces whose embedding is within `threshold` cosine
        similarity of any of `name`'s confirmed faces — powers /api/identities/{name}/similar-faces
        and the auto-calibrate-threshold feature's per-round batches/bulk-confirm.

        `ceiling`, if given, excludes score >= ceiling — scopes a calibration round to
        the still-unverified band below whatever threshold was already proven clean.
        `ascending`, when True, sorts lowest-score-first instead of the default
        best-first, so a calibration round samples the riskiest (closest to the
        threshold) candidates rather than the easy, already-confident ones.
        `limit=None` returns every qualifying row (used by the bulk-confirm endpoint).

        Returns {'results': [...limit], 'total': N} — `total` is the full count of
        qualifying rows before the limit slice, needed both for a round's "is this
        band empty" check and the final bulk-add candidate count."""
        ref_embeddings = manual.get_embeddings_for_identity(name)
        unidentified = _unpromoted_auto_faces(limit=None)
        if not ref_embeddings or not unidentified:
            return {'results': [], 'total': 0}
        import numpy as np
        ref_matrix = np.stack([np.frombuffer(e, dtype=np.float32) for e in ref_embeddings])
        scored = []
        for row in unidentified:
            face_id_, file_id_, path_, bbox_, emb_bytes = row
            if not emb_bytes:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(ref_matrix.dot(vec).max())
            if score >= threshold and (ceiling is None or score < ceiling):
                scored.append((score, face_id_, file_id_, path_))
        scored.sort(reverse=not ascending)
        sliced = scored if limit is None else scored[:limit]
        results = [
            {
                'ref': f"auto:{face_id_}",
                'file_id': file_id_,
                'path': path_,
                'filename': os.path.basename(path_),
                'score': round(score, 3),
            }
            for score, face_id_, file_id_, path_ in sliced
        ]
        return {'results': results, 'total': len(scored)}

    @app.get('/faces', response_class=HTMLResponse)
    def faces_page(request: Request, favorite: bool = False):
        all_tags = manual.list_all_tags()
        favorite_faces = []
        people = []
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
        else:
            for entry in manual.get_identity_summary():
                file_row = db.get_file_by_checksum(entry['checksum']) if entry['checksum'] else None
                if file_row is None:
                    continue
                people.append({
                    'ref': entry['ref'],
                    'file_id': file_row['id'],
                    'identity': entry['identity'],
                    'count': entry['count'],
                    'last_modified': entry['last_modified'],
                })
        return templates.TemplateResponse(request, 'faces.html', {
            'people': people,
            'people_count': len(people),
            'all_tags': all_tags,
            'all_categories': _all_categories_for_nav(),
            'favorite_only': favorite,
            'favorite_faces': favorite_faces,
        })

    @app.get('/api/faces/cleanup-queue')
    def api_faces_cleanup_queue(limit: int = 200):
        """Review queue for the "cleanup low confidence matches" feature: every
        confirmed face not yet marked reviewed_ok, ranked worst-first by its
        own max cosine similarity to the OTHER confirmed faces of the same
        identity (an outlier vs. its own identity's cohort = a likely
        mis-tag). Identities with fewer than 2 confirmed faces have nothing to
        compare against and contribute nothing. `compare_ref` is one randomly
        chosen sibling face per card, fixed for this queue build — the "peek
        at what we're comparing against" reference."""
        rows = manual.get_confirmed_faces_for_cleanup()
        by_identity = {}
        for face_id_, checksum, identity, emb_bytes, reviewed_ok in rows:
            by_identity.setdefault(identity, []).append((face_id_, checksum, emb_bytes, reviewed_ok))

        import numpy as np
        scored = []  # (score, face_id_, checksum, identity, compare_face_id)
        for identity, faces_for_identity in by_identity.items():
            if len(faces_for_identity) < 2:
                continue
            matrix = np.stack([
                np.frombuffer(emb_bytes, dtype=np.float32) for _, _, emb_bytes, _ in faces_for_identity
            ])
            for i, (face_id_, checksum, emb_bytes, reviewed_ok) in enumerate(faces_for_identity):
                if reviewed_ok:
                    continue
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                sims = matrix.dot(vec)
                sims[i] = -1.0  # exclude comparing a face against itself
                score = float(sims.max())
                # A random sibling to show as "what we're comparing against" —
                # per spec ("a random face example"), not necessarily the
                # closest match, just excluding itself.
                other_indices = [j for j in range(len(faces_for_identity)) if j != i]
                compare_face_id = faces_for_identity[random.choice(other_indices)][0]
                scored.append((score, face_id_, checksum, identity, compare_face_id))

        scored.sort(key=lambda t: t[0])
        cards = []
        for score, face_id_, checksum, identity, compare_face_id in scored[:limit]:
            file_row = db.get_file_by_checksum(checksum)
            if file_row is None:
                continue
            cards.append({
                'ref': f"manual:{face_id_}",
                'file_id': file_row['id'],
                'path': file_row['path'],
                'filename': os.path.basename(file_row['path']),
                'identity': identity,
                'score': round(score, 3),
                'compare_ref': f"manual:{compare_face_id}",
            })
        return {'cards': cards, 'total': len(scored)}

    @app.post('/api/faces/{face_id}/mark-reviewed')
    def api_mark_face_reviewed(face_id: str):
        """"Correct, leave as is, don't ask again" from the cleanup review."""
        kind, raw_id = _parse_face_ref(face_id)
        if kind != 'manual':
            raise HTTPException(status_code=400, detail='Only confirmed (manual) faces can be marked reviewed')
        if manual.get_face(raw_id) is None:
            raise HTTPException(status_code=404, detail='Face not found')
        manual.mark_face_reviewed_ok(raw_id)
        return {'ok': True}

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
        return FileResponse(crop_path, media_type='image/jpeg', headers=IMMUTABLE_CACHE_HEADERS)

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
        just one — since a name typo/correction should apply to the whole person.
        Renaming onto an exact-spelling existing identity already merges
        automatically (it's just a shared string key); renaming onto one that
        only matches case-insensitively (a likely typo, e.g. "joe" vs "Joe")
        warns first, since that silently combines two people who today read as
        distinct everywhere identity names are compared exactly."""
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail='Name must not be empty')
        if new_name != name:
            variant = manual.find_identity_variant(new_name)
            if variant is not None:
                if not body.confirm_merge:
                    return JSONResponse(status_code=409, content={
                        'conflict': 'identity',
                        'existing_identity': {'name': variant['identity'], 'count': variant['cnt']},
                    })
                # Merge onto the existing identity's actual spelling, not
                # whatever case the user typed — renaming onto the typed
                # spelling instead would leave two distinct identities
                # differing only by case rather than actually combining them.
                new_name = variant['identity']
        manual.rename_identity(name, new_name)
        return {'name': new_name}

    @app.get('/api/identities/{name}/aliases')
    def api_list_identity_aliases(name: str):
        return {'aliases': manual.get_aliases_for_identity(name)}

    @app.post('/api/identities/{name}/aliases')
    def api_add_identity_alias(name: str, body: IdentityAliasBody):
        alias = (body.alias or '').strip()
        if not alias:
            raise HTTPException(status_code=400, detail='Alias must not be empty')
        manual.add_identity_alias(name, alias)
        return {'aliases': manual.get_aliases_for_identity(name)}

    @app.delete('/api/identities/{name}/aliases/{alias}')
    def api_remove_identity_alias(name: str, alias: str):
        manual.remove_identity_alias(alias)
        return {'aliases': manual.get_aliases_for_identity(name)}

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

    @app.delete('/api/identities/{name}/exclude-set/{set_id}')
    def api_exclude_identity_from_set(name: str, set_id: int):
        """Remove a person from a set's People-present list regardless of which
        of the three sources (explicit link, whole-photo assignment, or a
        detected face on one of the set's own photos) put them there — see
        exclude_identity_from_set/get_people_present_in_set. Unlike the
        assign-set DELETE above (which only undoes an explicit link and leaves a
        detected-face-only person still showing), this is what the set page's
        single "×" on each People-present entry calls."""
        manual.exclude_identity_from_set(set_id, name)
        return {'excluded': True}

    def _identity_in_set_checksums(name, set_id):
        """Member checksums of `set_id` where `name` actually appears — via a named
        face or a whole-photo assignment. Shared by the count + scrub endpoints."""
        members = list(manual.get_files_by_set(set_id, limit=100_000_000))
        ident_map = manual.get_identities_for_checksums(members)   # {checksum: [names]}
        assigned = set(manual.get_photos_assigned_to_identity(name))
        return [cs for cs in members if name in ident_map.get(cs, []) or cs in assigned]

    @app.get('/api/identities/{name}/set-image-count/{set_id}')
    def api_identity_set_image_count(name: str, set_id: int):
        """How many of the set's images still contain this person — drives the
        'also remove from N images?' confirm on the set page."""
        return {'count': len(_identity_in_set_checksums(name, set_id))}

    @app.post('/api/identities/{name}/scrub-from-set/{set_id}')
    def api_scrub_identity_from_set(name: str, set_id: int):
        """Remove a wrongly-tagged person from a set AND from its member images:
        clear the face->identity link (reversible) and drop any whole-photo
        assignment on each affected member, then exclude them from the roster.
        NOTE: identity is library-global, so this affects those photos everywhere."""
        if manual.get_set(set_id) is None:
            raise HTTPException(status_code=404, detail='Set not found')
        assigned = set(manual.get_photos_assigned_to_identity(name))
        cleared_faces = 0
        cleared_assignments = 0
        images = 0
        for cs in _identity_in_set_checksums(name, set_id):
            touched = False
            n = manual.unassign_identity_for_checksum(cs, name) or 0
            if n:
                cleared_faces += n
                touched = True
            if cs in assigned:
                manual.remove_identity_photo_assignment(cs, name)
                cleared_assignments += 1
                touched = True
            if touched:
                images += 1
        manual.exclude_identity_from_set(set_id, name)
        return {'excluded': True, 'images': images,
                'cleared_faces': cleared_faces, 'cleared_assignments': cleared_assignments}

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
        return {'results': _find_similar_unknown_faces(name, threshold, limit=limit)['results']}

    @app.get('/api/identities/{name}/calibration-batch')
    def api_calibration_batch(name: str, floor: float = 0.10, ceiling: float = 1.0, count: int = 8):
        """Feeds the auto-calibrate-threshold feature's per-round review batch: the
        `count` riskiest (lowest-scoring) still-unreviewed candidates in [floor,
        ceiling), plus the true total count in that band (before `count` truncates
        it) so the client can tell an empty band apart from a merely-small one."""
        floor = max(0.0, min(1.0, floor))
        ceiling = max(floor, min(1.0, ceiling))
        return _find_similar_unknown_faces(name, floor, limit=count, ceiling=ceiling, ascending=True)

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

    @app.get('/face/{face_ref}', response_class=HTMLResponse)
    def face_page(request: Request, face_ref: str):
        """Landing page for a single face. A named face IS just that person, so we
        redirect to their existing /person page (the person view is the named-face
        view). An unknown face gets its own view: the crop, age/gender if known, a
        "name this face" action, a link back to its photo, and a grid of
        visually-similar faces to help decide who they are."""
        kind, raw_id = _parse_face_ref(face_ref)  # 400s on a malformed ref
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                raise HTTPException(status_code=404, detail='Face not found')
            identity = row['identity']
            file_row = db.get_file_by_checksum(row['checksum'])
            file_id = file_row['id'] if file_row is not None else None
        else:
            cursor = db.conn.cursor()
            cursor.execute('SELECT identity, file_id FROM faces WHERE id = ?', (raw_id,))
            row = cursor.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail='Face not found')
            identity = row['identity']
            file_id = row['file_id']
        if identity:
            return RedirectResponse(url=f'/person/{quote(identity)}', status_code=307)
        est = manual.get_age_estimate_for_face_ref(face_ref)
        age = est['age'] if est is not None and est['age'] is not None else None
        gender = est['gender'] if age is not None else None
        similar = _find_similar_faces_to_ref(face_ref, FACE_SIMILAR_THRESHOLD, limit=40)
        return templates.TemplateResponse(request, 'face.html', {
            'ref': face_ref,
            'file_id': file_id,
            'age': age,
            'gender': gender,
            'similar': similar,
        })

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

    def _confirm_auto_face(raw_id, name):
        """Core of confirming an auto-detected (media.db) face as `name`: looks up
        its file/bbox/embedding and promotes it into manual.db. Returns the new
        manual-face id, or None if the source auto face no longer exists. Shared
        by the single-face confirm endpoint below and the auto-calibrate-
        threshold feature's bulk-confirm-above-threshold endpoint."""
        cursor = db.conn.cursor()
        cursor.execute('SELECT file_id, bbox, embedding FROM faces WHERE id = ?', (raw_id,))
        src = cursor.fetchone()
        if src is None:
            return None
        src_file = db.get_file_by_id(src['file_id'])
        if src_file is None:
            return None
        bbox = json.loads(src['bbox'])
        return manual.promote_auto_face(raw_id, src_file['checksum'], bbox, src['embedding'], name, None, None)

    @app.post('/api/faces/{face_id}/identity')
    def api_assign_identity(face_id: str, body: IdentityBody):
        # A blank/omitted name confirms "this is a distinct person" without
        # requiring one to be typed — see manual.generate_placeholder_identity_name.
        # A placeholder behaves exactly like a real name everywhere else (rename,
        # suggestion matching, /person/{name}, ...), it's just waiting to be
        # renamed, or not.
        name = (body.name or '').strip() or manual.generate_placeholder_identity_name()
        kind, raw_id = _parse_face_ref(face_id)
        if kind == 'manual':
            row = manual.get_face(raw_id)
            if row is None:
                raise HTTPException(status_code=404, detail='Face not found')
            manual.assign_identity(raw_id, name)
            return {'face_id': face_id, 'identity': name}

        new_id = _confirm_auto_face(raw_id, name)
        if new_id is None:
            raise HTTPException(status_code=404, detail='Face not found')
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

    @app.post('/api/identities/{name}/confirm-above-threshold')
    def api_confirm_above_threshold(name: str, body: ConfirmAboveThresholdBody):
        """Bulk finish of the auto-calibrate-threshold feature: once the client's
        binary search has settled on a threshold with no observed false
        positives, confirm every remaining unreviewed candidate at or above it
        as `name` in one request, reusing the same _confirm_auto_face path the
        single-face confirm endpoint uses (no separate promotion logic)."""
        threshold = max(0.0, min(1.0, body.threshold))
        data = _find_similar_unknown_faces(name, threshold, limit=None, ascending=False)
        confirmed = 0
        for r in data['results']:
            kind, raw_id = _parse_face_ref(r['ref'])
            if kind == 'auto' and _confirm_auto_face(raw_id, name) is not None:
                confirmed += 1
        return {'confirmed': confirmed, 'threshold': threshold}

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
        count = manual.bump_face_favorite(raw_id, body.delta)
        return {'face_id': face_id, 'count': count}

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

    def _save_captured_still(parent_row, jpeg_bytes: bytes, time_ms: int):
        """Persist `jpeg_bytes` (a captured frame) as a hidden, real image file under
        captured_frames/ and link it back to `parent_row`'s content at `time_ms`.
        Returns the new file id. Idempotent per identical frame (keyed by checksum) —
        recapturing the same frame reuses the row. Shared by both capture endpoints
        (client canvas upload for videos, server-side extract for animated images)."""
        import xxhash
        checksum = xxhash.xxh64(jpeg_bytes).hexdigest()
        rel_path = 'captured_frames/' + checksum + '.jpg'
        abs_path = os.path.join(data_root, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        if not os.path.exists(abs_path):
            with open(abs_path, 'wb') as f:
                f.write(jpeg_bytes)
        new_id = db.upsert_file_path(rel_path, checksum, size=len(jpeg_bytes),
                                     modified_time=int(os.path.getmtime(abs_path)))
        db.set_file_hidden(new_id, True)   # commits (also persists the upsert above)
        manual.add_frame_capture(checksum, parent_row['checksum'], time_ms)
        return new_id

    @app.post('/api/files/{file_id}/capture-frame')
    async def api_capture_frame(file_id: int, frame: UploadFile = File(...), time_ms: int = Form(0)):
        """Save a manually-captured video frame (the canvas JPEG blob) as a hidden,
        real image file linked back to the source video at `time_ms`. Returns the new
        file id so the client can open it. Idempotent per identical frame (checksum)."""
        row = _file_or_404(file_id)
        # Videos and animated images (GIF/WEBP) both capture a frame the same way —
        # the client draws the currently-shown frame of <video>/<img> to a canvas.
        if os.path.splitext(row['path'])[1].lower() not in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS):
            raise HTTPException(status_code=400, detail='Frame capture needs a video or animated image')
        data = await frame.read()
        if not data:
            raise HTTPException(status_code=400, detail='Empty frame')
        new_id = _save_captured_still(row, data, time_ms)
        return {'file_id': new_id}

    @app.get('/api/files/{file_id}/frame/{frame_index}')
    def api_render_frame(file_id: int, frame_index: int):
        """Render one frame of an animated image (GIF/WEBP) as a JPEG. The <img>
        stepper on the photo page points its src here to show an exact, non-playing
        frame — the browser can't read the currently-displayed frame of an animation,
        so it asks the server to decode a specific one. frame_index is clamped."""
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        count = frames.get_frame_count(abs_path)
        idx = max(0, min(frame_index, count - 1))
        img = frames.extract_frame(abs_path, idx)
        if img is None:
            raise HTTPException(status_code=404, detail='Could not read that frame')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        return Response(content=buf.getvalue(), media_type='image/jpeg',
                        headers=IMMUTABLE_CACHE_HEADERS)

    @app.post('/api/files/{file_id}/capture-frame-index')
    def api_capture_frame_index(file_id: int, body: CaptureFrameIndexBody):
        """Capture a specific frame of an animated image by index — no client upload.
        The server decodes frame `body.frame_index` and saves it via the same helper
        the canvas-upload path uses. time_ms is derived from the frames' GIF/WEBP
        durations so the still gets a meaningful '@ Xs' label (0 if unavailable)."""
        row = _file_or_404(file_id)
        abs_path = _live_abs_path(file_id, row['path'])
        if abs_path is None:
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        if os.path.splitext(row['path'])[1].lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail='Frame-index capture needs an animated image')
        count = frames.get_frame_count(abs_path)
        idx = max(0, min(body.frame_index, count - 1))
        img = frames.extract_frame(abs_path, idx)
        if img is None:
            raise HTTPException(status_code=400, detail='Could not read that frame')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        time_ms = frames.frame_time_ms(abs_path, idx)
        new_id = _save_captured_still(row, buf.getvalue(), time_ms)
        return {'file_id': new_id}

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

            # Single-allocation matrix (one join + frombuffer, no per-row
            # np.stack) and a GIL-releasing numpy sort instead of a full Python
            # sorted() over every indexed face.
            matrix = np.frombuffer(
                b''.join(c[3] for c in combined), dtype=np.float32
            ).reshape(len(combined), -1)
            scores = matrix.dot(np.asarray(query_emb, dtype=np.float32))
            order = np.argsort(scores)[::-1]  # descending

            seen = {}
            results_out = []
            for i in order:
                ref, file_id, fpath, _emb, identity = combined[i]
                score = float(scores[i])
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

    @app.get('/api/worker/status')
    def worker_status(since: int = 0):
        client = _worker_client
        if client is None:
            return {'enabled': False, 'connected': False, 'address': None, 'recent': []}
        cfg = worker_config.load(data_root)
        return {
            'enabled': bool(cfg.get('enabled') and cfg.get('address')),
            'connected': client.is_available(),
            'address': cfg.get('address'),
            'recent': client.recent(since_id=since),
        }

    @app.on_event('shutdown')
    def _close_databases():
        db.close()
        errors.close()
        manual.close()

    return app


def create_app_from_env() -> FastAPI:
    """Factory for uvicorn's multi-worker mode (`media web --workers N`,
    N > 1) — uvicorn calls this with no arguments in each freshly-spawned
    worker process, so data_root has to travel via an env var rather than a
    normal function argument. Set by media.py's 'web' command before calling
    uvicorn.run(..., factory=True) with this function's import string."""
    return create_app(os.environ['MEDIA_WEB_DATA_ROOT'])
