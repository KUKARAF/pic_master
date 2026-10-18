"""
Concurrent broken media checker for images and videos.
Sets 'broken' column in DB to current Unix timestamp if file is corrupted.

Only a DECODER error marks a file broken. An infrastructure failure on our side
(fd exhaustion, I/O error, permissions, file vanished mid-scan, full disk, OOM) or
an unexpected exception says nothing about the file's bytes — those are logged and
skipped WITHOUT marking, so the file is simply re-checked on the next scan.
"""
import errno
import os
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError
import cv2

# recognised extensions (case-insensitive)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# OSError errnos that mean OUR side failed (fd exhaustion, I/O, permissions, the file
# vanishing mid-scan, a full disk) — never evidence that the file's bytes are bad.
INFRA_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.EIO, errno.EACCES,
                          errno.EPERM, errno.ENOENT, errno.ENOSPC})

# What a decoder raises on genuinely bad data. PIL reports truncated/corrupt streams
# as a bare OSError (errno None), hence OSError here — minus the INFRA_ERRNOS ones.
# (cv2.error guarded: headless tests stub cv2 with a MagicMock, not a type.)
_CV2_ERROR = cv2.error if isinstance(getattr(cv2, 'error', None), type) else ()
_DECODE_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError,
                  ValueError, EOFError, OSError, _CV2_ERROR)


def is_infra_error(exc):
    """True when `exc` is an our-side failure (see INFRA_ERRNOS, plus MemoryError)."""
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, OSError) and exc.errno in INFRA_ERRNOS


def is_decode_error(exc):
    """True when `exc` means the file's data itself is bad (→ mark it damaged).
    Infrastructure errors and anything unexpected (likely a bug) return False."""
    return isinstance(exc, _DECODE_ERRORS) and not is_infra_error(exc)


def _classify(exc):
    """(healthy, message) for an exception raised while checking a file: False =
    decoder says broken, None = couldn't check (infra/unexpected; don't mark)."""
    if is_decode_error(exc):
        return False, str(exc)
    return None, f'could not check ({type(exc).__name__}): {exc}'


def verify_image(path):
    """Return (healthy, message). healthy is True (decodes; message is a non-fatal
    decoder warning such as corrupt-but-recoverable JPEG data, or None), False
    (decoder error — message is its text), or None (couldn't check: an infra or
    unexpected error, see is_decode_error — the caller must NOT mark it broken)."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with Image.open(path) as img:
                img.verify()      # check header
            with Image.open(path) as img:
                img.load()          # actually decode pixel data
        if caught:
            return True, '; '.join(str(w.message) for w in caught)
        return True, None
    except Exception as exc:  # hook-allow: broad — classified: decode → broken, else skip
        return _classify(exc)

def verify_video(path):
    """Return (healthy, message) — same tri-state contract as verify_image."""
    try:
        # Deliberately NOT routed through video_decode.open_capture: a corruption probe
        # needs deterministic software decode. HW decoders (VAAPI/QSV) may mask or
        # differently surface corruption, and this path's whole job is to detect broken
        # files — so it must use the plain, predictable FFmpeg software decoder.
        cap = cv2.VideoCapture(str(path))
        # release() inside finally so an un-openable/corrupt file (isOpened() False but
        # a partial FFmpeg handle held) never leaks its fd on the early return.
        try:
            if not cap.isOpened():
                return False, 'could not open video'
            ret, _ = cap.read()
            return (True, None) if ret else (False, 'could not read a frame')
        finally:
            cap.release()
    except Exception as exc:  # hook-allow: broad — classified: decode → broken, else skip
        return _classify(exc)

def find_broken(root_path, db_conn, data_root, max_workers=8, error_log=None):
    """
    Walk root_path, verify images/videos, update DB.broken column.
    root_path: absolute path to scan (must exist)
    db_conn: sqlite3 connection
    data_root: repo root used for relative path storage
    max_workers: concurrent verification jobs
    error_log: optional ErrorLog — hard failures, non-fatal decoder warnings and
      could-not-check skips are all logged there (path relative to data_root)
    Returns number of newly-detected broken files.
    """
    root_path = Path(root_path).resolve()
    if not root_path.exists():
        raise RuntimeError(f"scan path does not exist: {root_path}")

    # .media/ holds our own cache/db files (thumbnails, face crops, media.db, ...) —
    # never treat it as photo/video content to verify.
    all_files = [
        p for p in root_path.rglob('*')
        if p.is_file() and '.media' not in p.relative_to(root_path).parts
    ]
    todo = []
    for p in all_files:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            todo.append((p, verify_image))
        elif ext in VIDEO_EXTS:
            todo.append((p, verify_video))

    cur = db_conn.cursor()
    broken_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # submit all verifications
        futures = {exe.submit(checker, path): (path, checker) for path, checker in todo}
        for fut in as_completed(futures):
            path, checker = futures[fut]
            try:
                healthy, message = fut.result()
            except Exception as exc:  # hook-allow: broad — checker crash is our bug, not the file's
                # The checkers classify their own errors, so anything escaping is a
                # bug on our side: log and skip (retried next scan), never mark.
                healthy, message = None, f'checker crashed ({type(exc).__name__}): {exc}'

            rel_path = os.path.relpath(str(path), data_root)

            if healthy is False:
                now = int(time.time())
                # mark broken (timestamp) or leave NULL if healthy. `broken` lives on
                # the content row (files), keyed by checksum; paths live in file_paths,
                # so resolve this path to its file_id rather than filtering files by a
                # `path` column it doesn't have.
                cur.execute(
                    'UPDATE files SET broken=? WHERE id=(SELECT file_id FROM file_paths WHERE path=?)',
                    (now, rel_path))
                if cur.rowcount:      # row actually existed
                    broken_count += 1

            if message and error_log is not None:
                error_log.log(rel_path, message)

    db_conn.commit()
    return broken_count
