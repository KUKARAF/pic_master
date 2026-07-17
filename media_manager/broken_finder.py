"""
Concurrent broken media checker for images and videos.
Sets 'broken' column in DB to current Unix timestamp if file is corrupted.
"""
import os
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import cv2

# recognised extensions (case-insensitive)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def verify_image(path):
    """Return (healthy, message). message is a non-fatal decoder warning
    (e.g. corrupt-but-recoverable JPEG data) when healthy=True, or the
    exception text when healthy=False. message is None when there's nothing
    to report."""
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
    except Exception as exc:
        return False, str(exc)

def verify_video(path):
    """Return (healthy, message)."""
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, 'could not open video'
        ret, _ = cap.read()
        cap.release()
        return (True, None) if ret else (False, 'could not read a frame')
    except Exception as exc:
        return False, str(exc)

def find_broken(root_path, db_conn, data_root, max_workers=8, error_log=None):
    """
    Walk root_path, verify images/videos, update DB.broken column.
    root_path: absolute path to scan (must exist)
    db_conn: sqlite3 connection
    data_root: repo root used for relative path storage
    max_workers: concurrent verification jobs
    error_log: optional ErrorLog — hard failures and non-fatal decoder
      warnings are both logged there (path relative to data_root)
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
            except Exception as exc:
                healthy, message = False, str(exc)     # treat worker crash as failure

            rel_path = os.path.relpath(str(path), data_root)

            if not healthy:
                now = int(time.time())
                # mark broken (timestamp) or leave NULL if healthy
                cur.execute('UPDATE files SET broken=? WHERE path=?', (now, rel_path))
                if cur.rowcount:      # row actually existed
                    broken_count += 1

            if message and error_log is not None:
                error_log.log(rel_path, message)

    db_conn.commit()
    return broken_count
