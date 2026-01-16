"""
Concurrent broken media checker for images and videos.
Sets 'broken' column in DB to current Unix timestamp if file is corrupted.
"""
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import cv2

# recognised extensions (case-insensitive)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def verify_image(path):
    """Return True if image file is healthy, else False."""
    try:
        with Image.open(path) as img:
            img.verify()      # check header
        with Image.open(path) as img:
            img.load()          # actually decode pixel data
        return True
    except Exception:
        return False

def verify_video(path):
    """Return True if video file can be opened and at least one frame read."""
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False

def find_broken(root_path, db_conn, data_root, max_workers=8):
    """
    Walk root_path, verify images/videos, update DB.broken column.
    root_path: absolute path to scan (must exist)
    db_conn: sqlite3 connection
    data_root: repo root used for relative path storage
    max_workers: concurrent verification jobs
    Returns number of newly-detected broken files.
    """
    root_path = Path(root_path).resolve()
    if not root_path.exists():
        raise RuntimeError(f"scan path does not exist: {root_path}")

    all_files = [p for p in root_path.rglob('*') if p.is_file()]
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
                healthy = fut.result()
            except Exception:
                healthy = False     # treat worker crash as failure

            if not healthy:
                rel_path = os.path.relpath(str(path), data_root)
                now = int(time.time())
                # mark broken (timestamp) or leave NULL if healthy
                cur.execute('UPDATE files SET broken=? WHERE path=?', (now, rel_path))
                if cur.rowcount:      # row actually existed
                    broken_count += 1

    db_conn.commit()
    return broken_count
