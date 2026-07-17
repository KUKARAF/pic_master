"""
Fast directory scanner using GNU find plus pv progress-bar and git-style ignore filtering.
Single find invocation, streamed parse, parallel content-hash, single DB transaction.
"""
import os
import subprocess
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from .ignore import IgnoreRules
from .formats import IMAGE_EXTENSIONS
from .hasher import FileHasher

def fast_scan(root_path, db, data_root, recursive=True, max_workers=8, dup_report_path=None, reindex=False):
    """
    Call GNU find, parse '%p|%s|%T@\n' lines, hash each candidate, upsert into DB.
    Reads .mediaignore (git-ignore syntax) in repo-root if present.
    root_path:       absolute directory to scan  (must exist)
    db:              Database instance (content-addressable schema — see database.py)
    data_root:       repo-root used to produce relative paths
    recursive:       if False restrict find to max-depth 1
    max_workers:     parallel hashing workers (this now does real per-file I/O, unlike
                     the old metadata-only scan, so it's parallelized like broken_finder.py)
    dup_report_path: where to write newline-separated 'new_path\\texisting_path' lines
                     for content found at a path that isn't already tracked for it (see
                     below). Defaults to duplicates.txt in the current working directory.
    reindex:         if True, files found already-tracked at the same path (see
                     already_indexed below) have their primary ML data (detections,
                     faces, frame-0 embedding) cleared, so the next index/embed/faces
                     run reprocesses them instead of skipping them as already-done.
    Returns (written_count, duplicate_count, already_indexed_count).
    """
    root_path = os.path.abspath(root_path)
    depth_args = [] if recursive else ['-maxdepth', '1']

    # load ignore rules once
    rules = IgnoreRules(data_root)
    hasher = FileHasher(db)

    # Build find command
    find_cmd = ['find', root_path] + depth_args + ['-type', 'f', '-printf', '%p|%s|%T@\n']

    # optional pv pipe
    if shutil.which('pv'):
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        pv_proc = subprocess.Popen(['pv', '-l'],
                                     stdin=find_proc.stdout,
                                     stdout=subprocess.PIPE, text=True)
        find_proc.stdout.close()
        stream = pv_proc.stdout
    else:
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE, text=True)
        stream = find_proc.stdout

    candidates = []  # (rel_path, abs_path, size, mtime)

    for raw in stream:
        raw = raw.rstrip('\n')
        if not raw:
            continue
        try:
            full_path, str_size, str_mtime = raw.split('|', 2)
            size = int(str_size)
            mtime = float(str_mtime)
        except ValueError as e:
            print(f"fast_scan: bad line '{raw[:80]}...'  ({e})", file=sys.stderr)
            continue

        ext = os.path.splitext(full_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        rel_path = os.path.relpath(full_path, data_root).replace(os.sep, '/')
        # .media/ holds our own cache/db files (thumbnails, face crops, media.db, ...) —
        # never treat it as photo content, regardless of .mediaignore.
        if rel_path == '.media' or rel_path.startswith('.media/'):
            continue
        # respect .mediaignore
        if rules.is_ignored(rel_path):
            continue

        candidates.append((rel_path, full_path, size, mtime))

    exit_code = find_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"find exited with code {exit_code}")

    # Content hash is the file's identity now (see database.py's files/file_paths split) —
    # every candidate has to be hashed before we can write anything, since the hash is
    # what tells us whether this is new content, known content at a new path, or an
    # already-known path whose content changed in place. Parallelized since this is now
    # real per-file I/O, not just a metadata stat.
    def _hash(entry):
        rel_path, abs_path, size, mtime = entry
        return rel_path, size, mtime, hasher.get_xxhash(abs_path)

    hashed = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_hash, c) for c in candidates]
        for fut in as_completed(futures):
            rel_path, size, mtime, checksum = fut.result()
            if checksum is None:
                # unreadable (permissions, vanished mid-scan, etc.) — skip, matching the
                # old scanner's silent OSError handling
                continue
            hashed.append((rel_path, size, mtime, checksum))

    # A checksum that already belongs to a *different* known path is a genuine
    # duplicate discovery, not a routine rescan — don't silently start tracking a
    # second location for it. Report it and leave the DB untouched for that path
    # instead, so duplicates get reviewed (and deleted, if that's the call) by a
    # human rather than quietly accumulating as extra tracked copies.
    #
    # A checksum that already belongs to *this same path* isn't a duplicate at all —
    # it's just the same file being rescanned. Count it separately as
    # "already indexed" instead of lumping it in with genuinely new files.
    hashed.sort()  # deterministic order when two brand-new files in this same scan collide
    duplicates = []
    written = 0
    already_indexed = 0
    for rel_path, size, mtime, checksum in hashed:
        existing_file_id = db.find_file_id_by_checksum(checksum)
        if existing_file_id is not None:
            known_paths = [p for p, _ in db.get_paths_for_file(existing_file_id)]
            if rel_path not in known_paths:
                duplicates.append((rel_path, known_paths[0] if known_paths else '(unknown)'))
                continue
            already_indexed += 1
            db.upsert_file_path(rel_path, checksum, size=size, modified_time=mtime)
            if reindex:
                db.clear_primary_ml_data(existing_file_id)
            continue
        db.upsert_file_path(rel_path, checksum, size=size, modified_time=mtime)
        written += 1
    db.conn.commit()

    if duplicates:
        report_path = dup_report_path or os.path.join(os.getcwd(), 'duplicates.txt')
        with open(report_path, 'a') as f:
            for new_path, existing_path in duplicates:
                f.write(f"{new_path}\t{existing_path}\n")
        print(
            f"fast_scan: {len(duplicates)} duplicate file(s) found and NOT tracked — "
            f"see {report_path}",
            file=sys.stderr,
        )

    if shutil.which('pv'):
        pv_proc.wait()
    return written, len(duplicates), already_indexed
