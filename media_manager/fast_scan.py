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
from .formats import MEDIA_EXTENSIONS
from .hasher import FileHasher

def fast_scan(root_path, db, data_root, recursive=True, max_workers=8, dup_report_path=None, reindex=False, manual=None,
              *, progress=None, use_pv=True, on_duplicates=None):
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
                     run reprocesses them instead of skipping them as already-done —
                     except faces for a checksum with any manual.db face decision
                     already on it (see manual param), which are left alone.
    manual:          ManualDB instance, used only when reindex=True to look up which
                     checksums already have a human face decision and must keep their
                     existing face detections untouched by the clear below. Optional —
                     omitting it (or passing reindex=False) just means nothing is
                     locked, matching the old unconditional-clear behavior.
    progress:        optional callable(dict) for callers that show their own progress
                     (the web's "Track this folder" job). Called with partial updates
                     of {'phase': 'listing'|'hashing'|'saving', 'done', 'total',
                     'ignored', 'added', 'already', 'duplicates', 'relinked'}. None (the CLI) means
                     no calls at all — the scan behaves exactly as without it.
    use_pv:          pipe find through `pv -l` when pv is installed (the CLI's progress
                     bar). The web job passes False: no pv subprocess, the callback is
                     its progress source.
    on_duplicates:   optional callable(list of (new_path, existing_path, checksum)) that
                     takes the duplicates instead of the dup_report_path file (the web
                     job puts them in the path-level trash). None = write the file.
    Returns (written_count, duplicate_count, already_indexed_count).
    """
    root_path = os.path.abspath(root_path)
    depth_args = [] if recursive else ['-maxdepth', '1']

    # load ignore rules once
    rules = IgnoreRules(data_root)
    hasher = FileHasher(db)

    # Build find command
    find_cmd = ['find', root_path] + depth_args + ['-type', 'f', '-printf', '%p|%s|%T@\n']

    def _report(**fields):
        if progress is not None:
            progress(fields)

    # optional pv pipe
    pv_proc = None
    if use_pv and shutil.which('pv'):
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
    listed = 0
    ignored = 0  # media-extension files skipped as system dirs or by .mediaignore
    _report(phase='listing', done=0, total=0, ignored=0)

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

        listed += 1
        if progress is not None and listed % 500 == 0:
            _report(done=listed, ignored=ignored)
        ext = os.path.splitext(full_path)[1].lower()
        if ext not in MEDIA_EXTENSIONS:
            continue

        rel_path = os.path.relpath(full_path, data_root).replace(os.sep, '/')
        # .media/ holds our own cache/db files (thumbnails, face crops, media.db, ...) —
        # never treat it as photo content, regardless of .mediaignore.
        if rel_path == '.media' or rel_path.startswith('.media/'):
            ignored += 1
            continue
        # captured_frames/ holds manually-grabbed video stills we register ourselves
        # (as hidden files) — skip so a re-scan never re-adds them as normal photos.
        if rel_path == 'captured_frames' or rel_path.startswith('captured_frames/'):
            ignored += 1
            continue
        # generated/ holds app-produced artifacts (collages, slideshows, and later
        # AI-generated images/video). They are deliberately kept OUT of the library —
        # tracked separately in manual.db's generated_artifacts registry and browsed
        # from their own view — so a scan must never fold them back in as originals.
        if rel_path == 'generated' or rel_path.startswith('generated/'):
            ignored += 1
            continue
        # respect .mediaignore
        if rules.is_ignored(rel_path):
            ignored += 1
            continue

        candidates.append((rel_path, full_path, size, mtime))

    exit_code = find_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"find exited with code {exit_code}")
    _report(phase='hashing', done=0, total=len(candidates), ignored=ignored)

    # Content hash is the file's identity now (see database.py's files/file_paths split) —
    # every candidate has to be hashed before we can write anything, since the hash is
    # what tells us whether this is new content, known content at a new path, or an
    # already-known path whose content changed in place. Parallelized since this is now
    # real per-file I/O, not just a metadata stat.
    def _hash(entry):
        rel_path, abs_path, size, mtime = entry
        return rel_path, size, mtime, hasher.get_xxhash(abs_path)

    hashed = []
    hashed_count = 0  # completed hashes, readable or not
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_hash, c) for c in candidates]
        for fut in as_completed(futures):
            rel_path, size, mtime, checksum = fut.result()
            hashed_count += 1
            if progress is not None:
                _report(done=hashed_count)
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
    # Exception: when the tracked copy of that content is flagged damaged, or none of its
    # tracked paths exist on disk any more, the new copy is NOT a duplicate to set aside —
    # it's the way back to that content. It gets tracked ("relinked") and the damaged flag
    # is cleared so the next thumbnail/scan re-checks the bytes (re-flagging them if the
    # damage is really in the content).
    hashed.sort()  # deterministic order when two brand-new files in this same scan collide
    duplicates = []
    relinked = 0
    written = 0
    already_indexed = 0
    # Computed once up front, not per file — a big rescan shouldn't turn into an
    # extra manual.db query per already-tracked file just to check this.
    locked_checksums = manual.get_all_checksums_with_manual_face_decision() if (reindex and manual is not None) else set()
    _report(phase='saving', done=0, total=len(hashed), added=0, already=0, duplicates=0)
    for rel_path, size, mtime, checksum in hashed:
        if progress is not None:
            _report(done=written + already_indexed + len(duplicates) + relinked,
                    added=written, already=already_indexed, duplicates=len(duplicates))
        existing_file_id = db.find_file_id_by_checksum(checksum)
        if existing_file_id is not None:
            known_paths = [p for p, _ in db.get_paths_for_file(existing_file_id)]
            if rel_path not in known_paths:
                existing = db.get_file_by_id(existing_file_id)
                damaged = existing is not None and existing['broken'] is not None
                gone = not any(os.path.isfile(os.path.join(data_root, p)) for p in known_paths)
                if damaged or gone:
                    db.upsert_file_path(rel_path, checksum, size=size, modified_time=mtime)
                    db.conn.execute('UPDATE files SET broken = NULL WHERE id = ?', (existing_file_id,))
                    relinked += 1
                    continue
                duplicates.append((rel_path, known_paths[0] if known_paths else '(unknown)', checksum))
                continue
            already_indexed += 1
            db.upsert_file_path(rel_path, checksum, size=size, modified_time=mtime)
            if reindex:
                db.clear_primary_ml_data(existing_file_id, skip_faces=checksum in locked_checksums)
            continue
        db.upsert_file_path(rel_path, checksum, size=size, modified_time=mtime)
        written += 1
    db.conn.commit()
    _report(done=len(hashed), added=written, already=already_indexed, duplicates=len(duplicates),
            relinked=relinked)
    if relinked:
        print(f"fast_scan: {relinked} file(s) relinked — their tracked copy was damaged or gone "
              f"from disk, so this copy is tracked now", file=sys.stderr)

    if duplicates and on_duplicates is not None:
        on_duplicates(duplicates)
    elif duplicates:
        report_path = dup_report_path or os.path.join(os.getcwd(), 'duplicates.txt')
        with open(report_path, 'a') as f:
            for new_path, existing_path, _checksum in duplicates:
                f.write(f"{new_path}\t{existing_path}\n")
        print(
            f"fast_scan: {len(duplicates)} duplicate file(s) found and NOT tracked — "
            f"see {report_path}",
            file=sys.stderr,
        )

    if pv_proc is not None:
        pv_proc.wait()
    return written, len(duplicates), already_indexed
