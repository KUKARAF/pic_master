"""
Per-folder marker files that drive bulk metadata application at scan time,
mirroring .mediaignore's "drop a file in a folder, it takes effect" pattern:

  .noface       (empty)  — assume there are no faces anywhere under this
                            folder; `media faces` never runs detection on
                            these files at all (see database.py's
                            set_noface_for_checksums / get_unface_indexed_files).
  .categories   (text)   — newline-separated category names (same blank-line/
                            #-comment convention as .mediaignore); every one
                            applies to every currently-tracked file under this
                            folder, recursively, creating the category if it
                            doesn't exist yet.
  .set          (text)   — first line is a set name; every following line is
                            "relative/path[ hash]" (path relative to the
                            folder the .set file itself is in); each resolves
                            to a tracked file and is added to that set.

Discovery (`discover_folder_markers`) and application (`apply_folder_markers`)
are separate steps so MediaManager.start_scan can call them once, right after
fast_scan has already tracked every file for this scan — see start_scan.
"""
import os
import subprocess


def discover_folder_markers(root_path):
    """One cheap `find` pass (mirrors fast_scan.py's own find invocation) for
    every .noface/.categories/.set file under root_path. Returns
    {dir_abs_path: {'noface': bool, 'categories': [str, ...],
                     'set': {'name': str, 'entries': [(relpath, hash_or_None), ...]} | None}}.
    """
    root_path = os.path.abspath(root_path)
    find_cmd = [
        'find', root_path, '-type', 'f', '(',
        '-name', '.noface', '-o', '-name', '.categories', '-o', '-name', '.set',
        ')',
    ]
    try:
        result = subprocess.run(find_cmd, stdout=subprocess.PIPE, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    markers = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        dir_path = os.path.dirname(line)
        name = os.path.basename(line)
        entry = markers.setdefault(dir_path, {'noface': False, 'categories': [], 'set': None})
        if name == '.noface':
            entry['noface'] = True
        elif name == '.categories':
            entry['categories'] = _read_lines(line)
        elif name == '.set':
            lines = _read_lines(line)
            if not lines:
                continue
            set_entries = []
            for raw in lines[1:]:
                parts = raw.split(None, 1)
                relpath = parts[0]
                file_hash = parts[1].strip() if len(parts) > 1 else None
                set_entries.append((relpath, file_hash))
            entry['set'] = {'name': lines[0], 'entries': set_entries}
    return markers


def _read_lines(path):
    """Same blank-line/#-comment convention as .mediaignore (ignore.py)."""
    with open(path, encoding='utf-8') as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith('#')]


def apply_folder_markers(db, manual, data_root, markers):
    """Applies every discovered marker (idempotent — safe to call on every
    scan). Returns a list of warning strings (e.g. unresolved .set entries)
    for the caller to print — a bad individual line is reported, not raised,
    so one typo doesn't abort an entire scan.

    A marker sitting directly at data_root itself (rel_dir == '') is skipped
    for .noface/.categories — find_files_under_folder requires a real
    sub-path — but a .set at the root still works (its own entries are
    resolved directly, independent of that helper)."""
    warnings = []
    for dir_path, marker in markers.items():
        rel_dir = os.path.relpath(dir_path, data_root).replace(os.sep, '/')
        if rel_dir == '.':
            rel_dir = ''

        if (marker['noface'] or marker['categories']) and rel_dir:
            matches = db.find_files_under_folder(rel_dir)
            checksums = [m[2] for m in matches]
            if marker['noface'] and checksums:
                db.set_noface_for_checksums(checksums)
            for cat_name in marker['categories']:
                cat_id = manual.create_category(cat_name)
                for checksum in checksums:
                    manual.add_file_category(checksum, cat_id)

        if marker['set']:
            set_info = marker['set']
            set_id = manual.create_set(set_info['name'])
            for relpath, file_hash in set_info['entries']:
                checksum = None
                if file_hash and db.find_file_id_by_checksum(file_hash) is not None:
                    checksum = file_hash
                if checksum is None:
                    full_relpath = os.path.normpath(os.path.join(rel_dir, relpath)).replace(os.sep, '/')
                    row = db.get_file_by_path(full_relpath)
                    if row is not None:
                        checksum = row['checksum']
                if checksum is None:
                    detail = f" (hash {file_hash})" if file_hash else ''
                    warnings.append(f"{dir_path}/.set: could not resolve '{relpath}'{detail}")
                    continue
                manual.assign_file_to_set(checksum, set_id)

    return warnings
