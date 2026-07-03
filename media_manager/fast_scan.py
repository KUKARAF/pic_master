"""
Fast directory scanner using GNU find plus pv progress-bar and git-style ignore filtering.
Single find invocation, streamed parse, single DB transaction.
"""
import os
import subprocess
import sqlite3
import shutil
import sys
from .ignore import IgnoreRules

def fast_scan(root_path, db_conn, data_root, recursive=True):
    """
    Call GNU find, parse '%p|%s|%T@\n' lines, insert / update DB in one go.
    Reads .mediaignore (git-ignore syntax) in repo-root if present.
    root_path: absolute directory to scan  (must exist)
    db_conn:   sqlite3 connection (with row_factory sqlite3.Row)
    data_root: repo-root used to produce relative paths
    recursive: if False restrict find to max-depth 1
    """
    root_path = os.path.abspath(root_path)
    depth_args = [] if recursive else ['-maxdepth', '1']

    # load ignore rules once
    rules = IgnoreRules(data_root)

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

    cur = db_conn.cursor()
    batch = []

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

        rel_path = os.path.relpath(full_path, data_root).replace(os.sep, '/')
        # respect .mediaignore
        if rules.is_ignored(rel_path):
            continue

        batch.append((rel_path, size, mtime, None, None, None))

    exit_code = find_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"find exited with code {exit_code}")

    # single bulk transaction
    cur.executemany(
        'INSERT OR REPLACE INTO files (path, size, modified_time, checksum, last_hashed, broken)'
        ' VALUES (?,?,?,?,?,?)', batch)
    db_conn.commit()

    if shutil.which('pv'):
        pv_proc.wait()
    return len(batch)
