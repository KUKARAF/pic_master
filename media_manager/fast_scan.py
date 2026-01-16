"""
Fast directory scanner using GNU find.
Single find invocation, streamed parse, single DB transaction.
"""
import os
import subprocess
import sqlite3

def fast_scan(root_path, db_conn, data_root, recursive=True):
    """
    Call GNU find, parse '%p|%s|%T@\n' lines, insert / update DB in one go.
    root_path: absolute directory to scan
    db_conn:   sqlite3 connection (with row_factory sqlite3.Row)
    data_root: repo-root used to produce relative paths
    recursive: if False restrict find to max-depth 1
    """
    root_path = os.path.abspath(root_path)
    if recursive:
        depth_args = []
    else:
        depth_args = ['-maxdepth', '1']

    cmd = (['find', root_path] + depth_args +
           ['-type', 'f', '-printf', '%p|%s|%T@\n'])

    cur = db_conn.cursor()
    # collect all rows first (avoids many tiny commits)
    batch = []

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for raw in proc.stdout:
            raw = raw.rstrip('\n')
            if not raw:
                continue
            try:
                full_path, size, mtime = raw.split('|', 2)
            except ValueError:
                continue
            rel_path = os.path.relpath(full_path, data_root)
            size = int(size)
            mtime = float(mtime)
            batch.append((rel_path, size, mtime, None, None))

    # single transaction
    cur.executemany(
        'INSERT OR REPLACE INTO files (path, size, modified_time, checksum, last_hashed)'
        ' VALUES (?,?,?,?,?)', batch)
    db_conn.commit()
    return len(batch)
