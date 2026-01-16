"""
Fast directory scanner using GNU find plus pv progress bar.
Single find invocation, streamed parse, single DB transaction.
"""
import os
import subprocess
import sqlite3
import shutil
import sys

def fast_scan(root_path, db_conn, data_root, recursive=True):
    """
    Call GNU find, parse '%p|%s|%T@' lines, insert/ update DB in one go.
    Spawns 'pv' if available for a simple line-count progress bar.
    root_path: absolute directory to scan  (must exist)
    db_conn:   sqlite3 connection (with row_factory sqlite3.Row)
    data_root: repo-root used to produce relative paths
    recursive: if False restrict find to max-depth 1
    """
    root_path = os.path.abspath(root_path)
    depth_args = [] if recursive else ['-maxdepth', '1']

    # Build find command
    find_cmd = ['find', root_path] + depth_args + ['-type', 'f', '-printf', '%p|%s|%T@\n']

    # Optional pv pipe
    if shutil.which('pv'):
        # Run find -> pv -> our parser
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        pv_proc   = subprocess.Popen(['pv', '-l'],
                                      stdin=find_proc.stdout,
                                      stdout=subprocess.PIPE,
                                      text=True)
        find_proc.stdout.close()          # let find receive SIGPIPE if pv dies
        stream = pv_proc.stdout
    else:
        # Plain find
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE, text=True)
        stream = find_proc.stdout

    cur   = db_conn.cursor()
    batch = []

    for raw in stream:
        raw = raw.rstrip('\n')
        if not raw:
            continue
        try:
            full_path, str_size, str_mtime = raw.split('|', 2)
            size  = int(str_size)
            mtime = float(str_mtime)
        except ValueError as e:
            # noisy failure: tell user we dropped a line
            print(f"fast_scan: bad line '{raw[:80]}...'  ({e})", file=sys.stderr)
            continue

        rel_path = os.path.relpath(full_path, data_root)
        batch.append((rel_path, size, mtime, None, None))

    # make sure find exited successfully
    exit_code = find_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"find exited with code {exit_code}")

    # single bulk transaction
    cur.executemany(
        'INSERT OR REPLACE INTO files (path, size, modified_time, checksum, last_hashed)'
        ' VALUES (?,?,?,?,?)', batch)
    db_conn.commit()

    # reap pv if used
    if shutil.which('pv'):
        pv_proc.wait()

    return len(batch)
