"""
Fast directory scanner using GNU find plus pv progress bar.
Single find invocation, streamed parse, single DB transaction.
"""
import os
import subprocess
import sqlite3
import shutil

def fast_scan(root_path, db_conn, data_root, recursive=True):
    """
    Call GNU find, parse '%p|%s|%T@' lines, insert/ update DB in one go.
    Spawns 'pv' if available for a simple line-count progress bar.
    root_path: absolute directory to scan
    db_conn:   sqlite3 connection (with row_factory sqlite3.Row)
    data_root: repo-root used to produce relative paths
    recursive: if False restrict find to max-depth 1
    """
    root_path = os.path.abspath(root_path)
    depth_args = [] if recursive else ['-maxdepth', '1']

    # find command (without shell=True)
    find_cmd = ['find', root_path] + depth_args + ['-type', 'f', '-printf', '%p|%s|%T@\n']

    if shutil.which('pv'):
        # pipe through pv for live line counter
        pv_cmd = ['pv', '-l']
        # use Python to create the pipe instead of the shell
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        pv_proc   = subprocess.Popen(pv_cmd, stdin=find_proc.stdout, stdout=subprocess.PIPE, text=True)
        find_proc.stdout.close()  # allow find_proc to receive SIGPIPE if pv dies
        in_stream = pv_proc.stdout
    else:
        # plain find output
        pv_proc = None
        in_stream = subprocess.Popen(find_cmd, stdout=subprocess.PIPE, text=True).stdout

    cur = db_conn.cursor()
    batch = []

    for raw in in_stream:
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

    # single bulk transaction
    cur.executemany(
        'INSERT OR REPLACE INTO files (path, size, modified_time, checksum, last_hashed)'
        ' VALUES (?,?,?,?,?)', batch)
    db_conn.commit()

    # clean-up
    if pv_proc:
        pv_proc.wait()

    return len(batch)
