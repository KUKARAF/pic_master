"""
One-time migration: convert an old path-identity media.db (files.path UNIQUE, one row
per path) to the content-hash-identity schema (files.checksum UNIQUE, file_paths
one-to-many). Content hash is now the real identity — a move/rename can never orphan
manual.db data again, and true duplicates (same bytes at multiple paths) collapse into
one record instead of being tagged/named separately. See database.py / the session's
plan doc for the full design.

Usage (run this yourself — it never writes without --apply):
    python migrate_file_identity.py /path/to/.media                # dry run, just reports
    python migrate_file_identity.py /path/to/.media --apply         # backs up both DBs, then migrates

What it does:
  1. Reads every row in the OLD `files` table (path-identity schema).
  2. Hashes any row with a NULL checksum, resolving its path relative to .media/'s
     parent directory (the actual library root). A file that can no longer be found
     on disk (moved/deleted since the last scan, or genuinely broken) gets a synthetic,
     unique pseudo-checksum instead of being dropped — its old id, path, and every
     reference to it (embeddings/detections/faces in media.db, tags/file_sets/faces in
     manual.db) are preserved exactly as before; it just can't be deduped against
     anything, since there's no content signal for it.
  3. Groups rows by checksum. Within each group, the LOWEST existing id is canonical —
     every other table that already references that id (embeddings, detections, faces,
     and manual.db's tags/file_sets/faces) needs no rewrite for the canonical row. Only
     the non-canonical ids in a group (true old-schema path duplicates) get their
     references repointed to the canonical id.
  4. Renames the old `files` table out of the way, creates the new files/file_paths
     schema, populates it (canonical rows keep their original id), repoints every
     reference in both DBs from a duplicate's old id to its group's canonical id, then
     drops the old table.
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time

from media_manager.hasher import FileHasher


def _table_cols(conn, table):
    return {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('media_dir', help='path to the .media directory')
    parser.add_argument('--apply', action='store_true', help='actually write the migration (default: dry run)')
    args = parser.parse_args()

    media_dir = os.path.abspath(args.media_dir)
    data_root = os.path.dirname(media_dir)  # library root — .media's parent
    media_db_path = os.path.join(media_dir, 'media.db')
    manual_db_path = os.path.join(media_dir, 'manual.db')

    media_conn = sqlite3.connect(media_db_path)
    media_conn.row_factory = sqlite3.Row
    files_cols = _table_cols(media_conn, 'files')

    if 'checksum' in files_cols and 'path' not in files_cols:
        print('media.db is already on the content-hash schema — nothing to migrate.')
        return
    if 'path' not in files_cols:
        print("ERROR: files table has neither 'path' nor the expected old-schema shape — aborting.", file=sys.stderr)
        sys.exit(1)

    hasher = FileHasher(None)
    old_rows = [dict(r) for r in media_conn.execute('SELECT * FROM files')]
    print(f'{len(old_rows)} rows in the old files table.')

    checksum_of = {}  # old_id -> checksum (real or synthetic)
    unhashed_now = 0
    for row in old_rows:
        cs = row.get('checksum')
        if not cs:
            abs_path = os.path.join(data_root, row['path'])
            cs = hasher.get_xxhash(abs_path)
        if not cs:
            cs = f"unhashed:{row['id']}"  # synthetic, unique — never dedupes with anything
            unhashed_now += 1
        checksum_of[row['id']] = cs

    groups = {}  # checksum -> [old_id, ...]
    for row in old_rows:
        groups.setdefault(checksum_of[row['id']], []).append(row['id'])

    canonical_of = {}  # old_id -> canonical_id
    dupes = 0
    for cs, ids in groups.items():
        canonical = min(ids)
        for i in ids:
            canonical_of[i] = canonical
            if i != canonical:
                dupes += 1

    print(f'{len(groups)} unique checksums -> {len(groups)} content records after migration.')
    print(f'{dupes} old rows collapse into an existing canonical id (true path-level duplicates).')
    print(f'{unhashed_now} rows could not be hashed (file missing/unreadable) — kept as their own unmergeable record.')
    if dupes:
        print('\nDuplicate groups:')
        for cs, ids in groups.items():
            if len(ids) > 1:
                canonical = min(ids)
                paths = {r['id']: r['path'] for r in old_rows if r['id'] in ids}
                print(f'  checksum {cs[:16]}... canonical id={canonical} ({paths[canonical]})')
                for i in ids:
                    if i != canonical:
                        print(f'    -> merges old id={i} ({paths[i]})')

    if not args.apply:
        print('\nDry run only — nothing was written. Re-run with --apply to migrate.')
        return

    manual_conn = sqlite3.connect(manual_db_path) if os.path.exists(manual_db_path) else None

    media_backup = f'{media_db_path}.bak-{time.strftime("%Y%m%d%H%M%S")}'
    shutil.copyfile(media_db_path, media_backup)
    print(f'Backed up media.db to {media_backup}')
    manual_backup = None
    if manual_conn is not None:
        manual_backup = f'{manual_db_path}.bak-{time.strftime("%Y%m%d%H%M%S")}'
        shutil.copyfile(manual_db_path, manual_backup)
        print(f'Backed up manual.db to {manual_backup}')

    try:
        cur = media_conn.cursor()
        cur.execute('ALTER TABLE files RENAME TO files_old')

        cur.execute('''
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checksum TEXT UNIQUE NOT NULL,
                size INTEGER,
                broken INTEGER,
                taken_at INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                metadata_checked_at INTEGER,
                first_seen INTEGER NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                path TEXT UNIQUE NOT NULL,
                modified_time REAL,
                last_seen_at INTEGER NOT NULL
            )
        ''')
        cur.execute('DROP VIEW IF EXISTS files_with_path')
        cur.execute('''
            CREATE VIEW files_with_path AS
            SELECT f.*, fp.path AS path, fp.modified_time AS modified_time
            FROM files f
            JOIN file_paths fp ON fp.id = (
                SELECT fp2.id FROM file_paths fp2
                WHERE fp2.file_id = f.id
                ORDER BY fp2.last_seen_at DESC LIMIT 1
            )
        ''')

        now = int(time.time())
        for cs, ids in groups.items():
            canonical = min(ids)
            canon_row = next(r for r in old_rows if r['id'] == canonical)
            first_seen = int(canon_row['modified_time']) if canon_row.get('modified_time') else now
            cur.execute('''
                INSERT INTO files (id, checksum, size, broken, taken_at, gps_lat, gps_lon,
                                    metadata_checked_at, first_seen)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (canonical, cs, canon_row.get('size'), canon_row.get('broken'),
                  canon_row.get('taken_at'), canon_row.get('gps_lat'), canon_row.get('gps_lon'),
                  canon_row.get('metadata_checked_at'), first_seen))

        for row in old_rows:
            cur.execute('''
                INSERT INTO file_paths (file_id, path, modified_time, last_seen_at)
                VALUES (?,?,?,?)
            ''', (canonical_of[row['id']], row['path'], row.get('modified_time'), now))

        # repoint every other media.db table's file_id for duplicate (non-canonical) ids —
        # canonical rows keep their original id, so nothing needs to change for them.
        remap = {old: canon for old, canon in canonical_of.items() if old != canon}
        for old_id, canon_id in remap.items():
            for table in ('embeddings', 'detections', 'faces'):
                cur.execute(f'UPDATE {table} SET file_id = ? WHERE file_id = ?', (canon_id, old_id))

        cur.execute('DROP TABLE files_old')
        media_conn.commit()
        print(f'\nmedia.db migrated: {len(groups)} content records, {len(old_rows)} known paths.')

        if manual_conn is not None and remap:
            mcur = manual_conn.cursor()
            for old_id, canon_id in remap.items():
                for table in ('tags', 'file_sets', 'faces'):
                    mcur.execute(f'UPDATE {table} SET file_id = ? WHERE file_id = ?', (canon_id, old_id))
            manual_conn.commit()
            print(f'manual.db repointed for {len(remap)} duplicate ids.')
    except Exception:
        media_conn.rollback()
        print(f'Something went wrong — rolled back media.db changes. '
              f'Your pre-write backups are still at {media_backup}'
              + (f' and {manual_backup}' if manual_backup else '') + ' regardless.')
        raise


if __name__ == '__main__':
    main()
