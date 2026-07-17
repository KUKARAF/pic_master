"""
Second-line recovery for the fast_scan.py id-scrambling bug, for the rows
recover_manual_db.py can't fix: file_sets and tags have no embedding to match
on. This script instead uses an *older backup* of media.db (from before the
scrambling happened, or before the most recent bit of it) to recover the
original file_id -> path mapping, then re-resolves each orphaned manual.db
row to whatever id that same path has in the *current* media.db.

Usage (run this yourself — it never writes without --apply):
    python recover_by_path.py /path/to/.media /path/to/media.db.bak-XXXXXXXX
    python recover_by_path.py /path/to/.media /path/to/media.db.bak-XXXXXXXX --apply

This only works if the backup file predates (or matches) the point at which
the orphaned old file_ids were still valid. If the backup is itself already
scrambled relative to those ids, rows will simply show up as unmatched —
nothing destructive happens either way.
"""
import argparse
import shutil
import sqlite3
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('media_dir', help='path to the .media directory (current, live)')
    parser.add_argument('old_media_db', help='path to an older media.db backup')
    parser.add_argument('--apply', action='store_true', help='actually write fixes (default: dry run)')
    args = parser.parse_args()

    current_db_path = f'{args.media_dir}/media.db'
    manual_db_path = f'{args.media_dir}/manual.db'

    old_conn = sqlite3.connect(args.old_media_db)
    current_conn = sqlite3.connect(current_db_path)
    manual_conn = sqlite3.connect(manual_db_path)
    manual_conn.row_factory = sqlite3.Row

    old_id_to_path = {r[0]: r[1] for r in old_conn.execute('SELECT id, path FROM files')}
    path_to_current_id = {r[0]: r[1] for r in current_conn.execute('SELECT path, id FROM files')}
    current_file_ids = set(path_to_current_id.values())

    print(f'Backup has {len(old_id_to_path)} files, current media.db has {len(path_to_current_id)} files.')

    def resolve(old_file_id):
        """old_file_id (as stored, now-stale, in manual.db) -> current file_id, or None.
        Returns (new_id_or_None, reason) where reason explains a miss."""
        path = old_id_to_path.get(old_file_id)
        if path is None:
            return None, 'id not found in backup'
        new_id = path_to_current_id.get(path)
        if new_id is None:
            return None, f'path {path!r} not in current media.db (moved/deleted?)'
        return new_id, None

    # ---------------- tags (no uniqueness constraint — simple 1:1 fix) ----------------
    tag_rows = [dict(r) for r in manual_conn.execute('SELECT * FROM tags')]
    orphaned_tags = [t for t in tag_rows if t['file_id'] not in current_file_ids]
    tag_updates = []
    tag_misses = []
    for t in orphaned_tags:
        new_id, reason = resolve(t['file_id'])
        if new_id is not None:
            tag_updates.append((t['id'], t['file_id'], new_id))
        else:
            tag_misses.append((t['id'], t['file_id'], reason))

    print(f'\ntags: {len(tag_rows)} total, {len(orphaned_tags)} orphaned')
    print(f'  Resolvable via backup path match: {len(tag_updates)}')
    for tag_id, old_fid, new_fid in tag_updates:
        print(f'    manual.db tag id={tag_id} old_file_id={old_fid} -> new_file_id={new_fid}')
    id_not_in_backup = sum(1 for _, _, r in tag_misses if r == 'id not found in backup')
    path_missing = len(tag_misses) - id_not_in_backup
    print(f'  Not resolvable: {len(tag_misses)}  '
          f'({id_not_in_backup} whose old id isn\'t in the backup at all, '
          f'{path_missing} whose path from the backup no longer exists in current media.db)')
    for tag_id, old_fid, reason in tag_misses[:10]:
        print(f'    manual.db tag id={tag_id} old_file_id={old_fid}: {reason}')
    if len(tag_misses) > 10:
        print(f'    ... and {len(tag_misses) - 10} more')

    # ---------------- file_sets (file_id is UNIQUE — a file is in at most one set) ----
    set_rows = [dict(r) for r in manual_conn.execute('SELECT * FROM file_sets')]
    orphaned_sets = [s for s in set_rows if s['file_id'] not in current_file_ids]
    existing_set_file_ids = {s['file_id'] for s in set_rows if s['file_id'] in current_file_ids}

    set_updates = []
    set_misses = []
    set_conflicts = 0
    claimed = set()
    for s in orphaned_sets:
        new_id, reason = resolve(s['file_id'])
        if new_id is None:
            set_misses.append((s['file_id'], s['set_id'], reason))
            continue
        if new_id in existing_set_file_ids or new_id in claimed:
            # Another row (already valid, or an earlier orphan in this same run)
            # already claims this file — updating would violate file_sets' UNIQUE
            # constraint. Leave it alone rather than guess which one is right.
            set_conflicts += 1
            continue
        claimed.add(new_id)
        set_updates.append((s['file_id'], s['set_id'], new_id))

    print(f'\nfile_sets: {len(set_rows)} total, {len(orphaned_sets)} orphaned')
    print(f'  Resolvable via backup path match: {len(set_updates)}')
    for old_fid, set_id, new_fid in set_updates:
        print(f'    manual.db file_sets old_file_id={old_fid} set_id={set_id} -> new_file_id={new_fid}')
    if set_conflicts:
        print(f'  Skipped (target file already claimed by another set-assignment): {set_conflicts}')
    id_not_in_backup = sum(1 for _, _, r in set_misses if r == 'id not found in backup')
    path_missing = len(set_misses) - id_not_in_backup
    print(f'  Not resolvable: {len(set_misses)}  '
          f'({id_not_in_backup} whose old id isn\'t in the backup at all, '
          f'{path_missing} whose path from the backup no longer exists in current media.db)')
    for old_fid, set_id, reason in set_misses[:10]:
        print(f'    manual.db file_sets old_file_id={old_fid} set_id={set_id}: {reason}')
    if len(set_misses) > 10:
        print(f'    ... and {len(set_misses) - 10} more')

    if not args.apply:
        print('\nDry run only — nothing was written. Re-run with --apply to write these fixes.')
        return

    if not tag_updates and not set_updates:
        print('\nNothing to apply.')
        return

    backup_path = f'{manual_db_path}.bak-{time.strftime("%Y%m%d%H%M%S")}'
    shutil.copyfile(manual_db_path, backup_path)
    print(f'\nBacked up manual.db to {backup_path}')

    try:
        for tag_id, old_fid, new_fid in tag_updates:
            manual_conn.execute('UPDATE tags SET file_id = ? WHERE id = ?', (new_fid, tag_id))
        for old_fid, set_id, new_fid in set_updates:
            manual_conn.execute(
                'UPDATE file_sets SET file_id = ? WHERE file_id = ? AND set_id = ?',
                (new_fid, old_fid, set_id)
            )
        manual_conn.commit()
    except Exception:
        manual_conn.rollback()
        print('Something went wrong — rolled back, manual.db is unchanged. '
              f'Your pre-write backup is still at {backup_path} regardless.')
        raise
    print(f'Applied {len(tag_updates)} tag fixes and {len(set_updates)} file_sets fixes to manual.db.')


if __name__ == '__main__':
    main()
