"""
One-off recovery script for the fast_scan.py "INSERT OR REPLACE" bug (now fixed),
which reassigned files.id on every re-scan and orphaned manual.db's references to it.

Usage (run this yourself — it never writes without --apply):
    python recover_manual_db.py /path/to/.media                # dry run, just reports
    python recover_manual_db.py /path/to/.media --apply         # backs up manual.db, then fixes faces

What it does:
  1. Reports how many manual.db rows (faces, file_sets, tags) reference a file_id
     that no longer exists in media.db's files table (orphaned by the bug).
  2. For orphaned *named faces* only: matches each orphaned face's embedding against
     every embedding in media.db's current (post-rescan) faces table via cosine
     similarity. Same photo + same model = near-identical embedding, so a match
     >= MATCH_THRESHOLD is treated as "this is the same face, just re-detected under
     a new file_id" and manual.db's file_id/source_face_id are corrected to match.
  3. file_sets and whole-image tags have no such signal (no embedding) and cannot be
     recovered this way — the script only reports their orphan count.
"""
import argparse
import shutil
import sqlite3
import sys
import time
import numpy as np

MATCH_THRESHOLD = 0.999  # near-exact — same image, same model, just re-run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('media_dir', help='path to the .media directory')
    parser.add_argument('--apply', action='store_true', help='actually write fixes (default: dry run)')
    args = parser.parse_args()

    media_db_path = f'{args.media_dir}/media.db'
    manual_db_path = f'{args.media_dir}/manual.db'

    media_conn = sqlite3.connect(media_db_path)
    media_conn.row_factory = sqlite3.Row
    manual_conn = sqlite3.connect(manual_db_path)
    manual_conn.row_factory = sqlite3.Row

    current_file_ids = {r[0] for r in media_conn.execute('SELECT id FROM files')}

    # --- report orphan counts ---
    for tbl in ['faces', 'file_sets', 'tags']:
        rows = manual_conn.execute(f'SELECT file_id FROM {tbl}').fetchall()
        orphaned = sum(1 for r in rows if r[0] not in current_file_ids)
        print(f'{tbl}: {len(rows)} total, {orphaned} orphaned (file_id no longer exists in media.db)')

    # --- recover named faces via embedding match ---
    orphaned_faces = [
        dict(r) for r in manual_conn.execute('SELECT * FROM faces')
        if r['file_id'] not in current_file_ids
    ]
    if not orphaned_faces:
        print('\nNo orphaned faces to recover.')
        return

    # Only match against faces whose file_id is still valid — media.db's faces table
    # can itself contain stale rows left over from before the fast_scan bug was fixed
    # (deletes from files never cascaded), so an unfiltered pool would happily "match"
    # an orphaned manual.db row to an equally-orphaned media.db row.
    current_faces = media_conn.execute(
        f'SELECT id, file_id, embedding FROM faces WHERE file_id IN '
        f'({",".join(str(i) for i in current_file_ids)})'
    ).fetchall() if current_file_ids else []
    if not current_faces:
        print('\nmedia.db has no current faces to match against — run `media faces` first if you haven\'t.')
        return

    matrix = np.stack([np.frombuffer(r['embedding'], dtype=np.float32) for r in current_faces])
    matched, unmatched = [], []

    for face in orphaned_faces:
        vec = np.frombuffer(face['embedding'], dtype=np.float32)
        scores = matrix.dot(vec)
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= MATCH_THRESHOLD:
            new_file_id = current_faces[best_idx]['file_id']
            new_source_id = current_faces[best_idx]['id']
            matched.append((face, new_file_id, new_source_id, best_score))
        else:
            unmatched.append((face, best_score))

    # `source_face_id` is UNIQUE (one media.db face promotes to at most one manual.db
    # row). Two orphaned rows can legitimately match the same current face — e.g. the
    # same person got named twice under different old ids before the bug — so keep
    # source_face_id only for the single best-scoring claimant per target; every other
    # row still gets its file_id fixed (that's what actually matters for search/lookup)
    # but source_face_id is left NULL rather than guessed, to avoid corrupting provenance.
    best_claim_for_source = {}
    for face, new_file_id, new_source_id, score in matched:
        current_best = best_claim_for_source.get(new_source_id)
        if current_best is None or score > current_best[3]:
            best_claim_for_source[new_source_id] = (face, new_file_id, new_source_id, score)

    existing_source_ids = {
        r[0] for r in manual_conn.execute(
            'SELECT source_face_id FROM faces WHERE source_face_id IS NOT NULL'
        )
    }
    final_updates = []  # (face, new_file_id, new_source_id_or_None, score)
    for face, new_file_id, new_source_id, score in matched:
        is_winner = best_claim_for_source[new_source_id][0]['id'] == face['id']
        # Only the highest-scoring claimant for a given target gets source_face_id set,
        # and only if nothing else (excluding this row's own stale old value) already
        # owns it — never risk violating the UNIQUE constraint. Every row still gets
        # its file_id fixed either way, which is what search/lookup actually depends on.
        already_owned_elsewhere = new_source_id in (existing_source_ids - {face['source_face_id']})
        if is_winner and not already_owned_elsewhere:
            final_updates.append((face, new_file_id, new_source_id, score))
        else:
            final_updates.append((face, new_file_id, None, score))

    dupe_count = sum(1 for f, _, sid, _ in final_updates if sid is None)

    print(f'\nOrphaned named faces: {len(orphaned_faces)}')
    print(f'  Recoverable via embedding match (>= {MATCH_THRESHOLD}): {len(matched)}')
    for face, new_file_id, new_source_id, score in final_updates:
        note = '' if new_source_id is not None else '  [file_id fixed, source_face_id left NULL — another row already claims that source]'
        print(f'    manual.db face id={face["id"]} identity={face["identity"]!r} '
              f'old_file_id={face["file_id"]} -> new_file_id={new_file_id} (score={score:.5f}){note}')
    if dupe_count:
        print(f'  ({dupe_count} of the matches above share a source with another row — see notes)')
    print(f'  Not recoverable (no close-enough match): {len(unmatched)}')
    for face, score in unmatched:
        print(f'    manual.db face id={face["id"]} identity={face["identity"]!r} '
              f'old_file_id={face["file_id"]} (best score only {score:.5f})')

    if not args.apply:
        print('\nDry run only — nothing was written. Re-run with --apply to fix the recoverable rows.')
        return

    backup_path = f'{manual_db_path}.bak-{time.strftime("%Y%m%d%H%M%S")}'
    shutil.copyfile(manual_db_path, backup_path)
    print(f'\nBacked up manual.db to {backup_path}')

    try:
        for face, new_file_id, new_source_id, score in final_updates:
            manual_conn.execute(
                'UPDATE faces SET file_id = ?, source_face_id = ? WHERE id = ?',
                (new_file_id, new_source_id, face['id'])
            )
        manual_conn.commit()
    except Exception:
        manual_conn.rollback()
        print('Something went wrong — rolled back, manual.db is unchanged. '
              f'Your pre-write backup is still at {backup_path} regardless.')
        raise
    print(f'Applied {len(final_updates)} fixes to manual.db.')


if __name__ == '__main__':
    main()
