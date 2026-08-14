"""Migration 0001: promote tags from a flat table (one repeated `label` string
per instance row — checksum, label, polarity, bbox, all on the same row) into
a real id'd `tags` definitions table (id, label UNIQUE, created_at) plus a
`file_tags` instance table (every column the old table had, minus `label`,
plus a `tag_id` FK) — the same shape as this app's existing categories/
file_categories split. See manual_db.py's ManualDB._create_tables for the
target schema these leave behind (the CREATE TABLE statements there are the
source of truth for a *fresh* install; this migration exists only to carry
forward an existing manual.db's data into that same shape)."""

ID = '0001_tags_table'


def describe_pending(conn):
    """None if there's nothing to do: either already migrated (no `checksum`
    column on `tags` — that column only ever existed on the old flat shape,
    the new definitions table never has one), or a fresh install where `tags`
    doesn't exist yet at all (PRAGMA table_info on a nonexistent table just
    returns no rows, so `cols` is empty and 'checksum' is trivially absent)."""
    cols = {row[1] for row in conn.execute('PRAGMA table_info(tags)')}
    if 'checksum' not in cols:
        return None
    total_instances = conn.execute('SELECT COUNT(*) FROM tags').fetchone()[0]
    distinct_labels = conn.execute('SELECT COUNT(DISTINCT label) FROM tags').fetchone()[0]
    return (
        f'would migrate {total_instances} tag instance row(s) across '
        f'{distinct_labels} distinct label(s) into a real tags/file_tags schema'
    )


def apply(conn):
    """Performs the migration. Assumes `conn` is already inside a transaction
    (see manual_db.run_migrations) — issues DDL/DML only; verifies row counts
    before returning and raises on mismatch so the caller rolls back the whole
    transaction, leaving the original `tags` table completely untouched."""
    total_instances = conn.execute('SELECT COUNT(*) FROM tags').fetchone()[0]
    distinct_labels = conn.execute('SELECT COUNT(DISTINCT label) FROM tags').fetchone()[0]

    conn.execute('''
        CREATE TABLE tags_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL
        )
    ''')
    conn.execute('''
        INSERT INTO tags_new (label, created_at)
        SELECT label, MIN(created_at) FROM tags GROUP BY label
    ''')
    conn.execute('''
        CREATE TABLE file_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checksum TEXT NOT NULL,
            tag_id INTEGER NOT NULL REFERENCES tags_new(id) ON DELETE CASCADE,
            polarity TEXT NOT NULL DEFAULT 'positive',
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            image_width INTEGER, image_height INTEGER,
            frame_index INTEGER,
            favorite INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )
    ''')
    conn.execute('''
        INSERT INTO file_tags (checksum, tag_id, polarity, x1, y1, x2, y2,
                               image_width, image_height, frame_index, favorite, created_at)
        SELECT t.checksum, tn.id, t.polarity, t.x1, t.y1, t.x2, t.y2,
               t.image_width, t.image_height, t.frame_index, t.favorite, t.created_at
        FROM tags t JOIN tags_new tn ON tn.label = t.label
    ''')

    new_instance_count = conn.execute('SELECT COUNT(*) FROM file_tags').fetchone()[0]
    new_def_count = conn.execute('SELECT COUNT(*) FROM tags_new').fetchone()[0]
    if new_instance_count != total_instances or new_def_count != distinct_labels:
        raise RuntimeError(
            f'verification failed: {total_instances} old instance row(s) vs '
            f'{new_instance_count} new, {new_def_count} new definition(s) vs '
            f'{distinct_labels} distinct label(s) expected'
        )

    conn.execute('DROP TABLE tags')
    conn.execute('ALTER TABLE tags_new RENAME TO tags')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_file_tags_checksum ON file_tags (checksum)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags (tag_id)')
    conn.execute('''
        CREATE VIEW IF NOT EXISTS file_tags_with_label AS
        SELECT ft.*, t.label AS label FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
    ''')
    return f'migrated {total_instances} tag instance row(s) across {distinct_labels} distinct label(s)'
