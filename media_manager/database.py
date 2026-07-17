"""
Database schema and operations for media management.
"""
import sqlite3
import time

class Database:
    def __init__(self, db_path="media.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()

    def create_tables(self):
        """Create the necessary tables if they don't exist.

        NOTE: files.checksum is the identity now (content-addressable), not path — a
        fresh DB gets the new two-table shape directly. An existing DB from before this
        change (path-identity, no file_paths table) is NOT auto-migrated here; run
        migrate_file_identity.py once first. See the plan doc for why: collapsing
        path-duplicates into one canonical files.id needs a dry-run/--apply recovery-style
        script, not a silent ALTER, given what happened last time an old-schema disk file
        was upgraded implicitly."""
        cursor = self.conn.cursor()

        # Fail loudly and immediately if this is an old-schema DB, rather than letting
        # 'CREATE TABLE IF NOT EXISTS' silently no-op and crash confusingly deep inside
        # a scan the first time something tries to write to the new file_paths table.
        existing_cols = {row[1] for row in cursor.execute('PRAGMA table_info(files)')}
        if 'path' in existing_cols and 'first_seen' not in existing_cols:
            raise RuntimeError(
                "This media.db is still on the old path-identity schema. Run the "
                "one-time migration first:\n"
                "    python migrate_file_identity.py /path/to/.media          # dry run\n"
                "    python migrate_file_identity.py /path/to/.media --apply  # then migrate\n"
                "Nothing has been written by this run."
            )

        # Files table: one row per unique piece of *content* (by checksum), not per path.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
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
        # file_paths: every location this content has been seen at. One-to-many — this
        # is where duplicates (same checksum, multiple paths) live. last_seen_at is
        # bumped on every scan that still finds the path on disk, so a path whose
        # last_seen_at predates the most recent scan is stale (moved away or deleted).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                path TEXT UNIQUE NOT NULL,
                modified_time REAL,
                last_seen_at INTEGER NOT NULL
            )
        ''')
        # Read-only convenience view: resolves each content record's *primary* path
        # (most recently seen) so the rest of this file's queries can keep selecting a
        # single 'path' column, same shape as the old one-row-per-path schema.
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS files_with_path AS
            SELECT f.*, fp.path AS path, fp.modified_time AS modified_time
            FROM files f
            JOIN file_paths fp ON fp.id = (
                SELECT fp2.id FROM file_paths fp2
                WHERE fp2.file_id = f.id
                ORDER BY fp2.last_seen_at DESC LIMIT 1
            )
        ''')
        # Sets table: named collections of images, e.g. a shoot from a studio
        # (superseded by manual.db's sets/file_sets — left here only so an old DB's
        # dead rows aren't dropped; nothing in this file reads/writes it anymore).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                studio TEXT,
                created_at INTEGER NOT NULL,
                UNIQUE(name, studio)
            )
        ''')
        # Embeddings table: stores CLIP embeddings for image search. frame_index is
        # part of the identity (not nullable) — an embedding is always "of some
        # specific image," so frame 0 (the default/primary embedding, same meaning as
        # before frame support existed) is a real value, not a NULL "whole file"
        # placeholder. A composite PK with NULL in one column wouldn't enforce the
        # uniqueness we want anyway, since SQL treats each NULL as distinct.
        # CREATE TABLE IF NOT EXISTS handles both fresh DBs and old DBs (migration).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                frame_index INTEGER NOT NULL DEFAULT 0,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                indexed_at INTEGER NOT NULL,
                PRIMARY KEY (file_id, frame_index)
            )
        ''')
        # Tags table: user-defined labels attached to files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(file_id, tag)
            )
        ''')
        # Detections table: stores YOLO-World detected objects for image search.
        # frame_index NULL = whole-file/primary-frame detection (today's behavior,
        # written by the batch `media index` CLI); a value = found only at that frame
        # of an animated file (written by the per-image "scan all frames" action).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                model TEXT NOT NULL,
                indexed_at INTEGER NOT NULL,
                frame_index INTEGER
            )
        ''')
        # Faces table: InsightFace detections + ArcFace embeddings. frame_index: same
        # NULL-means-primary-frame convention as detections.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                bbox        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                det_score   REAL NOT NULL,
                identity    TEXT,
                indexed_at  INTEGER NOT NULL,
                frame_index INTEGER
            )
        ''')
        # Body embeddings table: CLIP vectors of person crops, for find-by-body
        # (re-identifying a person by outfit/build when the face is hidden). Derived,
        # rebuildable data keyed by file_id like detections/embeddings — not part of
        # the checksum-keyed manual ground truth. bbox is a JSON [x1,y1,x2,y2] like
        # faces.bbox; the crop's source person box comes from YOLO detections.
        # frame_index: same NULL-means-primary-frame convention as detections/faces.
        # Sentinel for processed-but-no-person files: bbox='[]' with an empty blob.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS body_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                bbox        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                model       TEXT NOT NULL,
                indexed_at  INTEGER NOT NULL,
                frame_index INTEGER
            )
        ''')
        # Migrate DBs created before frame_index existed (nullable ADD COLUMN is safe
        # in-place — every existing row correctly becomes "primary frame" = NULL).
        detections_cols = {row[1] for row in cursor.execute('PRAGMA table_info(detections)')}
        if 'frame_index' not in detections_cols:
            cursor.execute('ALTER TABLE detections ADD COLUMN frame_index INTEGER')
        faces_cols = {row[1] for row in cursor.execute('PRAGMA table_info(faces)')}
        if 'frame_index' not in faces_cols:
            cursor.execute('ALTER TABLE faces ADD COLUMN frame_index INTEGER')

        # embeddings' PK changed shape (file_id -> (file_id, frame_index)), which SQLite
        # can't ALTER in place — rebuild the table if it's still the old single-column-PK
        # shape. Every existing row is by definition the file's only/primary embedding,
        # so it becomes frame_index=0, matching what a fresh scan would have written.
        embeddings_cols = {row[1] for row in cursor.execute('PRAGMA table_info(embeddings)')}
        if 'frame_index' not in embeddings_cols:
            cursor.execute('ALTER TABLE embeddings RENAME TO embeddings_old')
            cursor.execute('''
                CREATE TABLE embeddings (
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    frame_index INTEGER NOT NULL DEFAULT 0,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    indexed_at INTEGER NOT NULL,
                    PRIMARY KEY (file_id, frame_index)
                )
            ''')
            cursor.execute('''
                INSERT INTO embeddings (file_id, frame_index, embedding, model, indexed_at)
                SELECT file_id, 0, embedding, model, indexed_at FROM embeddings_old
            ''')
            cursor.execute('DROP TABLE embeddings_old')

        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_paths_path ON file_paths (path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_paths_file ON file_paths (file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON files (checksum)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_taken_at ON files (taken_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags (tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_class ON detections (class_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_file ON detections (file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_file ON faces (file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_identity ON faces (identity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sets_studio ON sets (studio)')
        self.conn.commit()

    def upsert_file_path(self, path, checksum, size=None, modified_time=None, seen_at=None):
        """Core write primitive for content-addressable identity (see fast_scan.py):
        find-or-create the content record for `checksum`, then point `path` at it.
        No commit — callers batch many of these per scan into one transaction.
        Returns file_id."""
        cursor = self.conn.cursor()
        now = seen_at if seen_at is not None else int(time.time())
        cursor.execute('SELECT id FROM files WHERE checksum = ?', (checksum,))
        row = cursor.fetchone()
        if row:
            file_id = row[0]
            cursor.execute('UPDATE files SET size = ? WHERE id = ?', (size, file_id))
        else:
            cursor.execute(
                'INSERT INTO files (checksum, size, first_seen) VALUES (?, ?, ?)',
                (checksum, size, now)
            )
            file_id = cursor.lastrowid
        cursor.execute('''
            INSERT INTO file_paths (file_id, path, modified_time, last_seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_id=excluded.file_id,
                modified_time=excluded.modified_time,
                last_seen_at=excluded.last_seen_at
        ''', (file_id, path, modified_time, now))
        return file_id

    def find_file_id_by_checksum(self, checksum):
        """Return the existing files.id for this checksum, or None. Used at scan time
        to detect a duplicate *before* writing anything — see fast_scan.py."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM files WHERE checksum = ?', (checksum,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_file_by_path(self, path):
        """Retrieve a file record (with its primary path) by any known path."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT fwp.* FROM files_with_path fwp
            JOIN file_paths fp ON fp.file_id = fwp.id
            WHERE fp.path = ?
        ''', (path,))
        return cursor.fetchone()

    def get_paths_for_file(self, file_id):
        """Return every known path for a piece of content, most-recently-seen first —
        powers a 'also found at...' hint and the duplicates view."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT path, last_seen_at FROM file_paths WHERE file_id = ? ORDER BY last_seen_at DESC',
            (file_id,)
        )
        return cursor.fetchall()

    def list_all_paths(self):
        """Return every (path, file_id) this DB currently knows about — used by
        `media status` to check which recorded paths no longer exist on disk (moved
        away or deleted). No disk I/O here; the caller checks existence."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT path, file_id FROM file_paths')
        return cursor.fetchall()

    def find_duplicates(self, limit=200):
        """Return [(file_id, path_count), ...] for content seen at more than one path —
        the actual point of switching to content-addressable identity: surfaces real
        duplicates for manual review/cleanup instead of just tagging/sorting each copy
        separately."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT file_id, COUNT(*) as path_count
            FROM file_paths
            GROUP BY file_id
            HAVING path_count > 1
            ORDER BY path_count DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def list_files(self, limit=100):
        """List files (id, path, size, checksum), most-recently-seen path shown."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, path, size, checksum FROM files_with_path LIMIT ?', (limit,))
        return cursor.fetchall()

    def count_files(self, limit=None):
        """Return the number of content records. limit: maximum rows to count (None == unlimited)."""
        cur = self.conn.cursor()
        sql = 'SELECT COUNT(*) FROM files'
        if limit is not None:
            sql += ' LIMIT ?'
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else 0

    def update_file_metadata(self, file_id, taken_at=None, gps_lat=None, gps_lon=None):
        """Store EXIF-derived capture time / GPS coordinates for a file, when present.
        Always stamps metadata_checked_at so a file with no EXIF isn't retried forever."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE files
            SET taken_at = ?, gps_lat = ?, gps_lon = ?, metadata_checked_at = ?
            WHERE id = ?
        ''', (taken_at, gps_lat, gps_lon, int(time.time()), file_id))
        self.conn.commit()

    def get_files_without_metadata(self, limit=None):
        """Return (id, path) for files whose EXIF metadata has never been checked."""
        cursor = self.conn.cursor()
        sql = 'SELECT id, path FROM files_with_path WHERE metadata_checked_at IS NULL'
        if limit is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        return cursor.fetchall()

    def count_broken_files(self):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM files WHERE broken IS NOT NULL')
        return cur.fetchone()[0]

    def list_broken_files(self, limit=100):
        cur = self.conn.cursor()
        cur.execute('SELECT path, broken FROM files_with_path WHERE broken IS NOT NULL LIMIT ?', (limit,))
        return cur.fetchall()

    def clear_broken(self, paths):
        cur = self.conn.cursor()
        cur.executemany('''
            UPDATE files SET broken = NULL
            WHERE id = (SELECT file_id FROM file_paths WHERE path = ?)
        ''', [(p,) for p in paths])
        self.conn.commit()
        return cur.rowcount

    def insert_embedding(self, file_id, embedding_bytes, model, frame_index=0):
        """Upsert the embedding for a file at a given frame (frame_index=0 is the
        default/primary embedding — same meaning as before per-frame support existed)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (file_id, frame_index, embedding, model, indexed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_id, frame_index, embedding_bytes, model, int(time.time())))
        self.conn.commit()

    def get_all_embeddings(self):
        """Return list of (file_id, path, embedding_bytes, checksum) joining with
        files. Scoped to the primary (frame_index=0) embedding only — frame-specific
        embeddings from 'scan all frames' don't participate in whole-file similarity
        search/gallery flags until that's deliberately built."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT e.file_id, f.path, e.embedding, f.checksum
            FROM embeddings e
            JOIN files_with_path f ON f.id = e.file_id
            WHERE e.frame_index = 0
        ''')
        return cursor.fetchall()

    def get_embeddings_for_files(self, file_ids):
        """Return [(file_id, embedding_bytes), ...] for a specific set of files'
        primary embeddings — used to build a representative CLIP vector for e.g.
        "images like this set"."""
        if not file_ids:
            return []
        placeholders = ','.join('?' for _ in file_ids)
        cursor = self.conn.cursor()
        cursor.execute(
            f'SELECT file_id, embedding FROM embeddings WHERE frame_index = 0 AND file_id IN ({placeholders})',
            tuple(file_ids)
        )
        return cursor.fetchall()

    def get_unindexed_files(self, limit=None):
        """Return (id, path) for files that have no primary (frame_index=0)
        embedding — independent of whether frame-specific embeddings exist."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files_with_path f
                LEFT JOIN embeddings e ON e.file_id = f.id AND e.frame_index = 0
                WHERE e.file_id IS NULL
            ''')
        else:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files_with_path f
                LEFT JOIN embeddings e ON e.file_id = f.id AND e.frame_index = 0
                WHERE e.file_id IS NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def count_indexed(self):
        """Return the count of files with a primary (frame_index=0) embedding."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings WHERE frame_index = 0')
        row = cursor.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Tag methods
    # ------------------------------------------------------------------

    def add_tag(self, file_id, tag):
        """Add a tag to a file (no-op if already exists)."""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO tags (file_id, tag, created_at) VALUES (?, ?, ?)',
            (file_id, tag.strip(), int(time.time()))
        )
        self.conn.commit()

    def remove_tag(self, file_id, tag):
        """Remove a tag from a file."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM tags WHERE file_id = ? AND tag = ?', (file_id, tag))
        self.conn.commit()

    def get_tags(self, file_id):
        """Return list of tag strings for a file."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT tag FROM tags WHERE file_id = ? ORDER BY tag', (file_id,))
        return [row[0] for row in cursor.fetchall()]

    def get_files_by_tag(self, tag, limit=100):
        """Return (file_id, path) rows for files that have the given tag."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.id, f.path
            FROM files_with_path f
            JOIN tags t ON t.file_id = f.id
            WHERE t.tag = ?
            ORDER BY f.path
            LIMIT ?
        ''', (tag, limit))
        return cursor.fetchall()

    def list_all_tags(self):
        """Return [(tag, count), ...] ordered by count descending."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT tag, COUNT(*) as cnt
            FROM tags
            GROUP BY tag
            ORDER BY cnt DESC
        ''')
        return cursor.fetchall()

    def get_file_by_id(self, file_id):
        """Retrieve a file record (with its primary path) by its id."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM files_with_path WHERE id = ?', (file_id,))
        return cursor.fetchone()

    def get_neighbor_ids(self, file_id):
        """Return (prev_id, next_id) — the adjacent files by id, matching the gallery's
        default ordering. Powers arrow-key next/previous navigation on the photo page.
        Scoped to files_with_path so navigation never lands on a dangling content
        record (content whose only path was repointed elsewhere after an in-place edit)."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(id) FROM files_with_path WHERE id < ?', (file_id,))
        prev_id = cursor.fetchone()[0]
        cursor.execute('SELECT MIN(id) FROM files_with_path WHERE id > ?', (file_id,))
        next_id = cursor.fetchone()[0]
        return prev_id, next_id

    def get_files_by_ids(self, ids):
        """Batched path lookup for a list of file ids — avoids N+1 queries when
        resolving paths for rows sourced from a different database (e.g. manual.db)."""
        if not ids:
            return []
        placeholders = ','.join('?' for _ in ids)
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM files_with_path WHERE id IN ({placeholders})', tuple(ids))
        return cursor.fetchall()

    def get_file_by_checksum(self, checksum):
        """Retrieve a file record (with its primary path) by content checksum — this is
        manual.db's identity now, not file_id (see manual_db.py)."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM files_with_path WHERE checksum = ?', (checksum,))
        return cursor.fetchone()

    def get_files_by_checksums(self, checksums):
        """Batched path lookup for a list of checksums — resolves manual.db query
        results (which are checksum-keyed) to current file_id/path rows."""
        if not checksums:
            return []
        placeholders = ','.join('?' for _ in checksums)
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM files_with_path WHERE checksum IN ({placeholders})', tuple(checksums))
        return cursor.fetchall()

    # Columns 'age' can sort by are handled entirely in Python (web.py) since ages
    # live in the separate manual.db — this SQL-level sort only ever sees 'added'/
    # 'modified'.
    _GALLERY_SORT_COLUMNS = {'added': 'f.first_seen', 'modified': 'f.modified_time'}

    def list_files_with_embedding_flag(self, limit=200, offset=0, sort='added', order='desc'):
        """Return (id, path, has_embedding, checksum) rows for gallery browsing.
        sort: 'added' (first_seen) or 'modified' (file's on-disk mtime). order: 'asc'/'desc'."""
        column = self._GALLERY_SORT_COLUMNS.get(sort, 'f.first_seen')
        direction = 'ASC' if order == 'asc' else 'DESC'
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT f.id, f.path,
                   CASE WHEN e.file_id IS NOT NULL THEN 1 ELSE 0 END AS has_embedding,
                   f.checksum
            FROM files_with_path f
            LEFT JOIN embeddings e ON e.file_id = f.id
            ORDER BY {column} {direction}, f.id {direction}
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        return cursor.fetchall()

    def get_embedding(self, file_id):
        """Return the primary (frame_index=0) embedding bytes for a file, or None."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM embeddings WHERE file_id = ? AND frame_index = 0', (file_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Detection methods (YOLO-World)
    # ------------------------------------------------------------------

    def insert_detections(self, file_id, detections, model, frame_index=None):
        """Upsert detections for a file (or, if frame_index is given, for just that
        frame of an animated file). detections is a list of
        (class_name, confidence, x1, y1, x2, y2). Always writes at least a sentinel
        row ('__indexed__') so the file/frame is never re-queued.

        The destructive DELETE is scoped to the same frame_index — this is what keeps
        the batch `media index` CLI (frame_index=None) from wiping out frame-specific
        rows written by the per-image 'scan all frames' action, and vice versa."""
        cursor = self.conn.cursor()
        if frame_index is None:
            cursor.execute('DELETE FROM detections WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        else:
            cursor.execute('DELETE FROM detections WHERE file_id = ? AND frame_index = ?', (file_id, frame_index))
        now = int(time.time())
        rows = [(file_id, cls, conf, x1, y1, x2, y2, model, now, frame_index)
                for cls, conf, x1, y1, x2, y2 in detections]
        if not rows:
            # sentinel: marks file/frame as processed even when nothing was detected
            rows = [(file_id, '__indexed__', 0.0, None, None, None, None, model, now, frame_index)]
        cursor.executemany(
            '''INSERT INTO detections (file_id, class_name, confidence, x1, y1, x2, y2, model, indexed_at, frame_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            rows
        )
        self.conn.commit()

    def get_detected_classes(self, file_id):
        """Return list of distinct detected class names for a file's primary frame,
        ordered by confidence descending."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT class_name FROM detections "
            "WHERE file_id = ? AND frame_index IS NULL AND class_name != '__indexed__' ORDER BY confidence DESC",
            (file_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def remove_detection(self, file_id, class_name):
        """Delete a specific auto-detected class for a file — used when a human marks it
        a false positive (negative tag). If this was the file's only detection, insert
        the sentinel row so a later `media index` run doesn't just regenerate it."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE file_id = ? AND class_name = ?', (file_id, class_name))
        cursor.execute('SELECT COUNT(*) FROM detections WHERE file_id = ?', (file_id,))
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO detections (file_id, class_name, confidence, x1, y1, x2, y2, model, indexed_at) "
                "VALUES (?, '__indexed__', 0.0, NULL, NULL, NULL, NULL, 'manual-correction', ?)",
                (file_id, int(time.time()))
            )
        self.conn.commit()

    def get_undetected_files(self, limit=None):
        """Return (id, path) for files that have no primary (frame_index IS NULL)
        detections row — independent of whether frame-specific rows exist."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files_with_path f
                LEFT JOIN detections d ON d.file_id = f.id AND d.frame_index IS NULL
                WHERE d.file_id IS NULL
            ''')
        else:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files_with_path f
                LEFT JOIN detections d ON d.file_id = f.id AND d.frame_index IS NULL
                WHERE d.file_id IS NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def search_by_classes(self, class_names, limit=20):
        """
        Return (file_id, path, score, checksum) rows where score = SUM(confidence) for
        matched classes. Uses LIKE substring matching so "couch" matches "couch" and
        "sofa couch" etc. Excludes sentinel rows (class_name = '__indexed__').
        """
        if not class_names:
            return []
        # Build: (class_name LIKE %tok1% OR class_name LIKE %tok2% OR ...)
        like_clauses = ' OR '.join('d.class_name LIKE ?' for _ in class_names)
        like_params = [f'%{t}%' for t in class_names]
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT f.id, f.path, SUM(d.confidence) as score, f.checksum
            FROM detections d
            JOIN files_with_path f ON f.id = d.file_id
            WHERE ({like_clauses})
              AND d.class_name != '__indexed__'
              AND d.frame_index IS NULL
            GROUP BY d.file_id
            ORDER BY score DESC
            LIMIT ?
        ''', (*like_params, limit))
        return cursor.fetchall()

    def search_by_path_substring(self, query, limit=50):
        """
        Return (file_id, path, checksum) rows where path contains `query` as a
        case-insensitive substring. Mirrors the (file_id, path, ..., checksum)
        shape of search_by_classes so callers can merge results easily, minus
        the score column since there's no similarity score for a plain
        substring match.
        """
        if not query:
            return []
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, path, checksum
            FROM files_with_path
            WHERE LOWER(path) LIKE LOWER(?)
            LIMIT ?
        ''', (f'%{query}%', limit))
        return cursor.fetchall()

    def count_detected(self):
        """Return count of distinct files with a primary (frame_index IS NULL) detections row."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(DISTINCT file_id) FROM detections WHERE frame_index IS NULL')
        row = cursor.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Face methods (InsightFace)
    # ------------------------------------------------------------------

    def insert_faces(self, file_id: int, faces: list, model: str) -> None:
        """Upsert primary-frame face rows for a file (the batch `media faces` CLI
        path — always writes frame_index=NULL).
        faces: list of {'bbox': [x1,y1,x2,y2], 'embedding': np.ndarray, 'det_score': float}
        Always writes at least one sentinel row so the file is never re-queued.
        The DELETE is scoped to frame_index IS NULL so re-running this never wipes
        out frame-specific rows written by the per-image 'scan all frames' action."""
        import json
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM faces WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        now = int(time.time())
        if not faces:
            cursor.execute(
                'INSERT INTO faces (file_id, bbox, embedding, det_score, identity, indexed_at) VALUES (?,?,?,?,?,?)',
                (file_id, '[]', b'', 0.0, '__indexed__', now)
            )
        else:
            for face in faces:
                cursor.execute(
                    'INSERT INTO faces (file_id, bbox, embedding, det_score, identity, indexed_at) VALUES (?,?,?,?,?,?)',
                    (file_id, json.dumps(face['bbox']), face['embedding'].tobytes(), face['det_score'], None, now)
                )
        self.conn.commit()

    def add_manual_face(self, file_id, bbox, embedding_bytes, det_score, frame_index=None) -> int:
        """Insert a single manually-added (or per-frame auto-detected) face row
        without deleting existing rows for this file (unlike insert_faces, which is
        destructive and is used only by the batch `media faces` CLI command).
        frame_index=None means a normal whole-file/primary-frame face; a value means
        this face was found at that specific frame of an animated file."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO faces (file_id, bbox, embedding, det_score, identity, indexed_at, frame_index) '
            'VALUES (?,?,?,?,?,?,?)',
            (file_id, json.dumps(bbox), embedding_bytes, det_score, None, int(time.time()), frame_index)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_unface_indexed_files(self, limit=None) -> list:
        """Return (id, path) for files that have no primary (frame_index IS NULL)
        row in faces table — independent of whether frame-specific rows exist."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT f.id, f.path
            FROM files_with_path f
            LEFT JOIN faces fa ON fa.file_id = f.id AND fa.frame_index IS NULL
            WHERE fa.file_id IS NULL
        '''
        if limit is not None:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        else:
            cursor.execute(sql)
        return cursor.fetchall()

    def get_faces_for_file(self, file_id: int) -> list:
        """Return all non-sentinel face rows for a file (excludes embedding blob),
        primary and frame-specific alike — the photo page shows everything found for
        this file, with a frame badge on rows where frame_index is not null."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, bbox, det_score, identity, frame_index FROM faces "
            "WHERE file_id = ? AND (identity IS NULL OR identity != '__indexed__') ORDER BY det_score DESC",
            (file_id,)
        )
        return cursor.fetchall()

    def get_face_embedding(self, face_id: int):
        """Return raw embedding bytes for a single face row."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM faces WHERE id = ?', (face_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_face_embeddings(self) -> list:
        """Return [(face_id, file_id, path, embedding_bytes)] excluding sentinels."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT fa.id, fa.file_id, f.path, fa.embedding
            FROM faces fa
            JOIN files_with_path f ON f.id = fa.file_id
            WHERE (fa.identity IS NULL OR fa.identity != '__indexed__') AND fa.embedding != x''
        ''')
        return cursor.fetchall()

    def assign_identity(self, face_id: int, name: str) -> None:
        """Set identity = name for a single face row."""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE faces SET identity = ? WHERE id = ?', (name.strip(), face_id))
        self.conn.commit()

    def get_files_by_face_identity(self, name: str, limit: int = 100) -> list:
        """Return (file_id, path) for files containing a face with given identity (case-insensitive)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT f.id, f.path
            FROM files_with_path f
            JOIN faces fa ON fa.file_id = f.id
            WHERE LOWER(fa.identity) LIKE LOWER(?)
            ORDER BY f.path
            LIMIT ?
        ''', (f'%{name}%', limit))
        return cursor.fetchall()

    def get_unidentified_faces(self, limit: int = 200) -> list:
        """Return (face_id, file_id, path, bbox, embedding) for faces where identity IS NULL."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT fa.id, fa.file_id, f.path, fa.bbox, fa.embedding
            FROM faces fa
            JOIN files_with_path f ON f.id = fa.file_id
            WHERE fa.identity IS NULL
            ORDER BY fa.id
        '''
        if limit is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        return cursor.fetchall()

    def get_named_face_embeddings(self) -> list:
        """Return [(identity, embedding_bytes)] for every already-named, non-sentinel face.
        Used to suggest a name for newly-unidentified faces that look like someone already known."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT identity, embedding FROM faces
            WHERE identity IS NOT NULL AND identity != '__indexed__' AND embedding != x''
        ''')
        return cursor.fetchall()

    def get_all_identities(self) -> list:
        """Return [(identity, count)] ordered by count DESC, excluding sentinels and NULL."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT identity, COUNT(*) as cnt
            FROM faces
            WHERE identity IS NOT NULL AND identity != '__indexed__'
            GROUP BY identity
            ORDER BY cnt DESC
        ''')
        return cursor.fetchall()

    # ------------------------------------------------------------------
    # Body embedding methods (find-by-body: CLIP-embedded person crops)
    # ------------------------------------------------------------------

    def insert_body_embeddings(self, file_id: int, bodies: list, model: str) -> None:
        """Upsert primary-frame body rows for a file.
        bodies: list of {'bbox': [x1,y1,x2,y2], 'embedding': np.ndarray}
        Always writes at least one sentinel row (bbox='[]', empty blob) so the file
        is never re-queued. DELETE scoped to frame_index IS NULL, mirroring
        insert_faces."""
        import json
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM body_embeddings WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        now = int(time.time())
        if not bodies:
            cursor.execute(
                'INSERT INTO body_embeddings (file_id, bbox, embedding, model, indexed_at) VALUES (?,?,?,?,?)',
                (file_id, '[]', b'', model, now)
            )
        else:
            for body in bodies:
                cursor.execute(
                    'INSERT INTO body_embeddings (file_id, bbox, embedding, model, indexed_at) VALUES (?,?,?,?,?)',
                    (file_id, json.dumps(body['bbox']), body['embedding'].tobytes(), model, now)
                )
        self.conn.commit()

    def get_body_embeddings_for_file(self, file_id: int) -> list:
        """Return non-sentinel (id, bbox, embedding) rows for a file."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, bbox, embedding FROM body_embeddings WHERE file_id = ? AND bbox != '[]'",
            (file_id,)
        )
        return cursor.fetchall()

    def get_body_embedding(self, body_id: int):
        """Return (file_id, bbox, embedding_bytes) for one body row, or None."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT file_id, bbox, embedding FROM body_embeddings WHERE id = ?', (body_id,))
        return cursor.fetchone()

    def get_all_body_embeddings(self) -> list:
        """Return [(body_id, file_id, path, bbox, embedding_bytes)] excluding sentinels."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT b.id, b.file_id, f.path, b.bbox, b.embedding
            FROM body_embeddings b
            JOIN files_with_path f ON f.id = b.file_id
            WHERE b.bbox != '[]' AND b.embedding != x''
        ''')
        return cursor.fetchall()

    def get_unbody_indexed_files(self, limit=None) -> list:
        """Return (id, path) for files that have no primary (frame_index IS NULL)
        row in body_embeddings — independent of whether frame-specific rows exist."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT f.id, f.path
            FROM files_with_path f
            LEFT JOIN body_embeddings b ON b.file_id = f.id AND b.frame_index IS NULL
            WHERE b.file_id IS NULL
        '''
        if limit is not None:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        else:
            cursor.execute(sql)
        return cursor.fetchall()

    def get_body_indexable_files(self) -> list:
        """Return (id, path) for files that have primary-frame detections (so person
        boxes are knowable without running a detector) but no body_embeddings rows
        yet. Files never YOLO-indexed are deliberately excluded — they become
        indexable after `media index` runs, rather than being sentineled as
        person-free here."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.id, f.path
            FROM files_with_path f
            JOIN detections d ON d.file_id = f.id AND d.frame_index IS NULL
            LEFT JOIN body_embeddings b ON b.file_id = f.id AND b.frame_index IS NULL
            WHERE b.file_id IS NULL
            GROUP BY f.id
        ''')
        return cursor.fetchall()

    def get_person_detections_for_file(self, file_id: int, min_conf: float = 0.3) -> list:
        """Return [x1,y1,x2,y2] person boxes from the primary-frame YOLO detections
        for a file. Only rows with real coordinates count — old rows and sentinels
        have NULL coords."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT x1, y1, x2, y2 FROM detections
            WHERE file_id = ? AND class_name = 'person' AND confidence >= ?
              AND frame_index IS NULL AND x1 IS NOT NULL
        ''', (file_id, min_conf))
        return [list(row) for row in cursor.fetchall()]

    def clear_primary_ml_data(self, file_id):
        """Delete primary-frame (frame_index NULL/0) detections, faces, body
        embeddings, and the frame-0 embedding for a file, so the next
        index/embed/faces run reprocesses it from scratch. Used by 'media
        add/commit --reindex' for files that were already tracked at the same path.
        Frame-specific rows from 'scan all frames' are left untouched."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        cursor.execute('DELETE FROM faces WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        cursor.execute('DELETE FROM embeddings WHERE file_id = ? AND frame_index = 0', (file_id,))
        cursor.execute('DELETE FROM body_embeddings WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
