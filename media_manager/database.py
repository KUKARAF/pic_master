"""
Database schema and operations for media management.
"""
import os
import sqlite3
import threading
import time


class ThreadLocalDB:
    """One sqlite connection per thread, opened lazily via the `conn` property.

    A single connection shared across threads — even with check_same_thread=False —
    is not safe for *concurrent* use: FastAPI's threadpool runs handlers in parallel
    and two overlapping cursor.execute() calls on one connection crash with
    'InterfaceError: bad parameter or other API misuse' (seen as random blank
    thumbnails in the gallery). Multiple connections to the same file are the
    supported way to do this; WAL mode lets readers proceed during a write and
    busy_timeout makes writers wait instead of erroring when they collide."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        # Persistent for the db file, so one-time here on the creating thread.
        self.conn.execute('PRAGMA journal_mode=WAL')

    @property
    def conn(self):
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA busy_timeout=5000')
            self._local.conn = conn
        return conn

    def close(self):
        """Close the calling thread's connection (other threads' stay open)."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @staticmethod
    def _chunked(items, size=500):
        """SQLite has a hard cap on bound parameters per query (varies by build,
        often as low as 999) — any `WHERE x IN (...)` lookup built from an
        *unbounded* candidate pool (e.g. every checksum in the whole library, not
        just a fixed-size page) needs to be split into chunks like this rather
        than bound in one query, or it raises 'too many SQL variables' once the
        library is large enough. Inherited by both Database and ManualDB so every
        checksum/id-batched lookup method gets this for free."""
        items = list(items)
        for i in range(0, len(items), size):
            yield items[i:i + size]


class Database(ThreadLocalDB):
    def __init__(self, db_path="media.db"):
        super().__init__(db_path)
        # --- Cached embedding matrices (write-invalidated) -----------------
        # get_all_*embeddings() fetchall() the whole table's float32 BLOBs on
        # every web request; callers then build a second numpy matrix. Under
        # concurrency that is a top OOM driver. The *_matrix() accessors below
        # build one shared numpy matrix, reuse it across requests, and rebuild
        # only when the underlying table changes. Each write bumps the matching
        # version counter; each accessor rebuilds iff its cached version is
        # stale. Single-process shared cache (fine for uvicorn --workers 1).
        self._emb_ver = 0
        self._face_ver = 0
        self._body_ver = 0
        self._tile_ver = 0
        self._matrix_lock = threading.Lock()
        # Each cache slot holds (version_it_was_built_at, built_result_tuple).
        self._emb_cache = None
        self._face_cache = None
        self._body_cache = None
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
        # noface: set by a .noface marker file (folder_markers.py) found in an
        # ancestor directory at scan time — "assume there are no faces here",
        # so `media faces` never selects this content for detection at all (see
        # get_unface_indexed_files below). Checksum-scoped like everything else
        # in this table, not per-path: the same content found again somewhere
        # else is still the same "no faces here" photo.
        files_cols = {row[1] for row in cursor.execute('PRAGMA table_info(files)')}
        if 'noface' not in files_cols:
            cursor.execute('ALTER TABLE files ADD COLUMN noface INTEGER NOT NULL DEFAULT 0')
        # hidden: derived content that shouldn't clutter the gallery/library listings
        # (currently manually-captured video frames — see web.py capture-frame). Still
        # a first-class file everywhere else (photo view, thumbnails, face search).
        if 'hidden' not in files_cols:
            cursor.execute('ALTER TABLE files ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0')
        # view_count / last_viewed_at: how many times /photo/{id} has been opened, and
        # when last. Drives the home "Needs attention" section's least-viewed-first
        # ordering (see get_least_viewed_files + web.py increment_view_count hook).
        if 'view_count' not in files_cols:
            cursor.execute('ALTER TABLE files ADD COLUMN view_count INTEGER NOT NULL DEFAULT 0')
        if 'last_viewed_at' not in files_cols:
            cursor.execute('ALTER TABLE files ADD COLUMN last_viewed_at INTEGER')
        # city_id: nearest known GeoNames city to this photo's EXIF GPS (see geonames.py
        # + nearest_city). Lets the UI show a place NAME instead of raw coordinates and
        # lets search filter by city. NULL until the "Match cities" job resolves it.
        if 'city_id' not in files_cols:
            cursor.execute('ALTER TABLE files ADD COLUMN city_id INTEGER')
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
        # Recreated (not IF NOT EXISTS) every init so `f.*` always reflects the
        # current files columns — an existing view is frozen at its creation-time
        # column list, so a newly ALTER-added column (e.g. hidden) wouldn't appear.
        cursor.execute('DROP VIEW IF EXISTS files_with_path')
        cursor.execute('''
            CREATE VIEW files_with_path AS
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
        # Tile embeddings table: per-image grid crops, each CLIP-embedded, backing
        # region search (a query crop can match a small/off-center region that the
        # whole-image embedding would wash out). Derived, rebuildable data keyed by
        # file_id like embeddings/detections/faces/body_embeddings — not part of the
        # checksum-keyed manual ground truth. bbox (x1,y1,x2,y2) is in ORIGINAL image
        # pixels; embedding is raw float32 (D,) np.tobytes(). Wholesale-replaced per
        # file on re-index (see insert_tile_embeddings), so no sentinel/frame_index
        # bookkeeping is needed. CREATE TABLE IF NOT EXISTS is itself the migration —
        # it runs on every init, giving existing DBs the table (and index) too.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tile_embeddings (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                tile_index INTEGER NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                embedding BLOB NOT NULL,
                model TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tile_file ON tile_embeddings(file_id)')
        # Cities: offline GeoNames reference data (cities15000, CC-BY 4.0) for
        # reverse-geocoding a photo's EXIF GPS to the nearest known city NAME
        # (files.city_id → cities.id). Optional/rebuildable — stays empty until
        # `media geo fetch-cities` populates it (see geonames.py). Created here (empty)
        # so nearest_city/find_cities never hit "no such table"; lat/lon are indexed
        # for the bounding-box prefilter nearest_city uses.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                country TEXT,
                admin1 TEXT,
                population INTEGER
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cities_lat ON cities(lat)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cities_lon ON cities(lon)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cities_name ON cities(name)')
        # Categories: single-value ML auto-match result per file, keyed by file_id
        # like detections/faces (derived, rebuildable data). Stores the category as
        # a name string rather than manual.db's numeric category id, since a cross-
        # database foreign key can't be enforced and would go stale silently if
        # manual.db is ever rebuilt — same rationale as faces.identity being a plain
        # string. Manual overrides live in manual.db and always take precedence over
        # this table at read time (see category_resolver.py).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_category_matches (
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                category_name TEXT NOT NULL,
                score REAL NOT NULL,
                model TEXT NOT NULL,
                matched_at INTEGER NOT NULL,
                PRIMARY KEY (file_id, category_name)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_category_matches_name ON file_category_matches (category_name)')
        # file_category_matches' PK changed shape (file_id -> (file_id, category_name)) —
        # categories are multi-valued now, so a file can independently clear threshold
        # for several categories at once instead of the ML matcher picking one winner.
        # Same rebuild-in-place pattern as embeddings' own PK-shape migration above.
        category_matches_pk_cols = {row[1] for row in cursor.execute('PRAGMA table_info(file_category_matches)') if row[5] > 0}
        if category_matches_pk_cols == {'file_id'}:
            cursor.execute('ALTER TABLE file_category_matches RENAME TO file_category_matches_old')
            cursor.execute('''
                CREATE TABLE file_category_matches (
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    category_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    model TEXT NOT NULL,
                    matched_at INTEGER NOT NULL,
                    PRIMARY KEY (file_id, category_name)
                )
            ''')
            cursor.execute('''
                INSERT INTO file_category_matches (file_id, category_name, score, model, matched_at)
                SELECT file_id, category_name, score, model, matched_at FROM file_category_matches_old
            ''')
            cursor.execute('DROP TABLE file_category_matches_old')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_category_matches_name ON file_category_matches (category_name)')

        # Migrate DBs created before frame_index existed (nullable ADD COLUMN is safe
        # in-place — every existing row correctly becomes "primary frame" = NULL).
        detections_cols = {row[1] for row in cursor.execute('PRAGMA table_info(detections)')}
        if 'frame_index' not in detections_cols:
            cursor.execute('ALTER TABLE detections ADD COLUMN frame_index INTEGER')
        faces_cols = {row[1] for row in cursor.execute('PRAGMA table_info(faces)')}
        if 'frame_index' not in faces_cols:
            cursor.execute('ALTER TABLE faces ADD COLUMN frame_index INTEGER')
        # Precomputed "closest known person" for an unidentified face (identity IS NULL):
        # the /find_all_faces review reads the top-scoring ones instead of re-scanning
        # every unidentified face against the named-face matrix on each buffer refill
        # (see compute_face_suggestions). NULL score = not computed yet.
        if 'suggested_identity' not in faces_cols:
            cursor.execute('ALTER TABLE faces ADD COLUMN suggested_identity TEXT')
        if 'suggested_score' not in faces_cols:
            cursor.execute('ALTER TABLE faces ADD COLUMN suggested_score REAL')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_suggested '
                           'ON faces(suggested_score) WHERE identity IS NULL')

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
        # Backs the gallery's default sort ('added' = first_seen DESC, the homepage's
        # default view) — without this, ordering by first_seen forces a full sort of
        # every tracked file before LIMIT/OFFSET can apply, on every single default
        # homepage load (confirmed via EXPLAIN QUERY PLAN: the top-level "USE TEMP
        # B-TREE FOR ORDER BY" step disappears with this index in place, for both
        # ASC and DESC). Doesn't help the 'modified' sort option — that value comes
        # from files_with_path's per-row correlated subquery, not a plain column.
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_first_seen_id ON files (first_seen DESC, id DESC)')
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

    def set_file_hidden(self, file_id, hidden=True):
        """Mark content as hidden (excluded from gallery/library listings) or not.
        Commits — also persists any pending upsert_file_path in the same transaction."""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE files SET hidden = ? WHERE id = ?', (1 if hidden else 0, file_id))
        self.conn.commit()

    def increment_view_count(self, file_id):
        """Record one more open of this photo's /photo page — bumps view_count and
        stamps last_viewed_at. Checksum-scoped (one files row per content), so all
        duplicate paths of the same photo share the count. Drives the home
        'Needs attention' least-viewed-first ordering."""
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE files SET view_count = view_count + 1, last_viewed_at = ? WHERE id = ?',
            (int(time.time()), file_id))
        self.conn.commit()

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
        """List files (id, path, size, checksum), most-recently-seen path shown.
        Hidden (derived, e.g. captured video frames) content is excluded."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, path, size, checksum FROM files_with_path WHERE hidden = 0 LIMIT ?', (limit,))
        return cursor.fetchall()

    def count_files(self, limit=None):
        """Return the number of (non-hidden) content records. limit: maximum rows to count."""
        cur = self.conn.cursor()
        sql = 'SELECT COUNT(*) FROM files WHERE hidden = 0'
        if limit is not None:
            sql += ' LIMIT ?'
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else 0

    def update_file_metadata(self, file_id, taken_at=None, gps_lat=None, gps_lon=None):
        """Store EXIF-derived capture time / GPS coordinates for a file, when present.
        Always stamps metadata_checked_at so a file with no EXIF isn't retried forever.
        When GPS is present and the offline cities table is loaded, also labels the file
        with its nearest city (no-op when cities were never fetched — nearest_city
        returns None)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE files
            SET taken_at = ?, gps_lat = ?, gps_lon = ?, metadata_checked_at = ?
            WHERE id = ?
        ''', (taken_at, gps_lat, gps_lon, int(time.time()), file_id))
        self.conn.commit()
        if gps_lat is not None and gps_lon is not None:
            city = self.nearest_city(gps_lat, gps_lon)
            if city is not None:
                self.set_file_city(file_id, city['id'])

    def get_files_without_metadata(self, limit=None):
        """Return (id, path) for files whose EXIF metadata has never been checked."""
        cursor = self.conn.cursor()
        sql = 'SELECT id, path FROM files_with_path WHERE metadata_checked_at IS NULL'
        if limit is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        return cursor.fetchall()

    def get_geotagged_checksums(self):
        """Checksums of files with EXIF GPS coordinates — the 'has location
        metadata' search filter unions this with manual location assignments."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT checksum FROM files WHERE gps_lat IS NOT NULL')
        return {row[0] for row in cursor.fetchall()}

    # --- Cities (offline GeoNames reverse-geocode; see geonames.py) ---------------

    def replace_cities(self, rows):
        """Wholesale-replace the `cities` table with (name, lat, lon, country, admin1,
        population) tuples — the atomic rebuild step for `media geo fetch-cities`.
        A rebuild changes ids, so any files.city_id is cleared too (it would otherwise
        point at a now-different city); re-run the 'Match cities' job afterwards."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM cities')
        cursor.execute('UPDATE files SET city_id = NULL WHERE city_id IS NOT NULL')
        cursor.executemany(
            'INSERT INTO cities (name, lat, lon, country, admin1, population) '
            'VALUES (?, ?, ?, ?, ?, ?)', rows)
        self.conn.commit()

    def count_cities(self):
        """How many cities are loaded — 0 means the table was never fetched (callers
        surface a 'run media geo fetch-cities' hint)."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM cities')
        return cursor.fetchone()[0]

    def nearest_city(self, lat, lon, max_km=None):
        """The `cities` row nearest to (lat, lon) as a dict, or None if the table is
        empty or nothing is within max_km. A bounding-box prefilter (±0.75°) keeps the
        haversine scan to a handful of candidates instead of the whole table."""
        if lat is None or lon is None:
            return None
        from media_manager.geonames import haversine
        cursor = self.conn.cursor()
        margin = 0.75
        cursor.execute(
            'SELECT id, name, lat, lon, country, admin1, population FROM cities '
            'WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?',
            (lat - margin, lat + margin, lon - margin, lon + margin))
        rows = cursor.fetchall()
        if not rows:
            # Sparse region (e.g. mid-ocean / remote) — fall back to a full scan so a
            # far-but-real nearest city is still found rather than silently returning None.
            cursor.execute('SELECT id, name, lat, lon, country, admin1, population FROM cities')
            rows = cursor.fetchall()
            if not rows:
                return None
        best, best_km = None, None
        for r in rows:
            km = haversine(lat, lon, r[2], r[3])
            if best_km is None or km < best_km:
                best, best_km = r, km
        if max_km is not None and best_km > max_km:
            return None
        return {'id': best[0], 'name': best[1], 'lat': best[2], 'lon': best[3],
                'country': best[4], 'admin1': best[5], 'population': best[6],
                'distance_km': round(best_km, 1)}

    def get_city(self, city_id):
        """One city row as a dict, or None. Used to render a photo's labeled place."""
        if city_id is None:
            return None
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, lat, lon, country, admin1, population '
                       'FROM cities WHERE id = ?', (city_id,))
        r = cursor.fetchone()
        if not r:
            return None
        return {'id': r[0], 'name': r[1], 'lat': r[2], 'lon': r[3],
                'country': r[4], 'admin1': r[5], 'population': r[6]}

    def find_cities(self, prefix, limit=20):
        """Cities whose name starts with `prefix` (case-insensitive), biggest first —
        backs the /api/cities search-palette autocomplete."""
        if not prefix or not prefix.strip():
            return []
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, name, lat, lon, country, admin1, population FROM cities '
            'WHERE name LIKE ? ORDER BY population DESC LIMIT ?',
            (prefix.strip() + '%', limit))
        return [{'id': r[0], 'name': r[1], 'lat': r[2], 'lon': r[3],
                 'country': r[4], 'admin1': r[5], 'population': r[6]}
                for r in cursor.fetchall()]

    def set_file_city(self, file_id, city_id):
        """Store this photo's nearest-city match (files.city_id → cities.id)."""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE files SET city_id = ? WHERE id = ?', (city_id, file_id))
        self.conn.commit()

    def get_geotagged_files_without_city(self, limit=None):
        """(id, gps_lat, gps_lon) for geotagged files not yet matched to a city — the
        'Match cities' bulk-job's work list."""
        cursor = self.conn.cursor()
        sql = ('SELECT id, gps_lat, gps_lon FROM files '
               'WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL AND city_id IS NULL')
        if limit is not None:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        else:
            cursor.execute(sql)
        return cursor.fetchall()

    def get_checksums_for_city(self, city_id):
        """Checksums of every photo labeled with this city — backs the `city:` search
        chip."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT checksum FROM files WHERE city_id = ?', (city_id,))
        return {row[0] for row in cursor.fetchall()}

    def get_used_cities(self):
        """Only the cities that actually label at least one photo, with photo counts —
        [{id, name, country, admin1, file_count}], biggest first. Bounded by the
        library's real diversity (not the whole ~26k table), so it's safe to preload
        into the search palette's `city` facet (mirrors how locations are listed)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT c.id, c.name, c.country, c.admin1, COUNT(f.id) AS file_count
            FROM cities c JOIN files f ON f.city_id = c.id
            GROUP BY c.id ORDER BY file_count DESC
        ''')
        return [{'id': r[0], 'name': r[1], 'country': r[2], 'admin1': r[3], 'file_count': r[4]}
                for r in cursor.fetchall()]

    # --- Service stats (for /stats + imdb sizing) ---------------------------------

    def embedding_stats(self):
        """Row count, vector dimension, and stacked-matrix byte size for each kind of
        embedding — the numbers that decide whether search needs to be offloaded (imdb)
        or an mmap'd index. `dim` is read from a real blob (len//4) so it's exact;
        `bytes` = rows × dim × 4 is the RAM the (N,D) float32 matrix would occupy."""
        cur = self.conn.cursor()

        def stat(count_sql, dim_sql):
            rows = cur.execute(count_sql).fetchone()[0]
            blob = cur.execute(dim_sql).fetchone()
            dim = (blob[0] // 4) if blob and blob[0] else 0
            return {'rows': rows, 'dim': dim, 'bytes': rows * dim * 4}

        return {
            'clip': stat("SELECT COUNT(*) FROM embeddings WHERE frame_index=0",
                         "SELECT length(embedding) FROM embeddings WHERE frame_index=0 "
                         "AND embedding IS NOT NULL LIMIT 1"),
            'face': stat("SELECT COUNT(*) FROM faces WHERE embedding != x''",
                         "SELECT length(embedding) FROM faces WHERE embedding != x'' LIMIT 1"),
            'body': stat("SELECT COUNT(*) FROM body_embeddings WHERE bbox != '[]'",
                         "SELECT length(embedding) FROM body_embeddings "
                         "WHERE bbox != '[]' AND embedding != x'' LIMIT 1"),
            'tile': stat("SELECT COUNT(*) FROM tile_embeddings",
                         "SELECT length(embedding) FROM tile_embeddings LIMIT 1"),
        }

    def loaded_matrix_bytes(self):
        """Actual RAM (bytes) of each embedding matrix currently resident in the
        write-invalidated caches, 0 if not yet built. Best-effort: reaches into the
        cache slots (each `(version, result)`, result a tuple whose last-or-known
        element is the numpy matrix), guarded so a cache-shape change can never break
        /stats."""
        out = {}
        # (label, cache attribute, index of the matrix within the cached result tuple)
        for label, attr, idx in (('clip', '_emb_cache', 2),
                                  ('face', '_face_cache', 2),
                                  ('body', '_body_cache', 3)):
            out[label] = 0
            try:
                slot = getattr(self, attr, None)
                result = slot[1] if slot else None
                if result is not None:
                    out[label] = int(result[idx].nbytes)
            except Exception:
                pass
        return out

    def db_file_bytes(self):
        """Size of this database file on disk, or 0 if missing."""
        return os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

    def iter_clip_embeddings(self, chunk=10000):
        """Stream (file_ids: list[int], vecs: bytes) chunks of whole-image CLIP
        embeddings via fetchmany — used to SHIP the CLIP matrix to the imdb worker
        without ever materializing all ~478 MB on this (low-RAM) host. `vecs` is the
        concatenated little-endian float32 blobs for the chunk's rows, in id order."""
        cur = self.conn.cursor()
        cur.execute('SELECT file_id, embedding FROM embeddings '
                    'WHERE frame_index = 0 AND embedding IS NOT NULL ORDER BY file_id')
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            yield [r[0] for r in rows], b''.join(r[1] for r in rows)

    def iter_face_embeddings(self, chunk=10000):
        """Stream (file_ids: list[int], vecs: bytes) chunks of face (ArcFace)
        embeddings — one row per detected face, `id` is its file_id so a face-search
        result maps straight to the photo. Streams like iter_clip_embeddings so the
        ~444 MB face matrix is never fully resident on the host."""
        cur = self.conn.cursor()
        cur.execute("SELECT file_id, embedding FROM faces "
                    "WHERE embedding != x'' ORDER BY id")
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            yield [r[0] for r in rows], b''.join(r[1] for r in rows)

    def get_geotagged_points(self):
        """(file_id, gps_lat, gps_lon) for every non-hidden EXIF-geotagged file —
        the points plotted on the offline locations map."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, gps_lat, gps_lon FROM files '
            'WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL AND COALESCE(hidden, 0) = 0')
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
        self._emb_ver += 1  # invalidate cached embeddings matrix

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

    @staticmethod
    def _stack_embeddings(blobs, label):
        """Stack a list of float32 embedding BLOBs into one (N, D) float32
        matrix in a single allocation. D is inferred from the first usable
        blob (dim usually 512). Any row whose blob is empty or whose byte
        length != D*4 is dropped so one bad row can't crash the whole matrix
        (shouldn't happen; each drop is printed loudly). Returns
        (matrix, keep) where keep is a bool list aligned with `blobs` marking
        which rows made it into the matrix, so the caller can filter its id
        arrays identically."""
        import numpy as np
        keep = [False] * len(blobs)
        # Infer D from the first non-empty blob.
        D = None
        for b in blobs:
            if b:
                D = len(b) // 4
                break
        if not D:
            return np.empty((0, 0), np.float32), keep
        row_nbytes = D * 4
        good = []
        for i, b in enumerate(blobs):
            if b and len(b) == row_nbytes:
                good.append(b)
                keep[i] = True
            else:
                print(f"[{label}] dropping row index {i}: blob length "
                      f"{len(b) if b else 0} != expected {row_nbytes} (D={D})")
        if not good:
            return np.empty((0, 0), np.float32), keep
        matrix = np.frombuffer(b''.join(good), np.float32).reshape(len(good), D)
        return matrix, keep

    def get_embeddings_matrix(self):
        """Cached, write-invalidated matrix form of get_all_embeddings for
        whole-file similarity search. Returns
        (file_ids: np.ndarray[int64], checksums: list[str], matrix: float32[N, D])
        over the same rows as get_all_embeddings (primary frame_index=0). The
        heavy numpy matrix is built ONCE and reused across requests until an
        embeddings write bumps self._emb_ver — avoiding the per-request
        full-table BLOB fetchall()+rebuild that drives the web-server OOM.
        `path` is intentionally not selected (resolved later for the top-K
        only)."""
        import numpy as np
        with self._matrix_lock:
            if self._emb_cache is not None and self._emb_cache[0] == self._emb_ver:
                return self._emb_cache[1]
            ver = self._emb_ver
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT e.file_id, f.checksum, e.embedding
                FROM embeddings e
                JOIN files_with_path f ON f.id = e.file_id
                WHERE e.frame_index = 0
            ''')
            rows = cursor.fetchall()
            if not rows:
                result = (np.array([], dtype=np.int64), [], np.empty((0, 0), np.float32))
                self._emb_cache = (ver, result)
                return result
            matrix, keep = self._stack_embeddings([r[2] for r in rows], 'embeddings-matrix')
            file_ids = np.array([rows[i][0] for i in range(len(rows)) if keep[i]], dtype=np.int64)
            checksums = [rows[i][1] for i in range(len(rows)) if keep[i]]
            result = (file_ids, checksums, matrix)
            self._emb_cache = (ver, result)
            return result

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
        results (which are checksum-keyed) to current file_id/path rows. Chunked
        (see _chunked) since callers can hand this an unbounded, library-wide
        checksum list, not just a fixed-size page."""
        if not checksums:
            return []
        cursor = self.conn.cursor()
        rows = []
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cursor.execute(f'SELECT * FROM files_with_path WHERE checksum IN ({placeholders})', tuple(chunk))
            rows.extend(cursor.fetchall())
        return rows

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
            WHERE f.hidden = 0
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

    def get_random_faceless_files(self, limit=200):
        """Random sample of files with NO real (non-sentinel) detected face, excluding
        hidden + noface-marked content. Returns (id, checksum) rows. Sampled on the
        base `files` table (no files_with_path correlated-subquery), and the face
        exclusion is an indexed NOT EXISTS (idx_faces_file) — the "needs attention"
        home section's candidate pool. Caller filters these against the set/tag
        exclusion sets from manual.db."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT id, checksum FROM files f
            WHERE f.hidden = 0 AND f.noface = 0
              AND NOT EXISTS (
                  SELECT 1 FROM faces fa
                  WHERE fa.file_id = f.id AND (fa.identity IS NULL OR fa.identity != '__indexed__')
              )
            ORDER BY RANDOM() LIMIT ?
        ''', (limit,))
        return cur.fetchall()

    def get_least_viewed_files(self, limit=400):
        """Least-viewed (non-hidden) files first, as (id, checksum) rows — the home
        'Needs attention' candidate pool. Unlike get_random_faceless_files this does
        NOT exclude photos that have a face: an UNNAMED face still needs attention, so
        eligibility (no name/set/tag/category) is applied by the caller against the
        manual.db exclusion sets. RANDOM() tie-break keeps variety while the whole
        library is still at view_count 0, and photos get viewed as they're worked on,
        so handled ones naturally sink out of the top."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT id, checksum FROM files
            WHERE hidden = 0 AND noface = 0
            ORDER BY view_count ASC, RANDOM() LIMIT ?
        ''', (limit,))
        return cur.fetchall()

    def get_random_file_checksums(self, limit=6):
        """Random (non-hidden) file checksums, sampled on the base files table."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum FROM files WHERE hidden = 0 ORDER BY RANDOM() LIMIT ?', (limit,))
        return [r[0] for r in cur.fetchall()]

    def count_unidentified_faces(self):
        """Cheap count of unidentified real faces (identity IS NULL) — a stat-tile
        approximation of 'unknown faces' that avoids the full row scan + join."""
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM faces WHERE identity IS NULL')
        row = cur.fetchone()
        return row[0] if row else 0

    def get_random_files(self, limit=50):
        """Random sample of (id, path, checksum) rows — bounded by `limit` regardless
        of library size, unlike a full-table pull. Seeds homepage 'in need of some
        love' candidate scanning (see _needs_love_highlight in web.py)."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, path, checksum FROM files_with_path ORDER BY RANDOM() LIMIT ?', (limit,))
        return cursor.fetchall()

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

    def get_detection_bboxes(self, file_id):
        """{class_name: [x1, y1, x2, y2]} for a file's primary-frame detections —
        each class's own highest-confidence box (mirrors get_detected_classes'
        one-representative-entry-per-class shape). Coordinates are absolute
        pixels in the ORIGINAL image (see detector.py's box.xyxy, what
        insert_detections was given) — deliberately not normalized, since the
        one current caller (photo.html's hover-to-highlight) already has the
        loaded <img>'s own naturalWidth/naturalHeight to scale against, the
        same technique the existing drag-to-draw box feature uses."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT class_name, x1, y1, x2, y2 FROM detections "
            "WHERE file_id = ? AND frame_index IS NULL AND class_name != '__indexed__' "
            "AND x1 IS NOT NULL ORDER BY confidence ASC",
            (file_id,)
        )
        # ASC order + dict overwrite = last-write-wins = highest confidence
        # ends up kept, without a second GROUP BY query.
        result = {}
        for class_name, x1, y1, x2, y2 in cursor.fetchall():
            result[class_name] = [x1, y1, x2, y2]
        return result

    def count_detected_classes(self):
        """Distinct auto-detected class names across the whole library (primary-frame
        detections only, sentinel excluded) — mirrors get_detected_classes' per-file
        filter, for the homepage stats section's 'automatic tags' number."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(DISTINCT class_name) FROM detections "
            "WHERE frame_index IS NULL AND class_name != '__indexed__'"
        )
        return cursor.fetchone()[0]

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

    def find_files_under_folder(self, folder_path):
        """
        Return (file_id, path, checksum) rows for every currently-tracked file whose
        stored path is under `folder_path`, matched as a real path-segment prefix (not
        a substring) — 'vacation/john/2023' matches 'vacation/john/2023/img1.jpg' but
        NOT the sibling 'vacation/john/2023-backup/img1.jpg'. Powers "add folder to
        set". Paths are stored relative to data_root with '/' separators (see
        upsert_file_path / fast_scan.py), so folder_path is expected in that same
        form; leading/trailing slashes are stripped from both sides before comparing.

        Queries file_paths directly (joined to files for the checksum) instead of
        through the files_with_path view — that view resolves each file's *primary*
        path via a scalar subquery correlated per files row (see its CREATE VIEW),
        which forces SQLite to evaluate that subquery for every tracked file in the
        whole library before it can even apply this method's path filter, i.e. O(every
        file ever tracked) regardless of how few actually live under folder_path.
        Querying file_paths directly also means a file matches here via ANY of its
        tracked paths, not just its current primary one (e.g. a duplicate copy that
        happens to live under this folder now counts, even if that file's "most
        recently seen" copy is elsewhere) — a strict improvement, not just a perf fix.

        Uses GLOB, not LIKE, for the prefix match: SQLite's LIKE is
        case-INSENSITIVE by default, which disqualifies it from the query
        planner's prefix-to-range-scan rewrite (it can't safely turn a
        case-insensitive comparison into a binary range seek), so a LIKE version
        of this query still fell back to a full index/table scan — GLOB is
        case-sensitive (matching real filesystem path semantics anyway) and
        reliably gets the range-scan treatment. Measured on a 100k-tracked-file
        library: the original files_with_path-view query ~150ms, LIKE directly
        against file_paths ~30ms (still a full scan, just a cheaper one), GLOB
        ~0.5ms (genuine index seek, confirmed via EXPLAIN QUERY PLAN — "SEARCH fp
        USING INDEX idx_file_paths_path (path>? AND path<?)"). folder_path is
        escaped against GLOB's own special characters (`*?[]`) since it's normally
        a real path string, not a pattern.

        Deliberately no GROUP BY here (a file could have two tracked paths both
        under this same folder, which without deduping would return it twice):
        adding one made SQLite's planner flip the join back to scanning `files` as
        the outer loop — same O(whole library) cost this method exists to avoid.
        Deduping in Python instead keeps the query itself a plain indexed seek;
        the second pass only iterates the (typically tiny) match set, not the
        whole library.
        """
        folder_path = (folder_path or '').strip().strip('/')
        if not folder_path:
            return []
        glob_escaped = folder_path.translate(str.maketrans({'*': '[*]', '?': '[?]', '[': '[[]', ']': '[]]'}))
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.id AS id, fp.path AS path, f.checksum AS checksum
            FROM file_paths fp
            JOIN files f ON f.id = fp.file_id
            WHERE (fp.path = ? OR fp.path GLOB ?) AND f.hidden = 0
        ''', (folder_path, f'{glob_escaped}/*'))
        seen_ids = set()
        result = []
        for row in cursor.fetchall():
            if row['id'] in seen_ids:
                continue
            seen_ids.add(row['id'])
            result.append(row)
        return result

    def browse_folder(self, prefix):
        """One level of the tracked-media folder tree under `prefix` (relative to
        data_root, '/'-separated, no leading/trailing slash; '' means the root).
        Returns (subfolders, file_rows):
          subfolders: [{'name', 'path', 'count'}, ...] sorted by name — the immediate
                      child directories, with count = tracked files anywhere under each
                      (recursive), for a "42 files" hint next to each folder.
          file_rows:  (id, path, checksum) Rows for files sitting DIRECTLY in `prefix`.

        Powers the /files browser. Mirrors find_files_under_folder's GLOB-escaping and
        strip('/') normalization; the one-level split is done in Python (like that
        method's Python-side dedup) so the query stays a single indexed range-scan.
        Root ('' prefix) necessarily scans every tracked path — that's inherent to
        enumerating top-level folders — but every non-root call is an indexed
        idx_file_paths_path seek over just that subtree."""
        prefix = (prefix or '').strip().strip('/')
        cur = self.conn.cursor()
        if prefix:
            esc = prefix.translate(str.maketrans({'*': '[*]', '?': '[?]', '[': '[[]', ']': '[]]'}))
            cur.execute('SELECT f.id AS id, fp.path AS path, f.checksum AS checksum '
                        'FROM file_paths fp JOIN files f ON f.id = fp.file_id '
                        'WHERE fp.path GLOB ? AND f.hidden = 0', (f'{esc}/*',))
        else:
            cur.execute('SELECT f.id AS id, fp.path AS path, f.checksum AS checksum '
                        'FROM file_paths fp JOIN files f ON f.id = fp.file_id WHERE f.hidden = 0')
        base = len(prefix) + 1 if prefix else 0
        folders = {}          # child folder name -> recursive file count
        files, seen = [], set()
        for row in cur.fetchall():
            rest = row['path'][base:]
            if '/' in rest:                        # lives in a subfolder of prefix
                seg = rest.split('/', 1)[0]
                folders[seg] = folders.get(seg, 0) + 1
            elif row['id'] not in seen:            # a file directly in prefix
                seen.add(row['id'])
                files.append(row)
        subfolders = [{'name': n, 'path': f'{prefix}/{n}' if prefix else n, 'count': c}
                      for n, c in sorted(folders.items())]
        return subfolders, files

    def remove_paths_under(self, path_prefix):
        """Untrack every currently-tracked path at or under `path_prefix` — a real
        path-segment prefix match, like find_files_under_folder above
        ('vacation/2023' matches 'vacation/2023/img.jpg' but not the sibling
        'vacation/2023-backup/img.jpg'), or an exact single-file path.

        Only removes rows from THIS database (file_paths, and — for any content
        whose last remaining tracked path was just removed — the owning `files`
        row plus every table keyed off its file_id: embeddings/tags/detections/
        faces/body_embeddings/tile_embeddings/file_category_matches). This connection never sets
        `PRAGMA foreign_keys=ON` (see ThreadLocalDB), so the schema's
        `ON DELETE CASCADE` declarations are not actually enforced — cleanup is
        done explicitly here instead of relying on them, to avoid leaving
        orphaned rows referencing a deleted file_id. Content that still has
        another tracked path elsewhere (a duplicate) keeps its `files` row and
        all derived data untouched.

        Deliberately does NOT touch manual.db — that database keys everything
        (tags, favorites, titles, sets, faces/identities) by checksum, not
        file_id, so untracking here doesn't lose any of it: re-adding the same
        content later (a fresh `media add`) reattaches it automatically. Nothing
        on disk is touched either — this is an untrack-only operation, mirroring
        `git rm --cached`, not a delete.

        Returns (paths_removed, files_removed)."""
        path_prefix = (path_prefix or '').strip().strip('/')
        if not path_prefix:
            return (0, 0)
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, file_id FROM file_paths WHERE path = ? OR path LIKE ?',
            (path_prefix, f'{path_prefix}/%')
        )
        rows = cursor.fetchall()
        if not rows:
            return (0, 0)
        file_path_ids = [r[0] for r in rows]
        affected_file_ids = {r[1] for r in rows}
        placeholders = ','.join('?' for _ in file_path_ids)
        cursor.execute(f'DELETE FROM file_paths WHERE id IN ({placeholders})', tuple(file_path_ids))

        files_removed = 0
        for file_id in affected_file_ids:
            cursor.execute('SELECT COUNT(*) FROM file_paths WHERE file_id = ?', (file_id,))
            if cursor.fetchone()[0] > 0:
                continue  # still tracked at another path (a duplicate) — keep its content row
            for table in ('embeddings', 'tags', 'detections', 'faces', 'body_embeddings', 'tile_embeddings', 'file_category_matches'):
                cursor.execute(f'DELETE FROM {table} WHERE file_id = ?', (file_id,))
            cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
            files_removed += 1
        self.conn.commit()
        if files_removed:
            # Deleted rows from embeddings/faces/body_embeddings/tile_embeddings — invalidate all.
            self._emb_ver += 1
            self._face_ver += 1
            self._body_ver += 1
            self._tile_ver += 1
        return (len(file_path_ids), files_removed)

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
        self._face_ver += 1  # invalidate cached face-embeddings matrix

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
        self._face_ver += 1  # invalidate cached face-embeddings matrix
        return cursor.lastrowid

    def get_unface_indexed_files(self, limit=None) -> list:
        """Return (id, path) for files that have no primary (frame_index IS NULL)
        row in faces table — independent of whether frame-specific rows exist.
        Excludes noface=1 files entirely (see set_noface_for_checksums) — a
        .noface-marked photo is never even offered as a detection candidate,
        rather than being detected and its results discarded."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT f.id, f.path
            FROM files_with_path f
            LEFT JOIN faces fa ON fa.file_id = f.id AND fa.frame_index IS NULL
            WHERE fa.file_id IS NULL AND f.noface = 0
        '''
        if limit is not None:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        else:
            cursor.execute(sql)
        return cursor.fetchall()

    def set_noface_for_checksums(self, checksums):
        """Mark every one of `checksums` as noface=1 (see get_unface_indexed_files) —
        applied by folder_markers.py for every file under a .noface-marked folder."""
        if not checksums:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in checksums)
        cursor.execute(f'UPDATE files SET noface = 1 WHERE checksum IN ({placeholders})', tuple(checksums))
        self.conn.commit()

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

    def get_face_embeddings_matrix(self):
        """Cached, write-invalidated matrix form of get_all_face_embeddings.
        Returns (face_ids: np.ndarray[int64], file_ids: np.ndarray[int64],
        matrix: float32[N, D]) over the same non-sentinel face rows
        (identity != '__indexed__' and embedding != b''). Built once and reused
        across requests until a faces write bumps self._face_ver. `path` is
        dropped (resolved later for the top-K only)."""
        import numpy as np
        with self._matrix_lock:
            if self._face_cache is not None and self._face_cache[0] == self._face_ver:
                return self._face_cache[1]
            ver = self._face_ver
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT fa.id, fa.file_id, fa.embedding
                FROM faces fa
                JOIN files_with_path f ON f.id = fa.file_id
                WHERE (fa.identity IS NULL OR fa.identity != '__indexed__') AND fa.embedding != x''
            ''')
            rows = cursor.fetchall()
            if not rows:
                result = (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                          np.empty((0, 0), np.float32))
                self._face_cache = (ver, result)
                return result
            matrix, keep = self._stack_embeddings([r[2] for r in rows], 'face-embeddings-matrix')
            face_ids = np.array([rows[i][0] for i in range(len(rows)) if keep[i]], dtype=np.int64)
            file_ids = np.array([rows[i][1] for i in range(len(rows)) if keep[i]], dtype=np.int64)
            result = (face_ids, file_ids, matrix)
            self._face_cache = (ver, result)
            return result

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

    # --- Precomputed face suggestions (see web.py compute_face_suggestions) --------

    def count_unsuggested_faces(self):
        """How many unidentified faces still need a suggestion computed (NULL score)."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faces WHERE identity IS NULL "
                    "AND suggested_score IS NULL AND embedding != x''")
        return cur.fetchone()[0]

    def iter_unsuggested_faces(self, chunk=5000):
        """Stream (face_ids: list[int], vecs: bytes) chunks of unidentified faces whose
        suggestion hasn't been computed yet — for the chunked, vectorized
        compute_face_suggestions (never materializes all ~227k face embeddings at once)."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, embedding FROM faces WHERE identity IS NULL "
                    "AND suggested_score IS NULL AND embedding != x'' ORDER BY id")
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            yield [r[0] for r in rows], b''.join(r[1] for r in rows)

    def set_face_suggestions(self, updates):
        """Batch-write precomputed (suggested_identity, suggested_score) — updates is a
        list of (identity, score, face_id) tuples."""
        cur = self.conn.cursor()
        cur.executemany('UPDATE faces SET suggested_identity = ?, suggested_score = ? WHERE id = ?',
                        updates)
        self.conn.commit()

    def clear_face_suggestions(self):
        """Reset all unidentified faces' suggestions (NULL) so the next compute recomputes
        everyone — used after the named-people set changes."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET suggested_identity = NULL, suggested_score = NULL '
                    'WHERE identity IS NULL')
        self.conn.commit()

    def get_suggested_unidentified_faces(self, threshold, limit):
        """Top unidentified faces by precomputed suggested_score (>= threshold), best
        first — the /find_all_faces review pool. Returns (face_id, file_id, path, bbox,
        suggested_identity, suggested_score). Caller drops promoted / already-buffered
        refs. Instant vs the old per-refill full-library scan."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT fa.id, fa.file_id, f.path, fa.bbox, fa.suggested_identity, fa.suggested_score
            FROM faces fa
            JOIN files_with_path f ON f.id = fa.file_id
            WHERE fa.identity IS NULL AND fa.suggested_score >= ?
            ORDER BY fa.suggested_score DESC
            LIMIT ?
        ''', (threshold, limit))
        return cur.fetchall()

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
        self._body_ver += 1  # invalidate cached body-embeddings matrix

    def add_manual_body(self, file_id: int, bbox: list, embedding_bytes: bytes, model: str) -> int:
        """Append one manually-drawn body crop (primary frame) and return its row id.
        Clears the 'no people' sentinel (bbox='[]') first so the manual crop isn't
        masked, but keeps any real body rows (unlike insert_body_embeddings, which
        replaces them wholesale). Mirrors add_manual_face for find-by-body."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM body_embeddings WHERE file_id = ? AND bbox = '[]' AND frame_index IS NULL",
            (file_id,)
        )
        cursor.execute(
            'INSERT INTO body_embeddings (file_id, bbox, embedding, model, indexed_at) VALUES (?,?,?,?,?)',
            (file_id, json.dumps(bbox), embedding_bytes, model, int(time.time()))
        )
        self.conn.commit()
        self._body_ver += 1  # invalidate cached body-embeddings matrix
        return cursor.lastrowid

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

    def get_body_embeddings_matrix(self):
        """Cached, write-invalidated matrix form of get_all_body_embeddings.
        Returns (body_ids: np.ndarray[int64], file_ids: np.ndarray[int64],
        bboxes: list, matrix: float32[N, D]) over the same non-sentinel body
        rows (bbox != '[]' and embedding != b''). Built once and reused across
        requests until a body write bumps self._body_ver. `path` is dropped
        (resolved later for the top-K only)."""
        import numpy as np
        with self._matrix_lock:
            if self._body_cache is not None and self._body_cache[0] == self._body_ver:
                return self._body_cache[1]
            ver = self._body_ver
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT b.id, b.file_id, b.bbox, b.embedding
                FROM body_embeddings b
                JOIN files_with_path f ON f.id = b.file_id
                WHERE b.bbox != '[]' AND b.embedding != x''
            ''')
            rows = cursor.fetchall()
            if not rows:
                result = (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                          [], np.empty((0, 0), np.float32))
                self._body_cache = (ver, result)
                return result
            matrix, keep = self._stack_embeddings([r[3] for r in rows], 'body-embeddings-matrix')
            body_ids = np.array([rows[i][0] for i in range(len(rows)) if keep[i]], dtype=np.int64)
            file_ids = np.array([rows[i][1] for i in range(len(rows)) if keep[i]], dtype=np.int64)
            bboxes = [rows[i][2] for i in range(len(rows)) if keep[i]]
            result = (body_ids, file_ids, bboxes, matrix)
            self._body_cache = (ver, result)
            return result

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

    # ------------------------------------------------------------------
    # Tile embeddings methods (region search)
    # ------------------------------------------------------------------

    def insert_tile_embeddings(self, file_id: int, tiles: list, model: str) -> None:
        """Replace the tile-embedding rows for a file (idempotent re-index).
        tiles: list of {'bbox': [x1,y1,x2,y2], 'embedding': np.ndarray} — one per
        grid crop, bbox in ORIGINAL image pixels. DELETEs any existing rows for the
        file first so a re-index never leaves stale/duplicate tiles, then inserts
        each with tile_index = its position in the list. Mirrors
        insert_body_embeddings; unlike it there is no sentinel row — a file with no
        tiles simply gets no rows (get_untiled_files re-queues it), which is fine
        because the tile scan skips non-images rather than marking them done here."""
        import numpy as np
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM tile_embeddings WHERE file_id = ?', (file_id,))
        for tile_index, tile in enumerate(tiles):
            x1, y1, x2, y2 = tile['bbox']
            cursor.execute(
                'INSERT INTO tile_embeddings (file_id, tile_index, x1, y1, x2, y2, embedding, model) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (file_id, tile_index, x1, y1, x2, y2,
                 tile['embedding'].astype(np.float32).tobytes(), model)
            )
        self.conn.commit()
        self._tile_ver += 1  # invalidate anything keyed off the tile version

    def count_tiled_files(self) -> int:
        """Number of distinct files that have at least one tile-embedding row.
        Used to decide tile-vs-whole-image search and to show a "pending" status."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(DISTINCT file_id) FROM tile_embeddings')
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_tiles_for_file(self, file_id):
        """(x1, y1, x2, y2, embedding_bytes) for every tile of one file, in tile
        order. Used to localize a whole-image match to its best sub-region (e.g.
        the swipe fullview highlight for centroid tag suggestions). [] if untiled."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT x1, y1, x2, y2, embedding FROM tile_embeddings '
            'WHERE file_id = ? ORDER BY tile_index', (file_id,))
        return cursor.fetchall()

    def get_untiled_files(self, limit=None) -> list:
        """Return (id, path) for tracked files that have NO tile_embeddings row.
        Mirrors get_unbody_indexed_files; the caller skips non-images, so there is
        no kind filter here."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT f.id, f.path
            FROM files_with_path f
            LEFT JOIN tile_embeddings t ON t.file_id = f.id
            WHERE t.file_id IS NULL
        '''
        if limit is not None:
            cursor.execute(sql + ' LIMIT ?', (limit,))
        else:
            cursor.execute(sql)
        return cursor.fetchall()

    def iter_tile_embeddings(self, batch_size=20000):
        """Yield (file_ids: np.ndarray[int64], matrix: np.ndarray[k, D] float32)
        chunks over ALL tile rows, ordered by id, up to batch_size rows per chunk.

        This is the RAM-bounded region-search primitive: the library may hold
        hundreds of thousands of tiles, so — unlike get_body_embeddings_matrix —
        we deliberately never build or cache one full matrix. Each chunk's blobs
        are joined and reshaped in a single allocation (D inferred from the first
        blob length // 4). A malformed/empty blob is skipped with a printed
        WARNING (the scan keeps going), and file_ids stay aligned to matrix rows
        because the skipped row is dropped from both."""
        import numpy as np
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, file_id, embedding FROM tile_embeddings ORDER BY id')
        batch_fids = []
        batch_blobs = []
        D = None

        def _flush():
            nonlocal D
            fids = np.array(batch_fids, dtype=np.int64)
            matrix = np.frombuffer(b''.join(batch_blobs), np.float32).reshape(len(batch_blobs), D)
            return fids, matrix

        for row_id, file_id, blob in cursor:
            if not blob or (len(blob) % 4) != 0:
                print(f"[iter_tile_embeddings] WARNING: skipping tile row id {row_id} "
                      f"(file_id {file_id}): blob length {len(blob) if blob else 0} not a float32 vector")
                continue
            row_D = len(blob) // 4
            if D is None:
                D = row_D
            elif row_D != D:
                print(f"[iter_tile_embeddings] WARNING: skipping tile row id {row_id} "
                      f"(file_id {file_id}): D={row_D} != expected D={D}")
                continue
            batch_fids.append(file_id)
            batch_blobs.append(blob)
            if len(batch_blobs) >= batch_size:
                yield _flush()
                batch_fids = []
                batch_blobs = []
        if batch_blobs:
            yield _flush()

    def has_object_detections(self, file_id: int) -> bool:
        """True if `media index` has already processed this file — a primary-frame
        detections row exists at all, even if it's just the sentinel meaning
        "confirmed zero classes". Lets body_index.py tell "never object-indexed"
        (worth a fresh person-only detection pass) apart from "object-indexed,
        genuinely no person" (trust that, don't re-detect)."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT 1 FROM detections WHERE file_id = ? AND frame_index IS NULL LIMIT 1',
            (file_id,)
        )
        return cursor.fetchone() is not None

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

    # ------------------------------------------------------------------
    # Category matches (ML auto-classification — see category_resolver.py for
    # how this is merged with manual.db's manual overrides, which always win)
    # ------------------------------------------------------------------

    def set_file_category_match(self, file_id, category_name, score, model):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO file_category_matches (file_id, category_name, score, model, matched_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_id, category_name, score, model, int(time.time())))
        self.conn.commit()

    def clear_file_category_match(self, file_id):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM file_category_matches WHERE file_id = ?', (file_id,))
        self.conn.commit()

    def get_file_category_matches(self, file_id):
        """Every ML auto-match for this file — a file can independently clear
        threshold for several categories at once now, so this returns a list,
        not a single best match."""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT category_name, score, model FROM file_category_matches WHERE file_id = ?',
            (file_id,)
        )
        return cursor.fetchall()

    def get_file_category_matches_for_files(self, file_ids):
        """Batched lookup: {file_id: [(category_name, score, model), ...]} — mirrors
        get_embeddings_for_files, but list-valued now (see get_file_category_matches).
        Chunked (see _chunked) since callers can hand this an unbounded, library-wide
        file_id list, not just a fixed-size page."""
        if not file_ids:
            return {}
        cursor = self.conn.cursor()
        result = {}
        for chunk in self._chunked(file_ids):
            placeholders = ','.join('?' for _ in chunk)
            cursor.execute(
                f'SELECT file_id, category_name, score, model FROM file_category_matches WHERE file_id IN ({placeholders})',
                tuple(chunk)
            )
            for file_id, category_name, score, model in cursor.fetchall():
                result.setdefault(file_id, []).append((category_name, score, model))
        return result

    def get_all_file_category_matches(self):
        """Return [(file_id, checksum, category_name, score), ...] joined for
        checksum — used by category_resolver's navbar-count/search-by-category
        helpers, mirrors get_all_embeddings's join-for-checksum style."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT m.file_id, f.checksum, m.category_name, m.score
            FROM file_category_matches m
            JOIN files_with_path f ON f.id = m.file_id
        ''')
        return cursor.fetchall()

    def clear_primary_ml_data(self, file_id, skip_faces=False):
        """Delete primary-frame (frame_index NULL/0) detections, faces, body
        embeddings, and the frame-0 embedding for a file, so the next
        index/embed/faces run reprocesses it from scratch. Used by 'media
        add/commit --reindex' for files that were already tracked at the same path.
        Frame-specific rows from 'scan all frames' are left untouched.

        skip_faces=True leaves the faces table alone entirely — for a file
        with a manual.db face decision already on it, wiping and letting
        `media faces` regenerate fresh auto-detected rows would break the
        source_face_id link that ties the decision to a specific detection,
        orphaning it and letting the same physical face resurface as a brand
        new, undecided candidate. Everything else (detections/embeddings/body)
        still reindexes normally — only face detection is locked."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        if not skip_faces:
            cursor.execute('DELETE FROM faces WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        cursor.execute('DELETE FROM embeddings WHERE file_id = ? AND frame_index = 0', (file_id,))
        cursor.execute('DELETE FROM body_embeddings WHERE file_id = ? AND frame_index IS NULL', (file_id,))
        self.conn.commit()
        # Invalidate all cached matrices: this cleared embeddings, body, and
        # (unless skip_faces) faces rows for the file.
        self._emb_ver += 1
        self._body_ver += 1
        if not skip_faces:
            self._face_ver += 1

    def close(self):
        """Close the database connection."""
        self.conn.close()
