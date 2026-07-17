"""Manual/hand-labeled data — deliberately its own sqlite file (.media/manual.db),
independent of media.db, because it is the ground truth for future fine-tuning
(YOLO / face-recognition) and must never be touched by bulk ML re-indexing.

Every table here identifies a file by its content checksum (media.db's files.checksum),
never by media.db's files.id. That id is an arbitrary AUTOINCREMENT value scoped to
whatever media.db happens to exist right now — it's reassigned on every rescan order
change and has no meaning at all if media.db is ever rebuilt from scratch. Checksum is
the one thing that's actually stable: the same photo bytes always hash the same way,
so a tag/face/set-membership recorded against a checksum stays correctly attached to
that photo forever, independent of anything that ever happens to media.db."""
import time

from media_manager.database import ThreadLocalDB

# Cosine similarity above which a newly detected face is auto-assigned to a known
# identity without asking — deliberately stricter than the 0.45 "suggest a match and
# let a human confirm" threshold used elsewhere, since auto-assignment has no human
# in the loop to catch a wrong guess.
AUTO_MATCH_THRESHOLD = 0.6


class ManualDB(ThreadLocalDB):
    def __init__(self, db_path):
        super().__init__(db_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        # Fail loudly if this is an old file_id-keyed manual.db, rather than letting
        # 'CREATE TABLE IF NOT EXISTS' silently no-op and crash confusingly on the
        # first query that references the (missing) checksum column.
        existing_tag_cols = {row[1] for row in cur.execute('PRAGMA table_info(tags)')}
        if 'file_id' in existing_tag_cols and 'checksum' not in existing_tag_cols:
            raise RuntimeError(
                "This manual.db is still on the old file_id-keyed schema (file_id "
                "referenced media.db's arbitrary, rebuild-unstable row id). There is "
                "no automatic migration for this — file_id alone can't be reliably "
                "mapped to a checksum after the fact. If you still have the old "
                "media.db this manual.db was created against, a one-off script could "
                "resolve file_id -> checksum from it before rebuilding either db. "
                "Otherwise, this manual.db needs to be recreated from scratch."
            )

        cur.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checksum TEXT NOT NULL,
                label TEXT NOT NULL,
                polarity TEXT NOT NULL DEFAULT 'positive',
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                image_width INTEGER, image_height INTEGER,
                created_at INTEGER NOT NULL,
                frame_index INTEGER,
                favorite INTEGER NOT NULL DEFAULT 0
            )
        ''')
        # Migrate manual.db files that predate frame_index (nullable ADD COLUMN is
        # safe in-place — every existing row correctly becomes "whole file", NULL).
        existing_tag_cols2 = {row[1] for row in cur.execute('PRAGMA table_info(tags)')}
        if 'frame_index' not in existing_tag_cols2:
            cur.execute('ALTER TABLE tags ADD COLUMN frame_index INTEGER')
        if 'favorite' not in existing_tag_cols2:
            cur.execute('ALTER TABLE tags ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_favorites (
                checksum TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_titles (
                checksum TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        ''')
        # Experimental age/gender estimation (MiVOLO, run in an isolated venv — see
        # age_estimator.py) — deliberately its own table, not touching files/faces at
        # all, so the whole feature can be dropped with a single DROP TABLE if it
        # doesn't work out. checksum+face_ref are logical references (the same
        # "auto:<id>"/"manual:<id>" strings used everywhere else for faces), not a
        # SQL foreign key, since a face's real row lives in one of two different
        # tables depending on whether it's auto-detected or manually confirmed.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS face_age_estimates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checksum TEXT NOT NULL,
                face_ref TEXT NOT NULL,
                age REAL,
                gender TEXT,
                model TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(face_ref, model)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_age_estimates_checksum ON face_age_estimates (checksum)')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                studio TEXT,
                created_at INTEGER NOT NULL,
                favorite INTEGER NOT NULL DEFAULT 0,
                UNIQUE(name, studio)
            )
        ''')
        existing_set_cols = {row[1] for row in cur.execute('PRAGMA table_info(sets)')}
        if 'favorite' not in existing_set_cols:
            cur.execute('ALTER TABLE sets ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_sets (
                checksum TEXT NOT NULL,
                set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (checksum, set_id)
            )
        ''')
        # file_sets used to have a bare UNIQUE on checksum, limiting a file to one set.
        # SQLite can't ALTER a table's primary key in place, so rebuild it if it's still
        # in that old single-set-per-file shape (detected by the composite PK being
        # absent — a freshly created table above already has it).
        file_sets_pk_cols = {row[1] for row in cur.execute('PRAGMA table_info(file_sets)') if row[5] > 0}
        if file_sets_pk_cols != {'checksum', 'set_id'}:
            cur.execute('ALTER TABLE file_sets RENAME TO file_sets_old')
            cur.execute('''
                CREATE TABLE file_sets (
                    checksum TEXT NOT NULL,
                    set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (checksum, set_id)
                )
            ''')
            cur.execute('''
                INSERT INTO file_sets (checksum, set_id, created_at)
                SELECT checksum, set_id, created_at FROM file_sets_old
            ''')
            cur.execute('DROP TABLE file_sets_old')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checksum TEXT NOT NULL,
                identity TEXT,
                x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
                embedding BLOB NOT NULL,
                bbox_source TEXT NOT NULL,
                source_face_id INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                rejected INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                frame_index INTEGER,
                favorite INTEGER NOT NULL DEFAULT 0
            )
        ''')
        existing_face_cols2 = {row[1] for row in cur.execute('PRAGMA table_info(faces)')}
        if 'frame_index' not in existing_face_cols2:
            cur.execute('ALTER TABLE faces ADD COLUMN frame_index INTEGER')
        if 'favorite' not in existing_face_cols2:
            cur.execute('ALTER TABLE faces ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS migration_state (
                key TEXT PRIMARY KEY,
                done_at INTEGER NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_manual_tags_checksum ON tags (checksum)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_manual_tags_label ON tags (label)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_manual_faces_checksum ON faces (checksum)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_manual_faces_identity ON faces (identity)')
        cur.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_manual_faces_source '
            'ON faces (source_face_id) WHERE source_face_id IS NOT NULL'
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_tag(self, checksum, label, polarity='positive', frame_index=None):
        """Whole-image tag (no bbox). polarity='negative' records a human correction —
        e.g. YOLO detected "car" but there is no car — so it can be excluded from search
        and used to suppress that false-positive detection chip in the UI. frame_index
        set means this tag applies to that one frame of an animated file, not the file
        as a whole."""
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO tags (checksum, label, polarity, x1,y1,x2,y2, image_width,image_height, created_at, frame_index) '
            'VALUES (?,?,?,NULL,NULL,NULL,NULL,NULL,NULL,?,?)',
            (checksum, label.strip(), polarity, int(time.time()), frame_index)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_spatial_tag(self, checksum, label, x1, y1, x2, y2, image_width, image_height, frame_index=None):
        """A tag scoped to a region — a YOLO-style (image, bbox, label) training
        example. frame_index set means this region is on that specific frame of an
        animated file."""
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO tags (checksum, label, polarity, x1,y1,x2,y2, image_width,image_height, created_at, frame_index) '
            "VALUES (?,?,'positive',?,?,?,?,?,?,?,?)",
            (checksum, label.strip(), x1, y1, x2, y2, image_width, image_height, int(time.time()), frame_index)
        )
        self.conn.commit()
        return cur.lastrowid

    def remove_tag(self, tag_id):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM tags WHERE id = ?', (tag_id,))
        self.conn.commit()

    def update_tag_label(self, tag_id, new_label):
        """Fix a typo in a tag's label in place, without a delete-and-re-add round trip."""
        cur = self.conn.cursor()
        cur.execute('UPDATE tags SET label = ? WHERE id = ?', (new_label.strip(), tag_id))
        self.conn.commit()

    def set_tag_favorite(self, tag_id, favorite):
        cur = self.conn.cursor()
        cur.execute('UPDATE tags SET favorite = ? WHERE id = ?', (1 if favorite else 0, tag_id))
        self.conn.commit()

    def rename_tag_label(self, old_label, new_label):
        """Renames a label everywhere it's used (every tag row with this label, any
        photo, any polarity) — the label-level equivalent of rename_identity, so
        fixing a typo doesn't require editing each photo's tag individually."""
        cur = self.conn.cursor()
        cur.execute('UPDATE tags SET label = ? WHERE label = ?', (new_label.strip(), old_label))
        self.conn.commit()

    def delete_tag_label(self, label):
        """Removes every tag row with this label, across every photo."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM tags WHERE label = ?', (label,))
        self.conn.commit()

    def get_tags(self, checksum):
        """Return full tag rows (id, label, polarity, x1..y2, image_width, image_height) for a file."""
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM tags WHERE checksum = ? ORDER BY id', (checksum,))
        return cur.fetchall()

    def get_whole_image_tag_labels(self, checksum):
        """Positive whole-file (non-spatial, non-frame-scoped) tag labels — feeds
        card/nav display, never includes negative tags since those are corrections,
        not "this file has X". Frame-scoped tags don't show up in this default list."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT label FROM tags WHERE checksum = ? AND x1 IS NULL AND frame_index IS NULL "
            "AND polarity = 'positive' ORDER BY label",
            (checksum,)
        )
        return [row[0] for row in cur.fetchall()]

    def get_negated_labels(self, checksum):
        """Labels a human has explicitly rejected for this file — used to suppress a
        matching auto-detected class from showing up as a "Detected objects" chip."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT label FROM tags WHERE checksum = ? AND polarity = 'negative'",
            (checksum,)
        )
        return {row[0] for row in cur.fetchall()}

    def get_files_by_tag(self, label, limit=100):
        """Positive matches only — a negative tag is a rejection, it must never match search.
        Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checksum FROM tags WHERE label = ? AND polarity = 'positive' ORDER BY checksum LIMIT ?",
            (label, limit)
        )
        return [row[0] for row in cur.fetchall()]

    def list_all_tags(self):
        """Return [(label, count), ...] for positive tags only, ordered by count descending."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT label, COUNT(*) as cnt FROM tags WHERE polarity = 'positive' GROUP BY label ORDER BY cnt DESC"
        )
        return cur.fetchall()

    def list_tags_for_checksums(self, checksums):
        """Batched positive whole-file (non-frame-scoped) tag lookup:
        {checksum: [label, ...]}. Avoids N+1 queries on list pages."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT checksum, label FROM tags WHERE x1 IS NULL AND frame_index IS NULL AND polarity = 'positive' "
            f'AND checksum IN ({placeholders}) ORDER BY label',
            tuple(checksums)
        )
        result = {}
        for checksum, label in cur.fetchall():
            result.setdefault(checksum, []).append(label)
        return result

    def get_all_positive_labels(self):
        """Distinct positive tag labels (whole-image + spatial) — expands the YOLO-World
        vocabulary with every object a human has confirmed, so it starts looking for
        things it wasn't originally told about."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT label FROM tags WHERE polarity = 'positive' ORDER BY label")
        return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # File favorites
    # ------------------------------------------------------------------

    def set_file_favorite(self, checksum, favorite):
        cur = self.conn.cursor()
        if favorite:
            cur.execute(
                'INSERT OR IGNORE INTO file_favorites (checksum, created_at) VALUES (?, ?)',
                (checksum, int(time.time()))
            )
        else:
            cur.execute('DELETE FROM file_favorites WHERE checksum = ?', (checksum,))
        self.conn.commit()

    def get_favorite_checksums(self, checksums):
        """Batched lookup: subset of `checksums` that are favorited."""
        if not checksums:
            return set()
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f'SELECT checksum FROM file_favorites WHERE checksum IN ({placeholders})',
            tuple(checksums)
        )
        return {row[0] for row in cur.fetchall()}

    def is_file_favorite(self, checksum):
        cur = self.conn.cursor()
        cur.execute('SELECT 1 FROM file_favorites WHERE checksum = ?', (checksum,))
        return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # File titles
    # ------------------------------------------------------------------

    def set_file_title(self, checksum, title):
        """A human-given title for a photo, independent of its filename. An empty/
        blank title clears it back to "no title" rather than storing an empty row."""
        cur = self.conn.cursor()
        title = title.strip() if title else ''
        if title:
            cur.execute('''
                INSERT INTO file_titles (checksum, title, created_at) VALUES (?, ?, ?)
                ON CONFLICT(checksum) DO UPDATE SET title=excluded.title
            ''', (checksum, title, int(time.time())))
        else:
            cur.execute('DELETE FROM file_titles WHERE checksum = ?', (checksum,))
        self.conn.commit()

    def get_file_title(self, checksum):
        cur = self.conn.cursor()
        cur.execute('SELECT title FROM file_titles WHERE checksum = ?', (checksum,))
        row = cur.fetchone()
        return row[0] if row is not None else None

    def get_titles_for_checksums(self, checksums):
        """Batched lookup: {checksum: title} for whichever of `checksums` have one —
        avoids N+1 queries on list pages (mirrors list_tags_for_checksums)."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f'SELECT checksum, title FROM file_titles WHERE checksum IN ({placeholders})',
            tuple(checksums)
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Age/gender estimates (experimental — see age_estimator.py)
    # ------------------------------------------------------------------

    def save_age_estimates(self, checksum, results, model):
        """Upsert each {'face_ref', 'age', 'gender'} result — re-running the estimate
        on the same face updates its row in place rather than piling up duplicates."""
        cur = self.conn.cursor()
        now = int(time.time())
        for r in results:
            cur.execute('''
                INSERT INTO face_age_estimates (checksum, face_ref, age, gender, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(face_ref, model) DO UPDATE SET age=excluded.age, gender=excluded.gender,
                    checksum=excluded.checksum, created_at=excluded.created_at
            ''', (checksum, r['face_ref'], r.get('age'), r.get('gender'), model, now))
        self.conn.commit()

    def get_age_estimates_for_checksum(self, checksum):
        cur = self.conn.cursor()
        cur.execute(
            'SELECT face_ref, age, gender, model FROM face_age_estimates WHERE checksum = ?',
            (checksum,)
        )
        return cur.fetchall()

    def get_average_ages_for_checksums(self, checksums):
        """Batched lookup: {checksum: average age} across every face with an estimate
        on that photo — feeds "sort by age" on gallery-style views. A photo with no
        estimate at all is simply absent from the result (not 0)."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f'''SELECT checksum, AVG(age) FROM face_age_estimates
                WHERE checksum IN ({placeholders}) AND age IS NOT NULL
                GROUP BY checksum''',
            tuple(checksums)
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------

    def create_set(self, name, studio=None):
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR IGNORE INTO sets (name, studio, created_at) VALUES (?, ?, ?)',
            (name.strip(), studio.strip() if studio else None, int(time.time()))
        )
        self.conn.commit()
        cur.execute(
            'SELECT id FROM sets WHERE name = ? AND studio IS ?',
            (name.strip(), studio.strip() if studio else None)
        )
        return cur.fetchone()[0]

    def find_set(self, name, studio=None):
        cur = self.conn.cursor()
        cur.execute(
            'SELECT * FROM sets WHERE name = ? AND studio IS ?',
            (name, studio)
        )
        return cur.fetchone()

    def get_set(self, set_id):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM sets WHERE id = ?', (set_id,))
        return cur.fetchone()

    def list_sets(self, favorite_only=False, studio=None):
        cur = self.conn.cursor()
        clauses = []
        params = []
        if favorite_only:
            clauses.append('s.favorite = 1')
        if studio is not None:
            clauses.append('s.studio = ?')
            params.append(studio)
        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        cur.execute(f'''
            SELECT s.id, s.name, s.studio, s.created_at, s.favorite, COUNT(fs.checksum) as image_count
            FROM sets s
            LEFT JOIN file_sets fs ON fs.set_id = s.id
            {where}
            GROUP BY s.id
            ORDER BY s.name
        ''', tuple(params))
        return cur.fetchall()

    def list_studios(self):
        """Distinct studios across all sets, with set count and total image count —
        feeds the Studios wall page."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT s.studio as studio, COUNT(DISTINCT s.id) as set_count, COUNT(fs.checksum) as image_count
            FROM sets s
            LEFT JOIN file_sets fs ON fs.set_id = s.id
            WHERE s.studio IS NOT NULL AND s.studio != ''
            GROUP BY s.studio
            ORDER BY s.studio
        ''')
        return cur.fetchall()

    def rename_set(self, set_id, name, studio=None):
        cur = self.conn.cursor()
        cur.execute(
            'UPDATE sets SET name = ?, studio = ? WHERE id = ?',
            (name.strip(), studio.strip() if studio else None, set_id)
        )
        self.conn.commit()

    def delete_set(self, set_id):
        """Deletes the set entity only — member files are untouched, `file_sets` rows
        for it are removed automatically via ON DELETE CASCADE."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM sets WHERE id = ?', (set_id,))
        self.conn.commit()

    def set_set_favorite(self, set_id, favorite):
        cur = self.conn.cursor()
        cur.execute('UPDATE sets SET favorite = ? WHERE id = ?', (1 if favorite else 0, set_id))
        self.conn.commit()

    def assign_file_to_set(self, checksum, set_id):
        """A file can belong to any number of sets — this adds a membership, it never
        replaces one."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO file_sets (checksum, set_id, created_at) VALUES (?, ?, ?)
        ''', (checksum, set_id, int(time.time())))
        self.conn.commit()

    def remove_file_from_set(self, checksum, set_id):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_sets WHERE checksum = ? AND set_id = ?', (checksum, set_id))
        self.conn.commit()

    def get_files_by_set(self, set_id, limit=200):
        """Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum FROM file_sets WHERE set_id = ? LIMIT ?', (set_id, limit))
        return [row[0] for row in cur.fetchall()]

    def get_sets_for_file(self, checksum):
        """Every set this file belongs to (a file can be in more than one)."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT s.* FROM sets s JOIN file_sets fs ON fs.set_id = s.id
            WHERE fs.checksum = ? ORDER BY s.name
        ''', (checksum,))
        return cur.fetchall()

    def get_sets_for_checksums(self, checksums):
        """Batched lookup: {checksum: [{'id', 'name', 'studio'}, ...]} — avoids N+1
        queries on list pages (mirrors list_tags_for_checksums)."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f'''SELECT fs.checksum, s.id, s.name, s.studio
                FROM file_sets fs JOIN sets s ON s.id = fs.set_id
                WHERE fs.checksum IN ({placeholders}) ORDER BY s.name''',
            tuple(checksums)
        )
        result = {}
        for checksum, set_id, name, studio in cur.fetchall():
            result.setdefault(checksum, []).append({'id': set_id, 'name': name, 'studio': studio})
        return result

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------

    def add_manual_face(self, checksum, bbox, embedding_bytes, image_width, image_height, frame_index=None):
        """A hand-drawn face box — entirely a human annotation. frame_index set means
        this box was drawn on that specific frame of an animated file."""
        x1, y1, x2, y2 = bbox
        cur = self.conn.cursor()
        cur.execute('''
            INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source, source_face_id,
                                image_width, image_height, created_at, frame_index)
            VALUES (?, NULL, ?,?,?,?, ?, 'manual', NULL, ?, ?, ?, ?)
        ''', (checksum, x1, y1, x2, y2, embedding_bytes, image_width, image_height, int(time.time()), frame_index))
        self.conn.commit()
        return cur.lastrowid

    def promote_auto_face(self, source_face_id, checksum, bbox, embedding_bytes, identity,
                           image_width, image_height, frame_index=None):
        """Naming an auto-detected face copies it into manual.db (upsert, keyed on source_face_id
        so renaming the same auto-detected face twice doesn't create a duplicate row). Clears
        any prior rejection — naming a face after all is a stronger, more recent signal.

        source_face_id is intentionally NOT the durable identity here (checksum is) — it's
        only a same-session dedup key scoped to the current media.db's faces table, which
        gets fully regenerated by `media faces`. If media.db is ever rebuilt, a re-promotion
        of "the same" face just becomes a fresh row instead of updating in place; harmless,
        just means occasionally re-confirming a name after a full rebuild.

        Promoting changes this face's ref from "auto:<source_face_id>" to
        "manual:<new_id>" — anything keyed to the old ref (namely a saved age/gender
        estimate, see face_age_estimates) would otherwise silently stop showing up
        the moment someone names the face, so carry it forward here rather than
        leaving it orphaned under a ref nothing displays anymore."""
        x1, y1, x2, y2 = bbox
        cur = self.conn.cursor()
        cur.execute('''
            INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source, source_face_id,
                                image_width, image_height, rejected, created_at, frame_index)
            VALUES (?, ?, ?,?,?,?, ?, 'auto', ?, ?, ?, 0, ?, ?)
            ON CONFLICT(source_face_id) WHERE source_face_id IS NOT NULL
                DO UPDATE SET identity=excluded.identity, rejected=0
        ''', (checksum, identity, x1, y1, x2, y2, embedding_bytes, source_face_id,
              image_width, image_height, int(time.time()), frame_index))
        cur.execute('SELECT id FROM faces WHERE source_face_id = ?', (source_face_id,))
        new_id = cur.fetchone()[0]

        old_ref, new_ref = f'auto:{source_face_id}', f'manual:{new_id}'
        cur.execute('SELECT 1 FROM face_age_estimates WHERE face_ref = ?', (new_ref,))
        if cur.fetchone() is None:
            # Only migrate if the new ref doesn't already have its own estimate —
            # e.g. re-estimated after an earlier promotion — which should win over
            # whatever was saved before this promotion.
            cur.execute(
                'UPDATE face_age_estimates SET face_ref = ? WHERE face_ref = ?',
                (new_ref, old_ref)
            )
        self.conn.commit()
        return new_id

    def reject_auto_face(self, source_face_id, checksum, bbox, embedding_bytes, image_width, image_height, frame_index=None):
        """A human said 'this is not a face' (or 'wrong person, remove it') for an
        auto-detected box. Kept as a row (rejected=1, identity=NULL) instead of deleted —
        a confirmed hard-negative is still useful signal for training a better detector,
        and the source_face_id upsert keeps it from ever reappearing as 'unidentified'."""
        x1, y1, x2, y2 = bbox
        cur = self.conn.cursor()
        cur.execute('''
            INSERT INTO faces (checksum, identity, x1,y1,x2,y2, embedding, bbox_source, source_face_id,
                                image_width, image_height, rejected, created_at, frame_index)
            VALUES (?, NULL, ?,?,?,?, ?, 'auto', ?, ?, ?, 1, ?, ?)
            ON CONFLICT(source_face_id) WHERE source_face_id IS NOT NULL
                DO UPDATE SET rejected=1, identity=NULL
        ''', (checksum, x1, y1, x2, y2, embedding_bytes, source_face_id,
              image_width, image_height, int(time.time()), frame_index))
        self.conn.commit()

    def reject_face(self, manual_face_id):
        """Reject an existing manual.db row (manually-added or already-promoted) — kept,
        not deleted, for the same reason as reject_auto_face."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET rejected = 1, identity = NULL WHERE id = ?', (manual_face_id,))
        self.conn.commit()

    def assign_identity(self, manual_face_id, name):
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET identity = ?, rejected = 0 WHERE id = ?', (name.strip(), manual_face_id))
        self.conn.commit()

    def rename_identity(self, old_name, new_name):
        """Renames a person everywhere they appear (every manual.db face row with this
        identity), not just a single face — fixes a typo/renaming once instead of
        re-naming each of that person's faces individually."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET identity = ? WHERE identity = ?', (new_name.strip(), old_name))
        self.conn.commit()

    def set_face_favorite(self, manual_face_id, favorite):
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET favorite = ? WHERE id = ?', (1 if favorite else 0, manual_face_id))
        self.conn.commit()

    def get_faces_for_file(self, checksum):
        """Non-rejected face rows for a file — feeds the photo-page face chips.
        Includes frame-specific rows alongside whole-file ones; frame_index in the
        result lets the UI badge frame-specific detections."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT id, x1,y1,x2,y2, identity, bbox_source, source_face_id, frame_index, favorite FROM faces '
            'WHERE checksum = ? AND rejected = 0 ORDER BY id',
            (checksum,)
        )
        return cur.fetchall()

    def get_identities_for_checksums(self, checksums):
        """Batched lookup: {checksum: [name, ...]} of *named* faces only (unknown/
        unidentified faces are deliberately excluded) — feeds card-level "who's in
        this photo" chips without an N+1 query per file."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(
            f'''SELECT DISTINCT checksum, identity FROM faces
                WHERE checksum IN ({placeholders}) AND identity IS NOT NULL AND rejected = 0
                ORDER BY identity''',
            tuple(checksums)
        )
        result = {}
        for checksum, identity in cur.fetchall():
            result.setdefault(checksum, []).append(identity)
        return result

    def get_favorite_faces(self):
        """Non-rejected, favorited manual.db face rows (always named or hand-drawn,
        since only manual rows can be favorited)."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT id, checksum, identity FROM faces WHERE favorite = 1 AND rejected = 0 ORDER BY identity, id'
        )
        return cur.fetchall()

    def get_promoted_source_ids(self):
        """Every media.db faces.id that already has a manual.db row — named, still-unnamed-
        but-manually-touched, or rejected. Used to compute 'auto faces not yet handled' via
        a Python-side set difference (no cross-db JOIN). Deliberately includes rejected rows
        too, so a rejected auto-detection never resurfaces as 'unidentified'."""
        cur = self.conn.cursor()
        cur.execute('SELECT source_face_id FROM faces WHERE source_face_id IS NOT NULL')
        return {row[0] for row in cur.fetchall()}

    def get_face(self, face_id):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
        return cur.fetchone()

    def get_all_faces_with_embedding(self):
        """Return [(id, checksum, identity, embedding_bytes), ...] for every non-rejected
        manual.db face."""
        cur = self.conn.cursor()
        cur.execute('SELECT id, checksum, identity, embedding FROM faces WHERE rejected = 0')
        return cur.fetchall()

    def get_named_face_embeddings(self):
        """Return [(identity, embedding_bytes), ...] for faces a human has named."""
        cur = self.conn.cursor()
        cur.execute("SELECT identity, embedding FROM faces WHERE identity IS NOT NULL AND rejected = 0")
        return cur.fetchall()

    def find_matching_identity(self, embedding_bytes, threshold=None):
        """Return (identity, score) for the closest known person if similarity clears
        `threshold` (defaults to AUTO_MATCH_THRESHOLD), else (None, None). Powers
        auto-matching a newly detected face to an existing identity without asking."""
        if threshold is None:
            threshold = AUTO_MATCH_THRESHOLD
        named = self.get_named_face_embeddings()
        if not named:
            return None, None
        import numpy as np
        query = np.frombuffer(embedding_bytes, dtype=np.float32)
        names = [n for n, _ in named]
        matrix = np.stack([np.frombuffer(e, dtype=np.float32) for _, e in named])
        scores = matrix.dot(query)
        best_idx = int(scores.argmax())
        if scores[best_idx] >= threshold:
            return names[best_idx], float(scores[best_idx])
        return None, None

    def get_embeddings_for_identity(self, name):
        """Return [embedding_bytes, ...] for every face already confirmed as this person
        (a person can have more than one confirmed face)."""
        cur = self.conn.cursor()
        cur.execute("SELECT embedding FROM faces WHERE identity = ? AND rejected = 0", (name,))
        return [row[0] for row in cur.fetchall()]

    def get_all_identities(self):
        """Return [(identity, count), ...] ordered by count descending."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT identity, COUNT(*) as cnt FROM faces
            WHERE identity IS NOT NULL AND rejected = 0
            GROUP BY identity
            ORDER BY cnt DESC
        ''')
        return cur.fetchall()

    def get_files_by_face_identity(self, name, limit=100):
        """Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checksum FROM faces WHERE LOWER(identity) LIKE LOWER(?) AND rejected = 0 LIMIT ?",
            (f'%{name}%', limit)
        )
        return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Migration bookkeeping (kept for future one-off migrations; nothing uses this today)
    # ------------------------------------------------------------------

    def is_migrated(self, key):
        cur = self.conn.cursor()
        cur.execute('SELECT 1 FROM migration_state WHERE key = ?', (key,))
        return cur.fetchone() is not None

    def mark_migrated(self, key):
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR IGNORE INTO migration_state (key, done_at) VALUES (?, ?)',
            (key, int(time.time()))
        )
        self.conn.commit()
