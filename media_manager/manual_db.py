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
        # A human-confirmed "this file does NOT belong in this set" — the
        # negative counterpart to file_sets, same (checksum, set_id) shape,
        # scoped per-set (unlike faces' single global `rejected` flag, since
        # a file can genuinely belong to one set while being a confirmed
        # non-match for another). Kept as real ground truth like everything
        # else in this file, not just a session-scoped UI convenience — see
        # this module's docstring on why manual decisions are never thrown
        # away here, even ones that only exist to say "not this".
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_set_exclusions (
                checksum TEXT NOT NULL,
                set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (checksum, set_id)
            )
        ''')
        # A human-confirmed "this person appears somewhere in this photo,"
        # independent of any specific detected face — no bbox, no embedding.
        # Exists for photos where face detection missed the person entirely
        # (turned away, obscured, out of frame at detection time, etc.) but a
        # human still recognizes them; the regular per-face path
        # (add_manual_face/promote_auto_face, which always carries a real
        # ArcFace embedding) stays how every other identity confirmation is
        # recorded, so this table is never mistaken for training data about
        # what a face looks like — it only ever answers "who's in this photo."
        cur.execute('''
            CREATE TABLE IF NOT EXISTS identity_photo_assignments (
                checksum TEXT NOT NULL,
                identity TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (checksum, identity)
            )
        ''')
        # A human-confirmed "this person appears in this set" — a single link, not
        # one row per member photo. Deliberately dynamic: which photos that implies
        # is resolved at read time from the set's current membership (see
        # get_sets_linked_to_identity's callers), so adding a photo to a linked set
        # later makes it "theirs" automatically, with no re-confirmation and no
        # extra row written.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS identity_set_assignments (
                set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                identity TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (set_id, identity)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_identity_set_assignments_identity ON identity_set_assignments (identity)')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                temperature REAL NOT NULL DEFAULT 0.75,
                created_at INTEGER NOT NULL
            )
        ''')
        # category_id NULL is a legal, meaningful value here — it records a human's
        # explicit "this file has no category" decision (see set_file_category),
        # which must still block ML auto-matching from claiming the file. Absence
        # of a row entirely (not this) means "no manual decision ever made".
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_category_overrides (
                checksum TEXT PRIMARY KEY,
                category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
                created_at INTEGER NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_category_overrides_category ON file_category_overrides (category_id)')

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

    def remove_tag_by_label(self, checksum, label, polarity):
        """Delete a single whole-file tag row by (checksum, label, polarity) rather
        than by numeric id — lets a caller (the tag-suggestion swipe stream's undo)
        reverse its own add_tag call without having to thread the row id it never
        asked for back through a fetch response."""
        cur = self.conn.cursor()
        cur.execute(
            'DELETE FROM tags WHERE checksum = ? AND label = ? AND polarity = ? AND x1 IS NULL',
            (checksum, label, polarity)
        )
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

    def get_decided_checksums_for_tag(self, label):
        """Checksums with ANY tag row for this label — positive (already tagged) or
        negative (explicitly rejected) — must never be re-suggested for this label."""
        cur = self.conn.cursor()
        cur.execute('SELECT DISTINCT checksum FROM tags WHERE label = ?', (label,))
        return {row[0] for row in cur.fetchall()}

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

    def get_age_estimate_for_face_ref(self, face_ref):
        """Single-face lookup — unlike get_age_estimates_for_checksum (every face on
        one photo), this is for callers that already know exactly which face they
        want (e.g. a set's "people present" representative face)."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT age, gender FROM face_age_estimates WHERE face_ref = ? ORDER BY created_at DESC LIMIT 1',
            (face_ref,)
        )
        return cur.fetchone()

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
        # A file can't be both a confirmed member and a confirmed exclusion for
        # the same set — a later confirm (e.g. from the photo page) always wins.
        cur.execute('DELETE FROM file_set_exclusions WHERE checksum = ? AND set_id = ?', (checksum, set_id))
        self.conn.commit()

    def remove_file_from_set(self, checksum, set_id):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_sets WHERE checksum = ? AND set_id = ?', (checksum, set_id))
        self.conn.commit()

    def exclude_file_from_set(self, checksum, set_id):
        """Record a human's 'not this set' decision for this file — the
        suggestion stream's reject action. Permanent ground truth, not a
        session-only UI convenience: a file excluded here is never offered
        as a suggestion for this same set again (see get_excluded_checksums_
        for_set), and this is exactly the kind of manual-vs-ML-derived
        negative example this database exists to preserve for future
        training."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO file_set_exclusions (checksum, set_id, created_at) VALUES (?, ?, ?)
        ''', (checksum, set_id, int(time.time())))
        self.conn.commit()

    def remove_set_exclusion(self, checksum, set_id):
        """Undo a prior exclude_file_from_set call (the reject-swipe's Ctrl+Z)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_set_exclusions WHERE checksum = ? AND set_id = ?', (checksum, set_id))
        self.conn.commit()

    def get_excluded_checksums_for_set(self, set_id):
        """Every checksum a human has confirmed does NOT belong in this set —
        used to permanently filter these out of future suggestion candidates
        for this set, the same way file_sets' own members are filtered out."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum FROM file_set_exclusions WHERE set_id = ?', (set_id,))
        return {row[0] for row in cur.fetchall()}

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

    def get_all_set_member_checksums(self):
        """Every checksum that belongs to at least one set, as a plain set() —
        one cheap query independent of any candidate list size, unlike
        get_sets_for_checksums (which needs one bound SQL parameter per
        checksum and has to be chunked for a large *caller-supplied* list).
        Used by the set-suggestion stream's avoid_existing check, which
        otherwise needs this same yes/no answer for every file in the whole
        library that clears the similarity threshold — recomputing that via
        chunked IN(...) lookups on every single background buffer refill was
        the actual slow part once a library got large, not just the SQLite
        bound-parameter crash it was originally chunked to avoid."""
        cur = self.conn.cursor()
        cur.execute('SELECT DISTINCT checksum FROM file_sets')
        return {row[0] for row in cur.fetchall()}

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
    # Categories
    # ------------------------------------------------------------------

    def _clamp_temperature(self, temperature):
        return max(0.0, min(1.0, temperature))

    def create_category(self, name, temperature=0.75):
        """Find-or-create (mirrors create_set): if the category already exists,
        its stored temperature is NOT updated — call set_category_temperature
        for that."""
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR IGNORE INTO categories (name, temperature, created_at) VALUES (?, ?, ?)',
            (name.strip(), self._clamp_temperature(temperature), int(time.time()))
        )
        self.conn.commit()
        cur.execute('SELECT id FROM categories WHERE name = ?', (name.strip(),))
        return cur.fetchone()[0]

    def find_category(self, name):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM categories WHERE name = ?', (name,))
        return cur.fetchone()

    def get_category(self, category_id):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM categories WHERE id = ?', (category_id,))
        return cur.fetchone()

    def list_categories(self):
        """Return category rows with id, name, temperature, created_at, image_count —
        image_count here is manual-assignment-only (file_category_overrides rows
        with a non-NULL category_id), mirroring list_sets's image_count."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT c.id, c.name, c.temperature, c.created_at,
                   COUNT(fco.checksum) as image_count
            FROM categories c
            LEFT JOIN file_category_overrides fco
                ON fco.category_id = c.id
            GROUP BY c.id
            ORDER BY c.name
        ''')
        return cur.fetchall()

    def rename_category(self, category_id, name):
        cur = self.conn.cursor()
        cur.execute('UPDATE categories SET name = ? WHERE id = ?', (name.strip(), category_id))
        self.conn.commit()

    def set_category_temperature(self, category_id, temperature):
        cur = self.conn.cursor()
        cur.execute(
            'UPDATE categories SET temperature = ? WHERE id = ?',
            (self._clamp_temperature(temperature), category_id)
        )
        self.conn.commit()

    def delete_category(self, category_id):
        """Deletes the category entity only. Any file_category_overrides row that
        pointed at it becomes category_id=NULL via ON DELETE SET NULL — i.e. those
        files become explicitly uncategorized, NOT reverted to 'no decision made'.
        Once a human has touched a file's category, ML must never silently reclaim
        it just because the specific category they picked was later removed."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM categories WHERE id = ?', (category_id,))
        self.conn.commit()

    def set_file_category(self, checksum, category_id):
        """Upsert a file's manual category override. category_id=None is a
        legitimate call — it records an explicit 'no category' human decision,
        which still outranks any ML auto-match (see category_resolver.py). This
        single method serves both 'assign' and 'clear' from the CLI/web UI."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT INTO file_category_overrides (checksum, category_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(checksum) DO UPDATE SET category_id=excluded.category_id, created_at=excluded.created_at
        ''', (checksum, category_id, int(time.time())))
        self.conn.commit()

    def clear_category_override(self, checksum):
        """Hard-delete the override row entirely — distinct from
        set_file_category(checksum, None), which UPSERTS an explicit 'no
        category' row. This restores "no manual decision was ever made", used
        to undo a category swipe decision (confirm or reject) back to the
        pre-decision state — safe because the suggestion stream only ever
        offers files that had no override to begin with."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_category_overrides WHERE checksum = ?', (checksum,))
        self.conn.commit()

    def get_category_override(self, checksum):
        """Return None if no manual decision was ever made for this file, else
        {'category_id', 'name'} — name is None for an explicit 'no category' row."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT fco.category_id, c.name
            FROM file_category_overrides fco
            LEFT JOIN categories c ON c.id = fco.category_id
            WHERE fco.checksum = ?
        ''', (checksum,))
        row = cur.fetchone()
        if row is None:
            return None
        return {'category_id': row[0], 'name': row[1]}

    def get_category_overrides_for_checksums(self, checksums):
        """Batched lookup: {checksum: {'category_id','name'}} for whichever of
        `checksums` have a manual decision recorded — a checksum missing from the
        result had no decision made at all (mirrors get_sets_for_checksums)."""
        if not checksums:
            return {}
        placeholders = ','.join('?' for _ in checksums)
        cur = self.conn.cursor()
        cur.execute(f'''
            SELECT fco.checksum, fco.category_id, c.name
            FROM file_category_overrides fco
            LEFT JOIN categories c ON c.id = fco.category_id
            WHERE fco.checksum IN ({placeholders})
        ''', tuple(checksums))
        return {row[0]: {'category_id': row[1], 'name': row[2]} for row in cur.fetchall()}

    def get_all_category_override_checksums(self):
        """Every checksum with ANY manual category decision recorded (assigned to
        some category, or an explicit "no category"), as a plain set() — one cheap
        query independent of candidate-list size, unlike
        get_category_overrides_for_checksums (which needs one bound SQL parameter
        per checksum and blows past SQLite's variable limit when called with
        every file in the library, as the category-suggestion stream does)."""
        cur = self.conn.cursor()
        cur.execute('SELECT DISTINCT checksum FROM file_category_overrides')
        return {row[0] for row in cur.fetchall()}

    def get_example_checksums_for_category(self, category_id, limit=1000):
        """Files manually assigned to this category — the ML training examples
        used to build a similarity centroid (mirrors get_files_by_set)."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT checksum FROM file_category_overrides WHERE category_id = ? LIMIT ?',
            (category_id, limit)
        )
        return [row[0] for row in cur.fetchall()]

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

    def delete_face_decision_by_source(self, source_face_id):
        """Hard-delete the manual.db row (if any) created by promote_auto_face or
        reject_auto_face for this auto-detected face — restoring it to fully
        unhandled, unlike reject_auto_face's kept-as-hard-negative row. Used to
        undo a face swipe decision (confirm or reject) back to the pre-decision
        state; safe because the face-suggestion stream only ever offers faces
        with no existing manual.db row for this source_face_id (see
        _unpromoted_auto_faces)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM faces WHERE source_face_id = ?', (source_face_id,))
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

    def unassign_identity_for_checksum(self, checksum, name):
        """Clear the identity (back to unidentified, not rejected) for every
        manual.db face row on this one photo currently named `name` — used by
        the person-search page's bulk "make unknown again" action, for when
        the auto-categorizer's confidence threshold wrongly merged someone
        into this identity, or a misclick joined the wrong two people.
        Returns the number of rows updated."""
        cur = self.conn.cursor()
        cur.execute(
            'UPDATE faces SET identity = NULL WHERE checksum = ? AND identity = ?',
            (checksum, name)
        )
        self.conn.commit()
        return cur.rowcount

    def rename_identity(self, old_name, new_name):
        """Renames a person everywhere they appear — every manual.db face row with
        this identity, every whole-photo assignment (identity_photo_assignments),
        and every set link (identity_set_assignments) — not just a single face,
        fixing a typo/renaming once instead of hunting down each place this string
        was written."""
        new_name = new_name.strip()
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET identity = ? WHERE identity = ?', (new_name, old_name))
        cur.execute('UPDATE identity_photo_assignments SET identity = ? WHERE identity = ?', (new_name, old_name))
        cur.execute('UPDATE identity_set_assignments SET identity = ? WHERE identity = ?', (new_name, old_name))
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

    def get_all_checksums_with_named_face(self):
        """Every checksum with at least one identified (named, non-rejected)
        face, as a plain set() — same one-cheap-query-instead-of-chunked-
        per-candidate-lookups shape as get_all_set_member_checksums, for the
        face-suggestion stream's avoid_existing check."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT checksum FROM faces WHERE identity IS NOT NULL AND rejected = 0")
        return {row[0] for row in cur.fetchall()}

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

    def get_representative_face_ids(self):
        """{identity: earliest non-rejected manual.db face id} for every named
        identity — feeds a thumbnail (a representative face crop) per person in
        the identity picker, same "first face for this name" query
        get_people_present_in_set uses for its own fallback lookup."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT identity, MIN(id) FROM faces
            WHERE identity IS NOT NULL AND rejected = 0
            GROUP BY identity
        ''')
        return {identity: face_id for identity, face_id in cur.fetchall()}

    def get_files_by_face_identity(self, name, limit=100):
        """Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checksum FROM faces WHERE LOWER(identity) LIKE LOWER(?) AND rejected = 0 LIMIT ?",
            (f'%{name}%', limit)
        )
        return [row[0] for row in cur.fetchall()]

    def assign_identity_to_photo(self, checksum, identity):
        """See identity_photo_assignments' schema comment: a whole-photo, no-bbox
        identity confirmation, for photos where no individual face crop exists
        to attach the identity to. Assigning to a whole set is a separate,
        single-row mechanism — see link_identity_to_set — not a loop over this."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO identity_photo_assignments (checksum, identity, created_at)
            VALUES (?, ?, ?)
        ''', (checksum, identity, int(time.time())))
        self.conn.commit()

    def remove_identity_photo_assignment(self, checksum, identity):
        cur = self.conn.cursor()
        cur.execute(
            'DELETE FROM identity_photo_assignments WHERE checksum = ? AND identity = ?',
            (checksum, identity)
        )
        self.conn.commit()

    def get_photos_assigned_to_identity(self, identity, limit=1000):
        """Checksums manually whole-photo-assigned to this identity (exact match,
        unlike get_files_by_face_identity's substring search — these are written
        with one exact name, not searched by a user-typed query) — a second,
        independent source of "files for this person" that callers building
        that combined list need to union in alongside get_files_by_face_identity."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT checksum FROM identity_photo_assignments WHERE identity = ? LIMIT ?',
            (identity, limit)
        )
        return [row[0] for row in cur.fetchall()]

    def link_identity_to_set(self, set_id, identity):
        """'This person appears in this set' — one row, not one per member photo.
        Which photos that implies is resolved dynamically by callers (join against
        the set's current membership at read time), so a photo added to this set
        later is automatically "theirs" too, with nothing further to write here."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO identity_set_assignments (set_id, identity, created_at)
            VALUES (?, ?, ?)
        ''', (set_id, identity, int(time.time())))
        self.conn.commit()

    def unlink_identity_from_set(self, set_id, identity):
        cur = self.conn.cursor()
        cur.execute(
            'DELETE FROM identity_set_assignments WHERE set_id = ? AND identity = ?',
            (set_id, identity)
        )
        self.conn.commit()

    def get_sets_linked_to_identity(self, identity):
        """Every set this identity has been linked to (see link_identity_to_set),
        as full set rows — callers resolve "files for this person" by expanding
        each one's current membership, and mark these as already-linked in the
        set picker UI."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT s.* FROM sets s
            JOIN identity_set_assignments isa ON isa.set_id = s.id
            WHERE isa.identity = ?
            ORDER BY s.name
        ''', (identity,))
        return cur.fetchall()

    def generate_placeholder_identity_name(self):
        """The next 'Unnamed N' not currently in use — lets a face/identity get
        confirmed without requiring a human-typed name up front (see
        assign_identity/promote_auto_face callers in web.py). Renaming later just
        goes through the normal rename_identity path; this only ever picks the
        starting string."""
        existing = {name for name, _count in self.get_all_identities()}
        n = 1
        while f'Unnamed {n}' in existing:
            n += 1
        return f'Unnamed {n}'

    def get_people_present_in_set(self, set_id, member_checksums):
        """Everyone confirmed to appear in this set, from all three sources a
        confirmation can come from: a whole-photo assignment
        (identity_photo_assignments) on one of the set's photos, a whole-set link
        (identity_set_assignments) to this set itself, or a real named face on
        one of the set's own photos. Returns an ordered {identity: representative
        manual.db face id, or None if no crop-able face exists anywhere} dict —
        manually *assigned* people (the first two sources) come first, then
        everyone who's only there via a detected face, each name appearing
        exactly once even if they came from more than one source (e.g. someone
        both manually linked to this set AND separately detected in one of its
        photos surfaces once, in the assigned group). A face id found directly
        in this set is always preferred for the thumbnail; names that only came
        from the manual sources fall back to any named face for them elsewhere
        in the library, and get None only if no such face exists at all."""
        cur = self.conn.cursor()
        face_people = {}
        if member_checksums:
            placeholders = ','.join('?' for _ in member_checksums)
            cur.execute(f'''
                SELECT identity, MIN(id) FROM faces
                WHERE checksum IN ({placeholders}) AND identity IS NOT NULL AND rejected = 0
                GROUP BY identity
            ''', tuple(member_checksums))
            for identity, face_id in cur.fetchall():
                face_people[identity] = face_id

        assigned_names = set()
        if member_checksums:
            placeholders = ','.join('?' for _ in member_checksums)
            cur.execute(f'''
                SELECT DISTINCT identity FROM identity_photo_assignments
                WHERE checksum IN ({placeholders})
            ''', tuple(member_checksums))
            assigned_names.update(row[0] for row in cur.fetchall())

        cur.execute('SELECT identity FROM identity_set_assignments WHERE set_id = ?', (set_id,))
        assigned_names.update(row[0] for row in cur.fetchall())

        people = {}
        for identity in sorted(assigned_names, key=str.lower):
            people[identity] = face_people.get(identity)
        for identity in sorted(face_people.keys(), key=str.lower):
            people.setdefault(identity, face_people[identity])

        missing = [name for name, face_id in people.items() if face_id is None]
        if missing:
            placeholders = ','.join('?' for _ in missing)
            cur.execute(f'''
                SELECT identity, MIN(id) FROM faces
                WHERE identity IN ({placeholders}) AND rejected = 0
                GROUP BY identity
            ''', tuple(missing))
            for identity, face_id in cur.fetchall():
                people[identity] = face_id

        return people

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
