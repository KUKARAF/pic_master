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
import os
import shutil
import sqlite3
import threading
import time
from collections import Counter

from media_manager.database import ThreadLocalDB

# Cosine similarity above which a newly detected face is auto-assigned to a known
# identity without asking — deliberately stricter than the 0.45 "suggest a match and
# let a human confirm" threshold used elsewhere, since auto-assignment has no human
# in the loop to catch a wrong guess.
AUTO_MATCH_THRESHOLD = 0.6


def run_migrations(db_path, apply=False):
    """Checks/applies every pending schema migration in media_manager/migrations/
    against a raw manual.db connection — never through ManualDB, whose
    _create_tables() fail-loud-guards against any pending migration (a ManualDB
    can't be constructed until migrations are applied). Backs `media migrate`
    (media.py), not run automatically at app startup.

    Tracked in the `migration_state` table (one row per migration id), so this
    is safe to run repeatedly — a migration already recorded there is skipped
    without even calling its describe_pending/apply. A migration whose
    describe_pending() says there's nothing to do (a fresh install, or a
    manual.db already in the target shape from before this migration existed)
    is recorded as applied immediately, with no write beyond that bookkeeping
    row, so it's not re-checked on every future startup either.

    apply=False (default): reports what's pending, writes nothing else.
    apply=True: backs up db_path once, before the first pending migration
    (shutil.copyfile with a timestamp suffix, same pattern as
    migrate_file_identity.py) — not once per migration, since one backup
    covers the whole run. Each migration then runs inside its own transaction
    — SQLite's DDL is fully transactional, so a failure rolls back that one
    migration cleanly (leaving its target tables completely untouched) and
    this stops immediately without attempting any later migrations.

    Returns a list of report strings, one per migration that was pending (or
    a single "nothing pending" message if none were). Raises on failure
    (never swallows an exception) — the caller (media.py's `migrate` command)
    is expected to let it propagate."""
    if not os.path.isfile(db_path):
        return [f"{db_path} does not exist yet — nothing to migrate."]

    from media_manager.migrations import MIGRATIONS

    conn = sqlite3.connect(db_path, isolation_level=None)  # explicit BEGIN/COMMIT below, not pysqlite's implicit one
    reports = []
    backup_path = None
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS migration_state (
                key TEXT PRIMARY KEY,
                done_at INTEGER NOT NULL
            )
        ''')
        for migration in MIGRATIONS:
            already_applied = conn.execute(
                'SELECT 1 FROM migration_state WHERE key = ?', (migration.ID,)
            ).fetchone() is not None
            if already_applied:
                continue

            pending_desc = migration.describe_pending(conn)
            if pending_desc is None:
                conn.execute(
                    'INSERT OR IGNORE INTO migration_state (key, done_at) VALUES (?, ?)',
                    (migration.ID, int(time.time()))
                )
                continue

            reports.append(f'{migration.ID}: {pending_desc}')
            if not apply:
                continue

            if backup_path is None:
                backup_path = f'{db_path}.bak-{time.strftime("%Y%m%d%H%M%S")}'
                shutil.copyfile(db_path, backup_path)

            try:
                conn.execute('BEGIN IMMEDIATE')
                result = migration.apply(conn)
                conn.execute(
                    'INSERT OR IGNORE INTO migration_state (key, done_at) VALUES (?, ?)',
                    (migration.ID, int(time.time()))
                )
                conn.execute('COMMIT')
                reports[-1] = f'{migration.ID}: {result}. Backup: {backup_path}'
            except Exception as exc:
                conn.execute('ROLLBACK')
                raise RuntimeError(
                    f"Migration '{migration.ID}' failed and was rolled back — manual.db "
                    f"is unchanged (a backup was also taken at {backup_path} before "
                    f"starting, just in case): {exc}"
                ) from exc

        if not reports:
            return ['Nothing pending — manual.db is already up to date.']
        if not apply:
            reports.append('Re-run with --apply to perform these migration(s).')
        return reports
    finally:
        conn.close()


class ManualDB(ThreadLocalDB):
    def __init__(self, db_path):
        super().__init__(db_path)
        self._create_tables()
        # Cache invalidation for the face-embedding matrices below. Every method
        # that inserts/updates/deletes a row in `faces` bumps self._face_ver; each
        # cached matrix accessor stores the version it was built at and rebuilds
        # only when that stored version != self._face_ver. Coarse over-invalidation
        # (bumping on a write that couldn't actually change the matrix, e.g. a
        # favorite bump) is deliberately fine — a stale-cache MISS is not. Shared
        # single-process cache (uvicorn --workers 1); the lock guards concurrent
        # rebuilds across request threads (ThreadLocalDB gives each thread its own
        # connection, but these Python-side attributes are shared on the instance).
        self._face_ver = 0
        self._matrix_lock = threading.Lock()
        self._named_matrix_cache = None   # (version, (identities, matrix))
        self._all_faces_matrix_cache = None  # (version, (face_ids, checksums, identities, matrix))

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
        # Fail loudly if any registered migration (media_manager/migrations/) is
        # still pending, rather than letting 'CREATE TABLE IF NOT EXISTS' silently
        # no-op against an old shape and crash confusingly on the first query that
        # expects the new one. Unlike the additive migrations elsewhere in this
        # method (simple ADD COLUMN + backfill, safe to run silently), a pending
        # migration here is a genuine data-rewriting change — deliberately NOT run
        # automatically. See `media migrate --apply`.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS migration_state (
                key TEXT PRIMARY KEY,
                done_at INTEGER NOT NULL
            )
        ''')
        from media_manager.migrations import MIGRATIONS
        pending_ids = []
        for migration in MIGRATIONS:
            applied = cur.execute('SELECT 1 FROM migration_state WHERE key = ?', (migration.ID,)).fetchone()
            if applied:
                continue
            if migration.describe_pending(cur) is not None:
                pending_ids.append(migration.ID)
        if pending_ids:
            raise RuntimeError(
                "manual.db has pending schema migration(s): " + ', '.join(pending_ids) + ". Run:\n"
                "    media migrate --dry-run   # see what would change, writes nothing\n"
                "    media migrate --apply     # backs up manual.db, then migrates\n"
                "first. Nothing has been written by this run."
            )

        cur.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checksum TEXT NOT NULL,
                tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                polarity TEXT NOT NULL DEFAULT 'positive',
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                image_width INTEGER, image_height INTEGER,
                frame_index INTEGER,
                favorite INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
        ''')
        # Read-only convenience view exposing every file_tags column plus the joined
        # tag label (mirrors sets_with_studio) — every read site that used to select
        # straight from the old flat `tags` table and read row['label'] directly can
        # just select FROM this instead of rewriting its own join.
        cur.execute('''
            CREATE VIEW IF NOT EXISTS file_tags_with_label AS
            SELECT ft.*, t.label AS label
            FROM file_tags ft
            JOIN tags t ON t.id = ft.tag_id
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_favorites (
                checksum TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1
            )
        ''')
        # Favorites are a counter now, not a toggle. A row's presence still means
        # "favorited" (count > 0) so all the existing presence-based queries stay
        # correct; the count column just records how many times it was bumped.
        existing_fav_cols = {row[1] for row in cur.execute('PRAGMA table_info(file_favorites)')}
        if 'count' not in existing_fav_cols:
            cur.execute('ALTER TABLE file_favorites ADD COLUMN count INTEGER NOT NULL DEFAULT 1')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_titles (
                checksum TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        ''')
        # Manually-captured video frames: each still (child_checksum, a hidden file in
        # media.db) links back to the source video (parent_checksum) + the timestamp it
        # was grabbed at. Lets the video's page list its saved frames and a captured
        # still link back to its video. Checksum-keyed like everything else here.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS frame_captures (
                child_checksum TEXT PRIMARY KEY,
                parent_checksum TEXT NOT NULL,
                time_ms INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_frame_captures_parent ON frame_captures (parent_checksum)')
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
        # A studio is one-per-set (not multi-valued like tags/categories), so this is
        # a plain id'd lookup table (same shape as `categories`) referenced by a
        # single FK column on `sets` — no join table needed.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS studios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL
            )
        ''')
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
        # studio used to be a free-text column directly on sets (no id — the
        # UNIQUE(name, studio) above was already dead weight even then, since
        # SQLite treats every NULL studio as distinct for UNIQUE purposes, see
        # find_sets_by_name's docstring), so renaming a studio meant a blind
        # UPDATE ... WHERE studio = ? string rewrite across every set instead of
        # renaming one row. Migrated once into the real `studios` table above,
        # dedup'd by exact name, via a plain ALTER TABLE ADD COLUMN + backfill
        # rather than the file_sets-style rename+recreate+copy+drop: sets.id is
        # referenced by file_sets/file_set_exclusions/identity_set_assignments/
        # set_aliases, so rebuilding this table would mean carefully preserving
        # every existing id and the sqlite_sequence counter — real risk for no
        # benefit when only a column (not the primary key) is changing.
        #
        # The old `studio` column itself is deliberately NOT dropped afterward
        # (app code never reads/writes it again once studio_id exists — see
        # sets_with_studio below) — SQLite's ALTER TABLE DROP COLUMN refuses to
        # drop a column that's part of a table-level constraint, and it's still
        # named in UNIQUE(name, studio) above; dropping that constraint needs
        # the same disruptive full rebuild this approach exists to avoid, so
        # the column is left behind as harmless dead weight instead.
        if 'studio' in existing_set_cols and 'studio_id' not in existing_set_cols:
            cur.execute('ALTER TABLE sets ADD COLUMN studio_id INTEGER REFERENCES studios(id)')
            cur.execute('''
                INSERT OR IGNORE INTO studios (name, created_at)
                SELECT DISTINCT studio, ? FROM sets WHERE studio IS NOT NULL AND studio != ''
            ''', (int(time.time()),))
            cur.execute('''
                UPDATE sets SET studio_id = (SELECT id FROM studios WHERE studios.name = sets.studio)
                WHERE sets.studio IS NOT NULL AND sets.studio != ''
            ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_sets_studio_id ON sets (studio_id)')
        # Read-only convenience view exposing every current `sets` column (the old
        # `studio` text column deliberately excluded, per above — replaced by the
        # freshly joined name under the same key) plus the joined studio *name*
        # (mirrors database.py's files_with_path view over files+file_paths) —
        # every read site that used to do `SELECT * FROM sets`/`s.*` and read
        # row['studio'] directly can just select FROM this instead of rewriting
        # its own join. Writes still go through the real `sets`/`studios` tables
        # — views aren't written to.
        cur.execute('''
            CREATE VIEW IF NOT EXISTS sets_with_studio AS
            SELECT s.id, s.name, s.created_at, s.favorite, s.studio_id, st.name AS studio
            FROM sets s
            LEFT JOIN studios st ON st.id = s.studio_id
        ''')

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
        # checksum (leftmost of the PK above) is already indexed via the PK itself,
        # but set_id alone is not — every "which files are in set X" lookup
        # (get_files_by_set, used once per set on /sets, /studios, and the
        # set-similarity ranking helpers) filters by set_id alone and was doing a
        # full table scan without this.
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_sets_set_id ON file_sets (set_id)')
        # Manual per-set ordering: a nullable rank so a user can arrange a set's
        # files in an arbitrary order on the web set page. NULL means "never
        # manually positioned" and sorts last, falling back to created_at (see
        # get_files_by_set's manual_order path). A plain ADD COLUMN is safe here
        # (unlike the PK rebuild above) since only a column, not the key, changes.
        file_sets_cols = {row[1] for row in cur.execute('PRAGMA table_info(file_sets)')}
        if 'position' not in file_sets_cols:
            cur.execute('ALTER TABLE file_sets ADD COLUMN position INTEGER')
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
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_set_exclusions_set_id ON file_set_exclusions (set_id)')
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
        cur.execute('CREATE INDEX IF NOT EXISTS idx_identity_photo_assignments_identity ON identity_photo_assignments (identity)')
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
        # "This person does NOT belong on this set's roster" — same reasoning as
        # file_set_exclusions above, just for a person instead of a photo. Needed
        # because a person can end up on a set's "People present" list without
        # ever being explicitly linked (identity_set_assignments): a detected
        # face, or a whole-photo assignment, on one of the set's own photos is
        # enough (see get_people_present_in_set/get_all_people_for_sets). Without
        # this table there'd be no way to remove such a person from the set's
        # people list — unlink_identity_from_set only has a row to delete for the
        # explicit-link source, so removal has to also suppress the other two.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS identity_set_exclusions (
                set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                identity TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (set_id, identity)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_identity_set_exclusions_identity ON identity_set_exclusions (identity)')
        # Alternate names for a set ("bts22" for "Behind The Scenes 2022") — same
        # shape/reasoning as identity_aliases below: a lookup convenience only,
        # never written onto the set's own name/studio columns, resolved back to
        # the owning set_id at search time (see resolve_set_alias). PK on alias
        # alone keeps an alias unambiguous across every set, same as identities.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS set_aliases (
                set_id INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (alias)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_set_aliases_set_id ON set_aliases (set_id)')
        # Alternate names for an identity ("girlfriend" / "babe" both meaning the
        # same person) — purely a lookup convenience layered on top of the single
        # canonical `identity` string every faces/assignment row actually stores;
        # an alias is never written onto a face/assignment row itself, only
        # resolved back to its canonical identity at lookup time (see
        # resolve_identity_name), so renaming the canonical identity doesn't
        # require touching every alias row's meaning, just its `identity` column.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS identity_aliases (
                identity TEXT NOT NULL,
                alias TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (alias)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_identity_aliases_identity ON identity_aliases (identity)')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                temperature REAL NOT NULL DEFAULT 0.75,
                created_at INTEGER NOT NULL
            )
        ''')
        # Categories are multi-valued: a file can carry any number of manual
        # category assignments (one row per (checksum, category_id) pair), plus
        # per-category rejections — "confirmed NOT in category X" — recorded the
        # same way file_set_exclusions already records "confirmed not in set X"
        # (see exclude_file_from_set's docstring for the same reasoning: this is
        # the category-suggestion swipe stream's reject action, permanent ground
        # truth, not a session-only convenience).
        #
        # file_category_overrides used to be the single table here: checksum as a
        # bare PRIMARY KEY (one category per file), with a category_id IS NULL row
        # meaning "human explicitly said no category" (a global flag, since only
        # one category could ever apply anyway). That shape can't hold more than
        # one real category per file, so it's split into the two tables below and
        # migrated once if the old table is still present — the old global
        # "no category" rows have no clean per-category equivalent to migrate
        # into and are simply dropped (a rare edge case; the file just starts
        # fresh with no manual categories and no exclusions instead).
        old_overrides_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='file_category_overrides'"
        ).fetchone() is not None
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_categories (
                checksum TEXT NOT NULL,
                category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (checksum, category_id)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_categories_category ON file_categories (category_id)')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS file_category_exclusions (
                checksum TEXT NOT NULL,
                category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (checksum, category_id)
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_category_exclusions_category ON file_category_exclusions (category_id)')
        if old_overrides_exists:
            cur.execute('''
                INSERT OR IGNORE INTO file_categories (checksum, category_id, created_at)
                SELECT checksum, category_id, created_at FROM file_category_overrides
                WHERE category_id IS NOT NULL
            ''')
            cur.execute('DROP TABLE file_category_overrides')

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
                favorite INTEGER NOT NULL DEFAULT 0,
                reviewed_ok INTEGER NOT NULL DEFAULT 0
            )
        ''')
        existing_face_cols2 = {row[1] for row in cur.execute('PRAGMA table_info(faces)')}
        if 'frame_index' not in existing_face_cols2:
            cur.execute('ALTER TABLE faces ADD COLUMN frame_index INTEGER')
        if 'favorite' not in existing_face_cols2:
            cur.execute('ALTER TABLE faces ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0')
        # Backs the "cleanup low confidence matches" feature's "correct, don't
        # ask again" action — a face marked this way is excluded from all
        # future cleanup-queue builds regardless of how low its computed
        # self-consistency score comes out.
        if 'reviewed_ok' not in existing_face_cols2:
            cur.execute('ALTER TABLE faces ADD COLUMN reviewed_ok INTEGER NOT NULL DEFAULT 0')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_tags_checksum ON file_tags (checksum)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags (tag_id)')
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

    def _find_or_create_tag(self, label):
        """Exact-match find-or-create for a tag definition row by label (mirrors
        _find_or_create_studio) — the write paths below always resolve/create a tag
        this way rather than storing the label text directly on the instance row."""
        cur = self.conn.cursor()
        row = cur.execute('SELECT id FROM tags WHERE label = ?', (label,)).fetchone()
        if row is not None:
            return row[0]
        cur.execute('INSERT INTO tags (label, created_at) VALUES (?, ?)', (label, int(time.time())))
        return cur.lastrowid

    def add_tag(self, checksum, label, polarity='positive', frame_index=None):
        """Whole-image tag (no bbox). polarity='negative' records a human correction —
        e.g. YOLO detected "car" but there is no car — so it can be excluded from search
        and used to suppress that false-positive detection chip in the UI. frame_index
        set means this tag applies to that one frame of an animated file, not the file
        as a whole."""
        tag_id = self._find_or_create_tag(label.strip())
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO file_tags (checksum, tag_id, polarity, x1,y1,x2,y2, image_width,image_height, created_at, frame_index) '
            'VALUES (?,?,?,NULL,NULL,NULL,NULL,NULL,NULL,?,?)',
            (checksum, tag_id, polarity, int(time.time()), frame_index)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_spatial_tag(self, checksum, label, x1, y1, x2, y2, image_width, image_height, frame_index=None, polarity='positive'):
        """A tag scoped to a region — a YOLO-style (image, bbox, label) training
        example. frame_index set means this region is on that specific frame of an
        animated file. polarity='negative' marks this specific region as a
        confirmed NON-example — e.g. a dog, boxed and labeled "monstera plant"
        with polarity='negative', a hard negative for the CLIP tag classifier
        (see clip_tag_classifier.py) — distinct from a whole-image negative
        (add_tag's polarity='negative'), which says "this photo has no X"
        rather than "this specific region here is definitely not X"."""
        tag_id = self._find_or_create_tag(label.strip())
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO file_tags (checksum, tag_id, polarity, x1,y1,x2,y2, image_width,image_height, created_at, frame_index) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (checksum, tag_id, polarity, x1, y1, x2, y2, image_width, image_height, int(time.time()), frame_index)
        )
        self.conn.commit()
        return cur.lastrowid

    def remove_tag(self, tag_id):
        """Deletes one tag *instance* row by its own id (the id param here is a
        file_tags.id, i.e. one photo's tag, not the shared tags.id definition)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_tags WHERE id = ?', (tag_id,))
        self.conn.commit()

    def remove_tag_by_label(self, checksum, label, polarity):
        """Delete a single whole-file tag row by (checksum, label, polarity) rather
        than by numeric id — lets a caller (the tag-suggestion swipe stream's undo)
        reverse its own add_tag call without having to thread the row id it never
        asked for back through a fetch response. A label with no matching tag
        definition at all has nothing to delete — no-op, not a create."""
        cur = self.conn.cursor()
        row = cur.execute('SELECT id FROM tags WHERE label = ?', (label,)).fetchone()
        if row is None:
            return
        cur.execute(
            'DELETE FROM file_tags WHERE checksum = ? AND tag_id = ? AND polarity = ? AND x1 IS NULL',
            (checksum, row[0], polarity)
        )
        self.conn.commit()

    def update_tag_label(self, tag_id, new_label):
        """Fix a typo on THIS ONE photo's tag instance (tag_id here is a file_tags.id) —
        deliberately a per-instance move, not a global rename: resolves/creates the tag
        definition for new_label and repoints just this one instance row to it, so
        every other photo still carrying the old label is unaffected. See
        rename_tag_label for the "rename everywhere" equivalent."""
        new_tag_id = self._find_or_create_tag(new_label.strip())
        cur = self.conn.cursor()
        cur.execute('UPDATE file_tags SET tag_id = ? WHERE id = ?', (new_tag_id, tag_id))
        self.conn.commit()

    def bump_tag_favorite(self, tag_id, delta):
        """Increment/decrement a tag instance's favorite count (floored at 0);
        returns the new count. tag_id is file_tags.id."""
        cur = self.conn.cursor()
        cur.execute('UPDATE file_tags SET favorite = max(0, favorite + ?) WHERE id = ?', (delta, tag_id))
        self.conn.commit()
        row = cur.execute('SELECT favorite FROM file_tags WHERE id = ?', (tag_id,)).fetchone()
        return row[0] if row else 0

    def rename_tag_label(self, old_label, new_label):
        """Renames a label everywhere it's used. If new_label already belongs to a
        *different* tag, this is a real merge — every instance row pointing at the old
        tag is repointed to the existing one and the now-empty old tag row is deleted
        (tags.label is UNIQUE, so two rows can never share a label); otherwise it's a
        plain single-row rename. Mirrors rename_studio exactly."""
        new_label = new_label.strip()
        cur = self.conn.cursor()
        old_row = cur.execute('SELECT id FROM tags WHERE label = ?', (old_label,)).fetchone()
        if old_row is None:
            return
        old_id = old_row[0]
        existing = cur.execute('SELECT id FROM tags WHERE label = ? AND id != ?', (new_label, old_id)).fetchone()
        if existing is not None:
            cur.execute('UPDATE file_tags SET tag_id = ? WHERE tag_id = ?', (existing[0], old_id))
            cur.execute('DELETE FROM tags WHERE id = ?', (old_id,))
        else:
            cur.execute('UPDATE tags SET label = ? WHERE id = ?', (new_label, old_id))
        self.conn.commit()

    def delete_tag_label(self, label):
        """Removes the tag everywhere — every instance row cascades away via
        file_tags.tag_id's ON DELETE CASCADE once the definition row is deleted."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM tags WHERE label = ?', (label,))
        self.conn.commit()

    def get_tags(self, checksum):
        """Return full tag rows (id, label, polarity, x1..y2, image_width, image_height) for a file."""
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM file_tags_with_label WHERE checksum = ? ORDER BY id', (checksum,))
        return cur.fetchall()

    def get_whole_image_tag_labels(self, checksum):
        """Positive whole-file (non-spatial, non-frame-scoped) tag labels — feeds
        card/nav display, never includes negative tags since those are corrections,
        not "this file has X". Frame-scoped tags don't show up in this default list."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT label FROM file_tags_with_label WHERE checksum = ? AND x1 IS NULL AND frame_index IS NULL "
            "AND polarity = 'positive' ORDER BY label",
            (checksum,)
        )
        return [row[0] for row in cur.fetchall()]

    def get_negated_labels(self, checksum):
        """Labels a human has explicitly rejected for this file — used to suppress a
        matching auto-detected class from showing up as a "Detected objects" chip."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT label FROM file_tags_with_label WHERE checksum = ? AND polarity = 'negative'",
            (checksum,)
        )
        return {row[0] for row in cur.fetchall()}

    def get_files_by_tag(self, label, limit=100):
        """Positive matches only — a negative tag is a rejection, it must never match search.
        Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checksum FROM file_tags_with_label WHERE label = ? AND polarity = 'positive' ORDER BY checksum LIMIT ?",
            (label, limit)
        )
        return [row[0] for row in cur.fetchall()]

    def get_decided_checksums_for_tag(self, label):
        """Checksums with ANY tag row for this label — positive (already tagged) or
        negative (explicitly rejected) — must never be re-suggested for this label."""
        cur = self.conn.cursor()
        cur.execute('SELECT DISTINCT checksum FROM file_tags_with_label WHERE label = ?', (label,))
        return {row[0] for row in cur.fetchall()}

    def get_whole_image_tag_polarities(self, label):
        """{checksum: polarity} for every whole-image (x1 IS NULL AND frame_index
        IS NULL) row of this label — the CLIP tag-classifier's training source:
        'positive' rows are this tag's confirmed examples, 'negative' rows are
        confirmed non-examples (a human explicitly rejected a detection), giving
        the classifier real negatives to learn from rather than just guessing
        from unlabeled photos."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT checksum, polarity FROM file_tags_with_label "
            "WHERE label = ? AND x1 IS NULL AND frame_index IS NULL",
            (label,)
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_spatial_tags_for_label(self, label):
        """[(checksum, x1, y1, x2, y2, image_width, image_height), ...] for this
        label's positive-polarity bbox rows (see add_spatial_tag) — the YOLO
        tag-classifier's training source, already in the same (image, box, label)
        shape a YOLO dataset needs. YOLO has no explicit "negative box" concept
        (an unboxed region in a training image is already an implicit negative),
        so negative-polarity regions are deliberately excluded here — see
        get_all_spatial_tags_for_label for the CLIP classifier's version, which
        needs both polarities as explicit hard-negative crops."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT checksum, x1, y1, x2, y2, image_width, image_height FROM file_tags_with_label "
            "WHERE label = ? AND x1 IS NOT NULL AND polarity = 'positive'",
            (label,)
        )
        return cur.fetchall()

    def get_all_spatial_tags_for_label(self, label):
        """[(checksum, x1, y1, x2, y2, image_width, image_height, polarity), ...]
        for this label's bbox rows of EITHER polarity — the CLIP tag
        classifier's region-level training source (see
        clip_tag_classifier.py): a positive region crop is "this specific area
        is X", a negative region crop is a hard negative — e.g. a dog boxed
        and labeled "monstera plant" with polarity='negative' — a much more
        precise training signal than a whole-image polarity when the same
        photo has both the target and a look-alike in it."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT checksum, x1, y1, x2, y2, image_width, image_height, polarity FROM file_tags_with_label "
            "WHERE label = ? AND x1 IS NOT NULL",
            (label,)
        )
        return cur.fetchall()

    def list_all_tags(self):
        """Return [(label, count), ...] for positive tags only, ordered by count descending."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT label, COUNT(*) as cnt FROM file_tags_with_label WHERE polarity = 'positive' GROUP BY label ORDER BY cnt DESC"
        )
        return cur.fetchall()

    def list_tags_for_checksums(self, checksums):
        """Batched positive whole-file (non-frame-scoped) tag lookup:
        {checksum: [label, ...]}. Avoids N+1 queries on list pages. Chunked (see
        _chunked) since callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f"SELECT checksum, label FROM file_tags_with_label WHERE x1 IS NULL AND frame_index IS NULL AND polarity = 'positive' "
                f'AND checksum IN ({placeholders}) ORDER BY label',
                tuple(chunk)
            )
            for checksum, label in cur.fetchall():
                result.setdefault(checksum, []).append(label)
        return result

    def get_all_positive_labels(self):
        """Distinct positive tag labels (whole-image + spatial) — expands the YOLO-World
        vocabulary with every object a human has confirmed, so it starts looking for
        things it wasn't originally told about."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT label FROM file_tags_with_label WHERE polarity = 'positive' ORDER BY label")
        return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # File favorites
    # ------------------------------------------------------------------

    def bump_file_favorite(self, checksum, delta):
        """Increment/decrement a photo's favorite count by `delta`, floored at 0.
        Presence in file_favorites still means favorited (count > 0): a decrement
        that reaches 0 deletes the row so every presence-based query stays correct.
        Returns the new count."""
        cur = self.conn.cursor()
        if delta > 0:
            cur.execute(
                'INSERT INTO file_favorites (checksum, count, created_at) VALUES (?, ?, ?) '
                'ON CONFLICT(checksum) DO UPDATE SET count = count + excluded.count',
                (checksum, delta, int(time.time()))
            )
        elif delta < 0:
            cur.execute('UPDATE file_favorites SET count = count + ? WHERE checksum = ?', (delta, checksum))
            cur.execute('DELETE FROM file_favorites WHERE checksum = ? AND count <= 0', (checksum,))
        self.conn.commit()
        return self.get_file_favorite_count(checksum)

    def add_frame_capture(self, child_checksum, parent_checksum, time_ms):
        """Record that `child_checksum` (a captured still) came from `parent_checksum`
        (a video) at `time_ms`. Idempotent — recapturing the identical frame reuses
        the row (same child checksum)."""
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO frame_captures (child_checksum, parent_checksum, time_ms, created_at) '
            'VALUES (?, ?, ?, ?) ON CONFLICT(child_checksum) DO UPDATE SET '
            'parent_checksum=excluded.parent_checksum, time_ms=excluded.time_ms',
            (child_checksum, parent_checksum, int(time_ms), int(time.time()))
        )
        self.conn.commit()

    def get_frame_captures_for(self, parent_checksum):
        """[{child_checksum, time_ms}] captured from this video, ordered by time."""
        cur = self.conn.cursor()
        cur.execute('SELECT child_checksum, time_ms FROM frame_captures WHERE parent_checksum = ? ORDER BY time_ms',
                    (parent_checksum,))
        return [{'child_checksum': r[0], 'time_ms': r[1]} for r in cur.fetchall()]

    def get_parent_capture(self, child_checksum):
        """{parent_checksum, time_ms} for a captured still, or None if not a capture."""
        cur = self.conn.cursor()
        cur.execute('SELECT parent_checksum, time_ms FROM frame_captures WHERE child_checksum = ?', (child_checksum,))
        row = cur.fetchone()
        return {'parent_checksum': row[0], 'time_ms': row[1]} if row else None

    def get_file_favorite_count(self, checksum):
        cur = self.conn.cursor()
        cur.execute('SELECT count FROM file_favorites WHERE checksum = ?', (checksum,))
        row = cur.fetchone()
        return row[0] if row else 0

    def get_favorite_counts(self, checksums):
        """Batched: {checksum: count} for the favorited subset of `checksums`
        (count > 0). Mirrors get_favorite_checksums; absent checksums are 0."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'SELECT checksum, count FROM file_favorites WHERE checksum IN ({placeholders})',
                tuple(chunk)
            )
            for row in cur.fetchall():
                result[row[0]] = row[1]
        return result

    def get_favorite_checksums(self, checksums):
        """Batched lookup: subset of `checksums` that are favorited. Chunked (see
        _chunked) since callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return set()
        cur = self.conn.cursor()
        result = set()
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'SELECT checksum FROM file_favorites WHERE checksum IN ({placeholders})',
                tuple(chunk)
            )
            result.update(row[0] for row in cur.fetchall())
        return result

    def is_file_favorite(self, checksum):
        cur = self.conn.cursor()
        cur.execute('SELECT 1 FROM file_favorites WHERE checksum = ?', (checksum,))
        return cur.fetchone() is not None

    def get_random_favorite_checksums(self, limit=6):
        """Random sample of favorited whole-photo checksums, bounded by `limit`
        regardless of how many photos are favorited — homepage Favorites teaser."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum FROM file_favorites ORDER BY RANDOM() LIMIT ?', (limit,))
        return [row[0] for row in cur.fetchall()]

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
        avoids N+1 queries on list pages (mirrors list_tags_for_checksums). Chunked
        (see _chunked) since callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'SELECT checksum, title FROM file_titles WHERE checksum IN ({placeholders})',
                tuple(chunk)
            )
            result.update(cur.fetchall())
        return result

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

    def get_ages_for_face_refs(self, face_refs):
        """Batched {face_ref: {'age': float, 'gender': str}} for a specific set of
        face refs — each ref's own most-recently-run estimate (mirrors
        get_age_estimate_for_face_ref's single-ref shape, batched). Unlike
        get_ages_for_identities_by_checksum (one value per (checksum, identity),
        averaged across however many of that identity's faces are on that photo),
        this resolves each INDIVIDUAL face instance's own age independently — needed
        when the same identity has more than one face on one photo (e.g. a
        before/after collage) and each occurrence must keep its own real estimate
        rather than being blended with the other's."""
        if not face_refs:
            return {}
        result = {}
        cur = self.conn.cursor()
        for chunk in self._chunked(list(face_refs)):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(f'''
                SELECT face_ref, age, gender FROM face_age_estimates
                WHERE face_ref IN ({placeholders})
                ORDER BY created_at DESC
            ''', tuple(chunk))
            for face_ref, age, gender in cur.fetchall():
                if face_ref not in result:
                    result[face_ref] = {'age': age, 'gender': gender}
        return result

    def get_ages_for_identities(self, identities):
        """Best-effort {identity: {'age': float, 'gender': str}} for every identity in
        `identities` that has an age/gender estimate on at least one of their faces
        (see face_age_estimates) — most identities won't, since estimating is a
        manual per-photo action, not automatic; callers should treat a missing key
        as "no estimate yet". An identity with estimates from more than one face
        uses the single most recently created one (mirrors
        get_age_estimate_for_face_ref's single-face resolution, just resolved
        across all of an identity's faces instead of one caller-chosen face)."""
        if not identities:
            return {}
        placeholders = ','.join('?' for _ in identities)
        cur = self.conn.cursor()
        cur.execute(f'''
            SELECT f.identity, fae.age, fae.gender
            FROM faces f
            JOIN face_age_estimates fae ON fae.face_ref = 'manual:' || f.id
            WHERE f.identity IN ({placeholders}) AND f.rejected = 0 AND fae.age IS NOT NULL
            ORDER BY fae.created_at DESC
        ''', tuple(identities))
        result = {}
        for identity, age, gender in cur.fetchall():
            if identity not in result:
                result[identity] = {'age': age, 'gender': gender}
        return result

    def get_average_ages_for_identities_in_sets(self, set_ids):
        """{(set_id, identity): {'age': float, 'gender': str}} averaged across only
        each set's own member photos (file_sets), not the identity's whole
        library — this is what a set's own roster (set_meta_line) should show,
        since "18" from some unrelated photo of the same person is meaningless
        context for this set. Gender is the most common non-null value across
        those faces (age estimates can drift photo to photo, but gender
        shouldn't); ties keep whichever SQLite happens to return first. Unlike
        get_ages_for_identities, this deliberately does NOT fall back to a
        global estimate when an identity has none within a given set — a
        missing key means "no estimate for this person within this set yet",
        not "no estimate anywhere". Batched across every set_id in one query
        (mirrors _attach_set_people's existing batching over multiple sets).

        Gender's "most common value" is resolved in Python from one flat
        query, not a correlated subquery re-scanning faces/face_age_estimates
        once per (set, identity) row — that pattern is O(rows × faces-per-set)
        and got slow as either grew; this is a single O(rows) pass."""
        if not set_ids:
            return {}
        placeholders = ','.join('?' for _ in set_ids)
        cur = self.conn.cursor()
        cur.execute(f'''
            SELECT fs.set_id, f.identity, fae.age, fae.gender
            FROM faces f
            JOIN face_age_estimates fae ON fae.face_ref = 'manual:' || f.id
            JOIN file_sets fs ON fs.checksum = f.checksum
            WHERE fs.set_id IN ({placeholders}) AND f.identity IS NOT NULL
                  AND f.rejected = 0 AND fae.age IS NOT NULL
        ''', tuple(set_ids))
        ages = {}      # (set_id, identity) -> [age, ...]
        genders = {}   # (set_id, identity) -> Counter(gender -> count)
        for set_id, identity, age, gender in cur.fetchall():
            key = (set_id, identity)
            ages.setdefault(key, []).append(age)
            if gender is not None:
                genders.setdefault(key, Counter())[gender] += 1
        return {
            key: {
                'age': sum(age_list) / len(age_list),
                'gender': genders[key].most_common(1)[0][0] if key in genders else None,
            }
            for key, age_list in ages.items()
        }

    def get_ages_for_identities_by_checksum(self, checksums):
        """{checksum: {identity: {'age': float, 'gender': str}}} — this specific
        photo's own estimate for each identity recognized in it, averaged across
        only that identity's face(s) on that one checksum (usually just one, but
        a person occasionally appears twice in a group photo). This is the
        per-photo counterpart to get_average_ages_for_identities_in_set: a photo
        card's "who's in this photo" chip should show what get_age_estimates_for_checksum
        would show if you clicked into that exact photo, not some other estimate
        of the same person from a different photo. Chunked (see _chunked) since
        callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(f'''
                SELECT f.checksum, f.identity, AVG(fae.age),
                       (SELECT gender FROM face_age_estimates fae2
                        JOIN faces f2 ON fae2.face_ref = 'manual:' || f2.id
                        WHERE f2.checksum = f.checksum AND f2.identity = f.identity
                              AND fae2.gender IS NOT NULL
                        GROUP BY fae2.gender ORDER BY COUNT(*) DESC LIMIT 1)
                FROM faces f
                JOIN face_age_estimates fae ON fae.face_ref = 'manual:' || f.id
                WHERE f.checksum IN ({placeholders}) AND f.identity IS NOT NULL
                      AND f.rejected = 0 AND fae.age IS NOT NULL
                GROUP BY f.checksum, f.identity
            ''', tuple(chunk))
            for checksum, identity, age, gender in cur.fetchall():
                result.setdefault(checksum, {})[identity] = {'age': age, 'gender': gender}
        return result

    def get_estimated_checksums(self, checksums):
        """Batched lookup: subset of `checksums` that already have at least one
        face_age_estimates row (regardless of whether age/gender actually resolved —
        a face_ref with a row here has already been run through the model). Mirrors
        get_favorite_checksums's exact shape. Lets "estimate all" on a person page
        skip photos already done on a previous run instead of redoing them —
        deliberately not get_average_ages_for_checksums, which filters out NULL ages
        and would wrongly re-queue a photo whose face(s) failed to yield an age.
        Chunked (see _chunked) since callers can hand this an unbounded, library-wide
        list."""
        if not checksums:
            return set()
        cur = self.conn.cursor()
        result = set()
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'SELECT DISTINCT checksum FROM face_age_estimates WHERE checksum IN ({placeholders})',
                tuple(chunk)
            )
            result.update(row[0] for row in cur.fetchall())
        return result

    def get_average_ages_for_checksums(self, checksums):
        """Batched lookup: {checksum: average age} across every face with an estimate
        on that photo — feeds "sort by age" on gallery-style views. A photo with no
        estimate at all is simply absent from the result (not 0). Chunked (see
        _chunked) since callers can hand this an unbounded, library-wide list —
        this is exactly the gallery's `sort=age` path, which enriches the whole
        library before slicing to a page."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'''SELECT checksum, AVG(age) FROM face_age_estimates
                    WHERE checksum IN ({placeholders}) AND age IS NOT NULL
                    GROUP BY checksum''',
                tuple(chunk)
            )
            result.update(cur.fetchall())
        return result

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------

    def _find_or_create_studio(self, name):
        """Exact-match find-or-create for a studio row by name. Deliberately not
        case-insensitive — the "did you mean the existing studio 'Acme'?" prompt
        for a near-miss spelling lives entirely in find_studio_variant, called by
        web.py *before* create_set/rename_set ever get a confirmed name; this just
        resolves or creates whatever exact string it's given."""
        cur = self.conn.cursor()
        row = cur.execute('SELECT id FROM studios WHERE name = ?', (name,)).fetchone()
        if row is not None:
            return row[0]
        cur.execute('INSERT INTO studios (name, created_at) VALUES (?, ?)', (name, int(time.time())))
        return cur.lastrowid

    def create_set(self, name, studio=None):
        """Idempotent: returns the existing set's id if (name, studio) already
        exists. Checks via find_set first rather than relying on `sets`'
        UNIQUE(name, studio) + INSERT OR IGNORE — SQLite treats every NULL
        studio as distinct for UNIQUE purposes, so that combination silently
        inserted a fresh duplicate row every time for the common
        studio-less case instead of ever actually being ignored."""
        name = name.strip()
        studio = studio.strip() if studio else None
        existing = self.find_set(name, studio)
        if existing is not None:
            return existing['id']
        studio_id = self._find_or_create_studio(studio) if studio else None
        cur = self.conn.cursor()
        cur.execute(
            'INSERT INTO sets (name, studio_id, created_at) VALUES (?, ?, ?)',
            (name, studio_id, int(time.time()))
        )
        self.conn.commit()
        return cur.lastrowid

    def find_set(self, name, studio=None):
        cur = self.conn.cursor()
        cur.execute(
            'SELECT * FROM sets_with_studio WHERE name = ? AND studio IS ?',
            (name, studio)
        )
        return cur.fetchone()

    def get_set(self, set_id):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM sets_with_studio WHERE id = ?', (set_id,))
        return cur.fetchone()

    def list_sets(self, favorite_only=False, studio=None):
        cur = self.conn.cursor()
        clauses = []
        params = []
        if favorite_only:
            clauses.append('s.favorite > 0')
        if studio is not None:
            clauses.append('s.studio = ?')
            params.append(studio)
        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        cur.execute(f'''
            SELECT s.id, s.name, s.studio, s.created_at, s.favorite, COUNT(fs.checksum) as image_count
            FROM sets_with_studio s
            LEFT JOIN file_sets fs ON fs.set_id = s.id
            {where}
            GROUP BY s.id
            ORDER BY s.name
        ''', tuple(params))
        return cur.fetchall()

    def count_sets(self):
        """Cheap set count, avoiding list_sets()'s LEFT JOIN/image_count aggregation
        — used by the homepage stats section, which runs on every page load."""
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM sets')
        return cur.fetchone()[0]

    def list_studios(self):
        """Every studio with at least one set, with set count and total image count
        — feeds the Studios wall page. Real join against the studios table now,
        not a GROUP BY over a free-text sets.studio column."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT st.name as studio, COUNT(DISTINCT s.id) as set_count, COUNT(fs.checksum) as image_count
            FROM studios st
            JOIN sets s ON s.studio_id = st.id
            LEFT JOIN file_sets fs ON fs.set_id = s.id
            GROUP BY st.id
            ORDER BY st.name
        ''')
        return cur.fetchall()

    def rename_studio(self, old_name, new_name):
        """Renames a studio everywhere it's used. If new_name already belongs to a
        *different* studio, this is a real merge — every set pointing at the old
        studio is repointed to the existing one and the now-empty old studio row
        is deleted (studios.name is UNIQUE, so two rows can never share a name);
        otherwise it's a plain single-row rename."""
        new_name = new_name.strip()
        cur = self.conn.cursor()
        old_row = cur.execute('SELECT id FROM studios WHERE name = ?', (old_name,)).fetchone()
        if old_row is None:
            return
        old_id = old_row[0]
        existing = cur.execute('SELECT id FROM studios WHERE name = ? AND id != ?', (new_name, old_id)).fetchone()
        if existing is not None:
            cur.execute('UPDATE sets SET studio_id = ? WHERE studio_id = ?', (existing[0], old_id))
            cur.execute('DELETE FROM studios WHERE id = ?', (old_id,))
        else:
            cur.execute('UPDATE studios SET name = ? WHERE id = ?', (new_name, old_id))
        self.conn.commit()

    def rename_set(self, set_id, name, studio=None):
        studio = studio.strip() if studio else None
        studio_id = self._find_or_create_studio(studio) if studio else None
        cur = self.conn.cursor()
        cur.execute(
            'UPDATE sets SET name = ?, studio_id = ? WHERE id = ?',
            (name.strip(), studio_id, set_id)
        )
        self.conn.commit()

    def find_sets_by_name(self, name, exclude_id=None):
        """Case-insensitive name match, ignoring studio — the `sets` table's own
        UNIQUE(name, studio) constraint doesn't actually stop duplicates in
        practice (SQLite treats every NULL studio as distinct for UNIQUE
        purposes, and studio is NULL on most sets), so this is what backs the
        rename-time "a set with this name already exists, merge them?" prompt."""
        cur = self.conn.cursor()
        query = 'SELECT * FROM sets_with_studio WHERE LOWER(name) = LOWER(?)'
        params = [name]
        if exclude_id is not None:
            query += ' AND id != ?'
            params.append(exclude_id)
        cur.execute(query, params)
        return cur.fetchall()

    def find_studio_variant(self, studio, exclude_set_id=None):
        """If a studio already exists elsewhere under a different exact
        spelling (case-insensitive match, different exact string — e.g. typed
        "acme" when sets already use "Acme"), return its existing spelling +
        how many sets/images use it — backs a prompt offering to reuse the
        existing spelling instead of fragmenting the same studio across two
        differently-cased studios rows."""
        if not studio:
            return None
        cur = self.conn.cursor()
        query = '''
            SELECT st.name as studio, COUNT(DISTINCT s.id) as set_count, COUNT(fs.checksum) as image_count
            FROM studios st
            JOIN sets s ON s.studio_id = st.id
            LEFT JOIN file_sets fs ON fs.set_id = s.id
            WHERE LOWER(st.name) = LOWER(?) AND st.name != ?
        '''
        params = [studio, studio]
        if exclude_set_id is not None:
            query += ' AND s.id != ?'
            params.append(exclude_set_id)
        query += ' GROUP BY st.id'
        cur.execute(query, params)
        return cur.fetchone()

    def find_identity_variant(self, name):
        """If a different person is already known under this name's exact
        case-insensitive spelling but a different exact string (e.g. renaming
        onto "joe" when "Joe" already exists), return that identity + its
        face count. A rename onto an *exact*-spelling match already merges
        automatically (identity is just a shared string key with no separate
        id — every face row simply now reads the same string) — this only
        catches the case-differs case, which wouldn't share that string and
        so wouldn't otherwise merge, backing a warn-before-rename prompt."""
        if not name:
            return None
        cur = self.conn.cursor()
        cur.execute('''
            SELECT identity, COUNT(*) as cnt FROM faces
            WHERE identity IS NOT NULL AND rejected = 0 AND LOWER(identity) = LOWER(?) AND identity != ?
            GROUP BY identity
        ''', (name, name))
        return cur.fetchone()

    def merge_sets(self, source_id, dest_id):
        """Folds source_id into dest_id: every member photo, exclusion, and
        linked identity moves over (skipped, not errored, wherever dest_id
        already has that same row — see the UPDATE OR IGNORE calls below),
        dest_id's favorite flag is OR'd with source's, then source_id itself
        is deleted. Deleting it cascades away whatever file_sets/
        file_set_exclusions/identity_set_assignments rows still point at it —
        exactly the ones skipped above because dest_id already had that same
        (checksum/identity) pairing, so it's correct for them to just vanish
        rather than raise a duplicate-key error. Returns dest_id."""
        cur = self.conn.cursor()
        cur.execute('UPDATE OR IGNORE file_sets SET set_id = ? WHERE set_id = ?', (dest_id, source_id))
        cur.execute('UPDATE OR IGNORE file_set_exclusions SET set_id = ? WHERE set_id = ?', (dest_id, source_id))
        cur.execute('UPDATE OR IGNORE identity_set_assignments SET set_id = ? WHERE set_id = ?', (dest_id, source_id))
        cur.execute('UPDATE OR IGNORE set_aliases SET set_id = ? WHERE set_id = ?', (dest_id, source_id))
        source = self.get_set(source_id)
        if source is not None and source['favorite']:
            # Favorites are counters now — sum the merged set's count into dest.
            cur.execute('UPDATE sets SET favorite = favorite + ? WHERE id = ?', (source['favorite'], dest_id))
        cur.execute('DELETE FROM sets WHERE id = ?', (source_id,))
        self.conn.commit()
        return dest_id

    def delete_set(self, set_id):
        """Deletes the set entity only — member files are untouched, `file_sets` rows
        for it are removed automatically via ON DELETE CASCADE."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM sets WHERE id = ?', (set_id,))
        self.conn.commit()

    def bump_set_favorite(self, set_id, delta):
        """Increment/decrement a set's favorite count (floored at 0); returns
        the new count."""
        cur = self.conn.cursor()
        cur.execute('UPDATE sets SET favorite = max(0, favorite + ?) WHERE id = ?', (delta, set_id))
        self.conn.commit()
        row = cur.execute('SELECT favorite FROM sets WHERE id = ?', (set_id,)).fetchone()
        return row[0] if row else 0

    def add_set_alias(self, set_id, alias):
        """Record `alias` as an alternate name for this set — same shape/
        reasoning as add_identity_alias: unique across every set (PRIMARY KEY
        on alias alone), silently repointed to this set_id if re-added
        elsewhere (INSERT OR REPLACE)."""
        alias = alias.strip()
        if not alias:
            return
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR REPLACE INTO set_aliases (set_id, alias, created_at) VALUES (?, ?, ?)',
            (set_id, alias, int(time.time()))
        )
        self.conn.commit()

    def remove_set_alias(self, alias):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM set_aliases WHERE alias = ?', (alias,))
        self.conn.commit()

    def get_aliases_for_set(self, set_id):
        cur = self.conn.cursor()
        cur.execute('SELECT alias FROM set_aliases WHERE set_id = ? ORDER BY alias', (set_id,))
        return [row[0] for row in cur.fetchall()]

    def get_aliases_for_sets(self, set_ids):
        """{set_id: [alias, ...]} for every set_id in set_ids, one query total —
        same batching reasoning as _attach_set_people's get_identities_for_sets,
        for /api/sets attaching aliases to every set at once."""
        if not set_ids:
            return {}
        cur = self.conn.cursor()
        placeholders = ','.join('?' for _ in set_ids)
        cur.execute(f'SELECT set_id, alias FROM set_aliases WHERE set_id IN ({placeholders}) ORDER BY alias', tuple(set_ids))
        result = {}
        for set_id, alias in cur.fetchall():
            result.setdefault(set_id, []).append(alias)
        return result

    def find_set_id_by_alias(self, alias):
        """Resolves an alias to its owning set_id, or None if unknown — the
        set-alias counterpart to resolve_identity_name."""
        cur = self.conn.cursor()
        cur.execute('SELECT set_id FROM set_aliases WHERE alias = ?', (alias,))
        row = cur.fetchone()
        return row[0] if row else None

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

    def get_files_by_set(self, set_id, limit=200, manual_order=False):
        """Returns checksums; resolve to current file_id/path via Database.get_files_by_checksums.

        With manual_order=True, orders by the user-assigned `position` (see
        set_file_positions), placing never-positioned members (NULL) last, tie-broken
        by created_at then checksum for a stable order. Without it, order is left to
        SQLite (rowid/insertion order) as before."""
        cur = self.conn.cursor()
        if manual_order:
            cur.execute('''
                SELECT checksum FROM file_sets WHERE set_id = ?
                ORDER BY (position IS NULL), position, created_at, checksum
                LIMIT ?
            ''', (set_id, limit))
        else:
            cur.execute('SELECT checksum FROM file_sets WHERE set_id = ? LIMIT ?', (set_id, limit))
        return [row[0] for row in cur.fetchall()]

    def set_file_positions(self, set_id, ordered_checksums):
        """Persist a full manual ordering for a set: writes position = index for each
        checksum in the given order. Writing the entire order (rather than swapping
        two rows) keeps positions dense and lets the up/down and number-input UI
        actions share one code path. Checksums not in this set are simply no-ops."""
        cur = self.conn.cursor()
        for idx, checksum in enumerate(ordered_checksums):
            cur.execute('UPDATE file_sets SET position = ? WHERE set_id = ? AND checksum = ?',
                        (idx, set_id, checksum))
        self.conn.commit()

    def get_sets_for_file(self, checksum):
        """Every set this file belongs to (a file can be in more than one)."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT s.* FROM sets_with_studio s JOIN file_sets fs ON fs.set_id = s.id
            WHERE fs.checksum = ? ORDER BY s.name
        ''', (checksum,))
        return cur.fetchall()

    def get_top_favorite_checksums(self, limit=60):
        """Favorited checksums, most-favorited first — (checksum, count) rows.
        file_favorites only holds favorited rows so this stays small/cheap."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum, count FROM file_favorites WHERE count > 0 '
                    'ORDER BY count DESC, checksum LIMIT ?', (limit,))
        return cur.fetchall()

    def get_checksums_with_manual_tags(self):
        """Whole-library set() of checksums carrying at least one positive manual tag —
        one indexed DISTINCT scan (idx_file_tags_checksum). For the home 'needs
        attention' exclusion."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT checksum FROM file_tags WHERE polarity = 'positive'")
        return {row[0] for row in cur.fetchall()}

    def count_set_member_checksums(self):
        """Cheap count of distinct set-member checksums (no set() materialization)."""
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT checksum) FROM file_sets')
        row = cur.fetchone()
        return row[0] if row else 0

    def count_identities(self):
        """Cheap count of distinct confirmed identities (mirrors get_identity_summary's
        filter) without the per-identity GROUP BY + thumbnail resolution."""
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT identity) FROM faces WHERE identity IS NOT NULL AND rejected = 0')
        row = cur.fetchone()
        return row[0] if row else 0

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
        queries on list pages (mirrors list_tags_for_checksums). Chunked (see
        _chunked) since callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'''SELECT fs.checksum, s.id, s.name, s.studio
                    FROM file_sets fs JOIN sets_with_studio s ON s.id = fs.set_id
                    WHERE fs.checksum IN ({placeholders}) ORDER BY s.name''',
                tuple(chunk)
            )
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
        image_count here is manual-assignment-only (file_categories rows), mirroring
        list_sets's image_count."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT c.id, c.name, c.temperature, c.created_at,
                   COUNT(fc.checksum) as image_count
            FROM categories c
            LEFT JOIN file_categories fc
                ON fc.category_id = c.id
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
        """Deletes the category entity only. Any file_categories row that pointed
        at it is removed automatically via ON DELETE CASCADE — those files simply
        lose this one category (they may still have others) rather than being
        pushed into "explicitly no categories" (file_category_excluded is a
        separate, unrelated decision — see its own docstring)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM categories WHERE id = ?', (category_id,))
        self.conn.commit()

    def add_file_category(self, checksum, category_id):
        """Add category_id to this file's manual categories — additive, never
        replaces any category the file already has (mirrors assign_file_to_set)."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO file_categories (checksum, category_id, created_at)
            VALUES (?, ?, ?)
        ''', (checksum, category_id, int(time.time())))
        self.conn.commit()

    def remove_file_category(self, checksum, category_id):
        """Remove just this one category from the file — any other categories it
        has are untouched (mirrors remove_file_from_set)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_categories WHERE checksum = ? AND category_id = ?', (checksum, category_id))
        self.conn.commit()

    def get_categories_for_checksum(self, checksum):
        """Every category manually assigned to this file — {'id', 'name'} per row."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT c.id, c.name FROM file_categories fc JOIN categories c ON c.id = fc.category_id
            WHERE fc.checksum = ? ORDER BY c.name
        ''', (checksum,))
        return [{'id': row[0], 'name': row[1]} for row in cur.fetchall()]

    def get_categories_for_checksums(self, checksums):
        """Batched lookup: {checksum: [{'id','name'}, ...]} — avoids N+1 queries on
        list pages (mirrors get_sets_for_checksums). Chunked (see _chunked) since
        callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(f'''
                SELECT fc.checksum, c.id, c.name
                FROM file_categories fc JOIN categories c ON c.id = fc.category_id
                WHERE fc.checksum IN ({placeholders}) ORDER BY c.name
            ''', tuple(chunk))
            for checksum, cat_id, name in cur.fetchall():
                result.setdefault(checksum, []).append({'id': cat_id, 'name': name})
        return result

    def exclude_file_from_category(self, checksum, category_id):
        """Record a human's 'not this category' decision for this file — the
        category-suggestion stream's reject action. Permanent ground truth, same
        shape/reasoning as exclude_file_from_set: a file rejected for category X
        is never offered as a suggestion for X again, but remains fully eligible
        for every other category."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO file_category_exclusions (checksum, category_id, created_at) VALUES (?, ?, ?)
        ''', (checksum, category_id, int(time.time())))
        self.conn.commit()

    def remove_category_exclusion(self, checksum, category_id):
        """Undo a prior exclude_file_from_category call (the reject-swipe's Ctrl+Z)."""
        cur = self.conn.cursor()
        cur.execute('DELETE FROM file_category_exclusions WHERE checksum = ? AND category_id = ?', (checksum, category_id))
        self.conn.commit()

    def get_excluded_checksums_for_category(self, category_id):
        """Every checksum a human has confirmed does NOT belong in this category —
        used to permanently filter these out of future suggestion candidates for
        this category (mirrors get_excluded_checksums_for_set)."""
        cur = self.conn.cursor()
        cur.execute('SELECT checksum FROM file_category_exclusions WHERE category_id = ?', (category_id,))
        return {row[0] for row in cur.fetchall()}

    def get_category_exclusions_for_checksums(self, checksums):
        """Batched lookup: {checksum: {category_id, ...}} — which categories each
        of `checksums` has been rejected for, used by the resolver to suppress
        just those specific auto-matches (mirrors get_categories_for_checksums'
        batching shape). Chunked (see _chunked) since callers can hand this an
        unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'SELECT checksum, category_id FROM file_category_exclusions WHERE checksum IN ({placeholders})',
                tuple(chunk)
            )
            for checksum, category_id in cur.fetchall():
                result.setdefault(checksum, set()).add(category_id)
        return result

    def get_example_checksums_for_category(self, category_id, limit=1000):
        """Files manually assigned to this category — the ML training examples
        used to build a similarity centroid (mirrors get_files_by_set)."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT checksum FROM file_categories WHERE category_id = ? LIMIT ?',
            (category_id, limit)
        )
        return [row[0] for row in cur.fetchall()]

    def get_all_category_checksums(self):
        """{category_id: set(checksum), ...} for every manual category assignment,
        one query total — avoids the one-query-per-category loop get_category_counts
        and the search palette's _category_checksums_by_name used to do (mirrors
        _set_checksums_by_id/_tag_checksums_by_label's batching idiom in web.py)."""
        cur = self.conn.cursor()
        cur.execute('SELECT category_id, checksum FROM file_categories')
        result = {}
        for category_id, checksum in cur.fetchall():
            result.setdefault(category_id, set()).add(checksum)
        return result

    def get_all_category_exclusions(self):
        """{category_id: set(checksum), ...} for every category exclusion, one
        query total — same batching reasoning as get_all_category_checksums."""
        cur = self.conn.cursor()
        cur.execute('SELECT category_id, checksum FROM file_category_exclusions')
        result = {}
        for category_id, checksum in cur.fetchall():
            result.setdefault(category_id, set()).add(checksum)
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
        self._face_ver += 1
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
        self._face_ver += 1
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
        self._face_ver += 1

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
        self._face_ver += 1

    def reject_face(self, manual_face_id):
        """Reject an existing manual.db row (manually-added or already-promoted) — kept,
        not deleted, for the same reason as reject_auto_face."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET rejected = 1, identity = NULL WHERE id = ?', (manual_face_id,))
        self.conn.commit()
        self._face_ver += 1

    def assign_identity(self, manual_face_id, name):
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET identity = ?, rejected = 0 WHERE id = ?', (name.strip(), manual_face_id))
        self.conn.commit()
        self._face_ver += 1

    def mark_face_reviewed_ok(self, manual_face_id):
        """"Correct, don't ask again" from the cleanup-low-confidence-matches
        review — excludes this face from all future cleanup-queue builds
        regardless of its computed self-consistency score."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET reviewed_ok = 1 WHERE id = ?', (manual_face_id,))
        self.conn.commit()
        self._face_ver += 1

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
        self._face_ver += 1
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
        cur.execute('UPDATE identity_aliases SET identity = ? WHERE identity = ?', (new_name, old_name))
        self.conn.commit()
        self._face_ver += 1

    def add_identity_alias(self, identity, alias):
        """Record `alias` as an alternate name for `identity` ("babe" for
        "girlfriend"). An alias is unique across the whole table (PRIMARY KEY on
        alias alone) — it can't simultaneously mean two different people, and it
        silently overwrites its own previous canonical identity if re-added
        pointing elsewhere (INSERT OR REPLACE), matching "aliases are just a
        lookup shortcut" rather than a decision worth erroring over."""
        alias = alias.strip()
        identity = identity.strip()
        if not alias or alias == identity:
            return
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR REPLACE INTO identity_aliases (identity, alias, created_at) VALUES (?, ?, ?)',
            (identity, alias, int(time.time()))
        )
        self.conn.commit()

    def remove_identity_alias(self, alias):
        cur = self.conn.cursor()
        cur.execute('DELETE FROM identity_aliases WHERE alias = ?', (alias,))
        self.conn.commit()

    def get_aliases_for_identity(self, identity):
        cur = self.conn.cursor()
        cur.execute('SELECT alias FROM identity_aliases WHERE identity = ? ORDER BY alias', (identity,))
        return [row[0] for row in cur.fetchall()]

    def resolve_identity_name(self, name):
        """If `name` is a known alias, return its canonical identity; otherwise
        return `name` unchanged (it's either already canonical, or genuinely
        unknown — callers that need to distinguish 'unknown' already check that
        separately, e.g. via _identity_checksums coming back empty)."""
        cur = self.conn.cursor()
        cur.execute('SELECT identity FROM identity_aliases WHERE alias = ?', (name,))
        row = cur.fetchone()
        return row[0] if row else name

    def bump_face_favorite(self, manual_face_id, delta):
        """Increment/decrement a manual face's favorite count (floored at 0);
        returns the new count."""
        cur = self.conn.cursor()
        cur.execute('UPDATE faces SET favorite = max(0, favorite + ?) WHERE id = ?', (delta, manual_face_id))
        self.conn.commit()
        self._face_ver += 1
        row = cur.execute('SELECT favorite FROM faces WHERE id = ?', (manual_face_id,)).fetchone()
        return row[0] if row else 0

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
        this photo" chips without an N+1 query per file. Chunked (see _chunked)
        since callers can hand this an unbounded, library-wide list."""
        if not checksums:
            return {}
        cur = self.conn.cursor()
        result = {}
        for chunk in self._chunked(checksums):
            placeholders = ','.join('?' for _ in chunk)
            cur.execute(
                f'''SELECT DISTINCT checksum, identity FROM faces
                    WHERE checksum IN ({placeholders}) AND identity IS NOT NULL AND rejected = 0
                    ORDER BY identity''',
                tuple(chunk)
            )
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

    def get_all_checksums_with_manual_face_decision(self):
        """Every checksum with at least one manual.db faces row at all —
        confirmed, rejected, or otherwise manually touched. Broader than
        get_all_checksums_with_named_face (which excludes rejected rows): a
        rejected-but-undecided face still represents real human work that a
        `--reindex` rescan must not silently discard by regenerating the
        underlying auto-detected face rows out from under it (see
        database.py's clear_primary_ml_data `skip_faces` param)."""
        cur = self.conn.cursor()
        cur.execute('SELECT DISTINCT checksum FROM faces')
        return {row[0] for row in cur.fetchall()}

    def get_favorite_faces(self):
        """Non-rejected, favorited manual.db face rows (always named or hand-drawn,
        since only manual rows can be favorited)."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT id, checksum, identity FROM faces WHERE favorite > 0 AND rejected = 0 ORDER BY identity, id'
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

    def get_confirmed_faces_for_cleanup(self):
        """Return [(id, checksum, identity, embedding_bytes, reviewed_ok), ...],
        deduplicated to one row per (checksum, identity) pair (see
        list_confirmed_faces for why duplicates happen) — the
        cleanup-low-confidence-matches feature needs ALL of an identity's
        (unique) faces, including already-reviewed ones, to build that
        identity's comparison pool, then filters to reviewed_ok = 0 for which
        faces actually become review candidates. Without deduping here, a
        duplicate confirmation of the exact same face would inflate its
        identity's comparison pool and, being near-identical to itself, would
        never look like the outlier it might actually be hiding."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT MAX(id) AS id, checksum, identity, embedding, reviewed_ok FROM faces
            WHERE identity IS NOT NULL AND rejected = 0
            GROUP BY checksum, identity
        ''')
        return cur.fetchall()

    def get_named_face_embeddings(self):
        """Return [(identity, embedding_bytes), ...] for faces a human has named."""
        cur = self.conn.cursor()
        cur.execute("SELECT identity, embedding FROM faces WHERE identity IS NOT NULL AND rejected = 0")
        return cur.fetchall()

    def get_named_face_matrix(self):
        """Cached, write-invalidated counterpart to get_named_face_embeddings for the
        hot auto-match path (find_matching_identity): returns
        (identities: list[str], matrix: np.ndarray[K, D] float32) where row i of
        matrix is the embedding of face identities[i]. The K×D matrix is built once
        and reused across requests, rebuilt only when the faces table changes (see
        self._face_ver) — instead of re-SELECTing every named embedding blob and
        re-stacking it on EVERY newly detected face, which is a web-server OOM cause.

        Empty result → ([], np.empty((0, 0), np.float32)). Rows with an empty or
        wrong-length embedding blob are skipped (with a printed warning) so one bad
        row can't crash the rebuild. Returns the SAME cached objects on a cache hit
        — callers must treat the matrix as read-only (don't mutate in place)."""
        import numpy as np
        with self._matrix_lock:
            cached = self._named_matrix_cache
            if cached is not None and cached[0] == self._face_ver:
                return cached[1]

            cur = self.conn.cursor()
            cur.execute(
                "SELECT identity, embedding FROM faces "
                "WHERE identity IS NOT NULL AND rejected = 0 AND embedding != x''"
            )
            rows = cur.fetchall()

            identities = []
            blobs = []
            dim = None
            for identity, blob in rows:
                if not blob:
                    print(f"WARNING: get_named_face_matrix dropping face for identity "
                          f"{identity!r} — empty embedding blob")
                    continue
                if len(blob) % 4 != 0:
                    print(f"WARNING: get_named_face_matrix dropping face for identity "
                          f"{identity!r} — embedding blob length {len(blob)} not a "
                          f"multiple of 4 (not float32)")
                    continue
                row_dim = len(blob) // 4
                if dim is None:
                    dim = row_dim
                elif row_dim != dim:
                    print(f"WARNING: get_named_face_matrix dropping face for identity "
                          f"{identity!r} — embedding dim {row_dim} != expected {dim}")
                    continue
                identities.append(identity)
                blobs.append(blob)

            if not blobs:
                result = ([], np.empty((0, 0), dtype=np.float32))
            else:
                matrix = np.frombuffer(b''.join(blobs), dtype=np.float32).reshape(len(blobs), dim)
                result = (identities, matrix)

            self._named_matrix_cache = (self._face_ver, result)
            return result

    def get_all_faces_matrix(self):
        """Cached, write-invalidated counterpart to get_all_faces_with_embedding:
        returns (face_ids: np.ndarray[int64], checksums: list[str],
        identities: list, matrix: np.ndarray[K, D] float32), where row i of matrix is
        the embedding of the face face_ids[i] / checksums[i] / identities[i] (identity
        may be None for an unnamed face). Built once and reused across requests,
        rebuilt only when the faces table changes (see self._face_ver) — instead of
        dumping every non-rejected embedding blob per request (a web-server OOM cause).

        Empty result → (np.empty(0, np.int64), [], [], np.empty((0, 0), np.float32)).
        Rows with an empty or wrong-length embedding blob are skipped (with a printed
        warning). Returns the SAME cached objects on a cache hit — callers must treat
        the returned arrays/lists as read-only (don't mutate in place)."""
        import numpy as np
        with self._matrix_lock:
            cached = self._all_faces_matrix_cache
            if cached is not None and cached[0] == self._face_ver:
                return cached[1]

            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, checksum, identity, embedding FROM faces "
                "WHERE rejected = 0 AND embedding != x''"
            )
            rows = cur.fetchall()

            face_ids = []
            checksums = []
            identities = []
            blobs = []
            dim = None
            for face_id, checksum, identity, blob in rows:
                if not blob:
                    print(f"WARNING: get_all_faces_matrix dropping face id {face_id} "
                          f"(checksum {checksum}) — empty embedding blob")
                    continue
                if len(blob) % 4 != 0:
                    print(f"WARNING: get_all_faces_matrix dropping face id {face_id} "
                          f"(checksum {checksum}) — embedding blob length {len(blob)} "
                          f"not a multiple of 4 (not float32)")
                    continue
                row_dim = len(blob) // 4
                if dim is None:
                    dim = row_dim
                elif row_dim != dim:
                    print(f"WARNING: get_all_faces_matrix dropping face id {face_id} "
                          f"(checksum {checksum}) — embedding dim {row_dim} != "
                          f"expected {dim}")
                    continue
                face_ids.append(face_id)
                checksums.append(checksum)
                identities.append(identity)
                blobs.append(blob)

            if not blobs:
                result = (np.empty(0, dtype=np.int64), [], [], np.empty((0, 0), dtype=np.float32))
            else:
                matrix = np.frombuffer(b''.join(blobs), dtype=np.float32).reshape(len(blobs), dim)
                result = (np.asarray(face_ids, dtype=np.int64), checksums, identities, matrix)

            self._all_faces_matrix_cache = (self._face_ver, result)
            return result

    def find_matching_identity(self, embedding_bytes, threshold=None):
        """Return (identity, score) for the closest known person if similarity clears
        `threshold` (defaults to AUTO_MATCH_THRESHOLD), else (None, None). Powers
        auto-matching a newly detected face to an existing identity without asking."""
        if threshold is None:
            threshold = AUTO_MATCH_THRESHOLD
        import numpy as np
        # Cached, write-invalidated K×D matrix — no per-call re-SELECT/re-stack of
        # every named embedding (this method runs once per newly detected face).
        names, matrix = self.get_named_face_matrix()
        if matrix.shape[0] == 0:
            return None, None
        query = np.frombuffer(embedding_bytes, dtype=np.float32)
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

    def get_faces_for_identity(self, name):
        """[(face_id, checksum, embedding_bytes), ...] for every real face row
        currently assigned to this identity — the candidate pool for the
        "Split" review (see get_embeddings_for_identity, same WHERE clause,
        plus id/checksum since split needs to reassign each specific face, not
        just compare embeddings)."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, checksum, embedding FROM faces WHERE identity = ? AND rejected = 0", (name,))
        return cur.fetchall()

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

    def get_identity_summary(self):
        """One row per unique identity — (identity, count, last_modified, ref,
        checksum) — for the single unique-faces grid on /faces. count and
        last_modified (a unix timestamp) are computed over every non-rejected
        confirmed face for that identity; last_modified is the created_at of
        the most recently confirmed one, used as a "last modified" proxy since
        faces has no updated_at column. The representative thumbnail is that
        same most-recent row (MAX(id)), so the grid shows the newest-confirmed
        face rather than an arbitrary one."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT identity, COUNT(*) as cnt, MAX(created_at) as last_modified, MAX(id) as rep_id
            FROM faces
            WHERE identity IS NOT NULL AND rejected = 0
            GROUP BY identity
        ''')
        rows = cur.fetchall()
        if not rows:
            return []
        rep_ids = [row['rep_id'] for row in rows]
        placeholders = ','.join('?' for _ in rep_ids)
        cur.execute(f'SELECT id, checksum FROM faces WHERE id IN ({placeholders})', rep_ids)
        checksum_by_id = {r['id']: r['checksum'] for r in cur.fetchall()}
        return [
            {
                'identity': row['identity'],
                'count': row['cnt'],
                'last_modified': row['last_modified'],
                'ref': f"manual:{row['rep_id']}",
                'checksum': checksum_by_id.get(row['rep_id']),
            }
            for row in rows
        ]

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

    def get_files_by_face_identity_exact(self, name, limit=1000):
        """Exact-match sibling of get_files_by_face_identity — identity names on
        faces are always assigned verbatim (never free-typed), so callers that
        already have a canonical name in hand (e.g. /person/{name}) don't need
        substring matching, and exact match lets this use idx_manual_faces_identity
        instead of forcing a full table scan the way LOWER(identity) LIKE '%...%'
        does (see _identity_checksums_by_name in web.py for prior art on this)."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checksum FROM faces WHERE identity = ? AND rejected = 0 LIMIT ?",
            (name, limit)
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

    def get_identities_assigned_to_photo(self, checksum):
        """Reverse of get_photos_assigned_to_identity: every name manually,
        faceless-ly assigned to this one photo — what photo.html's "Assigned
        without a face" list renders. Ordered by assignment time so the most
        recently confirmed name shows last, matching face-chip ordering
        elsewhere."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT identity FROM identity_photo_assignments WHERE checksum = ? ORDER BY created_at',
            (checksum,)
        )
        return [row[0] for row in cur.fetchall()]

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
        later is automatically "theirs" too, with nothing further to write here.
        Also clears any prior exclusion for this (set, identity) pair — explicitly
        re-adding someone should undo a previous removal, not silently no-op
        against it (see exclude_identity_from_set)."""
        cur = self.conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO identity_set_assignments (set_id, identity, created_at)
            VALUES (?, ?, ?)
        ''', (set_id, identity, int(time.time())))
        cur.execute(
            'DELETE FROM identity_set_exclusions WHERE set_id = ? AND identity = ?',
            (set_id, identity)
        )
        self.conn.commit()

    def unlink_identity_from_set(self, set_id, identity):
        cur = self.conn.cursor()
        cur.execute(
            'DELETE FROM identity_set_assignments WHERE set_id = ? AND identity = ?',
            (set_id, identity)
        )
        self.conn.commit()

    def exclude_identity_from_set(self, set_id, identity):
        """'This person does NOT belong on this set's roster' — see
        identity_set_exclusions' schema comment. Also removes any explicit link
        row, so this is the one call the "×" on a set's People-present entry
        needs regardless of which of the three sources (link, whole-photo
        assignment, or detected face) put that person there."""
        cur = self.conn.cursor()
        cur.execute(
            'DELETE FROM identity_set_assignments WHERE set_id = ? AND identity = ?',
            (set_id, identity)
        )
        cur.execute('''
            INSERT OR IGNORE INTO identity_set_exclusions (set_id, identity, created_at)
            VALUES (?, ?, ?)
        ''', (set_id, identity, int(time.time())))
        self.conn.commit()

    def get_sets_linked_to_identity(self, identity):
        """Every set this identity has been linked to (see link_identity_to_set),
        as full set rows — callers resolve "files for this person" by expanding
        each one's current membership, and mark these as already-linked in the
        set picker UI."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT s.* FROM sets_with_studio s
            JOIN identity_set_assignments isa ON isa.set_id = s.id
            WHERE isa.identity = ?
            ORDER BY s.name
        ''', (identity,))
        return cur.fetchall()

    def get_identities_for_sets(self, set_ids):
        """Batched reverse of get_sets_linked_to_identity: {set_id: [identity, ...]}
        (sorted, case-insensitively) for every set id given — avoids N+1 queries
        when a page renders many sets at once (mirrors get_sets_for_checksums)."""
        if not set_ids:
            return {}
        placeholders = ','.join('?' for _ in set_ids)
        cur = self.conn.cursor()
        cur.execute(
            f'''SELECT set_id, identity FROM identity_set_assignments
                WHERE set_id IN ({placeholders})''',
            tuple(set_ids)
        )
        result = {}
        for set_id, identity in cur.fetchall():
            result.setdefault(set_id, []).append(identity)
        for identities in result.values():
            identities.sort(key=str.lower)
        return result

    def get_all_people_for_sets(self, set_ids):
        """Batched {set_id: [identity, ...]} (sorted, case-insensitively) for a
        set's FULL roster — every identity linked via identity_set_assignments,
        PLUS anyone with a real named face or a whole-photo assignment on one of
        that set's own member photos (file_sets), minus anything excluded via
        identity_set_exclusions. This is the "sets always have @ with all the
        names, even when not manually set" list: set_meta_line/chip-set should
        show everyone actually present in the set, not just the subset someone
        happened to explicitly link. Mirrors get_people_present_in_set's three
        sources, batched across many sets at once instead of one set + its
        already-resolved member checksums (avoids the N+1 of calling that per
        set on pages rendering many sets, e.g. /sets or a card grid)."""
        if not set_ids:
            return {}
        placeholders = ','.join('?' for _ in set_ids)
        cur = self.conn.cursor()
        cur.execute(f'''
            SELECT fs.set_id, f.identity FROM faces f
            JOIN file_sets fs ON fs.checksum = f.checksum
            WHERE fs.set_id IN ({placeholders}) AND f.identity IS NOT NULL AND f.rejected = 0
            UNION
            SELECT fs.set_id, ipa.identity FROM identity_photo_assignments ipa
            JOIN file_sets fs ON fs.checksum = ipa.checksum
            WHERE fs.set_id IN ({placeholders})
            UNION
            SELECT isa.set_id, isa.identity FROM identity_set_assignments isa
            WHERE isa.set_id IN ({placeholders})
        ''', tuple(set_ids) * 3)
        result = {}
        for set_id, identity in cur.fetchall():
            result.setdefault(set_id, []).append(identity)

        cur.execute(f'''
            SELECT set_id, identity FROM identity_set_exclusions WHERE set_id IN ({placeholders})
        ''', tuple(set_ids))
        excluded = {}
        for set_id, identity in cur.fetchall():
            excluded.setdefault(set_id, set()).add(identity)

        for set_id, identities in result.items():
            names = [name for name in identities if name not in excluded.get(set_id, ())]
            result[set_id] = sorted(set(names), key=str.lower)
        return result

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
        in the library, and get None only if no such face exists at all.

        Each value is actually a (face_id, set_linked) pair — set_linked is True
        only for the identity_set_assignments source (an explicit "person X is in
        this set" link made via link_identity_to_set/the assign-set endpoint).
        Callers offering a single "remove" action for any entry regardless of
        source should call exclude_identity_from_set, not unlink_identity_from_set
        directly — the latter only has a row to delete for the linked source, so
        an excluded identity (identity_set_exclusions) is always filtered out
        below regardless of which of the three sources would otherwise surface
        it, and stays excluded until explicitly re-added via link_identity_to_set
        (which clears the exclusion)."""
        cur = self.conn.cursor()
        cur.execute('SELECT identity FROM identity_set_exclusions WHERE set_id = ?', (set_id,))
        excluded_names = {row[0] for row in cur.fetchall()}
        face_people = {}
        if member_checksums:
            placeholders = ','.join('?' for _ in member_checksums)
            cur.execute(f'''
                SELECT identity, MIN(id) FROM faces
                WHERE checksum IN ({placeholders}) AND identity IS NOT NULL AND rejected = 0
                GROUP BY identity
            ''', tuple(member_checksums))
            for identity, face_id in cur.fetchall():
                if identity not in excluded_names:
                    face_people[identity] = face_id

        photo_assigned_names = set()
        if member_checksums:
            placeholders = ','.join('?' for _ in member_checksums)
            cur.execute(f'''
                SELECT DISTINCT identity FROM identity_photo_assignments
                WHERE checksum IN ({placeholders})
            ''', tuple(member_checksums))
            photo_assigned_names.update(row[0] for row in cur.fetchall() if row[0] not in excluded_names)

        cur.execute('SELECT identity FROM identity_set_assignments WHERE set_id = ?', (set_id,))
        set_linked_names = {row[0] for row in cur.fetchall() if row[0] not in excluded_names}

        assigned_names = photo_assigned_names | set_linked_names

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

        return {identity: (face_id, identity in set_linked_names) for identity, face_id in people.items()}

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
