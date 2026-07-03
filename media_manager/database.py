"""
Database schema and operations for media management.
"""
import sqlite3
import time

class Database:
    def __init__(self, db_path="media.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()

    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        # Files table: stores file metadata and hash information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                size INTEGER,
                modified_time REAL,
                checksum TEXT,
                last_hashed INTEGER,
                broken INTEGER
            )
        ''')
        # Migrate existing DBs that predate the broken column
        existing = {row[1] for row in cursor.execute('PRAGMA table_info(files)')}
        if 'broken' not in existing:
            cursor.execute('ALTER TABLE files ADD COLUMN broken INTEGER')
        # Embeddings table: stores CLIP embeddings for image search.
        # CREATE TABLE IF NOT EXISTS handles both fresh DBs and old DBs (migration).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                indexed_at INTEGER NOT NULL
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                model TEXT NOT NULL,
                indexed_at INTEGER NOT NULL
            )
        ''')
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_path ON files (path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON files (checksum)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_hashed ON files (last_hashed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags (tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_class ON detections (class_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_file ON detections (file_id)')
        self.conn.commit()

    def insert_or_update_file(self, path, size=None, modified_time=None,
                              checksum=None, last_hashed=None):
        """Insert a new file record or update an existing one."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO files (path, size, modified_time, checksum, last_hashed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                modified_time=excluded.modified_time,
                checksum=excluded.checksum,
                last_hashed=excluded.last_hashed
        ''', (path, size, modified_time, checksum, last_hashed))
        self.conn.commit()
        return cursor.lastrowid

    def get_file_by_path(self, path):
        """Retrieve a file record by its path."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM files WHERE path = ?', (path,))
        row = cursor.fetchone()
        if row:
            return row
        return None

    def get_files_without_hash(self, limit=100):
        """Return a list of files that have NULL checksum."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files
                WHERE checksum IS NULL
            ''')
        else:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files
                WHERE checksum IS NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def get_files_with_hash(self, limit=100):
        """Return files that have been hashed."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files
                WHERE checksum IS NOT NULL
            ''')
        else:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files
                WHERE checksum IS NOT NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def list_files(self, limit=100, hashed_only=False, unhashed_only=False):
        """
        List files with optional filtering.
        Returns tuples of file records: (id, path, size, modified_time, checksum, last_hashed)
        """
        cursor = self.conn.cursor()
        if hashed_only:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files 
                WHERE checksum IS NOT NULL 
                LIMIT ?
            ''', (limit,))
        elif unhashed_only:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files 
                WHERE checksum IS NULL 
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT id, path, size, modified_time, checksum, last_hashed
                FROM files 
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def count_files(self, hashed_only=False, unhashed_only=False, limit=None):
        """
        Return the number of rows matching the filter.
        limit:  maximum rows to count (None == unlimited)
        """
        cur = self.conn.cursor()
        if hashed_only:
            sql = 'SELECT COUNT(*) FROM files WHERE checksum IS NOT NULL'
            if limit is not None:
                sql += ' LIMIT ?'
                cur.execute(sql, (limit,))
            else:
                cur.execute(sql)
        elif unhashed_only:
            sql = 'SELECT COUNT(*) FROM files WHERE checksum IS NULL'
            if limit is not None:
                sql += ' LIMIT ?'
                cur.execute(sql, (limit,))
            else:
                cur.execute(sql)
        else:
            sql = 'SELECT COUNT(*) FROM files'
            if limit is not None:
                sql += ' LIMIT ?'
                cur.execute(sql, (limit,))
            else:
                cur.execute(sql)
        row = cur.fetchone()
        return row[0] if row else 0

    def update_file_hash(self, file_id, checksum):
        """Update the checksum and last_hashed timestamp for a file."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE files
            SET checksum = ?, last_hashed = ?
            WHERE id = ?
        ''', (checksum, int(time.time()), file_id))
        self.conn.commit()

    def count_broken_files(self):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM files WHERE broken IS NOT NULL')
        return cur.fetchone()[0]

    def list_broken_files(self, limit=100):
        cur = self.conn.cursor()
        cur.execute('SELECT path, broken FROM files WHERE broken IS NOT NULL LIMIT ?', (limit,))
        return cur.fetchall()

    def clear_broken(self, paths):
        cur = self.conn.cursor()
        cur.executemany('UPDATE files SET broken = NULL WHERE path = ?', [(p,) for p in paths])
        self.conn.commit()
        return cur.rowcount

    def insert_embedding(self, file_id, embedding_bytes, model):
        """Upsert an embedding for a file."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (file_id, embedding, model, indexed_at)
            VALUES (?, ?, ?, ?)
        ''', (file_id, embedding_bytes, model, int(time.time())))
        self.conn.commit()

    def get_all_embeddings(self):
        """Return list of (file_id, path, embedding_bytes) joining with files."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT e.file_id, f.path, e.embedding
            FROM embeddings e
            JOIN files f ON f.id = e.file_id
        ''')
        return cursor.fetchall()

    def get_unindexed_files(self, limit=None):
        """Return (id, path) for files that have no row in embeddings."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files f
                LEFT JOIN embeddings e ON e.file_id = f.id
                WHERE e.file_id IS NULL
            ''')
        else:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files f
                LEFT JOIN embeddings e ON e.file_id = f.id
                WHERE e.file_id IS NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def count_indexed(self):
        """Return the count of rows in the embeddings table."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
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
            FROM files f
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
        """Retrieve a file record by its id."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM files WHERE id = ?', (file_id,))
        return cursor.fetchone()

    def list_files_with_embedding_flag(self, limit=200, offset=0):
        """Return (id, path, has_embedding) rows for gallery browsing."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.id, f.path,
                   CASE WHEN e.file_id IS NOT NULL THEN 1 ELSE 0 END AS has_embedding
            FROM files f
            LEFT JOIN embeddings e ON e.file_id = f.id
            ORDER BY f.id DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        return cursor.fetchall()

    def get_embedding(self, file_id):
        """Return the embedding bytes for a file, or None."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM embeddings WHERE file_id = ?', (file_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Detection methods (YOLO-World)
    # ------------------------------------------------------------------

    def insert_detections(self, file_id, detections, model):
        """Upsert detections for a file. detections is a list of (class_name, confidence, x1, y1, x2, y2).
        Always writes at least a sentinel row ('__indexed__') so the file is never re-queued."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE file_id = ?', (file_id,))
        now = int(time.time())
        rows = [(file_id, cls, conf, x1, y1, x2, y2, model, now)
                for cls, conf, x1, y1, x2, y2 in detections]
        if not rows:
            # sentinel: marks file as processed even when nothing was detected
            rows = [(file_id, '__indexed__', 0.0, None, None, None, None, model, now)]
        cursor.executemany(
            '''INSERT INTO detections (file_id, class_name, confidence, x1, y1, x2, y2, model, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            rows
        )
        self.conn.commit()

    def get_detected_classes(self, file_id):
        """Return list of distinct detected class names for a file, ordered by confidence descending."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT class_name FROM detections WHERE file_id = ? AND class_name != '__indexed__' ORDER BY confidence DESC",
            (file_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_undetected_files(self, limit=None):
        """Return (id, path) for files that have no row in detections."""
        cursor = self.conn.cursor()
        if limit is None:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files f
                LEFT JOIN detections d ON d.file_id = f.id
                WHERE d.file_id IS NULL
            ''')
        else:
            cursor.execute('''
                SELECT f.id, f.path
                FROM files f
                LEFT JOIN detections d ON d.file_id = f.id
                WHERE d.file_id IS NULL
                LIMIT ?
            ''', (limit,))
        return cursor.fetchall()

    def search_by_classes(self, class_names, limit=20):
        """
        Return (file_id, path, score) rows where score = SUM(confidence) for matched classes.
        Uses LIKE substring matching so "couch" matches "couch" and "sofa couch" etc.
        Excludes sentinel rows (class_name = '__indexed__').
        """
        if not class_names:
            return []
        # Build: (class_name LIKE %tok1% OR class_name LIKE %tok2% OR ...)
        like_clauses = ' OR '.join('d.class_name LIKE ?' for _ in class_names)
        like_params = [f'%{t}%' for t in class_names]
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT f.id, f.path, SUM(d.confidence) as score
            FROM detections d
            JOIN files f ON f.id = d.file_id
            WHERE ({like_clauses})
              AND d.class_name != '__indexed__'
            GROUP BY d.file_id
            ORDER BY score DESC
            LIMIT ?
        ''', (*like_params, limit))
        return cursor.fetchall()

    def count_detected(self):
        """Return count of distinct files with detections."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(DISTINCT file_id) FROM detections')
        row = cursor.fetchone()
        return row[0] if row else 0

    def close(self):
        """Close the database connection."""
        self.conn.close()
