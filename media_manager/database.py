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
                last_hashed INTEGER  -- Unix timestamp when hash was last computed
            )
        ''')
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_path ON files (path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON files (checksum)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_hashed ON files (last_hashed)')
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

    def close(self):
        """Close the database connection."""
        self.conn.close()
