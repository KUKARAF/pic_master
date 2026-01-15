"""
Database schema and operations for media management.
"""
import sqlite3
import time

class Database:
    def __init__(self, db_path="media.db"):
        self.conn = sqlite3.connect(db_path)
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
            return {
                'id': row[0],
                'path': row[1],
                'size': row[2],
                'modified_time': row[3],
                'checksum': row[4],
                'last_hashed': row[5]
            }
        return None

    def get_files_without_hash(self, limit=100):
        """Return a list of files that have NULL checksum."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, path, size, modified_time
            FROM files
            WHERE checksum IS NULL
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

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
