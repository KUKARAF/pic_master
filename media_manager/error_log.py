"""Separate error/warning log — deliberately its own sqlite file (.media/error.db),
independent of media.db, so it can be queried/reset without touching image metadata."""
import time

from media_manager.database import ThreadLocalDB


class ErrorLog(ThreadLocalDB):
    def __init__(self, db_path):
        super().__init__(db_path)
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                message TEXT NOT NULL,
                read INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_errors_read ON errors (read)')
        self.conn.commit()

    def log(self, path, message):
        cur = self.conn.cursor()
        cur.execute('INSERT INTO errors (path, message, read, created_at) VALUES (?,?,0,?)',
                    (path, message, int(time.time())))
        self.conn.commit()
        return cur.lastrowid

    def list_errors(self, unread_only=False, limit=50):
        cur = self.conn.cursor()
        sql = 'SELECT id, path, message, read, created_at FROM errors'
        if unread_only:
            sql += ' WHERE read = 0'
        sql += ' ORDER BY id DESC LIMIT ?'
        cur.execute(sql, (limit,))
        return cur.fetchall()

    def count_unread(self):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM errors WHERE read = 0')
        return cur.fetchone()[0]

    def mark_read(self, error_id):
        cur = self.conn.cursor()
        cur.execute('UPDATE errors SET read = 1 WHERE id = ?', (error_id,))
        self.conn.commit()

    def mark_all_read(self):
        cur = self.conn.cursor()
        cur.execute('UPDATE errors SET read = 1 WHERE read = 0')
        self.conn.commit()

