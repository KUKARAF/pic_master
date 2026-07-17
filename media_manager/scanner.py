"""
FileScanner class for discovering files. Superseded by fast_scan.py (used by
MediaManager.start_scan) for real scans — kept only for callers that want a plain
os.walk-based scanner instead of the GNU find backend.
"""
import os
from .hasher import FileHasher

class FileScanner:
    def __init__(self, database, data_root):
        self.db = database
        self.data_root = data_root
        self.hasher = FileHasher(database)

    def scan_directory(self, root_path, recursive=True):
        for dirpath, dirnames, filenames in os.walk(root_path):
            # .media/ holds our own cache/db files — never walk into it.
            dirnames[:] = [d for d in dirnames if d != '.media']
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                self._process_file(full_path)
            if not recursive:
                break
        return True

    def _process_file(self, full_path):
        """
        Hash a single file and upsert it (content is the identity — see database.py).
        Stores path relative to media_root.
        """
        try:
            stat = os.stat(full_path)
            checksum = self.hasher.get_xxhash(full_path)
            if checksum is None:
                return
            rel_path = os.path.relpath(full_path, self.data_root)
            self.db.upsert_file_path(
                rel_path, checksum,
                size=stat.st_size,
                modified_time=stat.st_mtime,
            )
            self.db.conn.commit()
        except OSError:
            # Skip files that cannot be accessed
            pass
