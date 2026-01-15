"""
FileScanner class for discovering files without hashing.
"""
import os

class FileScanner:
    def __init__(self, database):
        self.db = database

    def scan_directory(self, root_path, recursive=True):
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                self._process_file(full_path, dirpath, filename)
            if not recursive:
                break
        return True

    def _process_file(self, root, dirpath, filename):
        rel_dir = os.path.relpath(dirpath, os.path.join('..', 'data'))
        internal_path = os.path.join('data', rel_dir, filename)
        stat = os.stat(os.path.join(dirpath, filename))
        self.db.insert_or_update_file(
            path=internal_path,
            size=stat.st_size,
            modified_time=stat.st_mtime,
            checksum=None,
            last_hashed=None
        )
