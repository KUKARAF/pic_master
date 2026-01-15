"""
Main MediaManager class providing the primary interface for media management operations.
"""
import os
from .database import Database
from .scanner import FileScanner
from .hasher import FileHasher

class MediaManager:
    def __init__(self, db_path="media.db"):
        self.db = Database(db_path)
        self.scanner = FileScanner(self.db)
        self.hasher = FileHasher(self.db)

    def start_scan(self, path, recursive=True):
        """
        Scan a directory and store file metadata (without hashes).
        Path is always relative to ../data/
        """
        abs_path = os.path.join('..', 'data', path)
        return self.scanner.scan_directory(abs_path, recursive)

    def hash_files(self, batch_size=100):
        return self.hasher.process_null_hashes(batch_size)

    def get_file_info(self, path):
        return self.db.get_file_by_path(path)

    def close(self):
        self.db.close()
