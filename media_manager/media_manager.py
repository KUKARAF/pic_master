"""
Main MediaManager class providing the primary interface for media management operations.
"""
import os
from .database import Database
from .scanner import FileScanner
from .hasher import FileHasher

class MediaManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Find the .data directory by walking up the directory tree
            self.data_root = self._find_data_root()
            db_path = os.path.join(self.data_root, '.data', 'media.db')
        else:
            # If db_path is specified, derive data_root from it
            self.data_root = os.path.dirname(os.path.dirname(db_path))
            
        self.db = Database(db_path)
        self.scanner = FileScanner(self.db, self.data_root)
        self.hasher = FileHasher(self.db)

    def _find_data_root(self):
        """Find the .data directory by walking up the directory tree."""
        current = os.getcwd()
        while current != os.path.dirname(current):  # Stop at root
            data_dir = os.path.join(current, '.data')
            if os.path.isdir(data_dir):
                return current
            current = os.path.dirname(current)
        
        # If no .data found, create it in current directory
        data_dir = os.path.join(os.getcwd(), '.data')
        os.makedirs(data_dir, exist_ok=True)
        return os.getcwd()

    def start_scan(self, path, recursive=True):
        """
        Scan a directory and store file metadata (without hashes).
        Path is relative to data_root
        """
        abs_path = os.path.join(self.data_root, path)
        return self.scanner.scan_directory(abs_path, recursive)

    def hash_files(self, batch_size=100):
        return self.hasher.process_null_hashes(batch_size)

    def get_file_info(self, path):
        return self.db.get_file_by_path(path)

    def get_unhashed_files(self, limit=100):
        """
        Get database entries for files without a hash.
        """
        return self.db.get_files_without_hash(limit)

    def close(self):
        self.db.close()
