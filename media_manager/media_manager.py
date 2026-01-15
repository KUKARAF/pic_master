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
            # Find the .media directory by walking up the directory tree
            self.data_root = self._find_media_root()
            db_path = os.path.join(self.data_root, '.media', 'media.db')
        else:
            # If db_path is specified, derive data_root from it
            self.data_root = os.path.dirname(os.path.dirname(db_path))
            
        self.db = Database(db_path)
        self.scanner = FileScanner(self.db, self.data_root)
        self.hasher = FileHasher(self.db)

    def _find_media_root(self):
        """Find the .media directory by walking up the directory tree."""
        current = os.getcwd()
        while current != os.path.dirname(current):  # Stop at root
            media_dir = os.path.join(current, '.media')
            if os.path.isdir(media_dir):
                return current
            current = os.path.dirname(current)
        
        # If no .media found, create it in current directory
        media_dir = os.path.join(os.getcwd(), '.media')
        os.makedirs(media_dir, exist_ok=True)
        return os.getcwd()

    def start_scan(self, path, recursive=True):
        """
        Scan a directory and store file metadata (without hashes).
        Path is relative to media_root
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

    def get_hashed_files(self, limit=100):
        """
        Get database entries for files with hashes.
        """
        return self.db.get_files_with_hash(limit)

    def list_files(self, limit=100, hashed_only=False, unhashed_only=False):
        """
        List all files with optional filtering.
        """
        return self.db.list_files(limit=limit, hashed_only=hashed_only, unhashed_only=unhashed_only)

    def get_hash_for_file(self, path):
        """Return the stored hash for a single file (or None)."""
        info = self.get_file_info(path)
        return info['checksum'] if info else None

    def get_hash_list(self, path, recursive=False):
        """
        Return a list of (relative_path, checksum) tuples.
        If path is a file: single item if it has a hash.
        If path is a dir: all *hashed* files under it (respects recursive flag).
        """
        abs_path = os.path.join(self.data_root, path)
        out = []
        if os.path.isfile(abs_path):
            digest = self.get_hash_for_file(os.path.relpath(abs_path, self.data_root))
            if digest:
                out.append((os.path.relpath(abs_path, self.data_root), digest))
            return out

        # directory
        for root, _, files in os.walk(abs_path):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.data_root)
                digest = self.get_hash_for_file(rel)
                if digest:
                    out.append((rel, digest))
            if not recursive:
                break
        return out

    def close(self):
        self.db.close()
