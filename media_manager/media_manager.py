"""
Main MediaManager class providing the primary interface for media management operations.
"""
from .database import Database
from .scanner import FileScanner
from .hasher import FileHasher

class MediaManager:
    def __init__(self, db_path="media.db"):
        self.db = Database(db_path)
        self.scanner = FileScanner(self.db)
        self.hasher = FileHasher(self.db)

    def start_scan(self, path):
        """
        Scan a directory and store file metadata (without hashes).
        """
        return self.scanner.scan_directory(path)

    def hash_files(self, batch_size=100):
        """
        Process files that haven't been hashed yet.
        Returns number of files processed.
        """
        return self.hasher.process_null_hashes(batch_size)

    def get_file_info(self, path):
        """
        Retrieve file metadata and hash from the database.
        """
        return self.db.get_file_by_path(path)

    def verify_integrity(self, path):
        """
        Verify the integrity of a file by comparing stored and current hash.
        """
        file_info = self.db.get_file_by_path(path)
        if not file_info or file_info['checksum'] is None:
            return False
        return self.hasher.verify_file_integrity(file_info['id'], path)

    def close(self):
        """Close the database connection."""
        self.db.close()
