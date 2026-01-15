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

    def normalize_path(self, input_path):
        """
        Convert input path to internal format:
        - If input path starts with 'data/', keep it as is
        - Otherwise, find the 'data/' part in the path and extract from there
        - If no 'data/' found, prepend 'data/'
        """
        input_path = os.path.normpath(input_path)
        
        # If it's already in data/ format, return as is
        if input_path.startswith('data' + os.sep):
            return input_path
            
        # Find data directory in path
        parts = input_path.split(os.sep)
        try:
            data_idx = parts.index('data')
            return os.path.join(*parts[data_idx:])
        except ValueError:
            # If no 'data' found, prepend 'data'
            return os.path.join('data', input_path)

    def start_scan(self, path, recursive=True):
        """
        Scan a directory and store file metadata (without hashes).
        Converts relative paths to absolute paths starting from ../data
        """
        # Convert relative path to absolute path starting from ../data
        if not os.path.isabs(path):
            abs_path = os.path.join('..', 'data', path)
        else:
            abs_path = path
            
        return self.scanner.scan_directory(abs_path, recursive)

    def hash_files(self, batch_size=100):
        """
        Process files that haven't been hashed yet.
        Returns number of files processed.
        """
        return self.hasher.process_null_hashes(batch_size)

    def get_file_info(self, path):
        """
        Retrieve file metadata and hash from the database.
        Path should be in internal format (data/...)
        """
        normalized_path = self.normalize_path(path)
        return self.db.get_file_by_path(normalized_path)

    def verify_integrity(self, path):
        """
        Verify the integrity of a file by comparing stored and current hash.
        Path should be in internal format (data/...)
        """
        file_info = self.get_file_info(path)
        if not file_info or file_info['checksum'] is None:
            return False
            
        # Convert normalized path back to absolute path for verification
        abs_path = os.path.join('..', 'data', path.replace('data/', '', 1))
        return self.hasher.verify_file_integrity(file_info['id'], abs_path)

    def close(self):
        """Close the database connection."""
        self.db.close()
