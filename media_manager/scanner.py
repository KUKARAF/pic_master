"""
FileScanner class for discovering files without hashing.
"""
import os

class FileScanner:
    def __init__(self, database, data_root):
        self.db = database
        self.data_root = data_root

    def scan_directory(self, root_path, recursive=True):
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                self._process_file(full_path)
            if not recursive:
                break
        return True

    def _process_file(self, full_path):
        """
        Process a single file and store its metadata.
        Stores path relative to media_root
        """
        try:
            stat = os.stat(full_path)
            # Convert absolute path to relative path from media_root
            rel_path = os.path.relpath(full_path, self.data_root)
            self.db.insert_or_update_file(
                path=rel_path,
                size=stat.st_size,
                modified_time=stat.st_mtime,
                checksum=None,
                last_hashed=None
            )
        except OSError:
            # Skip files that cannot be accessed
            pass
