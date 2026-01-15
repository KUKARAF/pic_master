"""
FileScanner class for discovering files without hashing.
"""
import os
import time

class FileScanner:
    def __init__(self, database):
        self.db = database

    def scan_directory(self, root_path):
        """
        Recursively discover all files under root_path and store their metadata.
        Checksum and last_hashed are left as NULL.
        """
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                try:
                    stat = os.stat(full_path)
                    self.db.insert_or_update_file(
                        path=full_path,
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                        checksum=None,
                        last_hashed=None
                    )
                except (OSError, PermissionError):
                    # Skip files that cannot be accessed
                    continue
        return True
