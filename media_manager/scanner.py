"""
FileScanner class for discovering files without hashing.
"""
import os

class FileScanner:
    def __init__(self, database):
        self.db = database

    def scan_directory(self, root_path, recursive=True):
        """
        Recursively discover all files under root_path and store their metadata.
        Checksum and last_hashed are left as NULL.
        """
        if not recursive:
            # Only scan the immediate directory
            try:
                with os.scandir(root_path) as entries:
                    for entry in entries:
                        if entry.is_file():
                            self._process_file(entry.path)
            except (OSError, PermissionError):
                # Skip directories that cannot be accessed
                pass
            return True
        
        # Recursive scanning
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                try:
                    self._process_file(full_path)
                except (OSError, PermissionError):
                    # Skip files that cannot be accessed
                    continue
        return True

    def _process_file(self, full_path):
        """
        Process a single file and store its metadata.
        """
        try:
            stat = os.stat(full_path)
            # Normalize the path before storing
            normalized_path = self._normalize_path(full_path)
            self.db.insert_or_update_file(
                path=normalized_path,
                size=stat.st_size,
                modified_time=stat.st_mtime,
                checksum=None,
                last_hashed=None
            )
        except OSError:
            # Skip files that cannot be accessed
            pass

    def _normalize_path(self, full_path):
        """
        Convert absolute path to internal format starting from data/
        """
        # Convert to relative path from ../data
        root_data = os.path.abspath(os.path.join('..', 'data'))
        abs_path = os.path.abspath(full_path)
        
        if abs_path.startswith(root_data):
            rel_path = os.path.relpath(abs_path, root_data)
            return os.path.join('data', rel_path)
        else:
            # If path is not under ../data, try to find data/ in the path
            parts = full_path.split(os.sep)
            try:
                data_idx = parts.index('data')
                return os.path.join(*parts[data_idx:])
            except (ValueError, IndexError):
                # Fallback: just prepend 'data/'
                return os.path.join('data', os.path.basename(full_path))
