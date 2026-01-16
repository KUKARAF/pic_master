"""
FileHasher class for generating hashes on demand.
"""
import xxhash
import os

class FileHasher:
    def __init__(self, database):
        self.db = database

    def get_xxhash(self, file_path, buffer_size=65536):
        """Generate xxhash for a single file."""
        hasher = xxhash.xxh64()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    hasher.update(data)
            return hasher.hexdigest()
        except (OSError, IOError):
            return None

    def update_file_hash(self, file_id, file_path):
        """Compute hash for file and update the database record."""
        checksum = self.get_xxhash(file_path)
        if checksum is not None:
            self.db.update_file_hash(file_id, checksum)
            return True
        return False

    def process_null_hashes(self, batch_size=100):
        """
        Process files that have NULL checksum in the database.
        Returns the number of files successfully hashed.
        """
        files = self.db.get_files_without_hash(limit=batch_size)
        processed = 0
        for row in files:
            file_id = row['id']
            path = row['path']
            abs_path = os.path.join(self.db.data_root, path)  # <-- use repo root
            try:
                if self.update_file_hash(file_id, abs_path):
                    processed += 1
            except (OSError, IOError):
                # log and skip
                continue
        return processed

    def verify_file_integrity(self, file_id, file_path):
        """
        Compare stored hash with freshly computed hash.
        Returns True if they match, False otherwise.
        """
        file_info = self.db.get_file_by_path(file_path)
        if not file_info or file_info['checksum'] is None:
            return False
        abs_path = os.path.join(self.db.data_root, file_path)  # <-- repo root
        current_hash = self.get_xxhash(abs_path)
        return current_hash == file_info['checksum']
