"""
Data models for media management system.
"""
class File:
    """Represents a file record in the database."""
    def __init__(self, id=None, path=None, size=None, modified_time=None,
                 checksum=None, last_hashed=None):
        self.id = id
        self.path = path
        self.size = size
        self.modified_time = modified_time
        self.checksum = checksum
        self.last_hashed = last_hashed  # Unix timestamp of last hash generation
