"""
Main MediaManager class providing the primary interface for media management operations.
"""
import os
import time
import sys
from .database import Database
from .hasher import FileHasher
from .fast_scan import fast_scan
from .hash_estimator import HashEstimator
from .broken_finder import find_broken

_STOPWORDS = {'a', 'an', 'the', 'at', 'in', 'on', 'of', 'with', 'and', 'or', 'to', 'is', 'are', 'for', 'by', 'from'}

def _parse_query(query):
    tokens = query.lower().split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


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
        Path is relative to media_root – uses fast GNU find backend
        """
        abs_path = os.path.join(self.data_root, path)
        return fast_scan(abs_path, self.db.conn, self.data_root, recursive)

    def hash_files(self, batch_size=100):
        # wrap the low-level hasher call with live estimator
        return self._hash_with_progress(batch_size)

    def _hash_with_progress(self, batch_size):
        """
        Hash unhashed files while printing live progress.
        Commits every 5000 hashes by default.
        """
        unhashed = self.db.get_files_without_hash(limit=None)  # fetch all
        total = len(unhashed)

        # nothing to do?
        if total == 0:
            print("All files are already hashed; nothing to commit.")
            return 0

        COMMIT_EVERY = 5000
        processed = 0
        cursor = self.db.conn.cursor()

        # show progress header/footer even when total == 0
        with HashEstimator(total=total) as est:
            for idx, row in enumerate(unhashed, 1):
                file_id, path, *_ = row
                abs_path = os.path.join(self.data_root, path)
                # update hash (only in cursor, not committed yet)
                cursor.execute('''
                    UPDATE files
                    SET checksum = ?, last_hashed = ?
                    WHERE id = ?
                ''', (self.hasher.get_xxhash(abs_path), int(time.time()), file_id))
                processed += 1
                est.update(done=1)

                # bulk commit every COMMIT_EVERY rows
                if idx % COMMIT_EVERY == 0:
                    self.db.conn.commit()

        # final commit for remaining rows
        self.db.conn.commit()
        return processed

    def get_file_info(self, path):
        return self.db.get_file_by_path(path)

    def get_unhashed_files(self, limit=100):
        """Get database entries for files without a hash."""
        return self.db.get_files_without_hash(limit)

    def get_hashed_files(self, limit=100):
        """Get database entries for files with hashes."""
        return self.db.get_files_with_hash(limit)

    def list_files(self, limit=100, hashed_only=False, unhashed_only=False):
        """List all files with optional filtering."""
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

    def record_move(self, src, dst):
        """Update the DB so the file historically at <src> is now at <dst>."""
        info = self.db.get_file_by_path(src)
        if not info:
            return False
        # insert new row
        self.db.insert_or_update_file(
            path=dst,
            size=info['size'],
            modified_time=info['modified_time'],
            checksum=info['checksum'],
            last_hashed=info['last_hashed']
        )
        # remove old row
        cursor = self.db.conn.cursor()
        cursor.execute('DELETE FROM files WHERE path = ?', (src,))
        self.db.conn.commit()
        return True

    def find_moved_candidates(self, limit=20):
        """
        Return [(old_path, new_path, checksum), ...] for files that
        - exist in the DB but not at their recorded path
        - have the same size+checksum as another file that *does* exist somewhere else
        Limit caps the result set.
        """
        moved = []
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT path, size, checksum FROM files WHERE checksum IS NOT NULL')
        for row in cursor.fetchall():
            path, size, chksum = row
            if not os.path.exists(os.path.join(self.data_root, path)):
                cursor.execute('''
                    SELECT path FROM files
                    WHERE size=? AND checksum=? AND path!=?
                    LIMIT 1
                ''', (size, chksum, path))
                twin = cursor.fetchone()
                if twin:
                    moved.append((path, twin[0], chksum))
                    if limit is not None and len(moved) >= limit:
                        break
        return moved

    def update_moved_files(self):
        """
        Update the database to reflect moved files.
        For each moved file (old_path -> new_path), remove the old entry and keep the new one.
        Returns the number of moved files updated.
        """
        moved = self.find_moved_candidates(limit=None)  # Get all moved files
        count = 0
        cursor = self.db.conn.cursor()
        
        for old_path, new_path, _ in moved:
            # Remove the old entry (the one that doesn't exist anymore)
            cursor.execute('DELETE FROM files WHERE path = ?', (old_path,))
            count += 1
        
        self.db.conn.commit()
        return count

    def count_files(self, hashed_only=False, unhashed_only=False, limit=None):
        """
        Return the number of rows matching the filter.
        limit:  maximum rows to count (None == unlimited)
        """
        return self.db.count_files(hashed_only=hashed_only,
                                  unhashed_only=unhashed_only,
                                  limit=limit)

    def find_broken(self, path, max_workers=8):
        """
        Check images/videos under path for corruption.
        Sets 'broken' column to current unix timestamp for broken files.
        Returns number of newly-detected broken files.
        """
        abs_path = os.path.join(self.data_root, path)
        return find_broken(abs_path, self.db.conn, self.data_root, max_workers)

    def count_broken_files(self):
        """Return count of files marked as broken (broken NOT NULL)."""
        return self.db.count_broken_files()

    def list_broken_files(self, limit=100):
        """Return [(path, broken_timestamp), ...] for broken files."""
        return self.db.list_broken_files(limit=limit)

    def clear_broken(self, paths):
        """Clear broken flag (set to NULL) for given list of paths."""
        return self.db.clear_broken(paths)

    def index_files(self, path='.', batch_size=32, model_size='s', conf_threshold=0.15):
        """
        Run YOLO-World on undetected image files under path, store detections in DB.
        Returns (indexed_count, failed_count).
        """
        from .detector import YOLOWorldDetector, SUPPORTED_EXTENSIONS

        detector = YOLOWorldDetector(model_size=model_size, conf_threshold=conf_threshold)
        model_id = YOLOWorldDetector.model_id(model_size)

        undetected = self.db.get_undetected_files(limit=None)
        abs_path = os.path.abspath(os.path.join(self.data_root, path))
        candidates = []
        for file_id, rel_path in undetected:
            ext = os.path.splitext(rel_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            abs_file = os.path.join(self.data_root, rel_path)
            if not abs_file.startswith(abs_path):
                continue
            candidates.append((file_id, abs_file))

        total = len(candidates)
        indexed_count = 0
        failed_count = 0

        for batch_start in range(0, total, batch_size):
            batch = candidates[batch_start:batch_start + batch_size]
            batch_ids = [fid for fid, _ in batch]
            batch_paths = [fpath for _, fpath in batch]

            results = detector.detect_images(batch_paths)
            for file_id, (path_, detections, error) in zip(batch_ids, results):
                if error is not None:
                    failed_count += 1
                else:
                    self.db.insert_detections(file_id, detections, model_id)
                    indexed_count += 1

            if total > 0:
                done = min(batch_start + batch_size, total)
                print(f"Indexed {min(indexed_count + failed_count, done)}/{total}...")

        return indexed_count, failed_count

    def search(self, query, limit=20):
        """
        Search images by detected object classes.
        Returns [(path, score), ...] sorted by score descending.
        """
        tokens = _parse_query(query)
        if not tokens:
            return []
        rows = self.db.search_by_classes(tokens, limit=limit)
        return [(path, score) for _, path, score in rows]

    def close(self):
        self.db.close()
