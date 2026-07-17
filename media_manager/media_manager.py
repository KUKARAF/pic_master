"""
Main MediaManager class providing the primary interface for media management operations.
"""
import os
from .database import Database
from .fast_scan import fast_scan
from .broken_finder import find_broken
from .error_log import ErrorLog
from .manual_db import ManualDB

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
        self.errors = ErrorLog(os.path.join(self.data_root, '.media', 'error.db'))
        self.manual = ManualDB(os.path.join(self.data_root, '.media', 'manual.db'))

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

    def start_scan(self, path, recursive=True, reindex=False):
        """
        Scan a directory: find candidates, hash them (content is the file's identity —
        see database.py), upsert into the DB. Path is relative to media_root.
        reindex: force reprocessing (clear primary ML data) for files already tracked
        at the same path — see fast_scan.py.
        """
        abs_path = os.path.join(self.data_root, path)
        return fast_scan(abs_path, self.db, self.data_root, recursive, reindex=reindex)

    def get_file_info(self, path):
        return self.db.get_file_by_path(path)

    def find_duplicates(self, limit=200):
        """Return [(file_id, path_count), ...] for content seen at more than one path —
        the point of content-addressable identity: surfaces real duplicates in a messy
        library for manual review, instead of just tracking each copy separately."""
        return self.db.find_duplicates(limit=limit)

    def get_paths_for_file(self, file_id):
        """Every known path for a piece of content, most-recently-seen first."""
        return self.db.get_paths_for_file(file_id)

    def list_files(self, limit=100):
        """List all files."""
        return self.db.list_files(limit=limit)

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
        for root, dirnames, files in os.walk(abs_path):
            # .media/ holds our own cache/db files — never walk into it.
            dirnames[:] = [d for d in dirnames if d != '.media']
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.data_root)
                digest = self.get_hash_for_file(rel)
                if digest:
                    out.append((rel, digest))
            if not recursive:
                break
        return out

    def get_stale_paths(self, limit=None):
        """Return [(path, status, detail), ...] for tracked paths no longer found on
        disk. status is 'moved' (the same content now lives at another tracked path
        that *does* exist — detail is that path) or 'missing' (no live path left at
        all for this content — detail is None).

        Explicit move-tracking (media mv / record_move) is retired: content identity
        means a rescan (`media add`) already re-links a moved file to its existing
        tags/faces/sets automatically, no separate step needed. This just reports
        what a rescan would find, without requiring one first.
        """
        results = []
        for path, file_id in self.db.list_all_paths():
            if os.path.exists(os.path.join(self.data_root, path)):
                continue
            other_live = [
                p for p, _ in self.db.get_paths_for_file(file_id)
                if p != path and os.path.exists(os.path.join(self.data_root, p))
            ]
            if other_live:
                results.append((path, 'moved', other_live[0]))
            else:
                results.append((path, 'missing', None))
            if limit is not None and len(results) >= limit:
                break
        return results

    def count_files(self, limit=None):
        """
        Return the number of content records.
        limit:  maximum rows to count (None == unlimited)
        """
        return self.db.count_files(limit=limit)
    def find_broken(self, path, max_workers=8):
        """
        Check images/videos under path for corruption.
        Sets 'broken' column to current unix timestamp for broken files.
        Returns number of newly-detected broken files.
        """
        abs_path = os.path.join(self.data_root, path)
        return find_broken(abs_path, self.db.conn, self.data_root, max_workers, error_log=self.errors)

    def count_broken_files(self):
        """Return count of files marked as broken (broken NOT NULL)."""
        return self.db.count_broken_files()

    def list_broken_files(self, limit=100):
        """Return [(path, broken_timestamp), ...] for broken files."""
        return self.db.list_broken_files(limit=limit)

    def clear_broken(self, paths):
        """Clear broken flag (set to NULL) for given list of paths."""
        return self.db.clear_broken(paths)

    def extract_metadata(self, path='.', batch_size=200):
        """
        Read EXIF capture time + GPS coordinates for image files under path that
        haven't been checked yet, store in DB. Independent of hashing/checksums —
        runs off plain file discovery so it works whether or not you've committed.
        Returns (checked_count, found_count).
        """
        from .exif_reader import extract_exif_metadata
        from .formats import IMAGE_EXTENSIONS

        candidates_all = self.db.get_files_without_metadata(limit=None)
        abs_path = os.path.abspath(os.path.join(self.data_root, path))
        candidates = []
        for file_id, rel_path in candidates_all:
            abs_file = os.path.join(self.data_root, rel_path)
            if not abs_file.startswith(abs_path):
                continue
            ext = os.path.splitext(rel_path)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                # Not an image — nothing to extract, but stamp it checked so it
                # doesn't keep showing up in this query on every future run.
                self.db.update_file_metadata(file_id, None, None, None)
                continue
            candidates.append((file_id, abs_file))

        total = len(candidates)
        checked_count = 0
        found_count = 0

        for batch_start in range(0, total, batch_size):
            batch = candidates[batch_start:batch_start + batch_size]
            for file_id, abs_file in batch:
                taken_at, gps_lat, gps_lon = extract_exif_metadata(abs_file)
                self.db.update_file_metadata(file_id, taken_at, gps_lat, gps_lon)
                checked_count += 1
                if taken_at is not None or gps_lat is not None:
                    found_count += 1
            if total > 0:
                done = min(batch_start + batch_size, total)
                print(f"Checked metadata for {done}/{total}...")

        return checked_count, found_count

    def index_files(self, path='.', batch_size=32, model_size='s', conf_threshold=0.15):
        """
        Run YOLO-World on undetected image files under path, store detections in DB.
        Returns (indexed_count, failed_count).
        """
        from .detector import YOLOWorldDetector, SUPPORTED_EXTENSIONS, load_vocab_from_file, merge_vocab

        vocab_path = os.path.join(self.data_root, '.media', 'search_terms.txt')
        base_vocab = load_vocab_from_file(vocab_path)
        vocab = merge_vocab(base_vocab, self.manual.get_all_positive_labels())
        detector = YOLOWorldDetector(model_size=model_size, conf_threshold=conf_threshold, vocab=vocab)
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
                    self.errors.log(path_, error)
                else:
                    self.db.insert_detections(file_id, detections, model_id)
                    indexed_count += 1

            if total > 0:
                done = min(batch_start + batch_size, total)
                print(f"Indexed {min(indexed_count + failed_count, done)}/{total}...")

        return indexed_count, failed_count

    def embed_files(self, path='.', batch_size=32, model_name='ViT-B-32', pretrained='openai'):
        """
        Run CLIP on unembedded image files under path, store embeddings in DB.
        Powers the "find similar" visual-similarity feature (separate from YOLO-World
        object-class search). Returns (indexed_count, failed_count).
        """
        from .indexer import CLIPIndexer, SUPPORTED_EXTENSIONS

        indexer = CLIPIndexer(model_name=model_name, pretrained=pretrained)
        model_id = CLIPIndexer.model_id(model_name, pretrained)

        unindexed = self.db.get_unindexed_files(limit=None)
        abs_path = os.path.abspath(os.path.join(self.data_root, path))
        candidates = []
        for file_id, rel_path in unindexed:
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

            embeddings, failed = indexer.embed_images(batch_paths)
            failed_messages = dict(failed)
            emb_iter = iter(embeddings)
            for file_id, fpath in zip(batch_ids, batch_paths):
                if fpath in failed_messages:
                    failed_count += 1
                    self.errors.log(fpath, failed_messages[fpath])
                else:
                    self.db.insert_embedding(file_id, next(emb_iter).tobytes(), model_id)
                    indexed_count += 1

            if total > 0:
                done = min(batch_start + batch_size, total)
                print(f"Embedded {min(indexed_count + failed_count, done)}/{total}...")

        return indexed_count, failed_count

    def index_bodies(self, path='.', batch_size=16):
        """Crop+embed person boxes under path for find-by-body search (see
        body_index.py) — the CLI equivalent of the web UI's "Build body index"
        button. Reuses stored YOLO-World 'person' detections when a file already
        has them (from `media index`); for a file that's never been object-indexed,
        runs its own dedicated person-only detection pass instead of skipping it,
        so this doesn't require `media index` to have run first. Returns
        (indexed_count, failed_count)."""
        from . import body_index
        from .detector import YOLOWorldDetector
        from .indexer import CLIPIndexer, SUPPORTED_EXTENSIONS

        detector = YOLOWorldDetector(conf_threshold=body_index.MIN_PERSON_CONFIDENCE, vocab=['person'])
        clip_indexer = CLIPIndexer()

        unindexed = self.db.get_unbody_indexed_files()
        abs_path = os.path.abspath(os.path.join(self.data_root, path))
        candidates = []
        for file_id, rel_path in unindexed:
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

        for i, (file_id, abs_file) in enumerate(candidates, start=1):
            try:
                body_index.embed_bodies_for_file(self.db, clip_indexer, file_id, abs_file, detector=detector)
                indexed_count += 1
            except Exception as exc:
                failed_count += 1
                self.errors.log(abs_file, str(exc))
            if total > 0 and (i % batch_size == 0 or i == total):
                print(f"Body-indexed {i}/{total}...")

        return indexed_count, failed_count

    def detect_faces(self, path='.', batch_size=16, model_name='buffalo_l', det_thresh=0.5):
        """Run InsightFace on un-face-indexed image files under path.
        Returns (indexed_count, face_count, failed_count).
        """
        from .face_detector import FaceDetector, SUPPORTED_EXTENSIONS

        detector = FaceDetector(model_name=model_name, det_thresh=det_thresh)
        model_id = FaceDetector.model_id(model_name)

        unindexed = self.db.get_unface_indexed_files(limit=None)
        abs_path = os.path.abspath(os.path.join(self.data_root, path))
        candidates = []
        for file_id, rel_path in unindexed:
            ext = os.path.splitext(rel_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            abs_file = os.path.join(self.data_root, rel_path)
            if not abs_file.startswith(abs_path):
                continue
            candidates.append((file_id, abs_file))

        total = len(candidates)
        indexed_count = 0
        face_count = 0
        failed_count = 0

        for batch_start in range(0, total, batch_size):
            batch = candidates[batch_start:batch_start + batch_size]
            batch_ids = [fid for fid, _ in batch]
            batch_paths = [fpath for _, fpath in batch]

            results = detector.detect_faces(batch_paths)
            for file_id, (path_, faces, error) in zip(batch_ids, results):
                if error is not None:
                    failed_count += 1
                    self.errors.log(path_, error)
                else:
                    self.db.insert_faces(file_id, faces, model_id)
                    indexed_count += 1
                    face_count += len(faces)
                    self._auto_match_faces(file_id)

            done = min(batch_start + batch_size, total)
            print(f"Face-indexed {done}/{total}... ({face_count} faces found so far)")

        return indexed_count, face_count, failed_count

    def _auto_match_faces(self, file_id):
        """After detection, auto-promote any newly detected face in this file to an
        existing identity if it clears manual_db.AUTO_MATCH_THRESHOLD — no human
        confirmation needed for a high-confidence repeat appearance."""
        import json
        file_row = self.db.get_file_by_id(file_id)
        if file_row is None:
            return
        for row in self.db.get_faces_for_file(file_id):
            face_id = row['id']
            emb_bytes = self.db.get_face_embedding(face_id)
            if not emb_bytes:
                continue
            name, score = self.manual.find_matching_identity(emb_bytes)
            if name is None:
                continue
            bbox = json.loads(row['bbox'])
            self.manual.promote_auto_face(face_id, file_row['checksum'], bbox, emb_bytes, name, None, None)

    def search_by_face_image(self, image_path, limit=50, similarity_threshold=0.4):
        """Given an image path, detect the first face and find similar faces in the library.
        Returns [{'path', 'score', 'face_id', 'file_id'}, ...].
        """
        import numpy as np
        from .face_detector import FaceDetector

        detector = FaceDetector()
        results = detector.detect_faces([image_path])
        _, faces, error = results[0]
        if error or not faces:
            return []

        query_face = max(faces, key=lambda f: f['det_score'])
        query_emb = query_face['embedding']

        all_rows = self.db.get_all_face_embeddings()
        if not all_rows:
            return []

        face_ids  = [r[0] for r in all_rows]
        file_ids  = [r[1] for r in all_rows]
        paths     = [r[2] for r in all_rows]
        emb_bytes = [r[3] for r in all_rows]

        matrix = np.stack([np.frombuffer(b, dtype=np.float32) for b in emb_bytes])
        scores = matrix.dot(query_emb).tolist()

        ranked = sorted(
            zip(paths, scores, face_ids, file_ids),
            key=lambda x: x[1], reverse=True,
        )

        seen = {}
        deduped = []
        for fpath, score, face_id, file_id in ranked:
            if score < similarity_threshold:
                break
            if file_id not in seen:
                seen[file_id] = True
                deduped.append({'path': fpath, 'score': score, 'face_id': face_id, 'file_id': file_id})
            if len(deduped) >= limit:
                break
        return deduped

    def search_by_face_name(self, name, limit=100):
        """Return [(file_id, path), ...] for files containing a face named `name`."""
        return self.db.get_files_by_face_identity(name, limit=limit)

    def search(self, query, limit=20):
        """
        Search images by detected object classes.
        Returns [(path, score), ...] sorted by score descending.
        """
        tokens = _parse_query(query)
        if not tokens:
            return []
        rows = self.db.search_by_classes(tokens, limit=limit)
        return [(path, score) for _, path, score, _ in rows]

    def close(self):
        self.db.close()
        self.errors.close()
        self.manual.close()
