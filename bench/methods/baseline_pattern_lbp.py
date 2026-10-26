"""Current production pattern search: LBP+HSV+Gabor descriptor (pattern_descriptor.py)
per tile of the same grid, photo score = best-tile cosine to the query region's
descriptor (web.py api_pattern_search)."""
import os

import numpy as np

from harness import Method, max_per_group, top_k
from media_manager import pattern_descriptor
from media_manager.tile_index import generate_tiles


class PatternLBP(Method):
    name = 'baseline_pattern_lbp'
    kind = 'pattern'

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        f = os.path.join(cache_dir, 'tiles.npz')
        if not os.path.isfile(f):
            blocks = []
            for p in self.paths:
                with open(p, 'rb') as fh:
                    bgr = pattern_descriptor.decode_bgr(fh.read())
                descs = []
                if bgr is not None:
                    h, w = bgr.shape[:2]
                    boxes = [list(t) for t in generate_tiles(w, h)]
                    descs = [d for d in pattern_descriptor.descriptors_for_boxes(bgr, boxes) if d is not None]
                blocks.append(np.array(descs, dtype=np.float32).reshape(len(descs), -1))
            dim = max((b.shape[1] for b in blocks if len(b)), default=1)
            # An undecodable image keeps one zero row so every image owns >=1 row (score 0).
            blocks = [b if len(b) else np.zeros((1, dim), np.float32) for b in blocks]
            counts = np.array([len(b) for b in blocks])
            np.savez(f, matrix=np.concatenate(blocks), starts=np.cumsum(counts) - counts)
        z = np.load(f)
        self.matrix, self.starts = z['matrix'], z['starts']

    def query(self, image_path, bbox, k):
        with open(image_path, 'rb') as fh:
            bgr = pattern_descriptor.decode_bgr(fh.read())
        q = pattern_descriptor.descriptors_for_boxes(bgr, [list(bbox)])[0]
        scores = max_per_group(self.matrix.dot(np.asarray(q, dtype=np.float32)), self.starts)
        return top_k(self.paths, scores, k)


METHOD = PatternLBP()
