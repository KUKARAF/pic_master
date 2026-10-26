"""PATTERN, production path: media_manager.pattern_deep (fixed PCA-96 projection, int8 windows,
real describe_query + score_windows). The corpus windows are projected/quantised from the cached
raw ResNet18 stats of the E3-deep prototype (verified identical to pattern_deep.describe_image up
to the fp16 cache: mean |int8 diff| ~0.003), so the index build skips ~25 min of CPU inference."""
import os

import numpy as np

from harness import Method, top_k, CACHE_ROOT
from media_manager import pattern_deep as pdp

RAW = os.path.join(CACHE_ROOT, 'e3_raw', 'raw_deep.npz')


class PatternIntegrated(Method):
    name = 'pattern_integrated'
    kind = 'pattern'

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        z = np.load(RAW)
        raw, counts = z['feats'], z['counts']
        assert len(counts) == len(self.paths)
        os.makedirs(cache_dir, exist_ok=True)
        f = os.path.join(cache_dir, 'windows.npz')
        if not os.path.isfile(f):
            parts = [pdp.quantize(pdp.project(raw[i:i + 100000].astype(np.float32)))
                     for i in range(0, len(raw), 100000)]
            np.savez(f, w=np.concatenate(parts), counts=counts)
        z = np.load(f)
        self.w, self.counts = z['w'], z['counts']

    def warmup(self):
        pdp._get_net()

    def query(self, image_path, bbox, k):
        rgb = pdp.load_rgb(open(image_path, 'rb').read())
        q = pdp.describe_query(rgb, list(bbox))
        if q is None:
            return top_k(self.paths, np.zeros(len(self.paths)), k)
        scores = np.empty(len(self.paths), np.float32)
        step, start = 400, 0
        for i in range(0, len(self.counts), step):
            c = self.counts[i:i + step]
            n = int(c.sum())
            scores[i:i + step] = pdp.score_windows(self.w[start:start + n], c, q)
            start += n
        return top_k(self.paths, scores, k)


METHOD = PatternIntegrated()
