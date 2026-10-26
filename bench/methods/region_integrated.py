"""REGION, production path: media_manager.region_index (DINOv2-S windows int8 + 300 RootSIFT
features, real describe_query / query_sift / score_windows / rerank). Corpus DINO windows come from
the cached E1 token maps (same model/long side as the production index, so identical up to the fp16
token cache); SIFT is re-extracted with production rootsift (300 features), cached here.
"""
import os
from multiprocessing import Pool

import numpy as np

from harness import BENCH_DIR, Method, top_k
from proto_e1.prod_sift import sift_one
from methods.e1_dino_geom import _key, WORK
from media_manager import region_index as ri


class RegionIntegrated(Method):
    name = 'region_integrated' + os.environ.get('RI_SUFFIX', '')   # RI_NFEAT/RI_SUFFIX: SIFT-budget experiments
    kind = 'region'

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        f = os.path.join(cache_dir, 'index.npz')
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.isfile(f):
            wins, counts = [], []
            for p in self.paths:
                z = np.load(os.path.join(WORK, _key(p) + '.npz'))
                w = ri.quantize(ri.window_embeddings(z['tok'].astype(np.float32)))
                wins.append(w); counts.append(len(w))
            with Pool(3) as pool:
                sift = pool.map(sift_one, self.paths, chunksize=8)
            scounts = [len(s[0]) for s in sift]
            np.savez(f, wins=np.concatenate(wins), counts=np.array(counts),
                     pts=np.concatenate([s[0] for s in sift]), desc=np.concatenate([s[1] for s in sift]),
                     scounts=np.array(scounts))
        z = np.load(f)
        self.wins, self.counts = z['wins'], z['counts']
        self.pts, self.desc, self.scounts = z['pts'], z['desc'], z['scounts']
        self.sstarts = np.cumsum(self.scounts) - self.scounts

    def warmup(self):
        ri._get_model()

    def _sift(self, i):
        a = self.sstarts[i]
        return self.pts[a:a + self.scounts[i]], self.desc[a:a + self.scounts[i]]

    def query(self, image_path, bbox, k):
        pil = ri.load_pil(open(image_path, 'rb').read())
        q = ri.describe_query(pil, list(bbox))
        qp, qd = ri.query_sift(pil, list(bbox))
        scores = np.empty(len(self.paths), np.float32)
        start = 0
        for i in range(0, len(self.counts), 400):
            c = self.counts[i:i + 400]
            n = int(c.sum())
            scores[i:i + 400] = ri.score_windows(self.wins[start:start + n], c, q)
            start += n
        fused, _ = ri.rerank(np.arange(len(self.paths)), scores, qp, qd, self._sift)
        return top_k(self.paths, fused, k)


METHOD = RegionIntegrated()
