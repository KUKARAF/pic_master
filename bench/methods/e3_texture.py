"""E3 (PATTERN): native-scale multi-scale dense-window texture descriptors.

Index: every image -> pyramid (sqrt2 steps) of 48-px windows (stride 32) described by
riu2-LBP(R=1,2,3) + Gabor energy (rotation-sorted) + Lab moments (see proto_e3/descriptor.py).
Query: crop at its native pixel scale (+ down-scalings) -> mean window descriptor per scale;
image score = max over (query scale, window) of cosine in a corpus-whitened space.
Variants via env E3_VARIANT (default 'whiten'); raw windows are cached once and shared.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

from harness import Method, max_per_group, top_k, CACHE_ROOT
from proto_e3 import descriptor as D

RAW_DIR = os.path.join(CACHE_ROOT, 'e3_raw')
VARIANTS = {
    # whiten: shrinkage PCA whitening; bw = per-block weights (lbp, gab, col)
    'whiten': dict(whiten=0.3, bw=(1.0, 0.6, 0.4), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
    'nowhiten': dict(whiten=None, bw=(1.0, 0.6, 0.4), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
    'whiten_full': dict(whiten=0.05, bw=(1.0, 0.6, 0.4), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
    'whiten_qset': dict(whiten=0.3, bw=(1.0, 0.6, 0.4), agg='qset', qs=(1.0, 0.71, 0.5, 0.35)),
    'whiten_s1': dict(whiten=0.3, bw=(1.0, 0.6, 0.4), agg='mean', qs=(1.0,)),
    'lbp_only': dict(whiten=0.3, bw=(1.0, 0.0, 0.0), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
    'nocolor': dict(whiten=0.3, bw=(1.0, 0.6, 0.0), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
    'best': dict(whiten=0.3, bw=(1.0, 0.6, 1.0), agg='qset', qs=(1.0, 0.71, 0.5, 0.35)),
    'color_heavier': dict(whiten=0.3, bw=(1.0, 0.6, 1.6), agg='qset', qs=(1.0, 0.71, 0.5, 0.35)),
    'color_heavy': dict(whiten=0.3, bw=(1.0, 0.6, 1.0), agg='mean', qs=(1.0, 0.71, 0.5, 0.35)),
}


class E3Texture(Method):
    kind = 'pattern'

    mod = D
    raw_name = 'raw.npz'
    pre = staticmethod(lambda x: x)

    def __init__(self, variant='best'):
        self.variant = variant
        self.cfg = VARIANTS[variant]
        self.name = 'e3_texture' if variant == 'best' else f'e3_texture_{variant}'

    # ---------------------------------------------------------- index
    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        os.makedirs(RAW_DIR, exist_ok=True)
        f = os.path.join(RAW_DIR, self.raw_name)
        from harness import index_key
        key = index_key(self.paths)
        ok = False
        if os.path.isfile(f):
            z = np.load(f)
            ok = str(z['key']) == key
        if not ok:
            with Pool(2) as pool:
                res = pool.map(self.mod.index_one, self.paths, chunksize=8)
            counts = np.array([len(r[0]) for r in res])
            np.savez(f, key=key, feats=np.concatenate([r[0] for r in res]),
                     meta=np.concatenate([r[1] for r in res]), counts=counts)
            # stored size reported by the harness = cache_dir contents; mirror the raw file there
            os.makedirs(cache_dir, exist_ok=True)
            os.link(f, os.path.join(cache_dir, self.raw_name)) if not os.path.exists(os.path.join(cache_dir, self.raw_name)) else None
        z = np.load(f)
        raw, counts = z['feats'], z['counts']
        self.meta = z['meta']
        self.starts = np.cumsum(counts) - counts
        self.counts = counts
        self._prepare(raw)

    def _prepare(self, raw):
        c = self.cfg
        rng = np.random.default_rng(0)
        sub = self.pre(raw[rng.choice(len(raw), min(150000, len(raw)), replace=False)].astype(np.float32))
        self.mu, self.sd = sub.mean(0), sub.std(0) + 1e-4
        self.bwv = np.zeros(self.mod.DIM, np.float32)
        for (name, sl), w in zip(self.mod.BLOCKS.items(), c['bw']):
            self.bwv[sl] = w / np.sqrt(sl.stop - sl.start)
        z = (sub - self.mu) / self.sd * self.bwv
        if c['whiten'] is not None:
            cov = np.cov(z.T)
            ev, V = np.linalg.eigh(cov)
            keep = ev > 1e-6 * ev.max()
            ev, V = ev[keep], V[:, keep]
            self.W = (V / np.sqrt(ev + c['whiten'] * ev.mean())).astype(np.float32)
        else:
            self.W = np.eye(self.mod.DIM, dtype=np.float32)
        self.X = self._project(raw)
        # empty images own no windows: give them one -inf-score row via counts==0 handling
        self.empty = self.counts == 0

    def _project(self, raw, chunk=200000):
        out = np.empty((len(raw), self.W.shape[1]), np.float32)
        for i in range(0, len(raw), chunk):
            z = (self.pre(raw[i:i + chunk].astype(np.float32)) - self.mu) / self.sd * self.bwv
            z = z @ self.W
            z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
            out[i:i + chunk] = z
        return out

    # ---------------------------------------------------------- query
    def _score_vec(self, q):
        s = self.X @ q
        return s

    def _per_image(self, s):
        # images with zero windows have starts equal to the next image's start: mask them
        safe = np.minimum(self.starts, len(s) - 1)
        m = np.maximum.reduceat(s, safe)
        m = np.where(self.empty, -1.0, m)
        return m

    def query(self, image_path, bbox, k):
        rgb = D.load_rgb(image_path)
        qd = self.mod.describe_query(rgb, list(bbox), self.cfg['qs'])
        if not qd:
            return top_k(self.paths, np.zeros(len(self.paths)), k)
        # one pass over X for all (scale, query-window) vectors: S = X @ Q.T, then per-image max
        Q, grp = [], []
        for gi, (s, raw) in enumerate(qd.items()):
            P = self._project(raw)
            if self.cfg['agg'] == 'mean':
                q = P.mean(0)
                P = (q / (np.linalg.norm(q) + 1e-9))[None]
            else:                                        # qset: <=8 evenly spread query windows
                P = P[::max(1, len(P) // 8)][:8]
            Q.append(P); grp += [gi] * len(P)
        Q = np.concatenate(Q).astype(np.float32); grp = np.array(grp)
        per = np.empty((len(self.paths), len(Q)), np.float32)
        safe = np.minimum(self.starts, len(self.X) - 1)
        for c0 in range(0, len(Q), 8):
            per[:, c0:c0 + 8] = np.maximum.reduceat(self.X @ Q[c0:c0 + 8].T, safe, axis=0)
        per[self.empty] = -1.0
        best = np.max([per[:, grp == g].mean(1) for g in np.unique(grp)], axis=0)
        return top_k(self.paths, best, k)


def make():
    return E3Texture(os.environ.get('E3_VARIANT', 'best'))
