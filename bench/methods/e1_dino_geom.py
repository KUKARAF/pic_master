"""E1: DINOv2 multi-scale window prefilter + RootSIFT / MAGSAC geometric re-rank.

Stage 1: ViT-S/14-reg dense tokens (long side 448, aspect-preserving, EXIF-transposed),
mean-pooled into windows (whole, 1/2, 1/3 of each side, stride = half window); photo
score = max cosine over windows and over a few query variants.
Stage 2: top-N photos re-ranked by RootSIFT (ratio 0.8 + mutual, USAC_MAGSAC homography)
inlier count against the query crop; fused score = prefilter + w * log1p(max(0, inl - min)).

Env knobs (tuning): E1_N, E1_QMODE, E1_MININL, E1_W, E1_SIFTONLY=1 (stage 1 off: rerank
everything by SIFT, no DINO at all).
Heavy per-image artefacts (token maps) live in bench/.cache_e1_work (NOT part of the
reported index size; only windows + SIFT are).
"""
import hashlib
import json
import os
import time

import numpy as np
from harness import BENCH_DIR, Method, top_k
from proto_e1 import core

WORK = os.path.join(BENCH_DIR, '.cache_e1_work')
N_CAND = int(os.environ.get('E1_N', 200))
MIN_INL = float(os.environ.get('E1_MININL', 8))
W_INL = float(os.environ.get('E1_W', 1.0))
QMODE = os.environ.get('E1_QMODE', 'crop224,crop448,ctx')
SIFT_ONLY = os.environ.get('E1_SIFTONLY') == '1'
QUERY_SIFT_LONG = 640


def _key(p):
    st = os.stat(p)
    return hashlib.sha1(f'{p}\0{st.st_size}\0{int(st.st_mtime)}'.encode()).hexdigest()[:20]


class E1(Method):
    name = os.environ.get('E1_NAME', 'e1_dino_geom')
    kind = 'region'

    def __init__(self):
        self.dino = None
        self.cache = {}

    def warmup(self):
        if not SIFT_ONLY and self.dino is None:
            self.dino = core.Dino()

    # ------------------------------------------------------------------ index
    def _work(self, p):
        f = os.path.join(WORK, _key(p) + '.npz')
        if os.path.isfile(f):
            z = np.load(f)
            return {k: z[k] for k in z.files}
        t0 = time.perf_counter()
        im = core.load_rgb(p)
        out = {}
        if not SIFT_ONLY:
            tok, cls = self.dino.tokens(im)
            out['tok'] = tok.astype(np.float16)
        t1 = time.perf_counter()
        pts, desc = core.rootsift(im)
        s = min(3.0, core.SIFT_LONG / max(im.width, im.height))
        out['pts'] = (pts * s).astype(np.float16)
        out['desc'] = core.quant_desc(desc)
        out['t_dino'] = np.float32(t1 - t0)
        out['t_sift'] = np.float32(time.perf_counter() - t1)
        os.makedirs(WORK, exist_ok=True)
        np.savez(f + '.tmp.npz', **out)
        os.replace(f + '.tmp.npz', f)
        return out

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        f = os.path.join(cache_dir, 'index.npz')
        if not os.path.isfile(f):
            wins, starts, ptl, dl, sstarts = [], [], [], [], []
            n = tt = ts = 0
            for i, p in enumerate(self.paths):
                w = self._work(p)
                tt += float(w['t_dino']); ts += float(w['t_sift'])
                starts.append(sum(len(x) for x in wins))
                if 'tok' in w and not SIFT_ONLY:
                    e, _ = core.window_embeddings(w['tok'].astype(np.float32))
                    wins.append(e.astype(np.float16))
                sstarts.append(sum(len(x) for x in ptl))
                ptl.append(w['pts']); dl.append(w['desc'])
                if i % 200 == 0:
                    print(f'  e1 index {i}/{len(self.paths)}', flush=True)
            np.savez(f, wins=np.concatenate(wins) if wins else np.zeros((0, 1), np.float16),
                     starts=np.array(starts + [sum(len(x) for x in wins)]),
                     pts=np.concatenate(ptl), desc=np.concatenate(dl),
                     sstarts=np.array(sstarts + [sum(len(x) for x in ptl)]))
            with open(os.path.join(WORK, 'extract_times.json'), 'w') as fh:
                json.dump({'dino_s_per_img': tt / len(self.paths), 'sift_s_per_img': ts / len(self.paths)}, fh)
        z = np.load(f)
        self.wins = z['wins'].astype(np.float32)
        self.starts = z['starts']
        self.pts, self.desc, self.sstarts = z['pts'], z['desc'], z['sstarts']

    # ------------------------------------------------------------------ query
    def _query_vecs(self, im, bbox):
        W, H = im.size
        x0, y0, x1, y1 = [float(v) for v in bbox]
        x0, x1 = max(0, min(x0, W - 1)), max(x0 + 1, min(x1, W))
        y0, y1 = max(0, min(y0, H - 1)), max(y0 + 1, min(y1, H))
        crop = im.crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))))
        vecs = []
        for m in QMODE.split(','):
            if m.startswith('crop'):
                tok, _ = self.dino.tokens(crop, int(m[4:]))
                vecs.append(core.pool_box(tok, (0, 0, 1, 1)))
            elif m == 'ctx':
                tok, _ = self.dino.tokens(im)
                vecs.append(core.pool_box(tok, (x0 / W, y0 / H, x1 / W, y1 / H)))
        return np.stack(vecs), crop

    def prefilter(self, vecs):
        s = self.wins @ vecs.T                                  # [windows, nq]
        return np.maximum.reduceat(s.max(1), self.starts[:-1])

    def candidate_scores(self, image_path, bbox):
        """-> (dino prefilter score per photo, query crop)"""
        im = core.load_rgb(image_path)
        if SIFT_ONLY:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            return np.zeros(len(self.paths), np.float32), im.crop((x0, y0, x1, y1))
        vecs, crop = self._query_vecs(im, bbox)
        return self.prefilter(vecs), crop

    def verify(self, crop, idxs):
        qp, qd = core.rootsift(crop, long_side=QUERY_SIFT_LONG)
        out = {}
        for i in idxs:
            a, b = self.sstarts[i], self.sstarts[i + 1]
            cp = self.pts[a:b].astype(np.float32)
            cd = core.dequant_desc(self.desc[a:b])
            out[i] = core.match_inliers(qp, qd, cp, cd)[0]
        return out

    @staticmethod
    def fuse(dino, inl, min_inl=None, w=None):
        min_inl = MIN_INL if min_inl is None else min_inl
        w = W_INL if w is None else w
        return dino + w * np.log1p(max(0.0, inl - min_inl))

    def query(self, image_path, bbox, k):
        dino, crop = self.candidate_scores(image_path, bbox)
        n = len(self.paths) if SIFT_ONLY else N_CAND
        cand = np.argsort(-dino, kind='stable')[:n]
        inl = self.verify(crop, cand)
        scores = dino.copy()
        for i in cand:
            scores[i] = self.fuse(dino[i], inl[i])
        return top_k(self.paths, scores, k)


METHOD = E1()
