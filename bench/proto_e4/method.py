"""E4 method classes (prefilter + periodic/aperiodic verification)."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import cv2
import numpy as np

from harness import Method, CACHE_ROOT
from proto_e4 import core

K_WORDS = 1024
SHORTLIST = 200
PERIODIC_T = 0.30
THREADS = 2
SHARED = os.path.join(CACHE_ROOT, 'e4_periodic_verify')


def _feat(p):
    cv2.setNumThreads(1)
    return core.image_features(p)


def fuse(periodic, ncc, inl):
    sn = 1.0 - np.exp(-np.asarray(inl, np.float32) / 12.0)
    nn = np.clip((np.asarray(ncc, np.float32) - 0.3) / 0.6, 0, 1)
    return nn + 0.35 * sn if periodic else sn + 0.35 * nn


class E4(Method):
    name = 'e4_periodic_verify'
    kind = 'pattern'
    verify_all = False
    mode = 'fused'          # fused | routed | ncc | sift

    def __init__(self):
        self.diag = {}

    # ------------------------------------------------------------------ index
    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        d = SHARED
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, 'feat.npz')
        if not os.path.isfile(f):
            with Pool(THREADS) as pool:
                feats = pool.map(_feat, self.paths, chunksize=8)
            ncc = [x[0] for x in feats]
            kn = np.array([len(x[1]) for x in feats])
            xy = np.concatenate([x[1] for x in feats])
            des = np.concatenate([x[2] for x in feats])
            shapes = np.array([a.shape for a in ncc])
            flat = np.concatenate([a.ravel() for a in ncc])
            # codebook on a descriptor sample, then BoVW (sqrt tf, idf, L2)
            rng = np.random.default_rng(0)
            samp = des[rng.choice(len(des), min(150000, len(des)), replace=False)].astype(np.float32)
            cv2.setNumThreads(THREADS)
            _, _, centers = cv2.kmeans(samp, K_WORDS, None,
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1e-3),
                                       1, cv2.KMEANS_PP_CENTERS)
            cv2.setNumThreads(1)
            words = self._assign(des, centers)
            img_of = np.repeat(np.arange(len(self.paths)), kn)
            tf = np.zeros((len(self.paths), K_WORDS), np.float32)
            np.add.at(tf, (img_of, words), 1)
            df = (tf > 0).sum(0)
            idf = np.log((1 + len(self.paths)) / (1 + df)).astype(np.float32) + 1
            bovw = self._bovw_norm(tf, idf)
            np.savez(f, n=np.array(len(self.paths)), shapes=shapes, flat=flat, kn=kn, xy=xy, des=des,
                     centers=centers, idf=idf, bovw=bovw.astype(np.float16))
        z = np.load(f)
        if int(z['shapes'].max()) > core.NCC_SIDE:      # cache from an older NCC_SIDE: shrink in place
            sh, fl = z['shapes'], z['flat']
            o = np.concatenate([[0], np.cumsum(sh[:, 0] * sh[:, 1])])
            ims = [core.resize_long(fl[o[i]:o[i + 1]].reshape(tuple(sh[i])), core.NCC_SIDE)[0] for i in range(len(sh))]
            d = dict(z)
            d['shapes'] = np.array([a.shape for a in ims])
            d['flat'] = np.concatenate([a.ravel() for a in ims])
            np.savez(f, **d)
            z = np.load(f)
        self.shapes, self.kn = z['shapes'], z['kn']
        flat = z['flat']
        so = np.concatenate([[0], np.cumsum(self.shapes[:, 0] * self.shapes[:, 1])])
        self.ncc_imgs = [flat[so[i]:so[i + 1]].reshape(tuple(self.shapes[i])) for i in range(len(self.paths))]
        ko = np.concatenate([[0], np.cumsum(self.kn)])
        xy, des = z['xy'], z['des']
        self.kxy = [xy[ko[i]:ko[i + 1]] for i in range(len(self.paths))]
        self.kdes = [des[ko[i]:ko[i + 1]] for i in range(len(self.paths))]
        self.centers, self.idf = z['centers'], z['idf']
        self.bovw = z['bovw'].astype(np.float32)
        self.pool = ThreadPoolExecutor(THREADS)

    @staticmethod
    def _assign(des, centers):
        out = np.empty(len(des), np.int32)
        c2 = (centers ** 2).sum(1)
        for i in range(0, len(des), 50000):
            x = des[i:i + 50000].astype(np.float32)
            out[i:i + 50000] = np.argmin(c2[None] - 2 * x @ centers.T, 1)
        return out

    @staticmethod
    def _bovw_norm(tf, idf):
        v = np.sqrt(tf) * idf
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # ------------------------------------------------------------------ query
    def _verify_one(self, i, tpls, ctpls, qxy, qdesf):
        ncc = core.ncc_score(self.ncc_imgs[i], tpls, ctpls)
        inl = core.ransac_inliers(qxy, qdesf, self.kxy[i], self.kdes[i])
        return ncc, inl

    def query(self, image_path, bbox, k):
        crop, s, fq = core.prepare_query(image_path, bbox)
        strength, period = core.periodicity(crop)
        periodic = strength >= PERIODIC_T
        qxy, qdes = core.rootsift(crop)
        qdesf = qdes.astype(np.float32) / 255.0
        # prefilter: BoVW cosine of crop's visual words
        pre = np.zeros(len(self.paths), np.float32)
        if len(qdes):
            w = self._assign(qdes, self.centers)
            tf = np.bincount(w, minlength=K_WORDS).astype(np.float32)[None]
            pre = (self.bovw @ self._bovw_norm(tf, self.idf)[0]).astype(np.float32)
        cand = np.arange(len(self.paths)) if self.verify_all else np.argsort(-pre)[:SHORTLIST]
        tpls = core.ncc_prepare(core.make_templates(crop, period, periodic), fq)
        ctpls = core.coarse_templates(tpls)
        res = list(self.pool.map(lambda i: self._verify_one(i, tpls, ctpls, qxy, qdesf), cand))
        ncc = np.array([r[0] for r in res], np.float32)
        inl = np.array([r[1] for r in res], np.float32)
        self.diag[(image_path, tuple(bbox))] = dict(cand=cand, ncc=ncc, inl=inl, pre=pre,
                                                    strength=strength, period=period or 0.0)
        if self.mode == 'ncc':
            sc = ncc
        elif self.mode == 'sift':
            sc = inl
        else:
            sc = fuse(periodic, ncc, inl)
        # verified candidates first (score > 1 offset), the rest by prefilter score
        final = pre * 0.5 - 10.0
        final[cand] = sc + 1.0
        order = np.argsort(-final)[:k]
        return [(self.paths[i], float(final[i])) for i in order]
