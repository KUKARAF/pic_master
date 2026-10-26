"""E2: DINOv2-S/14-reg dense patch-token store + query-pyramid MaxSim (region + pattern).

Index per image: long side 448 (EXIF-transposed), ViT-S/14 + 4 registers, patch tokens
(384-d) -> PCA 64-d -> L2 -> int8. Stored as one flat token file + grid dims.
Query: crop -> pyramid (x rotations for pattern) -> query tokens -> MaxSim vs all image
tokens (chunked), plus optional re-rank stages (mutual-NN, offset voting) on a shortlist.
"""
import os
import time

import numpy as np
import torch
from PIL import Image, ImageOps

torch.set_num_threads(2)
LONG = 448
PATCH = 14
DIM = 64
HUB = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_Q = 128


def load_model():
    m = torch.hub.load(HUB, 'dinov2_vits14_reg', source='local', pretrained=True)
    return m.eval()


def to_tensor(img):
    a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def resize_to_grid(img, scale):
    """Resize so that the longer side ~ scale px, both sides multiples of 14 (>=14)."""
    w, h = img.size
    nw = max(PATCH, int(round(w * scale / PATCH)) * PATCH)
    nh = max(PATCH, int(round(h * scale / PATCH)) * PATCH)
    return img.resize((nw, nh), Image.BICUBIC), nh // PATCH, nw // PATCH


@torch.inference_mode()
def tokens_of(model, img):
    """-> (tokens [gh*gw,384] fp32 numpy, cls [384], gh, gw)."""
    t = to_tensor(img)
    out = model.forward_features(t)
    p = out['x_norm_patchtokens'][0].numpy()
    return p, out['x_norm_clstoken'][0].numpy(), t.shape[2] // PATCH, t.shape[3] // PATCH


def load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.load()
    return im


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


class PCA:
    def __init__(self, mean, comps):
        self.mean, self.comps = mean, comps

    def __call__(self, x):
        return l2((x - self.mean) @ self.comps)


def fit_pca(sample):
    mean = sample.mean(0)
    c = np.cov((sample - mean).T)
    w, v = np.linalg.eigh(c)
    return PCA(mean.astype(np.float32), v[:, ::-1][:, :DIM].astype(np.float32))


def quant(x):
    return np.clip(np.round(x * 127), -127, 127).astype(np.int8)


class E2Index:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = load_model()
        return self.model

    # ------------------------------------------------------------- index
    def build(self, paths, scratch):
        f = os.path.join(self.dir, 'tokens.i8')
        if os.path.isfile(os.path.join(self.dir, 'meta.npz')):
            return self.load(paths)
        model = self.get_model()
        os.makedirs(scratch, exist_ok=True)
        rawp, gp = os.path.join(scratch, 'raw.f16'), os.path.join(scratch, 'grids.txt')
        grids, clss = [], []
        if os.path.isfile(gp):   # resume
            for line in open(gp).read().splitlines():
                v = line.split()
                if len(v) == 386:
                    grids.append((int(v[0]), int(v[1])))
                    clss.append(np.array(v[2:], np.float32))
            with open(rawp, 'ab') as f:
                f.truncate(sum(h * w for h, w in grids) * 384 * 2)
        else:
            open(rawp, 'wb').close()
        raw, gf = open(rawp, 'ab'), open(gp, 'a')
        t_start = time.perf_counter()
        for n, p in enumerate(paths[len(grids):], len(grids)):
            im = load_rgb(p)
            img, gh, gw = resize_to_grid(im, LONG / max(im.size))
            tk, cls, gh, gw = tokens_of(model, img)
            raw.write(tk.astype(np.float16).tobytes())
            raw.flush()
            gf.write(f"{gh} {gw} {' '.join(repr(float(x)) for x in cls)}\n")
            gf.flush()
            grids.append((gh, gw))
            clss.append(cls)
            if n % 50 == 0:
                print('index', n, len(paths), round(time.perf_counter() - t_start), 's', flush=True)
        raw.close()
        gf.close()
        grids = np.array(grids, np.int32)
        counts = grids[:, 0] * grids[:, 1]
        starts = np.cumsum(counts) - counts
        n = int(counts.sum())
        mm = np.memmap(os.path.join(scratch, 'raw.f16'), np.float16, 'r', shape=(n, 384))
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, min(n, 80000), replace=False))
        pca = fit_pca(mm[idx].astype(np.float32))
        out = open(f, 'wb')
        for i in range(0, n, 20000):
            out.write(quant(pca(mm[i:i + 20000].astype(np.float32))).tobytes())
        out.close()
        np.savez(os.path.join(self.dir, 'meta.npz'), grids=grids, starts=starts,
                 pmean=pca.mean, pcomps=pca.comps, cls=np.stack(clss).astype(np.float16))
        del mm
        os.remove(os.path.join(scratch, 'raw.f16'))
        os.remove(gp)
        return self.load(paths)

    def load(self, paths):
        z = np.load(os.path.join(self.dir, 'meta.npz'))
        self.paths = list(paths)
        self.grids, self.starts = z['grids'], z['starts']
        self.counts = self.grids[:, 0] * self.grids[:, 1]
        self.pca = PCA(z['pmean'], z['pcomps'])
        self.N = int(self.counts.sum())
        self.tok = np.memmap(os.path.join(self.dir, 'tokens.i8'), np.int8, 'r', shape=(self.N, DIM))
        self.cls = z['cls'].astype(np.float32)
        self._coarse = None
        return self

    def coarse(self):
        """2x2-pooled tokens (RAM-resident prefilter): (tokens, starts, grids)."""
        if self._coarse is None:
            blocks, cg = [], []
            for i in range(len(self.paths)):
                gh, gw = self.grids[i]
                t = self.tok[self.starts[i]:self.starts[i] + gh * gw].astype(np.float32) / 127
                t = t.reshape(gh, gw, DIM)
                h2, w2 = gh // 2, gw // 2
                t = t[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, DIM).mean((1, 3))
                blocks.append(l2(t.reshape(-1, DIM)))
                cg.append((h2, w2))
            cg = np.array(cg, np.int32)
            cc = cg[:, 0] * cg[:, 1]
            self._coarse = (np.concatenate(blocks).astype(np.float32), np.cumsum(cc) - cc, cg)
        return self._coarse

    # ------------------------------------------------------------- query
    def query_tokens(self, image_path, bbox, mults, rots, src_scale=None):
        """List of (label, Q[nq,64] fp32, pos[nq,2], grid(h,w)) for the pyramid."""
        im = load_rgb(image_path)
        crop = im.crop(tuple(int(round(v)) for v in bbox))
        s0 = LONG / max(im.size)
        model = self.get_model()
        out, seen = [], set()
        for r in rots:
            c = crop if r == 0 else crop.rotate(r, expand=True)
            for m in mults:
                s = s0 * m
                w, h = c.size
                if max(w, h) * s > LONG:
                    s = LONG / max(w, h)
                if min(w, h) * s < 2 * PATCH:
                    continue
                key = (r, round(w * s / PATCH), round(h * s / PATCH))
                if key in seen:
                    continue
                seen.add(key)
                img, gh, gw = resize_to_grid(c, s)
                tk, cls, gh, gw = tokens_of(model, img)
                q = self.pca(tk)
                pos = np.stack(np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij'), -1).reshape(-1, 2)
                if len(q) > MAX_Q:
                    sel = np.sort(np.random.default_rng(0).choice(len(q), MAX_Q, replace=False))
                    q, pos = q[sel], pos[sel]
                out.append(((r, m), q.astype(np.float32), pos, (gh, gw)))
        return out

    def chunks(self, tok=None, starts=None, counts=None, target=60000):
        """Yield (i0, i1, T[float32], local starts) over groups of images."""
        if tok is None:
            tok, starts, counts = self.tok, self.starts, self.counts
        n = len(counts)
        i0 = 0
        while i0 < n:
            i1 = i0 + 1
            while i1 < n and starts[i1] + counts[i1] - starts[i0] <= target:
                i1 += 1
            a, b = starts[i0], starts[i1 - 1] + counts[i1 - 1]
            T = np.asarray(tok[a:b]).astype(np.float32)
            if T.dtype != np.float32 or tok.dtype == np.int8:
                T = T / 127.0
            yield i0, i1, T, (starts[i0:i1] - a)
            i0 = i1

    def maxsim_scores(self, qlist, coarse=False):
        """Per-image scores for variants over the pyramid: dict name -> [n_img]."""
        if coarse:
            tok, starts, cg = self.coarse()
            counts = cg[:, 0] * cg[:, 1]
        else:
            tok, starts, counts = self.tok, self.starts, self.counts
        n = len(counts)
        res = {k: np.full(n, -1e9, np.float32) for k in ('maxsim', 'trim', 'pooled')}
        for label, Q, pos, grid in qlist:
            if coarse:   # pool the query tokens 2x2 too
                gh, gw = grid
                h2, w2 = gh // 2, gw // 2
                if h2 < 1 or w2 < 1:
                    continue
                Qp = self._pool_query(label, Q, pos, grid)
                if Qp is None:
                    continue
                Q = Qp
            pq = l2(Q.mean(0, keepdims=True))[0]
            for i0, i1, T, ls in self.chunks(tok, starts, counts):
                S = Q @ T.T                                    # [nq, nt]
                best = np.maximum.reduceat(S, ls, axis=1)     # [nq, nimg]
                res['maxsim'][i0:i1] = np.maximum(res['maxsim'][i0:i1], best.mean(0))
                k = max(1, best.shape[0] // 2)
                part = np.partition(best, best.shape[0] - k, axis=0)[best.shape[0] - k:]
                res['trim'][i0:i1] = np.maximum(res['trim'][i0:i1], part.mean(0))
                pm = np.maximum.reduceat(T @ pq, ls)
                res['pooled'][i0:i1] = np.maximum(res['pooled'][i0:i1], pm)
        return res

    def _pool_query(self, label, Q, pos, grid):
        gh, gw = grid
        if len(Q) != gh * gw:    # subsampled: cannot pool exactly
            return Q
        t = Q.reshape(gh, gw, DIM)
        h2, w2 = gh // 2, gw // 2
        if h2 < 1 or w2 < 1:
            return None
        t = t[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, DIM).mean((1, 3))
        return l2(t.reshape(-1, DIM)).astype(np.float32)

    def rerank_scores(self, qlist, cand, vote_tau=0.0):
        """Per-candidate (mnn, vote) scores. cand: image indices."""
        mnn = np.zeros(len(cand), np.float32)
        vote = np.zeros(len(cand), np.float32)
        for ci, i in enumerate(cand):
            gh, gw = self.grids[i]
            T = self.tok[self.starts[i]:self.starts[i] + gh * gw].astype(np.float32) / 127.0
            tpos = np.stack(np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij'), -1).reshape(-1, 2)
            for label, Q, pos, grid in qlist:
                S = Q @ T.T
                nq = len(Q)
                j = S.argmax(1)
                sim = S[np.arange(nq), j]
                back = S.argmax(0)
                mutual = back[j] == np.arange(nq)
                mnn[ci] = max(mnn[ci], float((sim * mutual).sum() / nq))
                off = tpos[j] - pos                         # [nq,2]
                off = off - off.min(0) if False else off
                oh = off[:, 0] + 200
                ow = off[:, 1] + 200
                H = np.zeros((401, 401), np.float32)
                keep = sim > 0
                np.add.at(H, (oh[keep], ow[keep]), sim[keep])
                # 3x3 box sum
                P = np.pad(H, 1)
                Hs = sum(P[a:a + 401, b:b + 401] for a in range(3) for b in range(3))
                vote[ci] = max(vote[ci], float(Hs.max() / nq))
        return mnn, vote
