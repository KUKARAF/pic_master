"""DINOv2-L/14-reg dense-cell descriptors for find-by-pattern — the ALTERNATIVE to the ResNet18
texture windows in pattern_deep.py (selectable per search; pattern_deep stays the default until
bench/bench_pattern_dino.py shows this is better on the user's data).

Why a second algorithm: DINOv2's patch tokens are *semantic* local features (parts, prints,
objects, material categories) where ResNet18 layer2/3 statistics are *texture-statistical*.
Real-photo patterns (a dress print, a wallpaper behind a person, a logo texture) may favour
the former; clean swatches the latter (the earlier DINOv2-S prototype, bench/methods/
e2_dino_dense.py, scored pattern mAP .43 vs the ResNet's .73-.76 — parity on the real split,
far behind on synthetic swatches), so the intended use is either side by side or fused.

Index: the image is resized (aspect kept, long side LONG=448, both sides multiples of 28) and run
through DINOv2-L/14-reg at that resolution (<=32x32 patch tokens, 1024-d). Tokens are
avg-pooled 2x2 into cells (28 px at index scale, <=256 cells/image, ~190 for 4:3), each cell
vector is L2-normalised, centred, PCA-whitened (fixed projection data/pattern_dino_proj.npz,
fitted by fit_projection / bench/bench_pattern_dino.py --fit), L2-normalised and stored int8
(DIM B/cell, default 128 -> ~24 KB/image, same order as the ResNet index).

Query: the drawn box is described two ways and the vectors are *all* used (an image scores its
best (cell, query-vector) cosine, same scorer as pattern_deep.score_windows):
  mask  - the whole photo goes through the model at index resolution and the tokens under the box
          are averaged (token-overlap weighted) -> same context/distribution as the index cells;
  crop  - the box is cut out and run alone at CROP_SCALES (and optionally 90-degree rotations),
          mean of its tokens -> works when the box is smaller than a cell or its context misleads.

The two heavy halves (dense forward + decode) are shared with the worker; this module needs
numpy at import time only — torch/cv2/PIL load lazily, and constructing the DINOv2 model is the
caller's job (embedders.DINOv2Embedder, the SAME instance/weights the embedding pipeline uses).
Bump ALGO_BASE on any change to the cell definition; the projection hash is part of the tag so
refitting automatically marks the library "needs reindex".
"""
import hashlib
import io
import os
import threading
import time

import numpy as np

_PROJ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pattern_dino_proj.npz')

TOKEN_DIM = 1024                   # DINOv2-L
PATCH = 14
POOL = 2                           # tokens per cell side
CELL_PX = PATCH * POOL             # 28
LONG = 448                         # long side at index scale (32 tokens)
MIN_LONG = 56                      # smaller images are "indexed, nothing describable"
DEFAULT_DIM = 128
MIN_CROP = 56                      # query crops are run at >= 4x4 tokens
CROP_SCALES = (1.0, 1.5, 0.67)     # query crop zoom relative to its size at index scale
QUERY_MODES = ('mask', 'crop', 'both')
ALGO_BASE = 'dinol14reg-cell2-l448'
FIT_SHRINK = 0.3                   # whitening shrinkage (as the ResNet projection)

_proj = None
_proj_lock = threading.Lock()


# ------------------------------------------------------------------------------ projection

def _hash(mu, W):
    return hashlib.sha1(np.ascontiguousarray(mu, np.float32).tobytes()
                        + np.ascontiguousarray(W, np.float32).tobytes()).hexdigest()[:6]


def set_projection(mu, W):
    """Install a projection (tests / the benchmark, before it is written to data/)."""
    global _proj
    mu, W = np.asarray(mu, np.float32), np.asarray(W, np.float32)
    if mu.shape != (TOKEN_DIM,) or W.ndim != 2 or W.shape[0] != TOKEN_DIM:
        raise ValueError(f'projection must be mu ({TOKEN_DIM},), W ({TOKEN_DIM}, dim); got {mu.shape}, {W.shape}')
    with _proj_lock:
        _proj = {'mu': mu, 'W': W, 'hash': _hash(mu, W)}


def clear_projection():
    global _proj
    with _proj_lock:
        _proj = None


def _get_proj():
    """The fitted projection, or None when data/pattern_dino_proj.npz has not been produced yet."""
    global _proj
    if _proj is None and os.path.isfile(_PROJ_PATH):
        z = np.load(_PROJ_PATH)
        set_projection(z['mu'], z['W'])
    return _proj


def projection_ready():
    return _get_proj() is not None


def dim():
    p = _get_proj()
    return int(p['W'].shape[1]) if p else DEFAULT_DIM


def algo_tag():
    """Version tag stored with every row. Includes the projection hash: a refit (or a different
    dim) changes the tag, so old rows are ignored until re-indexed ("needs reindex")."""
    p = _get_proj()
    return f"{ALGO_BASE}-pca{p['W'].shape[1]}i8-p{p['hash']}-v1" if p else f'{ALGO_BASE}-unfitted'


def __getattr__(name):  # pattern_dino.ALGO_TAG / .DIM behave like pattern_deep's constants
    if name == 'ALGO_TAG':
        return algo_tag()
    if name == 'DIM':
        return dim()
    raise AttributeError(name)


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def fit_projection(cells, out_dim=DEFAULT_DIM, shrink=FIT_SHRINK, max_rows=300000, seed=0):
    """(mu, W) from raw pooled cells float (N, TOKEN_DIM): L2-normalise, centre, PCA, whiten with
    shrinkage `shrink * mean eigenvalue` (the same recipe as the ResNet projection, where whitening
    was worth ~+0.25 mAP). Does not install it — see set_projection / save_projection."""
    cells = np.asarray(cells, np.float32)
    if len(cells) > max_rows:
        cells = cells[np.random.default_rng(seed).choice(len(cells), max_rows, replace=False)]
    z = _l2(cells)
    mu = z.mean(0)
    ev, V = np.linalg.eigh(np.cov((z - mu).T))
    order = np.argsort(-ev)[:out_dim]
    ev_k = np.maximum(ev[order], 0)
    W = (V[:, order] / np.sqrt(ev_k + shrink * np.maximum(ev, 0).mean())).astype(np.float32)
    return mu.astype(np.float32), W


def save_projection(mu, W, path=None):
    path = path or _PROJ_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu=np.asarray(mu, np.float32), W=np.asarray(W, np.float32))
    return path


def project(raw, mu=None, W=None):
    """Raw pooled cells (n, TOKEN_DIM) -> L2-normalised float32 (n, dim)."""
    if mu is None:
        p = _get_proj()
        if p is None:
            raise RuntimeError('pattern_dino projection missing (data/pattern_dino_proj.npz): '
                               'run bench/bench_pattern_dino.py --fit on the GPU machine and redeploy')
        mu, W = p['mu'], p['W']
    z = (_l2(np.asarray(raw, np.float32)) - mu) @ W
    return _l2(z).astype(np.float32)


def quantize(vecs):
    """Unit-norm float32 (n, dim) -> int8 (same encoding as pattern_deep.quantize)."""
    return np.clip(np.rint(vecs * 127.0), -127, 127).astype(np.int8)


# ---------------------------------------------------------------------------- geometry/prep

def target_size(h, w):
    """(th, tw), both multiples of CELL_PX, for the long side LONG with the aspect kept, or None
    when the image is too small to describe."""
    if max(h, w) < MIN_LONG:
        return None
    s = LONG / max(h, w)
    th = min(LONG, max(CELL_PX, int(round(h * s / CELL_PX)) * CELL_PX))
    tw = min(LONG, max(CELL_PX, int(round(w * s / CELL_PX)) * CELL_PX))
    return th, tw


def _resize(rgb, th, tw):
    import cv2  # hook-allow: lazy — keeps module import minimal
    h, w = rgb.shape[:2]
    if (h, w) == (th, tw):
        return np.ascontiguousarray(rgb)
    interp = cv2.INTER_AREA if th * tw < h * w else cv2.INTER_CUBIC
    return cv2.resize(rgb, (tw, th), interpolation=interp)


def prepare(rgb, size=None):
    """RGB uint8 (h, w, 3) -> uint8 (th, tw, 3) at index scale, or None if too small.
    Normalisation happens on the device (uint8 buffers keep the decode pipeline light)."""
    size = size or target_size(*rgb.shape[:2])
    return None if size is None else _resize(rgb, *size)


# ------------------------------------------------------------------------------- the model

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _forward_cells(embedder, batch_u8, gpu_lock, stats=None, pooled=True):
    """uint8 (B, th, tw, 3) of ONE shape -> float32 numpy (B, n_cells, TOKEN_DIM) (pooled=True) or
    token-level (B, gh*gw, TOKEN_DIM). Takes `gpu_lock` around the forward + host copy only."""
    import torch  # hook-allow: lazy — heavy
    import torch.nn.functional as F  # hook-allow: lazy — heavy
    from .embedders import _autocast_ctx  # hook-allow: lazy — pulls torch
    B, th, tw, _ = batch_u8.shape
    gh, gw = th // PATCH, tw // PATCH
    t_wait = time.perf_counter()
    with gpu_lock:
        t_in = time.perf_counter()
        x = torch.from_numpy(batch_u8).to(embedder.device).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
        with torch.no_grad(), _autocast_ctx(embedder.device):
            out = embedder.model.forward_features((x - mean) / std)
        tok = (out['x_norm_patchtokens'] if isinstance(out, dict) else out).float()   # (B, gh*gw, C)
        if pooled:
            tok = F.avg_pool2d(tok.reshape(B, gh, gw, -1).permute(0, 3, 1, 2), POOL)
            tok = tok.flatten(2).transpose(1, 2)
        cells = tok.cpu().numpy()
        t_out = time.perf_counter()
    if stats is not None:
        stats['lock_wait_seconds'] = stats.get('lock_wait_seconds', 0.0) + (t_in - t_wait)
        stats['gpu_seconds'] = stats.get('gpu_seconds', 0.0) + (t_out - t_in)
    return cells


def _batch_size():
    return max(1, int(os.environ.get('MEDIA_PATTERN_DINO_BATCH', 16)))


def describe_files(embedder, sources, stats=None, gpu_lock=None, raw=False):
    """Index side. `sources`: str paths or encoded-image bytes. Returns (results, errors) aligned
    with `sources`: results[i] = int8 (n, dim) cells ((0, dim) = decoded but too small), or float32
    (n, TOKEN_DIM) pre-projection cells when raw=True; None + errors[i] (str) on failure.

    Decode+resize run in the embedder's thread pool a few batches ahead; images of equal grid shape
    are batched into one forward (most photos share 3-4 aspect ratios); the GPU lock is held per
    forward only. `stats` accumulates decode_seconds / gpu_seconds / lock_wait_seconds."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    from . import compute
    from .embedders import _decode_rgb
    lock = compute.GPU_LOCK if gpu_lock is None else gpu_lock
    st = stats if stats is not None else {}
    for k in ('decode_seconds', 'gpu_seconds', 'lock_wait_seconds'):
        st.setdefault(k, 0.0)
    st_lock = threading.Lock()
    bs = _batch_size()
    n = len(sources)
    results, errors = [None] * n, [None] * n
    d = TOKEN_DIM if raw else dim()
    if not raw and not projection_ready():
        raise RuntimeError('pattern_dino projection missing (data/pattern_dino_proj.npz)')
    empty = np.zeros((0, d), np.float32 if raw else np.int8)

    def decode(i):
        t = time.perf_counter()
        try:
            src = sources[i]
            img = _decode_rgb(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, 2 * LONG)
            return prepare(np.asarray(img))
        finally:
            with st_lock:
                st['decode_seconds'] += time.perf_counter() - t

    buckets, buffered = {}, [0]

    def flush(shape):
        items = buckets.pop(shape)
        buffered[0] -= len(items)
        for s in range(0, len(items), bs):
            part = items[s:s + bs]
            try:
                cells = _forward_cells(embedder, np.stack([a for _, a in part]), lock, st)
            except Exception as exc:  # hook-allow: broad — a bad batch must not sink the whole call
                for i, _ in part:
                    errors[i] = f'forward failed: {exc}'
                continue
            for (i, _), c in zip(part, cells):
                results[i] = c if raw else quantize(project(c))

    window = bs * 3
    pending = deque()

    def drain_one():
        i, fut = pending.popleft()
        try:
            arr = fut.result()
        except Exception as exc:  # hook-allow: broad — per-image decode resilience, reported as failed
            errors[i] = str(exc)
            return
        if arr is None:
            results[i] = empty
            return
        buckets.setdefault(arr.shape, []).append((i, arr))
        buffered[0] += 1
        if len(buckets[arr.shape]) >= bs:
            flush(arr.shape)
        elif buffered[0] >= 4 * bs:          # many distinct shapes: don't hoard decoded pixels
            flush(max(buckets, key=lambda k: len(buckets[k])))

    workers = max(1, int(getattr(embedder, 'num_workers', 4) or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(n):
            pending.append((i, pool.submit(decode, i)))
            while len(pending) >= window:
                drain_one()
        while pending:
            drain_one()
    for shape in list(buckets):
        flush(shape)
    return results, errors


# ------------------------------------------------------------------------------------ query

def query_raw(embedder, rgb, bbox, rots=(0,), gpu_lock=None):
    """Raw (pre-projection) query vectors for a drawn box (`bbox` px of the upright `rgb`).
    Returns [(kind, scale, rot, float32 (TOKEN_DIM,))] with kind 'mask' (scale 0, rot 0) or 'crop';
    [] when the box is too small / the image too small. Cheap enough per request (<=1 + 3*len(rots)
    batch-1 forwards)."""
    from . import compute
    lock = compute.GPU_LOCK if gpu_lock is None else gpu_lock
    size = target_size(*rgb.shape[:2])
    if size is None:
        return []
    th, tw = size
    frame = _resize(rgb, th, tw)
    sx, sy = tw / rgb.shape[1], th / rgb.shape[0]
    x1, y1 = max(0.0, bbox[0] * sx), max(0.0, bbox[1] * sy)
    x2, y2 = min(float(tw), bbox[2] * sx), min(float(th), bbox[3] * sy)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return []
    out = []
    # mask: dense tokens of the whole frame, averaged under the box (overlap-area weights)
    tok = _forward_cells(embedder, frame[None], lock, pooled=False)[0].reshape(th // PATCH, tw // PATCH, -1)
    w = box_token_weights(th // PATCH, tw // PATCH, (x1, y1, x2, y2))
    if w.sum() >= 0.5:
        out.append(('mask', 0.0, 0, ((w[:, :, None] * tok).sum((0, 1)) / w.sum()).astype(np.float32)))
    # crop: the region alone, at several zooms / rotations, mean of all its tokens
    crop = frame[int(y1):max(int(y1) + 1, int(np.ceil(y2))), int(x1):max(int(x1) + 1, int(np.ceil(x2)))]
    ch, cw = crop.shape[:2]
    seen = set()
    for s in CROP_SCALES:
        h2 = int(min(LONG, max(MIN_CROP, round(ch * s / PATCH) * PATCH)))
        w2 = int(min(LONG, max(MIN_CROP, round(cw * s / PATCH) * PATCH)))
        base = _resize(crop, h2, w2)
        for k in rots:
            if (h2, w2, k % 4) in seen:
                continue
            seen.add((h2, w2, k % 4))
            im = np.ascontiguousarray(np.rot90(base, k))
            t = _forward_cells(embedder, im[None], lock, pooled=False)[0]
            out.append(('crop', float(s), int(k) % 4, t.mean(0).astype(np.float32)))
    return out


def box_token_weights(gh, gw, box):
    """(gh, gw) float32: fraction of each PATCH x PATCH token square covered by `box` (px, x1 y1 x2 y2)."""
    x1, y1, x2, y2 = box
    xs = np.arange(gw) * PATCH
    ys = np.arange(gh) * PATCH
    ox = np.clip(np.minimum(xs + PATCH, x2) - np.maximum(xs, x1), 0, PATCH)
    oy = np.clip(np.minimum(ys + PATCH, y2) - np.maximum(ys, y1), 0, PATCH)
    return (np.outer(oy, ox) / float(PATCH * PATCH)).astype(np.float32)


def select_query(raw_vecs, mode='both', rots=None, scales=None):
    """Filter query_raw output by mode ('mask'|'crop'|'both'), rotations, crop scales ->
    float32 (S, TOKEN_DIM) raw vectors, or None when nothing is left. A 'mask'-only query whose
    box was too small for a mask vector falls back to the crops."""
    if mode not in QUERY_MODES:
        raise ValueError(f'mode {mode!r} not in {QUERY_MODES}')

    def pick(kinds):
        return [v for kind, s, r, v in raw_vecs
                if kind in kinds and (kind == 'mask' or ((rots is None or r in {x % 4 for x in rots})
                                                         and (scales is None or s in scales)))]
    vecs = pick({'mask'} if mode == 'mask' else {'crop'} if mode == 'crop' else {'mask', 'crop'})
    if not vecs and mode == 'mask':
        vecs = pick({'crop'})
    return np.stack(vecs).astype(np.float32) if vecs else None


def describe_query(embedder, rgb, bbox, mode='both', rots=(0,), gpu_lock=None):
    """Projected, unit-norm float32 (S, dim) query vectors, or None when the box can't be described."""
    sel = select_query(query_raw(embedder, rgb, bbox, rots, gpu_lock), mode)
    return None if sel is None else project(sel)


# --------------------------------------------------------------------------------- scoring

def score_cells(matrix, counts, queries):
    """Per-image best cosine over cells and query vectors for a chunk (same contract/maths as
    pattern_deep.score_windows): matrix int8 (sum(counts), dim), counts (k,), queries float32
    (S, dim) -> float32 (k,), -1 for images with no cells."""
    from .pattern_deep import score_windows
    return score_windows(matrix, counts, queries)


def fuse_scores(components):
    """Weighted z-score fusion of per-image scores from several indexes.
    components: [(file_ids int64[k], scores float32[k], weight)], scores <= -1 meaning "no
    descriptor" (ignored). Each component is z-scored over its own valid scores (the cosine
    scales of different spaces differ), then an image's fused score is the weight-normalised sum
    over the components that cover it. Returns (file_ids, fused) sorted by file_id."""
    live = []
    for fids, sc, wgt in components:
        fids, sc = np.asarray(fids, np.int64), np.asarray(sc, np.float32)
        ok = sc > -1.0
        if wgt > 0 and ok.any():
            v = sc[ok]
            live.append((fids[ok], (v - v.mean()) / (v.std() + 1e-6), float(wgt)))
    if not live:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    union = np.unique(np.concatenate([f for f, _, _ in live]))
    acc, wsum = np.zeros(len(union), np.float32), np.zeros(len(union), np.float32)
    for fids, z, wgt in live:
        idx = np.searchsorted(union, fids)
        acc[idx] += wgt * z
        wsum[idx] += wgt
    return union, acc / wsum
