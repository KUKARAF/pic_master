"""Instance-level region search: DINOv2 window prefilter + RootSIFT/MAGSAC geometric re-rank.

The CLIP tile index answers "what kind of thing is this crop" and so returns generic
same-category photos. This answers "find THIS object/region again":

  Index (per image): DINOv2 ViT-S/14-reg dense patch tokens (long side 448) mean-pooled
  into 35 windows (whole image, 1/2 and 1/3 of each side, stride half a window), L2-
  normalised and stored int8 (35 x 384 = 13 KB); plus up to SIFT_NFEAT RootSIFT features
  (long side 800) for verification (pts fp16 + desc uint8, ~40 KB at 300 features).

  Query: the drawn crop is embedded at 224 and 448 px plus the box's tokens inside the
  full-image token map ("ctx"); a photo's prefilter score is its best window cosine over
  those 3 vectors. The top N_CAND photos are verified: ratio+mutual RootSIFT matching of the
  crop's own features (so the match is restricted to the query box) and a USAC_MAGSAC
  homography. Final score = dino + W_INL * log1p(max(0, inliers - MIN_INL)).

Benchmark (bench/README.md): region mAP .646 (CLIP tiles) -> .720, hard-negative-above-
positive 22.7% -> 9.1%.

Shared by the worker (DINO + SIFT extraction) and the web (query SIFT + verification,
local fallback), so both must agree: bump ALGO_TAG on any change. torch loads lazily.
"""
import io
import os

import numpy as np

# SIFT budget per photo (~132 B each): 300 = ~40 KB/photo; 600 = ~79 KB and mAP .681 -> .715 on the
# bench. Part of ALGO_TAG, so changing it re-queues the library (worker and web must agree).
SIFT_NFEAT = int(os.environ.get('MEDIA_REGION_SIFT_NFEAT', '300'))
ALGO_TAG = f'dinov2s-win35i8+rsift{SIFT_NFEAT}-v1'
DIM = 384
TOKEN_LONG = 448          # long side fed to DINOv2 (multiple of 14)
PATCH = 14
SIFT_LONG = 800           # database SIFT long side
QUERY_SIFT_LONG = 640
WINDOW_FRACS = (1.0, 0.5, 0.34)
QUERY_LONGS = (224, 448)  # crop sizes embedded for the query (+ the "ctx" vector)
N_CAND = 200              # photos sent to geometric verification
MIN_INL = 8
W_INL = 1.0
_HUB = 'dinov2_vits14_reg'
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

_model = None


def _get_model():
    """DINOv2 ViT-S/14 with registers (torch.hub, same weights family as object_encoder.py)
    on the central compute device. object_encoder only exposes a fixed-224 CLS vector, so
    the dense token maps used here need their own forward."""
    global _model
    if _model is None:
        import torch  # hook-allow: lazy — optional heavy dep
        from . import compute
        model = torch.hub.load('facebookresearch/dinov2', _HUB)
        _model = (model.to(compute.torch_device()).eval(), torch)
    return _model


def load_pil(image_bytes):
    """Encoded bytes -> upright RGB PIL image (the frame query boxes are drawn in), or None."""
    from .formats import open_upright_rgb
    try:
        return open_upright_rgb(io.BytesIO(image_bytes))
    except Exception:  # hook-allow: broad — PIL raises many types on bad data; None = undecodable
        return None


def _fit_dims(w, h, long_side, min_tokens=2):
    s = long_side / max(w, h)
    return (max(min_tokens, int(round(w * s / PATCH))), max(min_tokens, int(round(h * s / PATCH))))


def tokens(pil, long_side=TOKEN_LONG):
    """Patch tokens [gh, gw, DIM] float32 (post-norm) of `pil` resized to long_side."""
    from PIL import Image  # hook-allow: lazy — keep module import numpy-only
    model, torch = _get_model()
    gw, gh = _fit_dims(pil.width, pil.height, long_side)
    im = pil.resize((gw * PATCH, gh * PATCH), Image.BICUBIC)
    a = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - _MEAN) / _STD
    dev = next(model.parameters()).device
    with torch.inference_mode():
        out = model.forward_features(torch.from_numpy(a)[None].to(dev))
    return out['x_norm_patchtokens'][0].float().cpu().numpy().reshape(gh, gw, -1)


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def window_boxes(gh, gw, fracs=WINDOW_FRACS):
    """(y0, y1, x0, x1) token windows; window = frac of each side, stride = half window."""
    out = []
    for f in fracs:
        h = max(1, int(round(f * gh)))
        w = max(1, int(round(f * gw)))
        n = 1 if f >= 0.999 else max(2, int(round(2.0 / f)) - 1)
        ys = sorted(set(int(round(v)) for v in np.linspace(0, gh - h, n)))
        xs = sorted(set(int(round(v)) for v in np.linspace(0, gw - w, n)))
        out += [(y, y + h, x, x + w) for y in ys for x in xs]
    return out


def window_embeddings(tok, fracs=WINDOW_FRACS):
    """Mean-pooled (integral image), L2-normalised window descriptors float32 [N, DIM]."""
    gh, gw, d = tok.shape
    ii = np.zeros((gh + 1, gw + 1, d), np.float64)
    ii[1:, 1:] = tok.astype(np.float64).cumsum(0).cumsum(1)
    e = np.stack([ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
                  for y0, y1, x0, x1 in window_boxes(gh, gw, fracs)])
    return _l2(e).astype(np.float32)


def pool_box(tok, box_frac):
    """Mean-pooled unit descriptor of the token cells under a fractional box (x0,y0,x1,y1)."""
    gh, gw, _ = tok.shape
    x0, y0, x1, y1 = box_frac
    xa = min(gw - 1, int(np.floor(x0 * gw))); xb = max(xa + 1, int(np.ceil(x1 * gw)))
    ya = min(gh - 1, int(np.floor(y0 * gh))); yb = max(ya + 1, int(np.ceil(y1 * gh)))
    return _l2(tok[ya:yb, xa:xb].reshape(-1, tok.shape[-1]).mean(0))


def quantize(vecs):
    """Unit-norm float32 (n, DIM) -> int8."""
    return np.clip(np.rint(vecs * 127.0), -127, 127).astype(np.int8)


# ------------------------------------------------------------------ SIFT / geometry

def rootsift(pil, nfeat=SIFT_NFEAT, long_side=SIFT_LONG):
    """-> (pts [N,2] in `pil` coordinates, RootSIFT desc [N,128] float32). The image is
    resized so its long side is `long_side` (up to 3x upscale) for detection."""
    import cv2  # hook-allow: lazy — keep module import numpy-only
    s = min(3.0, long_side / max(pil.width, pil.height))
    g = np.asarray(pil.convert('L'))
    if abs(s - 1) > 1e-3:
        g = cv2.resize(g, (max(8, int(round(pil.width * s))), max(8, int(round(pil.height * s)))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    kps, desc = cv2.SIFT_create(nfeatures=nfeat).detectAndCompute(g, None)
    if desc is None or not len(kps):
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], np.float32) / s
    desc = desc / np.maximum(desc.sum(1, keepdims=True), 1e-8)   # L1
    return pts, np.sqrt(desc).astype(np.float32)                  # Hellinger => RootSIFT


def quant_desc(d):
    return np.round(d * 512.0).clip(0, 255).astype(np.uint8)   # unit L2 => entries <= 1


def dequant_desc(q):
    return q.astype(np.float32) / 512.0


def match_inliers(qp, qd, cp, cd, ratio=0.8, thr=4.0):
    """Ratio test + mutual check, then USAC_MAGSAC homography -> inlier count (0 if none
    or the homography is degenerate)."""
    import cv2  # hook-allow: lazy — keep module import numpy-only
    if len(qd) < 4 or len(cd) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = {}
    for m in bf.knnMatch(qd, cd, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good[m[0].queryIdx] = m[0].trainIdx
    if good:
        bw = bf.knnMatch(cd, qd, k=1)
        good = {q: c for q, c in good.items() if bw[c] and bw[c][0].trainIdx == q}
    if len(good) < 4:
        return 0
    qi = np.fromiter(good.keys(), int)
    ci = np.fromiter(good.values(), int)
    try:
        H, mask = cv2.findHomography(qp[qi].astype(np.float64), cp[ci].astype(np.float64),
                                     cv2.USAC_MAGSAC, thr, confidence=0.995, maxIters=2000)
    except cv2.error:
        return 0
    if H is None or mask is None:
        return 0
    det = np.linalg.det(H[:2, :2])
    if not np.isfinite(det) or abs(det) < 1e-4 or abs(det) > 1e4 or abs(H[2, 0]) > 5e-3 or abs(H[2, 1]) > 5e-3:
        return 0
    return int(mask.sum())


# ------------------------------------------------------------------ index / query

def describe_image(pil):
    """Index side -> (windows int8 (n, DIM), sift pts float16 (k, 2) in SIFT-resized px,
    sift desc uint8 (k, 128))."""
    wins = quantize(window_embeddings(tokens(pil)))
    pts, desc = rootsift(pil)
    s = min(3.0, SIFT_LONG / max(pil.width, pil.height))
    return wins, (pts * s).astype(np.float16), quant_desc(desc)


def crop_box(pil, bbox):
    """Crop `pil` to the (clamped) drawn box."""
    W, H = pil.size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, x1 = max(0, min(x0, W - 1)), max(x0 + 1, min(x1, W))
    y0, y1 = max(0, min(y0, H - 1)), max(y0 + 1, min(y1, H))
    return pil.crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))), (x0 / W, y0 / H, x1 / W, y1 / H)


def describe_query(pil, bbox):
    """DINO query vectors float32 (3, DIM): crop@224, crop@448, box tokens in the full map."""
    crop, frac = crop_box(pil, bbox)
    vecs = [pool_box(tokens(crop, ls), (0, 0, 1, 1)) for ls in QUERY_LONGS]
    vecs.append(pool_box(tokens(pil), frac))
    return np.stack(vecs).astype(np.float32)


def query_sift(pil, bbox):
    """RootSIFT of the query crop only — restricts geometric matching to the drawn box."""
    return rootsift(crop_box(pil, bbox)[0], long_side=QUERY_SIFT_LONG)


def score_windows(matrix, counts, queries):
    """Prefilter score per image of a chunk: int8 windows (sum(counts), DIM), counts (k,),
    float32 queries (S, DIM) -> float32 (k,) best window cosine over all query vectors."""
    out = np.full(len(counts), -1.0, np.float32)
    if matrix.shape[0] == 0:
        return out
    sims = (matrix.astype(np.float32) @ queries.T).max(axis=1) / 127.0
    starts = np.cumsum(counts) - counts
    has = counts > 0
    out[has] = np.maximum.reduceat(sims, starts[has])
    return out


def fuse(dino, inliers):
    return dino + W_INL * np.log1p(max(0.0, inliers - MIN_INL))


def rerank(fids, scores, qp, qd, get_sift, n_cand=N_CAND):
    """Geometric re-rank: `fids`/`scores` are the prefilter ranking inputs (aligned arrays);
    the n_cand best are verified against the query features (qp, qd) and get the fused score,
    everyone else keeps the DINO score (always below the candidates'). get_sift(file_id) ->
    (pts f16, desc u8) or None. Returns (fused scores, {file_id: inliers})."""
    fused = np.asarray(scores, np.float32).copy()
    inliers = {}
    for i in np.argsort(-fused, kind='stable')[:n_cand]:
        sift = get_sift(int(fids[i]))
        inl = 0 if sift is None else match_inliers(qp, qd, sift[0].astype(np.float32), dequant_desc(sift[1]))
        inliers[int(fids[i])] = inl
        fused[i] = fuse(fused[i], inl)
    return fused, inliers
