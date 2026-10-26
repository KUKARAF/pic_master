"""E1 building blocks: DINOv2 dense token maps -> multi-scale window embeddings;
RootSIFT extraction, ratio+mutual matching, MAGSAC homography verification."""
import numpy as np
from PIL import Image, ImageOps

TOKEN_LONG = 448          # long side of the image fed to DINOv2 (multiple of 14)
PATCH = 14
SIFT_LONG = 800           # long side for database SIFT
SIFT_NFEAT = 600
WINDOW_FRACS = (1.0, 0.5, 0.34)
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert('RGB')
        im.load()
    return im


def fit_dims(w, h, long_side, patch=PATCH, min_tokens=2):
    """Aspect-preserving size whose sides are multiples of the patch size."""
    s = long_side / max(w, h)
    gw = max(min_tokens, int(round(w * s / patch)))
    gh = max(min_tokens, int(round(h * s / patch)))
    return gw, gh


class Dino:
    def __init__(self, hub='dinov2_vits14_reg'):
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.model = torch.hub.load('facebookresearch/dinov2', hub).eval()

    def tokens(self, pil, long_side=TOKEN_LONG):
        """-> (patch tokens [gh, gw, D] float32 (post-norm), cls [D])"""
        torch = self.torch
        gw, gh = fit_dims(pil.width, pil.height, long_side)
        im = pil.resize((gw * PATCH, gh * PATCH), Image.BICUBIC)
        a = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
        with torch.inference_mode():
            out = self.model.forward_features(torch.from_numpy(a)[None])
        pt = out['x_norm_patchtokens'][0].numpy().reshape(gh, gw, -1)
        return pt, out['x_norm_clstoken'][0].numpy()


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def window_boxes(gh, gw, fracs=WINDOW_FRACS):
    """(y0,y1,x0,x1) token windows; window = frac of each side, stride = half window."""
    out = []
    for f in fracs:
        h = max(1, int(round(f * gh)))
        w = max(1, int(round(f * gw)))
        n = 1 if f >= 0.999 else max(2, int(round(2.0 / f)) - 1)
        ys = sorted(set(int(round(v)) for v in np.linspace(0, gh - h, n)))
        xs = sorted(set(int(round(v)) for v in np.linspace(0, gw - w, n)))
        out += [(y, y + h, x, x + w) for y in ys for x in xs]
    return out


def window_embeddings(tokens, fracs=WINDOW_FRACS):
    """Mean-pooled (via integral image), L2-normalised window descriptors [N, D]."""
    gh, gw, d = tokens.shape
    ii = np.zeros((gh + 1, gw + 1, d), np.float64)
    ii[1:, 1:] = tokens.astype(np.float64).cumsum(0).cumsum(1)
    wins = window_boxes(gh, gw, fracs)
    e = np.stack([ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0] for y0, y1, x0, x1 in wins])
    return _l2(e).astype(np.float32), wins


def pool_box(tokens, box_frac):
    """Mean-pooled descriptor of the token cells under a fractional box (x0,y0,x1,y1)."""
    gh, gw, _ = tokens.shape
    x0, y0, x1, y1 = box_frac
    xa = min(gw - 1, int(np.floor(x0 * gw))); xb = max(xa + 1, int(np.ceil(x1 * gw)))
    ya = min(gh - 1, int(np.floor(y0 * gh))); yb = max(ya + 1, int(np.ceil(y1 * gh)))
    return _l2(tokens[ya:yb, xa:xb].reshape(-1, tokens.shape[-1]).mean(0))


# ------------------------------------------------------------------ SIFT / geometry

def _cv2():
    import cv2
    cv2.setNumThreads(2)
    return cv2


def rootsift(pil, nfeat=SIFT_NFEAT, long_side=SIFT_LONG, mask_box=None):
    """-> (pts [N,2] in coordinates of `pil`, rootsift desc [N,128] float32, sizes [N]).
    Image is resized so its long side is `long_side` (up to 3x upscale) for detection."""
    cv2 = _cv2()
    s = min(3.0, long_side / max(pil.width, pil.height))
    g = np.asarray(pil.convert('L'))
    if abs(s - 1) > 1e-3:
        g = cv2.resize(g, (max(8, int(round(pil.width * s))), max(8, int(round(pil.height * s)))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    sift = cv2.SIFT_create(nfeatures=nfeat)
    kps, desc = sift.detectAndCompute(g, None)
    if desc is None or not len(kps):
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], np.float32) / s
    desc = desc / np.maximum(desc.sum(1, keepdims=True), 1e-8)   # L1
    desc = np.sqrt(desc).astype(np.float32)                         # Hellinger => RootSIFT
    return pts, desc


def quant_desc(d):
    return np.round(d * 512.0).clip(0, 255).astype(np.uint8)   # unit L2 => entries <= 1


def dequant_desc(q):
    return q.astype(np.float32) / 512.0


def match_inliers(qp, qd, cp, cd, ratio=0.8, thr=4.0, mutual=True):
    """Ratio test (+mutual) then USAC_MAGSAC homography. -> (n_inliers, n_matches, H)"""
    cv2 = _cv2()
    if len(qd) < 4 or len(cd) < 4:
        return 0, 0, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fw = bf.knnMatch(qd, cd, k=2)
    good = {}
    for m in fw:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good[m[0].queryIdx] = m[0].trainIdx
    if mutual and good:
        bw = bf.knnMatch(cd, qd, k=1)
        good = {q: c for q, c in good.items() if bw[c] and bw[c][0].trainIdx == q}
    if len(good) < 4:
        return 0, len(good), None
    qi = np.fromiter(good.keys(), int); ci = np.fromiter(good.values(), int)
    src = qp[qi].astype(np.float64); dst = cp[ci].astype(np.float64)
    try:
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, thr, confidence=0.995, maxIters=2000)
    except cv2.error:
        return 0, len(good), None
    if H is None or mask is None:
        return 0, len(good), None
    # reject degenerate homographies (collapse / extreme perspective)
    det = np.linalg.det(H[:2, :2])
    if not np.isfinite(det) or abs(det) < 1e-4 or abs(det) > 1e4 or abs(H[2, 0]) > 5e-3 or abs(H[2, 1]) > 5e-3:
        return 0, len(good), H
    return int(mask.sum()), len(good), H
