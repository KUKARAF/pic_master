"""E4 helpers: features, periodicity test, NCC unit-cell verifier, SIFT/MAGSAC verifier."""
import cv2
import numpy as np

cv2.setNumThreads(1)
SIFT_SIDE = 640        # working size for SIFT (long side)
NCC_SIDE = 128         # working size for template matching (long side)
N_KP = 600
_sift = {}


def sift():
    import threading
    t = threading.get_ident()
    if t not in _sift:
        _sift[t] = cv2.SIFT_create(nfeatures=N_KP)
    return _sift[t]


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def resize_long(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1.0 and side == SIFT_SIDE:
        s = 1.0          # never upsample the corpus
    if s == 1.0:
        return img, 1.0
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def rootsift(gray, mask=None):
    kps, des = sift().detectAndCompute(gray, mask)
    if des is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    des = des / (des.sum(1, keepdims=True) + 1e-7)
    des = np.sqrt(des)                                   # RootSIFT, unit L2
    xy = np.array([k.pt for k in kps], np.float32)
    return xy, np.clip(des * 255.0 + 0.5, 0, 255).astype(np.uint8)


def image_features(path):
    """-> (ncc_gray uint8 (long side<=224), xy float32 (n,2), des uint8 (n,128))"""
    g = load_gray(path)
    if g is None:
        return np.zeros((8, 8), np.uint8), np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    gs, _ = resize_long(g, SIFT_SIDE)
    xy, des = rootsift(gs)
    gn, _ = resize_long(gs, NCC_SIDE)
    return gn, xy, des


# ------------------------------------------------------------- query preparation

def prepare_query(path, bbox):
    """Same normalisation as the corpus: long side 640 (never up), crop bbox in that frame."""
    g = load_gray(path)
    h0, w0 = g.shape
    gs, s = resize_long(g, SIFT_SIDE)
    x1, y1, x2, y2 = [int(round(v * s)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gs.shape[1], max(x2, x1 + 16)), min(gs.shape[0], max(y2, y1 + 16))
    return gs[y1:y2, x1:x2], s, NCC_SIDE / max(gs.shape)


def periodicity(crop):
    """Unbiased autocorrelation. -> (strength, period_px in crop coords or None)."""
    h, w = crop.shape
    s = 128.0 / max(h, w)
    g = cv2.resize(crop, (max(16, round(w * s)), max(16, round(h * s))), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    g -= g.mean()
    if g.std() < 1e-3:
        return 0.0, None
    H, W = g.shape
    F = np.fft.rfft2(g, s=(2 * H, 2 * W))
    ac = np.fft.irfft2(F * np.conj(F), s=(2 * H, 2 * W))
    m = np.ones_like(g)
    Fm = np.fft.rfft2(m, s=(2 * H, 2 * W))
    cnt = np.fft.irfft2(Fm * np.conj(Fm), s=(2 * H, 2 * W))
    ac = np.fft.fftshift(ac / np.maximum(cnt, 1.0))
    ac /= ac[H, W]
    cy, cx = H, W
    # central lobe = region around the origin above 0.2, grown a bit
    lobe = (ac > 0.2).astype(np.uint8)
    _, lab = cv2.connectedComponents(lobe)
    lobe = (lab == lab[cy, cx]).astype(np.uint8)
    lobe = cv2.dilate(lobe, np.ones((5, 5), np.uint8))
    yy, xx = np.mgrid[:2 * H, :2 * W]
    dy, dx = yy - cy, xx - cx
    valid = (np.abs(dy) <= 0.45 * H) & (np.abs(dx) <= 0.45 * W) & (lobe == 0)
    mx = cv2.dilate(ac, np.ones((5, 5), np.uint8))
    peaks = valid & (ac >= mx - 1e-9) & (ac > 0.1)
    if not peaks.any():
        return 0.0, None
    vals = np.where(peaks, ac, -1)
    i = np.unravel_index(np.argmax(vals), vals.shape)
    strength = float(vals[i])
    period = float(np.hypot(dy[i], dx[i])) / s
    return strength, period


ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
SCALES = (0.35, 0.5, 0.7, 1.0, 1.4)


def make_templates(crop, period, periodic):
    """Rotated, multi-scale NCC templates at NCC_SIDE resolution. `crop` is in SIFT_SIDE frame."""
    h, w = crop.shape
    side = min(h, w)
    if periodic and period:
        side = int(min(side, max(24, 2.4 * period)))
    y0, x0 = (h - side) // 2, (w - side) // 2
    base = crop[y0:y0 + side, x0:x0 + side]
    out = []
    # tile in ncc frame: crop (SIFT frame) -> NCC frame factor is norm_f
    for ang in ANGLES:
        M = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
        rot = cv2.warpAffine(base, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        inner = int(side / 1.4142)
        o = (side - inner) // 2
        rot = rot[o:o + inner, o:o + inner]
        for sc in SCALES:
            out.append((ang, sc, rot))
    return out


def ncc_prepare(templates, ncc_factor, cap=40):
    """Resize each template to final pixel size (relative to the NCC frame)."""
    res = []
    for ang, sc, t in templates:
        # native size in ncc frame = t * ncc_factor ; cap it, then apply relative scale
        size = max(t.shape) * ncc_factor
        size = min(size, cap) * sc
        n = int(round(size))
        if n < 10:
            continue
        tt = cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA if n < t.shape[0] else cv2.INTER_LINEAR)
        if tt.std() < 3:
            continue
        res.append(tt)
    return res


def _ncc_max(img, tpls):
    best = -1.0
    H, W = img.shape
    for t in tpls:
        if t.shape[0] >= H or t.shape[1] >= W:
            continue
        m = float(cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED).max())
        if m > best:
            best = m
    return best


def coarse_templates(tpls):
    out = []
    for t in tpls:
        n = t.shape[0] // 2
        if n >= 8:
            tt = cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA)
            if tt.std() >= 2:
                out.append(tt)
    return out


def ncc_score(img, tpls, ctpls, gate=0.45):
    """Coarse-to-fine: half-resolution pass over all templates; only images whose coarse peak
    passes `gate` get the full-resolution pass."""
    small = cv2.resize(img, (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    c = _ncc_max(small, ctpls)
    if c < gate:
        return max(c, 0.0)
    return max(c, _ncc_max(img, tpls), 0.0)


class Matcher:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_L2)


def ransac_inliers(qxy, qdesf, ixy, ides, ratio=0.85, thr=5.0, min_matches=8):
    """qdesf: float32 RootSIFT (unit L2) of the query; ides: uint8 RootSIFT of the image."""
    if len(qxy) < 4 or len(ixy) < 4:
        return 0
    idf = ides.astype(np.float32) * (1.0 / 255.0)
    sim = qdesf @ idf.T                                   # cosine; dist^2 = 2 - 2 sim
    if sim.shape[1] < 2:
        return 0
    top2 = np.argpartition(-sim, 1, axis=1)[:, :2]
    s2 = np.take_along_axis(sim, top2, 1)
    o = np.argsort(-s2, 1)
    top2 = np.take_along_axis(top2, o, 1)
    s2 = np.take_along_axis(s2, o, 1)
    d1 = np.sqrt(np.maximum(2 - 2 * s2[:, 0], 0))
    d2 = np.sqrt(np.maximum(2 - 2 * s2[:, 1], 0))
    good = d1 < ratio * d2
    if good.sum() < min_matches:
        return 0
    src = qxy[good]
    dst = ixy[top2[good, 0]]
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, thr, maxIters=1000, confidence=0.99)
    if H is None or mask is None:
        return 0
    n = int(mask.sum())
    return n if n >= min_matches else 0
