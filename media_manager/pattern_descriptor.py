"""Classical texture + colour descriptor for find-by-pattern.

CPU-only (numpy + OpenCV, both already deps). Where region-search uses CLIP to find
*what* is in a crop (a leaf, a wall), this captures the *pattern/texture* of a crop —
repeating micro-structure and colour distribution — and ignores semantics, so it can
find "that wallpaper" or "that fabric weave". Three blocks:

  - uniform Local Binary Patterns histogram (59 bins) — illumination-invariant micro-texture
  - HSV colour histogram (8x8 hue-sat + 8 value = 72 bins) — the pattern's colours
  - Gabor filter-bank energies (4 orientations x 2 wavelengths, mean+std = 16) — directional frequency

The blocks are concatenated and the whole vector is L2-normalised, so retrieval is a
dot product (cosine) — same as everything else in the app. Deterministic. Shared by the
worker (which does the compute) and the web (local fallback), so both must agree — bump
ALGO_TAG on any change and the search ignores descriptors with a different tag/length.
"""

import numpy as np
import cv2

ALGO_TAG = 'lbp59+hsv72+gabor16-v1'
_SIZE = 128   # every crop is resized to this before describing (tile-size independence)


def _build_uniform_lut():
    """256 -> 59 map: each of the 58 'uniform' 8-bit LBP codes (<=2 circular 0/1
    transitions) gets its own bin; all non-uniform codes share bin 58."""
    lut = np.empty(256, np.int16)
    nxt = 0
    for v in range(256):
        bits = [(v >> i) & 1 for i in range(8)]
        trans = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if trans <= 2:
            lut[v] = nxt
            nxt += 1
        else:
            lut[v] = 58
    return lut


_UNIFORM_LUT = _build_uniform_lut()

# Gabor kernels, precomputed once: 2 wavelengths x 4 orientations.
_GABOR = []
for _lam in (4.0, 8.0):
    for _theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        _GABOR.append(cv2.getGaborKernel((15, 15), _lam * 0.5, _theta, _lam, 0.5, 0.0).astype(np.float32))

DIM = 59 + 72 + 16   # 147


def _lbp_hist(gray):
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, np.uint8)
    for i, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                  (1, 1), (1, 0), (1, -1), (0, -1)]):
        neigh = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((neigh >= c).astype(np.uint8) << i)
    mapped = _UNIFORM_LUT[code.ravel()]
    hist = np.bincount(mapped, minlength=59).astype(np.float32)
    s = hist.sum()
    return hist / s if s else hist


def _color_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).astype(np.float32).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).astype(np.float32).ravel()
    hs /= (hs.sum() or 1.0)
    v /= (v.sum() or 1.0)
    return np.concatenate([hs, v])


def _gabor_feats(gray):
    g = gray.astype(np.float32) / 255.0
    feats = []
    for k in _GABOR:
        r = cv2.filter2D(g, cv2.CV_32F, k)
        feats.append(float(r.mean()))
        feats.append(float(r.std()))
    feats = np.asarray(feats, np.float32)
    n = np.linalg.norm(feats)
    return feats / n if n else feats


def compute_descriptor(bgr):
    """L2-normalised float32 descriptor for a BGR crop, or None if empty."""
    if bgr is None or bgr.size == 0:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(bgr, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    vec = np.concatenate([_lbp_hist(gray), _color_hist(crop), _gabor_feats(gray)]).astype(np.float32)
    n = np.linalg.norm(vec)
    return (vec / n).astype(np.float32) if n else vec


def decode_bgr(image_bytes):
    """Decode encoded image bytes to a BGR array, or None."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def descriptors_for_boxes(bgr, boxes):
    """boxes: list of [x1,y1,x2,y2] pixel boxes (None entry = whole image). Returns a
    list of descriptors (or None per bad crop), aligned to `boxes`."""
    h, w = bgr.shape[:2]
    out = []
    for box in boxes:
        if box is None:
            crop = bgr
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
            crop = bgr[y1:y2, x1:x2]
        out.append(compute_descriptor(crop))
    return out
