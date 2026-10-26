"""E3 dense multi-scale texture-window descriptors (classical, CPU-only).

Raw per-window feature (D=46): 3 x riu2-LBP(8, R=1,2,3) hist (10 bins each, sqrt/Hellinger)
 + Gabor quadrature energy at 2 wavelengths (log energy + 4 orientation fractions sorted
 => rotation invariant) + 6 Lab moments. Windows are 48 px on a 16-px cell grid (stride 32)
 over a scale pyramid of the image (base: long side <= 1024; ratio sqrt2 down), so a query
 crop described at its own pixel scale lives in the same frequency band as the window it
 should match.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

cv2.setNumThreads(1)
CELL = 16
WIN_CELLS = 3          # window = 48 px
STRIDE_CELLS = 2       # stride 32 px
BASE_LONG = 1024
LEVEL_RATIO = 2 ** -0.5
RADII = (1, 2, 3)
LAMBDAS = (4.0, 4.0)   # 2nd band = same kernel on a 2x-downsampled image (~lambda 8)
N_ORI = 4
DIM_LBP, DIM_GAB, DIM_COL = 30, 10, 6
DIM = DIM_LBP + DIM_GAB + DIM_COL
BLOCKS = {'lbp': slice(0, 30), 'gab': slice(30, 40), 'col': slice(40, 46)}


def _riu2_lut():
    lut = np.zeros(256, np.uint8)
    for c in range(256):
        bits = [(c >> i) & 1 for i in range(8)]
        u = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        lut[c] = sum(bits) if u <= 2 else 9
    return lut


LUT = _riu2_lut()
_OFFS = {R: [(int(round(-R * np.sin(2 * np.pi * k / 8))), int(round(R * np.cos(2 * np.pi * k / 8))))
             for k in range(8)] for R in RADII}


def _gabor_bank():
    bank = []
    for lam in LAMBDAS:
        sigma = 0.56 * lam
        half = int(round(2.5 * sigma))
        ys, xs = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
        ks = []
        for o in range(N_ORI):
            th = np.pi * o / N_ORI
            u = xs * np.cos(th) + ys * np.sin(th)
            ev = g * np.cos(2 * np.pi * u / lam)
            od = g * np.sin(2 * np.pi * u / lam)
            ev -= g * (ev.sum() / g.sum())  # zero DC
            ev /= np.abs(ev).sum(); od /= np.abs(od).sum()
            ks.append((ev.astype(np.float32), od.astype(np.float32)))
        bank.append(ks)
    return bank


BANK = _gabor_bank()


def load_rgb(path_or_bytes):
    import io
    im = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
    im = ImageOps.exif_transpose(im).convert('RGB')
    return np.asarray(im)


def base_resize(rgb, bbox=None):
    """Cap long side at BASE_LONG; returns (rgb, bbox scaled)."""
    h, w = rgb.shape[:2]
    s = min(1.0, BASE_LONG / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        if bbox is not None:
            bbox = [v * s for v in bbox]
    return rgb, bbox


def _cell_sum(a, nh, nw):
    """a: (H,W,...) cropped to cells -> (nh,nw,...) sums per CELL x CELL block."""
    sh = (nh, CELL, nw, CELL) + a.shape[2:]
    return a.reshape(sh).sum(axis=(1, 3))


def _window_sum(c):
    nh, nw = c.shape[:2]
    wh, ww = nh - WIN_CELLS + 1, nw - WIN_CELLS + 1
    out = 0
    for di in range(WIN_CELLS):
        for dj in range(WIN_CELLS):
            out = out + c[di:di + wh, dj:dj + ww]
    return out[::STRIDE_CELLS, ::STRIDE_CELLS]


def describe_level(rgb):
    """rgb uint8 (H,W,3) -> (raw features (n, DIM) float32, window top-left xy (n,2)) or None."""
    H, W = rgb.shape[:2]
    nh, nw = H // CELL, W // CELL
    if nh < WIN_CELLS or nw < WIN_CELLS:
        return None
    Hc, Wc = nh * CELL, nw * CELL
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    grey = lab[..., 0] * 2.55
    feats = []
    # --- LBP riu2, three radii
    pad = max(RADII)
    for R in RADII:
        g = cv2.GaussianBlur(grey, (0, 0), 0.6 + 0.35 * R) if R > 1 else cv2.GaussianBlur(grey, (0, 0), 0.7)
        gp = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        ctr = gp[pad:pad + Hc, pad:pad + Wc]
        code = np.zeros((Hc, Wc), np.uint8)
        for k, (dy, dx) in enumerate(_OFFS[R]):
            nb = gp[pad + dy:pad + dy + Hc, pad + dx:pad + dx + Wc]
            code |= ((nb >= ctr - 1.0).astype(np.uint8) << k)
        lab_map = LUT[code]
        idx = (np.arange(nh)[:, None, None, None] * nw + np.arange(nw)[None, None, :, None]) \
            * 10 + lab_map.reshape(nh, CELL, nw, CELL)
        cnt = np.bincount(idx.ravel(), minlength=nh * nw * 10).reshape(nh, nw, 10).astype(np.float32)
        ws = _window_sum(cnt)
        ws /= ws.sum(-1, keepdims=True) + 1e-6
        feats.append(np.sqrt(ws))
    # --- Gabor energy
    gz = grey[:Hc, :Wc] - grey[:Hc, :Wc].mean()
    for bi, ks in enumerate(BANK):
        src = gz if bi == 0 else cv2.resize(gz, (Wc // 2, Hc // 2), interpolation=cv2.INTER_AREA)
        cs = CELL if bi == 0 else CELL // 2
        en = []
        for ev, od in ks:
            re = cv2.filter2D(src, cv2.CV_32F, ev, borderType=cv2.BORDER_REFLECT)
            im = cv2.filter2D(src, cv2.CV_32F, od, borderType=cv2.BORDER_REFLECT)
            e = re * re + im * im
            en.append(e.reshape(nh, cs, nw, cs).sum(axis=(1, 3)) / float(cs * cs) * float(CELL * CELL))
        en = np.stack(en, -1)
        ws = _window_sum(en) / float((WIN_CELLS * CELL) ** 2)
        tot = ws.sum(-1, keepdims=True)
        frac = np.sort(ws / (tot + 1e-6), -1)[..., ::-1]
        feats.append(np.concatenate([0.5 * np.log(tot + 1.0), frac], -1))
    # --- Lab moments
    l = lab[:Hc, :Wc]
    m1 = _window_sum(_cell_sum(l, nh, nw)) / float((WIN_CELLS * CELL) ** 2)
    m2 = _window_sum(_cell_sum(l * l, nh, nw)) / float((WIN_CELLS * CELL) ** 2)
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0) + 1e-3)
    feats.append(np.concatenate([m1, sd], -1))
    f = np.concatenate(feats, -1)
    n_i, n_j = f.shape[:2]
    ys, xs = np.mgrid[0:n_i, 0:n_j]
    xy = np.stack([xs.ravel() * STRIDE_CELLS * CELL, ys.ravel() * STRIDE_CELLS * CELL], 1)
    return f.reshape(-1, f.shape[-1]).astype(np.float32), xy.astype(np.int16)


def pyramid(rgb, ratio=LEVEL_RATIO, min_side=WIN_CELLS * CELL, max_levels=9):
    h, w = rgb.shape[:2]
    s = 1.0
    for _ in range(max_levels):
        nh, nw = round(h * s), round(w * s)
        if min(nh, nw) < min_side:
            break
        yield s, (rgb if s == 1.0 else cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA))
        s *= ratio


def describe_image(rgb):
    """Index side: all pyramid levels -> (feats (N,DIM) f32, meta (N,3) int16 [level, x, y])."""
    F, M = [], []
    for li, (s, im) in enumerate(pyramid(rgb)):
        r = describe_level(im)
        if r is None:
            continue
        f, xy = r
        F.append(f)
        M.append(np.concatenate([np.full((len(f), 1), li, np.int16), xy], 1))
    if not F:
        return np.zeros((0, DIM), np.float32), np.zeros((0, 3), np.int16)
    return np.concatenate(F), np.concatenate(M)


def describe_query(rgb, bbox, scales=(1.0, 0.71, 0.5, 0.35)):
    """Query side: crop at native (base) pixel scale, re-described at several scales.
    Returns {scale: raw feats (n, DIM)} (all windows inside the crop at that scale)."""
    rgb, bbox = base_resize(rgb, bbox)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = rgb[y1:y2, x1:x2]
    out = {}
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 8:
        return out
    if min(ch, cw) < WIN_CELLS * CELL:      # tiny crop: upsample once so one window fits
        f = (WIN_CELLS * CELL + 2) / min(ch, cw)
        crop = cv2.resize(crop, (round(cw * f), round(ch * f)), interpolation=cv2.INTER_CUBIC)
    for s in scales:
        im = crop if s == 1.0 else cv2.resize(
            crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
        r = describe_level(im)
        if r is not None:
            out[s] = r[0]
    return out


def index_one(path):
    """Pool worker (must live in an importable module)."""
    import sys
    try:
        rgb, _ = base_resize(load_rgb(path))
        f, m = describe_image(rgb)
    except Exception as e:  # undecodable image keeps zero windows
        print('e3: failed', path, e, file=sys.stderr)
        f, m = np.zeros((0, DIM), np.float32), np.zeros((0, 3), np.int16)
    return f.astype(np.float16), m
