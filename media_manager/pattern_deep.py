"""Native-scale deep texture windows for find-by-pattern (replaces the 147-d LBP/HSV/Gabor
tile descriptor in pattern_descriptor.py, which stays as the benchmark baseline).

Where region-search asks "what is this", this asks "does this surface look like that":
orderless statistics of early ResNet18 features (layer2 mean + std, layer3 mean) over
48-px windows. The image is capped at 512 px on the long side and described over a
sqrt(2) pyramid (<=5 levels), so a query crop taken at its own pixel scale lives in the
same frequency band as the library window it should match. Each window's raw 512-d
statistic is sqrt-compressed, standardised, block-weighted, whitened and PCA-reduced to
DIM dims by a FIXED projection fitted offline (data/pattern_deep_proj.npz, see
bench/fit_pattern_projection.py), L2-normalised and stored int8 — ~255 windows x 96 B =
~24 KB/image, so 100k photos fit in ~2.5 GB while retrieval stays a plain dot product.

Query: the crop is described at scales (1, .71, .5), each scale's windows mean-pooled to
one vector; an image scores the best (scale, window) cosine. Benchmark (bench/README.md):
pattern mAP .365 (old descriptor) -> .757.

Shared by the worker (compute) and the web (local fallback), so both must agree: bump
ALGO_TAG on any change. torch/timm import lazily — importing this module needs only numpy.
"""
import os

import numpy as np

_PROJ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pattern_deep_proj.npz')
DIM = int(np.load(_PROJ_PATH)['W'].shape[1])   # stored descriptor width = projection's PCA dim
ALGO_TAG = f'rn18win48-pca{DIM}i8-v1'
DEEP_LONG = 512                    # long-side cap before describing
RATIO = 2 ** -0.5                  # pyramid step
MAX_LEVELS = 5
MIN_SIDE = 48                      # one window = 6x6 layer2 cells = 48 px
QUERY_SCALES = (1.0, 0.71, 0.5)

_net = None
_proj = None


def _get_net():
    """ResNet18 (timm a1_in1k) up to layer3, on the central compute device."""
    global _net
    if _net is None:
        import timm  # hook-allow: lazy — optional heavy dep
        import torch  # hook-allow: lazy — optional heavy dep
        from . import compute
        model = timm.create_model('resnet18.a1_in1k', pretrained=True, features_only=True,
                                  out_indices=(2, 3)).eval()
        _net = (model.to(compute.torch_device()), torch)
    return _net


def _get_proj():
    global _proj
    if _proj is None:
        z = np.load(_PROJ_PATH)
        _proj = {k: z[k] for k in ('mu', 'sd', 'bwv', 'W')}
    return _proj


def _level_feats(rgb):
    """rgb uint8 (h, w, 3) -> raw window stats (n, 512) float32, or None if too small."""
    import torch.nn.functional as F  # hook-allow: lazy — optional heavy dep
    h, w = rgb.shape[:2]
    if min(h, w) < MIN_SIDE:
        return None
    net, torch = _get_net()
    dev = next(net.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    with torch.inference_mode():
        f2, f3 = net(((x - mean) / std)[None].to(dev))
        # layer2 stride 8: 6x6 cells = 48 px, stride 4 cells; layer3 stride 16: 3x3 cells, stride 2
        m2 = F.avg_pool2d(f2, 6, 4)
        v2 = F.avg_pool2d(f2 * f2, 6, 4) - m2 * m2
        m3 = F.avg_pool2d(f3, 3, 2)
        nh, nw = min(m2.shape[2], m3.shape[2]), min(m2.shape[3], m3.shape[3])
        if nh < 1 or nw < 1:
            return None
        cat = torch.cat([m2[:, :, :nh, :nw], v2[:, :, :nh, :nw].clamp_min(0).sqrt(),
                         m3[:, :, :nh, :nw]], 1)[0]
        return cat.permute(1, 2, 0).reshape(-1, cat.shape[0]).float().cpu().numpy()


def project(raw):
    """Raw (n, 512) window stats -> L2-normalised float32 (n, DIM)."""
    p = _get_proj()
    z = (np.sqrt(np.maximum(raw, 0)) - p['mu']) / p['sd'] * p['bwv']
    z = z @ p['W']
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    return z.astype(np.float32)


def quantize(vecs):
    """Unit-norm float32 (n, DIM) -> int8 (components are in [-1, 1])."""
    return np.clip(np.rint(vecs * 127.0), -127, 127).astype(np.int8)


def load_rgb(image_bytes):
    """Encoded bytes -> RGB uint8 array in the EXIF-upright frame the UI draws boxes in,
    or None if undecodable."""
    import io  # hook-allow: lazy — cheap, keeps module import minimal
    from .formats import open_upright_rgb
    try:
        return np.asarray(open_upright_rgb(io.BytesIO(image_bytes)))
    except Exception:  # hook-allow: broad — PIL raises many types on bad data; None = undecodable
        return None


def _cap(rgb, scale_out=False):
    import cv2  # hook-allow: lazy — keeps module import minimal
    h, w = rgb.shape[:2]
    s = min(1.0, DEEP_LONG / max(h, w))
    out = rgb if s == 1.0 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return (out, s) if scale_out else out


def describe_image(rgb):
    """Index side: RGB array -> int8 (n, DIM) windows over the pyramid ((0, DIM) when
    the image is too small to hold a window)."""
    import cv2  # hook-allow: lazy — keeps module import minimal
    rgb = _cap(rgb)
    h, w = rgb.shape[:2]
    raws, s = [], 1.0
    for _ in range(MAX_LEVELS):
        nh, nw = round(h * s), round(w * s)
        if min(nh, nw) < MIN_SIDE:
            break
        im = rgb if s == 1.0 else cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        r = _level_feats(im)
        if r is not None:
            raws.append(r)
        s *= RATIO
    if not raws:
        return np.zeros((0, DIM), np.int8)
    return quantize(project(np.concatenate(raws)))


def describe_query(rgb, bbox):
    """Query side: crop `bbox` (px of `rgb`) at native scale, described at QUERY_SCALES
    -> float32 (S, DIM) unit vectors, one per scale; None if the crop is too small."""
    import cv2  # hook-allow: lazy — keeps module import minimal
    rgb, s0 = _cap(rgb, scale_out=True)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v * s0)) for v in bbox]
    crop = rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 8:
        return None
    if min(ch, cw) < 50:       # tiny crop: upsample so at least one window fits
        f = 50 / min(ch, cw)
        crop = cv2.resize(crop, (round(cw * f), round(ch * f)), interpolation=cv2.INTER_CUBIC)
    out = []
    for s in QUERY_SCALES:
        im = crop if s == 1.0 else cv2.resize(
            crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
            interpolation=cv2.INTER_AREA)
        r = _level_feats(im)
        if r is not None:
            q = project(r).mean(0)
            out.append(q / (np.linalg.norm(q) + 1e-9))
    return np.stack(out).astype(np.float32) if out else None


def score_windows(matrix, counts, queries):
    """Per-image score for a chunk: matrix int8 (sum(counts), DIM) of concatenated windows,
    counts (k,) windows per image, queries float32 (S, DIM) -> float32 (k,) = best cosine over
    windows and query scales (-1 for images with no windows)."""
    out = np.full(len(counts), -1.0, np.float32)
    if matrix.shape[0] == 0:
        return out
    sims = (matrix.astype(np.float32) @ queries.T).max(axis=1) / 127.0   # best scale per window
    starts = np.cumsum(counts) - counts
    has = counts > 0
    out[has] = np.maximum.reduceat(sims, starts[has])
    return out
