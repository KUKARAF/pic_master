"""E3 deep-texture variant: window mean/std statistics of early ResNet18 features
(layer2 stride-8 + layer3 stride-16), orderless pooling over 48-px windows."""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from . import descriptor as D

_net = None
DEEP_LONG = 512
RATIO = 2 ** -0.5


def net():
    global _net
    if _net is None:
        import timm
        torch.set_num_threads(2)
        _net = timm.create_model('resnet18.a1_in1k', pretrained=True, features_only=True,
                                 out_indices=(2, 3)).eval()
    return _net


def _prep(rgb):
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((x - mean) / std)[None]


@torch.inference_mode()
def level_feats(rgb):
    """rgb uint8 -> (n, 128*2+256) raw window stats, xy(n,2) at this level's pixel scale."""
    h, w = rgb.shape[:2]
    if min(h, w) < 48:
        return None
    f2, f3 = net()(_prep(rgb))
    # layer2 stride 8: 6x6 cells = 48 px, stride 4 cells (32 px); layer3 stride 16: 3x3 cells, stride 2
    m2 = F.avg_pool2d(f2, 6, 4); v2 = F.avg_pool2d(f2 * f2, 6, 4) - m2 * m2
    m3 = F.avg_pool2d(f3, 3, 2)
    n_h = min(m2.shape[2], m3.shape[2]); n_w = min(m2.shape[3], m3.shape[3])
    if n_h < 1 or n_w < 1:
        return None
    m2, v2, m3 = m2[:, :, :n_h, :n_w], v2[:, :, :n_h, :n_w], m3[:, :, :n_h, :n_w]
    cat = torch.cat([m2, v2.clamp_min(0).sqrt(), m3], 1)[0]            # (C, nh, nw)
    ys, xs = np.mgrid[0:n_h, 0:n_w]
    xy = np.stack([xs.ravel() * 32, ys.ravel() * 32], 1)
    return cat.permute(1, 2, 0).reshape(-1, cat.shape[0]).numpy(), xy


def describe_image(rgb, max_levels=5):
    rgb = _cap(rgb)
    F_, M = [], []
    for li, (s, im) in enumerate(D.pyramid(rgb, RATIO, max_levels=max_levels)):
        r = level_feats(im)
        if r is None:
            continue
        F_.append(r[0]); M.append(np.concatenate([np.full((len(r[0]), 1), li, np.int16), r[1].astype(np.int16)], 1))
    if not F_:
        return np.zeros((0, 512), np.float32), np.zeros((0, 3), np.int16)
    return np.concatenate(F_), np.concatenate(M)


def _cap(rgb, long=DEEP_LONG):
    h, w = rgb.shape[:2]
    s = min(1.0, long / max(h, w))
    return rgb if s == 1.0 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def describe_query(rgb, bbox, scales=(1.0, 0.71, 0.5)):
    h0, w0 = rgb.shape[:2]
    s0 = min(1.0, DEEP_LONG / max(h0, w0))
    x1, y1, x2, y2 = [int(round(v * s0)) for v in bbox]
    rgb = _cap(rgb)
    h, w = rgb.shape[:2]
    crop = rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 8:
        return {}
    if min(ch, cw) < 50:
        f = 50 / min(ch, cw)
        crop = cv2.resize(crop, (round(cw * f), round(ch * f)), interpolation=cv2.INTER_CUBIC)
    out = {}
    for s in scales:
        im = crop if s == 1.0 else cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                                              interpolation=cv2.INTER_AREA)
        r = level_feats(im)
        if r is not None:
            out[s] = r[0]
    return out


def index_one(path):
    import sys
    try:
        torch.set_num_threads(1)
        f, m = describe_image(D.load_rgb(path))
    except Exception as e:
        print('e3deep: failed', path, e, file=sys.stderr)
        f, m = np.zeros((0, 512), np.float32), np.zeros((0, 3), np.int16)
    return f.astype(np.float16), m


DIM = 512
BLOCKS = {'m2': slice(0, 128), 'sd2': slice(128, 256), 'm3': slice(256, 512)}
