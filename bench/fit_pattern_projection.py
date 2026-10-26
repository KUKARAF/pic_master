"""Fit the fixed whitening+PCA projection shipped for find-by-pattern (pattern_deep.py).

    python -I bench/fit_pattern_projection.py [--raw bench/.cache/e3_raw/raw_deep.npz] [--dim 96]

Input is the cached raw (N, 512) ResNet18 window stats of the E3-deep prototype. Mirrors
E3Texture._prepare (sqrt compression, standardise, block weights, shrinkage whitening
0.3) then keeps the top `dim` principal axes. Writes media_manager/data/pattern_deep_proj.npz.
"""
import argparse
import os

import numpy as np

BLOCKS = (128, 128, 256)          # m2, sd2, m3
BLOCK_W = (1.0, 0.6, 1.0)
SHRINK = 0.3


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(here, '.cache', 'e3_raw', 'raw_deep.npz'))
    ap.add_argument('--dim', type=int, default=96)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(here), 'media_manager', 'data',
                                                  'pattern_deep_proj.npz'))
    a = ap.parse_args()
    feats = np.load(a.raw)['feats']
    rng = np.random.default_rng(0)
    sub = np.sqrt(np.maximum(feats[rng.choice(len(feats), min(150000, len(feats)), replace=False)]
                             .astype(np.float32), 0))
    mu, sd = sub.mean(0), sub.std(0) + 1e-4
    bwv = np.concatenate([np.full(n, w / np.sqrt(n), np.float32) for n, w in zip(BLOCKS, BLOCK_W)])
    z = (sub - mu) / sd * bwv
    ev, V = np.linalg.eigh(np.cov(z.T))
    keep = ev > 1e-6 * ev.max()
    ev, V = ev[keep], V[:, keep]
    shrink = SHRINK * ev.mean()
    order = np.argsort(-ev)[:a.dim]
    W = (V[:, order] / np.sqrt(ev[order] + shrink)).astype(np.float32)
    print(f'kept {len(order)}/{len(ev)} axes, explained {ev[order].sum() / ev.sum():.3f}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, mu=mu.astype(np.float32), sd=sd.astype(np.float32), bwv=bwv, W=W)
    print('wrote', a.out, os.path.getsize(a.out), 'bytes')


if __name__ == '__main__':
    main()
