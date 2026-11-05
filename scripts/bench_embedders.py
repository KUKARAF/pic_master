#!/usr/bin/env python
"""Benchmark SigLIP2 / DINOv2 embedders on a folder; optional SigLIP2 text sanity.

  python scripts/bench_embedders.py FOLDER -n 200 [--devices auto,cpu] -q "a dog" -q "sunset"
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np  # noqa: E402

from media_manager.formats import IMAGE_EXTENSIONS  # noqa: E402


def collect(folder, n):
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                out.append(os.path.join(root, f))
                if len(out) >= n:
                    return out
    return out


def sync():
    import torch
    for b in ('xpu', 'cuda'):
        m = getattr(torch, b, None)
        if m is not None and getattr(m, 'is_available', lambda: False)():
            m.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('-n', type=int, default=200)
    ap.add_argument('--devices', default='auto', help='comma list of auto|cpu|cuda|xpu')
    ap.add_argument('--allow-cpu', action='store_true', help='one-off CPU comparison only (embedders are GPU-only)')
    ap.add_argument('--models', default='siglip2,dino')
    ap.add_argument('-q', '--query', action='append', default=[])
    a = ap.parse_args()
    paths = collect(a.folder, a.n)
    print(f'{len(paths)} images from {a.folder}')
    for dev in a.devices.split(','):
        if dev == 'auto':
            os.environ.pop('MEDIA_DEVICE', None)
        else:
            os.environ['MEDIA_DEVICE'] = dev
        from media_manager import compute
        compute.torch_device.cache_clear()
        for name in a.models.split(','):
            try:
                if name == 'siglip2':
                    from media_manager.embedders import SigLIP2Embedder as E
                    emb = E(_allow_cpu=a.allow_cpu)
                elif name == 'dino':
                    from media_manager.embedders import DINOv2Embedder as E
                    emb = E(_allow_cpu=a.allow_cpu)
                else:
                    raise ValueError(f'unknown model {name!r} (expected siglip2 or dino)')
                emb.embed_images(paths[:8])  # warmup (kernel compile / lazy init)
                sync()
                t = time.perf_counter()
                vecs, failed = emb.embed_images(paths)
                sync()
                dt = time.perf_counter() - t
                print(f'RESULT dev={emb.device} model={name} {len(vecs)/dt:.1f} img/s '
                      f'({len(vecs)} ok, {len(failed)} failed, {dt:.1f}s) dim={vecs.shape[-1] if vecs.ndim == 2 else 0}',
                      flush=True)
                if name == 'siglip2' and a.query:
                    ok = [p for p in paths if p not in {f[0] for f in failed}]
                    for q in a.query:
                        s = vecs @ emb.embed_text(q)
                        top = np.argsort(-s)[:5]
                        print(f'  query {q!r}:')
                        for i in top:
                            print(f'    {s[i]:.3f} {ok[i]}')
                if name == 'dino' and 'siglip2' in a.models.split(','):
                    from media_manager.embedders import SigLIP2Embedder, embed_shared
                    both = [SigLIP2Embedder(_allow_cpu=a.allow_cpu), emb]
                    embed_shared(both, paths[:8]); sync()
                    t = time.perf_counter()
                    embed_shared(both, paths); sync()
                    print(f'RESULT dev={emb.device} model=siglip2+dino(shared decode) {len(paths)/(time.perf_counter()-t):.1f} img/s', flush=True)
            except Exception as exc:  # hook-allow: broad — bench must report and continue
                print(f'FAIL dev={dev} model={name}: {type(exc).__name__}: {exc}', flush=True)


if __name__ == '__main__':
    main()
