"""Pattern-search benchmark: DINOv2-L dense cells (pattern_dino.py) vs the production ResNet18
windows (pattern_deep.py), fused, on the harness's `pattern` cases. RUN ON THE GPU MACHINE (bigboy);
DINOv2Embedder is GPU-only by design and raises on a CPU box.

    export LD_LIBRARY_PATH=...libGL...   # as in bench/README.md
    cd <repo>/media_manager
    # 1. benchmark everything (builds + caches raw DINO cells and ResNet windows, fits PCA in memory)
    .venv/bin/python -I bench/bench_pattern_dino.py --json bench/results/pattern_dino.json
    # 2. only if DINO wins/fuses well: fit the SHIPPED projection and write it into the package
    .venv/bin/python -I bench/bench_pattern_dino.py --fit --dim 128 --write-proj
    #    (add --fit-dir /path/to/library --fit-n 3000 to fit on a sample of your real library
    #     instead of the benchmark corpus), then commit data/pattern_dino_proj.npz and redeploy
    #     the worker (see reference_worker_bigboy).

Protocol and metrics are the harness's (same corpus, same cases, `harness.case_metrics`, query
image removed from the ranking, mAP over all positives, splits all/real/synthetic), so the numbers
compare directly with bench/README.md and the ResNet reference rows printed here (production
int8/PCA-96: mAP .731; float prototype .757; +LBP-fused prototype .79).

What it measures (one DINO forward per corpus image and per query, then everything else is numpy):
  * DINO variants: query mode {mask, crop, both} x crop rotations {0 | 0,90,180,270} x crop scales
    {all | 1.0 only} x PCA dim {--dims} (+ an un-projected "noproj" oracle: does int8/PCA cost accuracy?)
  * ResNet baseline (production code path) re-scored in the same run
  * fusion: z-score fusion (pattern_dino.fuse_scores) of the ResNet scores and each of the best DINO
    variants at DINO weights {.25,.5,.75}
  * cost: DINO index throughput (img/s, GPU-seconds), cells and KiB per image, extrapolation to 360k
Fit leakage: like the ResNet projection, the PCA is fitted unsupervised on the corpus images
(no labels used); use --fit-dir to fit on other photos for a leak-free number.
"""
import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import harness  # noqa: E402
from media_manager import pattern_deep as pdp  # noqa: E402
from media_manager import pattern_dino as pdn  # noqa: E402

REFERENCE = {'resnet production int8/PCA96': 0.731, 'resnet float prototype': 0.757,
             'resnet+LBP fused prototype': 0.790, 'DINOv2-S dense MaxSim prototype (e2)': 0.425,
             'LBP baseline': 0.365}
ROT4 = (0, 1, 2, 3)
LIBRARY_SIZE = 360_000


def make_embedder():
    """The GPU-only DINOv2-L embedder (same class/weights as the embedding pipeline). Tests swap this."""
    from media_manager.embedders import DINOv2Embedder
    return DINOv2Embedder()


# ------------------------------------------------------------------------------ index caches

def build_dino_cells(emb, paths, cache_dir, chunk=128):
    """Raw (pre-projection) pooled cells for the corpus: (feats float16 (N_cells, 1024), counts)."""
    key = f'{harness.index_key(paths)}|{pdn.ALGO_BASE}'
    f = os.path.join(cache_dir, 'raw_cells.npz')
    if os.path.isfile(f):
        z = np.load(f, allow_pickle=False)
        if str(z['key']) == key:
            return z['feats'], z['counts'], json.loads(str(z['cost']))
    stats, parts, counts = {}, [], []
    t0 = time.perf_counter()
    for i in range(0, len(paths), chunk):
        res, errs = pdn.describe_files(emb, paths[i:i + chunk], stats=stats, raw=True)
        for p, r, e in zip(paths[i:i + chunk], res, errs):
            if r is None:
                raise RuntimeError(f'cannot describe {p}: {e}')
            parts.append(r.astype(np.float16))
            counts.append(len(r))
        print(f'  dino cells {min(i + chunk, len(paths))}/{len(paths)} '
              f'({(i + chunk) / (time.perf_counter() - t0):.1f} img/s)', flush=True)
    wall = time.perf_counter() - t0
    feats, counts = np.concatenate(parts), np.array(counts, np.int64)
    cost = {'wall_s': wall, 'img_per_s': len(paths) / wall, 'gpu_s': stats.get('gpu_seconds', 0.0),
            'decode_thread_s': stats.get('decode_seconds', 0.0), 'cells_per_img': float(counts.mean())}
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(f, key=key, feats=feats, counts=counts, cost=json.dumps(cost))
    return feats, counts, cost


def build_resnet_windows(paths, cache_dir):
    """int8 production ResNet18 windows (pattern_deep.describe_image) for the corpus."""
    key = harness.index_key(paths)
    f = os.path.join(cache_dir, 'rn18_windows.npz')
    if os.path.isfile(f):
        z = np.load(f, allow_pickle=False)
        if str(z['key']) == key:
            return z['w'], z['counts']
    parts, counts = [], []
    for i, p in enumerate(paths):
        rgb = pdp.load_rgb(open(p, 'rb').read())
        w = np.zeros((0, pdp.DIM), np.int8) if rgb is None else pdp.describe_image(rgb)
        parts.append(w)
        counts.append(len(w))
        if (i + 1) % 100 == 0:
            print(f'  resnet windows {i + 1}/{len(paths)}', flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    w, counts = np.concatenate(parts), np.array(counts, np.int64)
    np.savez(f, key=key, w=w, counts=counts)
    return w, counts


# ----------------------------------------------------------------------------------- scoring

def score_matrix(matrix, counts, queries, step=400):
    """Per-image best cosine, chunked (int8 matrix, same scorer as production)."""
    out = np.empty(len(counts), np.float32)
    start = 0
    for i in range(0, len(counts), step):
        c = counts[i:i + step]
        n = int(c.sum())
        out[i:i + step] = pdn.score_cells(matrix[start:start + n], c, queries)
        start += n
    return out


def score_float(feats, counts, queries, mu, step=200):
    """Un-projected oracle: cosine of centred, L2-normalised raw cells vs centred query vectors."""
    q = pdn._l2(pdn._l2(queries) - mu)
    out = np.empty(len(counts), np.float32)
    start = 0
    for i in range(0, len(counts), step):
        c = counts[i:i + step]
        n = int(c.sum())
        x = pdn._l2(pdn._l2(feats[start:start + n].astype(np.float32)) - mu)
        res = np.full(len(c), -1.0, np.float32)
        if n:
            sims = (x @ q.T).max(1)
            s, has = np.cumsum(c) - c, c > 0
            res[has] = np.maximum.reduceat(sims, s[has])
        out[i:i + step] = res
        start += n
    return out


def evaluate(score_fn, cases, paths):
    """harness metrics for one scoring function (case -> scores over `paths`)."""
    norm = [os.path.normpath(p) for p in paths]
    results = []
    for c in cases:
        scores = score_fn(c)
        order = np.argsort(-scores, kind='stable')
        ranked = [norm[i] for i in order if norm[i] != c['query_image']]
        m = harness.case_metrics(ranked, c['positives'], c['hard_negatives'])
        m.update(id=c['id'], synthetic=c['synthetic'])
        results.append(m)
    split = {'all': results, 'real': [r for r in results if not r['synthetic']],
             'synthetic': [r for r in results if r['synthetic']]}
    return {k: harness.summarize(v) for k, v in split.items() if v}, results


# ------------------------------------------------------------------------------------- main

def fit_corpus_cells(emb, args, paths, feats, counts):
    """Raw cells to fit the projection on: the benchmark corpus, or a sample of --fit-dir."""
    if not args.fit_dir:
        return feats.astype(np.float32)
    cand = harness.corpus_paths(args.fit_dir)
    rng = np.random.default_rng(0)
    pick = [cand[i] for i in rng.choice(len(cand), min(args.fit_n, len(cand)), replace=False)]
    print(f'fitting on {len(pick)} images from {args.fit_dir}', flush=True)
    cells = []
    for i in range(0, len(pick), 128):
        res, errs = pdn.describe_files(emb, pick[i:i + 128], raw=True)
        cells += [r for r in res if r is not None and len(r)]
    return np.concatenate(cells)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=harness.DATA_DIR)
    ap.add_argument('--manifest')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--dims', default='64,128,256', help='PCA dims to compare')
    ap.add_argument('--dim', type=int, default=pdn.DEFAULT_DIM, help='dim of the projection --fit writes')
    ap.add_argument('--shrink', type=float, default=pdn.FIT_SHRINK)
    ap.add_argument('--fit', action='store_true', help='fit the shipped projection and stop')
    ap.add_argument('--write-proj', action='store_true', help='with --fit: write data/pattern_dino_proj.npz')
    ap.add_argument('--fit-dir', help='fit on a sample of photos from here instead of the corpus')
    ap.add_argument('--fit-n', type=int, default=3000)
    ap.add_argument('--skip-resnet', action='store_true')
    ap.add_argument('--no-oracle', action='store_true', help='skip the un-projected 1024-d oracle (needs RAM)')
    ap.add_argument('--cache-dir', default=os.path.join(harness.CACHE_ROOT, 'pattern_dino'))
    args = ap.parse_args(argv)

    paths = harness.corpus_paths(args.data_dir)
    cases = [c for c in harness.load_cases(args.data_dir, args.manifest) if c['task'] == 'pattern']
    emb = make_embedder()
    print(f'corpus {len(paths)} images, {len(cases)} pattern cases', flush=True)
    feats, counts, cost = build_dino_cells(emb, paths, args.cache_dir)

    if args.fit:
        mu, W = pdn.fit_projection(fit_corpus_cells(emb, args, paths, feats, counts), args.dim, args.shrink)
        pdn.set_projection(mu, W)
        print(f'fitted dim={args.dim}  algo tag {pdn.algo_tag()}')
        if args.write_proj:
            print('wrote', pdn.save_projection(mu, W))
        else:
            print('(not written: add --write-proj)')
        return None

    # ---- query-side raw vectors: one dense forward per case (+ crops); cached per run
    qraw, qlat = {}, []
    for c in cases:
        rgb = pdp.load_rgb(open(c['query_image'], 'rb').read())
        t0 = time.perf_counter()
        qraw[c['id']] = pdn.query_raw(emb, rgb, c['bbox'], rots=ROT4)
        qlat.append(time.perf_counter() - t0)

    report = {'corpus_images': len(paths), 'n_cases': len(cases), 'reference_mAP': REFERENCE,
              'cost': {**cost, 'extrapolated_360k_hours_at_img_per_s': LIBRARY_SIZE / cost['img_per_s'] / 3600,
                       'query_forward_p50_s': harness.percentile(qlat, 50)},
              'dino': {}, 'fusion': {}}
    rows = []     # (label, metrics)

    def dino_scores_for(dim, mode, rots, scales):
        W = projs[dim]
        def fn(c):
            sel = pdn.select_query(qraw[c['id']], mode, rots=rots, scales=scales)
            if sel is None:
                return np.zeros(len(paths), np.float32)
            return score_matrix(mats[dim], counts, pdn.project(sel, *W))
        return fn

    projs, mats = {}, {}
    fit_cells = fit_corpus_cells(emb, args, paths, feats, counts)
    for dim in [int(x) for x in args.dims.split(',')]:
        projs[dim] = pdn.fit_projection(fit_cells, dim, args.shrink)
        mats[dim] = np.concatenate([pdn.quantize(pdn.project(feats[i:i + 50000].astype(np.float32), *projs[dim]))
                                    for i in range(0, len(feats), 50000)])
    bytes_per_img = {d: float(counts.mean()) * d for d in projs}
    cache_scores = {}
    variants = list(itertools.product(sorted(projs), ('mask', 'crop', 'both'), ((0,), ROT4), (None, (1.0,))))
    for dim, mode, rots, scales in variants:
        if mode == 'mask' and (rots != (0,) or scales is not None):
            continue                                   # mask vectors have no rotations / scales
        label = f"dino pca{dim} {mode}" + ('' if mode == 'mask' else
                                           f" rots={'4' if rots == ROT4 else '1'} scales={'1.0' if scales else 'all'}")
        fn = dino_scores_for(dim, mode, rots, scales)
        store = {}
        def cached(c, fn=fn, store=store):
            store[c['id']] = fn(c)
            return store[c['id']]
        m, per_case = evaluate(cached, cases, paths)
        cache_scores[label] = store
        report['dino'][label] = {'metrics': m, 'KiB_per_img': bytes_per_img[dim] / 1024}
        rows.append((label, m))

    if not args.no_oracle:
        mu = projs[min(projs)][0]
        def oracle(c):
            sel = pdn.select_query(qraw[c['id']], 'both', rots=(0,))
            return score_float(feats, counts, sel, mu) if sel is not None else np.zeros(len(paths), np.float32)
        store = {}
        m, _ = evaluate(lambda c: store.setdefault(c['id'], oracle(c)), cases, paths)
        report['dino']['dino noproj(1024) both'] = {'metrics': m, 'KiB_per_img': float(counts.mean()) * pdn.TOKEN_DIM / 1024}
        rows.append(('dino noproj(1024) both  [oracle]', m))

    rn_store = None
    if not args.skip_resnet:
        w, rcounts = build_resnet_windows(paths, args.cache_dir)
        rn_store = {}
        def rn(c):
            if c['id'] not in rn_store:
                q = pdp.describe_query(pdp.load_rgb(open(c['query_image'], 'rb').read()), list(c['bbox']))
                rn_store[c['id']] = (np.zeros(len(paths), np.float32) if q is None else score_matrix(w, rcounts, q))
            return rn_store[c['id']]
        m, _ = evaluate(rn, cases, paths)
        report['resnet'] = {'metrics': m, 'KiB_per_img': float(rcounts.mean()) * pdp.DIM / 1024}
        rows.append(('resnet18 (production)', m))
        ranked_dino = sorted(cache_scores, key=lambda k: -report['dino'][k]['metrics']['all']['ap'])[:3]
        ids = np.arange(len(paths), dtype=np.int64)
        for label in ranked_dino:
            for wd in (0.25, 0.5, 0.75):
                def fused(c, label=label, wd=wd):
                    f, s = pdn.fuse_scores([(ids, rn_store[c['id']], 1 - wd), (ids, cache_scores[label][c['id']], wd)])
                    out = np.full(len(paths), -1e9, np.float32)
                    out[f] = s
                    return out
                for c in cases:
                    rn(c)
                m, _ = evaluate(fused, cases, paths)
                key = f'fused[{label}] w_dino={wd}'
                report['fusion'][key] = {'metrics': m}
                rows.append((key, m))

    print(f"\n{'variant':<62} {'mAP all':>8} {'real':>7} {'synth':>7} {'H@1':>6} {'R@10':>6} {'HN>pos':>7}")
    for label, m in sorted(rows, key=lambda r: -r[1]['all']['ap']):
        a = m['all']
        print(f"{label:<62} {a['ap']:8.3f} {m['real']['ap'] if 'real' in m else float('nan'):7.3f} "
              f"{m['synthetic']['ap'] if 'synthetic' in m else float('nan'):7.3f} {a['hit@1']:6.3f} "
              f"{a['recall@10']:6.3f} {a['hardneg_above_pos']:7.3f}")
    print('\nreference mAP (all): ' + ', '.join(f'{k} {v:.3f}' for k, v in REFERENCE.items()))
    print(f"DINO index: {cost['img_per_s']:.1f} img/s (GPU {cost['gpu_s']:.0f}s of {cost['wall_s']:.0f}s wall), "
          f"{cost['cells_per_img']:.0f} cells/img -> {LIBRARY_SIZE} photos ~ "
          f"{LIBRARY_SIZE / cost['img_per_s'] / 3600:.1f} h; storage per photo: "
          + ', '.join(f'pca{d} {b / 1024:.1f} KiB' for d, b in bytes_per_img.items()))
    print('Decision rule: ship DINO/fused as default only if its mAP(all) beats resnet (production) by a clear '
          'margin AND does not lose on the synthetic split (swatches); else keep resnet default and offer fused.')
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
        print('wrote', args.json)
    return report


if __name__ == '__main__':
    main()
