"""Offline tuning for E1: python -I bench/proto_e1/tune.py  (after the harness built the index).
Caches per-case DINO sub-scores + SIFT inliers, then grid-searches the fusion.
Tune split: synthetic + even-indexed real cases; report split: all."""
import json, os, pickle, sys, itertools
import numpy as np
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(BENCH), BENCH]
import harness
from harness import case_metrics
import importlib.util
spec = importlib.util.spec_from_file_location('e1m', os.path.join(BENCH, 'methods/e1_dino_geom.py'))
e1m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e1m)
from proto_e1 import core
M = e1m.METHOD
M.warmup()
paths = harness.corpus_paths(harness.DATA_DIR)
M.build_index(paths, os.path.join(harness.CACHE_ROOT, M.name))
cases = [c for c in harness.load_cases(harness.DATA_DIR) if c['task'] == 'region']
CF = os.path.join(BENCH, '.cache_e1_tune.pkl')
cache = pickle.load(open(CF, 'rb')) if os.path.isfile(CF) else {}
NV = 200
for c in cases:
    if c['id'] in cache: continue
    im = core.load_rgb(c['query_image'])
    W, H = im.size
    x0, y0, x1, y1 = c['bbox']
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    vecs = {}
    for L in (224, 336, 448):
        vecs[f'crop{L}'] = core.pool_box(M.dino.tokens(crop, L)[0], (0, 0, 1, 1))
    tok, _ = M.dino.tokens(im)
    vecs['ctx'] = core.pool_box(tok, (x0 / W, y0 / H, x1 / W, y1 / H))
    sub = {k: M.prefilter(v[None]) for k, v in vecs.items()}
    # union candidate pool = top-NV by max over variants
    comb = np.max(np.stack(list(sub.values())), 0)
    cand = np.argsort(-comb)[:NV]
    inl = M.verify(crop, cand)
    cache[c['id']] = {'sub': sub, 'cand': cand, 'inl': inl}
    print('tuned', c['id'], flush=True)
    pickle.dump(cache, open(CF, 'wb'))

def evaluate(cs, modes, N, minin, w, use_sift=True):
    res = []
    for c in cs:
        d = cache[c['id']]
        s = np.max(np.stack([d['sub'][m] for m in modes]), 0)
        order = np.argsort(-s)[:N]
        sc = s.copy()
        if use_sift:
            for i in order:
                if i in d['inl']:
                    sc[i] = s[i] + w * np.log1p(max(0.0, d['inl'][i] - minin))
        rk = [paths[i] for i in np.argsort(-sc, kind='stable') if paths[i] != c['query_image']]
        res.append(case_metrics(rk, c['positives'], c['hard_negatives']))
    return {k: float(np.mean([r[k] for r in res])) for k in ('recall@1', 'recall@10', 'ap', 'confusion@10', 'hit@1')}

tune = [c for c in cases if c['synthetic']] + [c for c in cases if not c['synthetic']][::2]
modesets = [('crop224',), ('crop336',), ('crop448',), ('ctx',), ('crop224', 'crop448'), ('crop224', 'crop336', 'crop448'),
            ('crop224', 'crop448', 'ctx'), ('crop224', 'crop336', 'crop448', 'ctx')]
print('--- prefilter only (tune split)')
for ms in modesets:
    print(ms, evaluate(tune, ms, 0, 0, 0, False))
print('--- prefilter recall: fraction of positives inside top-N (tune split)')
for ms in modesets[4:]:
    for N in (50, 100, 150, 200):
        r = []
        for c in tune:
            s = np.max(np.stack([cache[c['id']]['sub'][m] for m in ms]), 0)
            top = set(paths[i] for i in np.argsort(-s)[:N])
            r.append(np.mean([p in top for p in c['positives']]))
        print(ms, N, round(float(np.mean(r)), 3))
print('--- fusion grid (tune split)')
best = None
for ms in modesets[4:]:
    for N, minin, w in itertools.product((100, 200), (5, 8, 12, 20), (0.05, 0.15, 0.3, 1.0)):
        m = evaluate(tune, ms, N, minin, w)
        key = m['ap']
        if best is None or key > best[0]:
            best = (key, ms, N, minin, w, m)
print('BEST on tune split', best)
_, ms, N, minin, w, _m = best
print('ALL / real / synth with best:')
real = [c for c in cases if not c['synthetic']]; syn = [c for c in cases if c['synthetic']]
held = [c for c in real][1::2]
for name, cs in (('all', cases), ('real', real), ('synthetic', syn), ('heldout-real', held)):
    print(name, evaluate(cs, ms, N, minin, w), 'prefilter-only', evaluate(cs, ms, N, minin, w, False))
json.dump({'modes': ms, 'N': N, 'min_inl': minin, 'w': w}, open(os.path.join(BENCH, 'results/e1_tuned_params.json'), 'w'))
