"""Evaluate all E2 variants against the bench manifest (one pass, shared index).
   python -I bench/proto_e2/run_variants.py [--task region|pattern] [--limit N]"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path[:0] = [BENCH, os.path.dirname(BENCH)]
import harness
from proto_e2 import e2lib

CFG = {'region': dict(mults=(0.5, 0.7, 1.0, 1.4, 2.0), rots=(0,)),
       'pattern': dict(mults=(0.5, 0.7, 1.0, 1.4, 2.0), rots=(0, 90, 180, 270))}
TOP_RERANK = 100
SHORT = 300


def rank_of(paths, scores):
    order = np.argsort(-scores, kind='stable')
    return [paths[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    data = harness.DATA_DIR
    paths = harness.corpus_paths(data)
    cases = harness.load_cases(data)
    cache = os.path.join(harness.CACHE_ROOT, 'e2_dino_dense')
    os.makedirs(cache, exist_ok=True)
    ix = e2lib.E2Index(cache)
    meta_p = os.path.join(cache, harness.META_FILE)
    key = harness.index_key(paths)
    t0 = time.perf_counter()
    ix.get_model()
    ix.build(paths, '/var/home/rafa/.cache/e2_scratch')
    el = time.perf_counter() - t0
    if not os.path.isfile(meta_p):
        size = sum(os.path.getsize(os.path.join(cache, f)) for f in os.listdir(cache))
        json.dump({'key': key, 'seconds': el, 'bytes': size}, open(meta_p, 'w'))
    print('index ready', el, flush=True)
    out = {}
    for task in ([a.task] if a.task else ['region', 'pattern']):
        todo = [c for c in cases if c['task'] == task]
        if a.limit:
            todo = todo[:a.limit]
        per = {}   # variant -> list of case metric dicts
        lat = {}
        short_recall = []
        for c in todo:
            q = c['query_image']
            qi = paths.index(q)
            t1 = time.perf_counter()
            ql = ix.query_tokens(q, c['bbox'], **CFG[task])
            t2 = time.perf_counter()
            full = ix.maxsim_scores(ql)
            t3 = time.perf_counter()
            co = ix.maxsim_scores(ql, coarse=True)
            t4 = time.perf_counter()
            cand = np.argsort(-full['maxsim'])[:TOP_RERANK]
            mnn, vote = ix.rerank_scores(ql, cand)
            t5 = time.perf_counter()
            V = {}
            for k in ('maxsim', 'trim', 'pooled'):
                V[k] = full[k]
            # coarse shortlist -> exact
            sh = np.argsort(-co['maxsim'])[:SHORT]
            sc = co['maxsim'] - 10
            sc[sh] = full['maxsim'][sh]
            V['shortlist_maxsim'] = sc
            sc = co['trim'] - 10
            sc[sh] = full['trim'][sh]
            V['shortlist_trim'] = sc
            V['coarse'] = co['maxsim']
            def rr(base, extra):
                s = base.copy() - 10
                s[cand] = base[cand] + extra
                return s
            V['mnn'] = rr(full['maxsim'], 0) * 0 + rr(np.zeros_like(full['maxsim']), 0)
            s = full['maxsim'].copy() - 10; s[cand] = mnn + 10; V['mnn'] = s
            s = full['maxsim'].copy() - 10; s[cand] = vote + 10; V['vote'] = s
            s = full['maxsim'].copy() - 10; s[cand] = full['maxsim'][cand] + vote + 10; V['maxsim+vote'] = s
            s = full['maxsim'].copy() - 10; s[cand] = full['trim'][cand] + vote + 10; V['trim+vote'] = s
            s = full['maxsim'].copy() - 10; s[cand] = full['trim'][cand] + vote + mnn + 10; V['trim+vote+mnn'] = s
            pos_idx = [paths.index(p) for p in c['positives']]
            short_recall.append(float(np.mean([p in set(sh) for p in pos_idx])))
            for name, s in V.items():
                ranked = [p for p in rank_of(paths, s) if p != q]
                m = harness.case_metrics(ranked, c['positives'], c['hard_negatives'])
                m.update(id=c['id'], synthetic=c['synthetic'])
                per.setdefault(name, []).append(m)
            for k, v in dict(embed=t2 - t1, maxsim_full=t3 - t2, coarse=t4 - t3, rerank=t5 - t4).items():
                lat.setdefault(k, []).append(v)
            print(task, c['id'], 'ap maxsim %.2f trim %.2f vote %.2f t=%.1fs' % (
                per['maxsim'][-1]['ap'], per['trim'][-1]['ap'], per['trim+vote'][-1]['ap'], t5 - t1), flush=True)
        rep = {}
        for name, rs in per.items():
            rep[name] = {sp: harness.summarize([r for r in rs if (sp == 'all' or r['synthetic'] == (sp == 'synthetic'))])
                         for sp in ('all', 'real', 'synthetic') if any((sp == 'all' or r['synthetic'] == (sp == 'synthetic')) for r in rs)}
        out[task] = dict(variants=rep, per_case=per,
                         latency_mean_s={k: float(np.mean(v)) for k, v in lat.items()},
                         shortlist_positive_recall=float(np.mean(short_recall)), cfg=CFG[task], n_images=len(paths),
                         n_tokens=ix.N, coarse_tokens=int(ix.coarse()[1][-1] + 1))
        print(f'\n[{task}] variant   split   R@1  R@5 R@10 mAP HN@10')
        for name, d in rep.items():
            for sp, m in d.items():
                print(f"{name:18s} {sp:9s} n={m['n']:3d} {m['recall@1']:.3f} {m['recall@5']:.3f} {m['recall@10']:.3f} {m['ap']:.3f} {m['hardneg_in_top10']:.3f}")
        print('latency', out[task]['latency_mean_s'], 'shortlist pos recall', out[task]['shortlist_positive_recall'])
    os.makedirs(os.path.join(BENCH, 'results'), exist_ok=True)
    fn = os.path.join(BENCH, 'results', f'e2_variants{a.tag}.json')
    json.dump(out, open(fn, 'w'), indent=1)
    print('wrote', fn)

main()
