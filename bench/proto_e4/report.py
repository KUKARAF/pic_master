"""Offline ablations from analyze.py's raw scores -> bench/results/e4_ablation.json"""
import json, pickle, sys
import numpy as np
import harness
from proto_e4.method import fuse, PERIODIC_T

def rank_from(scores, paths):
    o = np.argsort(-scores, kind='stable')
    return [paths[i] for i in o]

def main(pkl, out):
    d = pickle.load(open(pkl, 'rb'))
    paths = d['paths']
    cases = [c for c in harness.load_cases(harness.DATA_DIR) if c['task'] == 'pattern']
    idx = {p: i for i, p in enumerate(paths)}
    variants = {}
    recall = {}
    for c in cases:
        s = d['store'][c['id']]
        ncc, inl, pre, per = s['ncc'], s['inl'], s['pre'], s['strength'] >= PERIODIC_T
        qi = idx[c['query_image']]
        sn = 1 - np.exp(-inl / 12.0)
        nn = np.clip((ncc - 0.3) / 0.6, 0, 1)
        cand = {
            'ncc_only': ncc, 'sift_only': inl + 1e-3 * ncc,
            'routed': (nn if per else sn) + 1e-3 * (sn if per else nn),
            'fused': fuse(per, ncc, inl),
            'max(nn,sn)': np.maximum(nn, sn) + 1e-3 * (nn + sn),
        }
        for name, sc in cand.items():
            for N in (None, 200, 100, 50):
                sc2 = sc.copy()
                if N is not None:
                    top = np.argsort(-pre)[:N]
                    m = np.full(len(sc), -10.0, np.float32); m += 0.5 * pre
                    m[top] = sc2[top] + 1.0
                    sc2 = m
                sc2 = sc2.copy(); sc2[qi] = -1e9
                ranked = [p for p in rank_from(sc2, paths) if p != c['query_image']]
                m_ = harness.case_metrics(ranked, c['positives'], c['hard_negatives'])
                m_.update(synthetic=c['synthetic'], periodic=bool(per), id=c['id'])
                variants.setdefault(f"{name}|{'ALL' if N is None else 'shortlist%d' % N}", []).append(m_)
        pre2 = pre.copy(); pre2[qi] = -1
        order = np.argsort(-pre2)
        for N in (50, 100, 200, 400):
            top = {paths[i] for i in order[:N]}
            recall.setdefault(N, []).append((c['synthetic'], np.mean([p in top for p in c['positives']])))
    rep = {'variants': {}, 'prefilter_recall': {}, 'n_periodic': sum(s['strength'] >= PERIODIC_T for s in d['store'].values())}
    for k, v in variants.items():
        row = {}
        for split, f in (('all', lambda r: True), ('real', lambda r: not r['synthetic']), ('synthetic', lambda r: r['synthetic']),
                         ('periodic', lambda r: r['periodic']), ('aperiodic', lambda r: not r['periodic'])):
            rs = [r for r in v if f(r)]
            if rs:
                row[split] = {'n': len(rs), **{m: float(np.mean([r[m] for r in rs])) for m in
                              ('recall@1', 'recall@5', 'recall@10', 'ap', 'mrr', 'hit@1', 'hardneg_in_top10', 'confusion@10')}}
        rep['variants'][k] = row
    for N, v in recall.items():
        rep['prefilter_recall'][N] = {'all': float(np.mean([x for _, x in v])),
                                      'real': float(np.mean([x for s, x in v if not s])),
                                      'synthetic': float(np.mean([x for s, x in v if s]))}
    json.dump(rep, open(out, 'w'), indent=1)
    for k, row in rep['variants'].items():
        a, r, s = row['all'], row.get('real', {}), row.get('synthetic', {})
        print(f"{k:28s} R@1 {a['recall@1']:.3f} R@5 {a['recall@5']:.3f} R@10 {a['recall@10']:.3f} mAP {a['ap']:.3f} "
              f"| real mAP {r.get('ap', 0):.3f} syn mAP {s.get('ap', 0):.3f} | HN@10 {a['hardneg_in_top10']:.2f} conf {a['confusion@10']:.3f}")
    print(json.dumps(rep['prefilter_recall']), 'periodic cases', rep['n_periodic'])

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
