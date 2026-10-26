"""Extra offline ablations: BoVW prefilter alone, and LBP(baseline)+E4 verifier score fusion."""
import json, pickle, sys, os
import numpy as np
import harness
from media_manager import pattern_descriptor
from proto_e4.method import fuse, PERIODIC_T

def main(pkl, out):
    d = pickle.load(open(pkl, 'rb')); paths = d['paths']
    idx = {p: i for i, p in enumerate(paths)}
    z = np.load(os.path.join(harness.CACHE_ROOT, 'baseline_pattern_lbp', 'tiles.npz'))
    M, starts = z['matrix'], z['starts']
    cases = [c for c in harness.load_cases(harness.DATA_DIR) if c['task'] == 'pattern']
    res = {}
    for c in cases:
        s = d['store'][c['id']]
        bgr = pattern_descriptor.decode_bgr(open(c['query_image'], 'rb').read())
        q = np.asarray(pattern_descriptor.descriptors_for_boxes(bgr, [list(c['bbox'])])[0], np.float32)
        lbp = harness.max_per_group(M @ q, starts)
        per = s['strength'] >= PERIODIC_T
        fz = fuse(per, s['ncc'], s['inl'])          # ~0..1.35
        sn = 1 - np.exp(-s['inl'] / 12.0)
        qi = idx[c['query_image']]
        cand = {'bovw_prefilter_only': s['pre'], 'lbp_baseline': lbp,
                'lbp+0.5*fused': lbp + 0.5 * fz, 'lbp+1.0*fused': lbp + 1.0 * fz,
                'lbp+0.5*sift': lbp + 0.5 * sn, 'lbp+0.2*fused': lbp + 0.2 * fz,
                'lbp+bovw': lbp + s['pre']}
        for name, sc in cand.items():
            sc = sc.copy(); sc[qi] = -1e9
            ranked = [paths[i] for i in np.argsort(-sc, kind='stable') if paths[i] != c['query_image']]
            m = harness.case_metrics(ranked, c['positives'], c['hard_negatives']); m['synthetic'] = c['synthetic']
            res.setdefault(name, []).append(m)
    rep = {}
    for k, v in res.items():
        row = {}
        for sp, f in (('all', lambda r: True), ('real', lambda r: not r['synthetic']), ('synthetic', lambda r: r['synthetic'])):
            rs = [r for r in v if f(r)]
            row[sp] = {m: float(np.mean([r[m] for r in rs])) for m in ('recall@1', 'recall@5', 'recall@10', 'ap', 'hardneg_in_top10', 'confusion@10')}
        rep[k] = row
        a = row['all']
        print(f"{k:22s} R@1 {a['recall@1']:.3f} R@5 {a['recall@5']:.3f} R@10 {a['recall@10']:.3f} mAP {a['ap']:.3f} | real {row['real']['ap']:.3f} syn {row['synthetic']['ap']:.3f} | HN@10 {a['hardneg_in_top10']:.2f} conf {a['confusion@10']:.3f}")
    json.dump(rep, open(out, 'w'), indent=1)
if __name__ == '__main__':
    main(*sys.argv[1:3])
