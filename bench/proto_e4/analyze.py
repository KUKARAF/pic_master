"""Run the verify-all variant on every pattern case, save raw per-image scores, and compute
metrics for ablations (prefilter shortlist recall, ncc-only, sift-only, routed, fused)."""
import json, os, sys, time, pickle
import numpy as np
import harness
from proto_e4 import method, core

def main(out):
    m = method.E4()
    m.verify_all = True
    paths = harness.corpus_paths(harness.DATA_DIR)
    m.build_index(paths, '/tmp/x')
    cases = [c for c in harness.load_cases(harness.DATA_DIR) if c['task'] == 'pattern']
    store = {}
    lat = []
    for c in cases:
        t = time.perf_counter()
        m.query(c['query_image'], c['bbox'], 10)
        lat.append(time.perf_counter() - t)
        d = m.diag[(c['query_image'], tuple(c['bbox']))]
        store[c['id']] = d
        print(c['id'], 'periodic %.2f' % d['strength'], '%.1fs' % lat[-1], flush=True)
    with open(out, 'wb') as f:
        pickle.dump(dict(paths=paths, store=store, lat=lat), f)

if __name__ == '__main__':
    main(sys.argv[1])
