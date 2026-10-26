"""E2: DINOv2-S/14-reg dense patch tokens (PCA-64, int8) + query-pyramid MaxSim,
optional top-100 re-rank with offset voting / mutual-NN. Region and pattern.
Variant via $E2_VARIANT (maxsim|trim|pooled|trim+vote|maxsim+vote|vote|mnn|shortlist_trim)."""
import os

import numpy as np

from harness import Method, top_k
from proto_e2 import e2lib

CFG = {'region': dict(mults=(0.5, 0.7, 1.0, 1.4, 2.0), rots=(0,)),
       'pattern': dict(mults=(0.5, 0.7, 1.0, 1.4, 2.0), rots=(0, 90, 180, 270))}
SCRATCH = os.environ.get('E2_SCRATCH', '/var/home/rafa/.cache/e2_scratch')


class E2(Method):
    name = 'e2_dino_dense'
    kind = 'both'

    def __init__(self):
        self.variant = os.environ.get('E2_VARIANT', 'trim+vote')
        self.ix = None

    def warmup(self):
        pass

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        self.ix = e2lib.E2Index(cache_dir)
        self.ix.build(self.paths, SCRATCH)

    def query(self, image_path, bbox, k):
        ix, v = self.ix, self.variant
        ql = ix.query_tokens(image_path, bbox, **CFG[self.task])
        if v.startswith('shortlist'):
            co = ix.maxsim_scores(ql, coarse=True)
            full = ix.maxsim_scores(ql)
            base = v.split('_')[1]
            sh = np.argsort(-co[base])[:300]
            s = co[base] - 10
            s[sh] = full[base][sh]
            return top_k(self.paths, s, k)
        full = ix.maxsim_scores(ql)
        if v in full:
            return top_k(self.paths, full[v], k)
        cand = np.argsort(-full['maxsim'])[:100]
        mnn, vote = ix.rerank_scores(ql, cand)
        extra = {'vote': vote, 'mnn': mnn, 'maxsim+vote': full['maxsim'][cand] + vote,
                 'trim+vote': full['trim'][cand] + vote}[v]
        s = full['maxsim'] - 10
        s[cand] = extra + 10
        return top_k(self.paths, s, k)


METHOD = E2()
