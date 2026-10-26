"""E3 fused (PATTERN): z-score late fusion of the classical window score and the deep-texture window score."""
import numpy as np

from harness import Method
from methods.e3_texture import E3Texture
from methods.e3_texture_deep import E3Deep

W_CLASSIC = 0.4


class E3Fused(Method):
    name = 'e3_texture_fused'
    kind = 'pattern'

    def __init__(self):
        self.a, self.b = E3Texture('best'), E3Deep()

    def warmup(self):
        self.b.warmup()

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        self.a.build_index(image_paths, cache_dir)
        self.b.build_index(image_paths, cache_dir)

    def query(self, image_path, bbox, k):
        n = len(self.paths)
        sa = dict(self.a.query(image_path, bbox, n))
        sb = dict(self.b.query(image_path, bbox, n))
        va = np.array([sa[p] for p in self.paths]); vb = np.array([sb[p] for p in self.paths])
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        from harness import top_k
        return top_k(self.paths, W_CLASSIC * z(va) + (1 - W_CLASSIC) * z(vb), k)


METHOD = E3Fused()
