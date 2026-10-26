"""E3 deep-texture variant (PATTERN): same dense-window pipeline as e3_texture but the
window descriptor is mean/std of early ResNet18 (layer2) + mean of layer3 features
(sqrt-compressed, standardised, whitened)."""
import numpy as np

from methods.e3_texture import E3Texture
from proto_e3 import deep


class E3Deep(E3Texture):
    mod = deep
    raw_name = 'raw_deep.npz'
    pre = staticmethod(lambda x: np.sqrt(np.maximum(x, 0)))

    def __init__(self):
        super().__init__('best')
        self.cfg = dict(whiten=0.3, bw=(1.0, 0.6, 1.0), agg='mean', qs=(1.0, 0.71, 0.5))
        self.name = 'e3_texture_deep'

    def warmup(self):
        deep.net()


METHOD = E3Deep()
