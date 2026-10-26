"""Pool worker for methods/region_integrated.py (must live in an importable module)."""
import os

import numpy as np

from media_manager import region_index as ri


def sift_one(p):
    pil = ri.load_pil(open(p, 'rb').read())
    pts, desc = ri.rootsift(pil, nfeat=int(os.environ.get('RI_NFEAT', ri.SIFT_NFEAT)))
    s = min(3.0, ri.SIFT_LONG / max(pil.width, pil.height))
    return (pts * s).astype(np.float16), ri.quant_desc(desc)
