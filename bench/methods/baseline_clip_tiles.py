"""Current production region search: CLIP ViT-B/32 over the overlapping tile grid
(tile_index.generate_tiles), photo score = best-tile cosine to the CLIP embedding of
the cropped query region (web.py api_region_search, tiles mode)."""
import os

import numpy as np
from PIL import Image

from harness import Method, max_per_group, top_k
from media_manager.tile_index import generate_tiles


class ClipTiles(Method):
    name = 'baseline_clip_tiles'
    kind = 'region'

    def __init__(self):
        self._clip = None

    def clip(self):
        if self._clip is None:
            from media_manager.indexer import CLIPIndexer  # pulls in torch/open_clip
            self._clip = CLIPIndexer()
        return self._clip

    def warmup(self):
        self.clip()

    def build_index(self, image_paths, cache_dir):
        self.paths = list(image_paths)
        f = os.path.join(cache_dir, 'tiles.npz')
        if not os.path.isfile(f):
            blocks = []
            for p in self.paths:
                with Image.open(p) as img:
                    img = img.convert('RGB')
                    crops = [img.crop(b) for b in generate_tiles(img.width, img.height)]
                blocks.append(self.clip().embed_pil_images(crops).astype(np.float32))
            counts = np.array([len(b) for b in blocks])
            np.savez(f, matrix=np.concatenate(blocks), starts=np.cumsum(counts) - counts)
        z = np.load(f)
        self.matrix, self.starts = z['matrix'], z['starts']

    def query(self, image_path, bbox, k):
        with Image.open(image_path) as im:
            crop = im.convert('RGB').crop(tuple(float(v) for v in bbox))
            crop.load()
        q = self.clip().embed_pil_images([crop])[0].astype(np.float32)
        scores = max_per_group(self.matrix.dot(q), self.starts)
        return top_k(self.paths, scores, k)


METHOD = ClipTiles()
