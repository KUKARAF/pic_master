"""CLIP-based image indexer."""
import os
import numpy as np
from PIL import Image
import open_clip
import torch

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS


class CLIPIndexer:
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        self.model_name = f"{model_name}/{pretrained}"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_images(self, paths):
        """
        Return (embeddings, failed).
        embeddings: numpy float32 array of shape (N_success, D)
        failed: list of (path, message) tuples for paths that couldn't be loaded
        """
        failed = []
        all_embeddings = []

        batch_size = 32
        batch_tensors = []
        batch_paths = []

        def _process_batch(tensors, bpaths):
            if not tensors:
                return
            all_embeddings.append(self._encode_tensor_batch(tensors))

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                failed.append((path, f'unsupported extension: {ext}'))
                continue
            try:
                img = Image.open(path).convert('RGB')
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
                batch_paths.append(path)
            except Exception as exc:
                failed.append((path, str(exc)))
                continue

            if len(batch_tensors) >= batch_size:
                _process_batch(batch_tensors, batch_paths)
                batch_tensors = []
                batch_paths = []

        # process remaining
        _process_batch(batch_tensors, batch_paths)

        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            embeddings = np.empty((0,), dtype=np.float32)

        return embeddings, failed

    def _encode_tensor_batch(self, tensors):
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def embed_pil_images(self, images):
        """Embed in-memory PIL images (e.g. person crops — no on-disk path to hand
        to embed_images). Returns an L2-normalized float32 array of shape (N, D) in
        the same space as embed_images."""
        if not images:
            return np.empty((0,), dtype=np.float32)
        all_embeddings = []
        batch_size = 32
        for start in range(0, len(images), batch_size):
            tensors = [self.preprocess(img.convert('RGB')) for img in images[start:start + batch_size]]
            all_embeddings.append(self._encode_tensor_batch(tensors))
        return np.concatenate(all_embeddings, axis=0)

    def embed_text(self, text):
        """Return normalized float32 numpy vector for a text query."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()[0]

    @staticmethod
    def model_id(model_name='ViT-B-32', pretrained='openai'):
        return f"{model_name}/{pretrained}"
