"""CLIP-based image indexer."""
import os
import numpy as np
from PIL import Image
import open_clip
import torch

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


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
        Return (embeddings, failed_paths).
        embeddings: numpy float32 array of shape (N_success, D)
        failed_paths: list of paths that couldn't be loaded
        """
        failed_paths = []
        all_embeddings = []

        batch_size = 32
        batch_tensors = []
        batch_paths = []

        def _process_batch(tensors, bpaths):
            if not tensors:
                return
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().float().numpy())

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                failed_paths.append(path)
                continue
            try:
                img = Image.open(path).convert('RGB')
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
                batch_paths.append(path)
            except Exception:
                failed_paths.append(path)
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

        return embeddings, failed_paths

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
