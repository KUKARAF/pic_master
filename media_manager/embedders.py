"""Torch image embedders: SigLIP 2 (text<->image) and DINOv2-with-registers (image<->image).

GPU-ONLY: construction raises RuntimeError when no cuda/xpu device is usable (no CPU
fallback; MEDIA_DEVICE=cpu raises too). ``embed_shared([sig, dino], paths)`` decodes each
file once for both models. (``_allow_cpu=True`` exists only for scripts/bench_embedders.py.)

Both expose the same interface:

* ``embed_images(paths) -> (np.float32 (N, D) L2-normalized, failed[(path, msg)])``
* ``embed_pil_images(images) -> np.float32 (N, D)``
* ``model_id()`` (works as static call with the env-configured default, or on an instance)

Device comes from ``compute.torch_device()`` (cuda -> xpu -> mps -> cpu, override
``MEDIA_DEVICE``). On cuda/xpu inference runs under autocast (``MEDIA_EMBED_DTYPE``
= ``bf16`` (default) | ``fp16`` | ``fp32``); on cpu/mps it is fp32.

Decoding (PIL, EXIF-transposed, frame 0 only, RGB) + preprocessing run in a thread
pool that stays a few batches ahead of the accelerator, so decode overlaps inference.
PIL/numpy/torch resize release the GIL, so threads are enough.

SigLIP2 variant: env ``MEDIA_SIGLIP_MODEL`` = open_clip model name (default
``ViT-SO400M-14-SigLIP2``, pretrained ``webli``, 1152-d).
Optional ``MEDIA_SIGLIP_PRETRAINED``.

DINOv2 variant: env ``MEDIA_DINO_MODEL`` = ``small`` | ``base`` | ``large`` (default)
(torch.hub ``dinov2_vit{s,b,l}14_reg``, same weights/cache as object_encoder.py).
Output is the CLS token only (384/768/1024-d; default large = 1024-d), L2-normalized: the CLS token is what
DINOv2's own retrieval/kNN benchmarks use, is a single compact global descriptor,
and with registers is not polluted by high-norm artifact patches. (Concatenating
mean-patch tokens doubles storage for a small gain; not worth it for an index.)
Input is squashed to 224x224 (no center crop, so nothing at the borders is lost).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps

from . import compute
from .compute import GPU_LOCK
from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

SIGLIP_DEFAULT_MODEL = 'ViT-SO400M-14-SigLIP2'
SIGLIP_DEFAULT_PRETRAINED = 'webli'
SIGLIP_DIM = {'ViT-B-16-SigLIP2': 768, 'ViT-B-32-SigLIP2-256': 768, 'ViT-L-16-SigLIP2-256': 1024,
              'ViT-SO400M-14-SigLIP2': 1152}

# size key -> (hub entrypoint, dim)
DINO_VARIANTS = {
    'small': ('dinov2_vits14_reg', 384),
    'base': ('dinov2_vitb14_reg', 768),
    'large': ('dinov2_vitl14_reg', 1024),
}
DINO_DEFAULT = 'large'
DINO_INPUT = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# JPEG DCT-domain downscale ("draft mode"): libjpeg decodes at 1/2, 1/4 or 1/8 size for a
# fraction of the cost. The draft target is DRAFT_MARGIN x the largest model input, so the
# later (properly antialiased) resize in each model's preprocess still has >= 2x the pixels to
# work with. NOTE: this changes embeddings very slightly vs a full-resolution decode (cosine
# similarity >0.99 in practice); set MEDIA_EMBED_DRAFT=0 to disable and match full decode.
DRAFT_MARGIN = 2


def _draft_enabled() -> bool:
    return os.environ.get('MEDIA_EMBED_DRAFT', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def _decode_rgb(src, draft_size: int | None = None) -> Image.Image:
    """Open path (or take a PIL image), EXIF-orient, frame 0 only, RGB.

    ``draft_size`` (pixels): for JPEG files, ask libjpeg for a reduced-size decode whose
    shorter side is still >= this (see DRAFT_MARGIN note above). Applied before
    exif_transpose, which only needs the EXIF tag, so orientation is unaffected."""
    if isinstance(src, Image.Image):
        img = src
    else:
        img = Image.open(src)
        try:
            img.seek(0)  # animated gif/webp/apng: first frame only
        except EOFError:
            pass
        if draft_size and img.format == 'JPEG' and _draft_enabled():
            img.draft(None, (draft_size, draft_size))
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')


def _draft_target(embedders) -> int:
    """Draft decode target: DRAFT_MARGIN x the largest input side among the embedders."""
    sizes = [int(getattr(e, 'input_size', 0) or DINO_INPUT) for e in embedders]
    return DRAFT_MARGIN * max(sizes)


def _autocast_ctx(device: str):
    kind = os.environ.get('MEDIA_EMBED_DTYPE', 'bf16').lower()
    if device not in ('cuda', 'xpu') or kind == 'fp32':
        return torch.autocast(device_type='cpu', enabled=False)
    dtype = torch.float16 if kind == 'fp16' else torch.bfloat16
    return torch.autocast(device_type=device, dtype=dtype)


def embed_shared(embedders, sources, labels=None, stats=None, gpu_lock=None):
    """Decode each image ONCE and feed every embedder (e.g. SigLIP2 + DINOv2 in one pass).

    Returns ([emb_per_embedder (N_ok, D_i)], failed). Rows are aligned across embedders
    (an image failing to decode is dropped from all). Decode + each model's preprocessing
    run in a thread pool a few batches ahead of the GPU.

    Forward passes run under ``gpu_lock`` (default: the process-wide compute.GPU_LOCK) and
    ONLY those: decode never holds it, so concurrent callers overlap one's decode with
    another's GPU work. If ``stats`` (a dict) is given it accumulates ``decode_seconds``
    (summed over decode threads), ``gpu_seconds`` (inside the lock: forward + host sync) and
    ``lock_wait_seconds`` (time blocked acquiring the lock).
    """
    embedders = list(embedders)
    bs = min(e.batch_size for e in embedders)
    workers = max(e.num_workers for e in embedders)
    failed, outs = [], [[] for _ in embedders]
    window = bs * 3
    pending = deque()
    tens = [[] for _ in embedders]
    lock = GPU_LOCK if gpu_lock is None else gpu_lock
    draft = _draft_target(embedders)
    st = stats if stats is not None else {}
    for k in ('decode_seconds', 'gpu_seconds', 'lock_wait_seconds'):
        st.setdefault(k, 0.0)
    st_lock = threading.Lock()

    def job(src):
        t = time.perf_counter()
        try:
            img = _decode_rgb(src, draft)
            return [e._prep(img) for e in embedders]
        finally:
            with st_lock:
                st['decode_seconds'] += time.perf_counter() - t

    def flush():
        if tens[0]:
            t = time.perf_counter()
            with lock:
                t_in = time.perf_counter()
                for i, e in enumerate(embedders):
                    outs[i].append(e._run(tens[i]))
                t_out = time.perf_counter()
            st['lock_wait_seconds'] += t_in - t
            st['gpu_seconds'] += t_out - t_in
            for t_ in tens:
                t_.clear()

    def drain_one():
        label, fut = pending.popleft()
        try:
            res = fut.result()
        except Exception as exc:  # hook-allow: broad — per-image decode resilience, reported as failed
            failed.append((label, str(exc)))
            return
        for i, t in enumerate(res):
            tens[i].append(t)
        if len(tens[0]) >= bs:
            flush()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, src in enumerate(sources):
            label = labels[i] if labels else src
            if isinstance(src, str):
                ext = os.path.splitext(src)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    failed.append((label, f'unsupported extension: {ext}'))
                    continue
            pending.append((label, pool.submit(job, src)))
            while len(pending) >= window:
                drain_one()
        while pending:
            drain_one()
    flush()
    empty = np.empty((0,), dtype=np.float32)
    return [np.concatenate(o, axis=0) if o else empty for o in outs], failed


class _BaseEmbedder:
    batch_size: int
    device: str
    num_workers: int

    def _init_common(self, batch_size, num_workers, _allow_cpu=False):
        self.device = compute.torch_device()
        if self.device not in ('cuda', 'xpu') and not _allow_cpu:
            raise RuntimeError(
                f"{type(self).__name__}: no GPU available (torch_device()={self.device!r}). "
                "These embedders are GPU-only by design (no CPU fallback): install the XPU/CUDA "
                "torch build, or set MEDIA_DEVICE=xpu|cuda.")
        self.batch_size = int(batch_size or os.environ.get('MEDIA_EMBED_BATCH', 32))
        self.num_workers = int(num_workers or min(8, os.cpu_count() or 2))

    def _prep(self, img):
        raise NotImplementedError

    def _encode(self, batch):
        raise NotImplementedError

    def _smoke(self, model):
        raise NotImplementedError

    def _to_device(self, model):
        try:
            model = model.to(self.device).eval()
            if self.device in ('cuda', 'xpu'):  # real forward now so failures are loud at construction
                self._smoke(model)
            return model
        except Exception as exc:  # hook-allow: broad — re-raised with context
            raise RuntimeError(f"{type(self).__name__} failed on device {self.device!r}: {exc}") from exc

    def _run(self, tensors) -> np.ndarray:
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad(), _autocast_ctx(self.device):
            feats = self._encode(batch)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype(np.float32)

    def _embed_sources(self, sources, labels=None):
        outs, failed = embed_shared([self], sources, labels)
        return outs[0], failed

    def embed_images(self, paths):
        """Return (embeddings (N_ok, D) float32 L2-normalized, failed [(path, message)])."""
        return self._embed_sources(list(paths))

    def embed_pil_images(self, images):
        """In-memory PIL images -> (N, D) float32 L2-normalized, same space as embed_images."""
        images = list(images)
        if not images:
            return np.empty((0,), dtype=np.float32)
        emb, failed = self._embed_sources(images, labels=[f'<pil:{i}>' for i in range(len(images))])
        if failed:
            raise ValueError(f'embed_pil_images failed for {failed[:3]}')
        return emb


class SigLIP2Embedder(_BaseEmbedder):
    def __init__(self, model_name=None, pretrained=None, batch_size=None, num_workers=None, _allow_cpu=False):
        import open_clip  # hook-allow: lazy — heavy dep
        self.model_name = model_name or os.environ.get('MEDIA_SIGLIP_MODEL', SIGLIP_DEFAULT_MODEL)
        self.pretrained = pretrained or os.environ.get('MEDIA_SIGLIP_PRETRAINED', SIGLIP_DEFAULT_PRETRAINED)
        self._init_common(batch_size, num_workers, _allow_cpu)
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model = self._to_device(model)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        size = getattr(model.visual, 'image_size', DINO_INPUT)
        self.input_size = int(max(size) if isinstance(size, (tuple, list)) else size)
        self.dim = int(getattr(model, 'text_projection', None).shape[-1]) if getattr(model, 'text_projection', None) is not None \
            else SIGLIP_DIM.get(self.model_name, 0)
        print(f"[siglip2] {self.model_name}/{self.pretrained} device={self.device} bs={self.batch_size}", flush=True)

    def _prep(self, img):
        return self.preprocess(img)

    def _smoke(self, model):
        size = getattr(model.visual, 'image_size', (224, 224))
        size = (size, size) if isinstance(size, int) else tuple(size)
        with torch.no_grad(), _autocast_ctx(self.device):
            model.encode_image(torch.zeros(1, 3, *size, device=self.device))

    def _encode(self, batch):
        return self.model.encode_image(batch)

    def embed_text(self, text):
        """L2-normalized float32 (D,) vector in the image embedding space."""
        tokens = self.tokenizer([text]).to(self.device)
        with GPU_LOCK, torch.no_grad(), _autocast_ctx(self.device):
            feats = self.model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)[0]

    def embed_texts(self, texts):
        tokens = self.tokenizer(list(texts)).to(self.device)
        with GPU_LOCK, torch.no_grad(), _autocast_ctx(self.device):
            feats = self.model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    @staticmethod
    def model_id(model_name=None, pretrained=None):
        return (f"{model_name or os.environ.get('MEDIA_SIGLIP_MODEL', SIGLIP_DEFAULT_MODEL)}/"
                f"{pretrained or os.environ.get('MEDIA_SIGLIP_PRETRAINED', SIGLIP_DEFAULT_PRETRAINED)}")


class DINOv2Embedder(_BaseEmbedder):
    def __init__(self, variant=None, batch_size=None, num_workers=None, _allow_cpu=False):
        self.variant = (variant or os.environ.get('MEDIA_DINO_MODEL', DINO_DEFAULT)).lower()
        if self.variant not in DINO_VARIANTS:
            raise ValueError(f"MEDIA_DINO_MODEL={self.variant!r} not in {tuple(DINO_VARIANTS)}")
        self.hub_name, self.dim = DINO_VARIANTS[self.variant]
        self.input_size = DINO_INPUT
        self._init_common(batch_size, num_workers, _allow_cpu)
        local = os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')
        if os.path.isdir(local):  # reuse the cache, skip the GitHub round-trip
            model = torch.hub.load(local, self.hub_name, source='local')
        else:
            model = torch.hub.load('facebookresearch/dinov2', self.hub_name)
        self.model = self._to_device(model)
        from torchvision import transforms as T  # hook-allow: lazy — heavy dep
        self._tf = T.Compose([
            T.Resize((DINO_INPUT, DINO_INPUT), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
        print(f"[dinov2] {self.hub_name} dim={self.dim} device={self.device} bs={self.batch_size}", flush=True)

    def _prep(self, img):
        return self._tf(img)

    def _smoke(self, model):
        with torch.no_grad(), _autocast_ctx(self.device):
            model(torch.zeros(1, 3, DINO_INPUT, DINO_INPUT, device=self.device))

    def _encode(self, batch):
        out = self.model(batch)  # hub forward -> CLS token (B, dim)
        return out if out.dim() == 2 else out[:, 0]

    @staticmethod
    def model_id(variant=None):
        v = (variant or os.environ.get('MEDIA_DINO_MODEL', DINO_DEFAULT)).lower()
        return f"{DINO_VARIANTS[v][0]}/cls"
