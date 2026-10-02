"""DINOv2 region/crop embedder for INSTANCE-level object retrieval.

Where the CLIP tile index (``tile_index.py``) answers "what *kind* of thing is in
this crop" (a leaf, a car, a wall), this answers "find *this specific* object/region
again across the library" — same-instance similarity, not category similarity. For
that, DINOv2 (self-supervised ViT, Apache-2.0) is markedly stronger than CLIP: its
patch/token features cluster same-instance crops far more tightly than CLIP's
semantic-but-instance-blind embedding. We use the **dinov2-with-registers** weights
(``facebook/dinov2-with-registers-*``): the register tokens absorb the high-norm
attention artifacts of plain DINOv2, giving cleaner dense/patch features and a
slightly better global (CLS) descriptor for retrieval.

DINOv2 is a plain ViT, so it runs through the same accelerator paths as everything
else in this app. Two interchangeable backends, picked at load time (never silently
degraded — see "no silent failures"):

  * **OpenVINO IR** (preferred on the Intel Arc box): a pre-converted ``.xml``/``.bin``
    loaded through the OpenVINO Runtime and compiled to ``GPU`` (Intel XPU). Chosen
    when ``MEDIA_OBJECT_DINOV2_OV`` points at the IR directory.
  * **torch / torch-hub** fallback: ``facebookresearch/dinov2`` hub weights on the
    central compute device (``compute.torch_device()`` → cuda / xpu / mps / cpu).
    Chosen when no IR dir is configured.

Both take a fixed 224x224 ImageNet-normalized input (patch 14 → a 16x16 = 256-patch
grid, + 1 CLS + 4 register tokens = 261 sequence positions) and use the pooled CLS
token (768-d for ViT-B/14) as the crop descriptor, L2-normalized so retrieval is a
plain dot product — same convention as the CLIP indexer and pattern_descriptor.

The module imports on a CPU-only box with none of the ML runtime deps installed:
``numpy`` and ``PIL`` are baseline, and ``openvino`` / ``torch`` / ``transformers``
are imported lazily inside the loader. A missing runtime or missing weights raises a
loud ``RuntimeError`` carrying the exact ``pip`` / convert command — it never quietly
falls back to a weaker path.

====================================================================================
B70 SETUP RUNBOOK (Intel Arc B70, OpenVINO backend — preferred)
====================================================================================
1. Packages (into the main .venv):

       pip install "openvino>=2024.3" "optimum-intel[openvino]" transformers

   (``transformers`` is only needed for the one-time conversion below; the OpenVINO
   backend itself needs only ``openvino`` at runtime.)

2. Convert the dinov2-with-registers weights to OpenVINO IR (one-time, host with
   internet; DINOv2 is Apache-2.0 — do NOT use DINOv3, its licence is restrictive):

       optimum-cli export openvino \\
         --model facebook/dinov2-with-registers-base \\
         --task feature-extraction \\
         --weight-format fp16 \\
         /path/to/dinov2-reg-vitb14-ov

   Produces openvino_model.xml + openvino_model.bin in that directory. (Swap -base
   for -small / -large to match MEDIA_OBJECT_MODEL.)

3. Env vars:

       MEDIA_OBJECT_MODEL=dinov2-reg-vitb14          # backbone (default)
       MEDIA_OBJECT_DINOV2_OV=/path/to/dinov2-reg-vitb14-ov   # IR dir → OpenVINO
       MEDIA_OBJECT_DEVICE=GPU                        # OpenVINO device (GPU/CPU/AUTO)
       MEDIA_OBJECT_BATCH=16                          # GPU batch chunk size
       # MEDIA_OBJECT_BACKEND=openvino|torch          # optional hard override

4. Verify GPU placement:

       python -c "import openvino as ov; print(ov.Core().available_devices)"
       # expect e.g. ['CPU', 'GPU'] — GPU present = Arc is visible to OpenVINO.
       python -c "from media_manager.object_encoder import ObjectEncoder; \\
                  e=ObjectEncoder(); print(e.model_id(), e.device_label())"
       # device_label() should read 'GPU (Intel XPU/OpenVINO)'.

Torch fallback (hosts without the IR): leave MEDIA_OBJECT_DINOV2_OV unset and

       pip install torch torchvision        # +xpu wheels for Arc:
       #   --extra-index-url https://download.pytorch.org/whl/xpu

   The weights download from torch.hub on first use. device_label() then follows
   the central compute device (e.g. 'GPU (Intel XPU)').
====================================================================================
"""
import os

import numpy as np
from PIL import Image

# Fixed network geometry for the ViT-B/14 (and -S/-L/14) dinov2-with-registers
# backbones: a 224x224 input at patch 14 gives a 16x16 patch grid. Static, so the
# OpenVINO IR can be reshaped to a fixed shape and compiled once.
_RESOLUTION = 224
_PATCH = 14
_PATCH_GRID = _RESOLUTION // _PATCH   # 16  → 256 patch tokens (+1 CLS, +4 register)

# ImageNet statistics DINOv2 was trained with (RGB, 0..1 scale).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Supported backbones. Each maps the stable env name to the HF checkpoint (for the
# IR convert), the torch.hub entrypoint (fallback), and the descriptor dimension.
_BACKBONES = {
    'dinov2-reg-vits14': {'hf': 'facebook/dinov2-with-registers-small',
                          'hub': 'dinov2_vits14_reg', 'dim': 384},
    'dinov2-reg-vitb14': {'hf': 'facebook/dinov2-with-registers-base',
                          'hub': 'dinov2_vitb14_reg', 'dim': 768},
    'dinov2-reg-vitl14': {'hf': 'facebook/dinov2-with-registers-large',
                          'hub': 'dinov2_vitl14_reg', 'dim': 1024},
}

_DEFAULT_BACKBONE = 'dinov2-reg-vitb14'


class ObjectEncoder:
    """DINOv2 crop/region embedder. Heavy deps load lazily on first embed."""

    def __init__(self, data_root=None, device='GPU'):
        self.data_root = data_root
        self._backbone = os.environ.get('MEDIA_OBJECT_MODEL', _DEFAULT_BACKBONE).strip() \
            or _DEFAULT_BACKBONE
        if self._backbone not in _BACKBONES:
            raise ValueError(
                f"MEDIA_OBJECT_MODEL={self._backbone!r} is not a known backbone; "
                f"choose one of {sorted(_BACKBONES)}"
            )
        spec = _BACKBONES[self._backbone]
        self._hf_id = spec['hf']
        self._hub_name = spec['hub']
        self._dim = spec['dim']

        # Device string: OpenVINO device name ('GPU'/'CPU'/'AUTO') when on the IR
        # backend; for torch it's informational only (compute.torch_device() picks
        # the real one). Env overrides the constructor default.
        self._device = (os.environ.get('MEDIA_OBJECT_DEVICE') or device or 'GPU').strip()

        try:
            self._batch = max(1, int(os.environ.get('MEDIA_OBJECT_BATCH', '16')))
        except ValueError:
            self._batch = 16

        self._ir_dir = (os.environ.get('MEDIA_OBJECT_DINOV2_OV') or '').strip()
        # Backend chosen by explicit signal, NOT by "try openvino, fall back to
        # torch" (that would silently degrade). An IR dir configured (or a hard
        # MEDIA_OBJECT_BACKEND override) selects OpenVINO; otherwise torch-hub.
        override = (os.environ.get('MEDIA_OBJECT_BACKEND') or '').strip().lower()
        if override in ('openvino', 'torch'):
            self._backend = override
        elif self._ir_dir:
            self._backend = 'openvino'
        else:
            self._backend = 'torch'

        # Lazily populated by _ensure_loaded().
        self._loaded = False
        self._compiled = None          # OpenVINO compiled model
        self._ov_input = None          # OpenVINO input port
        self._ov_output = None         # OpenVINO output port for the descriptor
        self._ov_cls_slice = False     # True when we must slice CLS out of a seq output
        self._torch = None             # torch module handle
        self._torch_model = None
        self._torch_device = None

    # -- identity / status ----------------------------------------------------
    def model_id(self) -> str:
        """Stable id for the DB 'model' column (encodes backbone, the register
        variant, and the descriptor dim). Changing backbone/dim changes this, so
        stored vectors from a different model read as stale and get rebuilt."""
        return f"{self._backbone}/d{self._dim}"

    def dim(self) -> int:
        return self._dim

    def device_label(self) -> str:
        """Human label for job status — cheap, no heavy imports required."""
        if self._backend == 'openvino':
            dev = (self._device or '').upper()
            if dev.startswith('GPU'):
                return 'GPU (Intel XPU/OpenVINO)'
            if dev == 'CPU':
                return 'CPU (OpenVINO)'
            if dev == 'AUTO':
                return 'AUTO (OpenVINO)'
            return f'{self._device} (OpenVINO)'
        # torch backend: defer to the central compute label (cuda/xpu/mps/cpu).
        try:
            from . import compute
            return compute.device_label()
        except Exception:
            return 'CPU'

    # -- public embedding API -------------------------------------------------
    def embed_crops(self, images, boxes=None) -> list:
        """Embed a list of PIL images (the query exemplars).

        images: list of PIL.Image. If ``boxes`` is given it is a parallel list of
        ``(x1, y1, x2, y2)`` boxes in each image's own pixel coords (or None for a
        whole-image crop); each image is cropped to its box first. Returns a list
        ALIGNED 1:1 with ``images`` — each entry an L2-normalized float32 ndarray of
        length dim(), or None where that crop could not be prepared. Embedding runs
        in GPU batches of MEDIA_OBJECT_BATCH."""
        n = len(images)
        out = [None] * n
        tensors = []
        idxs = []
        for i, img in enumerate(images):
            try:
                crop = img
                if boxes is not None and boxes[i] is not None:
                    crop = _crop_to_box(img, boxes[i])
                tensors.append(self._preprocess(crop))
                idxs.append(i)
            except Exception:
                # Per-crop prep failure is a legitimate None (e.g. degenerate box);
                # a missing runtime/weights is NOT swallowed — that raises in
                # _ensure_loaded below.
                out[i] = None
        if not tensors:
            return out
        self._ensure_loaded()
        vecs = self._infer_batched(np.stack(tensors, axis=0))
        for j, i in enumerate(idxs):
            out[i] = vecs[j]
        return out

    def embed_regions(self, image) -> list:
        """Embed the deterministic CLIP-tile grid of one image.

        image: a PIL.Image or a path. Reuses tile_index.generate_tiles so the boxes
        match the CLIP tile grid convention (whole image + 2x2 + 3x3, ~50% overlap).
        Returns ``[ ((x1,y1,x2,y2) in ORIGINAL pixels, vec float32 L2-norm), ... ]``,
        dropping any tile whose crop failed. This is what the library indexing job
        calls per image."""
        from .tile_index import generate_tiles

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as im:
                img = im.convert('RGB')
        else:
            img = image.convert('RGB')

        boxes = generate_tiles(img.width, img.height)
        if not boxes:
            return []
        vecs = self.embed_crops([img] * len(boxes), boxes)
        return [(box, vec) for box, vec in zip(boxes, vecs) if vec is not None]

    # -- preprocessing --------------------------------------------------------
    def _preprocess(self, img):
        """PIL image → float32 CHW array at the fixed resolution, ImageNet-normalized."""
        rgb = img.convert('RGB').resize((_RESOLUTION, _RESOLUTION), Image.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0   # HWC, 0..1
        arr = np.transpose(arr, (2, 0, 1))                # CHW
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(arr, dtype=np.float32)

    # -- inference ------------------------------------------------------------
    def _infer_batched(self, arr):
        """Run the model over an (N, 3, H, W) float32 array in GPU-sized chunks and
        return an (N, dim) L2-normalized float32 matrix."""
        chunks = []
        batch = self._batch
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            if self._backend == 'openvino':
                feats = self._infer_openvino(chunk)
            else:
                feats = self._infer_torch(chunk)
            chunks.append(np.asarray(feats, dtype=np.float32))
        mat = np.concatenate(chunks, axis=0)
        return _l2_normalize_rows(mat)

    def _infer_openvino(self, chunk):
        # The IR was reshaped+compiled to a STATIC batch of self._batch, so a short
        # final chunk is zero-padded up to the batch and sliced back afterwards.
        m = len(chunk)
        if m < self._batch:
            pad = np.zeros((self._batch - m,) + chunk.shape[1:], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        result = self._compiled({self._ov_input: chunk})
        feats = result[self._ov_output]
        if self._ov_cls_slice:
            feats = feats[:, 0, :]
        return feats[:m]

    def _infer_torch(self, chunk):
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(chunk)).to(self._torch_device)
        with torch.no_grad():
            feats = self._torch_model(tensor)
        # dinov2 hub backbones return the CLS token (B, dim); guard a seq output.
        if hasattr(feats, 'ndim') and feats.ndim == 3:
            feats = feats[:, 0, :]
        return feats.float().cpu().numpy()

    # -- lazy model load ------------------------------------------------------
    def _ensure_loaded(self):
        if self._loaded:
            return
        if self._backend == 'openvino':
            self._load_openvino()
        else:
            self._load_torch()
        self._loaded = True

    def _load_openvino(self):
        if not self._ir_dir:
            raise RuntimeError(
                "ObjectEncoder OpenVINO backend selected but MEDIA_OBJECT_DINOV2_OV "
                "is not set. Point it at a converted IR directory, e.g.\n"
                f"  optimum-cli export openvino --model {self._hf_id} "
                "--task feature-extraction --weight-format fp16 "
                "/path/to/dinov2-reg-ov\n"
                "then set MEDIA_OBJECT_DINOV2_OV=/path/to/dinov2-reg-ov"
            )
        xml = _find_ir_xml(self._ir_dir)
        if xml is None:
            raise RuntimeError(
                f"No OpenVINO IR (.xml) found under MEDIA_OBJECT_DINOV2_OV="
                f"{self._ir_dir!r}. Convert the weights first:\n"
                f"  optimum-cli export openvino --model {self._hf_id} "
                "--task feature-extraction --weight-format fp16 "
                f"{self._ir_dir}"
            )
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO runtime is not installed (needed for the ObjectEncoder "
                "OpenVINO backend). Install it with:\n"
                "  pip install \"openvino>=2024.3\"\n"
                "or unset MEDIA_OBJECT_DINOV2_OV to use the torch fallback."
            ) from exc

        core = ov.Core()
        model = core.read_model(xml)
        self._ov_input = model.input(0).get_any_name()
        # Fix a static NCHW shape so the GPU plugin compiles one optimal kernel.
        model.reshape({self._ov_input: [self._batch, 3, _RESOLUTION, _RESOLUTION]})
        try:
            self._compiled = core.compile_model(model, self._device)
        except Exception as exc:
            avail = ''
            try:
                avail = f" Available OpenVINO devices: {core.available_devices}."
            except Exception:
                pass
            raise RuntimeError(
                f"OpenVINO failed to compile the DINOv2 IR to device "
                f"{self._device!r}: {exc}.{avail} Set MEDIA_OBJECT_DEVICE to an "
                "available device (e.g. CPU or AUTO)."
            ) from exc

        # Prefer the pooled CLS descriptor; else slice CLS from the sequence output.
        self._ov_output, self._ov_cls_slice = self._pick_ov_output(self._compiled)
        print(f"[object_encoder] backend=openvino device={self._device} "
              f"model={self._backbone} dim={self._dim} ir={xml}", flush=True)

    @staticmethod
    def _pick_ov_output(compiled):
        """Return (output_port, needs_cls_slice). pooler_output is the ready pooled
        CLS descriptor; last_hidden_state needs its first (CLS) token sliced out."""
        pooler = None
        seq = None
        for port in compiled.outputs:
            names = set(port.get_names())
            if 'pooler_output' in names:
                pooler = port
            if 'last_hidden_state' in names:
                seq = port
        if pooler is not None:
            return pooler, False
        if seq is not None:
            return seq, True
        # Single-output IR with no recognizable name: assume it's the pooled vector.
        return compiled.outputs[0], False

    def _load_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is not installed (needed for the ObjectEncoder torch "
                "fallback backend). Install it with:\n"
                "  pip install torch torchvision\n"
                "(Intel Arc: add --extra-index-url "
                "https://download.pytorch.org/whl/xpu)\n"
                "or set MEDIA_OBJECT_DINOV2_OV to use the OpenVINO backend."
            ) from exc

        from . import compute
        self._torch = torch
        self._torch_device = compute.torch_device()
        try:
            model = torch.hub.load('facebookresearch/dinov2', self._hub_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv2 weights '{self._hub_name}' from torch.hub "
                f"(facebookresearch/dinov2): {exc}. This needs network access on "
                "first use (weights are cached under ~/.cache/torch/hub), or convert "
                "to OpenVINO IR and set MEDIA_OBJECT_DINOV2_OV instead."
            ) from exc
        self._torch_model = model.to(self._torch_device).eval()
        print(f"[object_encoder] backend=torch device={self._torch_device} "
              f"model={self._backbone} dim={self._dim} hub={self._hub_name}",
              flush=True)


# -- module helpers -----------------------------------------------------------
def _crop_to_box(img, box):
    """Crop a PIL image to an (x1,y1,x2,y2) pixel box, clamped to the image and to
    at least a 1px span (matches the degenerate-crop guard in pattern_descriptor)."""
    w, h = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return img.crop((x1, y1, x2, y2))


def _l2_normalize_rows(mat):
    """Row-wise L2 normalization, zero-safe, float32 out."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _find_ir_xml(ir_dir):
    """Locate the IR topology .xml in a converted OpenVINO directory. optimum-intel
    names it openvino_model.xml; accept any single .xml as a fallback."""
    if not os.path.isdir(ir_dir):
        if os.path.isfile(ir_dir) and ir_dir.endswith('.xml'):
            return ir_dir
        return None
    preferred = os.path.join(ir_dir, 'openvino_model.xml')
    if os.path.isfile(preferred):
        return preferred
    xmls = sorted(f for f in os.listdir(ir_dir) if f.endswith('.xml'))
    return os.path.join(ir_dir, xmls[0]) if xmls else None
