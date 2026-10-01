"""Visual Place Recognition image encoder (same-location retrieval, people-ignored).

This produces ONE L2-normalized float32 descriptor per image that answers "were
these two photos taken in the same *place*?" — same room, same building, same
view — while deliberately ignoring the *people* in the frame. It mirrors the
``CLIPIndexer`` interface (``media_manager/indexer.py``) so it drops straight into
the existing remote-or-local factory: ``embed_images(paths)`` /
``embed_pil_images(images)`` -> ``(embeddings, failed)``, plus ``model_id()`` and
``dim()`` for the DB ``model`` column and index sizing.

Primary backend = **AnyLoc** (DINOv2 ViT-g patch tokens + VLAD aggregation). AnyLoc
is unsupervised and domain-general, so it holds up on the indoor/home imagery this
library skews toward far better than a VPR net trained on street-level panoramas.

How "ignore people" works (AnyLoc-native, no retraining):
  * DINOv2 turns the image into a fixed grid of patch tokens (see resolution below).
  * VLAD is a permutation-invariant aggregation over a *set* of tokens, so dropping
    a subset leaves the result in-distribution — unlike masking pixels, which
    injects a black rectangle the backbone has never seen.
  * So: map each ORIGINAL-PIXEL person box to the grid cells it covers, DROP those
    tokens, and VLAD-aggregate ONLY the remaining (background) tokens.

A second, simpler **eigenplaces** backend (torch.hub ``gmberton/eigenplaces``,
ResNet-50, 2048-D GeM global descriptor, NO token dropping) exists only to bring up
and validate the Intel Arc Pro B70 (Battlemage) serving path end to end before
trusting the full AnyLoc graph. Backends sit behind a small internal dispatch so
more can be added.

================================================================================
B70 / Battlemage SETUP RUNBOOK (OpenVINO serving path — the production target)
================================================================================
torch 2.7.1+xpu exists but is flaky for serving; prefer OpenVINO. This module runs
BOTH on the prod/web host and inside the bigboy worker, so do this on each.

1. Runtime deps (into the PROJECT venv, not ~/ — see the venv-location memory):

     pip install openvino>=2025.0 onnxruntime-openvino>=1.19
     # conversion-only (can be a throwaway box with a GPU or even CPU):
     pip install optimum-intel[openvino]>=1.21 transformers torch

   onnxruntime-openvino is what compute.onnx_providers() already uses for faces;
   installing it also makes the OpenVINOExecutionProvider visible there. This
   module itself talks to OpenVINO directly via the ``openvino`` Runtime (full
   control over static shapes + explicit GPU targeting), and only needs
   ``optimum-intel`` at *conversion* time.

2. Convert DINOv2 ViT-g to OpenVINO IR (one-time; commit the IR to the model store,
   NOT to git). AnyLoc's SOTA config is the GIANT model (1536-D tokens):

     optimum-cli export openvino \
        --model facebook/dinov2-giant \
        --task image-feature-extraction \
        --weight-format fp16 \
        <data_root>/.media/dinov2_ov

   That writes openvino_model.xml / .bin. Override the location with
   MEDIA_PLACE_DINOV2_OV if you keep it elsewhere. (Intel also publishes
   'OpenVINO/dino_v2-fp16-ov' on the Hub, but confirm it is the *giant* variant and
   exposes the full patch-token sequence before using it — the small/base variants
   have 384/768-D tokens and will NOT match a giant-fit vocabulary.)

3. Fixed input resolution / static shapes. The OpenVINO GPU plugin wants static
   shapes, so we reshape the graph to a FIXED 1x3x518x518 input. DINOv2 patch size
   is 14, so 518/14 = 37 -> a 37x37 = 1369 patch-token grid (plus the CLS token we
   drop). This resolution and grid are baked into model_id(); changing them is a
   version change. The whole image is resized (anisotropically) to 518x518, which
   is why the person-box -> grid-cell math below is a plain linear rescale with no
   crop to undo.

4. VLAD vocabulary (cluster centers). AnyLoc ships per-DOMAIN vocabularies
   ('indoor', 'urban', 'aerial'); 'indoor' is the right one for a home library.
   Download AnyLoc's 'indoor' c_centers for ViT-g / layer 31 / facet 'value' /
   32 clusters from the AnyLoc release ("vocabulary" / "cluster-centers" assets,
   https://github.com/AnyLoc/AnyLoc), OR fit your own over a sample of the library
   (extract tokens for ~1-5k images, k-means to 32 centers). Store it as:

     <data_root>/.media/place_vocab/c_centers.pt      (torch tensor [32, 1536]), or
     <data_root>/.media/place_vocab/c_centers.npy     (numpy, no torch needed)

   CRITICAL: the vocabulary MUST be fit with the SAME DINOv2 facet+layer this module
   serves. A plain optimum IR export exposes the final-block TOKEN facet
   (last_hidden_state), which is NOT AnyLoc's default 'value' facet. So either (a)
   fit your vocabulary over the token facet of THIS exact IR (recommended — keeps
   the serving graph a stock export), or (b) export a custom IR that returns the
   layer-31 value facet and fit the vocab to match. MEDIA_PLACE_FACET records which
   facet is in play and is folded into model_id() so a mismatch is detectable in the
   DB. Missing vocab raises a LOUD error with the fix — it never silently degrades.

5. Optional PCA-whitening (AnyLoc raw dim is 32*1536 = 49152, too wide for a flat
   index). If <data_root>/.media/place_pca.npz exists (keys 'mean' [D_raw] and
   'components' [D_out, D_raw]), it is applied after VLAD, then L2-normalized; dim()
   then reports D_out (e.g. 2048). Absent -> raw 49152-D, L2-normalized. Fit PCA
   with sklearn over a sample of VLAD descriptors and np.savez it.

6. Env vars:
     MEDIA_PLACE_MODEL    'anyloc' (default) | 'eigenplaces'
     MEDIA_PLACE_DEVICE   OpenVINO device for anyloc: 'GPU' (default) | 'CPU' | 'AUTO';
                          for eigenplaces this is ignored (it uses compute.torch_device()).
     MEDIA_PLACE_DINOV2_OV   path to the DINOv2 OpenVINO IR dir (default
                             <data_root>/.media/dinov2_ov)
     MEDIA_PLACE_FACET    descriptor facet tag folded into model_id (default 'token')

7. Verify the graph actually runs on the GPU (no silent CPU fallback). We compile
   with the EXPLICIT device name (not 'AUTO'), so a missing GPU plugin raises here
   instead of quietly running on CPU. On construction this logs, e.g.:

     [place] anyloc dinov2_ov device=GPU exec=GPU.0 grid=37x37 dim=2048

   If 'exec' comes back 'CPU' while you asked for 'GPU', treat that as a failure to
   investigate (bad driver / Level-Zero / compute-runtime), not an acceptable
   fallback. `clinfo` / `python -c "import openvino; print(openvino.Core().available_devices)"`
   should list 'GPU'.

Per the project's "no silent failures" rule every missing-runtime / missing-weights
path raises a clear error carrying the exact pip/convert command to fix it, rather
than degrading to a different model or to zeros.
"""
from __future__ import annotations

import os

import numpy as np

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

# --- Fixed AnyLoc/DINOv2 serving geometry (baked into model_id; see runbook §3) ---
INPUT_RES = 518          # square input fed to the OpenVINO graph
PATCH = 14               # DINOv2 ViT patch size
GRID = INPUT_RES // PATCH  # 37 -> 37x37 = 1369 patch tokens
DINOV2_GIANT_DIM = 1536  # ViT-g token dimensionality

# ImageNet normalization — DINOv2's preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _default_data_root(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Mirror the app's convention: callers pass data_root; fall back to CWD so a
    # bare bring-up still has a place to look for .media/.
    return os.environ.get("MEDIA_DATA_ROOT", os.getcwd())


class PlaceEncoder:
    """Place/VPR encoder with a people-ignoring AnyLoc backend. Interface mirrors
    CLIPIndexer so it slots into the remote-or-local factory."""

    def __init__(self, model_name: str = None, device: str = "GPU", data_root: str = None):
        # model_name: backend id. Default from env, else 'anyloc'.
        self.model_name = (model_name or _env("MEDIA_PLACE_MODEL", "anyloc")).lower()
        # device: OpenVINO device name for anyloc ('GPU'/'CPU'/'AUTO'). The ctor
        # default is 'GPU' (the B70 target); env wins over the arg so deployments
        # can flip it without code changes.
        self.device = _env("MEDIA_PLACE_DEVICE", device or "GPU")
        self.data_root = _default_data_root(data_root)
        self._media_dir = os.path.join(self.data_root, ".media")
        self.facet = _env("MEDIA_PLACE_FACET", "token")

        # Lazy state — nothing heavy is imported or loaded here, so this ctor is
        # safe on a CPU-only dev box with none of the deps installed.
        self._loaded = False
        self._backend = None          # 'anyloc' | 'eigenplaces'
        self._ov_compiled = None      # OpenVINO CompiledModel (anyloc)
        self._ov_out_port = None
        self._centers = None          # np.ndarray [K, D] (anyloc)
        self._num_c = None
        self._desc_dim = None
        self._pca_mean = None         # np.ndarray [D_raw] or None
        self._pca_components = None   # np.ndarray [D_out, D_raw] or None
        self._torch_model = None      # eigenplaces model
        self._torch_device = None
        self._eigen_dim = 2048

        if self.model_name not in ("anyloc", "eigenplaces"):
            raise ValueError(
                f"MEDIA_PLACE_MODEL={self.model_name!r} is not a known place backend "
                f"(expected 'anyloc' or 'eigenplaces')."
            )

    # ------------------------------------------------------------------ metadata

    def model_id(self) -> str:
        """Stable string for the DB 'model' column. Encodes backend, geometry,
        facet, cluster count and PCA width so any version-affecting change (new
        resolution, different facet, re-fit PCA) produces a different id and the
        index can detect staleness. Cheap to call before load — reads only files."""
        if self.model_name == "eigenplaces":
            base = "eigenplaces-r50-gem-2048"
            return self._with_pca_suffix(base, self._eigen_pca_out_dim())
        # anyloc
        num_c = self._num_c if self._num_c is not None else self._peek_num_c()
        base = (
            f"anyloc-dinov2-vitg-vlad-c{num_c}"
            f"-{self.facet}-{INPUT_RES}px-g{GRID}"
        )
        return self._with_pca_suffix(base, self._pca_out_dim())

    def _with_pca_suffix(self, base: str, pca_out: int | None) -> str:
        return f"{base}-pca{pca_out}" if pca_out else base

    def dim(self) -> int:
        """Descriptor dimensionality AFTER any PCA (so callers can size the index).
        AnyLoc raw = num_c * desc_dim (~49152 for c32/ViT-g); PCA collapses that to
        'components' rows. Reads the PCA/vocab files without loading the model."""
        if self.model_name == "eigenplaces":
            return self._eigen_pca_out_dim() or self._eigen_dim
        pca_out = self._pca_out_dim()
        if pca_out:
            return pca_out
        num_c = self._num_c if self._num_c is not None else self._peek_num_c()
        return num_c * DINOV2_GIANT_DIM

    # ------------------------------------------------------------- public embed

    def embed_images(self, paths: list, person_boxes=None) -> tuple:
        """Mirror CLIPIndexer.embed_images EXACTLY (plus person_boxes).

        Returns ``(embeddings, failed)`` where ``embeddings`` is a list of
        np.ndarray (float32, L2-normalized, length == dim()) — one per SUCCESSFUL
        image, in input order — and ``failed`` is a list of ``(path, reason)`` for
        images that could not be processed.

        ``person_boxes``, when given, is parallel to ``paths``: ``person_boxes[i]``
        is a list of ``(x1, y1, x2, y2)`` ORIGINAL-PIXEL person rectangles for image
        ``i`` whose overlapping DINOv2 patch tokens are dropped before VLAD. None or
        [] means "use all tokens".
        """
        from PIL import Image

        self._ensure_loaded()
        embeddings: list = []
        failed: list = []
        for i, path in enumerate(paths):
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                failed.append((path, f"unsupported extension: {ext}"))
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:  # loud per-item failure, keeps going
                failed.append((path, str(exc)))
                continue
            boxes = self._boxes_for(person_boxes, i)
            try:
                embeddings.append(self._embed_one(img, boxes))
            except Exception as exc:
                failed.append((path, str(exc)))
        return embeddings, failed

    def embed_pil_images(self, images: list, person_boxes=None) -> tuple:
        """Same contract as embed_images, for already-loaded PIL images (the worker
        ships raw bytes, decoded to PIL upstream). ``failed`` entries use the image
        index as their 'path' since there is no on-disk path."""
        self._ensure_loaded()
        embeddings: list = []
        failed: list = []
        for i, img in enumerate(images):
            boxes = self._boxes_for(person_boxes, i)
            try:
                rgb = img.convert("RGB") if img.mode != "RGB" else img
                embeddings.append(self._embed_one(rgb, boxes))
            except Exception as exc:
                failed.append((i, str(exc)))
        return embeddings, failed

    @staticmethod
    def _boxes_for(person_boxes, i):
        if not person_boxes:
            return None
        if i >= len(person_boxes):
            return None
        return person_boxes[i]

    # --------------------------------------------------------------- dispatch

    def _embed_one(self, pil_img, boxes):
        if self._backend == "anyloc":
            return self._embed_anyloc(pil_img, boxes)
        if self._backend == "eigenplaces":
            return self._embed_eigenplaces(pil_img)
        raise RuntimeError(f"place backend {self._backend!r} not loaded")

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.model_name == "anyloc":
            self._load_anyloc()
            self._backend = "anyloc"
        else:
            self._load_eigenplaces()
            self._backend = "eigenplaces"
        self._loaded = True

    # ============================================================== AnyLoc path

    def _ir_dir(self) -> str:
        return _env("MEDIA_PLACE_DINOV2_OV", os.path.join(self._media_dir, "dinov2_ov"))

    def _vocab_dir(self) -> str:
        return os.path.join(self._media_dir, "place_vocab")

    def _pca_path(self) -> str:
        return os.path.join(self._media_dir, "place_pca.npz")

    def _load_anyloc(self):
        try:
            import openvino as ov
        except Exception as exc:
            raise RuntimeError(
                "AnyLoc place backend needs the OpenVINO runtime, which is not "
                "importable (%r). Install it into the project venv:\n"
                "    pip install openvino>=2025.0 onnxruntime-openvino>=1.19" % (exc,)
            )

        # --- locate the DINOv2 OpenVINO IR ---
        ir_dir = self._ir_dir()
        xml = self._find_ir_xml(ir_dir)

        # --- load vocabulary first (so a missing vocab fails before we spend time
        #     compiling the graph) ---
        self._centers = self._load_centers()
        self._num_c, self._desc_dim = self._centers.shape
        if self._desc_dim != DINOV2_GIANT_DIM:
            raise RuntimeError(
                f"VLAD vocabulary descriptor dim is {self._desc_dim} but this module "
                f"serves DINOv2 ViT-g ({DINOV2_GIANT_DIM}-D) tokens. The vocabulary "
                f"was fit for a different backbone — re-fit it for dinov2-giant or "
                f"point MEDIA_PLACE_DINOV2_OV at the matching IR."
            )

        # --- optional PCA ---
        self._load_pca_if_present()

        # --- compile the graph with an EXPLICIT device (no silent CPU fallback) ---
        core = ov.Core()
        try:
            model = core.read_model(xml)
        except Exception as exc:
            raise RuntimeError(f"Failed to read DINOv2 OpenVINO IR at {xml!r}: {exc}")
        # Static shape for the GPU plugin: single image input [1,3,518,518].
        try:
            model.reshape([1, 3, INPUT_RES, INPUT_RES])
        except Exception as exc:
            raise RuntimeError(
                f"Could not reshape the DINOv2 IR to a static [1,3,{INPUT_RES},"
                f"{INPUT_RES}] input ({exc}). The export must have a single image "
                f"input; re-export with 'optimum-cli export openvino --model "
                f"facebook/dinov2-giant --task image-feature-extraction ...'."
            )
        avail = list(core.available_devices)
        want = self.device.split(".")[0].upper()
        if want not in ("AUTO",) and want not in {d.split(".")[0].upper() for d in avail}:
            raise RuntimeError(
                f"MEDIA_PLACE_DEVICE={self.device!r} but OpenVINO reports no such "
                f"device. Available: {avail}. For the B70, 'GPU' must be present — "
                f"check the Intel GPU driver / Level-Zero / compute-runtime. Refusing "
                f"to silently run on CPU."
            )
        try:
            self._ov_compiled = core.compile_model(model, self.device)
        except Exception as exc:
            raise RuntimeError(
                f"OpenVINO failed to compile the DINOv2 graph on device "
                f"{self.device!r}: {exc}. (available={avail})"
            )
        self._ov_out_port = self._pick_patch_output(self._ov_compiled)

        exec_dev = self._exec_devices(self._ov_compiled)
        print(
            f"[place] anyloc dinov2_ov device={self.device} exec={exec_dev} "
            f"grid={GRID}x{GRID} dim={self.dim()}",
            flush=True,
        )

    @staticmethod
    def _find_ir_xml(ir_dir: str) -> str:
        if not os.path.isdir(ir_dir):
            raise RuntimeError(
                f"DINOv2 OpenVINO IR directory not found: {ir_dir!r}. Convert it once:\n"
                "    pip install optimum-intel[openvino]\n"
                "    optimum-cli export openvino --model facebook/dinov2-giant "
                "--task image-feature-extraction --weight-format fp16 "
                f"{ir_dir}\n"
                "or set MEDIA_PLACE_DINOV2_OV to an existing IR directory."
            )
        # optimum writes openvino_model.xml; accept any single .xml otherwise.
        cand = os.path.join(ir_dir, "openvino_model.xml")
        if os.path.isfile(cand):
            return cand
        xmls = [f for f in os.listdir(ir_dir) if f.endswith(".xml")]
        if len(xmls) == 1:
            return os.path.join(ir_dir, xmls[0])
        if not xmls:
            raise RuntimeError(
                f"No .xml OpenVINO IR found in {ir_dir!r}. Re-run the optimum-cli "
                f"export (see place_encoder runbook §2)."
            )
        raise RuntimeError(
            f"Multiple .xml files in {ir_dir!r} ({xmls}); keep only the DINOv2 IR or "
            f"point MEDIA_PLACE_DINOV2_OV at the exact file's directory."
        )

    def _load_centers(self) -> "np.ndarray":
        vdir = self._vocab_dir()
        npy = os.path.join(vdir, "c_centers.npy")
        pt = os.path.join(vdir, "c_centers.pt")
        if os.path.isfile(npy):
            arr = np.load(npy)
        elif os.path.isfile(pt):
            try:
                import torch
            except Exception as exc:
                raise RuntimeError(
                    f"VLAD vocabulary {pt!r} is a torch tensor but torch is not "
                    f"installed to read it ({exc}). Either install torch, or convert "
                    f"it once to numpy:\n"
                    f"    python -c \"import torch,numpy as np;"
                    f"np.save('{npy}', torch.load('{pt}', map_location='cpu')"
                    f".float().cpu().numpy())\""
                )
            t = torch.load(pt, map_location="cpu")
            arr = t.float().cpu().numpy()
        else:
            raise RuntimeError(
                f"VLAD vocabulary not found. Expected {npy!r} or {pt!r}. Download "
                f"AnyLoc's 'indoor' ViT-g cluster centers (c32, layer31/value) from "
                f"https://github.com/AnyLoc/AnyLoc releases, or fit your own over a "
                f"sample of the library, and place it there. See runbook §4. "
                f"Refusing to run without a vocabulary."
            )
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(
                f"VLAD vocabulary must be 2-D [num_clusters, desc_dim]; got shape "
                f"{arr.shape} from the vocab file."
            )
        return arr

    def _load_pca_if_present(self):
        p = self._pca_path()
        if not os.path.isfile(p):
            return
        data = np.load(p)
        if "mean" not in data or "components" not in data:
            raise RuntimeError(
                f"{p!r} exists but is missing required keys 'mean' and 'components'. "
                f"Re-fit the PCA and np.savez(mean=..., components=...)."
            )
        self._pca_mean = np.ascontiguousarray(data["mean"], dtype=np.float32)
        self._pca_components = np.ascontiguousarray(data["components"], dtype=np.float32)
        raw = self._num_c * self._desc_dim if self._num_c else None
        if raw is not None and self._pca_components.shape[1] != raw:
            raise RuntimeError(
                f"PCA components expect input dim {self._pca_components.shape[1]} but "
                f"VLAD raw dim is {raw}. The PCA was fit for a different VLAD config; "
                f"re-fit it against this vocabulary."
            )

    @staticmethod
    def _pick_patch_output(compiled):
        """Pick the output port that carries the patch-token sequence. optimum's
        image-feature-extraction export names it 'last_hidden_state'; fall back to
        the first 3-D output."""
        try:
            for port in compiled.outputs:
                names = set(port.get_names()) if hasattr(port, "get_names") else set()
                if any("last_hidden_state" in n for n in names):
                    return port
        except Exception:
            pass
        # fall back to the single/first output
        return compiled.output(0)

    @staticmethod
    def _exec_devices(compiled) -> str:
        try:
            return ",".join(compiled.get_property("EXECUTION_DEVICES"))
        except Exception:
            return "?"

    def _embed_anyloc(self, pil_img, boxes):
        import openvino as ov  # noqa: F401  (ensure present; lazy)

        orig_w, orig_h = pil_img.size
        inp = self._preprocess_dino(pil_img)  # [1,3,518,518] float32

        result = self._ov_compiled({0: inp})
        tokens = result[self._ov_out_port]
        tokens = np.asarray(tokens)
        patch = self._patch_tokens(tokens)  # [GRID*GRID, D] in row-major grid order

        keep = self._keep_mask(boxes, orig_w, orig_h)  # bool [GRID*GRID]
        if keep is not None:
            kept = patch[keep]
            # If person boxes cover the ENTIRE frame there is no background left.
            # Producing an empty VLAD would be a silent garbage descriptor, so we
            # fall back to all tokens (and the caller still gets a valid vector).
            if kept.shape[0] >= 1:
                patch = kept

        vlad = self._vlad(patch)                 # [num_c * desc_dim] float32
        desc = self._apply_pca_and_norm(vlad)    # [dim()] L2-normalized float32
        return desc

    def _preprocess_dino(self, pil_img):
        # Resize the WHOLE image to a fixed 518x518 square (anisotropic). This is
        # what makes the person-box -> grid mapping a plain linear rescale: there is
        # no crop to invert, each original pixel x maps to x * 518/W. For VPR the
        # mild aspect distortion is harmless and it keeps the token grid fixed at
        # 37x37, which the OpenVINO GPU plugin requires.
        img = pil_img.resize((INPUT_RES, INPUT_RES))
        a = np.asarray(img, dtype=np.float32) / 255.0  # HWC, RGB
        a = (a - _IMAGENET_MEAN) / _IMAGENET_STD
        a = np.transpose(a, (2, 0, 1))  # CHW
        return np.ascontiguousarray(a[None, ...], dtype=np.float32)

    def _patch_tokens(self, tokens):
        """From the raw model output -> [GRID*GRID, D] patch tokens, CLS/register
        tokens stripped. Row-major (patch row by row), matching how DINOv2 flattens
        the grid, so token index == row*GRID + col."""
        if tokens.ndim == 3:
            tokens = tokens[0]  # [T, D]
        if tokens.ndim != 2:
            raise RuntimeError(
                f"Unexpected DINOv2 output rank; got shape {tokens.shape}, expected "
                f"[1, tokens, dim]."
            )
        n = GRID * GRID
        t = tokens.shape[0]
        if t == n:
            patch = tokens
        elif t == n + 1:
            patch = tokens[1:]            # drop CLS
        elif t == n + 1 + 4:
            patch = tokens[1 + 4:]        # drop CLS + 4 register tokens (DINOv2-reg)
        elif t > n:
            patch = tokens[t - n:]        # assume leading extras, keep trailing grid
        else:
            raise RuntimeError(
                f"DINOv2 produced {t} tokens but the {GRID}x{GRID} grid needs {n} "
                f"patch tokens. Is the IR built for a {INPUT_RES}px input and patch "
                f"size {PATCH}?"
            )
        return np.ascontiguousarray(patch, dtype=np.float32)

    @staticmethod
    def _keep_mask(boxes, orig_w, orig_h):
        """Build a bool mask over the GRID*GRID patch tokens, False where a token
        cell overlaps any person box. Returns None when nothing to drop.

        Coordinate math (see _preprocess_dino): the full image is rescaled to
        518x518, so original pixel x -> input pixel x * 518/orig_w (and y likewise).
        A patch-grid column c spans input x in [c*14, (c+1)*14). A box therefore
        touches columns floor(ix1/14) .. ceil(ix2/14)-1 (inclusive), rows likewise.
        We drop every cell in that rectangle (overlap, not containment: any token
        whose cell the box touches at all)."""
        if not boxes:
            return None
        if orig_w <= 0 or orig_h <= 0:
            return None
        keep = np.ones(GRID * GRID, dtype=bool)
        sx = INPUT_RES / float(orig_w)
        sy = INPUT_RES / float(orig_h)
        dropped_any = False
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            ix1, ix2 = x1 * sx, x2 * sx
            iy1, iy2 = y1 * sy, y2 * sy
            c0 = int(np.floor(ix1 / PATCH))
            c1 = int(np.ceil(ix2 / PATCH))
            r0 = int(np.floor(iy1 / PATCH))
            r1 = int(np.ceil(iy2 / PATCH))
            c0 = max(0, min(GRID, c0)); c1 = max(0, min(GRID, c1))
            r0 = max(0, min(GRID, r0)); r1 = max(0, min(GRID, r1))
            if c1 <= c0 or r1 <= r0:
                continue
            for r in range(r0, r1):
                keep[r * GRID + c0: r * GRID + c1] = False
            dropped_any = True
        if not dropped_any:
            return None
        return keep

    def _vlad(self, descs):
        """AnyLoc-style hard-assignment VLAD over a token SUBSET.

        descs: [N, D] patch descriptors. centers: [K, D].
        Steps: L2-normalize each descriptor -> assign to nearest center ->
        sum residuals (desc - center) per cluster -> intra-normalize each cluster
        block (L2) -> concat to [K*D] -> global L2-normalize. Permutation-invariant
        over the token set, which is exactly why dropping person tokens is safe."""
        centers = self._centers  # [K, D]
        K, D = centers.shape
        if descs.shape[0] == 0:
            # Should not happen (caller guards), but never emit NaNs.
            return np.zeros(K * D, dtype=np.float32)
        # per-descriptor L2 norm
        dn = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-12)
        # nearest center: argmin ||d - c||^2 = argmax d.c (both unit-ish); use full
        # squared distance for correctness since centers need not be unit norm.
        # ||d-c||^2 = ||d||^2 - 2 d.c + ||c||^2 ; ||d||^2 constant per row -> use
        # -2 d.c + ||c||^2.
        c_sq = np.sum(centers * centers, axis=1)              # [K]
        dots = dn @ centers.T                                 # [N, K]
        dist = c_sq[None, :] - 2.0 * dots                     # [N, K] (up to +||d||^2)
        labels = np.argmin(dist, axis=1)                      # [N]
        vlad = np.zeros((K, D), dtype=np.float32)
        # accumulate residuals per assigned cluster
        for k in range(K):
            sel = labels == k
            if np.any(sel):
                vlad[k] = np.sum(dn[sel] - centers[k], axis=0)
        # intra-normalization (per cluster block)
        vlad = vlad / (np.linalg.norm(vlad, axis=1, keepdims=True) + 1e-12)
        flat = vlad.reshape(-1)
        flat = flat / (np.linalg.norm(flat) + 1e-12)
        return flat.astype(np.float32, copy=False)

    def _apply_pca_and_norm(self, vec):
        if self._pca_components is not None:
            v = vec - self._pca_mean
            v = v @ self._pca_components.T   # [D_out]
            v = v / (np.linalg.norm(v) + 1e-12)
            return v.astype(np.float32, copy=False)
        # already L2-normalized by _vlad; normalize again defensively (cheap).
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return v.astype(np.float32, copy=False)

    # ----------------------- metadata helpers that peek at files (no model load)

    def _peek_num_c(self) -> int:
        """Read num_clusters from the vocab file for model_id()/dim() before the
        model is loaded. Falls back to 32 (AnyLoc's default) if the vocab isn't
        present yet, so model_id() is still answerable off a fresh checkout."""
        if self._num_c is not None:
            return self._num_c
        vdir = self._vocab_dir()
        npy = os.path.join(vdir, "c_centers.npy")
        try:
            if os.path.isfile(npy):
                # read shape without loading the whole array into a copy
                arr = np.load(npy, mmap_mode="r")
                return int(arr.shape[0])
        except Exception:
            pass
        pt = os.path.join(vdir, "c_centers.pt")
        if os.path.isfile(pt):
            try:
                import torch
                t = torch.load(pt, map_location="cpu")
                return int(t.shape[0])
            except Exception:
                pass
        return 32

    def _pca_out_dim(self) -> int | None:
        if self._pca_components is not None:
            return int(self._pca_components.shape[0])
        p = self._pca_path()
        if os.path.isfile(p):
            try:
                with np.load(p) as data:
                    if "components" in data:
                        return int(data["components"].shape[0])
            except Exception:
                return None
        return None

    def _eigen_pca_out_dim(self) -> int | None:
        # eigenplaces shares the same optional PCA file convention.
        return self._pca_out_dim()

    # ========================================================== EigenPlaces path

    def _load_eigenplaces(self):
        """Bring-up/validation backend: gmberton/eigenplaces ResNet-50, 2048-D GeM
        global descriptor via torch.hub. NO token dropping (it has no patch tokens).
        Uses the project's central device pick (compute.torch_device()), so on the
        B70 it runs on 'xpu' when torch+XPU is present."""
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "eigenplaces place backend needs PyTorch, which is not importable "
                "(%r). Install it (XPU build for the B70):\n"
                "    pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/xpu" % (exc,)
            )
        from . import compute
        self._torch_device = compute.torch_device()
        try:
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load eigenplaces from torch.hub ({exc}). This needs "
                f"network access on first use (it caches to ~/.cache/torch/hub). "
                f"Ensure torch/torchvision are installed and the host can reach "
                f"github.com, or pre-populate the hub cache."
            )
        self._torch_model = model.to(self._torch_device).eval()
        self._load_pca_if_present_eigen()
        print(
            f"[place] eigenplaces r50 device={self._torch_device} dim={self.dim()}",
            flush=True,
        )

    def _load_pca_if_present_eigen(self):
        p = self._pca_path()
        if not os.path.isfile(p):
            return
        data = np.load(p)
        if "mean" in data and "components" in data:
            self._pca_mean = np.ascontiguousarray(data["mean"], dtype=np.float32)
            self._pca_components = np.ascontiguousarray(data["components"], dtype=np.float32)

    def _embed_eigenplaces(self, pil_img):
        import torch

        # 512x512 is EigenPlaces' standard test resolution.
        img = pil_img.resize((512, 512))
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - _IMAGENET_MEAN) / _IMAGENET_STD
        a = np.transpose(a, (2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(a[None, ...])).to(self._torch_device)
        with torch.no_grad():
            feat = self._torch_model(t)
        vec = feat.detach().float().cpu().numpy()[0]
        return self._apply_pca_and_norm(vec)
