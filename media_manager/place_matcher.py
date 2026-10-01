"""
place_matcher.py -- local-feature geometric verification for "same physical place?"

PURPOSE
=======
This module answers one narrow question: *do two photos show the same physical
place?* It is a RE-RANKER. A coarse global place-retrieval step (e.g. an image
embedding / VLAD / NetVLAD-style index) hands us the top ~50 candidate photos
for a query. We then run local-feature matching + geometric verification on
each (query, candidate) pair and sort by the number of geometric inliers. More
inliers == more likely the same rigid scene.

It deliberately IGNORES people. Keypoints that fall inside a person bounding box
are dropped (with their descriptors) during extraction, so a crowd of tourists
in front of two different landmarks cannot create spurious matches, and the same
person photographed in two different places cannot either. The filtering happens
in ``extract()`` so it is baked into the cached per-image features.

KEY ARCHITECTURE: EXTRACT ONCE, MATCH MANY
==========================================
Keypoints + descriptors are EXTRACTED ONCE PER IMAGE and CACHED in the DB
(one blob per file, produced by ``serialize()``). At query time we do NOT touch
the images again: we ``deserialize()`` the query features and the 50 candidate
features and run only the matcher + RANSAC, 50x. Extraction (the expensive,
GPU-friendly part) is therefore split cleanly from matching (the cheap part).

    # --- offline / ingest (once per image) -------------------------------
    pm = PlaceMatcher()                       # backend from env, device from env
    feat = pm.extract(path, person_boxes=boxes)
    db.store(file_id, model=pm.model_id(), blob=pm.serialize(feat))

    # --- query time (re-rank the 50 retrieval candidates) ----------------
    q = pm.deserialize(db.load(query_id))
    scored = []
    for cand_id in top50_from_retrieval:
        c = pm.deserialize(db.load(cand_id))
        s = pm.match_score(q, c)              # {'inliers','ratio','model'}
        scored.append((cand_id, s))
    scored.sort(key=lambda t: t[1]['inliers'], reverse=True)    # re-rank
    same = [cid for cid, s in scored if PlaceMatcher.is_same_place(s)]

BACKENDS
========
* ``xfeat`` (default, recommended) -- XFeat + LighterGlue.
    Apache-2.0, 64-D descriptors, sparse keypoints that cache in a few KB,
    runs in real time on CPU and accelerates well on Intel Arc via OpenVINO.
    LighterGlue is a distilled LightGlue (~3x faster) trained on XFeat features.
* ``sift`` -- OpenCV SIFT + mutual-NN ratio test.
    Pure CPU, zero heavy deps (needs only ``opencv-contrib-python`` or a cv2
    build with SIFT). This is the dependency-free fallback and the path the
    unit tests exercise on a CPU-only dev box.

Both backends end in the SAME geometric verifier: fit a fundamental matrix AND
a homography with ``cv2.USAC_MAGSAC`` and keep whichever explains more inliers.
(A planar facade is well described by H; a full 3-D scene by F -- trying both
and taking the max makes the inlier score robust across scene types.)

================================================================================
INTEL ARC PRO B70  --  SETUP RUNBOOK (xfeat backend, OpenVINO serving)
================================================================================
Runtime targets: the Intel Arc Pro B70 (discrete GPU) via OpenVINO, the bigboy
media worker, and the web host. On the B70 we serve XFeat through ONNX Runtime
with the OpenVINO Execution Provider.

1. PACKAGES
   Serving on the B70 (ONNX + OpenVINO):
       pip install "onnxruntime-openvino>=1.23.0"   # ships OpenVINO 2025.3
       pip install numpy pillow opencv-contrib-python
   The torch reference path (CPU / CUDA / XPU, also used for LighterGlue
   matching of cached features):
       pip install torch torchvision
       pip install git+https://github.com/verlab/accelerated_features.git
       #  ... or clone it and put the repo root on PYTHONPATH so that
       #  `from modules.xfeat import XFeat` and `modules.lighterglue` import.

   !! OpenVINO >= 2025.3 IS REQUIRED for the xfeat ONNX path. 2025.3 fixes an
   XFeat NMS accuracy regression in the OpenVINO graph (earlier builds mis-ran
   the keypoint non-max-suppression and silently returned degraded keypoints).
   Do NOT ship the B70 on < 2025.3. onnxruntime-openvino 1.23.0 bundles 2025.3.

2. WEIGHTS
   * torch:  XFeat + LighterGlue weights download automatically on first use
             (torch.hub / the `weights/` dir of accelerated_features). Both are
             Apache-2.0. Vendor them next to the repo for offline hosts.
   * onnx:   export XFeat to ONNX once and point MEDIA_PLACE_MATCHER_ONNX at it.
             Use the official repo's export, or fabio-sim/LightGlue-ONNX which
             ships XFeat and ALIKED exporters:
                 git clone https://github.com/fabio-sim/LightGlue-ONNX
                 python -m export --extractor xfeat \\
                     --num-keypoints 1024 --dynamic=False --fp16 \\
                     -o xfeat_1024_fp16.onnx
             Prefer STATIC shapes + a FIXED top-K (1024, or 2048 for hard,
             low-texture scenes) so OpenVINO compiles one optimal kernel. FP16
             is the right precision for the B70.

3. ENV VARS
       MEDIA_PLACE_MATCHER=xfeat|sift          (default: xfeat)
       MEDIA_PLACE_MATCHER_DEVICE=GPU|CPU|GPU.0 (default: GPU; OpenVINO device_type)
       MEDIA_PLACE_MATCHER_ONNX=/path/xfeat_1024_fp16.onnx
                                               (set -> use ONNX+OpenVINO for
                                                extraction; unset -> torch)
       MEDIA_PLACE_MATCHER_TOPK=1024           (keypoints per image; 2048 hard)
       MEDIA_PLACE_MATCHER_PRECISION=FP16|FP32 (OpenVINO precision; default FP16)
   On the B70, LighterGlue matching of the cached features still runs through
   the torch `accelerated_features` package (it is tiny and fast); install torch
   on the worker too, or swap in the ALIKED+LightGlue ONNX path (see below).

4. VERIFY GPU PLACEMENT
       python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   must list 'OpenVINOExecutionProvider'. Then run one extract() and confirm the
   session was built on OpenVINO -- this module logs (WARNING) the resolved
   providers and the chosen device at init and raises LOUDLY if the OpenVINO EP
   is unavailable while the ONNX path was requested. Watch GPU busy with
   `intel_gpu_top` (or `xpu-smi dump`) during a batch extract.

HIGHER-ACCURACY ALTERNATIVE (one swap): ALIKED + LightGlue, fully ONNX
----------------------------------------------------------------------
fabio-sim/LightGlue-ONNX exports an end-to-end ALIKED+LightGlue graph that runs
entirely under OpenVINO ('-d openvino' / '--device openvino') with no torch at
query time -- higher accuracy than XFeat at a modest speed cost. It is a drop-in
for this re-ranker: extract ALIKED keypoints/descriptors, cache them the same
way, and run the LightGlue ONNX matcher instead of LighterGlue. Keep the same
person-box filtering and the same MAGSAC F-vs-H verifier below.

PURE-CPU FALLBACK: OpenCV SIFT + MAGSAC
---------------------------------------
No learned weights, no GPU, no ONNX -- only OpenCV. Lower recall under big
viewpoint/illumination changes, but dependency-free and deterministic, so it is
the path that runs in CI / on the dev box. Set MEDIA_PLACE_MATCHER=sift.

NO SILENT FAILURES
------------------
Every heavy dependency is imported lazily inside the method that needs it, so
this file imports on a bare CPU-only box. When a backend's runtime or weights
are missing the module raises a clear error that names the exact pip/export
command to fix it -- it never quietly falls back to a weaker backend.
"""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger("media_manager.place_matcher")

# Stable schema version embedded in model_id(); bump when the cached-feature
# format or the matching semantics change so the DB can invalidate old blobs.
_SCHEMA_VERSION = "v1"

_DEFAULT_TOPK = 1024          # 1024 default; 2048 for hard, low-texture scenes
_DEFAULT_PRECISION = "FP16"
_MAGSAC_CONF = 0.999
_MAGSAC_MAXITERS = 10000
_BASE_REPROJ_PX = 3.0         # ~3px at 1024px long side, scaled to image size


class PlaceMatcher:
    """Extract-once / match-many local-feature place verifier.

    See the module docstring for the B70 runbook and the recommended re-rank
    usage. The only runtime deps are pulled in lazily, so constructing this
    object on a CPU-only box with nothing installed is always safe.
    """

    def __init__(self, backend: str = None, device: str = "GPU"):
        if backend is None:
            backend = os.environ.get("MEDIA_PLACE_MATCHER", "xfeat")
        backend = str(backend).strip().lower()
        if backend not in ("xfeat", "sift"):
            raise ValueError(
                "Unknown MEDIA_PLACE_MATCHER backend %r; expected 'xfeat' or 'sift'."
                % backend
            )
        self.backend = backend

        self.device = os.environ.get("MEDIA_PLACE_MATCHER_DEVICE", device) or "GPU"
        self.precision = (
            os.environ.get("MEDIA_PLACE_MATCHER_PRECISION", _DEFAULT_PRECISION) or _DEFAULT_PRECISION
        ).upper()

        try:
            self.top_k = int(os.environ.get("MEDIA_PLACE_MATCHER_TOPK", _DEFAULT_TOPK))
        except (TypeError, ValueError):
            raise ValueError(
                "MEDIA_PLACE_MATCHER_TOPK must be an integer (keypoints per image); "
                "got %r." % os.environ.get("MEDIA_PLACE_MATCHER_TOPK")
            )
        if self.top_k <= 0:
            raise ValueError("MEDIA_PLACE_MATCHER_TOPK must be > 0; got %d." % self.top_k)

        # ONNX path is selected (explicitly, never by silent auto-fallback) only
        # when the user points us at an exported XFeat ONNX model.
        self._onnx_path = os.environ.get("MEDIA_PLACE_MATCHER_ONNX") or None
        self._engine = "onnx" if (self.backend == "xfeat" and self._onnx_path) else (
            "torch" if self.backend == "xfeat" else "sift"
        )

        # Lazily-populated runtime handles (kept off __init__ so import is cheap).
        self._xfeat = None          # accelerated_features XFeat instance (torch)
        self._ort_session = None    # onnxruntime InferenceSession (OpenVINO EP)
        self._ort_io = None         # (input_name, output_name_map, (H, W))
        self._torch_device = None   # resolved torch device string
        self._sift = None           # cv2.SIFT
        self._model_id = None       # cached model_id() string

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    def model_id(self) -> str:
        """Stable version string for the DB 'model' column.

        Encodes backend + engine + device/precision + top_k + schema version so
        cached blobs are invalidated whenever any of those change. Never raises
        and never triggers a heavy import for its own sake.
        """
        if self._model_id is not None:
            return self._model_id

        if self.backend == "sift":
            try:
                import cv2  # noqa: lazy
                ver = cv2.__version__
            except Exception:
                ver = "unknown"
            mid = "sift-magsac/opencv%s/top%d/%s" % (ver, self.top_k, _SCHEMA_VERSION)
        elif self._engine == "onnx":
            mid = "xfeat-lighterglue/openvino-%s-%s/top%d/%s" % (
                self.device, self.precision.lower(), self.top_k, _SCHEMA_VERSION,
            )
        else:
            mid = "xfeat-lighterglue/torch-%s/top%d/%s" % (
                (self._torch_device or self.device), self.top_k, _SCHEMA_VERSION,
            )
        self._model_id = mid
        return mid

    # ------------------------------------------------------------------ #
    # Extraction (expensive; run once per image and cache the result)
    # ------------------------------------------------------------------ #
    def extract(self, image, person_boxes=None) -> dict:
        """Extract keypoints + descriptors from ``image``.

        image        : a filesystem path (str/os.PathLike), a PIL.Image, or a
                       numpy ndarray (HxW grayscale or HxWx3). RGB is assumed for
                       3-channel ndarrays.
        person_boxes : list of (x1, y1, x2, y2) rectangles in ORIGINAL pixel
                       coordinates. Any keypoint whose (x, y) falls inside any
                       box is dropped together with its descriptor BEFORE the
                       result is returned, so the cached features are already
                       person-free on both sides at query time.

        Returns a feature dict:
            {'kpts': float32 [N,2] in original pixels,
             'desc': float32 [N,D]  (D=64 xfeat, 128 sift),
             'size': (w, h),
             'model': model_id()}
        """
        import numpy as np

        rgb, (w, h) = _load_image_rgb(image)

        if self.backend == "sift":
            kpts, desc = self._extract_sift(rgb)
        else:
            kpts, desc = self._extract_xfeat(rgb)

        kpts = np.ascontiguousarray(kpts, dtype=np.float32).reshape(-1, 2)
        desc = np.ascontiguousarray(desc, dtype=np.float32)
        if desc.ndim != 2 or desc.shape[0] != kpts.shape[0]:
            raise RuntimeError(
                "place_matcher: backend %r returned mismatched kpts/desc "
                "(%s vs %s)." % (self.backend, kpts.shape, desc.shape)
            )

        kpts, desc = _drop_in_boxes(kpts, desc, person_boxes)

        return {
            "kpts": kpts,
            "desc": desc,
            "size": (int(w), int(h)),
            "model": self.model_id(),
        }

    # ------------------------------------------------------------------ #
    # Serialization for DB caching (self-describing .npz blob)
    # ------------------------------------------------------------------ #
    def serialize(self, feat: dict) -> bytes:
        """Pack a feature dict into compact, self-describing bytes for the DB."""
        import numpy as np

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            kpts=np.asarray(feat["kpts"], dtype=np.float32),
            desc=np.asarray(feat["desc"], dtype=np.float32),
            size=np.asarray(feat["size"], dtype=np.int32),
            model=np.asarray(str(feat.get("model", self.model_id()))),
            fmt=np.asarray(_SCHEMA_VERSION),
        )
        return buf.getvalue()

    def deserialize(self, blob: bytes) -> dict:
        """Inverse of :meth:`serialize`."""
        import numpy as np

        with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
            size = npz["size"]
            return {
                "kpts": np.ascontiguousarray(npz["kpts"], dtype=np.float32),
                "desc": np.ascontiguousarray(npz["desc"], dtype=np.float32),
                "size": (int(size[0]), int(size[1])),
                "model": str(npz["model"]),
            }

    # ------------------------------------------------------------------ #
    # Matching + geometric verification (cheap; run 50x per query)
    # ------------------------------------------------------------------ #
    def match_score(self, query_feat: dict, cand_feat: dict) -> dict:
        """Match two cached feature sets and geometrically verify them.

        For xfeat: LighterGlue produces confidence-filtered correspondences
        (only its own above-threshold matches are kept -- raw NN matches never
        reach RANSAC). For sift: a mutual nearest-neighbour ratio test does.

        The surviving correspondences are verified by fitting BOTH a fundamental
        matrix and a homography with cv2.USAC_MAGSAC and taking the model with
        more inliers.

        Returns {'inliers': int, 'ratio': float, 'model': 'F'|'H'} where
        ``inliers`` is the monotone re-rank score (higher = more likely the same
        place) and ``ratio`` is inliers / putative-matches.
        """
        import numpy as np

        q_kpts = np.asarray(query_feat["kpts"], dtype=np.float32)
        c_kpts = np.asarray(cand_feat["kpts"], dtype=np.float32)
        q_desc = np.asarray(query_feat["desc"], dtype=np.float32)
        c_desc = np.asarray(cand_feat["desc"], dtype=np.float32)

        if self.backend == "sift":
            pts0, pts1 = self._match_sift(q_kpts, q_desc, c_kpts, c_desc)
        else:
            pts0, pts1 = self._match_lighterglue(query_feat, cand_feat)

        # Verify against the query image size (both images filtered the same way;
        # the query frame is the reference for the reprojection threshold).
        return self._geometric_verify(pts0, pts1, query_feat.get("size"))

    @staticmethod
    def is_same_place(score: dict, min_inliers: int = 20, min_ratio: float = 0.3) -> bool:
        """Tunable decision rule over a :meth:`match_score` result.

        Defaults (20 inliers / 0.3 ratio) are a sane starting point for XFeat +
        LighterGlue on photo-scale images; tune per corpus. The ratio guard
        rejects pairs with many putative matches but a low geometric agreement
        (typical of repetitive texture / near-duplicate clutter).
        """
        if not score:
            return False
        return int(score.get("inliers", 0)) >= int(min_inliers) and \
            float(score.get("ratio", 0.0)) >= float(min_ratio)

    # ================================================================== #
    # XFeat backend internals
    # ================================================================== #
    def _extract_xfeat(self, rgb):
        if self._engine == "onnx":
            return self._extract_xfeat_onnx(rgb)
        return self._extract_xfeat_torch(rgb)

    def _load_xfeat_torch(self):
        """Load (once) the torch accelerated_features XFeat model."""
        if self._xfeat is not None:
            return self._xfeat
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The xfeat backend needs PyTorch. Install it, e.g.:\n"
                "    pip install torch torchvision\n"
                "or set MEDIA_PLACE_MATCHER=sift for the OpenCV-only fallback."
            ) from e
        try:
            from modules.xfeat import XFeat
        except ImportError as e:
            raise ImportError(
                "Could not import XFeat from the accelerated_features package.\n"
                "Install it:\n"
                "    pip install git+https://github.com/verlab/accelerated_features.git\n"
                "or clone it and put the repo root on PYTHONPATH so that "
                "`from modules.xfeat import XFeat` works."
            ) from e

        self._torch_device = self._resolve_torch_device()
        self._model_id = None  # recompute now that the device is resolved
        self._xfeat = XFeat(top_k=self.top_k)
        try:
            self._xfeat = self._xfeat.to(self._torch_device)
            self._xfeat.dev = self._torch_device
        except Exception:
            # XFeat manages its own .dev; if moving failed, keep the default.
            logger.warning("place_matcher: could not move XFeat to %r; using its default device.",
                            self._torch_device)
        logger.warning("place_matcher: xfeat/torch engine on device=%s (top_k=%d).",
                       self._torch_device, self.top_k)
        return self._xfeat

    def _resolve_torch_device(self):
        import torch
        want = (self.device or "GPU").upper()
        if want == "CPU":
            return "cpu"
        # want a GPU: prefer CUDA, then Intel XPU (IPEX), else CPU (surfaced).
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        logger.warning(
            "place_matcher: MEDIA_PLACE_MATCHER_DEVICE=%s requested but no torch "
            "GPU (CUDA/XPU) is available; XFeat will run on CPU. For Intel Arc B70 "
            "acceleration use the ONNX+OpenVINO path (set MEDIA_PLACE_MATCHER_ONNX).",
            self.device,
        )
        return "cpu"

    def _extract_xfeat_torch(self, rgb):
        import numpy as np
        xfeat = self._load_xfeat_torch()
        # XFeat.detectAndCompute accepts an HxWx3 uint8/float ndarray (it parses
        # it to a [1,C,H,W] tensor internally). Keypoints come back in input
        # pixel coordinates, so no rescaling is needed.
        out = xfeat.detectAndCompute(np.ascontiguousarray(rgb), top_k=self.top_k)[0]
        kpts = _to_numpy(out["keypoints"])
        desc = _to_numpy(out["descriptors"])
        return kpts, desc

    def _load_xfeat_onnx(self):
        """Build (once) the OpenVINO ONNX Runtime session for XFeat."""
        if self._ort_session is not None:
            return self._ort_session
        if not self._onnx_path or not os.path.exists(self._onnx_path):
            raise FileNotFoundError(
                "MEDIA_PLACE_MATCHER_ONNX points at a missing XFeat ONNX model: %r\n"
                "Export one (static shapes, fixed top-K, FP16), e.g.:\n"
                "    git clone https://github.com/fabio-sim/LightGlue-ONNX\n"
                "    python -m export --extractor xfeat --num-keypoints %d "
                "--dynamic=False --fp16 -o xfeat_%d_fp16.onnx"
                % (self._onnx_path, self.top_k, self.top_k)
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "The xfeat ONNX/OpenVINO path needs onnxruntime-openvino >= 1.23.0 "
                "(bundles OpenVINO 2025.3 with the XFeat NMS fix):\n"
                "    pip install 'onnxruntime-openvino>=1.23.0'"
            ) from e

        providers = ort.get_available_providers()
        if "OpenVINOExecutionProvider" not in providers:
            raise RuntimeError(
                "OpenVINOExecutionProvider is not available in this onnxruntime "
                "build (found: %s). Install the OpenVINO build and ensure "
                "OpenVINO >= 2025.3 (XFeat NMS fix):\n"
                "    pip install 'onnxruntime-openvino>=1.23.0'" % (providers,)
            )

        ov_opts = {"device_type": self.device, "precision": self.precision}
        session = ort.InferenceSession(
            self._onnx_path,
            providers=[("OpenVINOExecutionProvider", ov_opts)],
        )
        logger.warning(
            "place_matcher: xfeat/onnx engine on OpenVINO device_type=%s precision=%s; "
            "session providers=%s", self.device, self.precision, session.get_providers(),
        )

        inp = session.get_inputs()[0]
        shape = inp.shape  # e.g. [1, 3, H, W] for a static export
        hw = None
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            h_dim, w_dim = shape[2], shape[3]
            if isinstance(h_dim, int) and isinstance(w_dim, int):
                hw = (h_dim, w_dim)
        self._ort_io = (inp.name, [o.name for o in session.get_outputs()], hw)
        self._ort_session = session
        return session

    def _extract_xfeat_onnx(self, rgb):
        import numpy as np
        session = self._load_xfeat_onnx()
        in_name, out_names, hw = self._ort_io

        orig_h, orig_w = rgb.shape[0], rgb.shape[1]
        if hw is not None:
            net_h, net_w = hw
            resized, sx, sy = _resize_to(rgb, net_w, net_h)
        else:
            resized, sx, sy = rgb, 1.0, 1.0

        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1,3,H,W]
        x = np.ascontiguousarray(x)

        outputs = session.run(out_names, {in_name: x})
        named = dict(zip(out_names, outputs))
        kpts = _pick_output(named, want_cols=2, fallbacks=("keypoints", "mkpts", "kpts"))
        desc = _pick_output(named, want_cols=None, fallbacks=("descriptors", "desc", "feats"),
                            exclude=kpts)
        kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
        desc = np.asarray(desc, dtype=np.float32).reshape(kpts.shape[0], -1)

        # Undo the resize so keypoints land in ORIGINAL pixel coordinates.
        if sx != 1.0 or sy != 1.0:
            kpts = kpts.copy()
            kpts[:, 0] /= sx
            kpts[:, 1] /= sy
        # Clamp to the original frame (padding can push points just outside).
        np.clip(kpts[:, 0], 0, orig_w - 1, out=kpts[:, 0])
        np.clip(kpts[:, 1], 0, orig_h - 1, out=kpts[:, 1])
        return kpts, desc

    def _match_lighterglue(self, query_feat, cand_feat):
        """Run LighterGlue on two cached XFeat feature sets.

        LighterGlue applies its own confidence gating and returns only the
        surviving correspondences -- exactly what should feed RANSAC. Matching
        always goes through the torch accelerated_features LighterGlue (tiny and
        fast even on CPU); see the module docstring for the fully-ONNX
        ALIKED+LightGlue alternative.
        """
        import numpy as np
        import torch

        xfeat = self._load_xfeat_torch()
        dev = getattr(xfeat, "dev", self._torch_device or "cpu")

        def _pack(feat):
            k = np.asarray(feat["kpts"], dtype=np.float32)
            d = np.asarray(feat["desc"], dtype=np.float32)
            w, h = feat["size"]
            return {
                "keypoints": torch.from_numpy(k).to(dev),
                "descriptors": torch.from_numpy(d).to(dev),
                "image_size": (int(w), int(h)),  # (W, H) -- XFeat's convention
            }

        d0, d1 = _pack(query_feat), _pack(cand_feat)
        if d0["keypoints"].shape[0] == 0 or d1["keypoints"].shape[0] == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        res = xfeat.match_lighterglue(d0, d1)  # -> (mkpts0, mkpts1[, idxs])
        mkpts0, mkpts1 = _to_numpy(res[0]), _to_numpy(res[1])
        return (
            np.asarray(mkpts0, dtype=np.float32).reshape(-1, 2),
            np.asarray(mkpts1, dtype=np.float32).reshape(-1, 2),
        )

    # ================================================================== #
    # SIFT backend internals (pure OpenCV, CPU, dependency-free fallback)
    # ================================================================== #
    def _load_sift(self):
        if self._sift is not None:
            return self._sift
        cv2 = _load_cv2()
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "This OpenCV build has no SIFT (cv2.SIFT_create missing). Install "
                "a build that includes it:\n"
                "    pip install opencv-contrib-python"
            )
        self._sift = cv2.SIFT_create(nfeatures=self.top_k)
        return self._sift

    def _extract_sift(self, rgb):
        import numpy as np
        cv2 = _load_cv2()
        sift = self._load_sift()
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
        kpts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return kpts, np.asarray(desc, dtype=np.float32)

    def _match_sift(self, q_kpts, q_desc, c_kpts, c_desc):
        """Mutual nearest-neighbour + Lowe ratio test."""
        import numpy as np
        cv2 = _load_cv2()
        if q_desc.shape[0] < 2 or c_desc.shape[0] < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        ratio = 0.75

        def _ratio_nn(a, b):
            # best b-index for each a-index that passes the ratio test
            good = {}
            for m in bf.knnMatch(a, b, k=2):
                if len(m) < 2:
                    continue
                m0, m1 = m[0], m[1]
                if m0.distance < ratio * m1.distance:
                    good[m0.queryIdx] = m0.trainIdx
            return good

        fwd = _ratio_nn(q_desc, c_desc)   # q -> c
        bwd = _ratio_nn(c_desc, q_desc)   # c -> q
        pts0, pts1 = [], []
        for qi, ci in fwd.items():
            if bwd.get(ci) == qi:          # mutual
                pts0.append(q_kpts[qi])
                pts1.append(c_kpts[ci])
        if not pts0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return (np.asarray(pts0, dtype=np.float32), np.asarray(pts1, dtype=np.float32))

    # ================================================================== #
    # Shared geometric verifier: F vs H via USAC_MAGSAC, keep the winner
    # ================================================================== #
    def _geometric_verify(self, pts0, pts1, size) -> dict:
        import numpy as np
        cv2 = _load_cv2()

        if not hasattr(cv2, "USAC_MAGSAC"):
            raise RuntimeError(
                "cv2.USAC_MAGSAC is missing -- OpenCV is too old. Upgrade:\n"
                "    pip install -U 'opencv-contrib-python>=4.5.4'"
            )

        pts0 = np.ascontiguousarray(pts0, dtype=np.float64).reshape(-1, 2)
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64).reshape(-1, 2)
        n = int(pts0.shape[0])
        result = {"inliers": 0, "ratio": 0.0, "model": "H"}
        if n < 4:
            return result

        if size:
            w, h = size
            max_dim = max(int(w), int(h)) or 1024
        else:
            max_dim = 1024
        thr = max(1.0, _BASE_REPROJ_PX * (max_dim / 1024.0))

        # Homography (planar scenes / facades): needs >= 4 points.
        h_inl = 0
        try:
            _H, mask_h = cv2.findHomography(
                pts0, pts1, method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=thr, confidence=_MAGSAC_CONF,
                maxIters=_MAGSAC_MAXITERS,
            )
            if mask_h is not None:
                h_inl = int(mask_h.sum())
        except cv2.error:
            h_inl = 0

        # Fundamental matrix (full 3-D scenes): needs >= 8 points.
        f_inl = 0
        if n >= 8:
            try:
                F, mask_f = cv2.findFundamentalMat(
                    pts0, pts1, method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=thr, confidence=_MAGSAC_CONF,
                    maxIters=_MAGSAC_MAXITERS,
                )
                if mask_f is not None and F is not None and F.shape == (3, 3):
                    f_inl = int(mask_f.sum())
            except cv2.error:
                f_inl = 0

        if f_inl >= h_inl:
            result["inliers"], result["model"] = f_inl, "F"
        else:
            result["inliers"], result["model"] = h_inl, "H"
        result["ratio"] = float(result["inliers"]) / float(n) if n else 0.0
        return result


# ====================================================================== #
# Module-level helpers (all heavy imports stay lazy)
# ====================================================================== #
def _load_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV is required (image I/O, SIFT, and the MAGSAC verifier):\n"
            "    pip install opencv-contrib-python"
        ) from e


def _to_numpy(x):
    """Convert a torch tensor or array-like to a contiguous numpy array."""
    import numpy as np
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x)


def _load_image_rgb(image):
    """Return (rgb_uint8 HxWx3, (w, h)) from a path, PIL image, or ndarray."""
    import numpy as np

    # numpy ndarray
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                "place_matcher: unsupported ndarray image shape %r (want HxW or HxWx3)."
                % (image.shape,)
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.0 else \
                (arr * 255.0).clip(0, 255).astype(np.uint8)
        h, w = arr.shape[0], arr.shape[1]
        return np.ascontiguousarray(arr), (w, h)

    # PIL image (duck-typed so Pillow need not be importable otherwise)
    if hasattr(image, "convert") and hasattr(image, "size"):
        arr = np.asarray(image.convert("RGB"))
        h, w = arr.shape[0], arr.shape[1]
        return np.ascontiguousarray(arr), (w, h)

    # path-like
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.exists(path):
            raise FileNotFoundError("place_matcher: image not found: %r" % path)
        cv2 = _load_cv2()
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("place_matcher: OpenCV could not decode image: %r" % path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[0], rgb.shape[1]
        return np.ascontiguousarray(rgb), (w, h)

    raise TypeError(
        "place_matcher: image must be a path, a PIL.Image, or an ndarray; got %r."
        % type(image)
    )


def _resize_to(rgb, net_w, net_h):
    """Resize to (net_w, net_h); return (resized, scale_x, scale_y).

    scale_x = net_w / orig_w, so original_x = net_x / scale_x.
    """
    cv2 = _load_cv2()
    orig_h, orig_w = rgb.shape[0], rgb.shape[1]
    if orig_w == net_w and orig_h == net_h:
        return rgb, 1.0, 1.0
    resized = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_AREA)
    return resized, float(net_w) / float(orig_w), float(net_h) / float(orig_h)


def _pick_output(named, want_cols, fallbacks, exclude=None):
    """Pick an ONNX output by name, else by shape heuristic.

    want_cols=2 -> a keypoints array [N,2]; want_cols=None -> the descriptor
    array (the remaining 2-D float output). Raises loudly if nothing fits so a
    mismatched export never silently yields garbage.
    """
    import numpy as np
    for key in fallbacks:
        for name, val in named.items():
            if key in name.lower():
                return np.squeeze(np.asarray(val))

    exclude_id = id(exclude) if exclude is not None else None
    for name, val in named.items():
        arr = np.squeeze(np.asarray(val))
        if arr.ndim != 2:
            continue
        if exclude_id is not None and id(val) == exclude_id:
            continue
        if want_cols == 2 and arr.shape[1] == 2:
            return arr
        if want_cols is None and arr.shape[1] != 2:
            return arr
    raise RuntimeError(
        "place_matcher: could not identify the %s output in the XFeat ONNX model "
        "(outputs: %s). Re-export from the official repo / LightGlue-ONNX so "
        "outputs are named keypoints/descriptors, or adapt _pick_output()."
        % ("keypoints" if want_cols == 2 else "descriptor", list(named.keys()))
    )


def _drop_in_boxes(kpts, desc, person_boxes):
    """Drop keypoints (and their descriptors) that fall inside any person box."""
    import numpy as np
    if not person_boxes or kpts.shape[0] == 0:
        return kpts, desc
    x = kpts[:, 0]
    y = kpts[:, 1]
    inside = np.zeros(kpts.shape[0], dtype=bool)
    for box in person_boxes:
        x1, y1, x2, y2 = box
        lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
        lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
        inside |= (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    keep = ~inside
    return (np.ascontiguousarray(kpts[keep]), np.ascontiguousarray(desc[keep]))
