"""Reticulum media-worker *server*.

Runs the heavy ML models (InsightFace faces, CLIP embeddings, YOLO-World object
detection) and exposes them over Reticulum so a thin client on another host can
offload work to it. The wire contract (destination naming, request paths,
(de)serialization) lives in :mod:`worker_protocol` and is shared verbatim with
the client so the two never drift.

Design notes:
  * Models are lazily built and cached — ping must never force a heavy load.
  * A single lock serializes every model call. This both prevents the worker
    OOMing from concurrent inference and protects the shared YOLO-World
    detector's vocabulary from a set_vocab race between overlapping requests.
  * The real model methods read from *file paths*; the worker receives *bytes*.
    Each handler writes the bytes to a temp file preserving the original
    extension, runs the real method, then deletes the temp file — this reuses
    the exact model logic and error handling rather than re-implementing it.
  * No silent failures: every handler wraps its body in try/except, prints the
    full traceback, and returns the error string in the response — an error is
    always visible in both logs and the response, never swallowed into a
    fake success.
"""
import os
import io
import time
import random
import shutil
import signal
import threading
import tempfile
import traceback
import collections

import numpy as np

from . import worker_protocol
from . import compute
from . import face_timing
from . import clip_tag_classifier as ct
from . import yolo_tag_classifier as yt


DEFAULT_CONF_THRESHOLD = 0.15


class _InferenceLock:
    """``models.lock``: exclusive model-load lock PLUS the process-wide GPU lock.

    Every non-embedding handler (CLIP, faces, YOLO, MiVOLO, ...) keeps using
    ``with models.lock:`` and so still excludes every other model call, embedding forwards
    included. The SigLIP+DINO bulk-embed path instead takes only ``models.load_lock`` (lazy
    model construction) and then ``compute.GPU_LOCK`` around each forward pass, so one
    request's decode overlaps another's GPU work. Lock order is always load_lock -> GPU_LOCK
    (the embed path never holds GPU_LOCK while asking for load_lock), so there is no cycle."""

    def __init__(self, load_lock, gpu_lock):
        self.load_lock, self.gpu_lock = load_lock, gpu_lock

    def acquire(self, *a, **k):
        if not self.load_lock.acquire(*a, **k):
            return False
        self.gpu_lock.acquire()
        return True

    def release(self):
        self.gpu_lock.release()
        self.load_lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class WorkerModels:
    """Lazy, cached singletons for the three heavy model classes.

    A single lock serializes all model inference so the worker never runs
    concurrent inference (OOM risk) and the object detector's vocabulary can't
    be swapped mid-request by a concurrent caller.
    """

    def __init__(self):
        self.load_lock = threading.Lock()
        self.lock = _InferenceLock(self.load_lock, compute.GPU_LOCK)
        self._face_detector = None
        self._clip_indexer = None
        self._object_detector = None
        self._siglip = None
        self._dino = None

    def get_siglip_embedder(self):
        if self._siglip is None:
            print("[worker] loading SigLIP2Embedder...", flush=True)
            from .embedders import SigLIP2Embedder  # hook-allow: lazy — model load
            self._siglip = SigLIP2Embedder()
        return self._siglip

    def get_dino_embedder(self):
        if self._dino is None:
            print("[worker] loading DINOv2Embedder...", flush=True)
            from .embedders import DINOv2Embedder  # hook-allow: lazy — model load
            self._dino = DINOv2Embedder()
        return self._dino

    def get_face_detector(self):
        if self._face_detector is None:
            print("[worker] loading FaceDetector...", flush=True)
            from .face_detector import FaceDetector  # hook-allow: lazy — model load
            self._face_detector = FaceDetector()
        return self._face_detector

    def get_clip_indexer(self):
        if self._clip_indexer is None:
            print("[worker] loading CLIPIndexer...", flush=True)
            from .indexer import CLIPIndexer  # hook-allow: lazy — model load
            self._clip_indexer = CLIPIndexer()
        return self._clip_indexer

    def get_object_detector(self):
        if self._object_detector is None:
            print("[worker] loading YOLOWorldDetector...", flush=True)
            from .detector import YOLOWorldDetector  # hook-allow: lazy — model load
            self._object_detector = YOLOWorldDetector(
                model_size='s', conf_threshold=DEFAULT_CONF_THRESHOLD)
        return self._object_detector

    def loaded_model_ids(self):
        """Model ids for the models actually loaded so far (for ping)."""
        ids = []
        if self._face_detector is not None:
            ids.append(self._face_detector.model_id())
        if self._clip_indexer is not None:
            ids.append(self._clip_indexer.model_name)
        if self._object_detector is not None:
            ids.append(self._object_detector.model_id())
        if self._siglip is not None:
            ids.append(self._siglip.model_id())
        if self._dino is not None:
            ids.append(self._dino.model_id())
        return ids


# Module-level singleton holder shared by all request handlers.
models = WorkerModels()

# Age/gender (MiVOLO) runs in its OWN isolated .age-venv subprocess (see
# age_estimator.py) to avoid RAM contention with the three in-process models.
# BUT that subprocess still runs on the SAME Arc GPU (torch-XPU / Level-Zero), so
# running it WHILE an OpenVINO face/CLIP/YOLO inference is live means two GPU
# runtimes contend on one device — the intermittent "[GPU] CL_OUT_OF_RESOURCES"
# driver crash. So age estimation now takes the SAME models.lock as the OpenVINO
# paths: the venv stays isolated for RAM, but GPU work is serialized end-to-end
# (one GPU op at a time). Ping doesn't take the lock, so health checks stay
# responsive. Lazily built (its __init__ only resolves the venv path).
_age_estimator = None


def _get_age_estimator():
    global _age_estimator
    if _age_estimator is None:
        from .age_estimator import AgeGenderEstimator  # hook-allow: lazy — optional, self-contained feature
        _age_estimator = AgeGenderEstimator()
    return _age_estimator


def _ext_from_name(name):
    """Return a file extension (with dot) derived from the request's basename,
    falling back to '.png' when absent — the real model methods gate on the
    file extension, so it must be preserved for the bytes round-trip."""
    if name:
        ext = os.path.splitext(os.path.basename(name))[1]
        if ext:
            return ext
    return '.png'


def _write_temp(image_bytes, name):
    """Write received image bytes to a temp file preserving the extension.
    Returns the temp path; caller is responsible for deleting it."""
    ext = _ext_from_name(name)
    fd, path = tempfile.mkstemp(suffix=ext, prefix='mmworker_')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(image_bytes)
    except Exception:  # hook-allow: broad — cleanup then re-raise
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _cleanup(path):
    try:
        os.unlink(path)
    except OSError as e:
        # Surface rather than swallow — a persistent failure here means temp
        # images (full-size decoded frames) are piling up in the temp dir.
        print(f"[worker] WARNING: could not remove temp file {path}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Request handlers. build_app() dispatches each POST /<op> to the matching entry
# here via the 6-parameter signature these have always had (only `data` is used;
# the trailing four are legacy transport metadata, passed as None over HTTP):
#     handler(path, data, request_id, link_id, remote_identity, requested_at) -> bytes
# `data` is the raw bytes the client packed with worker_protocol.pack().
# Each returns worker_protocol.pack(resp_dict) (bytes).
# ---------------------------------------------------------------------------

def handle_ping(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        resp = {"ok": True, "models": models.loaded_model_ids(), "error": None}
        print(f"[worker] handled {worker_protocol.PATH_PING} () -> ok "
              f"models={resp['models']}", flush=True)
        return worker_protocol.pack(resp)
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_PING} () -> ERROR {e}", flush=True)
        return worker_protocol.pack({"ok": False, "models": [], "error": str(e)})


def handle_pattern(path, data, request_id, link_id, remote_identity, requested_at):
    """Find-by-pattern descriptors.
      req {image, deep, box}      -> {queries: float32 bytes (S, DIM), n: S, algo}   (deep query)
      req {image, deep}           -> {windows: int8 bytes (n, DIM), n, algo}         (deep index)
    `deep` selects the pattern_deep.py (ResNet18 window) pipeline — torch/timm, runs under
    models.lock on the worker's compute device. Without `deep` the legacy classical
    descriptor is served (numpy+OpenCV only):
      req {image, boxes}      -> {descriptors: [bytes,...]}  (one per box; for the query)
      req {image} (no boxes)  -> {tiles: [{bbox, descriptor}]} (auto grid; for indexing)"""
    name = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        if req.get("deep"):
            from media_manager import pattern_deep as pdeep  # hook-allow: lazy — pulls torch/timm
            rgb = pdeep.load_rgb(req["image"])
            if rgb is None:
                return worker_protocol.pack({"error": "could not decode image"})
            with models.lock:
                if req.get("box") is not None:
                    q = pdeep.describe_query(rgb, req["box"])
                    out = {"queries": q.tobytes() if q is not None else b"",
                           "n": 0 if q is None else int(q.shape[0])}
                else:
                    w = pdeep.describe_image(rgb)
                    out = {"windows": w.tobytes(), "n": int(w.shape[0])}
            print(f"[worker] handled {worker_protocol.PATH_PATTERN} ({name}) -> deep {out['n']} vectors",
                  flush=True)
            return worker_protocol.pack({**out, "algo": pdeep.ALGO_TAG, "error": None})
        from media_manager import pattern_descriptor as pd  # hook-allow: lazy — pulls cv2
        bgr = pd.decode_bgr(req["image"])
        if bgr is None:
            return worker_protocol.pack({"error": "could not decode image"})
        boxes = req.get("boxes")
        if boxes is not None:
            descs = pd.descriptors_for_boxes(bgr, boxes)
            out = [(d.tobytes() if d is not None else b"") for d in descs]
            print(f"[worker] handled {worker_protocol.PATH_PATTERN} ({name}) -> {len(out)} descriptors",
                  flush=True)
            return worker_protocol.pack({"descriptors": out, "algo": pd.ALGO_TAG, "error": None})
        # No boxes → index mode: lay down the shared tile grid and describe each tile.
        from media_manager.tile_index import generate_tiles  # hook-allow: lazy — pulls PIL
        h, w = bgr.shape[:2]
        tiles = generate_tiles(w, h)
        descs = pd.descriptors_for_boxes(bgr, [list(t) for t in tiles])
        out = [{"bbox": [float(v) for v in tiles[i]], "descriptor": descs[i].tobytes()}
               for i in range(len(tiles)) if descs[i] is not None]
        print(f"[worker] handled {worker_protocol.PATH_PATTERN} ({name}) -> {len(out)} tiles", flush=True)
        return worker_protocol.pack({"tiles": out, "algo": pd.ALGO_TAG, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_PATTERN} ({name}) -> ERROR {e}", flush=True)
        return worker_protocol.pack({"error": str(e)})


def _pattern_dino_meta(pdino):
    return {"algo": pdino.algo_tag(), "dim": pdino.dim(),
            "proto": worker_protocol.PATTERN_DINO_PROTOCOL_VERSION}


def handle_pattern_dino(path, data, request_id, link_id, remote_identity, requested_at):
    """Find-by-pattern, DINOv2-L dense cells (pattern_dino.py). Ops (see worker_protocol):
    info / query / index (upload items, or shared-disk paths under a validated data_root).
    Shares the worker's ONE DINOv2Embedder with the embedding pipeline (no second copy in VRAM);
    like embed_both only model LOADING takes models.lock — forwards take compute.GPU_LOCK
    themselves, so decode of one request overlaps another's GPU work."""
    from media_manager import pattern_dino as pdino  # hook-allow: lazy — pulls numpy only
    op = None
    try:
        req = worker_protocol.unpack(data)
        op = req.get("op")
        meta = _pattern_dino_meta(pdino)
        if op == "info":
            return worker_protocol.pack({**meta, "ready": pdino.projection_ready(), "error": None})
        if not pdino.projection_ready():
            raise RuntimeError("pattern_dino projection missing on the worker "
                               "(media_manager/data/pattern_dino_proj.npz): fit it with "
                               "bench/bench_pattern_dino.py --fit and redeploy")
        if models._dino is None:
            with models.lock:  # first use: construct (incl. smoke forward) under the full lock
                models.get_dino_embedder()
        dino = models.get_dino_embedder()
        t0 = time.time()
        if op == "query":
            from media_manager import pattern_deep  # hook-allow: lazy — decode helper
            rgb = pattern_deep.load_rgb(req["image"])
            if rgb is None:
                return worker_protocol.pack({**meta, "queries": b"", "n": 0, "error": "could not decode image"})
            q = pdino.describe_query(dino, rgb, req["box"], mode=req.get("mode") or "both",
                                     rots=tuple(req.get("rots") or (0,)))
            print(f"[worker] handled {worker_protocol.PATH_PATTERN_DINO} query -> "
                  f"{0 if q is None else len(q)} vectors in {time.time() - t0:.2f}s", flush=True)
            return worker_protocol.pack({**meta, "queries": q.tobytes() if q is not None else b"",
                                         "n": 0 if q is None else int(q.shape[0]), "error": None})
        if op != "index":
            raise ValueError(f"unknown op {op!r}")
        if "paths" in req:
            data_root = req.get("data_root")
            problem = _data_root_problem(data_root)
            if problem:
                raise ValueError(problem)
            root_real = os.path.realpath(data_root)
            names = list(req["paths"])
            results = [{"name": r, "cells": b"", "n": 0, "error": None} for r in names]
            sources = []
            for i, rel in enumerate(names):
                try:
                    sources.append((i, resolve_under_root(root_real, rel)))
                except (ValueError, OSError) as e:
                    results[i]["error"] = f"rejected: {e}"
        else:
            items = req["items"]
            results = [{"name": it.get("name"), "cells": b"", "n": 0, "error": None} for it in items]
            sources = [(i, it["image"]) for i, it in enumerate(items)]
        stats = {"decode_seconds": 0.0, "gpu_seconds": 0.0, "lock_wait_seconds": 0.0}
        if sources:
            cells, errs = pdino.describe_files(dino, [s for _, s in sources], stats=stats)
            for (i, _), c, err in zip(sources, cells, errs):
                if c is None:
                    results[i]["error"] = err or "no cells returned"
                else:
                    results[i]["cells"], results[i]["n"] = c.tobytes(), int(c.shape[0])
        wall = time.time() - t0
        print(f"[worker] handled {worker_protocol.PATH_PATTERN_DINO} index ({len(results)} items) -> "
              f"{sum(1 for r in results if r['error'] is None)} ok, wall {wall:.2f}s "
              f"(decode {stats['decode_seconds']:.2f}s summed, gpu {stats['gpu_seconds']:.2f}s, "
              f"lock wait {stats['lock_wait_seconds']:.2f}s)", flush=True)
        return worker_protocol.pack({**meta, "results": results, "gpu_seconds": stats["gpu_seconds"],
                                     "decode_seconds": stats["decode_seconds"],
                                     "lock_wait_seconds": stats["lock_wait_seconds"],
                                     "wall_seconds": wall, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_PATTERN_DINO} ({op}) -> ERROR {e}", flush=True)
        return worker_protocol.pack({"results": [], "queries": b"", "n": 0, "algo": None, "dim": None,
                                     "proto": worker_protocol.PATTERN_DINO_PROTOCOL_VERSION,
                                     "error": str(e)})


# DEPRECATION PLANNED: superseded by the SigLIP 2 + DINOv2 embedding pipeline; do not extend.
def handle_region(path, data, request_id, link_id, remote_identity, requested_at):
    """Instance-level region search (region_index.py), DINOv2 on the worker's compute device.
      req {image, box}  -> {vecs: float32 bytes (S, DIM), n: S, algo}                  (query)
      req {image}       -> {windows: int8 bytes (n, DIM), n, pts: f16 bytes (k, 2),
                            desc: u8 bytes (k, 128), nkp: k, algo}                      (index)"""
    name = None
    try:
        from media_manager import region_index as ri  # hook-allow: lazy — pulls torch/cv2
        req = worker_protocol.unpack(data)
        name = req.get("name")
        pil = ri.load_pil(req["image"])
        if pil is None:
            return worker_protocol.pack({"error": "could not decode image"})
        with models.lock:
            if req.get("box") is not None:
                v = ri.describe_query(pil, req["box"])
                out = {"vecs": v.tobytes(), "n": int(v.shape[0])}
            else:
                w, pts, desc = ri.describe_image(pil)
                out = {"windows": w.tobytes(), "n": int(w.shape[0]),
                       "pts": pts.tobytes(), "desc": desc.tobytes(), "nkp": int(pts.shape[0])}
        print(f"[worker] handled {worker_protocol.PATH_REGION} ({name}) -> {out['n']} vectors", flush=True)
        return worker_protocol.pack({**out, "algo": ri.ALGO_TAG, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_REGION} ({name}) -> ERROR {e}", flush=True)
        return worker_protocol.pack({"error": str(e)})


def _face_detector_ready():
    """The FaceDetector, constructed (first use only) under the FULL lock so no other forward
    runs during a model load; steady state skips the lock entirely."""
    if models._face_detector is None:
        with models.lock:
            models.get_face_detector()
    return models._face_detector


class _FaceLog:
    """Rolls per-request face timings up into ONE worker log line every `every` images
    (replacing the per-request 'handled ...' lines, which drown the log in a bulk scan)."""

    def __init__(self, every=100):
        self.every = every
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self._agg = {}
        self._t0 = None
        self._faces = 0
        self._errors = 0

    def note(self, timings, faces=0, errors=0):
        now = time.time()
        with self._lock:
            if self._t0 is None:
                # first request of the window: back-date by its own time so img/s isn't inflated
                self._t0 = now - float((timings or {}).get('total_s') or 0.0)
            for k, v in (timings or {}).items():
                if isinstance(v, (int, float)):
                    self._agg[k] = self._agg.get(k, 0) + v
            self._faces += faces
            self._errors += errors
            if self._agg.get('images', 0) < self.every:
                return
            line = face_timing.summary_line(self._agg, now - self._t0)
            line = f"{line}, {self._faces} faces, {self._errors} errors"
            self._reset()
        print(line, flush=True)


_face_log = _FaceLog()


def _detect_faces_concurrent(file_paths, timings=None):
    """FaceDetector.detect_faces over local files WITHOUT holding models.lock for the whole
    call: image decode, JPEG down-scaling and the CPU PCN cascade of concurrent requests
    overlap, and only the GPU-bound InsightFace calls serialize on compute.GPU_LOCK (taken
    inside the detector, per call). Lock order stays load_lock -> GPU_LOCK (the detector
    never asks for load_lock while holding GPU_LOCK), so there is no cycle, and every other
    handler's `with models.lock` still excludes these forwards."""
    return _face_detector_ready().detect_faces(file_paths, gpu_lock=compute.GPU_LOCK, timings=timings)


def _pack_faces(faces):
    return [{
        "bbox": [float(v) for v in face["bbox"]],
        "embedding": face["embedding"].astype(np.float32).tobytes(),
        "det_score": float(face["det_score"]),
        "angle": float(face.get("angle", 0.0)),
    } for face in faces]


def handle_detect_faces_paths(path, data, request_id, link_id, remote_identity, requested_at):
    """Shared-disk variant of detect_faces for N files: the worker reads <data_root>/<rel>
    itself (same data_root/.media check and realpath containment as embed_both_paths; a
    rejected item gets its own error and is never opened). Replies one result per requested
    path, in request order."""
    t0 = time.time()
    try:
        req = worker_protocol.unpack(data)
        data_root = req.get("data_root")
        problem = _data_root_problem(data_root)
        if problem:
            raise ValueError(problem)
        root_real = os.path.realpath(data_root)
        rels = list(req["paths"])
        results = [{"name": r, "faces": [], "error": None} for r in rels]
        timings = {}
        todo = []  # (result index, real path)
        for i, rel in enumerate(rels):
            try:
                todo.append((i, resolve_under_root(root_real, rel)))
            except (ValueError, OSError) as e:
                results[i]["error"] = f"rejected: {e}"
        if todo:
            got = _detect_faces_concurrent([rp for _, rp in todo], timings)
            for (i, _rp), (_p, faces, error) in zip(todo, got):
                results[i]["faces"] = _pack_faces(faces) if error is None else []
                results[i]["error"] = error
        n_faces = sum(len(r["faces"]) for r in results)
        n_err = sum(1 for r in results if r["error"])
        wall = time.time() - t0
        for r in [r for r in results if r["error"]][:5]:  # errors stay loud; successes roll up
            print(f"[worker] {worker_protocol.PATH_DETECT_FACES_PATHS} {r['name']} -> ERROR {r['error']}",
                  flush=True)
        _face_log.note(timings, n_faces, n_err)
        return worker_protocol.pack({
            "results": results, "wall_seconds": wall, "timings": timings,
            "proto": worker_protocol.FACE_PROTOCOL_VERSION,
            "orient": worker_protocol.ORIENT_GATE_VERSION, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_DETECT_FACES_PATHS} -> ERROR {e}", flush=True)
        return worker_protocol.pack({
            "results": [], "proto": worker_protocol.FACE_PROTOCOL_VERSION, "error": str(e)})


def handle_detect_faces(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        tmp = _write_temp(req["image"], name)
        timings = {}
        _, faces, error = _detect_faces_concurrent([tmp], timings)[0]
        out_faces = _pack_faces(faces)
        if error:
            print(f"[worker] {worker_protocol.PATH_DETECT_FACES} ({name}) -> ERROR {error}", flush=True)
        _face_log.note(timings, len(out_faces), 1 if error else 0)
        return worker_protocol.pack({"faces": out_faces, "error": error, "timings": timings,
                                     "orient": worker_protocol.ORIENT_GATE_VERSION})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_DETECT_FACES} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"faces": [], "error": str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


def handle_embed_bbox(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    try:
        import cv2  # hook-allow: lazy — heavy
        req = worker_protocol.unpack(data)
        name = req.get("name")
        img = cv2.imdecode(np.frombuffer(req["image"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None (undecodable image bytes)")
        bbox = req["bbox"]
        pad_ratio = req.get("pad_ratio", 0.3)
        with models.lock:
            d = models.get_face_detector().embed_bbox(
                img, bbox, pad_ratio, auto_orient=bool(req.get("auto_orient", True)))
        resp = {
            "bbox": d["bbox"],
            "embedding": d["embedding"].astype(np.float32).tobytes(),
            "det_score": float(d["det_score"]),
            "angle": float(d.get("angle", 0.0)),
            "orient": worker_protocol.ORIENT_GATE_VERSION,
            "error": None,
        }
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BBOX} ({name}) -> "
              f"det_score={resp['det_score']:.3f}", flush=True)
        return worker_protocol.pack(resp)
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BBOX} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({
            "bbox": None, "embedding": None, "det_score": 0.0, "angle": 0.0,
            "error": str(e)})


def handle_normalize_faces(path, data, request_id, link_id, remote_identity, requested_at):
    """Re-run in-plane rotation normalization on faces the host already has rows
    for. Same image transport as embed_bbox (encoded bytes straight into
    cv2.imdecode) because the detector needs the decoded array, not a file; the
    reply keeps the request's face order so the host can map results to DB rows
    positionally."""
    name = None
    try:
        import cv2  # hook-allow: lazy — heavy
        req = worker_protocol.unpack(data)
        name = req.get("name")
        img = cv2.imdecode(np.frombuffer(req["image"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None (undecodable image bytes)")
        faces = []
        for face in (req.get("faces") or []):
            emb = face.get("embedding")
            faces.append({
                "bbox": [float(v) for v in face["bbox"]],
                "embedding": (np.frombuffer(emb, dtype=np.float32).copy()
                              if emb else None),
                "det_score": float(face.get("det_score") or 0.0),
                "angle": float(face.get("angle") or 0.0),
            })
        with models.lock:
            out = models.get_face_detector().normalize_faces(img, faces)
        out_faces = []
        changed = 0
        for i, face in enumerate(out):
            emb = face.get("embedding")
            out_faces.append({
                "bbox": [float(v) for v in face["bbox"]],
                "embedding": (emb.astype(np.float32).tobytes()
                              if emb is not None else b""),
                "det_score": float(face["det_score"]),
                "angle": float(face.get("angle", 0.0)),
            })
            if face.get("angle", 0.0) != faces[i]["angle"]:
                changed += 1
        print(f"[worker] handled {worker_protocol.PATH_NORMALIZE_FACES} ({name}) -> "
              f"{len(out_faces)} faces, {changed} re-angled", flush=True)
        return worker_protocol.pack({"faces": out_faces, "error": None,
                                     "orient": worker_protocol.ORIENT_GATE_VERSION})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_NORMALIZE_FACES} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"faces": [], "error": str(e)})


def handle_embed_image(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        tmp = _write_temp(req["image"], name)
        with models.lock:
            emb, failed = models.get_clip_indexer().embed_images([tmp])
        if failed or emb.shape[0] == 0:
            err = failed[0][1] if failed else "no embedding"
            print(f"[worker] handled {worker_protocol.PATH_EMBED_IMAGE} ({name}) -> "
                  f"err={err}", flush=True)
            return worker_protocol.pack({"embedding": None, "error": err})
        print(f"[worker] handled {worker_protocol.PATH_EMBED_IMAGE} ({name}) -> "
              f"embedded dim={emb.shape[1]}", flush=True)
        return worker_protocol.pack({
            "embedding": emb[0].astype(np.float32).tobytes(), "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_IMAGE} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"embedding": None, "error": str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


def handle_embed_text(path, data, request_id, link_id, remote_identity, requested_at):
    text = None
    try:
        req = worker_protocol.unpack(data)
        text = req["text"]
        with models.lock:
            v = models.get_clip_indexer().embed_text(text)
        print(f"[worker] handled {worker_protocol.PATH_EMBED_TEXT} ({text!r}) -> "
              f"embedded dim={v.shape[0]}", flush=True)
        return worker_protocol.pack({
            "embedding": v.astype(np.float32).tobytes(), "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_TEXT} ({text!r}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"embedding": None, "error": str(e)})


def _embed_shared(embedders_, sources, stats=None, labels=None):
    """Indirection so the torch-importing embedders module loads lazily (and tests can fake it)."""
    from .embedders import embed_shared  # hook-allow: lazy — pulls torch
    return embed_shared(embedders_, sources, labels=labels, stats=stats)


def _embed_items_reply(path_name, results, idx_of, images=None):
    """Run SigLIP+DINO over the files in ``idx_of`` (file path -> [result indices]) and fill
    ``results`` in place; returns the packed reply shared by embed_both_batch / _paths.

    Only model LOADING holds the load lock; the forward passes take compute.GPU_LOCK inside
    embed_shared, so concurrent requests overlap decode with each other's GPU work."""
    t0 = time.time()
    stats = {"decode_seconds": 0.0, "gpu_seconds": 0.0, "lock_wait_seconds": 0.0}
    if models._siglip is None or models._dino is None:
        # First use: construct (incl. the smoke forward on the device) under the FULL lock so no
        # other forward runs during a load. Steady state skips the lock entirely.
        with models.lock:
            models.get_siglip_embedder()
            models.get_dino_embedder()
    sig, dino = models.get_siglip_embedder(), models.get_dino_embedder()
    order = list(idx_of)
    if order and images is not None:   # in-memory PIL images keyed by a string label (masked path)
        (sig_e, dino_e), failed = _embed_shared([sig, dino], [images[k] for k in order], stats, labels=order)
    elif order:
        (sig_e, dino_e), failed = _embed_shared([sig, dino], order, stats)
    else:
        sig_e = dino_e = None
        failed = []
    wall_s = time.time() - t0
    gpu_s = float(stats.get("gpu_seconds", 0.0))
    failed_paths = {}
    for label, msg in failed:
        failed_paths[label] = msg
    row = 0  # embed_shared drops failed rows; surviving rows keep input order
    n_ok = 0
    for fpath in order:
        idxs = idx_of[fpath]
        if fpath in failed_paths:
            for i in idxs:
                results[i]["error"] = failed_paths[fpath]
            continue
        sv = sig_e[row].astype(np.float32).tobytes()
        dv = dino_e[row].astype(np.float32).tobytes()
        for i in idxs:
            results[i]["siglip"], results[i]["dino"] = sv, dv
            n_ok += 1
        row += 1
    print(f"[worker] handled {path_name} ({len(results)} items) -> "
          f"{n_ok} ok, {len(results) - n_ok} failed, wall {wall_s:.2f}s "
          f"(decode {stats.get('decode_seconds', 0.0):.2f}s summed, gpu {gpu_s:.2f}s, "
          f"lock wait {stats.get('lock_wait_seconds', 0.0):.2f}s)", flush=True)
    return worker_protocol.pack({
        "results": results, "siglip_model": sig.model_id(), "dino_model": dino.model_id(),
        "gpu_seconds": gpu_s, "wall_seconds": wall_s,
        "decode_seconds": float(stats.get("decode_seconds", 0.0)),
        "lock_wait_seconds": float(stats.get("lock_wait_seconds", 0.0)),
        "proto": worker_protocol.EMBED_PROTOCOL_VERSION,
        "error": None})


def _data_root_problem(data_root):
    """None if ``data_root`` is an acceptable shared root, else a reason string. A root is
    accepted only if it is an absolute path with an existing ``.media`` directory (the marker
    of a media repo, like .git) -- so a client can't point the worker at an arbitrary tree."""
    if not isinstance(data_root, str) or not data_root or "\0" in data_root:
        return "data_root missing or invalid"
    if not os.path.isabs(data_root):
        return "data_root is not an absolute path"
    if not os.path.isdir(data_root):
        return "data_root does not exist on the worker"
    if not os.path.isdir(os.path.join(data_root, ".media")):
        return "data_root has no .media directory on the worker"
    return None


def resolve_under_root(root_real, rel):
    """Realpath of ``<root_real>/<rel>`` if it is a regular file strictly inside ``root_real``
    (symlinks resolved), else raise ValueError. Rejects absolute / traversal / NUL paths and
    symlinks that escape the root. ``root_real`` must already be os.path.realpath()'d."""
    if not isinstance(rel, str) or not rel or "\0" in rel:
        raise ValueError("invalid path")
    if os.path.isabs(rel) or rel.startswith(("/", "\\")):
        raise ValueError("absolute path rejected")
    real = os.path.realpath(os.path.join(root_real, rel))
    if os.path.commonpath([root_real, real]) != root_real or real == root_real:
        raise ValueError("path outside data_root")
    if not os.path.isfile(real):
        raise ValueError("not a file")
    return real


def handle_probe_data_root(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        reason = _data_root_problem(req.get("data_root"))
        return worker_protocol.pack({
            "visible": reason is None, "reason": reason or "ok",
            "proto": worker_protocol.EMBED_PROTOCOL_VERSION})
    except Exception as e:  # hook-allow: broad — report error to caller
        return worker_protocol.pack({
            "visible": False, "reason": str(e), "proto": worker_protocol.EMBED_PROTOCOL_VERSION})


def handle_embed_both_paths(path, data, request_id, link_id, remote_identity, requested_at):
    """Shared-disk variant of embed_both_batch: the worker reads ``<data_root>/<rel>`` itself.
    Whole request refused unless data_root has a .media dir; each file must realpath-resolve
    inside realpath(data_root) or that item gets an error (and is never opened)."""
    try:
        req = worker_protocol.unpack(data)
        data_root = req.get("data_root")
        problem = _data_root_problem(data_root)
        if problem:
            raise ValueError(problem)
        root_real = os.path.realpath(data_root)
        rels = req["paths"]
        results = [{"name": r, "siglip": None, "dino": None, "error": None} for r in rels]
        idx_of = {}  # real file path -> result indices
        for i, rel in enumerate(rels):
            try:
                idx_of.setdefault(resolve_under_root(root_real, rel), []).append(i)
            except (ValueError, OSError) as e:
                results[i]["error"] = f"rejected: {e}"
        return _embed_items_reply(worker_protocol.PATH_EMBED_BOTH_PATHS, results, idx_of)
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BOTH_PATHS} -> ERROR {e}", flush=True)
        return worker_protocol.pack({
            "results": [], "siglip_model": None, "dino_model": None, "gpu_seconds": 0.0,
            "proto": worker_protocol.EMBED_PROTOCOL_VERSION, "error": str(e)})


def handle_embed_both_batch(path, data, request_id, link_id, remote_identity, requested_at):
    """N images -> SigLIP2 + DINOv2 vectors in ONE decode pass. Per-item failures (bad image,
    unsupported extension) are reported in that item's ``error`` and never fail the batch."""
    tmps = []
    try:
        req = worker_protocol.unpack(data)
        items = req["items"]
        results = [{"name": it.get("name"), "siglip": None, "dino": None, "error": None}
                   for it in items]
        idx_of = {}  # temp path -> result index
        for i, it in enumerate(items):
            try:
                tmp = _write_temp(it["image"], it.get("name"))
            except Exception as e:  # hook-allow: broad — per-item isolation
                results[i]["error"] = f"write failed: {e}"
                continue
            tmps.append(tmp)
            idx_of[tmp] = i
        return _embed_items_reply(worker_protocol.PATH_EMBED_BOTH_BATCH, results,
                                  {p: [i] for p, i in idx_of.items()})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BOTH_BATCH} -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({
            "results": [], "siglip_model": None, "dino_model": None, "gpu_seconds": 0.0,
            "proto": worker_protocol.EMBED_PROTOCOL_VERSION, "error": str(e)})
    finally:
        for t in tmps:
            _cleanup(t)


def _mask_sources(sources, boxes_list, max_coverage, results, idx):
    """Decode + mask each source (a path or a temp file) in a thread pool; returns ({label:
    [result index]}, {label: PIL image}) for the ones that worked (PIL images are unhashable, so
    they travel under a string label). Failures / over-coverage go to results[i]["error"]."""
    from concurrent.futures import ThreadPoolExecutor
    from . import location_mask  # hook-allow: lazy -- PIL/numpy only
    out, images = {}, {}

    def one(k):
        return location_mask.decode_masked(sources[k], boxes_list[k])

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(sources)))) as pool:
        futs = [pool.submit(one, k) for k in range(len(sources))]
        for k, f in enumerate(futs):
            i = idx[k]
            try:
                img, cov = f.result()
            except Exception as e:  # hook-allow: broad -- per-item isolation
                results[i]["error"] = f"decode failed: {e}"
                continue
            results[i]["coverage"] = round(float(cov), 4)
            if cov > max_coverage:
                results[i]["error"] = f"mask covers {cov:.0%} of the frame (> {max_coverage:.0%})"
                continue
            out[f"mask:{i}"] = [i]
            images[f"mask:{i}"] = img
    return out, images


def _masked_reply(path_name, results, masked):
    imgs, images = masked
    reply = worker_protocol.unpack(_embed_items_reply(path_name, results, imgs, images)) if imgs else {
        "results": results, "siglip_model": None, "dino_model": None, "gpu_seconds": 0.0,
        "proto": worker_protocol.EMBED_PROTOCOL_VERSION, "error": None}
    reply["masked_proto"] = worker_protocol.MASKED_PROTOCOL_VERSION
    return worker_protocol.pack(reply)


def _masked_error(path_name, e):
    traceback.print_exc()
    print(f"[worker] handled {path_name} -> ERROR {e}", flush=True)
    return worker_protocol.pack({
        "results": [], "siglip_model": None, "dino_model": None, "gpu_seconds": 0.0,
        "proto": worker_protocol.EMBED_PROTOCOL_VERSION,
        "masked_proto": worker_protocol.MASKED_PROTOCOL_VERSION, "error": str(e)})


def handle_embed_masked_paths(path, data, request_id, link_id, remote_identity, requested_at):
    """Person-masked SigLIP 2 + DINOv2 over shared-disk files. Same data_root / containment rules
    as handle_embed_both_paths: whole request refused unless data_root has a .media dir; each file
    must realpath-resolve inside it or that item errors and is never opened."""
    try:
        req = worker_protocol.unpack(data)
        data_root = req.get("data_root")
        problem = _data_root_problem(data_root)
        if problem:
            raise ValueError(problem)
        root_real = os.path.realpath(data_root)
        rels, boxes = req["paths"], req["boxes"]
        if len(boxes) != len(rels):
            raise ValueError("boxes/paths length mismatch")
        max_cov = float(req.get("max_coverage", 0.5))
        results = [{"name": r, "siglip": None, "dino": None, "error": None, "coverage": None} for r in rels]
        srcs, bx, idx = [], [], []
        for i, rel in enumerate(rels):
            try:
                srcs.append(resolve_under_root(root_real, rel))
            except (ValueError, OSError) as e:
                results[i]["error"] = f"rejected: {e}"
                continue
            bx.append(boxes[i])
            idx.append(i)
        return _masked_reply(worker_protocol.PATH_EMBED_MASKED_PATHS, results,
                             _mask_sources(srcs, bx, max_cov, results, idx))
    except Exception as e:  # hook-allow: broad — report error to caller/job
        return _masked_error(worker_protocol.PATH_EMBED_MASKED_PATHS, e)


def handle_embed_masked_batch(path, data, request_id, link_id, remote_identity, requested_at):
    """Upload variant of handle_embed_masked_paths: items carry the image bytes."""
    tmps = []
    try:
        req = worker_protocol.unpack(data)
        items = req["items"]
        max_cov = float(req.get("max_coverage", 0.5))
        results = [{"name": it.get("name"), "siglip": None, "dino": None, "error": None, "coverage": None}
                   for it in items]
        srcs, bx, idx = [], [], []
        for i, it in enumerate(items):
            try:
                tmp = _write_temp(it["image"], it.get("name"))
            except Exception as e:  # hook-allow: broad — per-item isolation
                results[i]["error"] = f"write failed: {e}"
                continue
            tmps.append(tmp)
            srcs.append(tmp)
            bx.append(it.get("boxes") or [])
            idx.append(i)
        return _masked_reply(worker_protocol.PATH_EMBED_MASKED_BATCH, results,
                             _mask_sources(srcs, bx, max_cov, results, idx))
    except Exception as e:  # hook-allow: broad — report error to caller/job
        return _masked_error(worker_protocol.PATH_EMBED_MASKED_BATCH, e)
    finally:
        for t in tmps:
            _cleanup(t)


def handle_embed_siglip_text(path, data, request_id, link_id, remote_identity, requested_at):
    text = None
    try:
        req = worker_protocol.unpack(data)
        text = req["text"]
        if models._siglip is None:
            with models.lock:  # first use: load under the full lock
                models.get_siglip_embedder()
        sig = models.get_siglip_embedder()
        v = sig.embed_text(text)  # takes compute.GPU_LOCK itself
        return worker_protocol.pack({
            "embedding": np.asarray(v, dtype=np.float32).reshape(-1).tobytes(),
            "siglip_model": sig.model_id(),
            "proto": worker_protocol.EMBED_PROTOCOL_VERSION, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_SIGLIP_TEXT} ({text!r}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({
            "embedding": None, "siglip_model": None,
            "proto": worker_protocol.EMBED_PROTOCOL_VERSION, "error": str(e)})


def handle_detect_objects(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        vocab = req.get("vocab")
        tmp = _write_temp(req["image"], name)
        # Hold the lock across set_vocab + detect so a concurrent request can't
        # swap the shared detector's vocabulary between the two calls. The detector
        # is a shared singleton, so ALWAYS set the vocab for this request — falling
        # back to the default when the caller sent none, rather than silently reusing
        # whatever the previous request left on it.
        from .detector import DEFAULT_VOCAB  # hook-allow: lazy — model module
        with models.lock:
            det = models.get_object_detector()
            det.set_vocab(vocab if vocab else list(DEFAULT_VOCAB))
            res = det.detect_images([tmp])
        _, detections, error = res[0]
        out = [[d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])]
               for d in detections]
        summary = f"{len(out)} detections" if error is None else f"err={error}"
        print(f"[worker] handled {worker_protocol.PATH_DETECT_OBJECTS} ({name}) -> {summary}",
              flush=True)
        return worker_protocol.pack({"detections": out, "error": error})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_DETECT_OBJECTS} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"detections": [], "error": str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


def handle_estimate_age(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        # Map the wire faces to the shape AgeGenderEstimator.estimate expects
        # ('ref' + 'bbox'); it returns [{face_ref, age, gender}, ...] already.
        faces = [{"ref": f["face_ref"], "bbox": f["bbox"]} for f in (req.get("faces") or [])]
        bodies = [{"ref": b["face_ref"], "bbox": b["bbox"]} for b in (req.get("bodies") or [])]
        tmp = _write_temp(req["image"], name)
        # Shares models.lock (NOT a separate lock) so MiVOLO's GPU work never runs
        # concurrently with OpenVINO face/CLIP/YOLO inference — see the note by
        # _age_estimator above (prevents Arc CL_OUT_OF_RESOURCES).
        with models.lock:
            results = _get_age_estimator().estimate(tmp, faces, bodies)
        print(f"[worker] handled {worker_protocol.PATH_ESTIMATE_AGE} ({name}) -> "
              f"{len(results)} face(s)/body(ies)", flush=True)
        return worker_protocol.pack({"results": results, "error": None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_ESTIMATE_AGE} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"results": [], "error": str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


# ---------------------------------------------------------------------------
# Per-tag model training (job/poll) + fine-tuned inference.
#
# Training is stateful and long-running, unlike the stateless inference ops
# above: each job carries an id, accumulates uploaded data into a scratch dir,
# and runs in its OWN daemon thread that does NOT hold models.lock — so ping and
# inference stay responsive during a multi-minute fine-tune. YOLO checkpoints are
# persisted under TAG_MODELS_DIR and served by handle_tag_detect; the small CLIP
# artifact (weights.npz) is handed back in the final status response. Per-job
# progress/log is tracked explicitly (ultralytics' own stdout can't be captured
# per-job without clobbering other requests' prints).
# ---------------------------------------------------------------------------

TAG_MODELS_DIR = os.path.expanduser('~/.cache/media_manager/tag_models')
_TRAIN_DEFAULT_EPOCHS = 30
_MAX_JOBS = 32
_TAG_MODEL_CACHE_MAX = 4


class TrainingJob:
    def __init__(self, job_id, kind, tag, slug, epochs):
        self.job_id = job_id
        self.kind = kind
        self.tag = tag
        self.slug = slug
        self.total_epochs = epochs
        self.current_epoch = 0
        self.status = 'created'          # created|running|done|failed|cancelled
        self.error = None
        self.metrics = {}
        self.cancel_event = threading.Event()
        self.thread = None
        self.artifact_bytes = None       # clip weights.npz bytes, set on done
        self.scratch_dir = tempfile.mkdtemp(prefix=f'mmtrain_{slug}_')
        self._log = collections.deque(maxlen=300)
        # YOLO staging: [(staging_image_path, [[cx,cy,w,h] normalized, ...]), ...]
        self.yolo_examples = []
        self._yolo_counter = 0
        # embedding classifier (siglip|dino|legacy clip): whole-image [(emb_bytes, label)]
        # and region [(crop_jpeg, label)]
        self.clip_whole = []
        self.clip_regions = []

    def log(self, msg):
        self._log.append(f'[{int(time.time())}] {msg}')
        print(f'[worker][train {self.job_id}] {msg}', flush=True)

    def log_text(self):
        return '\n'.join(self._log)


class TrainingJobs:
    """In-memory registry of training jobs, guarded by its OWN lock (never
    models.lock) so status polls don't queue behind inference."""

    def __init__(self):
        self.lock = threading.Lock()
        self._jobs = collections.OrderedDict()
        self._counter = 0

    def create(self, kind, tag, slug, epochs):
        with self.lock:
            self._counter += 1
            job = TrainingJob(f'{os.getpid()}-{self._counter}', kind, tag, slug, epochs)
            self._jobs[job.job_id] = job
            # Evict oldest FINISHED jobs beyond the cap (and clean their scratch).
            for old_id in list(self._jobs):
                if len(self._jobs) <= _MAX_JOBS:
                    break
                old = self._jobs[old_id]
                if old.status in ('running', 'created'):
                    continue
                del self._jobs[old_id]
                shutil.rmtree(old.scratch_dir, ignore_errors=True)
            return job

    def get(self, job_id):
        with self.lock:
            return self._jobs.get(job_id)


training_jobs = TrainingJobs()

# Loaded per-tag YOLO checkpoints for inference, slug -> (mtime, model). Bounded
# LRU; only touched while holding models.lock (it's inference).
_tag_model_cache = collections.OrderedDict()


def _write_yolo_split(dataset_dir, split_name, examples):
    """Write one ultralytics split (images/{split} + labels/{split}) from
    (staging_image_path, normalized_boxes) tuples. Boxes are already
    [cx, cy, w, h] in [0,1] (the host normalized them before shipping)."""
    images_dir = os.path.join(dataset_dir, 'images', split_name)
    labels_dir = os.path.join(dataset_dir, 'labels', split_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for i, (src_path, boxes) in enumerate(examples):
        ext = os.path.splitext(src_path)[1] or '.jpg'
        name = f'{i:05d}'
        shutil.copyfile(src_path, os.path.join(images_dir, name + ext))
        lines = [f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for cx, cy, w, h in boxes]
        with open(os.path.join(labels_dir, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def _run_yolo_job(job):
    try:
        examples = job.yolo_examples
        if not examples:
            raise ValueError('no training images were uploaded')
        # 80/20 split by image (deterministic seed), same as the local trainer.
        rng = random.Random(0)
        shuffled = examples[:]
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.2))
        val_ex, train_ex = shuffled[:split], shuffled[split:]
        if not train_ex:
            train_ex, val_ex = val_ex, train_ex

        dataset_dir = os.path.join(job.scratch_dir, 'dataset')
        _write_yolo_split(dataset_dir, 'train', train_ex)
        _write_yolo_split(dataset_dir, 'val', val_ex or train_ex)
        data_yaml = os.path.join(dataset_dir, 'data.yaml')
        with open(data_yaml, 'w') as f:
            f.write(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n"
                    f"nc: 1\nnames: ['{job.tag}']\n")

        out_dir = os.path.join(TAG_MODELS_DIR, job.slug, 'yolo_model')
        n_boxes = sum(len(b) for _p, b in examples)
        job.log(f'training YOLO on {len(examples)} images / {n_boxes} boxes, {job.total_epochs} epochs')

        def progress(current_epoch, total_epochs):
            job.current_epoch = current_epoch
            job.total_epochs = total_epochs
            job.log(f'epoch {current_epoch}/{total_epochs}')

        res = yt.train_from_dataset(data_yaml, out_dir, epochs=job.total_epochs,
                                    progress_cb=progress, cancel_cb=job.cancel_event.is_set)
        job.metrics = {'held_out_map': res['held_out_map'],
                       'n_images': len(examples), 'n_boxes': n_boxes}
        job.status = 'done'
        job.log(f'done; checkpoint {res["best_path"]} map={res["held_out_map"]}')
    except yt.TrainingCancelled:
        job.status = 'cancelled'
        job.error = 'Cancelled by user.'
        job.log('cancelled')
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        job.status = 'failed'
        job.error = str(e)
        job.log(f'failed: {e}')
    finally:
        shutil.rmtree(os.path.join(job.scratch_dir, 'staging'), ignore_errors=True)


_EMBED_KIND_TO_SPACE = {'siglip_model': 'siglip', 'dino_model': 'dino', 'clip_model': 'clip'}


def _worker_embedder_for_space(space):
    """The worker's image embedder for a tag-classifier space. Caller holds models.lock."""
    if space == 'siglip':
        return models.get_siglip_embedder()
    if space == 'dino':
        return models.get_dino_embedder()
    return models.get_clip_indexer()  # DEPRECATION PLANNED: legacy CLIP; superseded by SigLIP 2 + DINOv2, do not extend.


def _run_clip_job(job):
    """Embedding-space tag classifier (kind siglip_model | dino_model | legacy clip_model)."""
    try:
        space = _EMBED_KIND_TO_SPACE[job.kind]
        xs, ys = [], []
        for emb_bytes, label in job.clip_whole:
            xs.append(np.frombuffer(emb_bytes, dtype=np.float32))
            ys.append(label)
        if job.clip_regions:
            from PIL import Image as PILImage  # hook-allow: lazy — heavy
            crops, labels = [], []
            for crop_bytes, label in job.clip_regions:
                crops.append(PILImage.open(io.BytesIO(crop_bytes)).convert('RGB'))
                labels.append(label)
            job.log(f'embedding {len(crops)} region crops on the worker {space} model')
            with models.lock:
                embs = _worker_embedder_for_space(space).embed_pil_images(crops)
            for e, l in zip(embs, labels):
                xs.append(e.astype(np.float32))
                ys.append(l)

        n_pos = sum(1 for y in ys if y == 1.0)
        n_neg = sum(1 for y in ys if y == 0.0)
        if n_pos < ct.MIN_EXAMPLES_PER_CLASS or n_neg < ct.MIN_EXAMPLES_PER_CLASS:
            raise ValueError(
                f'Need at least {ct.MIN_EXAMPLES_PER_CLASS} of each class; '
                f'have {n_pos} confirmed, {n_neg} rejected.')
        if job.cancel_event.is_set():
            job.status = 'cancelled'
            job.error = 'Cancelled by user.'
            job.log('cancelled')
            return

        X = np.stack(xs).astype(np.float32)
        y = np.array(ys, dtype=np.float32)
        dims = {x.shape[0] for x in xs}
        if len(dims) != 1:
            raise ValueError(f'mixed embedding dims {sorted(dims)} in {space} examples — re-embed, then retrain')
        job.log(f'fitting {space} logistic on {n_pos} pos / {n_neg} neg')
        w, b, cv = ct.fit_from_examples(X, y)

        out_dir = os.path.join(TAG_MODELS_DIR, job.slug, job.kind)
        os.makedirs(out_dir, exist_ok=True)
        weights_path = os.path.join(out_dir, 'weights.npz')
        ct.save_weights(weights_path, w, b, space)
        with open(weights_path, 'rb') as f:
            job.artifact_bytes = f.read()
        job.metrics = {'cv_accuracy': round(cv, 4), 'n_positive': n_pos, 'n_negative': n_neg}
        job.status = 'done'
        job.log(f'done; cv_accuracy={round(cv, 4)}')
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        job.status = 'failed'
        job.error = str(e)
        job.log(f'failed: {e}')


def _get_tag_model(slug, checkpoint):
    """Load (cached, LRU) a per-tag YOLO checkpoint for inference. MUST be called
    while holding models.lock (mirrors the other inference paths)."""
    from ultralytics import YOLOWorld  # hook-allow: lazy — heavy model dep
    mtime = os.path.getmtime(checkpoint)
    cached = _tag_model_cache.get(slug)
    if cached is not None and cached[0] == mtime:
        _tag_model_cache.move_to_end(slug)
        return cached[1]
    print(f"[worker] loading tag YOLO checkpoint for {slug}...", flush=True)
    model = YOLOWorld(checkpoint)
    _tag_model_cache[slug] = (mtime, model)
    _tag_model_cache.move_to_end(slug)
    while len(_tag_model_cache) > _TAG_MODEL_CACHE_MAX:
        _tag_model_cache.popitem(last=False)
    return model


def _parse_yolo_results(results):
    """[[class_name, conf, x1, y1, x2, y2], ...] with NORMALIZED (0..1) xyxy —
    the tag-suggestion swipe frontend positions a highlight box as plain CSS
    percentages, so it wants resolution-independent coords."""
    out = []
    for r in results:
        names = getattr(r, 'names', {}) or {}
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for b in boxes:
            cls = int(b.cls[0]) if b.cls is not None else 0
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            x1, y1, x2, y2 = [float(v) for v in b.xyxyn[0].tolist()]
            out.append([names.get(cls, str(cls)), conf, x1, y1, x2, y2])
    return out


def handle_train_create(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        kind = req['kind']
        if kind != 'yolo_model' and kind not in _EMBED_KIND_TO_SPACE:
            raise ValueError(f'unknown training kind {kind!r}')
        tag = req.get('tag', '')
        slug = req['slug']
        epochs = int(req.get('epochs') or _TRAIN_DEFAULT_EPOCHS)
        job = training_jobs.create(kind, tag, slug, epochs)
        job.log(f'created {kind} job for tag {tag!r} (slug={slug})')
        print(f"[worker] handled {worker_protocol.PATH_TRAIN_CREATE} ({kind} {slug}) "
              f"-> {job.job_id}", flush=True)
        return worker_protocol.pack({'job_id': job.job_id, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'job_id': None, 'error': str(e)})


def handle_train_add(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        job = training_jobs.get(req.get('job_id'))
        if job is None:
            raise ValueError(f"unknown job {req.get('job_id')!r}")
        if job.status != 'created':
            raise ValueError(f'job {job.job_id} is {job.status}; cannot add data')
        batch = req.get('batch') or []
        if job.kind == 'yolo_model':
            staging = os.path.join(job.scratch_dir, 'staging')
            os.makedirs(staging, exist_ok=True)
            for item in batch:
                ext = _ext_from_name(item.get('name'))
                if ext.lower() not in ('.jpg', '.jpeg', '.png'):
                    ext = '.jpg'  # host always ships JPEG; keep a decodable ext
                p = os.path.join(staging, f'{job._yolo_counter:06d}{ext}')
                with open(p, 'wb') as f:
                    f.write(item['image'])
                job._yolo_counter += 1
                job.yolo_examples.append(
                    (p, [[float(v) for v in box] for box in item['boxes']]))
        else:
            for item in batch:
                label = float(item['label'])
                if item.get('embedding') is not None:
                    job.clip_whole.append((item['embedding'], label))
                elif item.get('image') is not None:
                    job.clip_regions.append((item['image'], label))
        return worker_protocol.pack({'received': len(batch), 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'received': 0, 'error': str(e)})


def handle_train_run(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        job = training_jobs.get(req.get('job_id'))
        if job is None:
            raise ValueError(f"unknown job {req.get('job_id')!r}")
        if job.status != 'created':
            raise ValueError(f'job {job.job_id} already {job.status}')
        job.status = 'running'
        target = _run_yolo_job if job.kind == 'yolo_model' else _run_clip_job
        job.thread = threading.Thread(target=target, args=(job,), daemon=True)
        job.thread.start()
        print(f"[worker] handled {worker_protocol.PATH_TRAIN_RUN} ({job.job_id}) -> started",
              flush=True)
        return worker_protocol.pack({'ok': True, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'ok': False, 'error': str(e)})


def handle_train_status(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        job = training_jobs.get(req.get('job_id'))
        if job is None:
            return worker_protocol.pack({
                'status': 'failed', 'error': 'unknown job',
                'current_epoch': 0, 'total_epochs': 0, 'metrics': {},
                'log_tail': '', 'artifact': None})
        want_artifact = bool(req.get('want_artifact'))
        return worker_protocol.pack({
            'status': job.status,
            'current_epoch': job.current_epoch,
            'total_epochs': job.total_epochs,
            'metrics': job.metrics,
            'log_tail': job.log_text(),
            'error': job.error,
            'artifact': job.artifact_bytes if (want_artifact and job.status == 'done') else None,
        })
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({
            'status': 'failed', 'error': str(e),
            'current_epoch': 0, 'total_epochs': 0, 'metrics': {},
            'log_tail': '', 'artifact': None})


def handle_train_cancel(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        job = training_jobs.get(req.get('job_id'))
        if job is None:
            raise ValueError(f"unknown job {req.get('job_id')!r}")
        job.cancel_event.set()
        job.log('cancel requested')
        if job.status == 'created':   # never started — mark cancelled now
            job.status = 'cancelled'
            job.error = 'Cancelled by user.'
        return worker_protocol.pack({'ok': True, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'ok': False, 'error': str(e)})


def handle_tag_detect(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        slug = req['slug']
        name = req.get('name')
        conf = float(req.get('conf') or DEFAULT_CONF_THRESHOLD)
        checkpoint = os.path.join(TAG_MODELS_DIR, slug, 'yolo_model', 'best.pt')
        if not os.path.isfile(checkpoint):
            return worker_protocol.pack({
                'detections': [],
                'error': f'no trained model for slug {slug!r} on worker'})
        tmp = _write_temp(req['image'], name)
        with models.lock:
            model = _get_tag_model(slug, checkpoint)
            results = model.predict(tmp, conf=conf, verbose=False,
                                    device=compute.ultralytics_device())
        detections = _parse_yolo_results(results)
        print(f"[worker] handled {worker_protocol.PATH_TAG_DETECT} ({slug}/{name}) -> "
              f"{len(detections)} detections", flush=True)
        return worker_protocol.pack({'detections': detections, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_TAG_DETECT} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({'detections': [], 'error': str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


# --- imdb: resident in-memory search index --------------------------------------
class ImdbIndex:
    """The worker's resident CLIP + face embedding matrices for search offload, plus
    a receive buffer used WHILE a build streams in from the host. The host ships the
    matrix in bounded chunks (imdb_build_begin → many imdb_build_chunk → imdb_build_end)
    so neither side ever holds the whole ~1 GB blob in one message; only the final
    stacked matrix is kept resident. All state guarded by `lock`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.matrices = {}   # kind -> {'ids': int64[N], 'matrix': float32[N,D], 'built_at': int}
        self._buf = {}       # kind -> {'dim', 'ids': [int], 'chunks': [bytes], 'rows'} while building

    def begin(self, kind, dim):
        with self.lock:
            self._buf[kind] = {'dim': int(dim), 'ids': [], 'chunks': [], 'rows': 0}

    def add_chunk(self, kind, ids, vecs):
        with self.lock:
            buf = self._buf.get(kind)
            if buf is None:
                raise ValueError(f'no imdb build in progress for {kind!r} (call begin first)')
            buf['ids'].extend(ids)
            buf['chunks'].append(vecs)
            buf['rows'] += len(ids)
            return buf['rows']

    def end(self, kind, now):
        with self.lock:
            buf = self._buf.pop(kind, None)
            if buf is None:
                raise ValueError(f'no imdb build in progress for {kind!r}')
            dim = buf['dim']
            # One contiguous copy: join the chunk bytes, view as float32, own it, then
            # let the transient join-buffer go. ~2x the matrix size for a moment (fine
            # on the big worker box), vs never materializing it on the low-RAM host.
            matrix = np.frombuffer(b''.join(buf['chunks']), dtype=np.float32).reshape(-1, dim).copy()
            ids = np.array(buf['ids'], dtype=np.int64)
            if matrix.shape[0] != ids.shape[0]:
                raise ValueError(f'{kind}: {matrix.shape[0]} vectors but {ids.shape[0]} ids')
            self.matrices[kind] = {'ids': ids, 'matrix': matrix, 'built_at': now}
            return matrix.shape[0], dim, int(matrix.nbytes)

    def status(self):
        with self.lock:
            out = {}
            for kind in ('clip', 'face'):
                m = self.matrices.get(kind)
                building = kind in self._buf
                if m is not None:
                    out[kind] = {'rows': int(m['matrix'].shape[0]), 'dim': int(m['matrix'].shape[1]),
                                 'bytes': int(m['matrix'].nbytes), 'built_at': m['built_at'],
                                 'building': building}
                else:
                    out[kind] = {'rows': 0, 'dim': 0, 'bytes': 0, 'built_at': None, 'building': building}
            return out


imdb_index = ImdbIndex()


def handle_imdb_build_begin(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        kind = req['kind']
        imdb_index.begin(kind, req['dim'])
        print(f"[worker] imdb build begin kind={kind} dim={req['dim']} total={req.get('total_rows')}",
              flush=True)
        return worker_protocol.pack({'ok': True, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'ok': False, 'error': str(e)})


def handle_imdb_build_chunk(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        received = imdb_index.add_chunk(req['kind'], req['ids'], req['vecs'])
        return worker_protocol.pack({'received': received, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'received': 0, 'error': str(e)})


def handle_imdb_build_end(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        rows, dim, nbytes = imdb_index.end(req['kind'], int(time.time()))
        print(f"[worker] imdb build end kind={req['kind']} rows={rows} dim={dim} "
              f"bytes={nbytes} ({nbytes / 1048576:.1f} MB)", flush=True)
        return worker_protocol.pack({'rows': rows, 'dim': dim, 'bytes': nbytes, 'error': None})
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'rows': 0, 'dim': 0, 'bytes': 0, 'error': str(e)})


def handle_imdb_status(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        resp = imdb_index.status()
        resp['error'] = None
        return worker_protocol.pack(resp)
    except Exception as e:  # hook-allow: broad — report error to caller/job
        traceback.print_exc()
        return worker_protocol.pack({'clip': {}, 'face': {}, 'error': str(e)})


# Map of request path -> handler, used both by run() and the loopback test.
HANDLERS = {
    worker_protocol.PATH_PING: handle_ping,
    worker_protocol.PATH_PATTERN: handle_pattern,
    worker_protocol.PATH_PATTERN_DINO: handle_pattern_dino,
    worker_protocol.PATH_REGION: handle_region,
    worker_protocol.PATH_DETECT_FACES: handle_detect_faces,
    worker_protocol.PATH_DETECT_FACES_PATHS: handle_detect_faces_paths,
    worker_protocol.PATH_EMBED_BBOX: handle_embed_bbox,
    worker_protocol.PATH_NORMALIZE_FACES: handle_normalize_faces,
    worker_protocol.PATH_EMBED_IMAGE: handle_embed_image,
    worker_protocol.PATH_EMBED_TEXT: handle_embed_text,
    worker_protocol.PATH_EMBED_BOTH_BATCH: handle_embed_both_batch,
    worker_protocol.PATH_EMBED_BOTH_PATHS: handle_embed_both_paths,
    worker_protocol.PATH_PROBE_DATA_ROOT: handle_probe_data_root,
    worker_protocol.PATH_EMBED_MASKED_PATHS: handle_embed_masked_paths,
    worker_protocol.PATH_EMBED_MASKED_BATCH: handle_embed_masked_batch,
    worker_protocol.PATH_EMBED_SIGLIP_TEXT: handle_embed_siglip_text,
    worker_protocol.PATH_DETECT_OBJECTS: handle_detect_objects,
    worker_protocol.PATH_ESTIMATE_AGE: handle_estimate_age,
    worker_protocol.PATH_TRAIN_CREATE: handle_train_create,
    worker_protocol.PATH_TRAIN_ADD: handle_train_add,
    worker_protocol.PATH_TRAIN_RUN: handle_train_run,
    worker_protocol.PATH_TRAIN_STATUS: handle_train_status,
    worker_protocol.PATH_TRAIN_CANCEL: handle_train_cancel,
    worker_protocol.PATH_TAG_DETECT: handle_tag_detect,
    worker_protocol.PATH_IMDB_BUILD_BEGIN: handle_imdb_build_begin,
    worker_protocol.PATH_IMDB_BUILD_CHUNK: handle_imdb_build_chunk,
    worker_protocol.PATH_IMDB_BUILD_END: handle_imdb_build_end,
    worker_protocol.PATH_IMDB_STATUS: handle_imdb_status,
}


def build_app():
    """Build the worker's FastAPI app: one ``POST /<op>`` route per protocol path.

    Each route reads the raw msgpack request body and calls the matching entry in
    :data:`HANDLERS` — the exact same ``(path, data, request_id, link_id,
    remote_identity, requested_at) -> packed_bytes`` functions the RNS transport
    used (only ``data`` is read; the rest are passed as ``None``). The handler
    already catches its own errors and returns a packed dict with an ``error``
    field, so every route returns HTTP 200 with a packed body — transport failures
    (connection/timeout) are the client's job to distinguish, not our status code.
    Blocking model work runs in a threadpool so the event loop stays free and the
    process-wide ``models.lock`` still serializes actual inference."""
    # hook-allow: lazy — web stack only needed when serving
    from fastapi import FastAPI, Request, Response
    from starlette.concurrency import run_in_threadpool  # hook-allow: lazy

    app = FastAPI(title="media worker")

    @app.post("/{op}")
    async def dispatch(op: str, request: Request):
        handler = HANDLERS.get(op)
        if handler is None:
            return Response(status_code=404, content=b"unknown op")
        body = await request.body()
        packed = await run_in_threadpool(handler, op, body, None, None, None, None)
        return Response(content=packed, media_type="application/msgpack")

    return app


def run(host="127.0.0.1", port=4243, preload=False):
    """Start the media worker as a local HTTP service (uvicorn), exposing one
    ``POST`` route per protocol path. Serves until Ctrl-C / SIGTERM."""
    # Treat SIGTERM like Ctrl-C during the pre-serve window (uvicorn installs its
    # own signal handlers once it's running) so a parent that spawned us
    # (`media web --with-worker`) can always shut us down cleanly.
    def _sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)

    print(f"[worker] compute: {compute.describe()}", flush=True)

    if preload:
        # Preload in the BACKGROUND so the HTTP server binds immediately (ping
        # answers right away, the web badge goes green) and — crucially — a model
        # that fails to load logs loudly but does NOT take the whole worker down.
        # (A blocking preload that crashed on one model is exactly what made the
        # old worker die before it could serve anything.)
        def _preload():
            # SigLIP 2 + DINOv2 are the similarity/search embedders (every embed request goes
            # through them), so they load alongside the faces; legacy CLIP (body/tile/tag
            # classifier paths) stays lazy: its first request loads it on demand.
            for name, getter in (("FaceDetector", models.get_face_detector),
                                  ("SigLIP2Embedder", models.get_siglip_embedder),
                                  ("DINOv2Embedder", models.get_dino_embedder),
                                  ("YOLOWorldDetector", models.get_object_detector)):
                try:
                    print(f"[worker] preloading {name}...", flush=True)
                    with models.lock:
                        getter()
                except Exception:  # hook-allow: broad — a failed preload must not kill the worker
                    traceback.print_exc()
                    print(f"[worker] preload of {name} FAILED; continuing without it "
                          f"(its requests will error until it loads).", flush=True)
            print("[worker] preload finished.", flush=True)
        threading.Thread(target=_preload, name="preload", daemon=True).start()

    app = build_app()
    base = f"http://{host}:{port}"
    print("", flush=True)
    print("=" * 64, flush=True)
    print("  media worker is online (HTTP)", flush=True)
    print(f"  url : {base}", flush=True)
    print("", flush=True)
    print("  On the host (only needed cross-machine — `media web --with-worker`", flush=True)
    print("  wires this up automatically), run:", flush=True)
    print(f"    media worker-connect {base}", flush=True)
    print("=" * 64, flush=True)
    print("", flush=True)

    import uvicorn  # hook-allow: lazy — only needed when serving
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    print("[worker] shutting down.", flush=True)
