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
import time
import threading
import tempfile
import traceback

import numpy as np

import RNS

from . import worker_protocol


DEFAULT_IDENTITY_FILE = os.path.expanduser('~/.config/media_manager/worker_identity')
DEFAULT_CONF_THRESHOLD = 0.15


class WorkerModels:
    """Lazy, cached singletons for the three heavy model classes.

    A single lock serializes all model inference so the worker never runs
    concurrent inference (OOM risk) and the object detector's vocabulary can't
    be swapped mid-request by a concurrent caller.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._face_detector = None
        self._clip_indexer = None
        self._object_detector = None

    def get_face_detector(self):
        if self._face_detector is None:
            print("[worker] loading FaceDetector...", flush=True)
            from .face_detector import FaceDetector
            self._face_detector = FaceDetector()
        return self._face_detector

    def get_clip_indexer(self):
        if self._clip_indexer is None:
            print("[worker] loading CLIPIndexer...", flush=True)
            from .indexer import CLIPIndexer
            self._clip_indexer = CLIPIndexer()
        return self._clip_indexer

    def get_object_detector(self):
        if self._object_detector is None:
            print("[worker] loading YOLOWorldDetector...", flush=True)
            from .detector import YOLOWorldDetector
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
        return ids


# Module-level singleton holder shared by all request handlers.
models = WorkerModels()


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
    except Exception:
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
# Request handlers. RNS calls these with the 6-parameter response_generator
# signature confirmed against the installed RNS 1.4.2:
#     response_generator(path, data, request_id, link_id, remote_identity,
#                        requested_at) -> bytes
# `data` is the raw bytes the client packed with worker_protocol.pack().
# Each returns worker_protocol.pack(resp_dict) (bytes).
# ---------------------------------------------------------------------------

def handle_ping(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        resp = {"ok": True, "models": models.loaded_model_ids(), "error": None}
        print(f"[worker] handled {worker_protocol.PATH_PING} () -> ok "
              f"models={resp['models']}", flush=True)
        return worker_protocol.pack(resp)
    except Exception as e:
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_PING} () -> ERROR {e}", flush=True)
        return worker_protocol.pack({"ok": False, "models": [], "error": str(e)})


def handle_detect_faces(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        tmp = _write_temp(req["image"], name)
        with models.lock:
            res = models.get_face_detector().detect_faces([tmp])
        _, faces, error = res[0]
        out_faces = []
        for face in faces:
            out_faces.append({
                "bbox": [float(v) for v in face["bbox"]],
                "embedding": face["embedding"].astype(np.float32).tobytes(),
                "det_score": float(face["det_score"]),
            })
        summary = f"{len(out_faces)} faces" if error is None else f"err={error}"
        print(f"[worker] handled {worker_protocol.PATH_DETECT_FACES} ({name}) -> {summary}",
              flush=True)
        return worker_protocol.pack({"faces": out_faces, "error": error})
    except Exception as e:
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
        import cv2
        req = worker_protocol.unpack(data)
        name = req.get("name")
        img = cv2.imdecode(np.frombuffer(req["image"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None (undecodable image bytes)")
        bbox = req["bbox"]
        pad_ratio = req.get("pad_ratio", 0.3)
        with models.lock:
            d = models.get_face_detector().embed_bbox(img, bbox, pad_ratio)
        resp = {
            "bbox": d["bbox"],
            "embedding": d["embedding"].astype(np.float32).tobytes(),
            "det_score": float(d["det_score"]),
            "error": None,
        }
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BBOX} ({name}) -> "
              f"det_score={resp['det_score']:.3f}", flush=True)
        return worker_protocol.pack(resp)
    except Exception as e:
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_BBOX} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({
            "bbox": None, "embedding": None, "det_score": 0.0, "error": str(e)})


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
    except Exception as e:
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
    except Exception as e:
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_EMBED_TEXT} ({text!r}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"embedding": None, "error": str(e)})


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
        from .detector import DEFAULT_VOCAB
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
    except Exception as e:
        traceback.print_exc()
        print(f"[worker] handled {worker_protocol.PATH_DETECT_OBJECTS} ({name}) -> ERROR {e}",
              flush=True)
        return worker_protocol.pack({"detections": [], "error": str(e)})
    finally:
        if tmp is not None:
            _cleanup(tmp)


# Map of request path -> handler, used both by run() and the loopback test.
HANDLERS = {
    worker_protocol.PATH_PING: handle_ping,
    worker_protocol.PATH_DETECT_FACES: handle_detect_faces,
    worker_protocol.PATH_EMBED_BBOX: handle_embed_bbox,
    worker_protocol.PATH_EMBED_IMAGE: handle_embed_image,
    worker_protocol.PATH_EMBED_TEXT: handle_embed_text,
    worker_protocol.PATH_DETECT_OBJECTS: handle_detect_objects,
}


def build_destination(identity):
    """Build the IN/SINGLE worker destination with all request handlers
    registered. Shared by run() and the loopback smoke test."""
    dest = RNS.Destination(
        identity,
        RNS.Destination.IN,
        RNS.Destination.SINGLE,
        worker_protocol.APP_NAME,
        worker_protocol.ASPECT,
    )
    # Auto-prove inbound links so clients can establish them without an
    # app-level proof callback.
    dest.set_proof_strategy(RNS.Destination.PROVE_ALL)
    for path, handler in HANDLERS.items():
        dest.register_request_handler(
            path, response_generator=handler, allow=RNS.Destination.ALLOW_ALL)
    return dest


def _load_or_create_identity(identity_file):
    path = identity_file or DEFAULT_IDENTITY_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"[worker] loading identity from {path}", flush=True)
        identity = RNS.Identity.from_file(path)
        if identity is None:
            raise ValueError(f"Failed to load RNS identity from {path}")
        return identity
    print(f"[worker] creating new identity at {path}", flush=True)
    identity = RNS.Identity()
    identity.to_file(path)
    return identity


def run(identity_file=None, config_dir=None, announce_interval=300, preload=False):
    """Start the media worker server: init RNS, register handlers, announce,
    and loop announcing every ``announce_interval`` seconds until Ctrl-C."""
    RNS.Reticulum(config_dir)

    identity = _load_or_create_identity(identity_file)
    dest = build_destination(identity)

    print("", flush=True)
    print("=" * 64, flush=True)
    print("  media worker is online", flush=True)
    print(f"  destination : {RNS.prettyhexrep(dest.hash)}", flush=True)
    print(f"  address     : {dest.hash.hex()}", flush=True)
    print("", flush=True)
    print("  On the host, run:", flush=True)
    print(f"    media worker-connect {dest.hash.hex()}", flush=True)
    print("=" * 64, flush=True)
    print("", flush=True)

    if preload:
        print("[worker] preloading all models...", flush=True)
        with models.lock:
            models.get_face_detector()
            models.get_clip_indexer()
            models.get_object_detector()
        print("[worker] preload complete.", flush=True)

    dest.announce()
    print(f"[worker] announced; re-announcing every {announce_interval}s. Ctrl-C to stop.",
          flush=True)

    try:
        while True:
            # Sleep in small increments so Ctrl-C is responsive.
            slept = 0.0
            while slept < announce_interval:
                time.sleep(1.0)
                slept += 1.0
            dest.announce()
            print("[worker] re-announced.", flush=True)
    except KeyboardInterrupt:
        print("\n[worker] shutting down.", flush=True)
        return
