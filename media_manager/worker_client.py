"""Reticulum media-worker offload — CLIENT side.

Offloads the heavy ML (face detection/embedding, CLIP image/text embedding,
YOLO-World object detection) to a remote *worker* reached over Reticulum, so a
low-powered machine can index a library without carrying torch/insightface/YOLO
itself. The wire contract lives in :mod:`worker_protocol`; the connection config
(worker address + enabled flag) in :mod:`worker_config`.

This module provides:

* :class:`WorkerClient` — a per-``data_root`` singleton that owns the RNS
  instance + link and serializes requests to the worker.
* ``Remote*`` proxy classes that are DROP-IN replacements for
  ``FaceDetector`` / ``CLIPIndexer`` / ``YOLOWorldDetector`` — same method
  signatures, same return *types* (numpy float32 arrays of the same shape, same
  tuple/dict shapes) so the DB-writing code downstream is unchanged. On a
  per-item request failure they FALL BACK to the real local model for that item.

Per this project's "no silent failures" rule, every fallback logs a warning
explaining *why* it fell back; nothing is swallowed quietly.
"""
import io
import os
import sys
import time
import threading
import collections

import numpy as np
import cv2

import RNS

from . import worker_protocol
from . import worker_config


class WorkerUnavailable(Exception):
    """The worker could not be reached (no path, no identity, link/ping failed)."""


class WorkerError(Exception):
    """A request to the worker failed at transport level (timeout / failed_callback)."""


# Poll intervals / timeouts for link establishment (seconds).
_PATH_TIMEOUT = 8.0
_LINK_TIMEOUT = 8.0
_POLL_INTERVAL = 0.1
_AVAIL_TTL = 15.0


def _warn(msg: str) -> None:
    """Emit a one-line warning to stderr. We surface fallbacks loudly rather than
    swallowing them (project 'no silent failures' rule)."""
    print(f"[worker] WARNING: {msg}", file=sys.stderr, flush=True)


class WorkerClient:
    """Owns the Reticulum instance + link to the worker and dispatches requests.

    One instance per ``data_root`` (see :func:`get_client`). Thread-safe: a single
    ``threading.Lock`` guards link (re)establishment AND request dispatch, so there
    is at most one outstanding request at a time. That is the simplest correct v1;
    throughput can be improved later with per-request concurrency if needed.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.reticulum = None
        self._link = None
        self._lock = threading.Lock()

        # Availability cache: (bool_result, expiry_ts). TTL-bounded so is_available()
        # is cheap to call repeatedly without hammering the network.
        self._avail = None
        self._avail_expiry = 0.0
        self._connected = False

        # Activity ring buffer for the web status endpoint. Monotonic id counter.
        self._activity = collections.deque(maxlen=100)
        self._activity_counter = 0

    # -- RNS lifecycle ------------------------------------------------------

    def _ensure_rns(self):
        """Lazily obtain the process's Reticulum instance exactly once.

        A process may only init RNS once — ``RNS.Reticulum(None)`` raises OSError
        if an instance is already running in this process. So reuse a running
        instance via ``get_instance()`` (returns None when none is running) and
        only construct one when there isn't one yet.
        """
        with self._lock:
            if self.reticulum is None:
                self.reticulum = RNS.Reticulum.get_instance() or RNS.Reticulum(None)
        return self.reticulum

    def _ensure_link(self, address: str):
        """Return an ACTIVE link to the worker, (re)establishing it if needed.

        Must be called with an initialised RNS. Guarded by ``self._lock`` — callers
        that already hold the lock (like :meth:`request`) invoke ``_ensure_link_locked``.
        """
        with self._lock:
            return self._ensure_link_locked(address)

    def _ensure_link_locked(self, address: str):
        """Link establishment core; assumes ``self._lock`` is already held."""
        if self._link is not None and self._link.status == RNS.Link.ACTIVE:
            return self._link

        dest_hash = worker_config.address_hash_bytes(address)

        # Ensure we know a network path to the destination.
        if not RNS.Transport.has_path(dest_hash):
            RNS.Transport.request_path(dest_hash)
            deadline = time.time() + _PATH_TIMEOUT
            while not RNS.Transport.has_path(dest_hash):
                if time.time() > deadline:
                    raise WorkerUnavailable(
                        f"no path to worker {address[:8]} after {_PATH_TIMEOUT}s")
                time.sleep(_POLL_INTERVAL)

        server_identity = RNS.Identity.recall(dest_hash)
        if server_identity is None:
            raise WorkerUnavailable(
                f"could not recall identity for worker {address[:8]}")

        dest = RNS.Destination(
            server_identity,
            RNS.Destination.OUT,
            RNS.Destination.SINGLE,
            worker_protocol.APP_NAME,
            worker_protocol.ASPECT,
        )

        link = RNS.Link(dest)
        deadline = time.time() + _LINK_TIMEOUT
        while link.status != RNS.Link.ACTIVE:
            if time.time() > deadline:
                raise WorkerUnavailable(
                    f"link to worker {address[:8]} not active after {_LINK_TIMEOUT}s "
                    f"(status={link.status})")
            time.sleep(_POLL_INTERVAL)

        self._link = link
        return link

    # -- Request dispatch ---------------------------------------------------

    def request(self, path: str, req_dict: dict, timeout: float = 120) -> dict:
        """Send a request to the worker and return the unpacked response dict.

        All requests are serialized with ``self._lock`` — at most one outstanding
        request at a time (simplest correct v1). Raises :class:`WorkerUnavailable`
        if the link cannot be established, :class:`WorkerError` on transport failure
        or timeout. A response dict whose ``"error"`` field is set is a *per-item*
        processing error and is returned as-is (the proxy decides what to do).
        """
        address = self.address()
        if not address:
            raise WorkerUnavailable("no worker address configured")

        self._ensure_rns()

        with self._lock:
            link = self._ensure_link_locked(address)

            done = threading.Event()
            holder = {"response": None, "failed": False}

            def cb(request_receipt):
                holder["response"] = request_receipt.response
                done.set()

            def fcb(request_receipt):
                holder["failed"] = True
                done.set()

            link.request(
                path,
                data=worker_protocol.pack(req_dict),
                response_callback=cb,
                failed_callback=fcb,
                timeout=timeout,
            )

            # Wait slightly longer than the request timeout so RNS's own timeout
            # (which fires failed_callback) wins the race and gives a clear reason.
            if not done.wait(timeout + 5):
                raise WorkerError(f"worker request {path!r} timed out after {timeout}s")

            if holder["failed"]:
                raise WorkerError(f"worker request {path!r} failed (failed_callback)")

            response_bytes = holder["response"]
            if response_bytes is None:
                raise WorkerError(f"worker request {path!r} returned no response")

        return worker_protocol.unpack(response_bytes)

    # -- Availability -------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the worker is enabled, configured, and answers a ping.

        TTL-cached (~15s). Any exception is treated as unavailable but is surfaced
        as a one-line warning (never swallowed silently)."""
        now = time.time()
        if self._avail is not None and now < self._avail_expiry:
            return self._avail

        cfg = worker_config.load(self.data_root)
        if not cfg.get("enabled") or not cfg.get("address"):
            result = False
        else:
            try:
                self._ensure_rns()
                self._ensure_link(cfg["address"])
                resp = self.request(worker_protocol.PATH_PING, {}, timeout=8)
                result = resp.get("ok") is True
            except Exception as exc:
                _warn(f"availability check failed: {exc}")
                result = False

        self._avail = result
        self._avail_expiry = time.time() + _AVAIL_TTL
        self._connected = result
        return result

    # -- Activity / introspection ------------------------------------------

    def record(self, op: str, name: str) -> None:
        """Record an outsourced dispatch and emit the REQUIRED 'outsourced' log line."""
        self._activity_counter += 1
        self._activity.append({
            "id": self._activity_counter,
            "op": op,
            "name": name,
            "ts": time.time(),
        })
        addr = self.address()
        addr8 = addr[:8] if addr else "?"
        print(f"[worker] outsourced {op} ({name}) -> {addr8}", flush=True)

    def recent(self, since_id: int = 0) -> list:
        """Return activity entries with id > since_id (for the web status endpoint)."""
        return [e for e in self._activity if e["id"] > since_id]

    def address(self):
        """Return the configured worker address string, or None."""
        return worker_config.load(self.data_root).get("address")

    def connected(self) -> bool:
        """Return the result of the last availability check."""
        return self._connected


# Process-global singleton registry keyed by data_root.
_clients = {}
_clients_lock = threading.Lock()


def get_client(data_root: str) -> WorkerClient:
    """Return the process-global :class:`WorkerClient` for ``data_root``."""
    with _clients_lock:
        client = _clients.get(data_root)
        if client is None:
            client = WorkerClient(data_root)
            _clients[data_root] = client
        return client


# ---------------------------------------------------------------------------
# Drop-in proxies. Each holds a WorkerClient and a lazily-built local fallback of
# the REAL model (built on first fallback use, then cached). On a per-item request
# failure the proxy runs THAT item on the local model and logs why.
# ---------------------------------------------------------------------------


class RemoteFaceDetector:
    """Drop-in for :class:`face_detector.FaceDetector` backed by the worker."""

    def __init__(self, client: WorkerClient):
        self.client = client
        self._local = None

    def _local_model(self):
        if self._local is None:
            from .face_detector import FaceDetector
            self._local = FaceDetector()
        return self._local

    def detect_faces(self, paths: list) -> list:
        results = []
        for path in paths:
            basename = os.path.basename(path)
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
                self.client.record("detect_faces", basename)
                resp = self.client.request(
                    worker_protocol.PATH_DETECT_FACES,
                    {"name": basename, "image": image_bytes},
                )
                faces = []
                for face in (resp.get("faces") or []):
                    faces.append({
                        "bbox": [float(v) for v in face["bbox"]],
                        "embedding": np.frombuffer(face["embedding"], dtype=np.float32),
                        "det_score": float(face["det_score"]),
                    })
                results.append((path, faces, resp.get("error")))
            except (WorkerUnavailable, WorkerError) as exc:
                _warn(f"worker request failed: {exc}, running detect_faces locally for {basename}")
                results.extend(self._local_model().detect_faces([path]))
        return results

    def embed_bbox(self, img, bbox: list, pad_ratio: float = 0.3) -> dict:
        try:
            image_bytes = cv2.imencode('.png', img)[1].tobytes()
            self.client.record("embed_bbox", "<crop>")
            resp = self.client.request(
                worker_protocol.PATH_EMBED_BBOX,
                {"name": "<crop>", "image": image_bytes,
                 "bbox": [float(v) for v in bbox], "pad_ratio": float(pad_ratio)},
            )
            return {
                "bbox": resp["bbox"],
                "embedding": np.frombuffer(resp["embedding"], dtype=np.float32),
                "det_score": float(resp["det_score"]),
            }
        except (WorkerUnavailable, WorkerError) as exc:
            _warn(f"worker request failed: {exc}, running embed_bbox locally")
            return self._local_model().embed_bbox(img, bbox, pad_ratio)

    @staticmethod
    def model_id(*args, **kwargs) -> str:
        from .face_detector import FaceDetector
        return FaceDetector.model_id(*args, **kwargs)


class RemoteCLIPIndexer:
    """Drop-in for :class:`indexer.CLIPIndexer` backed by the worker."""

    def __init__(self, client: WorkerClient):
        self.client = client
        self._local = None
        # Must equal the model the worker uses so the DB 'model' column matches.
        from .indexer import CLIPIndexer
        self.model_name = CLIPIndexer.model_id()

    def _local_model(self):
        if self._local is None:
            from .indexer import CLIPIndexer
            self._local = CLIPIndexer()
        return self._local

    def embed_images(self, paths):
        vectors = []
        failed = []
        for path in paths:
            basename = os.path.basename(path)
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
                self.client.record("embed_image", basename)
                resp = self.client.request(
                    worker_protocol.PATH_EMBED_IMAGE,
                    {"name": basename, "image": image_bytes},
                )
                emb = resp.get("embedding")
                if emb is None or resp.get("error"):
                    failed.append((path, resp.get("error") or "no embedding returned"))
                else:
                    vectors.append(np.frombuffer(emb, dtype=np.float32))
            except (WorkerUnavailable, WorkerError) as exc:
                _warn(f"worker request failed: {exc}, running embed_image locally for {basename}")
                emb_arr, local_failed = self._local_model().embed_images([path])
                failed.extend(local_failed)
                # embed_images returns (N_success, D); merge each success row.
                if emb_arr.ndim == 2 and emb_arr.shape[0] > 0:
                    for row in emb_arr:
                        vectors.append(row.astype(np.float32))

        if vectors:
            embeddings = np.stack(vectors).astype(np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)
        return embeddings, failed

    def embed_pil_images(self, images):
        if not images:
            return np.empty((0,), dtype=np.float32)
        vectors = []
        for img in images:
            try:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                self.client.record("embed_image", "<crop>")
                resp = self.client.request(
                    worker_protocol.PATH_EMBED_IMAGE,
                    {"name": "<crop>", "image": image_bytes},
                )
                emb = resp.get("embedding")
                if emb is None or resp.get("error"):
                    raise WorkerError(resp.get("error") or "no embedding returned")
                vectors.append(np.frombuffer(emb, dtype=np.float32))
            except (WorkerUnavailable, WorkerError) as exc:
                _warn(f"worker request failed: {exc}, running embed_pil_images locally for one crop")
                local = self._local_model().embed_pil_images([img])
                for row in local:
                    vectors.append(row.astype(np.float32))
        return np.stack(vectors).astype(np.float32)

    def embed_text(self, text):
        try:
            self.client.record("embed_text", text[:40])
            resp = self.client.request(
                worker_protocol.PATH_EMBED_TEXT,
                {"text": text},
            )
            if resp.get("error"):
                raise WorkerError(resp["error"])
            return np.frombuffer(resp["embedding"], dtype=np.float32)
        except (WorkerUnavailable, WorkerError) as exc:
            _warn(f"worker request failed: {exc}, running embed_text locally")
            return self._local_model().embed_text(text)

    @staticmethod
    def model_id(*args, **kwargs):
        from .indexer import CLIPIndexer
        return CLIPIndexer.model_id(*args, **kwargs)


class RemoteYOLODetector:
    """Drop-in for :class:`detector.YOLOWorldDetector` backed by the worker."""

    def __init__(self, client: WorkerClient, model_size='s', conf_threshold=0.15, vocab=None):
        self.client = client
        self._model_size = model_size
        self._conf_threshold = conf_threshold
        self._vocab = list(vocab) if vocab is not None else None
        self._local = None

    def _local_model(self):
        if self._local is None:
            from .detector import YOLOWorldDetector
            self._local = YOLOWorldDetector(
                model_size=self._model_size,
                conf_threshold=self._conf_threshold,
                vocab=self._vocab,
            )
        return self._local

    def set_vocab(self, vocab):
        self._vocab = list(vocab)
        if self._local is not None:
            self._local.set_vocab(self._vocab)

    def set_classes(self, vocab):
        self.set_vocab(vocab)

    def detect_images(self, paths):
        results_out = []
        for path in paths:
            basename = os.path.basename(path)
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
                self.client.record("detect_objects", basename)
                resp = self.client.request(
                    worker_protocol.PATH_DETECT_OBJECTS,
                    {"name": basename, "image": image_bytes, "vocab": self._vocab},
                )
                detections = []
                for d in (resp.get("detections") or []):
                    class_name = d[0]
                    conf = float(d[1])
                    x1, y1, x2, y2 = (float(d[2]), float(d[3]), float(d[4]), float(d[5]))
                    detections.append((class_name, conf, x1, y1, x2, y2))
                results_out.append((path, detections, resp.get("error")))
            except (WorkerUnavailable, WorkerError) as exc:
                _warn(f"worker request failed: {exc}, running detect_objects locally for {basename}")
                local = self._local_model()
                if self._vocab is not None:
                    local.set_vocab(self._vocab)
                results_out.extend(local.detect_images([path]))
        return results_out

    @staticmethod
    def model_id(*args, **kwargs):
        from .detector import YOLOWorldDetector
        return YOLOWorldDetector.model_id(*args, **kwargs)
