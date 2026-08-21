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
# The availability probe (is_available) uses much shorter timeouts than a real
# request: it is a liveness check, often driven by a UI poll, and must degrade
# fast when the worker is configured-but-down instead of stalling a web request.
_PROBE_TIMEOUT = 3.0
_PROBE_PING_TIMEOUT = 4.0
# Real requests retry a couple of times (re-establishing the link) to ride out a
# transient worker blip WITHOUT falling back to a local model — the whole point of
# the offload is to keep heavy models out of a memory-constrained host. If the
# worker is genuinely gone the error is raised so the caller surfaces it.
_MAX_RETRIES = 1
_RETRY_BACKOFF = 1.5


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

        # Config cache (address/enabled). worker.json is read through here so a
        # per-item call like record() doesn't re-open+parse the file every image.
        self._cfg_cache = None
        self._cfg_expiry = 0.0

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

    def _ensure_link(self, address, path_timeout=_PATH_TIMEOUT, link_timeout=_LINK_TIMEOUT):
        """Return an ACTIVE link to the worker, (re)establishing it if needed.

        Must be called with an initialised RNS. Guarded by ``self._lock`` — callers
        that already hold the lock (like :meth:`request`) invoke ``_ensure_link_locked``.
        """
        with self._lock:
            return self._ensure_link_locked(address, path_timeout, link_timeout)

    def _ensure_link_locked(self, address, path_timeout=_PATH_TIMEOUT, link_timeout=_LINK_TIMEOUT):
        """Link establishment core; assumes ``self._lock`` is already held."""
        if self._link is not None and self._link.status == RNS.Link.ACTIVE:
            return self._link

        dest_hash = worker_config.address_hash_bytes(address)

        # Ensure we know a network path to the destination.
        if not RNS.Transport.has_path(dest_hash):
            RNS.Transport.request_path(dest_hash)
            deadline = time.time() + path_timeout
            while not RNS.Transport.has_path(dest_hash):
                if time.time() > deadline:
                    raise WorkerUnavailable(
                        f"no path to worker {address[:8]} after {path_timeout}s")
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
        deadline = time.time() + link_timeout
        while link.status != RNS.Link.ACTIVE:
            if time.time() > deadline:
                raise WorkerUnavailable(
                    f"link to worker {address[:8]} not active after {link_timeout}s "
                    f"(status={link.status})")
            time.sleep(_POLL_INTERVAL)

        self._link = link
        return link

    def _load_cfg(self):
        """Return the worker config (address/enabled), cached for ~_AVAIL_TTL so a
        per-item caller (record/address) doesn't re-read+parse worker.json each time."""
        now = time.time()
        if self._cfg_cache is not None and now < self._cfg_expiry:
            return self._cfg_cache
        cfg = worker_config.load(self.data_root)
        self._cfg_cache = cfg
        self._cfg_expiry = now + _AVAIL_TTL
        return cfg

    # -- Request dispatch ---------------------------------------------------

    def request(self, path: str, req_dict: dict, timeout: float = 120,
                retries: int = _MAX_RETRIES) -> dict:
        """Send a request to the worker and return the unpacked response dict.

        Retries up to ``retries`` times, dropping the (possibly dead) link between
        attempts, to ride out a transient worker blip. Raises :class:`WorkerUnavailable`
        / :class:`WorkerError` if it still can't reach the worker — callers surface that
        rather than loading a local model. A response dict whose ``"error"`` field is
        set is a *per-item* processing error (the worker ran but the model failed on
        that item) and is returned as-is; that is NOT retried.
        """
        address = self.address()
        if not address:
            raise WorkerUnavailable("no worker address configured")

        self._ensure_rns()

        last_exc = None
        for attempt in range(retries + 1):
            try:
                return self._request_once(path, req_dict, timeout)
            except (WorkerUnavailable, WorkerError) as exc:
                last_exc = exc
                with self._lock:
                    self._link = None  # force a fresh link on the next attempt
                if attempt < retries:
                    _warn(f"worker request {path!r} failed ({exc}); "
                          f"retry {attempt + 1}/{retries}")
                    time.sleep(_RETRY_BACKOFF * (attempt + 1))
        raise last_exc

    def _request_once(self, path: str, req_dict: dict, timeout: float) -> dict:
        address = self.address()
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

        cfg = self._load_cfg()
        if not cfg.get("enabled") or not cfg.get("address"):
            result = False
        elif not self._lock.acquire(blocking=False):
            # A request is in flight (the lock is held) — the worker is actively
            # reachable. Don't block this caller (often a UI status poll) behind a
            # long ML request; serve the last-known value and refresh next time.
            return self._avail if self._avail is not None else True
        else:
            self._lock.release()  # was only probing for contention
            try:
                self._ensure_rns()
                # Short timeouts: a liveness probe must fail fast when the worker
                # is down rather than stalling the caller for the full request budget.
                self._ensure_link(cfg["address"],
                                  path_timeout=_PROBE_TIMEOUT, link_timeout=_PROBE_TIMEOUT)
                resp = self.request(worker_protocol.PATH_PING, {},
                                    timeout=_PROBE_PING_TIMEOUT, retries=0)
                result = resp.get("ok") is True
            except Exception as exc:
                _warn(f"availability check failed: {exc}")
                result = False

        self._avail = result
        self._avail_expiry = time.time() + _AVAIL_TTL
        self._connected = result
        return result

    def is_configured(self) -> bool:
        """True if a worker is enabled and has an address — CHEAP (no network probe).

        Model getters use this (not :meth:`is_available`) to decide proxy-vs-local: when
        a worker is configured the host must ALWAYS use the proxy and NEVER build a local
        model, even if the worker is momentarily down (the proxy retries and then raises,
        which the caller surfaces). Loading a local model here is exactly what OOMs a
        memory-constrained web process, so we don't."""
        cfg = self._load_cfg()
        return bool(cfg.get("enabled") and cfg.get("address"))

    # -- Per-tag training + fine-tuned inference ---------------------------
    # These drive the worker's job/poll training protocol. Unlike the inference
    # proxies they are called from the host-side driver subprocess
    # (remote_tag_trainer.py) and, for tag_detect, from web.py's suggestion path.

    def train_create(self, kind: str, tag: str, slug: str, epochs=None) -> str:
        resp = self.request(
            worker_protocol.PATH_TRAIN_CREATE,
            {"kind": kind, "tag": tag, "slug": slug, "epochs": epochs},
            timeout=30)
        if resp.get("error") or not resp.get("job_id"):
            raise WorkerError(resp.get("error") or "train_create returned no job_id")
        return resp["job_id"]

    def train_add(self, job_id: str, kind: str, batch: list) -> int:
        # A batch of (downscaled) images can be a few MB — RNS ships it as a
        # segmented Resource, so allow a generous upload window.
        resp = self.request(
            worker_protocol.PATH_TRAIN_ADD,
            {"job_id": job_id, "kind": kind, "batch": batch},
            timeout=600)
        if resp.get("error"):
            raise WorkerError(resp["error"])
        return resp.get("received", 0)

    def train_run(self, job_id: str) -> None:
        resp = self.request(worker_protocol.PATH_TRAIN_RUN, {"job_id": job_id}, timeout=30)
        if resp.get("error") or not resp.get("ok"):
            raise WorkerError(resp.get("error") or "train_run failed")

    def train_status(self, job_id: str, want_artifact: bool = False) -> dict:
        # retries=0: this is polled in a loop, so a transient blip is handled by
        # the next poll rather than by an in-call retry that would double the wait.
        return self.request(
            worker_protocol.PATH_TRAIN_STATUS,
            {"job_id": job_id, "want_artifact": want_artifact},
            timeout=60, retries=0)

    def train_cancel(self, job_id: str) -> None:
        resp = self.request(worker_protocol.PATH_TRAIN_CANCEL, {"job_id": job_id}, timeout=30)
        if resp.get("error"):
            raise WorkerError(resp["error"])

    def tag_detect(self, slug: str, image_bytes: bytes, name: str = "<image>", conf=None) -> list:
        """Run a tag's fine-tuned YOLO checkpoint (kept on the worker) on one
        image. Returns [(class_name, conf, x1, y1, x2, y2), ...] — same shape as
        RemoteYOLODetector.detect_images per-image. Raises WorkerError if the
        worker has no trained model for this slug (host surfaces "retrain")."""
        self.record("tag_detect", name)
        resp = self.request(
            worker_protocol.PATH_TAG_DETECT,
            {"slug": slug, "kind": "yolo_model", "name": name,
             "image": image_bytes, "conf": conf})
        if resp.get("error"):
            raise WorkerError(resp["error"])
        out = []
        for d in (resp.get("detections") or []):
            out.append((d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])))
        return out

    # -- imdb: resident search index ---------------------------------------
    # The host (remote_index_builder.py) streams a matrix to the worker in bounded
    # chunks so neither side ever holds the whole ~1 GB blob. Control messages are
    # small; each chunk is a few tens of MB (RNS ships it as a segmented Resource),
    # so allow a generous per-chunk timeout.

    def imdb_build_begin(self, kind: str, dim: int, total_rows=None) -> None:
        resp = self.request(worker_protocol.PATH_IMDB_BUILD_BEGIN,
                            {"kind": kind, "dim": dim, "total_rows": total_rows}, timeout=60)
        if resp.get("error") or not resp.get("ok"):
            raise WorkerError(resp.get("error") or "imdb_build_begin failed")

    def imdb_build_chunk(self, kind: str, ids: list, vecs: bytes) -> int:
        resp = self.request(worker_protocol.PATH_IMDB_BUILD_CHUNK,
                            {"kind": kind, "ids": ids, "vecs": vecs}, timeout=600)
        if resp.get("error"):
            raise WorkerError(resp["error"])
        return resp.get("received", 0)

    def imdb_build_end(self, kind: str) -> dict:
        resp = self.request(worker_protocol.PATH_IMDB_BUILD_END, {"kind": kind}, timeout=300)
        if resp.get("error"):
            raise WorkerError(resp["error"])
        return resp

    def imdb_status(self) -> dict:
        return self.request(worker_protocol.PATH_IMDB_STATUS, {}, timeout=30, retries=0)

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
        return self._load_cfg().get("address")

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
# Drop-in proxies. Each holds a WorkerClient and forwards to the remote worker.
# They hold NO local model: when a request fails the WorkerClient has already
# retried, so the exception propagates and the caller surfaces it — the host must
# never load a heavy model to "fall back" (that reintroduces the OOM the offload
# exists to prevent). A getter returns a proxy only when a worker is configured;
# with no worker configured the getter returns the real local model instead.
# A response dict with "error" set is a per-item model failure (not a worker
# outage) and is surfaced through the normal return value / raised in-place.
# ---------------------------------------------------------------------------


class RemoteFaceDetector:
    """Drop-in for :class:`face_detector.FaceDetector` backed by the worker."""

    def __init__(self, client: WorkerClient):
        self.client = client
        # Callers (e.g. web.py detect-faces) read `detector._model_name` to compute
        # the DB model_id. The worker always runs the default buffalo_l, so mirror it.
        self._model_name = 'buffalo_l'

    def detect_faces(self, paths: list) -> list:
        results = []
        for path in paths:
            basename = os.path.basename(path)
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
                    # .copy() so the array owns its memory instead of pinning the
                    # whole response buffer for the lifetime of the embedding.
                    "embedding": np.frombuffer(face["embedding"], dtype=np.float32).copy(),
                    "det_score": float(face["det_score"]),
                })
            results.append((path, faces, resp.get("error")))
        return results

    def embed_bbox(self, img, bbox: list, pad_ratio: float = 0.3) -> dict:
        image_bytes = cv2.imencode('.png', img)[1].tobytes()
        self.client.record("embed_bbox", "<crop>")
        resp = self.client.request(
            worker_protocol.PATH_EMBED_BBOX,
            {"name": "<crop>", "image": image_bytes,
             "bbox": [float(v) for v in bbox], "pad_ratio": float(pad_ratio)},
        )
        if resp.get("error") or resp.get("embedding") is None:
            raise WorkerError(resp.get("error") or "embed_bbox returned no embedding")
        return {
            "bbox": resp["bbox"],
            "embedding": np.frombuffer(resp["embedding"], dtype=np.float32).copy(),
            "det_score": float(resp["det_score"]),
        }

    @staticmethod
    def model_id(*args, **kwargs) -> str:
        from .face_detector import FaceDetector
        return FaceDetector.model_id(*args, **kwargs)


class RemoteCLIPIndexer:
    """Drop-in for :class:`indexer.CLIPIndexer` backed by the worker."""

    def __init__(self, client: WorkerClient):
        self.client = client
        # Must equal the model the worker uses so the DB 'model' column matches.
        from .indexer import CLIPIndexer
        self.model_name = CLIPIndexer.model_id()

    def embed_images(self, paths):
        vectors = []
        failed = []
        for path in paths:
            basename = os.path.basename(path)
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
                vectors.append(np.frombuffer(emb, dtype=np.float32).copy())

        if vectors:
            embeddings = np.stack(vectors).astype(np.float32)
        else:
            embeddings = np.empty((0,), dtype=np.float32)  # match CLIPIndexer.embed_images
        return embeddings, failed

    def embed_pil_images(self, images):
        if not images:
            return np.empty((0,), dtype=np.float32)
        vectors = []
        for img in images:
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
            vectors.append(np.frombuffer(emb, dtype=np.float32).copy())
        return np.stack(vectors).astype(np.float32)

    def embed_text(self, text):
        self.client.record("embed_text", text[:40])
        resp = self.client.request(
            worker_protocol.PATH_EMBED_TEXT,
            {"text": text},
        )
        if resp.get("error"):
            raise WorkerError(resp["error"])
        return np.frombuffer(resp["embedding"], dtype=np.float32).copy()

    @staticmethod
    def model_id(*args, **kwargs):
        from .indexer import CLIPIndexer
        return CLIPIndexer.model_id(*args, **kwargs)


class RemoteAgeEstimator:
    """Drop-in for :class:`age_estimator.AgeGenderEstimator` backed by the worker.

    Same ``estimate(image_path, faces)`` signature — ships the image + face
    bboxes to the worker, which runs MiVOLO in its OWN isolated ``.age-venv``
    there. Raises :class:`RuntimeError` (matching the local estimator's error
    contract, which web.py's ``api_estimate_age`` already handles) rather than
    WorkerError, so a failure surfaces cleanly with no web.py change."""

    def __init__(self, client: WorkerClient):
        self.client = client

    def estimate(self, image_path: str, faces: list) -> list:
        if not faces:
            return []
        basename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        self.client.record("estimate_age", basename)
        # MiVOLO's per-call subprocess caps at 120s on the worker, plus the image
        # upload; give the round-trip generous headroom (first run also downloads
        # weights on the worker — that first estimate may need a retry).
        resp = self.client.request(
            worker_protocol.PATH_ESTIMATE_AGE,
            {"name": basename, "image": image_bytes,
             "faces": [{"face_ref": fc["ref"], "bbox": [float(v) for v in fc["bbox"]]}
                       for fc in faces]},
            timeout=200)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("results") or []


class RemoteYOLODetector:
    """Drop-in for :class:`detector.YOLOWorldDetector` backed by the worker."""

    def __init__(self, client: WorkerClient, model_size='s', conf_threshold=0.15, vocab=None):
        self.client = client
        self._model_size = model_size
        self._conf_threshold = conf_threshold
        self._vocab = list(vocab) if vocab is not None else None

    def set_vocab(self, vocab):
        self._vocab = list(vocab)

    def set_classes(self, vocab):
        self.set_vocab(vocab)

    def detect_images(self, paths):
        results_out = []
        for path in paths:
            basename = os.path.basename(path)
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
        return results_out

    @staticmethod
    def model_id(*args, **kwargs):
        from .detector import YOLOWorldDetector
        return YOLOWorldDetector.model_id(*args, **kwargs)
