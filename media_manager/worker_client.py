"""Media-worker offload — CLIENT side.

Offloads the heavy ML (face detection/embedding, CLIP image/text embedding,
YOLO-World object detection) to a *worker* reached over plain HTTP, so a
low-powered machine can index a library without carrying torch/insightface/YOLO
itself. The wire contract lives in :mod:`worker_protocol` (msgpack dicts sent as
``POST /<path>``); the connection config (worker base URL + enabled flag) in
:mod:`worker_config`.

This module provides:

* :class:`WorkerClient` — a per-``data_root`` singleton that owns an HTTP session
  and serializes requests to the worker.
* ``Remote*`` proxy classes that are DROP-IN replacements for
  ``FaceDetector`` / ``CLIPIndexer`` / ``YOLOWorldDetector`` — same method
  signatures, same return *types* (numpy float32 arrays of the same shape, same
  tuple/dict shapes) so the DB-writing code downstream is unchanged.

Per this project's "no silent failures" rule, every fallback logs a warning
explaining *why* it fell back; nothing is swallowed quietly.
"""
import io
import contextlib
import os
import sys
import time
import threading
import collections
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

import requests

from . import worker_protocol
from . import worker_config



def _read_for_upload(path):
    """(bytes, None), or (None, error) when the file can't be read (moved, deleted,
    unmounted drive, permissions) — so one unreadable file in a batch becomes that
    file's own error result, like the local models report it, instead of an
    exception that takes down the whole bulk job."""
    try:
        with open(path, "rb") as f:
            return f.read(), None
    except OSError as exc:
        return None, f"cannot read file: {exc.strerror or exc}"

class WorkerUnavailable(Exception):
    """The worker could not be reached (connection refused, DNS/route failure,
    or ping failed) — i.e. no HTTP service answering at the configured URL."""


class WorkerError(Exception):
    """A request reached the worker's HTTP service but failed at transport level
    (read timeout, non-200 status, or an undecodable body)."""


def _check_algo(resp: dict, algo: str) -> None:
    """Raise WorkerError on a worker-side error or when it answered with a different
    descriptor algo (an un-redeployed worker still serving the legacy classical format)."""
    if resp.get("error"):
        raise WorkerError(resp["error"])
    if resp.get("algo") != algo:
        raise WorkerError(f"worker pattern algo {resp.get('algo')!r} != expected {algo!r} "
                          "(redeploy the worker)")


# TCP connect timeout (seconds) — how long to wait to establish the HTTP
# connection before declaring the worker unreachable. Kept short so a
# configured-but-down worker fails fast rather than stalling a web request. (The
# per-request READ timeout is passed separately by each caller.)
_CONNECT_TIMEOUT = 8.0
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
        self._session = None
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

    # -- HTTP session -------------------------------------------------------

    def _ensure_session(self):
        """Lazily create the shared ``requests.Session`` (connection pooling +
        keep-alive). One outstanding request at a time is guaranteed by
        ``self._lock`` in :meth:`_request_once`, so a single session is fine."""
        with self._lock:
            if self._session is None:
                self._session = requests.Session()
        return self._session

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
                retries: int = _MAX_RETRIES, concurrent: bool = False) -> dict:
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

        self._ensure_session()

        last_exc = None
        for attempt in range(retries + 1):
            try:
                return self._request_once(path, req_dict, timeout, concurrent)
            except (WorkerUnavailable, WorkerError) as exc:
                last_exc = exc
                if attempt < retries:
                    _warn(f"worker request {path!r} failed ({exc}); "
                          f"retry {attempt + 1}/{retries}")
                    time.sleep(_RETRY_BACKOFF * (attempt + 1))
        raise last_exc

    def _request_once(self, path: str, req_dict: dict, timeout: float,
                      concurrent: bool = False) -> dict:
        base = worker_config.normalize_url(self.address())
        url = f"{base}/{path}"
        # One request at a time (matches the old single-link semantics and keeps
        # the worker's single-process model lock from being fought over).
        # concurrent=True (embed_both pipelining) skips the one-at-a-time lock; the worker
        # still serializes inference with its own models.lock, so this only overlaps
        # upload/decode of the next batch with GPU work on the current one.
        with (contextlib.nullcontext() if concurrent else self._lock):
            try:
                resp = self._session.post(
                    url,
                    data=worker_protocol.pack(req_dict),
                    headers={"Content-Type": "application/msgpack"},
                    # (connect, read): fail fast if the worker isn't listening,
                    # but allow the full per-call budget for the model to run.
                    timeout=(min(_CONNECT_TIMEOUT, timeout), timeout),
                )
            except requests.exceptions.Timeout as exc:
                raise WorkerError(f"worker request {path!r} timed out after {timeout}s") from exc
            except requests.exceptions.RequestException as exc:
                # connection refused / DNS / route → the worker isn't reachable.
                raise WorkerUnavailable(f"cannot reach worker at {base} ({exc})") from exc

        if resp.status_code == 404:
            raise WorkerError(f"worker has no handler for {path!r} (HTTP 404)")
        if resp.status_code != 200:
            raise WorkerError(
                f"worker request {path!r} failed: HTTP {resp.status_code} "
                f"{resp.text[:200]!r}")
        try:
            return worker_protocol.unpack(resp.content)
        except Exception as exc:  # hook-allow: broad — any msgpack decode failure → WorkerError
            raise WorkerError(
                f"worker request {path!r} returned an undecodable body ({exc})") from exc

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
                # Short read timeout: a liveness probe must fail fast when the
                # worker is down rather than stalling the caller for a full budget.
                # (A refused connection returns instantly; the connect timeout in
                # _request_once caps the wait when the host is unreachable.)
                self._ensure_session()
                resp = self.request(worker_protocol.PATH_PING, {},
                                    timeout=_PROBE_PING_TIMEOUT, retries=0)
                result = resp.get("ok") is True
            except Exception as exc:  # hook-allow: broad — any failure = unavailable (warned)
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
        # A batch of (downscaled) images can be a few MB — one HTTP POST body, so
        # allow a generous upload/read window.
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

    def pattern_query(self, image_bytes: bytes, box, name: str = "<query>"):
        """Descriptor for one region (find-by-pattern query). `box` is [x1,y1,x2,y2] in
        image pixels, or None for the whole image. Returns a float32 np.ndarray or None.
        Raises WorkerError on a worker-side failure."""
        self.record("pattern_query", name)
        resp = self.request(worker_protocol.PATH_PATTERN,
                            {"name": name, "image": image_bytes, "boxes": [box]})
        if resp.get("error"):
            raise WorkerError(resp["error"])
        ds = resp.get("descriptors") or []
        return np.frombuffer(ds[0], np.float32).copy() if ds and ds[0] else None

    def pattern_index(self, image_bytes: bytes, name: str = "<image>"):
        """Auto-tile an image and describe each tile (find-by-pattern indexing). Returns
        [(bbox: list[float], descriptor: np.ndarray[float32]), ...]."""
        self.record("pattern_index", name)
        resp = self.request(worker_protocol.PATH_PATTERN, {"name": name, "image": image_bytes})
        if resp.get("error"):
            raise WorkerError(resp["error"])
        return [(t["bbox"], np.frombuffer(t["descriptor"], np.float32).copy())
                for t in (resp.get("tiles") or [])]

    def pattern_deep_query(self, image_bytes: bytes, box, algo: str, name: str = "<query>"):
        """Deep-texture query vectors for one region (find-by-pattern): float32 (S, DIM), or
        None when the crop is too small. `box` is [x1,y1,x2,y2] in EXIF-upright image px.
        Raises WorkerError on failure or when the worker runs a different algo (stale worker)."""
        self.record("pattern_query", name)
        resp = self.request(worker_protocol.PATH_PATTERN,
                            {"name": name, "image": image_bytes, "deep": True, "box": list(box)})
        _check_algo(resp, algo)
        if not resp.get("n"):
            return None
        return np.frombuffer(resp["queries"], np.float32).reshape(resp["n"], -1).copy()

    def pattern_deep_index(self, image_bytes: bytes, algo: str, dim: int, name: str = "<image>"):
        """Deep-texture window descriptors for one image (find-by-pattern indexing): int8
        (n, dim); (0, dim) when nothing describable. Same failure contract as above."""
        self.record("pattern_index", name)
        resp = self.request(worker_protocol.PATH_PATTERN,
                            {"name": name, "image": image_bytes, "deep": True})
        _check_algo(resp, algo)
        return np.frombuffer(resp["windows"], np.int8).reshape(resp["n"], dim).copy()

    # DEPRECATION PLANNED: superseded by the SigLIP 2 + DINOv2 embedding pipeline; do not extend.
    def region_query(self, image_bytes: bytes, box, algo: str, name: str = "<query>"):
        """DINOv2 query vectors float32 (3, DIM) for one drawn region (instance-level region
        search). `box` is [x1,y1,x2,y2] in EXIF-upright image px. Raises WorkerError on
        failure or an algo mismatch (un-redeployed worker)."""
        self.record("region_query", name)
        resp = self.request(worker_protocol.PATH_REGION,
                            {"name": name, "image": image_bytes, "box": list(box)})
        _check_algo(resp, algo)
        return np.frombuffer(resp["vecs"], np.float32).reshape(resp["n"], -1).copy()

    # DEPRECATION PLANNED: superseded by the SigLIP 2 + DINOv2 embedding pipeline; do not extend.
    def region_index(self, image_bytes: bytes, algo: str, dim: int, name: str = "<image>"):
        """Region-index payload for one image: (windows int8 (n, dim), sift pts float16 (k, 2),
        sift desc uint8 (k, 128)). Same failure contract as region_query."""
        self.record("region_index", name)
        resp = self.request(worker_protocol.PATH_REGION, {"name": name, "image": image_bytes})
        _check_algo(resp, algo)
        k = resp["nkp"]
        return (np.frombuffer(resp["windows"], np.int8).reshape(resp["n"], dim).copy(),
                np.frombuffer(resp["pts"], np.float16).reshape(k, 2).copy(),
                np.frombuffer(resp["desc"], np.uint8).reshape(k, 128).copy())

    # -- imdb: resident search index ---------------------------------------
    # The host (remote_index_builder.py) streams a matrix to the worker in bounded
    # chunks so neither side ever holds the whole ~1 GB blob. Control messages are
    # small; each chunk is a few tens of MB (one HTTP POST body), so allow a
    # generous per-chunk timeout.

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
            image_bytes, read_error = _read_for_upload(path)
            if read_error:
                results.append((path, [], read_error))
                continue
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
                    "angle": float(face.get("angle") or 0.0),
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
            "angle": float(resp.get("angle") or 0.0),
        }

    def normalize_faces(self, img, faces):
        """Remote twin of :meth:`face_detector.FaceDetector.normalize_faces` — same
        length and order as `faces`, input dicts untouched. The image rides as
        encoded bytes exactly like embed_bbox's, since the worker needs the decoded
        array rather than a file on disk."""
        image_bytes = cv2.imencode('.png', img)[1].tobytes()
        payload = []
        for face in faces:
            emb = face.get("embedding")
            payload.append({
                "bbox": [float(v) for v in face["bbox"]],
                "embedding": (np.asarray(emb, dtype=np.float32).tobytes()
                              if emb is not None else b""),
                "det_score": float(face.get("det_score") or 0.0),
                "angle": float(face.get("angle") or 0.0),
            })
        self.client.record("normalize_faces", f"<{len(payload)} faces>")
        resp = self.client.request(
            worker_protocol.PATH_NORMALIZE_FACES,
            {"name": "<image>", "image": image_bytes, "faces": payload},
        )
        if resp.get("error"):
            raise WorkerError(resp["error"])
        out = []
        for face in (resp.get("faces") or []):
            emb = face.get("embedding")
            out.append({
                "bbox": [float(v) for v in face["bbox"]],
                "embedding": (np.frombuffer(emb, dtype=np.float32).copy()
                              if emb else None),
                "det_score": float(face["det_score"]),
                "angle": float(face.get("angle") or 0.0),
            })
        if len(out) != len(faces):
            # The caller maps results back to DB rows by position; a truncated reply
            # would silently write one face's vector onto another's row.
            raise WorkerError(
                f"normalize_faces returned {len(out)} faces for {len(faces)} sent")
        return out

    @staticmethod
    def model_id(*args, **kwargs) -> str:
        from .face_detector import FaceDetector  # hook-allow: lazy — model module, only for id
        return FaceDetector.model_id(*args, **kwargs)


class RemoteCLIPIndexer:
    """Drop-in for :class:`indexer.CLIPIndexer` backed by the worker."""

    def __init__(self, client: WorkerClient):
        self.client = client
        # Must equal the model the worker uses so the DB 'model' column matches.
        from .indexer import CLIPIndexer  # hook-allow: lazy — pulls torch/open_clip
        self.model_name = CLIPIndexer.model_id()

    def embed_images(self, paths):
        vectors = []
        failed = []
        for path in paths:
            basename = os.path.basename(path)
            image_bytes, read_error = _read_for_upload(path)
            if read_error:
                failed.append((path, read_error))
                continue
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
        from .indexer import CLIPIndexer  # hook-allow: lazy — pulls torch/open_clip
        return CLIPIndexer.model_id(*args, **kwargs)


# ---------------------------------------------------------------------------
# SigLIP 2 + DINOv2 embedding offload (embedders.py on the worker).
# ---------------------------------------------------------------------------

_EMBED_BATCH_TIMEOUT = 600


def _check_embed_proto(resp: dict, what: str) -> None:
    """Raise a clear WorkerError for a worker that predates / mismatches the embed protocol."""
    if resp.get("proto") != worker_protocol.EMBED_PROTOCOL_VERSION:
        raise WorkerError(
            f"{what}: worker embed protocol {resp.get('proto')!r} != expected "
            f"{worker_protocol.EMBED_PROTOCOL_VERSION} -- worker needs redeploy")


def _embed_request(client, path: str, req: dict, what: str, concurrent: bool = False) -> dict:
    try:
        resp = client.request(path, req, timeout=_EMBED_BATCH_TIMEOUT, concurrent=concurrent)
    except WorkerError as exc:
        if "no handler" in str(exc):  # HTTP 404 from an old worker
            raise WorkerError(f"{what}: worker needs redeploy (does not know {path!r})") from exc
        raise
    _check_embed_proto(resp, what)
    return resp


def _item_to_named_bytes(item, i: int):
    """str path | (name, bytes) | PIL image -> (name, encoded bytes)."""
    if isinstance(item, str):
        with open(item, "rb") as f:
            return os.path.basename(item), f.read()
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), bytes(item[1])
    buf = io.BytesIO()  # PIL image (lossless so the worker sees the same pixels)
    item.save(buf, format="PNG")
    return f"<pil:{i}>.png", buf.getvalue()


class BothResult:
    """Result of :func:`embed_both`; every list is aligned with the input items."""

    def __init__(self, n: int):
        self.siglip = [None] * n   # np.float32 (D1,) or None
        self.dino = [None] * n     # np.float32 (D2,) or None
        self.errors = [None] * n   # str or None
        self.siglip_model = None
        self.dino_model = None
        self.gpu_seconds = 0.0     # summed worker decode+inference time over all batches
        self.wall_seconds = 0.0    # client wall clock (wall - gpu ~ transport/overhead)
        self.n_batches = 0
        # 'paths' = worker read the files from the shared disk (no bytes sent), 'upload' = bytes
        # were sent, 'mixed' = some of each (items outside data_root fall back to upload).
        self.mode = "upload"


PATHS_BATCH_SIZE = 64          # path-mode batches carry no bytes, so they can be much larger
_PROBE_TTL_NEGATIVE = 300.0    # re-probe a "not visible" answer after this many seconds


def _worker_sees_data_root(client, data_root) -> bool:
    """One-time (cached per client+root) probe: can the worker read ``data_root`` itself?
    A positive answer is cached for the process; a negative one (incl. an old worker with no
    probe handler) for a few minutes; transport errors are not cached (-> upload this time)."""
    cache = client.__dict__.setdefault("_data_root_probe", {})
    hit = cache.get(data_root)
    if hit is not None and (hit[0] or time.time() < hit[1]):
        return hit[0]
    try:
        resp = client.request(worker_protocol.PATH_PROBE_DATA_ROOT, {"data_root": data_root},
                              timeout=15, retries=0)
    except WorkerError as exc:
        visible, reason = False, str(exc)
        if "no handler" not in reason:
            return False  # timeout / transient: don't cache
    except WorkerUnavailable:
        return False
    else:
        visible, reason = bool(resp.get("visible")), resp.get("reason")
    cache[data_root] = (visible, time.time() + _PROBE_TTL_NEGATIVE)
    print(f"[worker-client] data_root {data_root!r} visible to worker: {visible} ({reason})",
          flush=True)
    return visible


def _rel_under_root(item, root_real):
    """Relative '/'-path of file ``item`` inside ``root_real`` (symlinks resolved), or None if
    it is not a plain path string or resolves outside the root (-> upload that item)."""
    if not isinstance(item, str):
        return None
    real = os.path.realpath(item)
    try:
        if os.path.commonpath([root_real, real]) != root_real or real == root_real:
            return None
    except ValueError:
        return None
    return os.path.relpath(real, root_real).replace(os.sep, "/")


def embed_both(client, items, batch_size: int = 16, workers: int = 3, on_batch=None,
               data_root=None) -> BothResult:
    """SigLIP 2 + DINOv2 vectors for ``items`` via the worker, ``workers`` batches in flight.

    ``items``: str paths, (name, bytes) tuples, or PIL images. Per-item failures give
    ``None`` + an error string; a transport/worker-level failure raises WorkerError (a stale
    worker says "worker needs redeploy"). ``on_batch(done_items, total)`` is called as
    batches complete (from the calling thread).

    Shared-disk mode: when ``data_root`` is given and the worker's (cached) probe says it can see
    it, str items inside ``data_root`` are sent as relative paths (no bytes) with batches of at
    least PATHS_BATCH_SIZE. Everything else (and an old worker) uses the upload path.
    ``BothResult.mode`` reports what was used.
    """
    items = list(items)
    n = len(items)
    out = BothResult(n)
    t0 = time.time()
    bs = max(1, int(batch_size))

    rels = [None] * n  # rel path per item when it goes via shared disk
    if data_root and n and any(isinstance(it, str) for it in items) \
            and _worker_sees_data_root(client, data_root):
        root_real = os.path.realpath(data_root)
        rels = [_rel_under_root(it, root_real) for it in items]
        if any(r is not None for r in rels):
            bs = max(bs, PATHS_BATCH_SIZE)
    n_paths = sum(1 for r in rels if r is not None)
    out.mode = "paths" if n_paths == n and n else ("mixed" if n_paths else "upload")
    if n:
        print(f"[worker-client] embed_both: {n} items, mode={out.mode} "
              f"({n_paths} via shared disk), batch_size={bs}", flush=True)
    starts = list(range(0, n, bs))

    def upload_part(idxs, errs):
        payload = []
        for k in idxs:
            try:
                name, data = _item_to_named_bytes(items[k], k)
            except (OSError, ValueError) as exc:  # unreadable file: isolate to this item
                errs[k] = f"read failed: {exc}"
                continue
            payload.append((k, {"name": name, "image": data}))
        if not payload:
            return [], []
        client.record("embed_both", f"{len(payload)} items")
        resp = _embed_request(client, worker_protocol.PATH_EMBED_BOTH_BATCH,
                              {"items": [p for _, p in payload]}, "embed_both", concurrent=True)
        return [k for k, _ in payload], resp

    def paths_part(idxs):
        client.record("embed_both", f"{len(idxs)} paths")
        try:
            resp = _embed_request(client, worker_protocol.PATH_EMBED_BOTH_PATHS,
                                  {"data_root": data_root, "paths": [rels[k] for k in idxs]},
                                  "embed_both", concurrent=True)
        except WorkerError as exc:
            if "needs redeploy" not in str(exc):
                raise
            # Old worker without the path mode: remember, and upload these items instead.
            client.__dict__.setdefault("_data_root_probe", {})[data_root] = (
                False, time.time() + _PROBE_TTL_NEGATIVE)
            return None
        return idxs, resp

    def run_batch(start):
        idxs = list(range(start, min(n, start + bs)))
        errs, parts = {}, []  # parts: (item indices, reply)
        p_idx = [k for k in idxs if rels[k] is not None]
        u_idx = [k for k in idxs if rels[k] is None]
        if p_idx:
            got = paths_part(p_idx)
            if got is None:
                u_idx = idxs
            else:
                parts.append(got)
        if u_idx:
            ks, resp = upload_part(u_idx, errs)
            if ks:
                parts.append((ks, resp))
        gpu, models, merged = 0.0, (None, None), []
        for ks, resp in parts:
            if resp.get("error"):
                raise WorkerError(resp["error"])
            results = resp.get("results") or []
            if len(results) != len(ks):
                raise WorkerError(f"embed_both: worker returned {len(results)} results "
                                  f"for {len(ks)} items")
            gpu += float(resp.get("gpu_seconds") or 0.0)
            models = (resp.get("siglip_model"), resp.get("dino_model"))
            merged.extend(zip(ks, results))
        return merged, errs, gpu, models

    def merge(res):
        merged, errs, gpu, models = res
        for k, msg in errs.items():
            out.errors[k] = msg
        for k, r in merged:
            if r.get("error") or r.get("siglip") is None or r.get("dino") is None:
                out.errors[k] = r.get("error") or "no embedding returned"
                continue
            out.siglip[k] = np.frombuffer(r["siglip"], dtype=np.float32).copy()
            out.dino[k] = np.frombuffer(r["dino"], dtype=np.float32).copy()
        out.gpu_seconds += gpu
        out.n_batches += 1
        if models[0]:
            out.siglip_model, out.dino_model = models

    done = 0
    if starts:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = [pool.submit(run_batch, s) for s in starts]  # all submitted; pool bounds in-flight
            try:
                for f in futs:  # consume in order -> deterministic progress
                    merge(f.result())
                    done = min(n, done + bs)
                    if on_batch:
                        on_batch(done, n)
            except BaseException:
                for f in futs:
                    f.cancel()
                raise
    out.wall_seconds = time.time() - t0
    return out


class _RemoteSharedEmbedder:
    """Common bits of the SigLIP/DINO proxies: both ride on one embed_both batch."""
    _which = None  # "siglip" | "dino"

    def __init__(self, client: WorkerClient, batch_size: int = 16, workers: int = 3):
        self.client = client
        self.batch_size = batch_size
        self.workers = workers
        self._model_id = None

    def _run(self, items):
        res = embed_both(self.client, items, self.batch_size, self.workers)
        self.siglip_model, self.dino_model = res.siglip_model, res.dino_model
        if res.siglip_model:
            self._model_id = getattr(res, self._which + "_model")
        return res

    def embed_images(self, paths):
        paths = list(paths)
        res = self._run(paths)
        vecs = getattr(res, self._which)
        ok = [v for v in vecs if v is not None]
        failed = [(p, e) for p, e, v in zip(paths, res.errors, vecs) if v is None]
        emb = np.stack(ok).astype(np.float32) if ok else np.empty((0,), dtype=np.float32)
        return emb, failed

    def embed_pil_images(self, images):
        images = list(images)
        if not images:
            return np.empty((0,), dtype=np.float32)
        res = self._run(images)
        bad = [(i, e) for i, (e, v) in enumerate(zip(res.errors, getattr(res, self._which)))
               if v is None]
        if bad:
            raise WorkerError(f"embed_pil_images failed for {bad[:3]}")
        return np.stack(getattr(res, self._which)).astype(np.float32)

    def model_id(self) -> str:
        """Model id the WORKER reports (so DB rows record what actually ran). Cached; the
        first call on a fresh proxy costs one text probe (siglip) or tiny embed (dino)."""
        if self._model_id is None:
            self._probe_model_id()
        return self._model_id

    def _probe_model_id(self):
        raise NotImplementedError


class RemoteSigLIPEmbedder(_RemoteSharedEmbedder):
    """Drop-in for :class:`embedders.SigLIP2Embedder` backed by the worker."""
    _which = "siglip"

    def embed_text(self, text):
        self.client.record("embed_siglip_text", text[:40])
        resp = _embed_request(self.client, worker_protocol.PATH_EMBED_SIGLIP_TEXT,
                              {"text": text}, "embed_siglip_text")
        if resp.get("error") or resp.get("embedding") is None:
            raise WorkerError(resp.get("error") or "no embedding returned")
        if resp.get("siglip_model"):
            self._model_id = resp["siglip_model"]
        return np.frombuffer(resp["embedding"], dtype=np.float32).copy()

    def _probe_model_id(self):
        self.embed_text("a")


class RemoteDINOEmbedder(_RemoteSharedEmbedder):
    """Drop-in for :class:`embedders.DINOv2Embedder` backed by the worker."""
    _which = "dino"

    def _probe_model_id(self):
        # No text path for DINO; a 1x1 image through embed_both reports both model ids.
        from PIL import Image  # hook-allow: lazy — only for the probe
        res = self._run([Image.new("RGB", (32, 32))])
        if self._model_id is None:
            raise WorkerError(f"dino model id probe failed: {res.errors[:1]}")


class RemoteAgeEstimator:
    """Drop-in for :class:`age_estimator.AgeGenderEstimator` backed by the worker.

    Same ``estimate(image_path, faces, bodies=None)`` signature — ships the image
    + face (and face-less body) bboxes to the worker, which runs MiVOLO in its OWN isolated ``.age-venv``
    there. Raises :class:`RuntimeError` (matching the local estimator's error
    contract, which web.py's ``api_estimate_age`` already handles) rather than
    WorkerError, so a failure surfaces cleanly with no web.py change."""

    def __init__(self, client: WorkerClient):
        self.client = client

    def estimate(self, image_path: str, faces: list, bodies: list | None = None) -> list:
        bodies = bodies or []
        if not faces and not bodies:
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
                       for fc in faces],
             "bodies": [{"face_ref": b["ref"], "bbox": [float(v) for v in b["bbox"]]}
                        for b in bodies]},
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
            image_bytes, read_error = _read_for_upload(path)
            if read_error:
                results_out.append((path, [], read_error))
                continue
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
        from .detector import YOLOWorldDetector  # hook-allow: lazy — model module, only for id
        return YOLOWorldDetector.model_id(*args, **kwargs)
