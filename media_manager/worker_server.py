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
import shutil
import threading
import tempfile
import traceback
import collections

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

# Age/gender (MiVOLO) runs in its OWN isolated .age-venv subprocess (see
# age_estimator.py), NOT in this process — so it doesn't contend with the three
# in-process models above, and gets its own lock rather than models.lock. The
# lock just serializes concurrent age requests so two MiVOLO subprocesses can't
# spike RAM at once. Lazily built (its __init__ only resolves the venv path).
_age_estimator = None
_age_lock = threading.Lock()


def _get_age_estimator():
    global _age_estimator
    if _age_estimator is None:
        from .age_estimator import AgeGenderEstimator
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


def handle_estimate_age(path, data, request_id, link_id, remote_identity, requested_at):
    name = None
    tmp = None
    try:
        req = worker_protocol.unpack(data)
        name = req.get("name")
        # Map the wire faces to the shape AgeGenderEstimator.estimate expects
        # ('ref' + 'bbox'); it returns [{face_ref, age, gender}, ...] already.
        faces = [{"ref": f["face_ref"], "bbox": f["bbox"]} for f in (req.get("faces") or [])]
        tmp = _write_temp(req["image"], name)
        with _age_lock:
            results = _get_age_estimator().estimate(tmp, faces)
        print(f"[worker] handled {worker_protocol.PATH_ESTIMATE_AGE} ({name}) -> "
              f"{len(results)} face(s)", flush=True)
        return worker_protocol.pack({"results": results, "error": None})
    except Exception as e:
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
        # CLIP: whole-image [(emb_bytes, label)] and region [(crop_jpeg, label)]
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
    from media_manager import yolo_tag_classifier as yt
    try:
        examples = job.yolo_examples
        if not examples:
            raise ValueError('no training images were uploaded')
        # 80/20 split by image (deterministic seed), same as the local trainer.
        import random
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
    except Exception as e:
        traceback.print_exc()
        job.status = 'failed'
        job.error = str(e)
        job.log(f'failed: {e}')
    finally:
        shutil.rmtree(os.path.join(job.scratch_dir, 'staging'), ignore_errors=True)


def _run_clip_job(job):
    from media_manager import clip_tag_classifier as ct
    try:
        xs, ys = [], []
        for emb_bytes, label in job.clip_whole:
            xs.append(np.frombuffer(emb_bytes, dtype=np.float32))
            ys.append(label)
        if job.clip_regions:
            from PIL import Image as PILImage
            crops, labels = [], []
            for crop_bytes, label in job.clip_regions:
                crops.append(PILImage.open(io.BytesIO(crop_bytes)).convert('RGB'))
                labels.append(label)
            job.log(f'embedding {len(crops)} region crops on the worker CLIP model')
            with models.lock:
                embs = models.get_clip_indexer().embed_pil_images(crops)
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
        job.log(f'fitting CLIP logistic on {n_pos} pos / {n_neg} neg')
        w, b, cv = ct.fit_from_examples(X, y)

        out_dir = os.path.join(TAG_MODELS_DIR, job.slug, 'clip_model')
        os.makedirs(out_dir, exist_ok=True)
        weights_path = os.path.join(out_dir, 'weights.npz')
        np.savez(weights_path, w=w, b=np.float32(b), embedding_model='ViT-B-32/openai')
        with open(weights_path, 'rb') as f:
            job.artifact_bytes = f.read()
        job.metrics = {'cv_accuracy': round(cv, 4), 'n_positive': n_pos, 'n_negative': n_neg}
        job.status = 'done'
        job.log(f'done; cv_accuracy={round(cv, 4)}')
    except Exception as e:
        traceback.print_exc()
        job.status = 'failed'
        job.error = str(e)
        job.log(f'failed: {e}')


def _get_tag_model(slug, checkpoint):
    """Load (cached, LRU) a per-tag YOLO checkpoint for inference. MUST be called
    while holding models.lock (mirrors the other inference paths)."""
    from ultralytics import YOLOWorld
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
        if kind not in ('yolo_model', 'clip_model'):
            raise ValueError(f'unknown training kind {kind!r}')
        tag = req.get('tag', '')
        slug = req['slug']
        epochs = int(req.get('epochs') or _TRAIN_DEFAULT_EPOCHS)
        job = training_jobs.create(kind, tag, slug, epochs)
        job.log(f'created {kind} job for tag {tag!r} (slug={slug})')
        print(f"[worker] handled {worker_protocol.PATH_TRAIN_CREATE} ({kind} {slug}) "
              f"-> {job.job_id}", flush=True)
        return worker_protocol.pack({'job_id': job.job_id, 'error': None})
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
            results = model.predict(tmp, conf=conf, verbose=False)
        detections = _parse_yolo_results(results)
        print(f"[worker] handled {worker_protocol.PATH_TAG_DETECT} ({slug}/{name}) -> "
              f"{len(detections)} detections", flush=True)
        return worker_protocol.pack({'detections': detections, 'error': None})
    except Exception as e:
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
    except Exception as e:
        traceback.print_exc()
        return worker_protocol.pack({'ok': False, 'error': str(e)})


def handle_imdb_build_chunk(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        received = imdb_index.add_chunk(req['kind'], req['ids'], req['vecs'])
        return worker_protocol.pack({'received': received, 'error': None})
    except Exception as e:
        traceback.print_exc()
        return worker_protocol.pack({'received': 0, 'error': str(e)})


def handle_imdb_build_end(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        req = worker_protocol.unpack(data)
        rows, dim, nbytes = imdb_index.end(req['kind'], int(time.time()))
        print(f"[worker] imdb build end kind={req['kind']} rows={rows} dim={dim} "
              f"bytes={nbytes} ({nbytes / 1048576:.1f} MB)", flush=True)
        return worker_protocol.pack({'rows': rows, 'dim': dim, 'bytes': nbytes, 'error': None})
    except Exception as e:
        traceback.print_exc()
        return worker_protocol.pack({'rows': 0, 'dim': 0, 'bytes': 0, 'error': str(e)})


def handle_imdb_status(path, data, request_id, link_id, remote_identity, requested_at):
    try:
        resp = imdb_index.status()
        resp['error'] = None
        return worker_protocol.pack(resp)
    except Exception as e:
        traceback.print_exc()
        return worker_protocol.pack({'clip': {}, 'face': {}, 'error': str(e)})


# Map of request path -> handler, used both by run() and the loopback test.
HANDLERS = {
    worker_protocol.PATH_PING: handle_ping,
    worker_protocol.PATH_DETECT_FACES: handle_detect_faces,
    worker_protocol.PATH_EMBED_BBOX: handle_embed_bbox,
    worker_protocol.PATH_EMBED_IMAGE: handle_embed_image,
    worker_protocol.PATH_EMBED_TEXT: handle_embed_text,
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
