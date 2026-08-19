"""Shared Reticulum (RNS) media-worker offload protocol.

This module is the single source of truth for the wire contract between the
media-worker *server* (which runs the heavy ML: face/body/CLIP/YOLO) and the
*client* that offloads work to it over Reticulum. Both sides import this module
so the destination naming, request paths, and (de)serialization stay in lockstep.

The worker is reached as an RNS destination with app name ``media_manager`` and
aspect ``worker``. Requests are dispatched by the path constants below; payloads
are packed with the umsgpack bundled inside RNS so no extra dependency is needed
and raw ``bytes`` (image data, float32 embeddings) survive the round trip intact.
"""
from RNS.vendor import umsgpack

APP_NAME = "media_manager"
ASPECT = "worker"

# RNS request handler paths.
PATH_PING = "ping"
PATH_DETECT_FACES = "detect_faces"
PATH_EMBED_BBOX = "embed_bbox"
PATH_EMBED_IMAGE = "embed_image"
PATH_EMBED_TEXT = "embed_text"
PATH_DETECT_OBJECTS = "detect_objects"
# Per-tag model TRAINING (job/poll pattern) + fine-tuned inference. Unlike the
# stateless inference ops above, these carry a job_id and the worker keeps
# state (a scratch dataset while training, a persisted checkpoint after).
PATH_TRAIN_CREATE = "train_create"
PATH_TRAIN_ADD = "train_add"
PATH_TRAIN_RUN = "train_run"
PATH_TRAIN_STATUS = "train_status"
PATH_TRAIN_CANCEL = "train_cancel"
PATH_TAG_DETECT = "tag_detect"

ALL_PATHS = [
    PATH_PING,
    PATH_DETECT_FACES,
    PATH_EMBED_BBOX,
    PATH_EMBED_IMAGE,
    PATH_EMBED_TEXT,
    PATH_DETECT_OBJECTS,
    PATH_TRAIN_CREATE,
    PATH_TRAIN_ADD,
    PATH_TRAIN_RUN,
    PATH_TRAIN_STATUS,
    PATH_TRAIN_CANCEL,
    PATH_TAG_DETECT,
]

# ---------------------------------------------------------------------------
# Request / response schemas (reference for other implementers).
#
# All requests and responses are dicts, packed/unpacked with pack()/unpack().
# Embeddings are raw little-endian float32 bytes (use numpy frombuffer/tobytes).
# Bounding boxes are [x1, y1, x2, y2] as floats.
#
#   PATH_PING
#     req:  {}
#     resp: {"ok": True, "models": [<str>, ...], "error": None}
#
#   PATH_DETECT_FACES
#     req:  {"name": <str basename>, "image": <bytes: encoded image>}
#     resp: {"faces": [{"bbox": [x1, y1, x2, y2] (floats),
#                       "embedding": <bytes: float32 (512,)>,
#                       "det_score": <float>}],
#            "error": <None|str>}
#
#   PATH_EMBED_BBOX
#     req:  {"name": <str>, "image": <bytes: encoded image>,
#            "bbox": [x1, y1, x2, y2], "pad_ratio": <float>}
#     resp: {"bbox": [<floats>]|None, "embedding": <bytes: float32 (512,)>,
#            "det_score": <float>, "error": <None|str>}
#
#   PATH_EMBED_IMAGE
#     req:  {"name": <str>, "image": <bytes: encoded image>}
#     resp: {"embedding": <bytes: float32 (D,)>|None, "error": <None|str>}
#
#   PATH_EMBED_TEXT
#     req:  {"text": <str>}
#     resp: {"embedding": <bytes: float32 (D,)>, "error": <None|str>}
#
#   PATH_DETECT_OBJECTS
#     req:  {"name": <str>, "image": <bytes>, "vocab": [<str>, ...]|None}
#     resp: {"detections": [[class_name(str), conf(float),
#                            x1, y1, x2, y2 (floats)]],
#            "error": <None|str>}
#
# --- Per-tag training (job/poll) + fine-tuned inference --------------------
# The host driver (remote_tag_trainer.py) creates a job, uploads the training
# data in batches (small requests, so RNS never has to ship one huge payload),
# starts it, then polls status until a terminal state. YOLO checkpoints stay on
# the worker (served by PATH_TAG_DETECT); the CLIP artifact (a ~2 KB weights.npz)
# rides back in the final status response.
#
#   PATH_TRAIN_CREATE
#     req:  {"kind": "yolo_model"|"clip_model", "tag": <str>, "slug": <str>,
#            "epochs": <int|None>}
#     resp: {"job_id": <str>, "error": <None|str>}
#
#   PATH_TRAIN_ADD   (append one batch of examples; call repeatedly)
#     req:  {"job_id": <str>, "kind": <str>, "batch": [ <item>, ... ]}
#           yolo item:  {"name": <str>, "image": <bytes: <=640px JPEG>,
#                        "boxes": [[cx, cy, w, h] normalized floats, ...]}
#           clip item:  {"label": 0|1,
#                        "embedding": <bytes: float32 (D,)>}   # whole-image
#                        | {"image": <bytes: cropped+downscaled JPEG>}  # region
#     resp: {"received": <int>, "error": <None|str>}
#
#   PATH_TRAIN_RUN   (start the detached training thread)
#     req:  {"job_id": <str>}
#     resp: {"ok": <bool>, "error": <None|str>}
#
#   PATH_TRAIN_STATUS
#     req:  {"job_id": <str>, "want_artifact": <bool>}
#     resp: {"status": "created"|"running"|"done"|"failed"|"cancelled",
#            "current_epoch": <int>, "total_epochs": <int>,
#            "metrics": {<str>: <num|None>, ...}, "log_tail": <str>,
#            "artifact": <bytes|None>,   # clip weights.npz, only if want_artifact & done
#            "error": <None|str>}
#
#   PATH_TRAIN_CANCEL
#     req:  {"job_id": <str>}
#     resp: {"ok": <bool>, "error": <None|str>}
#
#   PATH_TAG_DETECT   (run a tag's fine-tuned YOLO checkpoint on one image)
#     req:  {"slug": <str>, "kind": "yolo_model", "name": <str>,
#            "image": <bytes: <=640px JPEG>, "conf": <float|None>}
#     resp: {"detections": [[class_name, conf, x1, y1, x2, y2], ...],
#            "error": <None|str>}   # error set (e.g. "no model") when untrained
# ---------------------------------------------------------------------------


def pack(obj) -> bytes:
    """Serialize a Python object to msgpack bytes for the wire."""
    return umsgpack.packb(obj)


def unpack(data: bytes):
    """Deserialize msgpack bytes received off the wire back to Python objects."""
    return umsgpack.unpackb(data)
