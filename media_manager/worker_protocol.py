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

ALL_PATHS = [
    PATH_PING,
    PATH_DETECT_FACES,
    PATH_EMBED_BBOX,
    PATH_EMBED_IMAGE,
    PATH_EMBED_TEXT,
    PATH_DETECT_OBJECTS,
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
# ---------------------------------------------------------------------------


def pack(obj) -> bytes:
    """Serialize a Python object to msgpack bytes for the wire."""
    return umsgpack.packb(obj)


def unpack(data: bytes):
    """Deserialize msgpack bytes received off the wire back to Python objects."""
    return umsgpack.unpackb(data)
