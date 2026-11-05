"""Shared media-worker offload protocol.

This module is the single source of truth for the wire contract between the
media-worker *server* (which runs the heavy ML: face/body/SigLIP/DINO/YOLO) and the
*client* that offloads work to it. Both sides import this module so the request
paths and (de)serialization stay in lockstep.

Transport is plain HTTP: the worker is a small local FastAPI/uvicorn service and
each path below is a ``POST /<path>`` route. Payloads are msgpack — one packed
dict per request, one per response — because it round-trips raw ``bytes`` (image
data, float32 embeddings) intact without a schema. (This used to ride over
Reticulum with the umsgpack vendored inside RNS; the wire dicts are identical, so
only the codec import and the transport changed.)
"""
import msgpack

# Request handler paths (HTTP route names).
PATH_PING = "ping"
PATH_DETECT_FACES = "detect_faces"
PATH_EMBED_BBOX = "embed_bbox"
# Re-run in-plane rotation normalization on faces ALREADY stored in the host's DB
# (the renormalization backfill). Stateless like the ops above, but the request
# carries the existing face dicts so the worker can apply the margin guardrails
# against the scores the host already has.
PATH_NORMALIZE_FACES = "normalize_faces"
# SigLIP 2 + DINOv2 embedding pipeline (embedders.py). Batched: one request carries N
# images, decoded ONCE on the worker and fed to both models (embedders.embed_shared).
PATH_EMBED_BOTH_BATCH = "embed_both_batch"
PATH_EMBED_SIGLIP_TEXT = "embed_siglip_text"
# Shared-disk mode (web server and worker on the same machine): instead of uploading bytes the
# client sends {data_root, paths:[rel...]} and the worker reads <data_root>/<rel> itself.
# probe_data_root ({data_root} -> {visible, reason}) is the cheap check that it can.
PATH_EMBED_BOTH_PATHS = "embed_both_paths"
PATH_PROBE_DATA_ROOT = "probe_data_root"
# Bumped when the embed_both_batch / embed_siglip_text wire shape changes. Replies carry it as
# "proto" so a client can tell a stale (un-redeployed) worker from a real failure.
EMBED_PROTOCOL_VERSION = 1
# Shared-disk face detection (same data_root contract as embed_both_paths): the worker reads
# <data_root>/<rel> itself and returns one result per path, in request order.
#   req:  {"data_root": <str>, "paths": [<rel '/'-path>, ...]}
#   resp: {"results": [{"name": <rel>, "faces": [<same face dicts as PATH_DETECT_FACES>],
#                       "error": <None|str>}],
#          "proto": FACE_PROTOCOL_VERSION, "gpu_seconds": <float>, "wall_seconds": <float>,
#          "error": <None|str: whole-request failure>}
# An old worker answers HTTP 404 (no handler) -> the client falls back to PATH_DETECT_FACES.
PATH_DETECT_FACES_PATHS = "detect_faces_paths"
FACE_PROTOCOL_VERSION = 1
# Optional field "orient": ORIENT_GATE_VERSION in the replies of detect_faces,
# detect_faces_paths, embed_bbox and normalize_faces, set by a worker whose detector runs
# the always-on orientation gate (landmark-roll triggered quarter-turn re-embed). Absent on
# an older worker, which keeps working with its previous (det_score-gated) behaviour; the
# host logs that fallback once. embed_bbox requests may carry "auto_orient": bool (default
# true; false = embed the region exactly as given, for the explicit rotate controls).
ORIENT_GATE_VERSION = 1
PATH_DETECT_OBJECTS = "detect_objects"
PATH_ESTIMATE_AGE = "estimate_age"
# Per-tag model TRAINING (job/poll pattern) + fine-tuned inference. Unlike the
# stateless inference ops above, these carry a job_id and the worker keeps
# state (a scratch dataset while training, a persisted checkpoint after).
PATH_TRAIN_CREATE = "train_create"
PATH_TRAIN_ADD = "train_add"
PATH_TRAIN_RUN = "train_run"
PATH_TRAIN_STATUS = "train_status"
PATH_TRAIN_CANCEL = "train_cancel"
PATH_TAG_DETECT = "tag_detect"
# In-memory search index ("imdb"): the worker holds the big SigLIP 2 + face embedding
# matrices resident in RAM and answers ranking queries, so the low-RAM host never
# loads them (the OOM this exists to prevent). The host STREAMS the matrix in
# bounded chunks (imdb_build_*) so neither side ever materializes the whole ~1 GB
# blob; imdb_search ranks a query vector (optionally within a host-supplied id set).
PATH_IMDB_BUILD_BEGIN = "imdb_build_begin"
PATH_IMDB_BUILD_CHUNK = "imdb_build_chunk"
PATH_IMDB_BUILD_END = "imdb_build_end"
PATH_IMDB_STATUS = "imdb_status"
PATH_IMDB_SEARCH = "imdb_search"
PATH_PATTERN = "pattern_descriptor"
# Instance-level region search (region_index.py): DINOv2 windows + RootSIFT for indexing,
# DINOv2 query vectors for searching.
PATH_REGION = "region_descriptor"
# Find-by-pattern, DINOv2-L dense-cell algorithm (pattern_dino.py). Batched index op (upload or
# shared-disk paths, same data_root contract as embed_both_paths) + a query op. A worker that
# predates it answers HTTP 404 ("no handler"), a stale one a different "proto"/"algo": the client
# turns both into a "worker needs redeploy" WorkerError (no silent fallback).
#   req  {"op": "index", "items": [{"name", "image": bytes}]}           (upload)
#        {"op": "index", "data_root": <str>, "paths": [<rel '/'-path>]} (shared disk)
#   resp {"results": [{"name", "cells": int8 bytes (n, dim), "n", "error"}], "algo", "dim",
#         "gpu_seconds", "decode_seconds", "lock_wait_seconds", "wall_seconds",
#         "proto": PATTERN_DINO_PROTOCOL_VERSION, "error"}
#   req  {"op": "query", "image": bytes, "box": [x1,y1,x2,y2], "mode": "mask|crop|both", "rots": [0..3]}
#   resp {"queries": float32 bytes (S, dim), "n": S, "algo", "dim", "proto", "error"}
#   req  {"op": "info"} -> {"ready": bool, "algo", "dim", "proto", "error"}
PATH_PATTERN_DINO = "pattern_dino"
PATTERN_DINO_PROTOCOL_VERSION = 1

# Person-masked embedding (location_mask.py, 'place' location model stage 2). Same two transports
# as embed_both: shared-disk {"data_root", "paths": [rel...], "boxes": [[[x1,y1,x2,y2]...]...],
# "max_coverage"} (same data_root / realpath containment checks as embed_both_paths) and upload
# {"items": [{"name", "image", "boxes"}...], "max_coverage"}. boxes are NORMALISED (0..1) person
# regions of the upright image; the worker fills them with the mean colour of the rest of the frame
# and embeds the result with SigLIP 2 + DINOv2. Reply = embed_both's, each result also carrying
# "coverage" (masked fraction; an item over max_coverage gets an error instead of a vector), plus
# "masked_proto": MASKED_PROTOCOL_VERSION. An old worker answers HTTP 404 -> the client reports
# "worker needs redeploy" and the host serves stage 1 only, with a visible notice.
PATH_EMBED_MASKED_PATHS = "embed_masked_paths"
PATH_EMBED_MASKED_BATCH = "embed_masked_batch"
MASKED_PROTOCOL_VERSION = 1

ALL_PATHS = [
    PATH_EMBED_MASKED_PATHS,
    PATH_EMBED_MASKED_BATCH,
    PATH_PING,
    PATH_PATTERN,
    PATH_PATTERN_DINO,
    PATH_REGION,
    PATH_DETECT_FACES,
    PATH_EMBED_BBOX,
    PATH_NORMALIZE_FACES,
    PATH_EMBED_BOTH_BATCH,
    PATH_EMBED_SIGLIP_TEXT,
    PATH_DETECT_OBJECTS,
    PATH_ESTIMATE_AGE,
    PATH_TRAIN_CREATE,
    PATH_TRAIN_ADD,
    PATH_TRAIN_RUN,
    PATH_TRAIN_STATUS,
    PATH_TRAIN_CANCEL,
    PATH_TAG_DETECT,
    PATH_IMDB_BUILD_BEGIN,
    PATH_IMDB_BUILD_CHUNK,
    PATH_IMDB_BUILD_END,
    PATH_IMDB_STATUS,
    PATH_IMDB_SEARCH,
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
#                       "det_score": <float>,
#                       "angle": <float>}],   # in-plane roll, CW from upright,
#                                             # (-180, 180]; 0.0 = upright
#            "error": <None|str>}
#
#   PATH_EMBED_BBOX
#     req:  {"name": <str>, "image": <bytes: encoded image>,
#            "bbox": [x1, y1, x2, y2], "pad_ratio": <float>}
#     resp: {"bbox": [<floats>]|None, "embedding": <bytes: float32 (512,)>,
#            "det_score": <float>, "angle": <float>, "error": <None|str>}
#
#   PATH_NORMALIZE_FACES
#     req:  {"name": <str>, "image": <bytes: encoded image>,
#            "faces": [{"bbox": [x1, y1, x2, y2],
#                       "embedding": <bytes: float32 (512,)>,
#                       "det_score": <float>, "angle": <float>}, ...]}
#     resp: {"faces": [ <same shape as the request's faces> ],  # SAME length and
#                       # order as the request: the caller maps them back to its
#                       # own DB rows positionally
#            "error": <None|str>}
#
#   PATH_EMBED_BOTH_BATCH
#     req:  {"items": [{"name": <str>, "image": <bytes: encoded image>}, ...]}
#     resp: {"results": [{"name": <str>, "siglip": <bytes float32 (D1,)>|None,
#                         "dino": <bytes float32 (D2,)>|None, "error": <None|str>}, ...],
#                         # SAME length/order as req items; a bad image sets only its own error
#            "siglip_model": <str>, "dino_model": <str>,
#            "gpu_seconds": <float: time inside forward passes + host sync, under the GPU lock>,
#            "decode_seconds": <float: decode+preprocess time summed over decode threads>,
#            "lock_wait_seconds": <float: time blocked waiting for the GPU lock>,
#            "wall_seconds": <float: worker wall time for this request>,
#            (the three new fields are optional/additive; absent from older workers)
#            "proto": EMBED_PROTOCOL_VERSION, "error": <None|str: whole-batch failure>}
#
#   PATH_EMBED_SIGLIP_TEXT
#     req:  {"text": <str>}
#     resp: {"embedding": <bytes float32 (D1,)>|None, "siglip_model": <str>,
#            "proto": EMBED_PROTOCOL_VERSION, "error": <None|str>}
#
#   PATH_DETECT_OBJECTS
#     req:  {"name": <str>, "image": <bytes>, "vocab": [<str>, ...]|None}
#     resp: {"detections": [[class_name(str), conf(float),
#                            x1, y1, x2, y2 (floats)]],
#            "error": <None|str>}
#
#   PATH_ESTIMATE_AGE   (MiVOLO age/gender, run in the worker's isolated .age-venv)
#     req:  {"name": <str>, "image": <bytes: encoded image>,
#            "faces": [{"face_ref": <str>, "bbox": [x1, y1, x2, y2]}, ...],
#            "bodies": [{"face_ref": "body:<id>", "bbox": [...]}, ...]}  # optional,
#                                   # face-less named bodies (body-only MiVOLO)
#     resp: {"results": [{"face_ref": <str>, "age": <float|None>,
#                         "gender": <str|None>}, ...],
#            "error": <None|str>}   # error e.g. "run media age-setup" when the
#                                   # worker has no .age-venv
#
# --- Per-tag training (job/poll) + fine-tuned inference --------------------
# The host driver (remote_tag_trainer.py) creates a job, uploads the training
# data in batches (small requests, so RNS never has to ship one huge payload),
# starts it, then polls status until a terminal state. YOLO checkpoints stay on
# the worker (served by PATH_TAG_DETECT); the embedding-classifier artifact (a small
# weights.npz) rides back in the final status response. The embedding space is part of
# the kind: "siglip_model" (SigLIP 2, default) or "dino_model". A worker that predates a
# kind (or is sent the retired "clip_model") rejects it in train_create ("unknown training
# kind"), so a stale worker can never train in the wrong space.
#
#   PATH_TRAIN_CREATE
#     req:  {"kind": "yolo_model"|"siglip_model"|"dino_model", "tag": <str>, "slug": <str>,
#            "epochs": <int|None>}
#     resp: {"job_id": <str>, "error": <None|str>}
#
#   PATH_TRAIN_ADD   (append one batch of examples; call repeatedly)
#     req:  {"job_id": <str>, "kind": <str>, "batch": [ <item>, ... ]}
#           yolo item:  {"name": <str>, "image": <bytes: <=640px JPEG>,
#                        "boxes": [[cx, cy, w, h] normalized floats, ...]}
#           embedding item (all *_model kinds except yolo):  {"label": 0|1,
#                        "embedding": <bytes: float32 (D,), in the kind's space>}   # whole-image
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
#            "artifact": <bytes|None>,   # weights.npz (embedding kinds), only if want_artifact & done
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
#
# --- In-memory search index ("imdb") --------------------------------------
# The host (remote_index_builder.py) rebuilds a matrix by streaming it in chunks:
# begin (reset the receive buffer), many chunk (append rows), end (finalize the
# buffer into one resident (N,D) float32 matrix + parallel id array). "kind" is
# "siglip" or "face". ids are the row's file_id (for face: the file_id per face row,
# so a search maps to photos). Vectors are contiguous little-endian float32,
# count*D of them per chunk.
#
#   PATH_IMDB_BUILD_BEGIN
#     req:  {"kind": "siglip"|"face", "dim": <int>, "total_rows": <int|None>}
#     resp: {"ok": <bool>, "error": <None|str>}
#
#   PATH_IMDB_BUILD_CHUNK   (append one block of rows; call repeatedly)
#     req:  {"kind": <str>, "ids": [<int>, ...], "vecs": <bytes: float32 count*D>}
#     resp: {"received": <int total rows buffered>, "error": <None|str>}
#
#   PATH_IMDB_BUILD_END   (finalize the buffer into the resident matrix)
#     req:  {"kind": <str>}
#     resp: {"rows": <int>, "dim": <int>, "bytes": <int>, "error": <None|str>}
#
#   PATH_IMDB_STATUS
#     req:  {}
#     resp: {"siglip": {"rows": <int>, "dim": <int>, "bytes": <int>,
#                     "built_at": <int|None>, "building": <bool>},
#            "face": {...}, "error": <None|str>}
#
#   PATH_IMDB_SEARCH   (M2 — rank a query vector against a resident matrix)
#     req:  {"kind": <str>, "query": <bytes: float32 (D,)>, "k": <int>,
#            "allowed_ids": [<int>, ...]|None}   # None = rank all
#     resp: {"results": [[id(int), score(float)], ...], "error": <None|str>}
# ---------------------------------------------------------------------------


def pack(obj) -> bytes:
    """Serialize a Python object to msgpack bytes for the wire.

    ``use_bin_type=True`` keeps raw ``bytes`` (image data, float32 embeddings)
    encoded as msgpack bin rather than str, so they round-trip byte-for-byte."""
    return msgpack.packb(obj, use_bin_type=True)


def unpack(data: bytes):
    """Deserialize msgpack bytes received off the wire back to Python objects.

    ``raw=False`` decodes msgpack str to ``str`` (bin stays ``bytes``);
    ``strict_map_key=False`` tolerates non-str dict keys."""
    return msgpack.unpackb(data, raw=False, strict_map_key=False)
