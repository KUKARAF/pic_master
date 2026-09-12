"""GPU-accelerated similarity search with matrices kept RESIDENT in VRAM.

Similarity search is a mat-vec: a pre-built ``[N, D]`` float32 embedding matrix
(D=512, N up to ~233k → ~478 MB) dotted against one ``[D]`` query. On CPU that is
pure numpy BLAS and it is *memory-bandwidth* bound, not compute bound. Naively
"doing it on the GPU" per request would be SLOWER than CPU, because you'd pay to
upload the whole 478 MB matrix across PCIe on every single call and the matmul
itself is trivial next to that transfer.

The ONLY way a GPU wins here is to upload each matrix to VRAM exactly ONCE and
keep it resident, reusing it across every request; then each call ships only the
tiny ``[D]`` query. That is what this module does: a process-global cache maps a
string key to ``(version, device_tensor)``. The caller passes the numpy matrix
plus the integer version counter it already maintains for cache invalidation
(``Database._emb_ver`` / ``_face_ver``, ``ManualDB._face_ver``); when the version
differs from what's resident we rebuild the device tensor from the given matrix
and drop the old one (freeing its VRAM on GC). So residency is tied to exactly
the same write-invalidation the numpy matrix caches already use — never stale.

CPU-passthrough contract (the important half): if there is no usable GPU
(``compute.torch_device()`` is ``'cpu'``, or torch isn't importable at all) EVERY
public function returns ``None``. Callers treat ``None`` as "no GPU — keep doing
exactly what you did before" and fall back to their existing numpy ``matrix.dot``.
This guarantees byte-identical behavior on non-GPU machines: on CPU we import
nothing heavier than torch (via :mod:`compute`) and touch no device at all.

Per the project "no silent failures" rule (see ``compute.py``): the CPU fallback
is a *documented contract*, not a swallowed error — torch-not-installed is the
one thing treated as "no GPU". Once we're actually on a GPU, every build/upload/
matmul error propagates loudly; we never quietly fall back to CPU mid-flight and
hide a broken accelerator.

VRAM note: each resident matrix is ~0.5 GB (233k×512×4 bytes) and lives in VRAM
*alongside* the ML models (CLIP, InsightFace, etc.). Two resident matrices
('emb' whole-image + 'all_faces') plus the models is acceptable headroom on the
16 GB Intel Arc this targets, but it is NOT free — keep the resident key set
small and be aware new resident matrices add up against model VRAM.
"""
from __future__ import annotations

import threading

import numpy as np

from . import compute

# key -> (version_int, tensor_on_device). Process-global on purpose: the web
# server runs single-process (uvicorn --workers 1), so one resident copy is
# shared by every request thread — the whole point is to upload once. Guarded by
# _LOCK because FastAPI's threadpool calls these concurrently and a rebuild
# mutates the dict.
_RESIDENT: dict = {}
_LOCK = threading.Lock()


def _resolve_device():
    """Return the torch device string to run on ('cuda'|'xpu'|'mps'), or ``None``
    to mean "no GPU — caller keeps its numpy path".

    torch not being importable is the documented CPU-passthrough case, so an
    ImportError here maps to ``None`` (not an error). A misconfiguration that
    :func:`compute.torch_device` deliberately raises on (e.g. ``MEDIA_DEVICE=cuda``
    with no CUDA) is NOT swallowed — it propagates, per "no silent failures"."""
    try:
        dev = compute.torch_device()
    except ImportError:
        # torch isn't installed at all → genuinely no GPU. Passthrough.
        return None
    return None if dev == "cpu" else dev


def _resident_tensor(key, version, matrix, device, torch):
    """Fetch (or (re)build) the resident device tensor for ``key`` at ``version``.

    Rebuilds when the cached version differs, replacing the entry so the old
    tensor loses its last reference and its VRAM is reclaimed on GC. Prints a
    loud one-liner on every (re)build so residency changes are visible in logs.
    ``matrix`` is forced float32 + C-contiguous because ``torch.from_numpy`` is a
    zero-copy view and a non-contiguous / wrong-dtype source would otherwise error
    or silently misread bytes."""
    m = np.ascontiguousarray(matrix, dtype=np.float32)
    with _LOCK:
        cached = _RESIDENT.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
        tensor = torch.from_numpy(m).to(device)
        _RESIDENT[key] = (version, tensor)
        d = m.shape[1] if m.ndim > 1 else 0
        print(f"[gpu_search] resident {key!r} N={m.shape[0]} D={d} on {device}")
        return tensor


def matvec(key, version, matrix, query):
    """Score a resident ``[N, D]`` matrix against a ``[D]`` query on the GPU:
    ``matrix @ query -> [N]``, returned as a float32 numpy array.

    On a GPU: ensures the matrix is resident for (key, version) — uploading it
    once and reusing it across calls — then uploads only the query and computes
    ``resident @ q``. Returns ``None`` when there's no GPU (caller falls back to
    numpy) or when the matrix is empty (nothing to make resident; the caller's
    numpy path handles the empty case identically)."""
    device = _resolve_device()
    if device is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return None
    import torch

    resident = _resident_tensor(key, version, matrix, device, torch)
    q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32)).to(device)
    scores = resident @ q
    return scores.detach().to("cpu").numpy().astype(np.float32, copy=False)


def gemm(a, b):
    """One-shot (NOT cached) matmul ``[N, D] @ [D, K] -> [N, K]`` on the GPU,
    returned as a float32 numpy array.

    For the offline batch face-suggestion scan, which walks the whole faces table
    in chunks against the named-reference matrix. That's a whole-run job, so the
    per-call upload of both operands is amortized over the entire scan and there's
    nothing worth keeping resident — hence no key/version, unlike :func:`matvec`.
    Returns ``None`` on CPU (caller falls back to numpy ``a @ b``) or if either
    operand is empty."""
    device = _resolve_device()
    if device is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return None
    import torch

    ta = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    tb = torch.from_numpy(np.ascontiguousarray(b)).to(device)
    out = ta @ tb
    return out.detach().to("cpu").numpy().astype(np.float32, copy=False)
