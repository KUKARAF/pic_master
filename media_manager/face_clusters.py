"""Find the biggest groups of mutually similar faces -- GPU ONLY (torch cuda/xpu).

Used by the /find_all_faces "face group review": over every not-yet-named face,
surface the largest sets of faces that look like the same person so the user can
name/approve a whole set at once.

NO CPU PATH. `find_face_groups` resolves `compute.torch_device()`; anything but
cuda/xpu raises `GpuUnavailableError` (the web job shows it as the status `error`).
The only way to run on another device is the explicit `device=` argument, a TEST HOOK
(the unit tests pass 'cpu'); no production call site passes it and nothing picks it
automatically.

Algorithm (greedy, density-seeded, centroid-checked; NOT single linkage)
  0. UPLOAD ONCE. The N x D matrix goes to the device a single time and stays resident
     for the whole job (fp32; see PRECISION) and is freed in a `finally`.
  1. DENSITY PASS (on device). Tiled all-pairs similarity (R x C blocks, one matmul
     each). The `>= threshold` comparison and the row/column sums run on the device
     and accumulate into an int32[N] counts tensor; a similarity block is NEVER copied
     to the host (only `counts`, N ints, comes back once). Symmetric, so only the
     upper triangle of tiles is multiplied and each tile feeds both sides' counters.
  2. GROUP LOOP (on device). Seeds come from a host max-heap of (count, index) built
     from the counts. Lazy greedy: a popped seed's count is only an upper bound, so it
     is recomputed against the still-free pool (one N x D matvec against the resident
     matrix; the pool is a boolean mask on the device) and re-pushed if it fell below
     the next candidate. The seed's neighbours (cos >= threshold to the SEED) form the
     first member set; the centroid (re-normalised mean, computed as mask @ X, no
     gather copy) is refined: re-score every free face against the centroid, keep
     those >= threshold, recompute, until the member set stops changing.
  3. INVARIANT. The returned centroid is exactly the one the final membership test was
     run against, so EVERY member has cosine >= threshold to the group centroid
     (`scores` are those cosines). With centroid-similarity >= t two members are at
     least 2t^2-1 similar, so a chain a~b~c~d (single linkage) cannot form.
  4. Members leave the pool; repeat until the best remaining density cannot reach
     `min_size` (or `max_groups`). Only small index/score arrays (one group at a time)
     and the D-float centroid cross back to the host. Output sorted by size desc.

PRECISION. fp32 resident, fp32 matmul (torch's default 'highest' float32 matmul
precision; no TF32/bf16). Threshold decisions near 0.55 therefore need no second
verification pass, and the host-visible `scores` are the very values compared with the
threshold. fp16/bf16 would use the matrix engines (several x faster) but has ~1e-3
error in a cosine, which flips borderline pairs; the whole point of fp32 is that 32 GB
makes it affordable.

VRAM / TIME (N faces, D dims; B70 = 32 GB). Resident matrix N*D*4 bytes (1M x 512 = 2.0
GiB). A tile of edge T needs ~5*T^2 bytes (fp32 scores + bool mask) plus a headroom
factor; `pick_tile_size` takes at most 60% of what is FREE after the matrix and caps the
edge at 16384 (1.3 GiB score tile): that is the answer for anything from 100k to several
M faces on a 32 GB card, and a smaller card just gets smaller tiles. Density pass flops
= N^2*D (1M x 512 -> 5e14: tens of seconds on a ~20 TFLOPS fp32 part); the group loop
is G * iters * (2 matvecs of N*D). Intel Arc has a 4 GiB single-allocation limit unless
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1: a matrix bigger than 3.5 GiB (~1.8M faces at
D=512) is refused with a clear error instead of an opaque allocator failure.

Deterministic: ties break on array index / face id; no randomness.
"""
from __future__ import annotations

import gc
import heapq
import math
import os
from typing import Callable, Optional, Sequence

import numpy as np

GPU_DEVICES = ('cuda', 'xpu')
GPU_REQUIRED_MESSAGE = ('face grouping needs a GPU (xpu/cuda); none available - set MEDIA_DEVICE '
                        'or install the XPU torch build')
MAX_TILE = 16384
MIN_TILE = 256
TILE_VRAM_FRACTION = 0.6        # of the VRAM still free AFTER the matrix is resident
TILE_BYTES_PER_CELL = 5 * 1.25  # fp32 score + bool mask, +25% allocator/temporary headroom
PER_FACE_SCRATCH = 32           # counts, free mask, matvec result, ... per face (bytes)
SINGLE_ALLOC_LIMIT = int(3.5 * 2 ** 30)   # Arc/Level-Zero 4 GiB per-allocation cap
SYNC_EVERY_TILES = 8            # bound the async queue so cancel/progress stay real


class GroupSearchCancelled(Exception):
    """Raised by find_face_groups when should_stop() turned true."""


class GpuUnavailableError(RuntimeError):
    """Face grouping is GPU-only and no usable cuda/xpu device exists."""


class GpuMemoryError(GpuUnavailableError):
    """The GPU is present but cannot hold the matrix plus a minimal tile."""


def resolve_device(device: Optional[str] = None) -> str:
    """The device grouping runs on. `device` is a TEST HOOK (returned as-is); without it
    only compute.torch_device() in ('cuda', 'xpu') is accepted -- cpu/mps/no torch raise
    GpuUnavailableError."""
    if device:
        return str(device)
    try:
        from media_manager import compute  # hook-allow: lazy - pulls torch on first use
        dev = compute.torch_device()
    except ImportError as exc:
        raise GpuUnavailableError(GPU_REQUIRED_MESSAGE + ' (torch is not installed)') from exc
    except (RuntimeError, ValueError) as exc:
        raise GpuUnavailableError('%s (%s)' % (GPU_REQUIRED_MESSAGE, exc)) from exc
    if dev not in GPU_DEVICES:
        raise GpuUnavailableError('%s (torch device is %r)' % (GPU_REQUIRED_MESSAGE, dev))
    return dev


def pick_tile_size(n: int, dim: int, free_bytes: Optional[int], max_tile: int = MAX_TILE) -> int:
    """Tile edge for the density pass from the VRAM free BEFORE the matrix upload.
    Takes TILE_VRAM_FRACTION of what is left after the resident matrix and per-face
    scratch, ~TILE_BYTES_PER_CELL per tile cell, rounded down to a multiple of 256,
    clamped to [MIN_TILE, max_tile] (and never above n). Unknown free memory -> 4096.
    Raises GpuMemoryError when not even MIN_TILE fits."""
    n = max(1, int(n))
    if free_bytes is None:
        return max(1, min(n, 4096, max_tile))
    after = int(free_bytes) - n * int(dim) * 4 - n * PER_FACE_SCRATCH
    edge = int(math.sqrt(max(0.0, after * TILE_VRAM_FRACTION) / TILE_BYTES_PER_CELL))
    if edge < min(MIN_TILE, n):
        raise GpuMemoryError(
            'face grouping needs a GPU with more free memory: %d faces x %d dims need %.2f GiB '
            'resident plus a tile, only %.2f GiB free' % (
                n, dim, n * dim * 4 / 2 ** 30, free_bytes / 2 ** 30))
    edge = (edge // MIN_TILE) * MIN_TILE if edge >= MIN_TILE else edge
    return max(1, min(edge, max_tile, n))


def _torch():
    import torch  # hook-allow: lazy - heavy dep, GPU-only module
    return torch


def _free_bytes(device: str) -> Optional[int]:
    torch = _torch()
    mod = getattr(torch, torch.device(device).type, None)
    if mod is None:
        return None
    try:
        return int(mod.mem_get_info()[0])
    except Exception:  # hook-allow: broad - older torch/driver; try the property fallback
        pass
    try:
        total = mod.get_device_properties(0).total_memory
        return max(0, int(total * 0.8) - int(mod.memory_allocated()))
    except Exception:  # hook-allow: broad - unknown budget -> conservative default tile
        return None


def _sync(device: str) -> None:
    torch = _torch()
    mod = getattr(torch, torch.device(device).type, None)
    if mod is not None and hasattr(mod, 'synchronize'):
        mod.synchronize()


def _release(device: str) -> None:
    gc.collect()
    torch = _torch()
    mod = getattr(torch, torch.device(device).type, None)
    if mod is not None and hasattr(mod, 'empty_cache'):
        mod.empty_cache()


def _to_device(arr: np.ndarray, device: str):
    """Chunked upload straight into one preallocated device tensor."""
    torch = _torch()
    out = torch.empty(arr.shape, dtype=torch.float32, device=device)
    step = 65536
    for i in range(0, arr.shape[0], step):
        out[i:i + step] = torch.from_numpy(arr[i:i + step]).to(device)
    return out


def _to_host(t) -> np.ndarray:
    """The ONLY device->host path. Everything crossing it is O(N) or smaller (counts,
    one group's indices/scores, a D-float centroid); tests assert that."""
    return t.detach().cpu().numpy()


def _as_unit_matrix(embeddings) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if x.ndim != 2:
        raise ValueError('embeddings must be a 2-D (N, D) array')
    return x


def _neighbour_counts(xd, threshold, tile, device, should_stop, on_tile):
    """counts[i] = #{j != i : x_i . x_j >= threshold}, all on the device (int32[N])."""
    torch = _torch()
    n = xd.shape[0]
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    starts = list(range(0, n, tile))
    k = 0
    for bi, i0 in enumerate(starts):
        i1 = min(n, i0 + tile)
        a = xd[i0:i1]
        for j0 in starts[bi:]:
            if should_stop():
                raise GroupSearchCancelled()
            j1 = min(n, j0 + tile)
            hit = (a @ xd[j0:j1].T) >= threshold          # bool (R, C), stays on device
            if j0 == i0:
                # Diagonal tile holds every ordered pair of its block: row sums are
                # complete; subtract the self match.
                counts[i0:i1] += hit.sum(dim=1, dtype=torch.int32) - 1
            else:
                counts[i0:i1] += hit.sum(dim=1, dtype=torch.int32)
                counts[j0:j1] += hit.sum(dim=0, dtype=torch.int32)
            del hit
            k += 1
            if k % SYNC_EVERY_TILES == 0:
                _sync(device)
            on_tile()
    _sync(device)
    return counts


def find_face_groups(face_ids: Sequence[int], embeddings, threshold: float = 0.55,
                     min_size: int = 3, max_groups: Optional[int] = None,
                     progress: Optional[Callable[[int, int], None]] = None,
                     should_stop: Optional[Callable[[], bool]] = None,
                     refine_iters: int = 6, tile: Optional[int] = None,
                     device: Optional[str] = None) -> list[dict]:
    """Groups of faces whose every member is >= `threshold` cosine-similar to the
    group's centroid. GPU only; see the module docstring.

    face_ids      N ids, aligned with the rows of `embeddings`
    embeddings    (N, D) float32, L2-normalised rows (dot product == cosine)
    min_size      smallest group returned (>= 2)
    max_groups    stop after this many groups (they are found roughly biggest-first)
    progress      progress(done, total): tiles first, then faces consumed
    should_stop   polled between tiles / groups; True raises GroupSearchCancelled
    tile          tile edge override; default is picked from the device's free VRAM
    device        TEST HOOK ONLY (e.g. 'cpu'); default resolves to cuda/xpu or raises
                  GpuUnavailableError. Production callers must not pass it.

    Returns [{'face_ids': [...], 'centroid': float32[D], 'scores': [...]}], sorted by
    size desc (ties: higher mean score, then lowest face id), members by score desc.
    """
    device = resolve_device(device)
    x = _as_unit_matrix(embeddings)
    n = x.shape[0]
    if len(face_ids) != n:
        raise ValueError('face_ids and embeddings differ in length')
    min_size = max(2, int(min_size))
    stop = should_stop or (lambda: False)
    ids = np.asarray(list(face_ids), dtype=np.int64)
    if n < min_size:
        return []
    dim = x.shape[1]
    torch = _torch()
    dev_type = torch.device(device).type
    if (dev_type == 'xpu' and n * dim * 4 > SINGLE_ALLOC_LIMIT
            and not os.environ.get('UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS')):
        raise GpuMemoryError(
            'face matrix (%.1f GiB) exceeds the Intel GPU single-allocation limit; set '
            'UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 for the web process' % (n * dim * 4 / 2 ** 30))
    chunk = max(1, int(tile)) if tile else pick_tile_size(n, dim, _free_bytes(device))
    chunk = min(chunk, n)

    n_tiles = (n + chunk - 1) // chunk
    n_tiles = n_tiles * (n_tiles + 1) // 2
    total = n_tiles + n
    state = {'done': 0}

    def _report(done):
        if progress:
            progress(min(done, total), total)

    def _tile_done():
        state['done'] += 1
        _report(state['done'])

    xd = free = None
    try:
        xd = _to_device(x, device)
        del x
        _report(0)
        counts = _to_host(_neighbour_counts(xd, threshold, chunk, device, stop, _tile_done))
        free = torch.ones(n, dtype=torch.bool, device=device)
        heap = [(-int(c), i) for i, c in enumerate(counts) if c + 1 >= min_size]
        del counts
        heapq.heapify(heap)
        groups: list[dict] = []
        consumed = 0

        def _free_hits(vec):
            s = xd @ vec
            return (s >= threshold) & free, s

        while heap:
            if stop():
                raise GroupSearchCancelled()
            if max_groups is not None and len(groups) >= max_groups:
                break
            neg, seed = heapq.heappop(heap)
            if not bool(free[seed]):
                continue
            mask, _ = _free_hits(xd[seed])
            cur = int(mask.sum())                  # includes the seed itself
            if cur < min_size:
                continue                           # cannot seed a group any more
            if heap and cur < -heap[0][0] + 1:
                heapq.heappush(heap, (-(cur - 1), seed))   # stale bound: denser seed first
                continue

            centroid = xd[seed]
            members = mask
            for _ in range(max(1, refine_iters)):
                c = members.to(torch.float32) @ xd     # sum of member rows, no gather copy
                norm = float(torch.linalg.vector_norm(c))
                if norm == 0.0:
                    break
                centroid = c / norm
                new_members, _s = _free_hits(centroid)
                if torch.equal(new_members, members):
                    break
                members = new_members
                if int(members.sum()) < min_size:
                    break
            # Re-test once against the final centroid so member . centroid >= t holds exactly.
            members, scores = _free_hits(centroid)
            idx_d = torch.nonzero(members).squeeze(1)
            if idx_d.numel() < min_size:
                continue                           # seed stays in the pool for other groups
            idx = _to_host(idx_d)
            sc = _to_host(scores[idx_d])
            order = np.lexsort((ids[idx], -sc))
            idx, sc = idx[order], sc[order]
            groups.append({'face_ids': [int(v) for v in ids[idx]],
                           'centroid': _to_host(centroid).astype(np.float32),
                           'scores': [float(v) for v in sc]})
            free[idx_d] = False
            consumed += idx.size
            _report(n_tiles + consumed)
    finally:
        del xd, free
        _release(device)

    groups.sort(key=lambda g: (-len(g['face_ids']), -float(np.mean(g['scores'])),
                               g['face_ids'][0]))
    _report(total)
    return groups
