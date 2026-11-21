"""Find the biggest groups of mutually similar faces (pure numpy; no web/db imports).

Used by the /find_all_faces "face group review": over every not-yet-named face,
surface the largest sets of faces that look like the same person so the user can
name/approve a whole set at once.

Algorithm (greedy, density-seeded, centroid-checked; NOT single linkage)
  1. DENSITY PASS. Tiled all-pairs similarity (R x C blocks, each block one matmul,
     through `gemm` when given -- the GPU path -- else BLAS). Per face we only keep
     the COUNT of other faces with cosine >= threshold. Symmetric, so only the upper
     triangle of tiles is multiplied and each tile feeds both sides' counters.
     Memory is one tile (chunk x chunk floats, 64 MB at the default 4096), never N x N.
  2. GROUP LOOP. Seeds come from a max-heap of (count, index). Lazy greedy: a popped
     seed's count is only an upper bound (members may have been taken since), so it is
     recomputed against the still-free pool (one N x D matvec) and re-pushed if it
     fell below the next candidate. The seed's neighbours (cos >= threshold to the
     SEED) form the first member set; the centroid (re-normalised mean) is then
     refined: re-score every free face against the centroid, keep those >= threshold,
     recompute, until the member set stops changing (max `refine_iters` rounds).
  3. INVARIANT. The returned centroid is exactly the one the final membership test was
     run against, so EVERY member has cosine >= threshold to the group centroid
     (`scores` are those cosines). Members are not required to be similar to each
     other pairwise beyond what the triangle inequality on the sphere gives: with
     centroid-similarity >= t, two members are at least 2t^2-1 similar, which is why
     a chain a~b~c~d (single-linkage) cannot form: the far end would drop below
     threshold against the centroid and be shed.
  4. Members are removed from the pool; repeat until the best remaining density cannot
     reach `min_size` (or `max_groups` groups exist). Output is sorted by size desc.

Complexity for N faces, D dims: density pass ~ N^2*D/2 flops (227k x 512 -> ~1.3e13:
roughly 1-3 minutes on a multi-core CPU, seconds on a GPU); group loop ~ G * iters *
N*D (G groups; each iteration is one matvec, ~30 ms at 227k x 512). Peak memory: the
N x D float32 matrix (465 MB at 227k x 512) plus one tile.

Deterministic: ties break on array index; no randomness.
"""
from __future__ import annotations

import heapq
from typing import Callable, Optional, Sequence

import numpy as np


class GroupSearchCancelled(Exception):
    """Raised by find_face_groups when should_stop() turned true."""


def _as_unit_matrix(embeddings) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if x.ndim != 2:
        raise ValueError('embeddings must be a 2-D (N, D) array')
    return x


def _tile_scores(a: np.ndarray, b: np.ndarray, gemm) -> np.ndarray:
    """a (R,D) @ b.T (D,C) -> (R,C) float32, GPU when `gemm` returns a result."""
    if gemm is not None:
        out = gemm(a, np.ascontiguousarray(b.T))
        if out is not None:
            return out
    return a @ b.T


def _neighbour_counts(x, threshold, gemm, chunk, should_stop, on_tile):
    """counts[i] = #{j != i : x_i . x_j >= threshold}, via upper-triangle tiles."""
    n = x.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    starts = list(range(0, n, chunk))
    for bi, i0 in enumerate(starts):
        i1 = min(n, i0 + chunk)
        a = x[i0:i1]
        for j0 in starts[bi:]:
            if should_stop():
                raise GroupSearchCancelled()
            j1 = min(n, j0 + chunk)
            hit = _tile_scores(a, x[j0:j1], gemm) >= threshold
            if j0 == i0:
                # A diagonal tile holds every ordered pair of its block, so row sums
                # alone are complete; subtract the self match.
                counts[i0:i1] += hit.sum(axis=1) - 1
            else:
                counts[i0:i1] += hit.sum(axis=1)
                counts[j0:j1] += hit.sum(axis=0)
            on_tile()
    return counts


def find_face_groups(face_ids: Sequence[int], embeddings, threshold: float = 0.55,
                     min_size: int = 3, max_groups: Optional[int] = None,
                     gemm: Optional[Callable] = None, chunk: int = 4096,
                     progress: Optional[Callable[[int, int], None]] = None,
                     should_stop: Optional[Callable[[], bool]] = None,
                     refine_iters: int = 6) -> list[dict]:
    """Groups of faces whose every member is >= `threshold` cosine-similar to the
    group's centroid. See the module docstring for the algorithm.

    face_ids      N ids, aligned with the rows of `embeddings`
    embeddings    (N, D) float32, L2-normalised rows (dot product == cosine)
    min_size      smallest group returned (>= 2)
    max_groups    stop after this many groups (they are found roughly biggest-first)
    gemm          optional accelerated matmul (a (R,D), b (D,C)) -> (R,C) | None
    chunk         tile edge for the density pass (peak scratch = chunk^2 * 4 bytes)
    progress      progress(done, total): tiles first, then faces consumed
    should_stop   polled between tiles / groups; True raises GroupSearchCancelled

    Returns [{'face_ids': [...], 'centroid': float32[D], 'scores': [...]}], sorted by
    size desc (ties: higher mean score, then lowest face id), members by score desc.
    """
    x = _as_unit_matrix(embeddings)
    n = x.shape[0]
    if len(face_ids) != n:
        raise ValueError('face_ids and embeddings differ in length')
    min_size = max(2, int(min_size))
    chunk = max(1, int(chunk))
    stop = should_stop or (lambda: False)
    ids = np.asarray(list(face_ids), dtype=np.int64)
    if n < min_size:
        return []

    n_tiles = ((n + chunk - 1) // chunk)
    n_tiles = n_tiles * (n_tiles + 1) // 2
    total = n_tiles + n
    state = {'done': 0}

    def _report(done):
        if progress:
            progress(min(done, total), total)

    def _tile_done():
        state['done'] += 1
        _report(state['done'])

    _report(0)
    counts = _neighbour_counts(x, threshold, gemm, chunk, stop, _tile_done)

    free = np.ones(n, dtype=bool)      # still in the pool
    heap = [(-int(c), i) for i, c in enumerate(counts) if c + 1 >= min_size]
    heapq.heapify(heap)
    groups: list[dict] = []
    consumed = 0

    def _free_hits(vec):
        """Boolean mask of free faces with cos(vec) >= threshold, and all scores."""
        s = x @ vec
        return (s >= threshold) & free, s

    while heap:
        if stop():
            raise GroupSearchCancelled()
        if max_groups is not None and len(groups) >= max_groups:
            break
        neg, seed = heapq.heappop(heap)
        if not free[seed]:
            continue
        mask, _ = _free_hits(x[seed])
        cur = int(mask.sum())                      # includes the seed itself
        if cur < min_size:
            continue                               # cannot seed a group any more
        if heap and cur < -heap[0][0] + 1:
            heapq.heappush(heap, (-(cur - 1), seed))   # stale bound: let a denser seed go first
            continue

        centroid = x[seed]
        members = mask
        scores = None
        for _ in range(max(1, refine_iters)):
            c = x[members].mean(axis=0)
            norm = float(np.linalg.norm(c))
            if norm == 0.0:
                break
            centroid = (c / norm).astype(np.float32)
            new_members, scores = _free_hits(centroid)
            if np.array_equal(new_members, members):
                break
            members = new_members
            if int(members.sum()) < min_size:
                break
        # members was last tested against `centroid` only if the loop converged or ran
        # out: re-test once so the invariant (member . centroid >= t) holds exactly.
        members, scores = _free_hits(centroid)
        idx = np.flatnonzero(members)
        if idx.size < min_size:
            continue                               # seed stays in the pool for other groups
        order = np.lexsort((ids[idx], -scores[idx]))
        idx = idx[order]
        groups.append({'face_ids': [int(v) for v in ids[idx]],
                       'centroid': centroid,
                       'scores': [float(v) for v in scores[idx]]})
        free[idx] = False
        consumed += idx.size
        _report(n_tiles + consumed)

    groups.sort(key=lambda g: (-len(g['face_ids']), -float(np.mean(g['scores'])),
                               g['face_ids'][0]))
    _report(total)
    return groups
