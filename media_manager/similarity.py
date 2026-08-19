"""Shared CLIP-embedding centroid/ranking math — extracted so both media.py's
category auto-matching and web.py's set-similarity features use the exact same
computation, without media.py (CLI/core layer) importing web.py."""
import numpy as np


def mean_normalized_centroid(embedding_bytes_list):
    """Mean of a list of embedding byte-blobs, L2-normalized. None if the list
    is empty or the mean is the zero vector."""
    if not embedding_bytes_list:
        return None
    vecs = np.stack([np.frombuffer(e, dtype=np.float32) for e in embedding_bytes_list])
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return None if norm == 0 else centroid / norm


def adjusted_centroid(positive_embeddings, negative_embeddings, negative_weight=0.5):
    """Rocchio-style relevance-feedback centroid: the plain positive centroid
    (mean_normalized_centroid of positive_embeddings), pulled away from the
    mean direction of negative_embeddings when there are any, then
    re-normalized. `negative_weight` controls how hard: 0 reduces to the plain
    positive centroid, 1 subtracts the negative centroid at full strength.
    None if positive_embeddings is empty (mirrors mean_normalized_centroid);
    falls back to the plain positive centroid if negative_embeddings is empty
    or its own mean is the zero vector."""
    positive_centroid = mean_normalized_centroid(positive_embeddings)
    if positive_centroid is None or not negative_embeddings:
        return positive_centroid
    negative_centroid = mean_normalized_centroid(negative_embeddings)
    if negative_centroid is None:
        return positive_centroid
    adjusted = positive_centroid - negative_weight * negative_centroid
    norm = np.linalg.norm(adjusted)
    return positive_centroid if norm == 0 else adjusted / norm


def top_k_indices(scores, k=None):
    """Indices into `scores` (a 1-D np array) for the top-k values, descending.
    Uses argpartition (O(N)) to avoid a full sort when k < N; both argpartition
    and argsort release the GIL, unlike Python's sorted(). k=None (or k>=N)
    returns every index in descending order."""
    n = int(scores.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if k is None or k >= n:
        return np.argsort(scores)[::-1]
    # top-k partition, then sort just those k descending
    part = np.argpartition(scores, n - k)[n - k:]
    return part[np.argsort(scores[part])[::-1]]


def rank_matrix(centroid, matrix, k=None):
    """Dot a pre-built [N, D] float32 matrix against `centroid`, returning
    (indices, scores) for the top-k rows descending. Vectorized end-to-end
    (BLAS dot + argpartition) — the path that keeps the GIL free. Assumes rows
    are already unit-normalized; only the centroid is normalized by callers as
    needed."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    scores = matrix.dot(np.asarray(centroid, dtype=np.float32))
    idx = top_k_indices(scores, k)
    return idx, scores[idx]


def rank_by_similarity(centroid, candidates, embedding_index=2, k=None):
    """candidates: rows containing embedding bytes at embedding_index (matches
    the shape of db.get_all_embeddings()/get_embeddings_for_files() rows).
    Returns [(candidate, score), ...] sorted descending; pass k to get only the
    top-k (argpartition, no full sort). Assumes stored embeddings are already
    unit-normalized — only the centroid is explicitly normalized here.

    The matrix is built in a single allocation (one join + frombuffer) rather
    than a per-row np.stack, and ranking uses numpy sorts (GIL-releasing)
    instead of Python's sorted()."""
    if not candidates:
        return []
    matrix = np.frombuffer(
        b''.join(c[embedding_index] for c in candidates), dtype=np.float32
    ).reshape(len(candidates), -1)
    idx, scores = rank_matrix(centroid, matrix, k=k)
    return [(candidates[i], float(scores[j])) for j, i in enumerate(idx)]
