"""Shared embedding centroid/ranking math — extracted so both media.py's
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


def fuse_zscores(parts):
    """Rank-fusion of several embedding spaces by weighted mean of z-scored cosines.

    parts: [(file_ids int64[n], checksums[n], scores float32[n], weight)], one per space
    (spaces may cover different files). Each space's scores are z-scored over its own
    whole matrix, so models with differently-shaped cosine distributions (DINOv2 vs
    SigLIP 2) are commensurable; a file embedded in only some spaces is scored by the
    weight-renormalised mean of the ones it has. Returns (file_ids, checksums, rank,
    display) over the union of files: `rank` is the fused z (order by this), `display` the
    same weighted mean of the RAW cosines (a 0..1-ish number a card can show)."""
    parts = [p for p in parts if p[2].shape[0] > 0]
    if not parts:
        return (np.empty(0, np.int64), [], np.empty(0, np.float32), np.empty(0, np.float32))
    all_ids = np.concatenate([np.asarray(p[0], dtype=np.int64) for p in parts])
    union, first = np.unique(all_ids, return_index=True)
    all_cs = [c for p in parts for c in p[1]]
    checksums = [all_cs[i] for i in first]
    num = np.zeros(len(union), np.float64)
    raw = np.zeros(len(union), np.float64)
    den = np.zeros(len(union), np.float64)
    for ids, _cs, sc, w in parts:
        sc = np.asarray(sc, dtype=np.float64)
        z = (sc - sc.mean()) / max(float(sc.std()), 1e-6)
        pos = np.searchsorted(union, np.asarray(ids, dtype=np.int64))
        np.add.at(num, pos, w * z)
        np.add.at(raw, pos, w * sc)
        np.add.at(den, pos, w)
    return union, checksums, (num / den).astype(np.float32), (raw / den).astype(np.float32)
