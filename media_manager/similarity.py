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


def rank_by_similarity(centroid, candidates, embedding_index=2):
    """candidates: rows containing embedding bytes at embedding_index (matches
    the shape of db.get_all_embeddings()/get_embeddings_for_files() rows).
    Returns [(candidate, score), ...] sorted descending. Assumes stored
    embeddings are already unit-normalized — only the centroid is explicitly
    normalized here."""
    if not candidates:
        return []
    matrix = np.stack([np.frombuffer(c[embedding_index], dtype=np.float32) for c in candidates])
    scores = matrix.dot(centroid)
    return sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
