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
