"""Background-only matching on the CLIP tile index — "same place", ignoring the people.

A whole-image CLIP embedding is dominated by whoever is in the photo, so ranking
"photos probably taken at this location" by it surfaces the same PERSON, not the same
place. The tile index (tile_index.py: whole image + 2x2 + 3x3 overlapping crops per
photo) already has embeddings of parts of each photo; this module keeps only the
BACKGROUND tiles — those that YOLO person boxes (the Auto-tags detections) cover by at
most MAX_PERSON_COVER — and ranks photos by how well their background tiles match the
background tiles of the photos already filed at a location.

Pure numpy over arrays handed in by the caller (no DB access), so it's testable alone:
  person_coverage()   fraction of each tile covered by any person box
  background_mask()   which tile rows are background, per file
  rank_files()        per-photo score = mean of its TOP_K best background tiles, each
                      scored by its best match among the query (location) tiles,
                      optionally pushed down by a better match to rejected tiles
"""
import numpy as np

# A tile with more than this fraction of its area under person boxes isn't background.
MAX_PERSON_COVER = 0.25
# Sample points per tile side when measuring person coverage (16x16 = 256 points).
COVER_GRID = 16
# A photo's score is the mean of its best TOP_K background tiles (fewer if it has fewer)
# — one lucky tile isn't enough to claim "same place", two agreeing tiles are.
TOP_K = 2
# honor_negatives: a tile that matches a rejected photo better than the location loses
# this much of the difference.
NEG_WEIGHT = 0.5
# Cap on query tiles (a location with hundreds of photos): evenly subsampled.
MAX_QUERY_TILES = 4000


def person_coverage(tiles, persons, grid=COVER_GRID):
    """(n,) fraction of each tile (n,4 x1,y1,x2,y2) covered by the UNION of person boxes
    (m,4), estimated on a grid x grid lattice of sample points inside each tile."""
    tiles = np.asarray(tiles, dtype=np.float32).reshape(-1, 4)
    persons = np.asarray(persons, dtype=np.float32).reshape(-1, 4)
    if len(tiles) == 0:
        return np.zeros(0, np.float32)
    if len(persons) == 0:
        return np.zeros(len(tiles), np.float32)
    steps = (np.arange(grid, dtype=np.float32) + 0.5) / grid                 # (g,)
    xs = tiles[:, 0:1] + steps[None, :] * (tiles[:, 2:3] - tiles[:, 0:1])    # (n,g)
    ys = tiles[:, 1:2] + steps[None, :] * (tiles[:, 3:4] - tiles[:, 1:2])    # (n,g)
    px = xs[:, None, :, None]                                                 # (n,1,g,1)
    py = ys[:, None, None, :]                                                 # (n,1,1,g)
    b = persons[None, :, :, None, None]                                       # (1,m,4,1,1)
    inside_x = (px >= b[:, :, 0]) & (px <= b[:, :, 2])                        # (n,m,g,1)
    inside_y = (py >= b[:, :, 1]) & (py <= b[:, :, 3])                        # (n,m,1,g)
    covered = (inside_x & inside_y).any(axis=1)                               # (n,g,g)
    return covered.mean(axis=(1, 2)).astype(np.float32)


def background_mask(file_ids, tile_idx, boxes, persons_by_file, max_cover=MAX_PERSON_COVER):
    """Bool (n,) over tile rows: True = background tile.

    file_ids (n,), tile_idx (n,) (0 = the whole-image tile), boxes (n,4) in the same
    upright-pixel frame as the person boxes; persons_by_file {file_id: [[x1,y1,x2,y2]]}.
    A photo with no person boxes keeps every tile (its whole-image tile included). A
    photo with people never keeps its whole-image tile — that's the person embedding."""
    file_ids = np.asarray(file_ids)
    tile_idx = np.asarray(tile_idx)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    mask = np.ones(len(file_ids), dtype=bool)
    if not persons_by_file or len(file_ids) == 0:
        return mask
    order = np.argsort(file_ids, kind='stable')
    sorted_fids = file_ids[order]
    uniq, starts = np.unique(sorted_fids, return_index=True)
    ends = np.append(starts[1:], len(sorted_fids))
    for fid, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        persons = persons_by_file.get(fid)
        if not persons:
            continue
        rows = order[s:e]
        cover = person_coverage(boxes[rows], persons)
        mask[rows] = (cover <= max_cover) & (tile_idx[rows] != 0)
    return mask


def _l2(m):
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def _cap(m, limit=MAX_QUERY_TILES):
    if len(m) <= limit:
        return m
    return m[np.linspace(0, len(m) - 1, limit).astype(int)]


def rank_files(chunks, query, negatives=None, top_k=TOP_K):
    """Rank photos by background similarity to `query` tiles.

    chunks: iterable of (file_ids (k,), matrix (k,D)) — background tiles only.
    query: (q,D) background tiles of the location's photos; negatives: (r,D) background
    tiles of photos rejected for it, or None. Returns [(file_id, score)] best first."""
    query = _cap(_l2(query))
    neg = _cap(_l2(negatives)) if negatives is not None and len(negatives) else None
    all_fids, all_scores = [], []
    for fids, matrix in chunks:
        if len(fids) == 0:
            continue
        m = _l2(matrix)
        pos = (m @ query.T).max(axis=1)
        if neg is not None:
            nmax = (m @ neg.T).max(axis=1)
            pos = pos - NEG_WEIGHT * np.maximum(0.0, nmax - pos)
        all_fids.append(np.asarray(fids, dtype=np.int64))
        all_scores.append(pos.astype(np.float32))
    if not all_fids:
        return []
    fids = np.concatenate(all_fids)
    scores = np.concatenate(all_scores)
    order = np.lexsort((-scores, fids))          # by file, best tile first
    fids, scores = fids[order], scores[order]
    uniq, starts, counts = np.unique(fids, return_index=True, return_counts=True)
    take = np.minimum(counts, top_k)
    sums = np.zeros(len(uniq), np.float64)
    for j in range(top_k):
        has = take > j
        sums[has] += scores[starts[has] + j]
    per_file = sums / take
    best = np.argsort(-per_file, kind='stable')
    return [(int(uniq[i]), float(per_file[i])) for i in best]
