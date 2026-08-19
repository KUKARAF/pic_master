"""Region search via CLIP "tiles": embed a deterministic overlapping multi-scale
grid of crops per image so a query can match a *part* of a photo, not just the
whole frame. Unlike body_index.py this needs no object detector — the tiles are
pure geometry (fixed grid), so every image gets the same coverage regardless of
what's in it. Web-only corpus build, same shape as build_body_index: iterate the
un-tiled files, CLIP-embed each image's crops, store them through the DB layer,
and leave failures un-tiled so a rebuild retries them.

The whole image is always included as one tile, so region search still works when
the query is essentially the entire frame (i.e. it degrades gracefully to plain
whole-image CLIP search).
"""
import os

from PIL import Image

from .formats import IMAGE_EXTENSIONS

# Tiles smaller than this on either side embed to noise, so drop them (matches the
# degenerate-crop guard in body_index.crop_bodies).
MIN_TILE_PX = 8

# Grid scales to lay down on top of the whole-image tile: a 2x2 and a 3x3, each
# with ~50% overlap so an object straddling a cell boundary still lands wholly
# inside some neighbouring tile.
GRID_SCALES = (2, 3)


def generate_tiles(width, height):
    """Deterministic overlapping multi-scale grid of (x1, y1, x2, y2) crops in
    ORIGINAL pixel coords.

    Scheme:
      - Tile 1 is always the WHOLE image (0, 0, width, height).
      - For each grid n in {2, 3} we lay down an n x n grid of ~50%-overlapping
        windows. Each tile is `dim * 2/(n+1)` on a side (so a 2x2 gives half-size
        tiles, a 3x3 gives ~half-size tiles too) and the top-left corners step by
        `dim / (n+1)`, i.e. half a tile — a classic 50%-overlap sliding window.
        That yields (n+1-1)=n positions per axis, hence up to n*n tiles per scale.
      - Coords are clamped to [0, width] x [0, height], rounded to ints, and any
        tile under MIN_TILE_PX on a side (or a duplicate of one already emitted) is
        dropped — which is what makes tiny images degrade to just the whole tile.

    For a landscape image this is 1 (whole) + 4 (2x2) + 9 (3x3) = 14 tiles; after
    de-duping the near-identical whole-frame-ish tiles it lands around ~13. Tiny
    images (e.g. 20x20) collapse to mostly just the whole tile once the sub-tiles
    fall under MIN_TILE_PX or dedupe away.
    """
    tiles = []
    seen = set()

    def _add(x1, y1, x2, y2):
        x1 = max(0, min(width, int(round(x1))))
        y1 = max(0, min(height, int(round(y1))))
        x2 = max(0, min(width, int(round(x2))))
        y2 = max(0, min(height, int(round(y2))))
        if x2 - x1 < MIN_TILE_PX or y2 - y1 < MIN_TILE_PX:
            return
        key = (x1, y1, x2, y2)
        if key in seen:
            return
        seen.add(key)
        tiles.append((x1, y1, x2, y2))

    # Whole image first so it's always tile 0.
    _add(0, 0, width, height)

    for n in GRID_SCALES:
        tile_w = width * 2.0 / (n + 1)
        tile_h = height * 2.0 / (n + 1)
        step_x = width / (n + 1)
        step_y = height / (n + 1)
        for row in range(n):
            for col in range(n):
                x1 = col * step_x
                y1 = row * step_y
                _add(x1, y1, x1 + tile_w, y1 + tile_h)

    return tiles


def embed_tiles_for_file(db, clip_indexer, file_id, abs_path, model=None):
    """Open one image, tile it, CLIP-embed the crops, and store the tile rows.

    Returns the number of tiles stored. A degenerate image with no valid tiles
    still stores nothing but doesn't error (generate_tiles always yields the whole
    tile for any image at least MIN_TILE_PX on both sides)."""
    with Image.open(abs_path) as img:
        img = img.convert('RGB')
        boxes = generate_tiles(img.width, img.height)
        crops = [img.crop(box) for box in boxes]
    tiles = []
    if crops:
        embeddings = clip_indexer.embed_pil_images(crops)
        tiles = [{'bbox': list(box), 'embedding': embeddings[i]}
                 for i, box in enumerate(boxes)]
    db.insert_tile_embeddings(file_id, tiles, model or clip_indexer.model_id())
    return len(tiles)


def build_tile_index(db, errors, clip_indexer, data_root, on_progress=None):
    """Tile-index every tracked file that has no tile row yet — the background
    corpus build behind the web UI. Non-image files are counted as processed but
    skipped (get_untiled_files doesn't filter by kind). Failures go to the error
    log (same policy as the batch ML passes); a failed file is left un-tiled so a
    rebuild retries it. Returns (processed, total)."""
    files = db.get_untiled_files()
    total = len(files)
    processed = 0
    for file_id, rel_path in files:
        if os.path.splitext(rel_path)[1].lower() not in IMAGE_EXTENSIONS:
            processed += 1
            if on_progress:
                on_progress(processed, total)
            continue
        try:
            abs_path = os.path.join(data_root, rel_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError('file missing on disk')
            embed_tiles_for_file(db, clip_indexer, file_id, abs_path)
        except Exception as exc:
            errors.log(rel_path, f'tile index: {exc}')
        processed += 1
        if on_progress:
            on_progress(processed, total)
    return processed, total
