"""Shared build loop for the location-by-scene (place) index.

For each un-indexed image this computes a people-masked scene descriptor (via the
place encoder) and caches local features (via the place matcher) for the same-spot
re-rank. Both the web bulk job and the `media place-index` CLI drive this one loop;
callers supply the person-box lookup, path resolution, progress callback, and error
log so the loop itself stays free of heavy/web-only dependencies.

Performance: images are decoded ONCE (JPEG draft mode, near the working resolution)
in a thread pool, then fed in BATCHES to the encoder + matcher so the GPU does many
images per forward instead of one at a time. The expensive per-image CPU work (decode)
overlaps across cores; the GPU work is batched. Keypoint extraction is always done —
it's a core feature, just done efficiently (XFeat batches on GPU; SIFT loops on CPU).

SQLite safety: all DB access (person boxes, inserts) happens on the CALLER's thread.
The decode workers touch only PIL — never the db — because sqlite connections are
per-thread.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _decode_draft(abs_path, long_side):
    """Decode an image near `long_side` on its long edge, cheaply. For JPEGs this uses
    PIL draft mode (decode at 1/2, 1/4, 1/8 instead of full res), then a final resize to
    land at ~long_side. Returns (PIL.Image RGB, (sx, sy)) where sx/sy map ORIGINAL-pixel
    coordinates to the decoded image (so person boxes can be scaled to match), or
    (None, (1.0, 1.0)) if it couldn't be decoded."""
    from PIL import Image
    try:
        im = Image.open(abs_path)
        ow, oh = im.size
        try:
            im.draft('RGB', (long_side, long_side))  # JPEG-only; coarse 1/2^n decode
        except Exception:
            pass
        im = im.convert('RGB')
        w, h = im.size
        m = max(w, h)
        if m > long_side:  # draft lands on powers of two; trim to the target
            r = long_side / float(m)
            im = im.resize((max(1, int(round(w * r))), max(1, int(round(h * r)))))
        dw, dh = im.size
        return im, ((dw / ow) if ow else 1.0, (dh / oh) if oh else 1.0)
    except Exception:
        return None, (1.0, 1.0)


def _scale_boxes(boxes, scale):
    sx, sy = scale
    out = []
    for b in boxes or []:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        out.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out


def build_place_index(db, encoder, matcher, data_root, *, image_exts,
                      person_boxes_fn, abs_path_fn=None, exclude_ids=None,
                      candidates=None, on_progress=None, log=None):
    """Encode + cache features for every image lacking a place index row.

    Decodes once, batches the encoder/matcher. `encoder` must expose model_id() and
    either embed_batch(pils, person_boxes) (preferred, batched, returns a list aligned
    1:1 with inputs of float32 vectors or None) or embed_images(paths, person_boxes).
    `matcher` (or None) must expose model_id(), serialize(), and either
    extract_batch(images, person_boxes) (aligned list of feature-dicts/None) or
    extract(image, person_boxes). person_boxes_fn(file_id) returns ORIGINAL-pixel boxes.

    Returns the number of candidates processed.
    """
    if candidates is None:
        excl = exclude_ids or set()
        candidates = [
            (fid, rel) for (fid, rel) in db.get_unplace_indexed_files()
            if os.path.splitext(rel)[1].lower() in image_exts and fid not in excl
        ]
    total = len(candidates)
    if on_progress:
        on_progress(0, total)
    enc_model = encoder.model_id()
    mat_model = matcher.model_id() if matcher is not None else None

    batch = max(1, int(os.environ.get('MEDIA_PLACE_BATCH', '16') or 16))
    workers = int(os.environ.get('MEDIA_PLACE_DECODE_WORKERS', '') or min(8, (os.cpu_count() or 4)))
    long_side = max(256, int(os.environ.get('MEDIA_PLACE_DECODE_PX', '1024') or 1024))
    use_batch_embed = hasattr(encoder, 'embed_batch')
    use_batch_extract = matcher is not None and hasattr(matcher, 'extract_batch')

    def _resolve(fid, rel):
        if abs_path_fn is not None:
            return abs_path_fn(fid, rel)
        p = os.path.join(data_root, rel)
        return p if os.path.isfile(p) else None

    def _write_embedding(fid, rel, vec):
        if vec is None:
            db.insert_place_embedding(fid, b'', enc_model)  # sentinel — don't re-queue
            if log:
                log(rel, 'place embed failed')
        else:
            db.insert_place_embedding(fid, np.asarray(vec, dtype=np.float32).tobytes(), enc_model)

    def _write_keypoints(fid, rel, feat):
        if matcher is None:
            return
        if feat is None:
            if log:
                log(rel, 'place keypoints failed')
            return
        try:
            db.insert_place_keypoints(fid, matcher.serialize(feat), mat_model)
        except Exception as exc:
            if log:
                log(rel, f'place keypoints serialize error: {exc}')

    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for i in range(0, total, batch):
            chunk = candidates[i:i + batch]
            # --- DB + path work on THIS thread (sqlite is per-thread) ---
            prepped = []  # (fid, rel, abs_path_or_None, original_boxes)
            for fid, rel in chunk:
                ap = _resolve(fid, rel)
                boxes = person_boxes_fn(fid) if ap is not None else []
                prepped.append((fid, rel, ap, boxes))

            # --- parallel decode (PIL only, no DB) ---
            def _decode(item):
                fid, rel, ap, boxes = item
                if ap is None:
                    return (fid, rel, 'nopath', None, None)
                im, scale = _decode_draft(ap, long_side)
                if im is None:
                    return (fid, rel, 'decodefail', None, None)
                return (fid, rel, 'ok', im, _scale_boxes(boxes, scale))

            decoded = list(pool.map(_decode, prepped))

            ok_pos = [j for j, d in enumerate(decoded) if d[2] == 'ok']
            pils = [decoded[j][3] for j in ok_pos]
            boxes_list = [decoded[j][4] for j in ok_pos]

            # --- batched (or fallback) embed + extract for the decoded images ---
            if pils and use_batch_embed:
                embs = encoder.embed_batch(pils, boxes_list)
            elif pils:
                embs = []
                for im, bx in zip(pils, boxes_list):
                    try:
                        e, failed = encoder.embed_pil_images([im], [bx])
                        embs.append(e[0] if (e and not failed) else None)
                    except Exception:
                        embs.append(None)
            else:
                embs = []

            if pils and use_batch_extract:
                feats = matcher.extract_batch(pils, boxes_list)
            elif pils and matcher is not None:
                feats = []
                for im, bx in zip(pils, boxes_list):
                    try:
                        feats.append(matcher.extract(im, person_boxes=bx))
                    except Exception:
                        feats.append(None)
            else:
                feats = [None] * len(pils)

            emb_by_pos = {ok_pos[k]: embs[k] for k in range(len(ok_pos))}
            feat_by_pos = {ok_pos[k]: feats[k] for k in range(len(ok_pos))}

            # --- writes on THIS thread ---
            for j, (fid, rel, status, _im, _bx) in enumerate(decoded):
                if status == 'nopath':
                    pass  # file offline/moved — skip without a sentinel so it retries later
                elif status == 'decodefail':
                    db.insert_place_embedding(fid, b'', enc_model)  # corrupt — sentinel
                    if log:
                        log(rel, 'place decode failed')
                else:
                    _write_embedding(fid, rel, emb_by_pos.get(j))
                    _write_keypoints(fid, rel, feat_by_pos.get(j))
                done += 1
            if on_progress:
                on_progress(done, total)
    finally:
        pool.shutdown(wait=True)
    return total
