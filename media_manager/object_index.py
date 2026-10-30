"""Build the object-region index for instance-level object-exemplar search.

For each un-indexed image this computes DINOv2 region embeddings (an overlapping
multi-scale grid of crops) and stores them with their boxes in ORIGINAL pixels. These
are the library-wide candidate pool a marked query box is matched against.

Decode happens ONCE per image (JPEG draft mode, in a thread pool) and the per-region
embedding is batched by the encoder. NO keypoints are extracted here: geometric
verification (object_matcher) is a query-time step over a small shortlist only.

SQLite safety: all DB access runs on the CALLER's thread; decode workers touch only PIL.

Embed failures: only a per-image DATA error (ValueError — e.g. a degenerate image)
writes the empty-blob sentinel that marks the file indexed. Anything else (GPU/runtime
error, OOM, a bug) is logged and the file is left un-indexed for a retry; after
MAX_CONSECUTIVE_EMBED_FAILURES of those in a row the build raises, so the job status
shows the error instead of grinding through the library.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAX_CONSECUTIVE_EMBED_FAILURES = 5


def _decode_draft(abs_path, long_side):
    """Decode an image near `long_side` on its long edge (JPEG draft mode for cheap
    1/2,1/4,1/8 decode, then a resize to land at ~long_side). Returns (PIL.Image RGB,
    (sx, sy)) mapping DECODED-image coords back to ORIGINAL pixels (so region boxes can
    be stored in original coordinates), or (None, (1.0, 1.0)) on failure."""
    from PIL import Image  # hook-allow: lazy — PIL only for decode
    try:
        im = Image.open(abs_path)
        ow, oh = im.size
        try:
            im.draft('RGB', (long_side, long_side))
        except Exception:  # hook-allow: broad — draft is an optional speedup
            pass
        im = im.convert('RGB')
        w, h = im.size
        m = max(w, h)
        if m > long_side:
            r = long_side / float(m)
            im = im.resize((max(1, int(round(w * r))), max(1, int(round(h * r)))))
        dw, dh = im.size
        # scale to convert a DECODED-coord value back to ORIGINAL pixels
        return im, (ow / dw if dw else 1.0, oh / dh if dh else 1.0)
    except Exception:  # hook-allow: broad — third-party decode, failure is a sentinel
        return None, (1.0, 1.0)


def build_object_index(db, encoder, data_root, *, image_exts, abs_path_fn=None,
                       exclude_ids=None, candidates=None, on_progress=None, log=None,
                       should_stop=None):
    """Encode + store DINOv2 region embeddings for every image lacking an object-region
    index row. `encoder` must expose model_id() and embed_regions(pil) -> list of
    ((x1,y1,x2,y2) in the passed image's pixels, float32 L2-normalized vector).
    `should_stop()` is checked before each image's write — True ends the build early
    (images already decoded in that batch are left un-indexed).

    Returns the number of candidates processed.
    """
    if candidates is None:
        excl = exclude_ids or set()
        candidates = [
            (fid, rel) for (fid, rel) in db.get_unobject_indexed_files()
            if os.path.splitext(rel)[1].lower() in image_exts and fid not in excl
        ]
    total = len(candidates)
    if on_progress:
        on_progress(0, total)
    enc_model = encoder.model_id()

    workers = int(os.environ.get('MEDIA_OBJECT_DECODE_WORKERS', '') or min(8, (os.cpu_count() or 4)))
    long_side = max(256, int(os.environ.get('MEDIA_OBJECT_DECODE_PX', '1024') or 1024))
    batch = max(1, int(os.environ.get('MEDIA_OBJECT_DECODE_BATCH', '8') or 8))

    def _resolve(fid, rel):
        if abs_path_fn is not None:
            return abs_path_fn(fid, rel)
        p = os.path.join(data_root, rel)
        return p if os.path.isfile(p) else None

    done = 0
    consecutive_failures = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    stopped = False
    try:
        for i in range(0, total, batch):
            if stopped:
                break
            chunk = candidates[i:i + batch]
            # path resolution on this thread
            resolved = [(fid, rel, _resolve(fid, rel)) for fid, rel in chunk]

            def _decode(item):
                fid, rel, ap = item
                if ap is None:
                    return (fid, rel, 'nopath', None, None)
                im, scale = _decode_draft(ap, long_side)
                if im is None:
                    return (fid, rel, 'decodefail', None, None)
                return (fid, rel, 'ok', im, scale)

            for fid, rel, status, im, scale in pool.map(_decode, resolved):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if status == 'nopath':
                    pass  # offline/moved — leave un-indexed so it retries later
                elif status == 'decodefail':
                    db.insert_object_regions(fid, [((0.0, 0.0, 0.0, 0.0), b'')], enc_model)  # sentinel
                    if log:
                        log(rel, 'object-index decode failed')
                else:
                    try:
                        regions = encoder.embed_regions(im)  # [(box_in_decoded_px, vec), ...]
                        consecutive_failures = 0
                    except ValueError as exc:
                        # The image itself is degenerate — retrying won't help.
                        db.insert_object_regions(fid, [((0.0, 0.0, 0.0, 0.0), b'')], enc_model)  # sentinel
                        if log:
                            log(rel, f'object-index embed error: {exc}')
                        regions = None
                    except Exception as exc:  # hook-allow: broad — our-side failure: log, no sentinel, abort on a run
                        # RuntimeError/MemoryError/anything else: not the image's fault.
                        # Leave it un-indexed (retried next run) and stop if it keeps
                        # happening — the encoder/GPU is broken, not the library.
                        if log:
                            log(rel, f'object-index embed failed ({type(exc).__name__}): {exc} '
                                     '(not indexed; will retry)')
                        regions = None
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
                            raise RuntimeError(
                                f'Aborted: object embedding failed on {consecutive_failures} '
                                f'images in a row (last: {type(exc).__name__}: {exc}). The '
                                'encoder looks broken; those images stay un-indexed.') from exc
                    if regions is not None:
                        sx, sy = scale
                        rows = []
                        for box, vec in regions:
                            if vec is None:
                                continue
                            x1, y1, x2, y2 = box
                            rows.append(((x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                                         np.asarray(vec, dtype=np.float32).tobytes()))
                        if rows:
                            db.insert_object_regions(fid, rows, enc_model)
                        else:
                            db.insert_object_regions(fid, [((0.0, 0.0, 0.0, 0.0), b'')], enc_model)  # sentinel
                            if log:
                                log(rel, 'object-index produced no regions')
                done += 1
            if on_progress:
                on_progress(done, total)
    finally:
        pool.shutdown(wait=True)
    return total
