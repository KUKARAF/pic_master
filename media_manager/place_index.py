"""Shared build loop for the location-by-scene (place) index.

For each un-indexed image this computes a people-masked scene descriptor (via the
place encoder) and caches local features (via the place matcher) for the same-spot
re-rank. Both the web bulk job and the `media place-index` CLI drive this one loop;
callers supply the person-box lookup, path resolution, progress callback, and error
log so the loop itself stays free of heavy/web-only dependencies.
"""
import os


def build_place_index(db, encoder, matcher, data_root, *, image_exts,
                      person_boxes_fn, abs_path_fn=None, exclude_ids=None,
                      candidates=None, on_progress=None, log=None):
    """Encode + cache features for every image lacking a place index row.

    Args:
        db: Database with get_unplace_indexed_files / insert_place_embedding /
            insert_place_keypoints.
        encoder: place encoder (embed_images, model_id).
        matcher: place matcher (extract, serialize, model_id) or None to skip features.
        data_root: repo root for default path resolution.
        image_exts: set/collection of lowercase extensions (incl. dot) to include.
        person_boxes_fn: callable(file_id) -> person boxes to mask out.
        abs_path_fn: optional callable(file_id, rel) -> abs path or None (defaults to
            data_root join, requiring the file to exist on disk).
        exclude_ids: optional set of file ids to skip when computing candidates.
        candidates: optional pre-computed [(file_id, rel), ...]; if None it is derived
            from db.get_unplace_indexed_files().
        on_progress: optional callable(done, total).
        log: optional callable(path, message) for non-fatal per-file errors.

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
    done = 0
    for fid, rel in candidates:
        abs_path = (abs_path_fn(fid, rel) if abs_path_fn is not None
                    else (os.path.join(data_root, rel)
                          if os.path.isfile(os.path.join(data_root, rel)) else None))
        if abs_path is None:
            done += 1
            if on_progress:
                on_progress(done, total)
            continue
        boxes = person_boxes_fn(fid)
        try:
            embs, failed = encoder.embed_images([abs_path], [boxes])
            if not failed and embs:
                db.insert_place_embedding(fid, embs[0].astype('float32').tobytes(), enc_model)
            else:
                db.insert_place_embedding(fid, b'', enc_model)  # sentinel
                if log:
                    log(rel, f'place embed failed: {failed}')
        except Exception as e:
            db.insert_place_embedding(fid, b'', enc_model)  # sentinel, don't re-queue
            if log:
                log(rel, f'place embed error: {e}')
        if matcher is not None:
            try:
                feat = matcher.extract(abs_path, person_boxes=boxes)
                db.insert_place_keypoints(fid, matcher.serialize(feat), matcher.model_id())
            except Exception as e:
                if log:
                    log(rel, f'place keypoints error: {e}')
        done += 1
        if on_progress:
            on_progress(done, total)
    return total
