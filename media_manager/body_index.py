"""Find-by-body: find/crop person boxes and CLIP-embed the crops, so a person can
be re-identified by outfit/build when their face is hidden or blurry — which means
this must NOT depend on a face, or even on `media index` having run, existing for
a photo. Matching is appearance-based: strong within the same event/outfit, weak
across clothing changes. Web-only by design — no CLI command calls this; web.py
triggers both the background corpus build and the on-demand per-photo embedding
on the find-by-body page.

Person boxes are reused from `media index`'s stored YOLO-World detections when a
file already has them (cheap — no model call), but for a file that's never been
object-indexed this runs its own dedicated person-only detection pass instead of
skipping the file — otherwise the body-search corpus would silently exclude every
photo that only ever went through `media add`/`media faces`, regardless of how
visible the person in it is (confirmed: YOLO-World's 'person' confidence barely
drops with the face occluded or entirely out of frame — detection quality was
never the bottleneck, the missing `media index` prerequisite was).

match_face_to_body duplicates the logic in age_estimator_worker.py, which runs in
an isolated venv and can't import from the main app — keep the two in sync.
"""
import os

from PIL import Image

# Fraction of the face box's area that must fall inside a person box to call that
# person box the face's body — containment, not IoU, since a small face box inside
# a much larger body box has a tiny IoU but is obviously "in" that body.
MIN_FACE_IN_BODY_OVERLAP = 0.7

# Stored YOLO-World detections go down to conf 0.15; body crops want more certain
# person boxes than tag search does, so filter harder here.
MIN_PERSON_CONFIDENCE = 0.3


def _containment(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    ix1, iy1 = max(fx1, px1), max(fy1, py1)
    ix2, iy2 = min(fx2, px2), min(fy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    face_area = max(1e-6, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def match_face_to_body(face_bbox, person_bboxes):
    """Best-matching person box for this face, or None if nothing overlaps enough."""
    best_score = 0.0
    best_bbox = None
    for person_bbox in person_bboxes:
        score = _containment(face_bbox, person_bbox)
        if score > best_score:
            best_score = score
            best_bbox = person_bbox
    if best_score >= MIN_FACE_IN_BODY_OVERLAP:
        return best_bbox
    return None


def crop_bodies(abs_path, bboxes):
    """Crop person boxes out of one image. Returns [(bbox, PIL image)], dropping
    boxes that are degenerate after clamping to the image bounds."""
    pairs = []
    with Image.open(abs_path) as img:
        img = img.convert('RGB')
        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(img.width, int(bbox[2]))
            y2 = min(img.height, int(bbox[3]))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            pairs.append(([x1, y1, x2, y2], img.crop((x1, y1, x2, y2))))
    return pairs


def embed_bodies_for_file(db, clip_indexer, file_id, abs_path, person_boxes=None, detector=None):
    """Crop + embed one file's person boxes and upsert its body_embeddings rows
    (sentinel when none). person_boxes, when given, skips detection entirely (used
    when a caller already has fresh boxes on hand). Otherwise: reuse this file's
    stored YOLO-World detections if `media index` already ran on it; for a file
    that's never been object-indexed at all, run a dedicated person-only pass with
    `detector` instead of treating "no stored detections" as "no person" — that
    distinction (via db.has_object_detections) is what makes this self-sufficient
    rather than silently skipping every never-object-indexed file. Returns the
    number of bodies embedded."""
    if person_boxes is None:
        person_boxes = db.get_person_detections_for_file(file_id, min_conf=MIN_PERSON_CONFIDENCE)
        if not person_boxes and not db.has_object_detections(file_id):
            if detector is None:
                raise ValueError('file has no stored object index and no detector was given')
            detector.set_vocab(['person'])
            _, detections, error = detector.detect_images([abs_path])[0]
            if error:
                raise RuntimeError(f'person detection failed: {error}')
            person_boxes = [[x1, y1, x2, y2] for _cls, conf, x1, y1, x2, y2 in detections
                            if conf >= MIN_PERSON_CONFIDENCE]
    bodies = []
    if person_boxes:
        pairs = crop_bodies(abs_path, person_boxes)
        if pairs:
            embeddings = clip_indexer.embed_pil_images([crop for _, crop in pairs])
            bodies = [{'bbox': bbox, 'embedding': embeddings[i]}
                      for i, (bbox, _) in enumerate(pairs)]
    db.insert_body_embeddings(file_id, bodies, clip_indexer.model_name)
    return len(bodies)


def build_body_index(db, errors, clip_indexer, detector, data_root, on_progress=None):
    """Body-index every tracked file that has no body row yet — the background
    corpus build behind the web UI's "Build body index" button. No longer requires
    `media index` to have run first: files with stored YOLO-World detections reuse
    them (fast path, no model call), everything else gets a dedicated person-only
    detection pass via `detector`. Failures go to the error log (same policy as the
    batch ML passes in media_manager.py); a failed file is left un-sentineled so a
    rebuild retries it. Returns (processed, total)."""
    files = db.get_unbody_indexed_files()
    total = len(files)
    processed = 0
    for file_id, rel_path in files:
        try:
            abs_path = os.path.join(data_root, rel_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError('file missing on disk')
            embed_bodies_for_file(db, clip_indexer, file_id, abs_path, detector=detector)
        except Exception as exc:
            errors.log(rel_path, f'body index: {exc}')
        processed += 1
        if on_progress:
            on_progress(processed, total)
    return processed, total
