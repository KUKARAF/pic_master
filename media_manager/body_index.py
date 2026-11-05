"""Find-by-body: find/crop person boxes and embed the crops, so a person can
be re-identified by outfit/build when their face is hidden or blurry — which means
this must NOT depend on a face, or even on `media index` having run, existing for
a photo. Matching is appearance-based: strong within the same event/outfit, weak
across clothing changes. Web-only by design — no CLI command calls this; web.py
triggers both the background corpus build and the on-demand per-photo embedding
on the find-by-body page.

Person boxes are reused from `media index`'s stored YOLO-World detections when a
file already has a person-like box (cheap — no model call). Crucially, "person-like"
is NOT just the exact label 'person': `media index` runs YOLO-World over the full
DEFAULT_VOCAB, which is open-vocabulary and assigns ONE best label per box, so a
person is frequently stored as 'portrait'/'selfie'/'child' rather than 'person'.
Matching only 'person' silently dropped all of those from the body corpus (they
appeared indexable but returned no crops), so PERSON_LIKE_CLASSES widens the reuse
to every single-person label. 'crowd'/'group of people' are deliberately excluded:
their box spans many people, so the crop is a useless blend for re-ID.

When NO person-like box is stored — the file was never object-indexed, OR it was but
YOLO-World gave the people only excluded labels (crowd/group) or a scene label — we
do NOT trust that as "no person": we run a dedicated person-only detection pass, the
same one behind the find-by-body "Reindex now" button. The (possibly empty) result
is sentineled by the caller, so this one-time pass never repeats for a genuinely
peopleless photo. (Confirmed: YOLO-World's 'person' confidence barely drops with the
face occluded or entirely out of frame — detection quality was never the bottleneck,
trusting the general index's labels was.)

EMBEDDING MODEL (was CLIP; now DINOv2-L by default, SigLIP 2 optional)
-----------------------------------------------------------------------
Select with env ``MEDIA_BODY_MODEL`` = ``dino`` (default) | ``siglip``. The crops go
through the shared worker/local embedders (web._get_dino_embedder / _get_siglip_embedder),
never CLIP.

Why DINOv2-L by default (decided from the literature and this repo's own bench, NOT
measured on this task -- no GPU was available when it was chosen; compare on bigboy
with ``bench/bench_body_model.py`` and flip the env var if SigLIP wins):
  * Person re-ID by outfit/build is an INSTANCE-level, appearance-matching problem: the
    same garment colours/patterns/silhouette must match across photos while a different
    person in a similar-looking scene must not. DINOv2's self-supervised, non-semantic
    features are the strongest off-the-shelf instance-retrieval features (Oquab et al.
    2023: Oxford/Paris, Met, copy-detection); SigLIP/CLIP-style contrastive features
    are trained on alt-text and collapse "a man in a blue shirt" photos together
    (semantic category dominates the vector), which is exactly the failure that made the
    old CLIP body search weak. DINO-based person re-ID (e.g. DINOv2 fine-tunes/linear
    probes on Market-1501/MSMT17, CLIP-ReID needs fine-tuning + prompts to get there)
    follows the same pattern.
  * Repo evidence: bench/bench_pattern_dino.py found DINOv2-L cell descriptors beat the
    ResNet pattern windows for local appearance/texture matching, the same signal a
    clothing crop needs; SigLIP 2 is used here for text<->image (semantic) search.
  * Same worker op / GPU lock / cache as the already-deployed similarity embedder, 1024-d
    (vs 1152-d for SigLIP 2 SO400M), so no new model to ship.
Caveats: DINOv2's input is squashed to 224x224 (no crop) so tall person boxes are
anisotropically resized, identically for index and query; thresholds below are per model
and UNTUNED. SigLIP's strength is semantic (gender/age/garment type), so it may win on
coarse "same kind of outfit" recall -- hence the switch.

STALE ROWS: body_embeddings.model and body_identities.model hold ``model_tag()``
(e.g. ``dino:large:1024``). Rows with another tag (the old CLIP ones) are NEVER mixed
into a search (matrices, anchors and per-file reads are filtered by tag) and count as
MISSING for the 'bodies' stage, so a rerun re-embeds them in place (row ids, and thus
the swipe's source_body_id decisions, are preserved when the person box is unchanged).
Ground-truth labels in manual.db are not deleted: refresh_stale_labels() re-crops and
re-embeds them under the new tag. 'No person' sentinels are model-independent.

match_face_to_body duplicates the logic in age_estimator_worker.py, which runs in
an isolated venv and can't import from the main app — keep the two in sync.
"""
import os

import numpy as np
from PIL import Image

from . import compute
from .formats import IMAGE_EXTENSIONS

# Fraction of the face box's area that must fall inside a person box to call that
# person box the face's body — containment, not IoU, since a small face box inside
# a much larger body box has a tiny IoU but is obviously "in" that body.
MIN_FACE_IN_BODY_OVERLAP = 0.7

# Stored YOLO-World detections go down to conf 0.15; body crops want more certain
# person boxes than tag search does, so filter harder here.
MIN_PERSON_CONFIDENCE = 0.3

# DEFAULT_VOCAB labels that denote ONE croppable person, any of which the general
# object index may have assigned to a person instead of 'person' (open-vocab, one
# best label per box). Body re-ID must treat all of them as person boxes — see the
# module docstring. 'crowd'/'group of people' are excluded on purpose: one box over
# many people crops to a blend that's useless for re-identification.
PERSON_LIKE_CLASSES = ('person', 'child', 'selfie', 'portrait')


BODY_MODEL_CHOICES = ('dino', 'siglip')
BODY_MODEL_DEFAULT = 'dino'

# Cosine floors per space for "show this body as a candidate / match". UNTUNED guesses
# (DINOv2 CLS cosines of different people sit lower than SigLIP image-image ones);
# env MEDIA_BODY_THRESHOLD overrides both. Tune on bigboy with bench/bench_body_model.py.
BODY_THRESHOLDS = {'dino': 0.5, 'siglip': 0.6}

# Dim tables duplicated from embedders.py (which imports torch; this module must not).
# tests/test_body_model.py asserts they stay in sync.
_DINO_DIMS = {'small': 384, 'base': 768, 'large': 1024}
_SIGLIP_DIMS = {'ViT-B-16-SigLIP2': 768, 'ViT-B-32-SigLIP2-256': 768,
                'ViT-L-16-SigLIP2-256': 1024, 'ViT-SO400M-14-SigLIP2': 1152}


def model_key():
    """'dino' | 'siglip' from MEDIA_BODY_MODEL (default dino)."""
    key = os.environ.get('MEDIA_BODY_MODEL', BODY_MODEL_DEFAULT).strip().lower()
    if key not in BODY_MODEL_CHOICES:
        raise ValueError(f'MEDIA_BODY_MODEL={key!r} not in {BODY_MODEL_CHOICES}')
    return key


def model_dim():
    if model_key() == 'dino':
        return _DINO_DIMS[os.environ.get('MEDIA_DINO_MODEL', 'large').strip().lower()]
    return _SIGLIP_DIMS.get(os.environ.get('MEDIA_SIGLIP_MODEL', 'ViT-SO400M-14-SigLIP2'), 0)


def model_tag():
    """Space tag stored with every body vector, e.g. 'dino:large:1024' or
    'siglip:ViT-SO400M-14-SigLIP2:1152'. Pure env arithmetic (no torch, no worker
    round-trip) so DB queries / scan_status can use it freely."""
    key = model_key()
    if key == 'dino':
        variant = os.environ.get('MEDIA_DINO_MODEL', 'large').strip().lower()
        return f'dino:{variant}:{_DINO_DIMS[variant]}'
    name = os.environ.get('MEDIA_SIGLIP_MODEL', 'ViT-SO400M-14-SigLIP2')
    return f'siglip:{name}:{_SIGLIP_DIMS.get(name, 0)}'


def threshold():
    env = os.environ.get('MEDIA_BODY_THRESHOLD')
    return float(env) if env else BODY_THRESHOLDS[model_key()]


def embed_crops(embedder, crops):
    """Embed PIL crops and verify the vectors live in the configured space."""
    embs = np.asarray(embedder.embed_pil_images(list(crops)), dtype=np.float32)
    dim = model_dim()
    if embs.ndim != 2 or (dim and embs.shape[1] != dim):
        raise RuntimeError(f'body embedder returned shape {embs.shape}, expected (N, {dim}) for '
                           f'{model_tag()} -- is the worker running a different model than '
                           f'MEDIA_BODY_MODEL/MEDIA_DINO_MODEL/MEDIA_SIGLIP_MODEL?')
    return embs


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


def embed_bodies_for_file(db, embedder, file_id, abs_path, person_boxes=None, detector=None):
    """Crop + embed one file's person boxes and upsert its body_embeddings rows
    (sentinel when none). person_boxes, when given, skips detection entirely (used
    when a caller already has fresh boxes on hand). Otherwise: reuse this file's
    stored PERSON_LIKE_CLASSES boxes from `media index` if it has any (cheap, no
    model call — and this counts 'portrait'/'selfie'/'child', not just the exact
    'person' label, since YOLO-World often labels a lone person that way); if it has
    none, run a dedicated person-only pass with `detector` rather than treating the
    absence of a person-like box as "no person" (see the module docstring). Returns
    the number of bodies embedded."""
    if person_boxes is None:
        person_boxes = db.get_person_detections_for_file(
            file_id, min_conf=MIN_PERSON_CONFIDENCE, class_names=PERSON_LIKE_CLASSES)
        if not person_boxes:
            # No stored person-like box. Do NOT trust that as "no person" — the file
            # may never have been object-indexed, or the general index may have given
            # the people only excluded labels (crowd/group) or a scene label. Run the
            # dedicated person-only pass (same as the "Reindex now" button); its result
            # gets sentineled by insert_body_embeddings, so this never repeats.
            if detector is None:
                raise ValueError('file has no stored person box and no detector was given')
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
            embeddings = embed_crops(embedder, [crop for _, crop in pairs])
            bodies = [{'bbox': bbox, 'embedding': embeddings[i]}
                      for i, (bbox, _) in enumerate(pairs)]
    db.insert_body_embeddings(file_id, bodies, model_tag())
    return len(bodies)


def build_body_index(db, errors, embedder, detector, data_root, on_progress=None, exclude_ids=None,
                     files=None, should_stop=None, manual=None):
    """Body-index every tracked file with no body row in the current space (model_tag()) yet — the background
    corpus build behind the web UI's "Build body index" button. No longer requires
    `media index` to have run first: files with stored YOLO-World detections reuse
    them (fast path, no model call), everything else gets a dedicated person-only
    detection pass via `detector`. `exclude_ids` (e.g. trashed file ids) are dropped up
    front. Failures go to the error log (same policy as the batch ML passes in
    media_manager.py); a failed file is left un-sentineled so a rebuild retries it.
    `files` [(file_id, rel_path)] restricts the build to those (the file manager's
    folder job; default: every un-body-indexed file). `should_stop()` is checked
    before each file — True ends the build early. `manual`, when given, also gets its
    stale (old-space) ground-truth body labels re-embedded at the end. Returns (processed, total)."""
    if files is None:
        files = db.get_unbody_indexed_files(model_tag())
    if exclude_ids:
        files = [f for f in files if f[0] not in exclude_ids]
    total = len(files)
    processed = 0
    for file_id, rel_path in files:
        if should_stop is not None and should_stop():
            break
        # Body detection is image-only (PIL/YOLO); videos are tracked but have no
        # body crops — skip them (get_unbody_indexed_files doesn't filter by kind).
        if os.path.splitext(rel_path)[1].lower() not in IMAGE_EXTENSIONS:
            processed += 1
            if on_progress:
                on_progress(processed, total)
            continue
        try:
            abs_path = os.path.join(data_root, rel_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError('file missing on disk')
            embed_bodies_for_file(db, embedder, file_id, abs_path, detector=detector)
        except Exception as exc:  # hook-allow: broad — per-file resilience, logged
            errors.log(rel_path, f'body index: {exc}')
            compute.raise_if_gpu_fatal(exc)   # GPU gone: stop, don't fail every remaining file
        processed += 1
        if on_progress:
            on_progress(processed, total)
    if manual is not None and not (should_stop is not None and should_stop()):
        refresh_stale_labels(db, manual, embedder, data_root, errors=errors)
    return processed, total


def refresh_stale_labels(db, manual, embedder, data_root, identity=None, errors=None):
    """Re-crop and re-embed ground-truth body labels (manual.db body_identities) whose
    stored vector is in another space (old CLIP rows), in place under model_tag(). The
    label (identity, box, checksum, source_body_id) is untouched -- only the vector
    and its tag change. `identity` limits it to one person. Labels whose photo is gone
    or whose crop fails are skipped (they stay stale and out of searches). Returns the
    number refreshed."""
    tag = model_tag()
    done = 0
    for label_id, checksum, bbox in manual.get_stale_body_labels(tag, identity):
        try:
            frow = db.get_file_by_checksum(checksum)
            if frow is None:
                continue
            abs_path = os.path.join(data_root, frow['path'])
            if not os.path.isfile(abs_path):
                continue
            pairs = crop_bodies(abs_path, [bbox])
            if not pairs:
                continue
            emb = embed_crops(embedder, [pairs[0][1]])[0]
            manual.update_body_label_embedding(label_id, emb.tobytes(), tag)
            done += 1
        except Exception as exc:  # hook-allow: broad -- per-label resilience, logged
            if errors is not None:
                errors.log(checksum, f'body label refresh: {exc}')
            compute.raise_if_gpu_fatal(exc)
    return done
