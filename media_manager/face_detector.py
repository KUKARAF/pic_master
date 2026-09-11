"""InsightFace-based face detector and embedder."""
import os
import math
import numpy as np

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

# --- Angle convention: the single source of truth for the whole app -----------
# Every face dict carries 'angle' = the in-plane roll of the face AS IT SITS IN
# THE SOURCE IMAGE, measured CLOCKWISE from upright, folded into (-180, 180].
# 0.0 means upright / nothing was corrected. To render such a face upright you
# rotate its region COUNTER-clockwise by `angle` (PIL: region.rotate(angle,
# expand=True); cv2: getRotationMatrix2D(center, angle, 1) — both read positive
# degrees as counter-clockwise).
#
# This is the NEGATION of PCN's own Window.angle: pcn_detect.crop_face warps a
# face upright by re-rotating its corners through `-angle`, which in image
# coordinates (y down) makes PCN's angle the COUNTER-clockwise roll. Verified
# end to end in tests/test_pcn_rotation.py — rotate a real photo 90 deg
# counter-clockwise and detection reports angle = -90. The negation happens here,
# once, at the only place PCN's frame is ever visible, so no consumer (DB, web,
# templates) has to think about it.

# --- Rotation normalization ----------------------------------------------------
# InsightFace's SCRFD is trained on roughly-upright faces. On an in-plane-rotated
# face it either misses it outright (a recall hole) or — worse — finds the box and
# fits sloppy landmarks, so the aligned 112x112 patch it hands ArcFace is
# distorted: a confident-looking detection with a near-useless embedding.
# Measured on a 6-face photo, cosine similarity of each face's embedding against
# the same face upright: ~1.00 unrotated, 0.93-0.99 at 40 deg, 0.70-0.99 at
# 90 deg, 0.37-0.84 upside down — while det_score stays as high as 0.85. Hence:
# det_score alone never tells us the embedding is broken, so PCN (which is
# rotation-invariant) supplies the angle and we re-embed from an upright warp,
# which measured 0.98+ similarity at every rotation.
#
# Arm 1 (RECOVER) — a PCN face no InsightFace box overlaps. Nothing corroborates
# it and a false positive becomes a permanent DB row, so the floor stays high;
# PCN stage-3 is precise enough for that to still find faces.
PCN_MIN_CONF = 0.98
# Arm 2 (CORRECT) — a PCN face that DOES overlap an InsightFace box: the
# mis-aligned case above. A lower floor is safe here because InsightFace
# independently agrees a face is there; PCN only has to supply the angle.
PCN_CORRECT_MIN_CONF = 0.90
# Below this |angle| we leave the face alone. PCN's angle carries ~25 deg of noise
# (upright faces in the measurement above were reported anywhere in -13..+24), so
# a lower gate would burn a recognition call per face on nothing.
PCN_CORRECT_MIN_ANGLE = 20.0
# A PCN face overlapping an InsightFace one (by IoU) is the same face.
PCN_DEDUP_IOU = 0.3
# Backstop on how many faces we re-embed per image, correction and recovery
# together (a pathological crowd photo shouldn't fan out into dozens of extra
# recognition calls).
PCN_MAX_RECOVER = 30
# SCRFD needs CONTEXT around a face: it is an anchor-based detector and a face
# filling the whole frame falls outside its scale range. Measured on the same
# photo, warping PCN's bare face square found 0 faces out of 6 at any crop size;
# growing the square by PCN_WARP_PAD before the warp found 6 out of 6 at 0.77-0.87
# det_score. (This is why the previous zero-pad recovery arm silently recovered
# nothing.) The crop is rendered large enough that the face itself still lands at
# ~145px, comfortably above ArcFace's 112px input.
PCN_WARP_PAD = 1.2
PCN_WARP_SIZE = 320
# Context manufactured around an already-tight crop in embed_bbox's PCN tier, as
# a fraction of the crop's long side. Measured: a bare face crop is invisible to
# PCN, the same crop with this much replicated border is found at conf 1.0.
PCN_CROP_BORDER = 0.4
# Same story for the axis-aligned re-crop the quarter-turn sweep uses: at 0.1 pad
# the padded crop detected nothing, at 0.5 it matched the whole-image score.
ROTATION_TTA_PAD = 0.5
# Never swap an embedding for a re-detected one unless it CLEARLY beats the
# incumbent: a tie means we only shuffled noise, and without the margin the
# rotation paths would flip-flop between runs instead of converging — which is
# also what makes normalize_faces idempotent.
ROTATION_CORRECT_MARGIN = 0.05
# The score guardrail alone is not enough, and the measurement says why: an
# upside-down face can keep a 0.88 whole-image score AND a 0.88 un-rotated warp
# score while its embedding sits at 0.41 similarity to the same face upright. What
# does separate the two is the incumbent read's OWN landmarks: whenever they sat
# more than this far off upright, ArcFace's similarity-transform alignment had
# degraded (0.37-0.84), while the PCN upright warp measured 0.98+ at every
# rotation. So a materially-rotated incumbent alignment is grounds to adopt the
# rotated re-embed even without a score win. Above PCN's own ~25 deg angle noise,
# so a merely tilted head never trips it.
ROTATION_ALIGN_TOLERANCE = 45.0
# Quarter-turn sweep trigger. A face SCRFD reports confidently is *located* well,
# so the sweep would mostly re-confirm it at real cost; a weak detection is the
# signature of a rotated or badly framed face — including ones PCN itself missed.
ROTATION_TTA_TRIGGER = 0.72
# Slack around a user-drawn box in embed_bbox. Unchanged default: it is enough for
# SCRFD to re-find the face (measured 0.77-0.85) while staying tight enough that a
# neighbouring face in a crowd shot doesn't wander into the crop.
FACE_CROP_PAD_RATIO = 0.3
# A re-detect inside a padded crop must be the face we cropped FOR, not a
# neighbour that came along with the padding. The target is centred by
# construction, so accept only detections whose centre is within this fraction of
# the crop's short side from the crop centre.
CROP_CENTER_TOLERANCE = 0.25


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _norm_angle(deg):
    """Fold a rotation into (-180, 180] so -10 and 350 are the same tilt and the
    |angle| thresholds above mean what they read like."""
    a = float(deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def _kps_angle(kps):
    """In-plane roll, in our convention, implied by InsightFace's own 5 landmarks:
    the eye-centre -> mouth-centre vector points straight down on an upright face.
    Built from midpoints so a left/right swap in the landmark order — which is
    exactly what SCRFD does on a rotated face — can't flip the result."""
    k = np.asarray(kps, dtype=np.float64)
    if k.shape[0] < 5:
        return 0.0
    vx, vy = ((k[3] + k[4]) - (k[0] + k[1])) / 2.0
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return _norm_angle(math.degrees(math.atan2(-vx, vy)))


def _alignment_rank(face):
    """Sort key for choosing between reads of the same face at different
    rotations. We are choosing an ALIGNMENT, not a detection: a read whose own
    landmarks come out upright beats a higher-scoring one that doesn't, and score
    only breaks ties inside each group."""
    return (abs(_kps_angle(face.kps)) >= ROTATION_ALIGN_TOLERANCE,
            -float(face.det_score))


def _padded_crop(img, bbox, pad_ratio):
    """Crop `bbox` with `pad_ratio` slack, clamped to the image. Returns
    (crop, x_offset, y_offset) so a detection inside the crop can be mapped back
    to original-image coordinates; the crop is empty for a degenerate box."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return img[cy1:cy2, cx1:cx2], cx1, cy1


def _inflate_square(bbox, pad_ratio):
    """Grow a box about its own centre. The centre must not move: it is the pivot
    PCN's warp rotates around, so shifting it would rotate the wrong point."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = (x2 - x1) / 2.0 * (1.0 + pad_ratio)
    return [cx - half, cy - half, cx + half, cy + half]


class FaceDetector:
    def __init__(self, model_name='buffalo_l', det_thresh=0.5, ctx_id=0):
        from insightface.app import FaceAnalysis
        self.det_thresh = det_thresh
        self._model_name = model_name
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._pcn = None
        self._pcn_failed = False

    def _get_pcn(self):
        """Lazily construct the PCN rotation-invariant detector (onnxruntime, no
        torch). If it can't load, disable it and fall back to InsightFace-only
        detection — PCN is a supplementary recall boost, never a hard requirement."""
        if self._pcn is None and not self._pcn_failed:
            try:
                from media_manager.pcn import PCNDetector
                self._pcn = PCNDetector()
            except Exception:
                self._pcn_failed = True
        return self._pcn

    def detect_faces(self, paths: list) -> list:
        """Run detection + embedding on a list of image paths.

        Returns list of (path, faces, error):
          faces = [{'bbox': [x1,y1,x2,y2], 'embedding': np.ndarray,
                    'det_score': float, 'angle': float}]
          angle = in-plane roll in degrees, clockwise from upright (module header);
                  0.0 when the face needed no rotation correction
          error = None on success, str on failure
        """
        import cv2
        results = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                results.append((path, [], 'unsupported extension'))
                continue
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError('cv2.imread returned None')
                faces = []
                for face in self.app.get(img):
                    if face.det_score < self.det_thresh:
                        continue
                    faces.append({
                        'bbox': [float(v) for v in face.bbox],
                        # normed_embedding, not embedding — InsightFace's raw .embedding
                        # is NOT unit length, so a plain dot product isn't cosine
                        # similarity (scores can exceed 1, thresholds become meaningless).
                        'embedding': face.normed_embedding.astype(np.float32),
                        'det_score': float(face.det_score),
                        'angle': 0.0,
                    })
                self._normalize_in_place(img, faces)
                results.append((path, faces, None))
            except Exception as exc:
                results.append((path, [], str(exc)))
        return results

    def normalize_faces(self, img, faces):
        """Given a full-size BGR image and existing face dicts (bbox/embedding/
        det_score/angle, e.g. re-hydrated from the DB), re-run in-plane rotation
        normalization on each and return a NEW list with corrected 'embedding' +
        'angle'. Runs the exact same normalization detect_faces does, so it is
        deterministic and idempotent: a weak or unaligned vector is recomputed
        through the upright path, a face the pipeline considers correctly aligned
        comes back byte-identical, and a second pass changes nothing. Backs the
        re-normalization backfill job.

        Same length and order as `faces` (the backfill maps results back to DB rows
        positionally), and the input dicts are never mutated. Faces InsightFace
        originally missed are NOT discovered here — inventing rows is detection's
        job; the backfill can only rewrite the rows it was handed.
        """
        working = []
        for face in faces:
            emb = face.get('embedding')
            working.append({
                'bbox': [float(v) for v in face['bbox']],
                # Left alone when no arm fires: every decision below is taken
                # against the image, not against the stored vector, so a face the
                # pipeline considers correct must come back byte-identical — that
                # is what makes re-running the backfill a no-op.
                'embedding': None if emb is None else np.asarray(emb, dtype=np.float32),
                'det_score': float(face.get('det_score') or 0.0),
                'angle': _norm_angle(face.get('angle') or 0.0),
            })
        return self._normalize_in_place(img, working, recover=False)

    def _normalize_in_place(self, img, faces, recover=True):
        """THE in-plane rotation normalization pass: PCN correct+recover, then the
        quarter-turn sweep for weak faces PCN never reported. detect_faces and
        normalize_faces both route through here so the live pipeline and the
        backfill cannot drift into two different normalizations. Mutates and
        returns `faces`."""
        claimed = self._pcn_correct_and_recover(img, faces, recover=recover)
        self._quarter_turn_backstop(img, faces, claimed)
        return faces

    def _pcn_correct_and_recover(self, img, faces, recover=True):
        """One PCN pass that CORRECTS mis-aligned faces and RECOVERS missed ones.

        PCN sees faces at any in-plane rotation, so each of its candidates lands in
        one of two arms:

        * Overlaps an InsightFace box (IoU >= PCN_DEDUP_IOU) and is materially
          rotated -> InsightFace found the face but aligned it badly, so its
          embedding is distorted. Re-embed from PCN's upright warp and REPLACE
          embedding/det_score/angle in place, keeping InsightFace's bbox: SCRFD
          located the face correctly, only the alignment was wrong, and PCN's box
          is the looser axis-aligned bound of a rotated square.
        * Overlaps nothing -> a face InsightFace missed entirely; append it.

        Mutates `faces` (corrections in place, recoveries appended when `recover`)
        and returns the set of indices it CLAIMED — every face a candidate overlaps,
        adopted or not. The quarter-turn sweep skips those: PCN already has an
        opinion about them, and a blind 90-degree sweep that disagrees with it
        writes a confidently WRONG angle (measured: it turned an upside-down face
        into a "+90" one, which then renders wrong everywhere). Best-effort by
        design: any failure (PCN unavailable, a bad crop) leaves normal detection
        standing.
        """
        claimed = set()
        try:
            pcn = self._get_pcn()
            if pcn is None:
                return claimed
            # One floor per arm, so the corroborated correction arm isn't held to
            # the high bar the uncorroborated recovery arm needs.
            floor = min(PCN_MIN_CONF, PCN_CORRECT_MIN_CONF) if recover else PCN_CORRECT_MIN_CONF
            candidates = [c for c in pcn.detect(img) if c['conf'] >= floor]
            # Highest-confidence first so the per-image cap keeps the best faces.
            candidates.sort(key=lambda c: c['conf'], reverse=True)
            for cand in candidates:
                if len(claimed) >= PCN_MAX_RECOVER:
                    break
                angle = _norm_angle(-cand['angle'])  # PCN's frame -> ours
                hit = -1
                for i, face in enumerate(faces):
                    if _iou(cand['bbox'], face['bbox']) >= PCN_DEDUP_IOU:
                        hit = i
                        break
                if hit >= 0:
                    self._correct_face(pcn, img, cand, angle, faces, hit, claimed)
                    claimed.add(hit)
                    continue
                if not recover or cand['conf'] < PCN_MIN_CONF:
                    continue
                got = self._embed_warped(pcn, img, cand, cand['angle'])
                if got is None:
                    continue
                # Original-image bbox = axis-aligned bounds of PCN's rotated square.
                xs = [p[0] for p in cand['points']]
                ys = [p[1] for p in cand['points']]
                faces.append({
                    'bbox': [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                    'embedding': got['embedding'],
                    'det_score': got['det_score'],
                    'angle': angle,
                })
                claimed.add(len(faces) - 1)
            return claimed
        except Exception:
            return claimed

    def _correct_face(self, pcn, img, cand, angle, faces, hit, claimed):
        """Correction arm for one PCN candidate that overlaps `faces[hit]`. Returns
        True when the face's embedding/det_score/angle were replaced."""
        if hit in claimed:
            return False  # two candidates on one face: the confident one already won
        if cand['conf'] < PCN_CORRECT_MIN_CONF:
            return False
        if abs(angle) < PCN_CORRECT_MIN_ANGLE:
            return False  # near-upright: SCRFD's own alignment is fine
        rotated = self._embed_warped(pcn, img, cand, cand['angle'])
        if rotated is None:
            return False  # PCN has no recognition model; if ArcFace's detector
            # can't see a face in the upright warp we must not trust PCN alone
        # Guardrail (the one the original rotation TTA had): the rotated read must
        # clearly beat the un-rotated one. Measured against the SAME warped region
        # rather than the stored det_score, because a whole-image SCRFD score and a
        # tight-crop score are not comparable — an upside-down face keeps a 0.85
        # whole-image score while its un-rotated warp is often not detected at all,
        # so comparing against the stored number would reject exactly the
        # corrections this exists to make. Un-rotated non-detection scores 0.0: the
        # incumbent alignment can't even be reproduced, so it cannot be defended.
        flat = self._embed_warped(pcn, img, cand, 0.0)
        baseline = flat['det_score'] if flat is not None else 0.0
        if rotated['det_score'] < baseline + ROTATION_CORRECT_MARGIN:
            # Second chance: the incumbent read's own landmarks betray it. If they
            # sit materially off upright, the embedding we'd be defending came out
            # of a heavily rotated alignment — the regime where ArcFace measurably
            # falls apart — so a score tie is no reason to keep it.
            if flat is None or abs(_kps_angle(flat['kps'])) < ROTATION_ALIGN_TOLERANCE:
                return False
        faces[hit]['embedding'] = rotated['embedding']
        faces[hit]['det_score'] = rotated['det_score']
        faces[hit]['angle'] = angle
        return True

    def _embed_warped(self, pcn, img, cand, pcn_angle):
        """Warp the padded region around a PCN candidate so `pcn_angle` (PCN's own
        frame) becomes upright, then re-run InsightFace on it so the embedding
        comes from real, correctly-oriented landmarks. `pcn_angle=0.0` gives the
        un-rotated read of the very same region, which is the only fair baseline
        for the guardrail. Returns {'embedding', 'det_score', 'kps'} or None."""
        win = {'bbox': _inflate_square(cand['bbox'], PCN_WARP_PAD),
               'angle': pcn_angle, 'conf': cand['conf']}
        crop = pcn.upright_crop(img, win, PCN_WARP_SIZE)
        if crop is None or crop.size == 0:
            return None
        face = self._centered_face(self.app.get(crop), crop.shape)
        if face is None:
            return None
        return {
            'embedding': face.normed_embedding.astype(np.float32),
            'det_score': float(face.det_score),
            # Carried so the caller can ask what roll this read's own landmarks
            # imply — the only runtime evidence of a bad alignment we get for free.
            'kps': face.kps,
        }

    def _quarter_turn_backstop(self, img, faces, claimed):
        """Quarter-turn test-time augmentation for weak faces PCN never reported —
        it has its own recall holes, and before this existed a sideways face SCRFD
        half-saw kept its distorted vector forever. Faces in `claimed` are PCN's
        call, not ours.

        Re-crops each weak face on its own (so it fills more of the frame) and
        re-detects it at 0/90/180/270 degrees, then picks the read whose own
        landmarks come out upright — NOT the highest-scoring one. That distinction
        is load-bearing: on an upside-down face SCRFD happily scores the sideways
        quarter turn highest, and picking by score alone stamped a confidently
        wrong "+90" angle on it. A rotation is adopted only if it beats the upright
        read of the same crop by ROTATION_CORRECT_MARGIN (same region, same size,
        same detector — a fair comparison) or that upright read's landmarks are
        themselves materially rotated, i.e. its alignment cannot be defended.
        Failing all that, an upright read that clearly beats the incumbent
        detection is still adopted: that is how a vector from embed_bbox's
        unaligned last resort (det_score 0.0) gets recomputed through the properly
        aligned upright path. Confidently-detected faces skip the sweep entirely,
        so they cost nothing.
        """
        import cv2
        turns = ((1, cv2.ROTATE_90_CLOCKWISE),
                 (2, cv2.ROTATE_180),
                 (3, cv2.ROTATE_90_COUNTERCLOCKWISE))
        for i, face in enumerate(faces):
            if i in claimed or face['det_score'] >= ROTATION_TTA_TRIGGER:
                continue
            try:
                crop, _x, _y = _padded_crop(img, face['bbox'], ROTATION_TTA_PAD)
                if crop.size == 0:
                    continue
                upright = self._centered_face(self.app.get(crop), crop.shape)
                reads = [] if upright is None else [(0, upright)]
                for k, code in turns:
                    rot = cv2.rotate(crop, code)
                    got = self._centered_face(self.app.get(rot), rot.shape)
                    if got is not None:
                        reads.append((k, got))
                if not reads:
                    continue
                best_k, best = min(reads, key=lambda r: _alignment_rank(r[1]))
                flat_score = float(upright.det_score) if upright is not None else 0.0
                if best_k:
                    tilted = (upright is None
                              or abs(_kps_angle(upright.kps)) >= ROTATION_ALIGN_TOLERANCE)
                    if not tilted and float(best.det_score) < flat_score + ROTATION_CORRECT_MARGIN:
                        continue
                elif flat_score < face['det_score'] + ROTATION_CORRECT_MARGIN:
                    continue
                face['embedding'] = best.normed_embedding.astype(np.float32)
                face['det_score'] = float(best.det_score)
                # We rotated the crop CLOCKWISE by 90*best_k to bring the face
                # upright, so in the source image the face is rolled
                # counter-clockwise by 90*best_k, i.e. clockwise by -90*best_k —
                # which is exactly what 'angle' reports.
                face['angle'] = _norm_angle(-90.0 * best_k)
                claimed.add(i)
            except Exception:
                continue  # one bad crop must not cost us the rest of the image

    def _centered_face(self, got, shape):
        """Pick the detection a padded crop was made FOR: the target sits at the
        centre by construction, so a face off in a corner is a neighbour that the
        padding dragged in, and adopting it would write one person's vector onto
        another person's row. Returns None when nothing central clears
        det_thresh."""
        if not got:
            return None
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0
        limit = CROP_CENTER_TOLERANCE * min(h, w)
        best, best_dist = None, None
        for face in got:
            if face.det_score < self.det_thresh:
                continue
            bx1, by1, bx2, by2 = face.bbox
            dist = (((bx1 + bx2) / 2.0 - cx) ** 2 + ((by1 + by2) / 2.0 - cy) ** 2) ** 0.5
            if dist <= limit and (best_dist is None or dist < best_dist):
                best, best_dist = face, dist
        return best

    def embed_bbox(self, img, bbox: list, pad_ratio: float = FACE_CROP_PAD_RATIO) -> dict:
        """Given a full-size BGR numpy image and a user-drawn bbox [x1,y1,x2,y2],
        crop with padding, re-run detection on the crop (so the detector gets a
        second chance now the face fills more of the frame + proper landmarks
        for alignment), and return an embedding translated back to original
        image coordinates (bbox=None in the result signals "use the caller's
        original bbox"). Three tiers, best first:

          1. InsightFace finds the face in the padded crop -> aligned embedding.
          2. It doesn't, but PCN does -> the face is in-plane rotated, so warp it
             upright and re-detect; we still get real landmarks, plus the angle.
          3. Neither -> an unaligned resize, which is barely an embedding at all.
        """
        crop, cx1, cy1 = _padded_crop(img, bbox, pad_ratio)

        raw_faces = self.app.get(crop) if crop.size else None
        face = max(raw_faces, key=lambda f: f.det_score) if raw_faces else None
        if face is not None:
            fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox]
            return {
                'bbox': [fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1],
                'embedding': face.normed_embedding.astype(np.float32),
                'det_score': float(face.det_score),
                'angle': 0.0,
            }

        rotated = self._embed_rotated_crop(crop, cx1, cy1) if crop.size else None
        if rotated is not None:
            return rotated

        import cv2
        # LAST RESORT: an UNALIGNED 112x112 resize straight into ArcFace. With no
        # landmarks the face is never warped onto the canonical template, so this
        # vector is only loosely comparable to properly aligned ones — it exists so
        # a hand-drawn box still yields *something*, not because it is any good.
        resized = cv2.resize(crop, (112, 112))
        feat = self.app.models['recognition'].get_feat(resized)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        feat = feat / np.linalg.norm(feat)  # get_feat() is raw too — normalize to unit length
        return {
            'bbox': None,  # caller falls back to the user-drawn bbox
            'embedding': feat,
            'det_score': 0.0,
            'angle': 0.0,
        }

    def _embed_rotated_crop(self, crop, cx1, cy1):
        """Tier 2 of embed_bbox: let PCN look for an in-plane-rotated face inside a
        crop InsightFace saw nothing in, then embed it from the upright warp. This
        is what keeps the manual "rotate and re-score" path off the unaligned
        fallback. Returns a face dict in original-image coordinates, or None."""
        try:
            import cv2
            pcn = self._get_pcn()
            if pcn is None:
                return None
            # A face that fills its whole frame is invisible to BOTH detectors
            # (measured: PCN found nothing in a bare face crop, and found it with
            # conf 1.0 once the crop had a border). web.py's rotate-and-rescore
            # path hands us exactly such a crop — the padded re-detect above has
            # no room to pad — so manufacture the missing context. Replicated
            # edges, not black: the detectors respond to them far better.
            b = max(1, int(max(crop.shape[:2]) * PCN_CROP_BORDER))
            padded = cv2.copyMakeBorder(crop, b, b, b, b, cv2.BORDER_REPLICATE)
            cands = [c for c in pcn.detect(padded) if c['conf'] >= PCN_CORRECT_MIN_CONF]
            if not cands:
                return None
            cand = max(cands, key=lambda c: c['conf'])
            got = self._embed_warped(pcn, padded, cand, cand['angle'])
            if got is None:
                return None
            # PCN's rotated square -> axis-aligned bounds, out of the border's
            # frame and back into the original image's.
            h, w = crop.shape[:2]
            xs = [min(max(p[0] - b, 0), w) for p in cand['points']]
            ys = [min(max(p[1] - b, 0), h) for p in cand['points']]
            return {
                'bbox': [float(min(xs)) + cx1, float(min(ys)) + cy1,
                         float(max(xs)) + cx1, float(max(ys)) + cy1],
                'embedding': got['embedding'],
                'det_score': got['det_score'],
                'angle': _norm_angle(-cand['angle']),
            }
        except Exception:
            return None

    @staticmethod
    def model_id(model_name='buffalo_l') -> str:
        return f'insightface-{model_name}'
