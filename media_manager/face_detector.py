"""InsightFace-based face detector and embedder."""
import os
import json
import numpy as np

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

# PCN (rotation-invariant detector) recovery: only keep PCN faces InsightFace's own
# detector missed (its recall drops on in-plane-rotated faces), deduped against what
# InsightFace already found. High conf floor since PCN stage-3 is already precise.
PCN_MIN_CONF = 0.98
# A PCN face overlapping an InsightFace one (by IoU) is the same face — skip it.
PCN_DEDUP_IOU = 0.3
# Backstop on how many rotated faces we re-embed per image (a pathological crowd
# photo shouldn't fan out into dozens of extra recognition calls).
PCN_MAX_RECOVER = 30


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


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
          faces = [{'bbox': [x1,y1,x2,y2], 'embedding': np.ndarray, 'det_score': float}]
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
                raw_faces = self.app.get(img)
                faces = []
                boxes = []  # kept boxes, for PCN dedup
                for face in raw_faces:
                    if face.det_score < self.det_thresh:
                        continue
                    bbox = [float(v) for v in face.bbox]
                    faces.append({
                        'bbox': bbox,
                        # normed_embedding, not embedding — InsightFace's raw .embedding
                        # is NOT unit length, so a plain dot product isn't cosine
                        # similarity (scores can exceed 1, thresholds become meaningless).
                        'embedding': face.normed_embedding.astype(np.float32),
                        'det_score': float(face.det_score),
                    })
                    boxes.append(bbox)
                # Rotation-invariant recovery: InsightFace's detector misses heavily
                # in-plane-rotated faces (an upside-down / sideways person). PCN finds
                # faces at any angle; add the ones InsightFace missed, embedded from an
                # upright-warped crop so the ArcFace vector is correct.
                faces.extend(self._pcn_recover(img, boxes))
                results.append((path, faces, None))
            except Exception as exc:
                results.append((path, [], str(exc)))
        return results

    def _pcn_recover(self, img, existing_boxes):
        """Detect faces at any in-plane rotation with PCN (one pass), keep only those
        InsightFace didn't already find (IoU dedup), warp each upright and embed it
        with InsightFace. Returns a list of face dicts. Best-effort: any failure
        (PCN unavailable, a bad crop) is swallowed so normal detection still stands."""
        try:
            pcn = self._get_pcn()
            if pcn is None:
                return []
            candidates = [c for c in pcn.detect(img) if c['conf'] >= PCN_MIN_CONF]
            # Highest-confidence first so the per-image cap keeps the best faces.
            candidates.sort(key=lambda c: c['conf'], reverse=True)
            recovered = []
            for cand in candidates:
                if len(recovered) >= PCN_MAX_RECOVER:
                    break
                if any(_iou(cand['bbox'], b) >= PCN_DEDUP_IOU for b in existing_boxes):
                    continue  # InsightFace already has this face
                crop = pcn.upright_crop(img, cand, 200)
                if crop is None or crop.size == 0:
                    continue
                got = self.app.get(crop)
                if not got:
                    continue
                f = max(got, key=lambda x: x.det_score)
                if f.det_score < self.det_thresh:
                    continue
                # Original-image bbox = axis-aligned bounds of PCN's rotated square.
                xs = [p[0] for p in cand['points']]
                ys = [p[1] for p in cand['points']]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                recovered.append({
                    'bbox': bbox,
                    'embedding': f.normed_embedding.astype(np.float32),
                    'det_score': float(f.det_score),
                })
                existing_boxes.append(bbox)
            return recovered
        except Exception:
            return []

    def embed_bbox(self, img, bbox: list, pad_ratio: float = 0.3) -> dict:
        """Given a full-size BGR numpy image and a user-drawn bbox [x1,y1,x2,y2],
        crop with padding, re-run detection on the crop (so the detector gets a
        second chance now the face fills more of the frame + proper landmarks
        for alignment), and return an embedding translated back to original
        image coordinates. Falls back to an unaligned 112x112 resize + direct
        recognition-model call if no face is found even in the padded crop
        (bbox=None in the result signals "use the caller's original bbox").
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        crop = img[cy1:cy2, cx1:cx2]

        raw_faces = self.app.get(crop)
        if raw_faces:
            face = max(raw_faces, key=lambda f: f.det_score)
            fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox]
            return {
                'bbox': [fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1],
                'embedding': face.normed_embedding.astype(np.float32),
                'det_score': float(face.det_score),
            }

        import cv2
        resized = cv2.resize(crop, (112, 112))
        feat = self.app.models['recognition'].get_feat(resized)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        feat = feat / np.linalg.norm(feat)  # get_feat() is raw too — normalize to unit length
        return {
            'bbox': None,  # caller falls back to the user-drawn bbox
            'embedding': feat,
            'det_score': 0.0,
        }

    @staticmethod
    def model_id(model_name='buffalo_l') -> str:
        return f'insightface-{model_name}'
