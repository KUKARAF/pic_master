"""InsightFace-based face detector and embedder."""
import os
import json
import numpy as np

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

# A face that is confidently upright (det_score at/above this) is not re-checked for
# rotation — the sweep below only fires for the marginal ones, so upright faces cost
# nothing extra.
ROTATION_TTA_TRIGGER = 0.72
# Only adopt a rotated orientation when it beats the upright crop's det_score by this
# margin — avoids churning a barely-different embedding on a face that was fine.
ROTATION_TTA_MARGIN = 0.05


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
                for face in raw_faces:
                    if face.det_score < self.det_thresh:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in face.bbox]
                    # normed_embedding, not embedding — InsightFace's raw .embedding is
                    # NOT unit length, so a plain dot product against it isn't cosine
                    # similarity (scores can exceed 1, thresholds become meaningless).
                    embedding = face.normed_embedding.astype(np.float32)
                    det_score = float(face.det_score)
                    # An in-plane-rotated face (e.g. one person upside-down in an
                    # otherwise-upright photo) is often detected with wrong landmarks,
                    # so its aligned crop — and thus embedding — is scrambled and gets
                    # mislabeled. Re-crop just this face and pick the orientation the
                    # detector is most confident in. Only for the not-clearly-upright.
                    if det_score < ROTATION_TTA_TRIGGER:
                        better = self._best_oriented_embedding(img, [x1, y1, x2, y2], det_score)
                        if better is not None:
                            embedding, det_score = better
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'embedding': embedding,
                        'det_score': det_score,
                    })
                results.append((path, faces, None))
            except Exception as exc:
                results.append((path, [], str(exc)))
        return results

    def _best_oriented_embedding(self, img, bbox: list, base_score: float, pad_ratio: float = 0.3):
        """Re-crop the face at `bbox` and re-detect it at 0/90/180/270° in-plane
        rotations, returning `(embedding, det_score)` from whichever orientation the
        detector is most confident in — but ONLY when a rotated orientation beats the
        upright crop by ROTATION_TTA_MARGIN (else None, meaning "keep what you had").

        det_score is a reliable which-way-is-up oracle: the same face scores far
        higher upright than upside-down, and once the crop is upright InsightFace's
        landmark alignment produces a correct ArcFace embedding instead of the
        scrambled one a wrong-landmark detection gave. The bbox is left as the
        caller's (it located the face fine); only the embedding was the problem."""
        import cv2
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        # None == upright; the three cv2 codes are the lossless 90° steps.
        scores, embs = {}, {}
        for code in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rot = crop if code is None else cv2.rotate(crop, code)
            found = self.app.get(rot)
            if not found:
                continue
            f = max(found, key=lambda ff: ff.det_score)
            scores[code] = float(f.det_score)
            embs[code] = f.normed_embedding.astype(np.float32)
        if not scores:
            return None
        upright = scores.get(None, base_score)
        best_code = max(scores, key=scores.get)
        if best_code is not None and scores[best_code] > upright + ROTATION_TTA_MARGIN:
            return embs[best_code], scores[best_code]
        return None

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
