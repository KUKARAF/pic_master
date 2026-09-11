"""In-plane face rotation normalization.

Covers: the shipped .onnx weights load and run (torch-free); the ANGLE SIGN
CONVENTION the whole app depends on (faces.angle, /face-crop rendering, the
renormalization backfill) proven both against PCN's own crop_face geometry and,
when the real models happen to be installed, end to end on a rotated photo;
FaceDetector's PCN correct+recover merge pass; the quarter-turn backstop; and
normalize_faces. Everything except section [3] runs on fakes, so no model
download or real face image is needed.
"""
import os, sys, math
import numpy as np
import cv2

FAILS = []
def check(n, c, e=''):
    print(f"  {'PASS' if c else 'FAIL'}: {n}" + (f"  [{e}]" if e else ''))
    if not c: FAILS.append(n)


print("[1] shipped ONNX weights load and run without torch")
assert 'torch' not in sys.modules, "torch must not be imported by the PCN path"
from media_manager.pcn import PCNDetector
det = PCNDetector()  # loads media_manager/pcn/pcn{1,2,3}.onnx
check('3 onnx sessions loaded', len(det.nets) == 3)
blank = np.zeros((200, 300, 3), dtype=np.uint8)
check('blank image detects nothing (no crash)', det.detect(blank) == [])
check('torch was never imported', 'torch' not in sys.modules)


print("\n[2] angle convention: PCN's frame vs ours")
from media_manager.pcn.pcn_detect import Window, crop_face
from media_manager.face_detector import _norm_angle, _kps_angle

# A synthetic "face": white on the TOP half of a square, black below. After a
# correct upright warp the white must be back on top.
S = 400
base = np.zeros((S, S, 3), np.uint8)
base[100:200, 100:300] = 255

def rotate_ccw(img, deg):
    """Rotate image content COUNTER-clockwise by deg (cv2 and PIL both read a
    positive angle as counter-clockwise)."""
    M = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (S, S))

def upright_score(crop):
    """> 0 when the bright half is on top, i.e. the crop came out upright."""
    h = crop.shape[0]
    return float(crop[:h // 2].mean() - crop[h // 2:].mean())

# PCN's Window.angle is the COUNTER-clockwise roll: content rotated CCW by D is
# warped upright by crop_face with angle = +D, and made worse by -D.
for D in (30, 90, -30):
    rot = rotate_ccw(base, D)
    same = upright_score(crop_face(rot, Window(100, 100, 200, D, 0.99), 200)[0])
    flip = upright_score(crop_face(rot, Window(100, 100, 200, -D, 0.99), 200)[0])
    check(f'crop_face(angle=+{D}) undoes a {D} deg CCW rotation', same > 200, f'{same:.0f}')
    check(f'crop_face(angle=-{D}) does not', flip < same, f'{flip:.0f} vs {same:.0f}')
    # Which fixes our own convention: angle = CLOCKWISE roll = -PCN's angle.
    check(f'our angle for a {D} deg CCW roll is {-D}', _norm_angle(-D) == float(-D))

check('_norm_angle folds into (-180, 180]', [_norm_angle(v) for v in (0, 180, -180, 190, 350, 540)]
      == [0.0, 180.0, 180.0, -170.0, -10.0, 180.0])

# The landmark-implied roll must read in the SAME convention, since the detector
# uses it to decide whether an incumbent alignment is rotated.
def kps_for(cx, cy, roll, d=20.0, e=8.0):
    """5 InsightFace-style landmarks for a face rolled `roll` degrees clockwise."""
    t = math.radians(roll)
    down = (-math.sin(t), math.cos(t))
    right = (math.cos(t), math.sin(t))
    eye = (cx - down[0] * d, cy - down[1] * d)
    mouth = (cx + down[0] * d, cy + down[1] * d)
    return np.array([
        [eye[0] - right[0] * e, eye[1] - right[1] * e],
        [eye[0] + right[0] * e, eye[1] + right[1] * e],
        [cx, cy],
        [mouth[0] - right[0] * e, mouth[1] - right[1] * e],
        [mouth[0] + right[0] * e, mouth[1] + right[1] * e],
    ], dtype=np.float32)

for roll in (0.0, 30.0, -45.0, 90.0, 180.0):
    got = _kps_angle(kps_for(50, 50, roll))
    check(f'_kps_angle reads a {roll} deg clockwise roll', abs(_norm_angle(got - roll)) < 1e-3,
          f'{got:.2f}')


print("\n[3] angle sign end to end on a real rotated photo (needs insightface + models)")

def _real_setup():
    """A real detector plus a real photo with detectable faces, or (None, why)."""
    try:
        import insightface
        from media_manager.face_detector import FaceDetector
        photo = os.path.join(os.path.dirname(insightface.__file__),
                             'data', 'images', 't1.jpg')
        if not os.path.exists(photo):
            return None, f'no sample photo at {photo}'
        img = cv2.imread(photo)
        if img is None:
            return None, 'sample photo unreadable'
        return (FaceDetector(), img), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'

real, why = _real_setup()
if real is None:
    print(f"  SKIP: real models unavailable ({why}).")
    print("  The convention is still pinned by section [2] (PCN crop_face geometry)"
          " and [5] (quarter-turn mapping).")
else:
    import tempfile
    real_det, photo = real
    for applied_ccw, expected in ((90.0, -90.0), (180.0, 180.0)):
        h, w = photo.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), applied_ccw, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2.0 - w / 2.0
        M[1, 2] += nh / 2.0 - h / 2.0
        tmp = os.path.join(tempfile.gettempdir(), f'pcn_rot_{int(applied_ccw)}.jpg')
        cv2.imwrite(tmp, cv2.warpAffine(photo, M, (nw, nh)))
        _p, faces, err = real_det.detect_faces([tmp])[0]
        os.unlink(tmp)
        angles = [f['angle'] for f in faces if f['angle'] != 0.0]
        off = [abs(_norm_angle(a - expected)) for a in angles]
        check(f'photo rotated {applied_ccw:.0f} deg CCW -> most faces report angle '
              f'{expected:.0f}', err is None and len(angles) >= max(1, len(faces) // 2),
              f'{len(angles)}/{len(faces)} corrected, err={err}')
        check(f'every reported angle is within 35 deg of {expected:.0f}',
              bool(off) and max(off) <= 35.0,
              ', '.join(f'{a:.1f}' for a in angles) or 'none')


print("\n[4] PCN correct + recover merge pass (fakes)")
from media_manager.face_detector import (
    FaceDetector, PCN_MAX_RECOVER, PCN_MIN_CONF, PCN_CORRECT_MIN_CONF,
    PCN_CORRECT_MIN_ANGLE, ROTATION_CORRECT_MARGIN, ROTATION_TTA_TRIGGER,
    ROTATION_ALIGN_TOLERANCE)

class FakeFace:
    """Stands in for an insightface Face: score, embedding, box and landmarks."""
    def __init__(self, score, emb_k=0, roll=0.0, shape=(200, 200)):
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0
        half = min(h, w) * 0.2
        self.det_score = score
        self.bbox = np.array([cx - half, cy - half, cx + half, cy + half], np.float32)
        self.kps = kps_for(cx, cy, roll, d=half * 0.5, e=half * 0.2)
        e = np.zeros(512, np.float32); e[emb_k] = 1.0
        self.normed_embedding = e

def cand(bbox, conf, angle=90):
    x1, y1, x2, y2 = bbox
    return {'bbox': [float(x1), float(y1), float(x2), float(y2)], 'angle': angle,
            'conf': conf, 'points': [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]}

class FakePCN:
    """Records which angle the caller asked the warp for, so the fake recognition
    app below can answer differently for the rotated read and the flat baseline."""
    def __init__(self, cands):
        self._c = cands
        self.last_angle = None
    def detect(self, img):
        return list(self._c)
    def upright_crop(self, img, win, size=200):
        self.last_angle = float(win['angle'] if isinstance(win, dict) else win.angle)
        return np.zeros((size, size, 3), np.uint8)

class WarpApp:
    """Fake recognition app keyed on the warp angle PCN was last asked for:
    `reads` maps that angle to (det_score, emb_k, landmark_roll) or None."""
    def __init__(self, pcn, reads):
        self.pcn = pcn
        self.reads = reads
        self.calls = 0
    def get(self, crop):
        self.calls += 1
        spec = self.reads.get(self.pcn.last_angle)
        if spec is None:
            return []
        score, emb_k, roll = spec
        return [FakeFace(score, emb_k, roll, crop.shape)]

def make_det(app, pcn):
    d = FaceDetector.__new__(FaceDetector)
    d.app = app; d.det_thresh = 0.5; d._model_name = 'fake'
    d._pcn = pcn; d._pcn_failed = False
    return d

def face(bbox, score, emb_k=0, angle=0.0):
    e = np.zeros(512, np.float32); e[emb_k] = 1.0
    return {'bbox': [float(v) for v in bbox], 'embedding': e,
            'det_score': float(score), 'angle': angle}

img = np.zeros((500, 500, 3), np.uint8)

# a) THE BUG: InsightFace found the face but mis-aligned it. PCN reports a large
#    angle, the upright re-read is clearly better -> embedding REPLACED, bbox kept.
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 7, 0.0), 0.0: None}), pcn)
faces = [face([12, 12, 112, 112], 0.55, emb_k=0)]
claimed = d._pcn_correct_and_recover(img, faces)
check('overlapping mis-aligned face is corrected, not skipped', claimed == {0}, str(claimed))
check('embedding REPLACED from the upright warp', faces[0]['embedding'][7] == 1.0)
check('det_score replaced', faces[0]['det_score'] == 0.9, str(faces[0]['det_score']))
check('angle = -(PCN angle), i.e. clockwise roll', faces[0]['angle'] == -90.0,
      str(faces[0]['angle']))
check('InsightFace bbox preserved', faces[0]['bbox'] == [12.0, 12.0, 112.0, 112.0])
check('no face invented for an overlapping candidate', len(faces) == 1)

# b) Margin guardrail: the flat read of the same region is just as good (and its
#    own landmarks say upright), so the rotated read is rejected.
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.90, 7, 0.0),
                           0.0: (0.90 - ROTATION_CORRECT_MARGIN / 2, 3, 0.0)}), pcn)
faces = [face([12, 12, 112, 112], 0.55, emb_k=0)]
d._pcn_correct_and_recover(img, faces)
check('re-detect that misses the margin is rejected', faces[0]['embedding'][0] == 1.0)
check('rejected correction leaves angle 0.0', faces[0]['angle'] == 0.0)
check('rejected correction leaves det_score', faces[0]['det_score'] == 0.55)

# c) ...unless the flat read's OWN landmarks are rotated: then a score tie is no
#    defence (this is the upside-down face that kept a high det_score).
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=180)])
d = make_det(WarpApp(pcn, {180.0: (0.90, 7, 0.0), 0.0: (0.90, 3, 180.0)}), pcn)
faces = [face([12, 12, 112, 112], 0.88, emb_k=0)]
d._pcn_correct_and_recover(img, faces)
check('tie is overridden when the incumbent alignment is itself rotated',
      faces[0]['embedding'][7] == 1.0)
check('angle folds to 180', faces[0]['angle'] == 180.0, str(faces[0]['angle']))

# d) A near-upright overlapping candidate costs nothing at all.
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=PCN_CORRECT_MIN_ANGLE - 1)])
app = WarpApp(pcn, {})
d = make_det(app, pcn)
faces = [face([12, 12, 112, 112], 0.55, emb_k=0)]
d._pcn_correct_and_recover(img, faces)
check('near-upright overlap not corrected', faces[0]['angle'] == 0.0
      and faces[0]['embedding'][0] == 1.0)
check('near-upright overlap runs no recognition call', app.calls == 0, str(app.calls))

# e) A genuinely new face is recovered: bbox from PCN's corners, angle negated.
pcn = FakePCN([cand([200, 50, 300, 150], 0.99, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 5, 0.0)}), pcn)
faces = [face([0, 0, 40, 40], 0.9)]
claimed = d._pcn_correct_and_recover(img, faces)
check('missed rotated face recovered', len(faces) == 2 and claimed == {1}, str(claimed))
if len(faces) == 2:
    check('bbox = axis-aligned bounds of PCN corners',
          faces[1]['bbox'] == [200.0, 50.0, 300.0, 150.0], str(faces[1]['bbox']))
    check('recovered embedding comes from the upright warp', faces[1]['embedding'][5] == 1.0)
    check('recovered det_score from the upright warp', faces[1]['det_score'] == 0.9)
    check('recovered angle negates PCN', faces[1]['angle'] == -90.0)

# f) The two arms have different confidence floors: a candidate good enough to
#    correct with is NOT good enough to invent a face from.
mid = (PCN_MIN_CONF + PCN_CORRECT_MIN_CONF) / 2
pcn = FakePCN([cand([200, 50, 300, 150], mid, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 5, 0.0)}), pcn)
faces = []
d._pcn_correct_and_recover(img, faces)
check('mid-confidence candidate does not recover a new face', faces == [], str(faces))
pcn = FakePCN([cand([10, 10, 110, 110], mid, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 7, 0.0), 0.0: None}), pcn)
faces = [face([12, 12, 112, 112], 0.55)]
d._pcn_correct_and_recover(img, faces)
check('mid-confidence candidate still corrects an existing face',
      faces[0]['embedding'][7] == 1.0)
pcn = FakePCN([cand([10, 10, 110, 110], PCN_CORRECT_MIN_CONF - 0.01, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 7, 0.0), 0.0: None}), pcn)
faces = [face([12, 12, 112, 112], 0.55)]
d._pcn_correct_and_recover(img, faces)
check('below the correction floor nothing is touched', faces[0]['embedding'][0] == 1.0)

# g) A crop InsightFace still can't read is not adopted (PCN has no recognizer).
pcn = FakePCN([cand([200, 50, 300, 150], 0.99, angle=90)])
d = make_det(WarpApp(pcn, {}), pcn)
faces = []
d._pcn_correct_and_recover(img, faces)
check('unembeddable warp skipped', faces == [])

# h) Per-image cap covers both arms together.
many = [cand([(i % 20) * 25, (i // 20) * 25, (i % 20) * 25 + 20, (i // 20) * 25 + 20],
             0.99, angle=90) for i in range(PCN_MAX_RECOVER + 10)]
pcn = FakePCN(many)
d = make_det(WarpApp(pcn, {90.0: (0.9, 5, 0.0)}), pcn)
faces = []
d._pcn_correct_and_recover(img, faces)
check('merge pass capped at PCN_MAX_RECOVER', len(faces) == PCN_MAX_RECOVER, str(len(faces)))

# i) PCN blowing up never costs us normal detection.
class BoomPCN:
    def detect(self, img): raise RuntimeError('boom')
d = make_det(WarpApp(FakePCN([]), {}), BoomPCN())
faces = [face([12, 12, 112, 112], 0.55)]
check('PCN exception swallowed', d._pcn_correct_and_recover(img, faces) == set())
check('faces survive a PCN failure untouched', faces[0]['angle'] == 0.0)


print("\n[5] quarter-turn backstop (fakes)")

def _bright_side(crop):
    """Which side of the crop centre the bright band sits on — the fake app's
    stand-in for 'which way is the top of the head pointing'. Centroid-based, so
    it survives the padding the sweep adds and cv2's real rotations."""
    ys, xs = np.nonzero(crop[:, :, 0] > 128)
    if len(xs) == 0:
        return 'none'
    h, w = crop.shape[:2]
    dx, dy = xs.mean() - (w - 1) / 2.0, ys.mean() - (h - 1) / 2.0
    if abs(dx) > abs(dy):
        return 'left' if dx < 0 else 'right'
    return 'top' if dy < 0 else 'bottom'

# Landmark roll implied by each orientation: upright only when the bright band
# (top of the head) is at the top of the frame.
SIDE_ROLL = {'top': 0.0, 'left': -90.0, 'right': 90.0, 'bottom': 180.0}

class OrientedApp:
    """Reads a crop's orientation off its content, so the real cv2.rotate calls in
    the sweep drive real differences. `scores` maps the bright side to a score."""
    def __init__(self, scores, rolls=None):
        self.scores = scores
        self.rolls = rolls or SIDE_ROLL
        self.calls = 0
    def get(self, crop):
        self.calls += 1
        side = _bright_side(crop)
        score = self.scores.get(side)
        if not score:
            return []
        return [FakeFace(score, emb_k={'top': 1, 'left': 2, 'right': 3, 'bottom': 4}[side],
                         roll=self.rolls[side], shape=crop.shape)]

def striped(side):
    """500x500 image whose face box carries a bright band on `side` — i.e. the top
    of the head points that way in the source image."""
    im = np.zeros((500, 500, 3), np.uint8)
    if side == 'left':
        im[150:350, 150:190] = 255
    elif side == 'bottom':
        im[310:350, 150:350] = 255
    else:
        im[150:190, 150:350] = 255
    return im

BOX = [150, 150, 350, 350]

# a) Top of the head points LEFT: the face is rolled counter-clockwise 90, so the
#    reported clockwise angle must be -90, and the adopted read is the upright one.
app = OrientedApp({'top': 0.9, 'left': 0.5, 'right': 0.5, 'bottom': 0.5})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.55, emb_k=0)]
d._quarter_turn_backstop(striped('left'), faces, set())
check('sideways face swept and corrected', faces[0]['embedding'][1] == 1.0)
check('angle -90 for a head pointing left', faces[0]['angle'] == -90.0, str(faces[0]['angle']))
check('det_score taken from the adopted read', faces[0]['det_score'] == 0.9)
check('bbox untouched by the sweep', faces[0]['bbox'] == [float(v) for v in BOX])

# b) Upside down -> 180.
app = OrientedApp({'top': 0.9, 'left': 0.5, 'right': 0.5, 'bottom': 0.5})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.55, emb_k=0)]
d._quarter_turn_backstop(striped('bottom'), faces, set())
check('angle 180 for an upside-down face', faces[0]['angle'] == 180.0, str(faces[0]['angle']))

# c) The sweep picks the UPRIGHT read, not the best-scoring one — a sideways read
#    that scores higher must not win (it would stamp a wrong angle on the face).
app = OrientedApp({'top': 0.75, 'left': 0.95, 'right': 0.95, 'bottom': 0.95})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.55, emb_k=0)]
d._quarter_turn_backstop(striped('left'), faces, set())
check('higher-scoring sideways read does not win', faces[0]['angle'] == -90.0,
      str(faces[0]['angle']))

# d) A confidently-detected face is not swept at all.
app = OrientedApp({'top': 0.9, 'left': 0.5, 'right': 0.5, 'bottom': 0.5})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, ROTATION_TTA_TRIGGER, emb_k=0)]
d._quarter_turn_backstop(striped('left'), faces, set())
check('confident face skips the sweep entirely', app.calls == 0, str(app.calls))
check('confident face keeps its embedding and angle',
      faces[0]['embedding'][0] == 1.0 and faces[0]['angle'] == 0.0)

# e) A face the PCN pass already claimed is PCN's call, not the sweep's: the sweep
#    must not overwrite it with a contradictory quarter turn.
app = OrientedApp({'top': 0.9, 'left': 0.5, 'right': 0.5, 'bottom': 0.5})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.55, emb_k=0)]
d._quarter_turn_backstop(striped('left'), faces, {0})
check('PCN-claimed face skipped by the sweep', app.calls == 0 and faces[0]['angle'] == 0.0)

# f) Margin guardrail: every orientation reads as upright landmarks, and the best
#    rotation only ties the flat read -> rejected.
app = OrientedApp({'top': 0.60, 'left': 0.62, 'right': 0.62, 'bottom': 0.62},
                  rolls={k: 0.0 for k in SIDE_ROLL})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.55, emb_k=0)]
d._quarter_turn_backstop(striped('top'), faces, set())
check('rotation that misses the margin is rejected', faces[0]['embedding'][0] == 1.0)
check('rejected sweep leaves angle 0.0', faces[0]['angle'] == 0.0, str(faces[0]['angle']))

# g) An upright face whose stored vector is embed_bbox's unaligned last resort
#    (det_score 0.0) is re-embedded through the aligned upright path, angle 0.
app = OrientedApp({'top': 0.85, 'left': 0.0, 'right': 0.0, 'bottom': 0.0})
d = make_det(app, None); d._pcn_failed = True
faces = [face(BOX, 0.0, emb_k=0)]
d._quarter_turn_backstop(striped('top'), faces, set())
check('unaligned vector recomputed through the upright path',
      faces[0]['embedding'][1] == 1.0 and faces[0]['det_score'] == 0.85)
check('upright repair reports angle 0.0', faces[0]['angle'] == 0.0)


print("\n[6] normalize_faces (fakes)")

# A rotated stored row is repaired; the call is a pure function of the image.
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=90)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 7, 0.0), 0.0: None}), pcn)
stored = [face([12, 12, 112, 112], 0.55, emb_k=0)]
once = d.normalize_faces(img, stored)
check('normalize_faces returns one row per input row', len(once) == 1)
check('rotated stored face gets the corrected vector', once[0]['embedding'][7] == 1.0)
check('rotated stored face gets its angle', once[0]['angle'] == -90.0, str(once[0]['angle']))
check('bbox preserved', once[0]['bbox'] == [12.0, 12.0, 112.0, 112.0])
check('input dicts never mutated', stored[0]['angle'] == 0.0
      and stored[0]['embedding'][0] == 1.0)
twice = d.normalize_faces(img, once)
check('idempotent: second pass returns the same vector',
      float(np.dot(twice[0]['embedding'], once[0]['embedding'])) == 1.0)
check('idempotent: second pass returns the same angle', twice[0]['angle'] == once[0]['angle'])

# An upright face comes back untouched...
pcn = FakePCN([cand([10, 10, 110, 110], 0.99, angle=3)])
app = WarpApp(pcn, {})
d = make_det(app, pcn)
stored = [face([12, 12, 112, 112], 0.8, emb_k=2, angle=0.0)]
out = d.normalize_faces(img, stored)
check('upright face keeps its vector', out[0]['embedding'][2] == 1.0)
check('upright face keeps angle 0.0', out[0]['angle'] == 0.0)
check('upright face costs no recognition call', app.calls == 0, str(app.calls))

# ...and the backfill never invents rows it has no DB id for.
pcn = FakePCN([cand([200, 50, 300, 150], 0.99, angle=90),
               cand([10, 10, 110, 110], 0.99, angle=3)])
d = make_det(WarpApp(pcn, {90.0: (0.9, 5, 0.0)}), pcn)
stored = [face([12, 12, 112, 112], 0.8, emb_k=2)]
out = d.normalize_faces(img, stored)
check('normalize_faces recovers no new faces', len(out) == 1, str(len(out)))


print("\n" + ("ALL ROTATION TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S): {FAILS}"))
sys.exit(1 if FAILS else 0)
