"""PCN rotation-invariant detection integration.

Covers: the shipped .onnx weights load and run (torch-free), and FaceDetector's
_pcn_recover merge logic (dedup vs InsightFace, upright-warp + embed, bbox mapping,
per-image cap, confidence floor) — using fakes so no real face image or InsightFace
model is needed.
"""
import os, sys
import numpy as np

FAILS = []
def check(n, c, e=''):
    print(f"  {'PASS' if c else 'FAIL'}: {n}" + (f"  [{e}]" if e else ''))
    if not c: FAILS.append(n)


print("[1] shipped ONNX weights load and run without torch")
import importlib
assert 'torch' not in sys.modules, "torch must not be imported by the PCN path"
from media_manager.pcn import PCNDetector
det = PCNDetector()  # loads media_manager/pcn/pcn{1,2,3}.onnx
check('3 onnx sessions loaded', len(det.nets) == 3)
blank = np.zeros((200, 300, 3), dtype=np.uint8)
check('blank image detects nothing (no crash)', det.detect(blank) == [])
check('torch was never imported', 'torch' not in sys.modules)


print("\n[2] FaceDetector._pcn_recover merge logic (fakes)")
from media_manager.face_detector import FaceDetector, PCN_MAX_RECOVER

class FakeFace:
    def __init__(self, score, emb_k=0):
        self.det_score = score
        e = np.zeros(512, np.float32); e[emb_k] = 1.0
        self.normed_embedding = e

class FakeApp:
    def __init__(self, score=0.9): self.score = score
    def get(self, crop): return [FakeFace(self.score)] if self.score else []

def cand(bbox, conf, angle=90):
    x1, y1, x2, y2 = bbox
    return {'bbox': [float(x1), float(y1), float(x2), float(y2)], 'angle': angle,
            'conf': conf, 'points': [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]}

class FakePCN:
    def __init__(self, cands): self._c = cands
    def detect(self, img): return self._c
    def upright_crop(self, img, c, size=200): return np.zeros((size, size, 3), np.uint8)

def make_det(app, pcn):
    d = FaceDetector.__new__(FaceDetector)
    d.app = app; d.det_thresh = 0.5; d._model_name = 'fake'
    d._pcn = pcn; d._pcn_failed = False
    return d

img = np.zeros((500, 500, 3), np.uint8)

# a) a PCN face overlapping an existing InsightFace box is deduped away
d = make_det(FakeApp(0.9), FakePCN([cand([10, 10, 110, 110], 0.99)]))
out = d._pcn_recover(img, existing_boxes=[[12, 12, 112, 112]])  # ~same box
check('overlapping face deduped (0 recovered)', len(out) == 0, str(len(out)))

# b) a genuinely new face is recovered, bbox = bounds of PCN points, embedding from crop
d = make_det(FakeApp(0.9), FakePCN([cand([200, 50, 300, 150], 0.99)]))
out = d._pcn_recover(img, existing_boxes=[[0, 0, 40, 40]])
check('new rotated face recovered', len(out) == 1, str(len(out)))
if out:
    check('bbox = axis-aligned bounds of PCN corners', out[0]['bbox'] == [200.0, 50.0, 300.0, 150.0], str(out[0]['bbox']))
    check('embedding carried from the upright crop', out[0]['embedding'].shape == (512,))
    check('det_score from the upright crop', out[0]['det_score'] == 0.9)

# c) confidence floor: a below-PCN_MIN_CONF candidate is dropped
d = make_det(FakeApp(0.9), FakePCN([cand([200, 50, 300, 150], 0.5)]))
check('low-confidence PCN face dropped', d._pcn_recover(img, []) == [])

# d) crop that InsightFace still can't embed (empty) is skipped
d = make_det(FakeApp(0.0), FakePCN([cand([200, 50, 300, 150], 0.99)]))
check('unembeddable crop skipped', d._pcn_recover(img, []) == [])

# e) per-image cap honored (non-overlapping boxes so dedup doesn't trim them first)
many = [cand([(i % 20) * 50, (i // 20) * 50, (i % 20) * 50 + 40, (i // 20) * 50 + 40], 0.99)
        for i in range(PCN_MAX_RECOVER + 10)]
d = make_det(FakeApp(0.9), FakePCN(many))
check('recovery capped at PCN_MAX_RECOVER', len(d._pcn_recover(img, [])) == PCN_MAX_RECOVER,
      str(len(d._pcn_recover(img, []))))

# f) PCN failure never breaks detection (returns [])
class BoomPCN:
    def detect(self, img): raise RuntimeError('boom')
d = make_det(FakeApp(0.9), BoomPCN())
check('PCN exception swallowed', d._pcn_recover(img, []) == [])

print("\n" + ("ALL PCN TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S): {FAILS}"))
sys.exit(1 if FAILS else 0)
