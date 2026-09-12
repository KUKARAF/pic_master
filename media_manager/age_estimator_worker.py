"""Runs INSIDE the isolated `.age-venv` (see age_estimator.py), never imported by the
main app process. MiVOLO's own code hard-pins an old timm/ultralytics that conflict
with this project's YOLO-World detector and CLIP indexer, so this whole feature lives
in its own virtualenv and talks to the main process over stdin/stdout JSON instead of
being importable in-process.

Input (stdin, one JSON object):
    {"image_path": "/abs/path/to/photo.jpg",
     "faces": [{"face_ref": "manual:1", "bbox": [x1, y1, x2, y2]}, ...]}

Output (stdout, one JSON object):
    {"results": [{"face_ref": "manual:1", "age": 27.4, "gender": "male"}, ...]}
    or {"error": "message"} on failure (nonzero exit code).

A face with no usable estimate (e.g. inference failed for just that one crop) still
appears in "results" with age/gender set to null, rather than being dropped silently.
"""
import json
import os
import sys

# torch >= 2.6 — required for Intel Arc Battlemage (B70) XPU support — defaults
# torch.load to weights_only=True, which the old ultralytics/MiVOLO checkpoints
# aren't written for and fail to load under. Force the pre-2.6 behavior BEFORE
# torch is imported anywhere below so both the YOLO person detector and the
# MiVOLO weights still load. Harmless on the old CPU-only torch (ignored there).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Person (body) detection — plain YOLOv8 person class, not YOLO-World; this is a
# different, much smaller model than this app's main object detector and is only
# ever loaded in this isolated process.
PERSON_MODEL_NAME = "yolov8n.pt"
PERSON_CLASS_ID = 0  # COCO "person"
PERSON_CONF_THRESHOLD = 0.5

# A face's body is whichever person box overlaps it enough, biased toward the face
# sitting in the box's upper portion (a body detector's box spans head-to-feet, so a
# real match's face bbox should be near the top of its person box, not floating
# outside it or centered in its lower half).
MIN_FACE_IN_BODY_OVERLAP = 0.7

MODEL_REPO = "iitolstykh/mivolo_v2"


def _iou_and_containment(face_bbox, person_bbox):
    """Returns the fraction of the face box's area that falls inside the person box —
    containment, not IoU, since a small face box inside a much larger body box has a
    tiny IoU but is obviously "in" that body."""
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
        score = _iou_and_containment(face_bbox, person_bbox)
        if score > best_score:
            best_score = score
            best_bbox = person_bbox
    if best_score >= MIN_FACE_IN_BODY_OVERLAP:
        return best_bbox
    return None


def detect_person_boxes(image, device=None):
    from ultralytics import YOLO

    model = YOLO(PERSON_MODEL_NAME)
    # device (e.g. 'xpu' for the Arc GPU) is threaded in from run(); None keeps
    # ultralytics' own auto-selection. Only inject it when set so behavior is
    # unchanged on CPU-only setups.
    predict_kwargs = dict(classes=[PERSON_CLASS_ID], conf=PERSON_CONF_THRESHOLD, verbose=False)
    if device is not None:
        predict_kwargs['device'] = device
    result = model.predict(image, **predict_kwargs)[0]
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((x1, y1, x2, y2))
    return boxes


def crop(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _pick_device(torch):
    """Choose a torch device string ('cuda'|'xpu'|'mps'|'cpu') for THIS venv.

    Deliberately duplicates the logic in media_manager/compute.py: this worker
    runs in the isolated .age-venv, which pins torch==2.5.1 and does NOT have the
    main package on its path, so it cannot `from media_manager import compute`
    (same precedent as this file's other duplicated helpers, e.g. the person
    detector living here instead of reusing detector.py). The channel from the
    parent is the inherited MEDIA_DEVICE env var (see age_estimator.py).

    Per the project's no-silent-failures rule: if a backend is requested or probed
    but isn't actually usable in this venv's torch, we print a loud one-line note
    to stderr and drop to CPU (visibly), never quietly. Since .age-venv pins torch
    2.5.1 without IPEX, 'xpu' will usually be unavailable here and this correctly
    falls back to CPU with a visible note — that is expected, not a bug."""
    want = os.environ.get("MEDIA_DEVICE", "").strip().lower()

    def _xpu_ok():
        # Native XPU (torch>=2.5) needs no IPEX, but older Intel stacks only
        # register the 'xpu' backend after importing IPEX — try it, guarded.
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except Exception:
            pass
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def _mps_ok():
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if want:
        if want not in ("cuda", "xpu", "mps", "cpu"):
            print(f"[age_worker] MEDIA_DEVICE={want!r} is not one of "
                  f"cuda|xpu|mps|cpu — using CPU.", file=sys.stderr)
            return "cpu"
        if want == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[age_worker] MEDIA_DEVICE=cuda but torch.cuda.is_available() "
                  "is False in .age-venv — using CPU.", file=sys.stderr)
            return "cpu"
        if want == "xpu":
            if _xpu_ok():
                return "xpu"
            print("[age_worker] MEDIA_DEVICE=xpu but torch.xpu is unavailable in "
                  ".age-venv (its pinned torch==2.5.1 has no IPEX/XPU build) — "
                  "using CPU.", file=sys.stderr)
            return "cpu"
        if want == "mps":
            if _mps_ok():
                return "mps"
            print("[age_worker] MEDIA_DEVICE=mps but torch's MPS backend is "
                  "unavailable in .age-venv — using CPU.", file=sys.stderr)
            return "cpu"
        return "cpu"

    # No MEDIA_DEVICE set: auto-probe cuda -> xpu -> cpu. mps is skipped unless
    # explicitly requested (matches compute.py's cautious auto-order intent for
    # this always-headless worker).
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_ok():
        return "xpu"
    return "cpu"


def run(image_path, faces):
    import cv2
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # Pick the device this venv can actually use (see _pick_device) once, up front,
    # so BOTH the YOLO person detector and the MiVOLO model run on it (e.g. the Arc
    # GPU via 'xpu'). A model and its input tensors must share a device or torch
    # raises, so the same `dev` is reused for model.to()/inputs below.
    dev = _pick_device(torch)

    person_boxes = detect_person_boxes(image, device=dev)

    face_crops = []
    body_crops = []
    refs = []
    for face in faces:
        face_bbox = face["bbox"]
        face_crop_img = crop(image, face_bbox)
        if face_crop_img is None:
            # Nothing usable for this face — still report it, just with no estimate.
            refs.append(face["face_ref"])
            face_crops.append(None)
            body_crops.append(None)
            continue
        matched_body_bbox = match_face_to_body(face_bbox, person_boxes)
        body_crop_img = crop(image, matched_body_bbox) if matched_body_bbox else None
        refs.append(face["face_ref"])
        face_crops.append(face_crop_img)
        body_crops.append(body_crop_img)

    config = AutoConfig.from_pretrained(MODEL_REPO, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL_REPO, trust_remote_code=True, dtype=torch.float32)
    processor = AutoImageProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
    model.eval()
    # Move the MiVOLO model onto the device chosen above. The per-face
    # pixel_values tensors are moved to the same device just before each forward
    # pass below.
    model.to(dev)

    results = []
    # One at a time rather than batched: crops vary in whether a body is present, and
    # keeping the per-face try/except means one bad crop can't take out the whole
    # photo's results.
    for face_ref, face_crop_img, body_crop_img in zip(refs, face_crops, body_crops):
        if face_crop_img is None:
            results.append({"face_ref": face_ref, "age": None, "gender": None})
            continue
        try:
            # processor(...)["pixel_values"] is a tensor; move it onto the same
            # device as the model (a no-op when dev == 'cpu').
            faces_input = processor(images=[face_crop_img])["pixel_values"].to(dev)
            body_input = processor(images=[body_crop_img])["pixel_values"].to(dev)
            with torch.no_grad():
                output = model(faces_input=faces_input, body_input=body_input)
            age = round(output.age_output[0].item(), 1)
            gender = config.gender_id2label[output.gender_class_idx[0].item()]
            results.append({"face_ref": face_ref, "age": age, "gender": gender})
        except Exception as exc:  # noqa: BLE001 - one bad face must not sink the batch
            results.append({"face_ref": face_ref, "age": None, "gender": None, "error": str(exc)})

    return results


def main():
    try:
        payload = json.loads(sys.stdin.read())
        results = run(payload["image_path"], payload["faces"])
        json.dump({"results": results}, sys.stdout)
    except Exception as exc:  # noqa: BLE001 - report, don't traceback, to the caller
        json.dump({"error": str(exc)}, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
