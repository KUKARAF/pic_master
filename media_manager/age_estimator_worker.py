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
import sys

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


def detect_person_boxes(image):
    from ultralytics import YOLO

    model = YOLO(PERSON_MODEL_NAME)
    result = model.predict(image, classes=[PERSON_CLASS_ID], conf=PERSON_CONF_THRESHOLD, verbose=False)[0]
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


def run(image_path, faces):
    import cv2
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    person_boxes = detect_person_boxes(image)

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

    results = []
    # One at a time rather than batched: crops vary in whether a body is present, and
    # keeping the per-face try/except means one bad crop can't take out the whole
    # photo's results.
    for face_ref, face_crop_img, body_crop_img in zip(refs, face_crops, body_crops):
        if face_crop_img is None:
            results.append({"face_ref": face_ref, "age": None, "gender": None})
            continue
        try:
            faces_input = processor(images=[face_crop_img])["pixel_values"]
            body_input = processor(images=[body_crop_img])["pixel_values"]
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
