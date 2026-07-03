"""YOLO-World based image detector for open-vocabulary object search."""
import os
import time

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

VOCAB = [
#should not be defined here insgtead be read from .media/search_terms.txt newline seperated
]


class YOLOWorldDetector:
    def __init__(self, model_size='s', conf_threshold=0.15):
        from ultralytics import YOLOWorld as _YOLOWorld
        self.conf_threshold = conf_threshold
        self._model_size = model_size
        self.model = _YOLOWorld(f'yolov8{model_size}-worldv2.pt')
        self.model.set_classes(VOCAB)

    def detect_images(self, paths):
        """
        Run detection on a list of image paths.
        Returns list of (path, detections, error):
          detections = [(class_name, confidence, x1, y1, x2, y2), ...]
          error      = None on success, str on failure
        """
        results_out = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                results_out.append((path, [], "unsupported extension"))
                continue
            try:
                results = self.model.predict(path, conf=self.conf_threshold, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = VOCAB[cls_id] if cls_id < len(VOCAB) else self.model.names.get(cls_id, "unknown")
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                        detections.append((class_name, confidence, x1, y1, x2, y2))
                results_out.append((path, detections, None))
            except Exception as e:
                results_out.append((path, [], str(e)))
        return results_out

    @staticmethod
    def model_id(model_size='s'):
        return f"yolo-world-v2-{model_size}"
