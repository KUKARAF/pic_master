"""YOLO-World based image detector for open-vocabulary object search."""
import os

from .formats import IMAGE_EXTENSIONS as SUPPORTED_EXTENSIONS

# Default vocabulary used when .media/search_terms.txt does not exist.
# Add/remove terms freely, or create .media/search_terms.txt (one term per line)
# to override this list entirely without touching code.
DEFAULT_VOCAB = [
    # People & activities
    "person", "child", "crowd", "group of people", "selfie", "portrait",
    "wedding", "graduation", "birthday party", "sports",
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "pig", "rabbit",
    "fish", "bear", "deer", "fox", "wolf", "lion", "tiger", "elephant",
    "monkey", "chicken", "duck", "butterfly", "bee",
    # Nature & landscape
    "tree", "forest", "flower", "grass", "mountain", "river", "lake",
    "ocean", "beach", "sunset", "sunrise", "sky", "cloud", "snow",
    "rain", "desert", "waterfall", "field", "garden", "park",
    # Food & drink
    "food", "pizza", "burger", "sandwich", "salad", "cake", "bread",
    "fruit", "apple", "banana", "vegetables", "coffee", "wine", "beer",
    # Vehicles & transport
    "car", "truck", "bus", "motorcycle", "bicycle", "boat", "airplane",
    "train", "helicopter", "scooter",
    # Buildings & places
    "house", "building", "church", "bridge", "road", "street",
    "skyscraper", "tower", "castle", "ruins", "market", "stadium",
    # Everyday objects
    "book", "phone", "laptop", "camera", "tv", "chair", "table",
    "sofa", "bed", "lamp", "bottle", "cup", "bag", "shoes",
    "watch", "glasses", "hat", "umbrella", "clock",
    # Art & misc
    "painting", "sculpture", "graffiti", "flag", "fire", "fireworks",
    "map", "sign", "text", "logo",
]


def load_vocab_from_file(vocab_path=None):
    """Return vocabulary list. Reads vocab_path if given and it exists, else DEFAULT_VOCAB."""
    if vocab_path and os.path.isfile(vocab_path):
        with open(vocab_path) as f:
            terms = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if terms:
            return terms
    return DEFAULT_VOCAB


def merge_vocab(base_vocab, extra_labels=None, exclude=None):
    """Combine a base vocabulary with additional labels (e.g. human-confirmed tags
    from manual.db), case-insensitively deduped, dropping anything in `exclude`.

    This is also the practical answer to "can YOLO-World be told not to look for
    something" — it has no native negative-prompt concept, so the only real way to
    stop it finding a class is to simply never put that class in the vocabulary
    handed to set_classes()."""
    exclude_set = {e.lower() for e in (exclude or [])}
    seen = set()
    result = []
    for label in list(base_vocab) + list(extra_labels or []):
        key = label.lower()
        if key in exclude_set or key in seen:
            continue
        seen.add(key)
        result.append(label)
    return result


# Shared cache dir so the checkpoint is downloaded once and reused across every
# .media repo, instead of once per cwd (ultralytics downloads relative to cwd
# when given a bare filename).
WEIGHTS_DIR = os.path.expanduser('~/.cache/media_manager/weights')


class YOLOWorldDetector:
    def __init__(self, model_size='s', conf_threshold=0.15, vocab_path=None, vocab=None):
        from ultralytics import YOLOWorld as _YOLOWorld
        self.conf_threshold = conf_threshold
        self._model_size = model_size
        self.vocab = vocab if vocab is not None else load_vocab_from_file(vocab_path)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        weights_path = os.path.join(WEIGHTS_DIR, f'yolov8{model_size}-worldv2.pt')
        self.model = _YOLOWorld(weights_path)
        self.model.set_classes(self.vocab)

    def set_vocab(self, vocab):
        """Update classes on the already-loaded model — cheap (re-embeds the class
        names) compared to constructing a fresh YOLOWorldDetector() (which loads the
        full checkpoint). Lets one cached instance serve requests with a different
        vocabulary per call, e.g. a specific image's negative-tag-excluded list."""
        self.vocab = vocab
        self.model.set_classes(vocab)

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
                        class_name = self.vocab[cls_id] if cls_id < len(self.vocab) else self.model.names.get(cls_id, "unknown")
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
