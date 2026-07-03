"""YOLO-World based image detector for open-vocabulary object search."""
import os
import time

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

VOCAB = [
    "girl", "baby", "toddler", "teenager", "adult", "elderly person",
    "cum", "asshole", "stockings", "red", "costume", "blowjob", "sex",
    "anal", "pain", "smile", "happy", "girls", "underwear", "wet"
    "wedding", "wedding dress", "wedding cake", "birthday party", "birthday cake",
    "toy", "doll", "teddy bear", "dildo", "latina", "feet up", "handjob", 
    "2 girls", "oil", "socks", "white", "black", "red", "picnic", "with mom", 
    "young girl with mom", "gangbang", "oral sex", "pussy","vagina", "bending over", 
    "bikini", "swimsuite", "school uniform", "school", "staircase", "skirt", 
    "curly" , "blond", "bangs", "sucking", "with mom", "mom teaching", "shower", 
    "bathtub", "multiple girls", "curled up", "legs open", "sitting on dick", "dick",
    "nonude", "foam", "long hair", "ready for sex",
    # animals
    "dog", "chicken", "bee",
    # nature & landscape
    "beach", "ocean", "sea", "waves", "sand", "sunset", "sunrise",
    "mountain", "hill", "valley", "cliff", "waterfall", "river", "lake",
    "forest", "trees", "jungle", "desert", "field", "meadow", "farm",
    "sky", "clouds", "rainbow", "lightning", "fog", "mist",
    "snow", "ice", "glacier", "rain", "storm",
    "flower", "rose", "grass", "leaves", "garden", "park",
    # food & drink
    "fruit", "vegetables"
    # vehicles & transport
    "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
    "boat", "sailboat", "ship", "ferry", "kayak",
    "airplane", "helicopter", "hot air balloon",
    "train", "subway", "tram",
    # places & architecture
    "building", "skyscraper", "house", "apartment", "cottage",
    "church", "mosque", "temple", "cathedral",
    "bridge", "road", "street", "alley", "highway",
    "restaurant", "cafe", "bar", "market", "shop", "mall",
    "school", "hospital", "library", "museum", "stadium",
    "swimming pool", "playground", "cemetery",
    "city", "village", "suburb", "countryside",
    "chair", "table", "sofa", "couch", "armchair", "bed", "desk",
    "phone", "smartphone", "laptop", "computer", "tablet", "camera",
    "television", "monitor", "keyboard",
    "book", "newspaper", "magazine",
    "bag", "backpack", "suitcase", "handbag",
    "umbrella", "hat", "sunglasses", "watch",
    "flag", "sign", "graffiti",
    "fire", "smoke", "fireworks",
    "window", "door", "stairs", "fence", "gate",
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
