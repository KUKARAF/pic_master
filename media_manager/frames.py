"""Frame extraction for multi-frame image formats (animated GIF / WEBP).

Every ML pipeline in this codebase (face_detector.py's cv2.imread, indexer.py's
PIL.Image.open, detector.py's Ultralytics predict) only ever reads frame 0 of a
multi-frame file — there's no other frame-reading code anywhere else in the project.
This module is the one shared place that knows how to look past frame 0.
"""
from PIL import Image


def get_frame_count(path) -> int:
    """Return how many frames `path` has. 1 for any non-animated (or single-frame)
    image — Pillow's n_frames reads the file's frame directory, it doesn't decode
    every frame, so this is cheap even for a large animation."""
    with Image.open(path) as img:
        return getattr(img, 'n_frames', 1)


def extract_frame(path, frame_index: int):
    """Return frame `frame_index` of `path` as an RGB PIL.Image, or None if that
    frame doesn't exist / the file can't be read."""
    try:
        with Image.open(path) as img:
            img.seek(frame_index)
            return img.convert('RGB')
    except (EOFError, OSError, ValueError):
        return None
