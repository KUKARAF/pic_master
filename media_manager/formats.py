"""Shared supported file extensions."""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
# Web-playable video containers only — these all play inline in an HTML5 <video>
# element (unlike e.g. .mkv/.avi). Kept separate from IMAGE_EXTENSIONS so the ML
# pipelines (object index, CLIP embed, faces, EXIF) stay image-only and skip video.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v'}
# Everything worth scanning into the index (the scan gate uses this).
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
