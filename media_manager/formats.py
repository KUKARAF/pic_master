"""Shared supported file extensions."""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
# Web-playable video containers only — these all play inline in an HTML5 <video>
# element (unlike e.g. .mkv/.avi). Kept separate from IMAGE_EXTENSIONS so the ML
# pipelines (object index, CLIP embed, faces, EXIF) stay image-only and skip video.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v'}
# Explicit content types for serving video — don't rely on mimetypes.guess_type,
# which returns None for .m4v (and, on minimal systems without /etc/mime.types,
# possibly others), causing FileResponse to fall back to application/octet-stream
# and the HTML5 <video> element to reject the source as an unsupported MIME type.
VIDEO_MIME_TYPES = {
    '.mp4': 'video/mp4',
    '.m4v': 'video/mp4',      # M4V is an MPEG-4 container; browsers play it as video/mp4
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
}
# Everything worth scanning into the index (the scan gate uses this).
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
