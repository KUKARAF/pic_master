"""Shared supported file extensions."""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
# Video containers the library tracks. Kept separate from IMAGE_EXTENSIONS so the ML
# pipelines (object index, CLIP embed, faces, EXIF) stay image-only and skip video.
# Frames are decoded with cv2/ffmpeg, which reads all of these.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v', '.avi', '.mpg', '.mpeg'}
# Containers no browser plays inline: /photo goes straight to the cached H.264
# transcode (/playable/{id}) instead of waiting for the <video> element to fail.
TRANSCODE_EXTENSIONS = {'.avi', '.mpg', '.mpeg'}
# Explicit content types for serving video — don't rely on mimetypes.guess_type,
# which returns None for .m4v (and, on minimal systems without /etc/mime.types,
# possibly others), causing FileResponse to fall back to application/octet-stream
# and the HTML5 <video> element to reject the source as an unsupported MIME type.
VIDEO_MIME_TYPES = {
    '.mp4': 'video/mp4',
    '.m4v': 'video/mp4',      # M4V is an MPEG-4 container; browsers play it as video/mp4
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
    '.avi': 'video/x-msvideo',
    '.mpg': 'video/mpeg',
    '.mpeg': 'video/mpeg',
}
# Everything worth scanning into the index (the scan gate uses this).
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def open_upright_rgb(path_or_file):
    """Open an image as RGB with its EXIF orientation applied, i.e. in the frame the
    browser displays. Region/pattern boxes are drawn on that rotated view, so every
    crop made from them (and every tile laid down at indexing) must use this frame."""
    from PIL import Image, ImageOps  # hook-allow: lazy — keep this module import-light
    with Image.open(path_or_file) as im:
        out = ImageOps.exif_transpose(im).convert('RGB')
        out.load()
    return out
