"""Shared supported file extensions + the lenient image opener."""
import io
import os
import zlib

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


_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_PNG_TEXT_CHUNKS = {b'tEXt', b'zTXt', b'iTXt'}


def png_without_bad_metadata(data):
    """PNG bytes with every text chunk (tEXt/zTXt/iTXt) and every ancillary chunk whose
    CRC doesn't match removed — critical chunks (IHDR/PLTE/IDAT/IEND) are copied as is.
    None when `data` isn't a PNG. Image viewers ignore exactly these problems, while
    Pillow refuses the whole file: a compressed text chunk over its MAX_TEXT_CHUNK (big
    AI-workflow / XMP metadata) or a bad CRC on a metadata chunk."""
    if not data.startswith(_PNG_SIGNATURE):
        return None
    out, pos = [_PNG_SIGNATURE], len(_PNG_SIGNATURE)
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], 'big')
        ctype = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            out.append(data[pos:])          # truncated: leave it to the decoder
            break
        ancillary = bool(ctype[0] & 0x20)
        crc_ok = (zlib.crc32(data[pos + 4:pos + 8 + length]) & 0xFFFFFFFF
                  == int.from_bytes(data[end - 4:end], 'big'))
        if not (ancillary and (ctype in _PNG_TEXT_CHUNKS or not crc_ok)):
            out.append(data[pos:end])
        pos = end
        if ctype == b'IEND':
            break
    return b''.join(out)


def open_rgb(path, upright=False):
    """Fully decoded RGB PIL image (EXIF orientation applied when `upright`). If Pillow
    rejects a PNG, retries once without its text / bad-CRC metadata chunks (see
    png_without_bad_metadata) and re-raises the ORIGINAL error if that fails too — so
    only genuinely undecodable pixels count as damaged."""
    from PIL import Image, ImageOps  # hook-allow: lazy — keep this module import-light

    def _decode(src):
        with Image.open(src) as im:
            out = (ImageOps.exif_transpose(im) if upright else im).convert('RGB')
            out.load()
        return out

    try:
        return _decode(path)
    except (OSError, ValueError, SyntaxError) as exc:
        if isinstance(exc, OSError) and exc.errno is not None:
            raise                            # missing / unreadable file, not bad data
        with open(path, 'rb') as fh:
            cleaned = png_without_bad_metadata(fh.read())
        if cleaned is None:
            raise
        try:
            return _decode(io.BytesIO(cleaned))
        except (OSError, ValueError, SyntaxError):
            raise exc from None


def open_upright_rgb(path_or_file):
    """Open an image as RGB with its EXIF orientation applied, i.e. in the frame the
    browser displays. Region/pattern boxes are drawn on that rotated view, so every
    crop made from them (and every tile laid down at indexing) must use this frame.
    Paths get open_rgb's lenient PNG retry."""
    if isinstance(path_or_file, (str, bytes, os.PathLike)):
        return open_rgb(path_or_file, upright=True)
    from PIL import Image, ImageOps  # hook-allow: lazy — keep this module import-light
    with Image.open(path_or_file) as im:
        out = ImageOps.exif_transpose(im).convert('RGB')
        out.load()
    return out
