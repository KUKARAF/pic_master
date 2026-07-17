"""EXIF metadata extraction — capture time and GPS coordinates, when present.
Kept separate from the file's filesystem modified_time, which just reflects the
last time the file itself was touched (e.g. after a copy/transfer) and isn't a
reliable stand-in for when the photo was actually taken."""
import time

# EXIF tag ids (standard, not Pillow-specific): DateTimeOriginal, DateTimeDigitized,
# DateTime — checked in this priority order since DateTimeOriginal is the most
# reliable "when the shutter was pressed" field when a camera provides one.
_DATE_TAGS = (36867, 36868, 306)
_GPS_IFD_TAG = 0x8825


def _parse_exif_datetime(raw):
    """EXIF datetimes look like '2023:05:17 14:30:00'. Return a unix timestamp or None."""
    try:
        return int(time.mktime(time.strptime(raw, '%Y:%m:%d %H:%M:%S')))
    except (ValueError, TypeError):
        return None


def _dms_to_degrees(dms, ref):
    """Convert EXIF's (degrees, minutes, seconds) + hemisphere ref to signed decimal degrees."""
    if not dms or len(dms) != 3:
        return None
    try:
        degrees, minutes, seconds = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    value = degrees + minutes / 60 + seconds / 3600
    if ref in ('S', 'W'):
        value = -value
    return value


def extract_exif_metadata(abs_path):
    """Return (taken_at, gps_lat, gps_lon) — each None if not present in the file's EXIF."""
    try:
        from PIL import Image
        with Image.open(abs_path) as img:
            exif = img.getexif()
            if not exif:
                return None, None, None

            taken_at = None
            for tag_id in _DATE_TAGS:
                raw = exif.get(tag_id)
                if raw:
                    taken_at = _parse_exif_datetime(raw)
                    if taken_at is not None:
                        break

            gps_lat = gps_lon = None
            gps_ifd = exif.get_ifd(_GPS_IFD_TAG)
            if gps_ifd:
                gps_lat = _dms_to_degrees(gps_ifd.get(2), gps_ifd.get(1))
                gps_lon = _dms_to_degrees(gps_ifd.get(4), gps_ifd.get(3))

            return taken_at, gps_lat, gps_lon
    except Exception:
        return None, None, None
