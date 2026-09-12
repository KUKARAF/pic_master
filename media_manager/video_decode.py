"""Hardware-accelerated video decode for the Intel Arc (Battlemage) media engine.

Every video path in this project decodes with `cv2.VideoCapture(path)`, which hands
the stream to FFmpeg's SOFTWARE decoders. That burns CPU we don't need to burn: the
Arc GPU has fixed-function HEVC/H.264/AV1/VP9 decode blocks sitting idle. This module
routes captures through the GPU media engine (VAAPI or QSV) when it's available, and
falls back to software LOUDLY when it isn't — losing the accelerator is a performance
regression we always want to see, but it must never break decoding (video features
still have to work on a CPU-only box, e.g. this dev machine with no GPU / no libGL).

Design notes:
- torch-free by design; this is the plain-cv2 decode path.
- cv2 is imported lazily INSIDE the functions, mirroring phasher.py / the thumbnail
  path: importing this module must not drag in (or fail on) opencv. On boxes where
  cv2 can't even import (no libGL), `describe()` still answers without touching cv2.
- The cv2 HW-accel constants (CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_VAAPI, ...)
  only exist in opencv >= ~4.5.2 AND only when the wheel was built with the VideoIO
  hardware-accel API. Every one of them is fetched with getattr and guarded; a missing
  constant means "HW unavailable" — we warn once and use software, never crash.
- "No silent failures": any time we ask for HW and don't get it, we print a one-time,
  explicit stderr warning naming what was requested vs. what actually engaged.
"""

import os
import sys

# --- Configuration ---------------------------------------------------------------

# MEDIA_VIDEO_HWACCEL selects the decode backend:
#   auto  (default) — try VAAPI, then QSV, then software
#   vaapi           — try VAAPI only, else software
#   qsv             — try QSV only, else software
#   none            — plain software cv2.VideoCapture(path), no HW attempt at all
_VALID_MODES = ('auto', 'vaapi', 'qsv', 'none')


def hwaccel_requested() -> str:
    """The normalised MEDIA_VIDEO_HWACCEL mode ('auto'|'vaapi'|'qsv'|'none').

    An unrecognised value is treated as 'auto' and warned about once (below), so a
    typo in the env var degrades to the sensible default instead of silently doing
    something surprising.
    """
    raw = (os.environ.get('MEDIA_VIDEO_HWACCEL') or 'auto').strip().lower()
    if raw not in _VALID_MODES:
        _warn_once(
            f"MEDIA_VIDEO_HWACCEL={raw!r} is not one of {_VALID_MODES}; "
            "using 'auto'."
        )
        return 'auto'
    return raw


# --- One-time, loud warning plumbing ---------------------------------------------

# Module-level flags so the warnings print ONCE per process, not once per video —
# otherwise a bulk job over thousands of clips would drown stderr in identical lines.
_warned_msgs = set()          # de-dupe arbitrary warning strings
_engaged_backend = None       # cache of the backend that actually engaged (or 'software')


def _warn_once(msg: str) -> None:
    """Print `msg` to stderr exactly once per process (keyed on the message text)."""
    if msg in _warned_msgs:
        return
    _warned_msgs.add(msg)
    print(f"[video_decode] WARNING: {msg}", file=sys.stderr, flush=True)


# --- Backend probing --------------------------------------------------------------

def _accel_constant(cv2, mode):
    """Return (accel_enum, missing_names) for a HW mode.

    accel_enum is the cv2.VIDEO_ACCELERATION_* value to pass, or None if any constant
    this path needs is absent from the installed opencv. missing_names lists what was
    missing (for the loud warning). We check BOTH the property id (CAP_PROP_HW_*) and
    the acceleration enum, because an old/oddly-built wheel can be missing either.
    """
    needed = {
        'CAP_PROP_HW_ACCELERATION': getattr(cv2, 'CAP_PROP_HW_ACCELERATION', None),
        'CAP_PROP_HW_ACCELERATION_USE_OPENCL':
            getattr(cv2, 'CAP_PROP_HW_ACCELERATION_USE_OPENCL', None),
    }
    if mode == 'vaapi':
        needed['VIDEO_ACCELERATION_VAAPI'] = getattr(cv2, 'VIDEO_ACCELERATION_VAAPI', None)
        enum_name = 'VIDEO_ACCELERATION_VAAPI'
    elif mode == 'qsv':
        needed['VIDEO_ACCELERATION_QSV'] = getattr(cv2, 'VIDEO_ACCELERATION_QSV', None)
        enum_name = 'VIDEO_ACCELERATION_QSV'
    else:
        return None, [mode]

    missing = [name for name, val in needed.items() if val is None]
    if missing:
        return None, missing
    return needed[enum_name], []


def _try_open_hw(cv2, path, mode):
    """Attempt to open `path` with HW acceleration `mode` ('vaapi'|'qsv').

    Returns (cap_or_None, ok:bool). Only returns a capture when it both opened AND the
    driver reports the requested accelerator actually engaged (VideoWriter/Capture can
    silently downgrade to software, and we refuse to claim HW we didn't get). On any
    miss the (possibly-open) capture is released and (None, False) is returned so the
    caller can warn and fall back.
    """
    accel, missing = _accel_constant(cv2, mode)
    if accel is None:
        _warn_once(
            f"opencv is missing constant(s) {missing} needed for {mode.upper()} "
            "hardware decode (opencv < 4.5.2 or built without the VideoIO HW API); "
            "treating hardware decode as unavailable."
        )
        return None, False

    # CAP_PROP_HW_ACCELERATION_USE_OPENCL=0: we want the decoded frames back on the CPU
    # as normal BGR numpy arrays (the rest of the pipeline — imencode, numpy hashing —
    # is CPU), not left as OpenCL surfaces. cv2 downloads them for us.
    params = [
        int(cv2.CAP_PROP_HW_ACCELERATION), int(accel),
        int(cv2.CAP_PROP_HW_ACCELERATION_USE_OPENCL), 0,
    ]
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    if not cap.isOpened():
        cap.release()
        return None, False

    # Confirm what actually engaged. If the wheel/driver downgraded us to software,
    # the property reads back as VIDEO_ACCELERATION_NONE (or the value we didn't ask
    # for). Treat anything other than the requested accel as "did not engage".
    got = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
    if got != int(accel):
        cap.release()
        return None, False
    return cap, True


def _open_software(cv2, path):
    """Plain software capture — the always-available fallback."""
    return cv2.VideoCapture(path)


# --- Public API -------------------------------------------------------------------

def open_capture(path):
    """Open `path` as a cv2.VideoCapture, using the Arc media engine when possible.

    Honours MEDIA_VIDEO_HWACCEL (see hwaccel_requested()). For 'auto' we try VAAPI
    then QSV then software; for an explicit 'vaapi'/'qsv' we try just that one; 'none'
    goes straight to software. Whenever HW was requested but did not engage, a loud
    ONE-TIME stderr warning names what was asked for vs. what we fell back to, then a
    working software capture is returned so decoding keeps functioning.

    Always returns a cv2.VideoCapture object (the caller still checks .isOpened() and
    handles a genuinely un-decodable/corrupt file — this function's job is backend
    selection, not validating the media).
    """
    global _engaged_backend
    import cv2  # heavy; lazy like the rest of the decode path (and phasher.py)

    mode = hwaccel_requested()

    if mode == 'none':
        _engaged_backend = 'software'
        return _open_software(cv2, path)

    # Ordered list of HW backends to try for this mode.
    candidates = ['vaapi', 'qsv'] if mode == 'auto' else [mode]

    for backend in candidates:
        cap, ok = _try_open_hw(cv2, path, backend)
        if ok:
            _engaged_backend = backend
            return cap

    # Nothing HW engaged — fall back to software, loudly.
    _engaged_backend = 'software'
    requested = 'VAAPI or QSV' if mode == 'auto' else mode.upper()
    _warn_once(
        f"{requested} hardware video decode was requested (MEDIA_VIDEO_HWACCEL="
        f"{mode!r}) but did not engage; falling back to CPU software decode. "
        "Video features still work, but the Arc media engine is NOT being used."
    )
    return _open_software(cv2, path)


def describe() -> str:
    """One-line human-readable decode-backend summary for a startup log line.

    Reflects what has ACTUALLY engaged if a capture has been opened this process
    (e.g. "video decode: vaapi"); otherwise it states the request and, when HW was
    asked for, that engagement is still unverified — we can't know a HW backend works
    until we open a real file, and probing cv2 constants here would need to import
    cv2, which may not even be importable on this box.
    """
    mode = hwaccel_requested()
    if _engaged_backend is not None:
        if _engaged_backend == 'software' and mode not in ('none',):
            requested = 'VAAPI or QSV' if mode == 'auto' else mode.upper()
            return f"video decode: software ({requested} requested but unavailable)"
        return f"video decode: {_engaged_backend}"

    # Nothing opened yet.
    if mode == 'none':
        return "video decode: software (forced by MEDIA_VIDEO_HWACCEL=none)"
    requested = 'VAAPI or QSV' if mode == 'auto' else mode.upper()
    return f"video decode: {requested} requested (engages on first video open)"
