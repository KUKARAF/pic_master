"""Support layer for media generated FROM a set — see docs/generate-from-set.md.

The *engines* that actually create content (face-swap / diffusion for images;
RIFE-or-FLF2V frame-morph for video) are model-based and run on the GPU box — they
are NOT here. This module is the transport-agnostic plumbing they all share and
that runs anywhere:

  * output_path()/generated_dir()/slugify() — where generated files go
    (``<data_root>/generated/<set-slug>/…``, a tree the scanner skips).
  * save_image_with_provenance() — write a produced PIL image with an embedded
    provenance mark (EXIF/PNG-text) so "AI-generated" survives export.
  * frames_to_video() — assemble an ordered list of frame images into an mp4 with
    the B70's VAAPI hardware HEVC encoder (software fallback), tagging the file
    with the same provenance mark. Both video engines (RIFE interpolation and Wan
    FLF2V) produce frames, then hand them here.

``origin`` is 'ai' for synthesized content (the only kind this feature makes);
'composite' exists only for completeness. Callers also record each output in
manual.db's generated_artifacts registry.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time

from PIL import PngImagePlugin

GENERATED_DIRNAME = "generated"

PROV_AI = "pic_master AI-generated"
PROV_COMPOSITE = "pic_master composite (real photos, not AI-generated)"


def _prov_text(origin: str) -> str:
    return PROV_AI if origin == "ai" else PROV_COMPOSITE


def generated_dir(data_root: str) -> str:
    """`<data_root>/generated` — the segregated output tree (never indexed)."""
    return os.path.join(data_root, GENERATED_DIRNAME)


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "set").strip()).strip("-").lower()
    return s or "set"


def output_path(data_root: str, set_name: str, kind: str, ext: str, stamp: str = None) -> str:
    """Build (and mkdir) generated/<set-slug>/<kind>-<ts>.<ext>."""
    d = os.path.join(generated_dir(data_root), slugify(set_name))
    os.makedirs(d, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(d, f"{kind}-{stamp}.{ext}")


def save_image_with_provenance(img, out_path, origin="ai"):
    """Save a produced PIL image, embedding a provenance mark (PNG text chunk or
    EXIF Software/ImageDescription) so its origin survives download/export."""
    mark = _prov_text(origin)
    ext = os.path.splitext(out_path)[1].lower()
    tmp = out_path + ".tmp"
    if ext == ".png":
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Software", "pic_master")
        meta.add_text("Comment", mark)
        img.save(tmp, "PNG", pnginfo=meta)
    else:
        exif = img.getexif()
        exif[0x0131] = "pic_master"   # Software
        exif[0x010E] = mark           # ImageDescription
        img.save(tmp, "JPEG", quality=92, exif=exif)
    os.replace(tmp, out_path)
    return out_path


def register_generated_file(db, manual, data_root, abs_path, *, set_id=None,
                            kind="morph", origin="ai", model=None,
                            media_type=None, params=None):
    """Make a just-produced file under generated/ a first-class (but flagged)
    library entry so it can be a set member and get thumbnails/cards:
      * hash it (streamed xxhash — no loading a big mp4 into memory);
      * add it to the files table via upsert_file_path, then mark it hidden=1
        (keeps it out of the normal gallery/feeds) and ai_generated=1 (AI badge + /ai);
      * assign it to its source set (so it shows in that set's grid);
      * record it in the generated_artifacts provenance registry.
    Returns (file_id, checksum, artifact_id)."""
    import xxhash
    rel_path = os.path.relpath(abs_path, data_root)
    h = xxhash.xxh64()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    checksum = h.hexdigest()
    file_id = db.upsert_file_path(rel_path, checksum,
                                  size=os.path.getsize(abs_path),
                                  modified_time=int(os.path.getmtime(abs_path)))
    db.set_file_hidden(file_id, True)
    db.set_file_ai_generated(file_id, True)
    if set_id is not None:
        manual.assign_file_to_set(checksum, set_id)
    # Put every generated item in the "AI" category (find-or-create). This is the
    # single label the rest of the app uses to keep synthetic media OUT of AI
    # generation inputs and model training — no feedback loops / model collapse.
    manual.add_file_category(checksum, manual.create_category("AI"))
    artifact_id = manual.add_generated_artifact(
        kind=kind, origin=origin, path=rel_path, set_id=set_id,
        media_type=media_type, model=model, params=params)
    return file_id, checksum, artifact_id


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def _vaapi_render_node():
    """A usable VAAPI render node (/dev/dri/renderD*) or None. On the B70 this
    enables hardware HEVC/AV1 encode; absent → software encode."""
    node = os.environ.get("MEDIA_VAAPI_DEVICE")
    if node:
        return node if os.path.exists(node) else None
    for n in ("/dev/dri/renderD128", "/dev/dri/renderD129"):
        if os.path.exists(n):
            return n
    return None


def frames_to_video(frame_paths, out_path, fps=30, hardware=True, origin="ai"):
    """Assemble an ORDERED list of frame image paths into an mp4.

    This is the last step for both video engines: an interpolation/generative
    model produces the morphed frames, then hands the ordered list here. Encodes
    with the B70's VAAPI HEVC encoder when a render node is present, falling back
    to software x264 (also the path on a machine with no Arc). Embeds the
    provenance mark as the container ``comment``. Returns out_path."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH — required to assemble video")
    frames = [p for p in frame_paths if p and os.path.isfile(p)]
    if not frames:
        raise ValueError("frames_to_video: no frame images given")

    # Symlink the frames into a temp dir as a zero-padded sequence so ffmpeg's
    # image2 reader ingests them in order without copying pixels.
    seqdir = tempfile.mkdtemp(prefix="pm_frames_")
    ext = os.path.splitext(frames[0])[1] or ".png"
    try:
        for i, p in enumerate(frames):
            os.symlink(os.path.abspath(p), os.path.join(seqdir, f"f_{i:06d}{ext}"))
        pattern = os.path.join(seqdir, f"f_%06d{ext}")
        meta = ["-metadata", f"comment={_prov_text(origin)}", "-metadata", "encoder=pic_master"]
        node = _vaapi_render_node() if hardware else None
        attempts = []
        if node:
            attempts.append(("vaapi", [
                "ffmpeg", "-y", "-vaapi_device", node,
                "-framerate", str(fps), "-i", pattern,
                "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi",
                "-rc_mode", "CQP", "-qp", "24", *meta, out_path,
            ]))
        attempts.append(("software", [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-pix_fmt", "yuv420p", *meta, out_path,
        ]))
        last_err = None
        for label, cmd in attempts:
            tmp = out_path + ".tmp.mp4"
            cmd[-1] = tmp
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, out_path)
                print(f"[set_render] assembled {len(frames)} frames via {label}: {out_path}",
                      flush=True)
                return out_path
            last_err = (r.stderr or "")[-600:] or f"exit {r.returncode}"
            if os.path.exists(tmp):
                os.unlink(tmp)
            print(f"[set_render] {label} encode failed, "
                  f"{'falling back' if label == 'vaapi' else 'giving up'}: {last_err}",
                  flush=True)
        raise RuntimeError(f"frames_to_video: ffmpeg failed — {last_err}")
    finally:
        shutil.rmtree(seqdir, ignore_errors=True)
