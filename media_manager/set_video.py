"""Set → morph video via Wan 2.2 FLF2V (generative), assembled on the B70.

Pipeline (see docs/generate-from-set.md, Plan B):
  ordered set members → for each consecutive pair (A→B, B→C, …) send the two
  images to Wan 2.2 FLF2V (first-frame/last-frame) via the ComfyUI gen service,
  which *generates* the in-between frames → collect all frames in order (dropping
  the duplicated boundary frame between pairs) → assemble to mp4 with the B70's
  hardware encoder (set_render.frames_to_video). Output is marked origin='ai'.

The FLF2V ComfyUI workflow is an API-format JSON *template* finalized on the B70;
this module just substitutes the two input-image names + params into it. Template
placeholders: ``__FIRST_IMAGE__``, ``__LAST_IMAGE__``, ``__PROMPT__``,
``__FRAMES__``, ``__SEED__``. Point ``MEDIA_FLF2V_WORKFLOW`` at the file.

No local fallback: if the gen service or workflow isn't configured/reachable the
caller gets a loud error (generation has no CPU substitute).
"""
import json
import os
import tempfile

from . import set_render
from .gen_service import ComfyUIClient, GenServiceUnavailable, GenServiceError


# Default location for the workflow, so no env var is needed: drop the ComfyUI
# API-format FLF2V graph here and it's picked up automatically.
DEFAULT_WORKFLOW_RELPATH = os.path.join(".media", "flf2v.api.json")


def load_workflow_template(path=None, data_root=None) -> dict:
    """Resolve the FLF2V workflow JSON: explicit path → MEDIA_FLF2V_WORKFLOW →
    <data_root>/.media/flf2v.api.json. Raises with guidance if none exists."""
    candidates = [path, os.environ.get("MEDIA_FLF2V_WORKFLOW")]
    if data_root:
        candidates.append(os.path.join(data_root, DEFAULT_WORKFLOW_RELPATH))
    for c in candidates:
        if c and os.path.isfile(c):
            with open(c) as f:
                return json.load(f)
    raise GenServiceUnavailable(
        "FLF2V workflow not found — put the ComfyUI API-format workflow at "
        f"<library>/{DEFAULT_WORKFLOW_RELPATH} (or set MEDIA_FLF2V_WORKFLOW)")


def build_workflow(template: dict, first_image: str, last_image: str, params: dict) -> dict:
    """Return a copy of the workflow with this pair's two frames (and prompt) filled in.

    No hand-editing required: just "Save (API Format)" from ComfyUI and drop the
    file in. We AUTO-DETECT the inputs to fill:
      * the two ``LoadImage`` nodes (lowest node-id = first frame, next = last)
        get their ``image`` set to the uploaded names;
      * if a motion prompt was given, the positive ``CLIPTextEncode`` (a node
        whose title contains "pos", else the only CLIPTextEncode) gets its ``text``.
    Everything else (steps, frame count, seed, sampler) is whatever you baked into
    the graph in ComfyUI.

    Advanced/opt-in: if the exported JSON contains the tokens ``__FIRST_IMAGE__`` /
    ``__LAST_IMAGE__`` / ``__PROMPT__`` / ``__FRAMES__`` / ``__SEED__``, we
    string-substitute those instead (full manual control)."""
    raw = json.dumps(template)
    if "__FIRST_IMAGE__" in raw or "__LAST_IMAGE__" in raw:
        for token, value in (("__FIRST_IMAGE__", first_image),
                             ("__LAST_IMAGE__", last_image),
                             ("__PROMPT__", str(params.get("prompt", ""))),
                             ("__FRAMES__", str(params.get("frames", 49))),
                             ("__SEED__", str(params.get("seed", 0)))):
            raw = raw.replace(token, value)
        return json.loads(raw)

    wf = json.loads(raw)  # deep copy

    def nodes_of(class_type):
        got = [(nid, n) for nid, n in wf.items()
               if isinstance(n, dict) and n.get("class_type") == class_type]
        # ascending node id (numeric when possible) so "first"/"last" are stable
        return sorted(got, key=lambda kv: (len(kv[0]), kv[0]))

    loads = nodes_of("LoadImage")
    if len(loads) < 2:
        raise GenServiceError(
            f"FLF2V workflow needs two LoadImage nodes (first & last frame); found "
            f"{len(loads)}. Use two LoadImage nodes, or add __FIRST_IMAGE__/"
            f"__LAST_IMAGE__ tokens for manual control.")
    loads[0][1].setdefault("inputs", {})["image"] = first_image
    loads[1][1].setdefault("inputs", {})["image"] = last_image

    prompt = params.get("prompt")
    if prompt:
        encoders = nodes_of("CLIPTextEncode")
        target = None
        for _nid, n in encoders:
            if "pos" in (n.get("_meta", {}) or {}).get("title", "").lower():
                target = n
                break
        if target is None and len(encoders) == 1:
            target = encoders[0][1]
        if target is not None and "text" in target.get("inputs", {}):
            target["inputs"]["text"] = prompt
    return wf


def morph_from_set(member_paths, out_path, gen=None, workflow_template=None,
                   data_root=None, fps=16, params=None, progress=None):
    """Generate a morph video across ordered set members with Wan FLF2V per pair.

    `gen` is a ComfyUIClient (injected for testing); `progress(done, total)` is an
    optional callback. Requires a reachable gen service. Returns out_path. Raises
    GenServiceUnavailable/GenServiceError (no local fallback)."""
    params = params or {}
    imgs = [p for p in member_paths if p and os.path.isfile(p)]
    if len(imgs) < 2:
        raise ValueError("morph needs at least 2 readable set members")

    gen = gen or ComfyUIClient()
    if not gen.is_configured():
        raise GenServiceUnavailable("no generation service configured (MEDIA_GEN_SERVICE_URL)")
    template = load_workflow_template(workflow_template, data_root)

    n_pairs = len(imgs) - 1
    frame_dir = tempfile.mkdtemp(prefix="pm_morph_")
    ordered_frames = []
    try:
        frame_no = 0
        for i in range(n_pairs):
            if progress:
                progress(i, n_pairs)
            with open(imgs[i], "rb") as fa:
                first_name, _ = gen.upload_image(fa.read(),
                                                 f"morph_{i}_a{os.path.splitext(imgs[i])[1] or '.png'}")
            with open(imgs[i + 1], "rb") as fb:
                last_name, _ = gen.upload_image(fb.read(),
                                                f"morph_{i}_b{os.path.splitext(imgs[i + 1])[1] or '.png'}")
            workflow = build_workflow(template, first_name, last_name, params)
            outputs = gen.run_workflow(workflow, timeout=params.get("timeout", 1800))
            files = ComfyUIClient.collect_outputs(outputs)
            if not files:
                raise GenServiceError(f"FLF2V pair {i} produced no frames")
            # Drop the first frame of every pair after the first: it equals the
            # previous pair's last frame (the shared boundary image).
            for j, (fn, sub, typ) in enumerate(files):
                if i > 0 and j == 0:
                    continue
                data = gen.fetch(fn, sub, typ)
                fp = os.path.join(frame_dir, f"{frame_no:06d}{os.path.splitext(fn)[1] or '.png'}")
                with open(fp, "wb") as out:
                    out.write(data)
                ordered_frames.append(fp)
                frame_no += 1
        if progress:
            progress(n_pairs, n_pairs)
        if not ordered_frames:
            raise GenServiceError("morph produced no frames")
        set_render.frames_to_video(ordered_frames, out_path, fps=fps, origin="ai")
        return out_path
    finally:
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Image-to-video: animate a SINGLE image (used by the /photo 🙌 button). Distinct
# from the morph above (which needs two frames) — this runs one still through an
# image-to-video ComfyUI workflow at .media/i2v.api.json.
# --------------------------------------------------------------------------
DEFAULT_I2V_WORKFLOW_RELPATH = os.path.join(".media", "i2v.api.json")


def load_i2v_workflow(path=None, data_root=None) -> dict:
    candidates = [path, os.environ.get("MEDIA_I2V_WORKFLOW")]
    if data_root:
        candidates.append(os.path.join(data_root, DEFAULT_I2V_WORKFLOW_RELPATH))
    for c in candidates:
        if c and os.path.isfile(c):
            with open(c) as f:
                return json.load(f)
    raise GenServiceUnavailable(
        "image-to-video workflow not found — put the ComfyUI API-format workflow at "
        f"<library>/{DEFAULT_I2V_WORKFLOW_RELPATH} (or set MEDIA_I2V_WORKFLOW)")


def image_to_video(image_path, out_path, gen=None, workflow_template=None,
                   data_root=None, fps=16, params=None):
    """Animate a single image into a video via an image-to-video ComfyUI workflow
    (auto-detects the LoadImage node + positive prompt; frames → mp4). Returns
    out_path. Raises GenServiceUnavailable/GenServiceError."""
    import shutil
    from . import set_image
    params = params or {}
    if not (image_path and os.path.isfile(image_path)):
        raise ValueError("image_to_video needs a readable image")
    gen = gen or ComfyUIClient()
    if not gen.is_configured():
        raise GenServiceUnavailable("no generation service configured (MEDIA_GEN_SERVICE_URL)")
    template = load_i2v_workflow(workflow_template, data_root)
    with open(image_path, "rb") as f:
        name, _ = gen.upload_image(f.read(), f"i2v{os.path.splitext(image_path)[1] or '.png'}")
    workflow = set_image.build_image_workflow(template, [name], params.get("prompt", ""))
    outputs = gen.run_workflow(workflow, timeout=params.get("timeout", 1800))
    files = ComfyUIClient.collect_outputs(outputs)
    if not files:
        raise GenServiceError("image-to-video workflow produced no frames")
    frame_dir = tempfile.mkdtemp(prefix="pm_i2v_")
    ordered = []
    try:
        for i, (fn, sub, typ) in enumerate(files):
            data = gen.fetch(fn, sub, typ)
            fp = os.path.join(frame_dir, f"{i:06d}{os.path.splitext(fn)[1] or '.png'}")
            with open(fp, "wb") as o:
                o.write(data)
            ordered.append(fp)
        set_render.frames_to_video(ordered, out_path, fps=fps, origin="ai")
        return out_path
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
