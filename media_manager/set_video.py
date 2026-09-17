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


def fill_template(template: dict, first_image: str, last_image: str, params: dict) -> dict:
    """Substitute the two uploaded image names + params into the workflow template
    (string replacement so it's agnostic to the exact node layout)."""
    s = json.dumps(template)
    repl = {
        "__FIRST_IMAGE__": first_image,
        "__LAST_IMAGE__": last_image,
        "__PROMPT__": str(params.get("prompt", "")),
        "__FRAMES__": str(params.get("frames", 49)),
        "__SEED__": str(params.get("seed", 0)),
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return json.loads(s)


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
            workflow = fill_template(template, first_name, last_name, params)
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
