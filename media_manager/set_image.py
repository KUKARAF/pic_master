"""Set → a NEW generated image via a ComfyUI workflow (face-swap / img2img /
identity), zero-config — the image sibling of set_video.

Uploads one or more of the set's member images to the ComfyUI gen service, runs an
image workflow (auto-detecting the workflow's ``LoadImage`` nodes + positive
prompt), and saves the produced image (marked AI). The workflow is *your* ComfyUI
graph, exported API-format, at ``<library>/.media/image.api.json`` (or point
``MEDIA_IMAGE_WORKFLOW`` at it). No env vars needed co-located.

Placeholders (optional manual override): ``__IMAGE_1__``, ``__IMAGE_2__``, … and
``__PROMPT__``. Otherwise the workflow's ``LoadImage`` nodes are filled in id order
with the uploaded members (the last is reused if the graph has more slots than
we send), and the positive ``CLIPTextEncode`` gets the prompt.
"""
import io
import json
import os

from . import set_render
from .gen_service import ComfyUIClient, GenServiceUnavailable, GenServiceError

DEFAULT_WORKFLOW_RELPATH = os.path.join(".media", "image.api.json")


def load_image_workflow(path=None, data_root=None) -> dict:
    candidates = [path, os.environ.get("MEDIA_IMAGE_WORKFLOW")]
    if data_root:
        candidates.append(os.path.join(data_root, DEFAULT_WORKFLOW_RELPATH))
    for c in candidates:
        if c and os.path.isfile(c):
            with open(c) as f:
                return json.load(f)
    raise GenServiceUnavailable(
        "image workflow not found — put the ComfyUI API-format workflow at "
        f"<library>/{DEFAULT_WORKFLOW_RELPATH} (or set MEDIA_IMAGE_WORKFLOW)")


def _loadimage_ids(wf):
    return sorted([nid for nid, n in wf.items()
                   if isinstance(n, dict) and n.get("class_type") == "LoadImage"],
                  key=lambda k: (len(k), k))


def build_image_workflow(template: dict, image_names, prompt="") -> dict:
    """Fill the workflow's LoadImage nodes with the uploaded member image names (in
    id order) and set the positive prompt. Token override supported."""
    raw = json.dumps(template)
    if "__IMAGE_1__" in raw or "__PROMPT__" in raw:
        for i, name in enumerate(image_names, 1):
            raw = raw.replace(f"__IMAGE_{i}__", name)
        raw = raw.replace("__PROMPT__", prompt or "")
        return json.loads(raw)

    wf = json.loads(raw)
    ids = _loadimage_ids(wf)
    if not ids:
        raise GenServiceError(
            "image workflow has no LoadImage node to receive the set image "
            "(or add __IMAGE_1__ tokens for manual control).")
    for i, nid in enumerate(ids):
        name = image_names[i] if i < len(image_names) else image_names[-1]
        wf[nid].setdefault("inputs", {})["image"] = name
    if prompt:
        encoders = sorted([(nid, n) for nid, n in wf.items()
                           if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"],
                          key=lambda kv: (len(kv[0]), kv[0]))
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


def image_from_set(member_paths, out_path, gen=None, workflow_template=None,
                   data_root=None, params=None):
    """Generate one new image from a set's members via the ComfyUI image workflow.
    Returns out_path. Raises GenServiceUnavailable/GenServiceError (no local
    fallback)."""
    params = params or {}
    imgs = [p for p in member_paths if p and os.path.isfile(p)]
    if not imgs:
        raise ValueError("image generation needs at least 1 readable member image")

    gen = gen or ComfyUIClient()
    if not gen.is_configured():
        raise GenServiceUnavailable("no generation service configured (MEDIA_GEN_SERVICE_URL)")
    template = load_image_workflow(workflow_template, data_root)

    n_slots = max(1, len(_loadimage_ids(template)))
    chosen = imgs[:n_slots]
    uploaded = []
    for i, p in enumerate(chosen):
        with open(p, "rb") as f:
            name, _ = gen.upload_image(f.read(),
                                       f"srcimg_{i}{os.path.splitext(p)[1] or '.png'}")
        uploaded.append(name)

    workflow = build_image_workflow(template, uploaded, params.get("prompt", ""))
    outputs = gen.run_workflow(workflow, timeout=params.get("timeout", 600))
    files = ComfyUIClient.collect_outputs(outputs)
    if not files:
        raise GenServiceError("image workflow produced no output")
    fn, sub, typ = files[0]
    data = gen.fetch(fn, sub, typ)
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    set_render.save_image_with_provenance(img, out_path, origin="ai")
    return out_path
