"""Client for the generative ComfyUI service (the Intel ``llm-scaler`` ComfyUI-XPU
container on the B70) used to generate images/video FROM a set.

Only the transport lives here — the app owns no models. It submits a ComfyUI
API-format *workflow*, uploads input images, polls to completion, and fetches the
outputs. The actual workflows (Wan 2.2 FLF2V for the video morph; SDXL/Qwen/
face-swap for images) are templates finalized on the B70 (see
docs/generate-from-set.md). If the service isn't configured or reachable the
caller surfaces it loudly — generation has no local fallback (the models don't
run on a low-power host, which is the whole reason for the offload).

Config: ``MEDIA_GEN_SERVICE_URL`` (e.g. ``http://127.0.0.1:8188``).
"""
import io
import json
import os
import time
import uuid

import requests

_CONNECT_TIMEOUT = 8.0


class GenServiceUnavailable(Exception):
    """The generation service isn't configured or can't be reached."""


class GenServiceError(Exception):
    """The service was reached but a request/workflow failed."""


# ComfyUI's default port; co-located on the B70 this "just works" with no export.
# Set MEDIA_GEN_SERVICE_URL only to point at a different host/port.
DEFAULT_SERVICE_URL = "http://127.0.0.1:8188"


def service_url():
    return (os.environ.get("MEDIA_GEN_SERVICE_URL") or DEFAULT_SERVICE_URL).strip().rstrip("/")


class ComfyUIClient:
    """Minimal ComfyUI HTTP client: upload → submit workflow → poll → fetch."""

    def __init__(self, base_url=None, session=None):
        self.base_url = base_url if base_url is not None else service_url()
        self.client_id = uuid.uuid4().hex
        self._session = session or requests.Session()

    def _url(self, path):
        if not self.base_url:
            raise GenServiceUnavailable("MEDIA_GEN_SERVICE_URL is not set")
        return f"{self.base_url}/{path.lstrip('/')}"

    def is_configured(self):
        return bool(self.base_url)

    def is_available(self, timeout=4.0):
        if not self.base_url:
            return False
        try:
            r = self._session.get(self._url("system_stats"),
                                  timeout=(min(_CONNECT_TIMEOUT, timeout), timeout))
            return r.status_code == 200
        except requests.RequestException:
            return False

    def upload_image(self, data: bytes, name: str):
        """Upload input image bytes; returns (server_name, subfolder)."""
        try:
            r = self._session.post(
                self._url("upload/image"),
                files={"image": (name, io.BytesIO(data), "application/octet-stream")},
                data={"overwrite": "true"},
                timeout=(_CONNECT_TIMEOUT, 60),
            )
        except requests.RequestException as e:
            raise GenServiceUnavailable(f"image upload failed: {e}") from e
        if r.status_code != 200:
            raise GenServiceError(f"upload HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        return j.get("name", name), j.get("subfolder", "")

    def submit(self, workflow: dict) -> str:
        """Queue a workflow; returns its prompt_id."""
        try:
            r = self._session.post(
                self._url("prompt"),
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=(_CONNECT_TIMEOUT, 30),
            )
        except requests.RequestException as e:
            raise GenServiceUnavailable(f"workflow submit failed: {e}") from e
        if r.status_code != 200:
            raise GenServiceError(f"submit HTTP {r.status_code}: {r.text[:300]}")
        pid = (r.json() or {}).get("prompt_id")
        if not pid:
            raise GenServiceError("submit returned no prompt_id")
        return pid

    def wait(self, prompt_id: str, timeout=1800, poll=2.0) -> dict:
        """Poll /history until the prompt finishes; returns its `outputs` dict."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self._session.get(self._url(f"history/{prompt_id}"),
                                      timeout=(_CONNECT_TIMEOUT, 30))
            except requests.RequestException as e:
                raise GenServiceUnavailable(f"history poll failed: {e}") from e
            if r.status_code == 200:
                hist = (r.json() or {}).get(prompt_id)
                if hist:
                    status = hist.get("status", {}) or {}
                    if status.get("status_str") == "error":
                        raise GenServiceError(f"workflow error: {json.dumps(status)[:300]}")
                    if hist.get("outputs"):
                        return hist["outputs"]
            time.sleep(poll)
        raise GenServiceError(f"workflow {prompt_id} did not finish within {timeout}s")

    def fetch(self, filename: str, subfolder="", type_="output") -> bytes:
        """Download one output file."""
        try:
            r = self._session.get(
                self._url("view"),
                params={"filename": filename, "subfolder": subfolder, "type": type_},
                timeout=(_CONNECT_TIMEOUT, 120),
            )
        except requests.RequestException as e:
            raise GenServiceUnavailable(f"output fetch failed: {e}") from e
        if r.status_code != 200:
            raise GenServiceError(f"fetch HTTP {r.status_code} for {filename!r}")
        return r.content

    def run_workflow(self, workflow: dict, timeout=1800) -> dict:
        """submit + wait; returns the outputs dict (node_id -> {images/gifs: [...]})."""
        return self.wait(self.submit(workflow), timeout=timeout)

    @staticmethod
    def collect_outputs(outputs: dict):
        """Flatten a ComfyUI outputs dict into an ordered list of
        (filename, subfolder, type). Node ids are visited in numeric order so a
        SaveImage sequence comes back in frame order; 'images' and 'gifs'
        (VHS video output) are both included."""
        files = []
        for node_id in sorted(outputs.keys(), key=lambda k: (len(k), k)):
            node = outputs[node_id] or {}
            for item in node.get("images", []) + node.get("gifs", []):
                files.append((item["filename"], item.get("subfolder", ""),
                              item.get("type", "output")))
        return files
