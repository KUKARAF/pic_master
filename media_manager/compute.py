"""Central compute-device selection for the ML models.

Historically each model picked its own device: CLIP did
``'cuda' if torch.cuda.is_available() else 'cpu'`` and the InsightFace detector
hardcoded ``providers=['CUDAExecutionProvider', 'CPUExecutionProvider']``. That
only ever accelerated on NVIDIA/CUDA — on an Intel Arc GPU (or anything without
CUDA) every model silently ran on CPU and left the GPU idle.

This module is the single place that answers "what should I run on?" for both
the torch models (CLIP, YOLO-World, MiVOLO age/gender) and the onnxruntime
models (InsightFace face detect/embed, PCN). Point every model at these helpers
so adding a backend is a one-line change here, not a repo-wide grep.

Priority (highest first), overridable with the ``MEDIA_DEVICE`` env var
(``cuda|xpu|mps|cpu``, case-insensitive):

* ``cuda`` — NVIDIA, via torch's CUDA build + onnxruntime ``CUDAExecutionProvider``
* ``xpu``  — Intel GPU (Arc / Data Center GPU Flex/Max) via torch's XPU backend
  and onnxruntime ``OpenVINOExecutionProvider`` (device_type=GPU)
* ``mps``  — Apple Silicon (torch only; no onnxruntime EP, faces fall to CPU)
* ``cpu``  — always-available fallback

Per this project's "no silent failures" rule: if ``MEDIA_DEVICE`` names a
backend that isn't actually usable we raise instead of quietly dropping to CPU,
and if the torch device is a GPU but the matching onnxruntime provider isn't
installed we print a one-time warning to stderr (so a half-accelerated setup is
visible) rather than hiding it.
"""
from __future__ import annotations

import functools
import os
import sys

_VALID = ("cuda", "xpu", "mps", "cpu")


def _override() -> str:
    return os.environ.get("MEDIA_DEVICE", "").strip().lower()


@functools.lru_cache(maxsize=1)
def _maybe_import_ipex() -> None:
    """Import Intel Extension for PyTorch if present. Native XPU (torch>=2.5)
    needs no IPEX, but older Intel stacks only register the ``xpu`` backend after
    this import. Absent → fine (we just won't see an XPU device). Cached so the
    (potentially slow) import is attempted at most once."""
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception:
        pass


@functools.lru_cache(maxsize=1)
def torch_device() -> str:
    """Best torch device string: ``'cuda'`` | ``'xpu'`` | ``'mps'`` | ``'cpu'``.

    Cached — the answer can't change within a process and probing the backends
    isn't free. torch is imported lazily so callers that only need the
    onnxruntime answer (or the torch-free PCN path) don't pay for it.
    """
    import torch

    ov = _override()
    if ov:
        if ov not in _VALID:
            raise ValueError(f"MEDIA_DEVICE={ov!r} is not one of {_VALID}")
        if ov == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MEDIA_DEVICE=cuda but torch.cuda.is_available() is False — "
                "install a CUDA build of torch and check the NVIDIA driver."
            )
        if ov == "xpu":
            _maybe_import_ipex()
        if ov == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "MEDIA_DEVICE=xpu but torch.xpu is unavailable — install the XPU "
                "build (pip install torch --index-url "
                "https://download.pytorch.org/whl/xpu) and the Intel GPU runtime "
                "(Level-Zero + compute-runtime)."
            )
        if ov == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MEDIA_DEVICE=mps but torch's MPS backend is unavailable.")
        return ov

    if torch.cuda.is_available():
        return "cuda"
    _maybe_import_ipex()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=1)
def onnx_providers():
    """onnxruntime provider list matching :func:`torch_device`, intersected with
    what this onnxruntime build actually ships.

    Suitable to pass straight to ``InferenceSession(providers=...)`` /
    ``FaceAnalysis(providers=...)``. OpenVINO's GPU target is expressed as a
    ``(name, options)`` tuple. ``CPUExecutionProvider`` is always the final
    fallback so a session can still build if the accelerator EP is missing.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    dev = torch_device()
    chain: list = []
    if dev == "cuda":
        if "CUDAExecutionProvider" in available:
            chain.append("CUDAExecutionProvider")
        else:
            _warn_missing_ep(dev, "onnxruntime-gpu (CUDAExecutionProvider)")
    elif dev == "xpu":
        if "OpenVINOExecutionProvider" in available:
            chain.append(("OpenVINOExecutionProvider", {"device_type": "GPU"}))
        else:
            _warn_missing_ep(dev, "onnxruntime-openvino (OpenVINOExecutionProvider)")
    # mps has no reliable onnxruntime EP for the InsightFace models — CPU it is.
    chain.append("CPUExecutionProvider")
    return chain


def insightface_providers():
    """``(providers, provider_options)`` for InsightFace's ``FaceAnalysis``.

    ``FaceAnalysis`` forwards these to onnxruntime but, unlike a raw
    ``InferenceSession`` (which takes ``(name, opts)`` tuples inside the
    providers list — that's what :func:`onnx_providers` returns), it wants a
    plain name list plus a positionally-aligned ``provider_options`` list. We
    split :func:`onnx_providers` into that shape here. ``provider_options`` is
    ``None`` unless a non-default device is actually targeted (OpenVINO GPU), so
    InsightFace builds that don't accept the kwarg keep working on CPU/CUDA."""
    provs = onnx_providers()
    names = [p[0] if isinstance(p, tuple) else p for p in provs]
    opts = [p[1] if isinstance(p, tuple) else {} for p in provs]
    if all(not o for o in opts):
        return names, None
    return names, opts


def insightface_ctx_id() -> int:
    """InsightFace ``prepare(ctx_id=...)`` value: a non-negative GPU index when
    the torch device is a GPU, else ``-1`` (CPU). InsightFace uses this to decide
    whether to attempt GPU init; the actual EP still comes from
    :func:`onnx_providers`."""
    return 0 if torch_device() in ("cuda", "xpu") else -1


def ultralytics_device():
    """Device value understood by ultralytics ``predict``/``train``/``val``:
    ``0`` for CUDA (ultralytics wants the int index), ``'xpu'``/``'mps'`` for
    those backends, ``'cpu'`` otherwise."""
    dev = torch_device()
    if dev == "cuda":
        return 0
    return dev


def describe() -> str:
    """One-line human summary for startup logs, e.g.
    ``torch=xpu onnx=OpenVINOExecutionProvider,CPUExecutionProvider``."""
    provs = ",".join(p[0] if isinstance(p, tuple) else p for p in onnx_providers())
    return f"torch={torch_device()} onnx={provs}"


@functools.lru_cache(maxsize=None)
def _warn_missing_ep(device: str, want: str) -> None:
    """One-time stderr warning when the torch device is a GPU but the matching
    onnxruntime EP isn't installed (cached on args so it prints once per case)."""
    print(
        f"[compute] torch is using {device!r} but {want} is not installed — "
        f"InsightFace/PCN will run on CPU. Install it to accelerate faces too.",
        file=sys.stderr,
    )
