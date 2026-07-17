"""Age/gender estimation (MiVOLO) — thin client for the main app process.

MiVOLO's own code hard-pins an old timm/ultralytics that conflict with this project's
YOLO-World detector and CLIP indexer (confirmed: installing it into the main venv
broke `from ultralytics import YOLOWorld`). So the actual model, person detector, and
face<->body matching all live in `age_estimator_worker.py`, run as a subprocess inside
a separate, isolated virtualenv (`.age-venv`) — never imported here. This file just
shells out to it and speaks JSON over stdin/stdout.

This isolation is also what makes the whole feature trivially removable: delete this
file + age_estimator_worker.py, drop the `.age-venv` directory, remove the one web.py
endpoint + lazy accessor and the one photo.html section — nothing else in the app
imports or depends on any of it.
"""
import json
import os
import subprocess
from pathlib import Path

MODEL_ID = "mivolo"

_HERE = Path(__file__).parent
_WORKER_SCRIPT = _HERE / "age_estimator_worker.py"

# Repo layout for this project: media_manager/media_manager/age_estimator.py -> up
# two levels is the repo root, where `.age-venv` lives alongside the main `.venv`.
# Only meaningful for a source checkout; for a pip-installed package this resolves
# somewhere inside site-packages and simply never exists.
_REPO_CHECKOUT_VENV_DIR = _HERE.parent.parent / ".age-venv"

_REQUIREMENTS_FILE = _HERE / "requirements-age-estimator.txt"

# The venv must run an older Python: the pinned torch==2.5.1 (required for MiVOLO's
# old timm/ultralytics, see requirements-age-estimator.txt) publishes no wheels for
# Python 3.14+.
_AGE_VENV_PYTHON_SERIES = "3.12"

# MiVOLO's setup.py does `import pkg_resources`, which modern setuptools no longer
# ships — so an isolated build env (which gets latest setuptools) can't build it.
# Instead we pre-install the last setuptools line that still bundles pkg_resources
# into the venv and build mivolo against the venv itself (no build isolation).
_BUILD_SETUPTOOLS_PIN = "setuptools<81"

# Written into the venv after a fully successful install; its absence means a
# previous age-setup run died partway (venv exists, packages don't) and the venv
# should be rebuilt rather than reported as already set up.
_SETUP_COMPLETE_MARKER = ".setup-complete"

ESTIMATE_TIMEOUT_SECONDS = 120


def default_age_venv_dir() -> Path:
    """Where `media age-setup` installs the venv and where lookup falls back to:
    $XDG_DATA_HOME/media_manager/age-venv (~/.local/share/media_manager/age-venv)."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "media_manager" / "age-venv"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _find_compatible_python() -> str | None:
    """PATH lookup for a Python matching _AGE_VENV_PYTHON_SERIES (pip fallback path
    only — uv resolves/downloads interpreters itself)."""
    import shutil
    return shutil.which(f"python{_AGE_VENV_PYTHON_SERIES}")


def _venv_python_path() -> Path:
    override = os.environ.get("MEDIA_AGE_VENV_PYTHON")
    if override:
        return Path(override)
    repo_python = _venv_python(_REPO_CHECKOUT_VENV_DIR)
    if repo_python.is_file():
        return repo_python
    return _venv_python(default_age_venv_dir())


def setup_age_venv(dest=None, force: bool = False) -> Path:
    """Create the isolated MiVOLO venv and install the pinned requirements into it.

    Backs `media age-setup`. Returns the venv's python executable path. Uses uv when
    available (much faster, shares its wheel cache), else stdlib venv + pip.
    """
    import shutil

    venv_dir = Path(dest).expanduser() if dest else default_age_venv_dir()
    python = _venv_python(venv_dir)
    marker = venv_dir / _SETUP_COMPLETE_MARKER

    if python.is_file() and marker.is_file() and not force:
        print(f"Age estimator venv already set up at {venv_dir} — use --force to recreate.")
        return python
    if venv_dir.exists():
        if python.is_file() and not marker.is_file():
            print(f"Found leftovers of an interrupted setup at {venv_dir} — rebuilding.")
        shutil.rmtree(venv_dir)

    print(f"Creating isolated age-estimator venv at {venv_dir}")
    print("(separate from the main environment: MiVOLO pins an old ultralytics/timm "
          "that conflict with this app's object detector and CLIP indexer)")
    uv = shutil.which("uv")
    if uv:
        # A compatible interpreter, not the system Python (which may be too new for
        # the pinned torch): a system pythonX.Y if installed, else uv's managed one.
        # Distro uv packages often set python-downloads to 'manual', so the download
        # has to be opted into explicitly for this one command.
        base_python = _find_compatible_python() or _AGE_VENV_PYTHON_SERIES
        if base_python == _AGE_VENV_PYTHON_SERIES:
            print(f"No system python{_AGE_VENV_PYTHON_SERIES} found — "
                  f"letting uv download a managed Python {_AGE_VENV_PYTHON_SERIES}.")
        subprocess.run(
            [uv, "venv", "--python", base_python, str(venv_dir)],
            check=True,
            env={**os.environ, "UV_PYTHON_DOWNLOADS": "automatic"},
        )
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), _BUILD_SETUPTOOLS_PIN, "wheel"],
            check=True,
        )
        subprocess.run(
            [uv, "pip", "install", "--python", str(python),
             "--no-build-isolation-package", "mivolo",
             "-r", str(_REQUIREMENTS_FILE)],
            check=True,
        )
    else:
        base_python = _find_compatible_python()
        import venv as venv_module
        if base_python is None:
            raise RuntimeError(
                f"No Python {_AGE_VENV_PYTHON_SERIES} interpreter found on PATH and uv is "
                f"not installed. The age estimator needs Python {_AGE_VENV_PYTHON_SERIES} "
                "(its pinned torch==2.5.1 has no wheels for newer Pythons). Install uv "
                "(it auto-downloads a compatible interpreter) or install "
                f"python{_AGE_VENV_PYTHON_SERIES} and re-run `media age-setup`."
            )
        subprocess.run([base_python, "-m", "venv", str(venv_dir)], check=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", _BUILD_SETUPTOOLS_PIN, "wheel"],
            check=True,
        )
        # Global --no-build-isolation is fine here: everything except mivolo installs
        # from wheels, and mivolo's build needs are covered by the setuptools above.
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-build-isolation",
             "-r", str(_REQUIREMENTS_FILE)],
            check=True,
        )

    marker.touch()
    print(f"Done. Age estimation is ready (venv: {venv_dir}).")
    print("Model weights download automatically on the first estimate.")
    return python


class AgeGenderEstimator:
    def __init__(self):
        self.venv_python = _venv_python_path()
        if not self.venv_python.is_file():
            raise RuntimeError(
                f"Age estimator venv not found at '{self.venv_python}'. "
                "Run `media age-setup` to create it (it must stay separate from the main "
                "environment — MiVOLO pins an old timm/ultralytics that conflict with this "
                "app's object detector), or point MEDIA_AGE_VENV_PYTHON at an existing "
                "isolated venv's python executable."
            )

    def estimate(self, image_path: str, faces: list) -> list:
        """faces: [{'ref': ..., 'bbox': [x1,y1,x2,y2], ...}, ...] — the same shape
        web.py's _combined_faces_for_file already returns. Returns
        [{'face_ref': ..., 'age': float|None, 'gender': str|None}, ...], one entry per
        input face, in the same order (a failed individual face is still present with
        null age/gender rather than dropped from the list)."""
        if not faces:
            return []

        payload = {
            "image_path": image_path,
            "faces": [{"face_ref": f["ref"], "bbox": f["bbox"]} for f in faces],
        }
        try:
            proc = subprocess.run(
                [str(self.venv_python), str(_WORKER_SCRIPT)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=ESTIMATE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Age estimation timed out after {ESTIMATE_TIMEOUT_SECONDS}s")

        if not proc.stdout:
            raise RuntimeError(f"Age estimator produced no output (exit {proc.returncode}): {proc.stderr[-2000:]}")

        # The worker's own final json.dump is always the last line it prints — anything
        # before that (e.g. a one-time "downloading yolov8n.pt..." notice on first run)
        # is noise from a third-party library, not something we control the format of.
        last_line = proc.stdout.strip().splitlines()[-1]
        try:
            data = json.loads(last_line)
        except json.JSONDecodeError:
            raise RuntimeError(f"Age estimator returned invalid output: {proc.stdout[-2000:]}")

        if "error" in data:
            raise RuntimeError(f"Age estimator failed: {data['error']}")

        return data["results"]
