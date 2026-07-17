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

    if python.is_file() and not force:
        print(f"Age estimator venv already set up at {venv_dir} — use --force to recreate.")
        return python
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)

    print(f"Creating isolated age-estimator venv at {venv_dir}")
    print("(separate from the main environment: MiVOLO pins an old ultralytics/timm "
          "that conflict with this app's object detector and CLIP indexer)")
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", str(venv_dir)], check=True)
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "-r", str(_REQUIREMENTS_FILE)],
            check=True,
        )
    else:
        import venv as venv_module
        venv_module.create(str(venv_dir), with_pip=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(_REQUIREMENTS_FILE)],
            check=True,
        )

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
