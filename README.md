# Media Manager

Media manager — like git for your media files. Scan directories, track files by
content hash, then layer ML on top: object detection (YOLO-World), visual
similarity search (CLIP), face detection/recognition (InsightFace), EXIF
metadata, and a local web gallery.

## Installation

Straight from GitHub:

```bash
pip install git+https://github.com/KUKARAF/pic_master.git
```

Or for development:

```bash
git clone https://github.com/KUKARAF/pic_master.git
cd pic_master
pip install -e .
```

This installs the `media` CLI. Model weights (YOLO, CLIP, InsightFace) are
downloaded automatically on first use.

GPU note: `requirements.txt` pins `onnxruntime` (CPU). Swap in
`onnxruntime-gpu` for CUDA-accelerated face detection.

## Quick Start

```bash
cd /path/to/your/media
media init                       # create the .media/ repo (index db lives here)
media add .                      # scan + hash files
media commit . --with-full-ml    # scan + EXIF + object index + CLIP embed + faces
media web                        # browse at http://127.0.0.1:8000/
```

## CLI overview

| Command | What it does |
| --- | --- |
| `media init` | Initialize a media repository (`.media/`) |
| `media add <path>` | Scan and hash files (content-hash identity: moved files re-link automatically) |
| `media commit [path] [--with-full-ml]` | Scan, optionally running the full ML pipeline |
| `media status` / `media ls` / `media count` | Inspect tracked files |
| `media duplicates` | List content present at more than one path |
| `media find_broken [path]` | Find corrupted images/videos |
| `media index [path]` | Detect objects with YOLO-World |
| `media search <query>` | Search by detected object class |
| `media embed [path]` | Build CLIP embeddings for similarity search |
| `media faces [path]` | Detect and embed faces (InsightFace) |
| `media who <image>` | Find which known people appear in an image |
| `media metadata [path]` | Read EXIF capture time + GPS |
| `media set create/ls/assign/files` | Manage named sets (e.g. a studio shoot) |
| `media web` | Launch the FastAPI gallery UI |

## Optional: age/gender estimation (MiVOLO)

MiVOLO pins old `ultralytics`/`timm` versions that conflict with this app's own
detector and indexer, so it lives in a **separate, isolated virtualenv** — never
in the main environment. From the repo root:

```bash
uv venv .age-venv
uv pip install --python .age-venv/bin/python -r requirements-age-estimator.txt
```

The app talks to it via a subprocess (`media_manager/age_estimator.py`); set
`MEDIA_AGE_VENV_PYTHON` to point at the venv's python if you put it elsewhere.
Everything else works fine without this step.

## Project Structure

```
media_manager/
├── media.py              # CLI entry point (`media`)
├── media_manager.py      # Main MediaManager class
├── scanner.py            # File discovery
├── hasher.py             # Content hashing (xxhash)
├── database.py           # SQLite schema and operations
├── detector.py           # YOLO-World object detection
├── indexer.py            # CLIP embedding / similarity
├── face_detector.py      # InsightFace detection + embeddings
├── exif_reader.py        # EXIF capture time + GPS
├── age_estimator.py      # MiVOLO client (isolated-venv subprocess)
├── web.py                # FastAPI gallery
├── templates/, static/   # Web UI assets
└── ...
```

## Development

See [TODO.md](TODO.md) and [features.todo.md](features.todo.md) for the roadmap.
