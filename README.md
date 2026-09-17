# Media Manager

Media manager — like git for your media files. Scan directories, track files by
content hash, then layer ML on top: object detection (YOLO-World), visual
similarity search (CLIP), face detection/recognition (InsightFace), find-by-body
(re-identify a person by outfit/build when their face is hidden — web UI only),
EXIF metadata, and a local web gallery.

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

GPU note: the compute device is auto-selected — **CUDA → XPU (Intel Arc) → MPS →
CPU** — and can be forced with the `MEDIA_DEVICE=cuda|xpu|mps|cpu` env var (it
fails loudly if you name a backend that isn't usable, rather than silently
running on CPU). Out of the box `requirements.txt` pins CPU `onnxruntime` and a
CPU/CUDA `torch`. To actually use a GPU:

- **NVIDIA (CUDA):** swap `onnxruntime` → `onnxruntime-gpu`; install CUDA
  `torch`+`torchvision` **together** (open_clip pulls torchvision, and a
  mismatched pair fails with `operator torchvision::nms does not exist`).
- **Intel Arc (XPU), incl. Arc Pro B70 / Battlemage:** swap `onnxruntime` →
  `onnxruntime-openvino`, and install the XPU builds of **both** torch and
  torchvision from the same index — they must match or you get `operator
  torchvision::nms does not exist`:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu`
  — on a host with the Intel GPU runtime (kernel i915/xe + Level-Zero +
  compute-runtime). CLIP and YOLO-World run
  on the Arc GPU; InsightFace faces run via OpenVINO. Age/gender (MiVOLO) runs in
  its own isolated venv, which `media age-setup` builds with `torch==2.7.1+xpu`
  (Battlemage needs torch ≥ 2.6, so the age venv no longer uses the old 2.5.1) —
  this bump is **unvalidated against MiVOLO's model build**, so verify age
  estimation on the GPU box after `age-setup`; if MiVOLO breaks, pin a compatible
  `timm` in `requirements-age-estimator.txt`.

All device selection funnels through `media_manager/compute.py`.

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
| `media bodies [path]` | Crop + embed person boxes for find-by-body search |
| `media who <image>` | Find which known people appear in an image |
| `media metadata [path]` | Read EXIF capture time + GPS |
| `media geo fetch-cities` | Download the offline GeoNames city database for reverse-geocoding |
| `media set create/ls/assign/files` | Manage named sets (e.g. a studio shoot) |
| `media web` | Launch the FastAPI gallery UI |
| `media worker` | Run the ML-offload worker HTTP service on a beefy machine (see below) |
| `media worker-connect <url>` | Point this host at a worker (base URL) to offload heavy ML |

## Optional: offline city names (reverse-geocoding)

Run `media geo fetch-cities` once to download the [GeoNames](https://www.geonames.org/)
`cities15000` dataset (~26k cities, licensed CC-BY 4.0) into a local `cities` table — no
runtime network calls. Then the `🏙 Match cities` bulk action (⚡ menu) labels each
geotagged photo with its nearest city, so photos show a place **name** (and are searchable
by the `city:` facet) instead of raw coordinates. City data © GeoNames, CC-BY 4.0.

## Optional: duplicate & damaged-file detection (czkawka)

Near-duplicate and corrupt-file scanning are powered by the external
[czkawka](https://github.com/qarmin/czkawka) CLI (`czkawka_cli` ≥ 12) — install it on the
server and put it on `PATH`. On startup the app probes for it; if it's missing it logs
`czkawka not found, deduplication will not be available`, disables the two bulk actions
(**🔎 Find near-duplicates** and **🩹 Find damaged files** in the ⚡ menu), and everything
else keeps working. **czkawka is only ever used to *find*** — this app never passes it a
delete/move flag; removing anything stays your own reversible **Trash** / mark-damaged
action. Similar-**video** scanning additionally needs `ffmpeg` + `ffprobe` on `PATH`; the
damaged-file scan validates images/PDF/archive/music (not video streams).

## Optional: age/gender estimation (MiVOLO)

MiVOLO pins old `ultralytics`/`timm` versions that conflict with this app's own
detector and indexer, so it lives in a **separate, isolated virtualenv** — never
in the main environment. One command sets it up:

```bash
media age-setup
```

This creates the venv at `~/.local/share/media_manager/age-venv` (override with
`--dest`) and installs the pinned requirements bundled with the package. The app
talks to it via a subprocess (`media_manager/age_estimator.py`). To use a venv
you built yourself, set `MEDIA_AGE_VENV_PYTHON` to its python executable; a
repo checkout's `.age-venv` is also still picked up automatically for
development. Everything else works fine without this step.

## Optional: ML worker (offload heavy models to a local HTTP service)

Face detection/embedding, CLIP, and YOLO-World are memory-hungry. On a
low-RAM host (or one running the web UI with multiple workers) they can OOM. The
`media worker` command runs those models as a small **HTTP service** — on the
same box or a separate, beefier one — and the host offloads to it. When a worker
is reachable, `media faces` / `media index` / `media embed` / `media bodies`, the
web reindex/embed/face endpoints, find-by-body, **per-tag classifier training**
(the CLIP and YOLO-World "Train" buttons), **and age/gender estimation** all send
their work to it — each dispatch is logged (`[worker] outsourced …`) and shown in
the web UI's worker badge. For the *inference* offloads the host transparently
falls back to running locally if the worker is unreachable; **training and
age/gender do not fall back** — see below.

The transport is plain HTTP with msgpack payloads (`worker_protocol.py`): the
worker is a FastAPI/uvicorn service exposing one `POST /<op>` route per operation
and the host is a `requests` client (`worker_client.py`); discovery is just the
worker's base URL. It's unencrypted, so on anything but a trusted LAN put it
behind Tailscale/WireGuard rather than exposing the port.

### Single box, both roles: `media web --with-worker`

If one machine runs *both* the web UI and the worker, you don't need any config —
no `worker-connect`, no `worker.json`:

```bash
media web --with-worker            # add --preload-worker to load models at boot
```

This spawns `media worker` as a child HTTP service on `127.0.0.1:4243` (override
with `--worker-port`), points the web process at it, and shuts it down when the
web server exits. It's mainly a way to keep heavy models out of the web process's
own address space on a single box; if you don't care about that, plain `media web`
runs every model in-process and needs no worker at all. The cross-machine setup
below is only for running the worker on a *separate* beefier host.

Only the *models* run remotely; your media files never need to live on the
worker (images are streamed to it per request). Results come back as embeddings
and are written to the host's `.media/` database exactly as if computed locally.

### Cross-machine: worker on a separate box

**1. Install the package on both machines.** The worker also needs the ML
dependencies (they ship in `requirements.txt`); use a Python with wheels for your
ML stack (3.11/3.12 are safe). For GPU acceleration on the worker, see the GPU
note near the top of this README.

**2. Start the worker, bound so the host can reach it:**

```bash
media worker --host 0.0.0.0 --port 4243     # add --preload to load models at startup
```

It prints its URL and serves until Ctrl-C. Leave it running (tmux/screen, a
systemd unit, or `nohup media worker --host 0.0.0.0 &`). Make sure the port is
reachable from the host; on anything but a trusted LAN put it behind
Tailscale/WireGuard rather than exposing it directly.

**3. Point the host at the worker** — inside your media repo:

```bash
media worker-connect http://<worker-ip>:4243
```

This writes `.media/worker.json`. From now on `media` commands and the web UI
offload to the worker. You can also set `MEDIA_WORKER_ADDR` (a URL) /
`MEDIA_WORKER_ENABLED` as environment variables instead of the file, and
`media worker-connect <url> --disable` saves the address but turns offloading off.

### Per-tag classifier training on the worker

When a worker is configured, clicking **Train** on a tag trains its CLIP linear
classifier and/or YOLO-World fine-tune **on the worker**, not the host — the
host only gathers the tag's examples, downscales the images, and streams them
over. Progress, logs, and cancel work exactly as with local training (the same
metadata/status the UI already polls). Two things to know:

- **The worker keeps its trained YOLO checkpoints** (under
  `~/.cache/media_manager/tag_models/` on the worker) and also serves the
  fine-tuned detection for the "find more" swipe, so the heavy YOLO model never
  loads on the host. The small CLIP weights come back to the host. If the
  worker's cache is wiped or the worker is replaced, a tag's YOLO suggestions
  will report a missing model — just retrain it.
- **No local fallback for training.** If the worker is configured but
  unreachable, a Train click fails with a clear "worker unavailable" status
  rather than falling back to training on the (low-RAM) host. Start the worker,
  then retry.

### Age/gender estimation on the worker

Age/gender (MiVOLO) is also offloaded when a worker is configured — the
`/person/` "🎂 Estimate all" button and the per-photo estimate run on the
worker, not the host. MiVOLO pins an old torch/timm/ultralytics, so it lives in
its **own isolated venv** separate from the worker's main environment; set it up
once **on the worker**:

```bash
media age-setup      # creates ~/.local/share/media_manager/age-venv (uses uv if present)
```

Like the other offloads it is **worker-only, no local fallback**: if the worker
is unreachable — or you skipped `media age-setup` on it — an Estimate click fails
loudly rather than running MiVOLO on the host.

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
├── worker_server.py      # `media worker` — ML-offload HTTP service (FastAPI/uvicorn)
├── worker_client.py      # host-side HTTP client + drop-in Remote* model proxies
├── worker_protocol.py    # shared wire contract (paths + msgpack pack/unpack)
├── worker_config.py      # .media/worker.json + MEDIA_WORKER_* env
├── web.py                # FastAPI gallery
├── templates/, static/   # Web UI assets
└── ...
```

## Development

See [TODO.md](TODO.md) and [features.todo.md](features.todo.md) for the roadmap.
