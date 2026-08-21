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
| `media bodies [path]` | Crop + embed person boxes for find-by-body search |
| `media who <image>` | Find which known people appear in an image |
| `media metadata [path]` | Read EXIF capture time + GPS |
| `media geo fetch-cities` | Download the offline GeoNames city database for reverse-geocoding |
| `media set create/ls/assign/files` | Manage named sets (e.g. a studio shoot) |
| `media web` | Launch the FastAPI gallery UI |
| `media worker` | Run the remote ML-offload worker on a beefy machine (see below) |
| `media worker-connect <hash>` | Point this host at a worker to offload heavy ML |

## Optional: offline city names (reverse-geocoding)

Run `media geo fetch-cities` once to download the [GeoNames](https://www.geonames.org/)
`cities15000` dataset (~26k cities, licensed CC-BY 4.0) into a local `cities` table — no
runtime network calls. Then the `🏙 Match cities` bulk action (⚡ menu) labels each
geotagged photo with its nearest city, so photos show a place **name** (and are searchable
by the `city:` facet) instead of raw coordinates. City data © GeoNames, CC-BY 4.0.

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

## Optional: remote ML worker (offload heavy models over Reticulum)

Face detection/embedding, CLIP, and YOLO-World are memory-hungry. On a
low-RAM host (or one running the web UI with multiple workers) they can OOM. The
`media worker` command runs those models on a **separate, beefier machine** and
the host offloads to it over [Reticulum](https://reticulum.network/) (RNS, an
encrypted networking stack). When a worker is reachable, `media faces` /
`media index` / `media embed` / `media bodies`, the web reindex/embed/face
endpoints, find-by-body, **per-tag classifier training** (the CLIP and
YOLO-World "Train" buttons), **and age/gender estimation** all send their work
to it — each dispatch is logged
(`[worker] outsourced …`) and shown in the web UI's worker badge. For the
*inference* offloads the host transparently falls back to running locally if the
worker is unreachable; **training does not fall back** — see below.

Only the *models* run remotely; your media files never need to live on the
worker (images are streamed to it per request). Results come back as embeddings
and are written to the host's `.media/` database exactly as if computed locally.

### 1. Install the package on both machines

Install `media` (this package) on the host and the worker the same way. The
worker also needs the ML dependencies (they ship in `requirements.txt`); use a
Python with wheels for your ML stack (3.11/3.12 are safe — very new interpreters
may lack torch/onnxruntime wheels).

### 2. Give both machines a Reticulum path to each other

The two machines must share a Reticulum network. The simplest reliable setup is
an explicit TCP link: run a **TCP server** interface on the worker and a **TCP
client** interface on the host. Create `~/.reticulum/config` on each:

**Worker** (`~/.reticulum/config`):

```ini
[reticulum]
  enable_transport = No
  share_instance = No

[interfaces]
  [[TCP Server Interface]]
    type = TCPServerInterface
    interface_enabled = yes
    listen_ip = 0.0.0.0
    listen_port = 4242
```

**Host** (`~/.reticulum/config`) — point `target_host` at the worker's IP:

```ini
[reticulum]
  enable_transport = No
  share_instance = No

[interfaces]
  [[Worker link]]
    type = TCPClientInterface
    interface_enabled = yes
    target_host = 192.168.1.66
    target_port = 4242
```

Make sure the worker's port (4242 here) is reachable from the host (open it in
any firewall). On a single flat LAN you can instead rely on Reticulum's default
`AutoInterface` (no IPs needed), but an explicit TCP interface is more
predictable — especially on a machine with many virtual/bridge interfaces (e.g.
a Docker host), where AutoInterface gets noisy.

### 3. Start the worker

On the worker machine:

```bash
media worker            # add --preload to load all models at startup
```

It prints its **destination address** (a hex hash) and keeps running, announcing
itself periodically. The address is stable across restarts (the worker persists
its identity under `~/.config/media_manager/`). Leave it running (in tmux/screen,
or `nohup media worker &`).

### 4. Point the host at the worker

On the host, inside your media repo:

```bash
media worker-connect <hex-address-from-step-3>
```

This writes `.media/worker.json`. From now on `media` commands and the web UI
offload to the worker. You can also set `MEDIA_WORKER_ADDR` /
`MEDIA_WORKER_ENABLED` as environment variables instead of the file, and
`media worker-connect <hash> --disable` saves the address but turns offloading
off.

### 5. Per-tag classifier training on the worker

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

### 6. Age/gender estimation on the worker

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
├── worker_server.py      # `media worker` — remote ML-offload server (Reticulum)
├── worker_client.py      # host-side client + drop-in Remote* model proxies
├── worker_protocol.py    # shared RNS wire contract
├── worker_config.py      # .media/worker.json + MEDIA_WORKER_* env
├── web.py                # FastAPI gallery
├── templates/, static/   # Web UI assets
└── ...
```

## Development

See [TODO.md](TODO.md) and [features.todo.md](features.todo.md) for the roadmap.
