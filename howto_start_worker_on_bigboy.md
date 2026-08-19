# How to start the media worker on **bigboy** (local runbook — do not commit)

Concrete, machine-specific steps for *this* setup. General docs are in the
README's "remote ML worker" section.

## The players

| Role   | Machine | Notes |
| ------ | ------- | ----- |
| Worker | **bigboy** — `rafa@bigboy` | 62 GB RAM, Docker host. Runs the heavy models. Repo/venv at `/mnt/raid/pic_master`. |
| Host   | the box running `media web` | Low RAM. Offloads to bigboy. |

- Worker **destination address**: `420ddbf528615e317e3f2286d24988c6`
  (stable — persisted at `~/.config/media_manager/worker_identity` on bigboy).
- bigboy is reachable from the host on the **LAN** `192.168.1.66` and over
  **Tailscale** `100.99.33.116`. Worker listens on TCP **4242** (`0.0.0.0`).
- bigboy's host firewall is inactive; port 4242 is open.

## One-time setup on bigboy

bigboy's `/mnt/raid/pic_master/.venv` is a `uv` venv on **Python 3.12** (the
default empty 3.14 venv had no ML wheels — recreate with `uv venv --python 3.12`
if it's ever wrong).

```bash
ssh rafa@bigboy
export PATH=$HOME/.local/bin:$PATH        # uv lives here
cd /mnt/raid/pic_master

# (re)create a 3.12 venv if needed:
# uv venv --python 3.12 .venv

# install / update the package from main (deps are cached after the first run):
uv pip install --python .venv/bin/python \
  "git+https://github.com/KUKARAF/pic_master.git"

# RNS config: a TCP server on 4242 (avoids AutoInterface fighting bigboy's ~40
# docker bridges). Written once:
mkdir -p ~/.reticulum
cat > ~/.reticulum/config <<'EOF'
[reticulum]
  enable_transport = No
  share_instance = No
  panic_on_interface_error = No

[logging]
  loglevel = 4

[interfaces]

  [[TCP Server Interface]]
    type = TCPServerInterface
    interface_enabled = yes
    listen_ip = 0.0.0.0
    listen_port = 4242
EOF
```

## Start the worker (every time)

```bash
ssh rafa@bigboy
cd /mnt/raid/pic_master
nohup .venv/bin/media worker --announce-interval 20 > worker.log 2>&1 &
tail -f worker.log         # should print "media worker is online" + the address
```

Foreground alternative (Ctrl-C to stop): `.venv/bin/media worker`.
Add `--preload` to load all three models at startup instead of lazily.

Check it's up / watch what it handles:

```bash
ssh rafa@bigboy 'pgrep -x media && tail -n 20 /mnt/raid/pic_master/worker.log'
# log lines: "[worker] loading FaceDetector...", "[worker] handled detect_faces (...) -> N faces"
```

Stop it: `ssh rafa@bigboy 'pkill -x media'`.

## One-time setup on the host

```bash
# RNS config: TCP client to bigboy. Use the LAN IP if same network, else Tailscale.
mkdir -p ~/.reticulum
cat > ~/.reticulum/config <<'EOF'
[reticulum]
  enable_transport = No
  share_instance = No
  panic_on_interface_error = No

[logging]
  loglevel = 4

[interfaces]

  [[TCP Client to bigboy]]
    type = TCPClientInterface
    interface_enabled = yes
    target_host = 192.168.1.66
    target_port = 4242
EOF
# (over Tailscale instead: target_host = 100.99.33.116)

# point this repo at the worker (writes .media/worker.json):
cd /path/to/your/media
media worker-connect 420ddbf528615e317e3f2286d24988c6
```

## Verify the link

```bash
# TCP reachable?
bash -c '</dev/tcp/192.168.1.66/4242 && echo reachable'

# real offload test (host RAM should stay flat; bigboy loads the model):
media faces            # look for "[media] offloading face detection to remote worker"
                       # and, on bigboy, "[worker] handled detect_faces ..."
```

In the web UI, the nav shows a green **Worker** badge when connected, and a
transient "Outsourced …" toast per dispatch. First badge can take ~15 s after
`media web` starts (cold RNS link).

## Gotchas

- **Worker won't start: "no media repo found"** — you're on an old build; the
  worker no longer needs a `.media/` repo. Reinstall from `main`.
- **Badge stuck offline but offloading works** — cold-start timing; it flips
  green within a poll cycle. If it never connects, re-check the TCP config /
  that `media worker` is running / port 4242 is reachable.
- **`uv pip install` rebuilds torch and fails** — the venv is on a too-new
  Python; recreate with `uv venv --python 3.12 .venv` and reinstall.
- **Worker offline → tasks run locally** (by design). On a low-RAM host that can
  re-introduce the OOM, so keep the worker up while indexing.
- The worker only runs models; it does **not** need your media files or a copy
  of the library.
