"""Host-side driver that runs per-tag training on the remote Reticulum worker
instead of locally — a drop-in replacement subprocess for
``clip_tag_classifier`` / ``yolo_tag_classifier`` when a worker is configured.

It honours the SAME on-disk contract those local trainers use (metadata.json via
``write_status``, a ``train.log`` under ``classifier_dir``), so web.py's status /
log / cancel / button-state code and the templates are untouched: the only
difference is where the heavy work runs. Launched exactly like the local
trainers, plus a ``--kind``:

    python -m media_manager.remote_tag_trainer --data-root <d> --tag <t> --kind yolo_model

This process stays LIGHT — it never imports torch/ultralytics/insightface. It
gathers the tag's examples with the same DB accessors the local trainers use,
downscales the images (YOLO trains at imgsz=640; boxes are stored normalized, so
this is label-invariant and cuts RNS transfer ~10-50x), ships them to the worker
in batches, then polls the worker job and mirrors its progress into
metadata.json / train.log. YOLO checkpoints stay on the worker (served by
tag_detect); the small CLIP weights.npz artifact is pulled back so score_all can
run locally.

Fail-loud: if the worker is unreachable there is NO local fallback (that would
reintroduce the OOM the offload exists to prevent) — the job is recorded failed
with a clear message.
"""
import argparse
import io
import os
import signal
import sys
import time

from media_manager.clip_tag_classifier import (
    classifier_dir, slugify, write_status, MIN_EXAMPLES_PER_CLASS)
from media_manager.yolo_tag_classifier import MIN_BOXES

_MAX_LONG_SIDE = 640          # ultralytics trains at imgsz=640; no point shipping bigger
_YOLO_UPLOAD_BATCH = 16       # images per train_add — keeps each RNS request modest
_CLIP_UPLOAD_BATCH = 64
_POLL_INTERVAL = 3.0
_MAX_POLL_ERRORS = 5          # consecutive status-poll failures before giving up

# Set once the remote job exists, so the SIGTERM handler (web.py's cancel path
# kills this driver) can cancel the worker-side job.
_active = {'client': None, 'job_id': None}

# train.log is owned by this driver (not via stdout redirect): it combines the
# driver's own progress with the worker's cumulative log tail on each poll.
_driver_lines = []
_worker_tail = ['']


def _flush_log(out_dir):
    text = '\n'.join(_driver_lines)
    if _worker_tail[0]:
        text += '\n\n--- worker ---\n' + _worker_tail[0]
    with open(os.path.join(out_dir, 'train.log'), 'w') as f:
        f.write(text + '\n')


def _log(out_dir, msg):
    _driver_lines.append(f'[{int(time.time())}] {msg}')
    _flush_log(out_dir)


def _client(data_root):
    from media_manager import worker_client
    return worker_client.get_client(data_root)


def _downscale_jpeg(img, max_long_side=_MAX_LONG_SIDE):
    """PIL image -> JPEG bytes, downscaled so its long side is <= max_long_side."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_long_side:
        scale = max_long_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def train_for_tag(data_root, tag_label, kind):
    """Entry point mirroring the local trainers' train_for_tag. Never lets an
    exception escape without first recording status='failed' so a crash always
    leaves a readable status behind."""
    out_dir = classifier_dir(data_root, tag_label, kind)
    os.makedirs(out_dir, exist_ok=True)
    pid = os.getpid()
    started_at = int(time.time())
    write_status(out_dir, status='running', pid=pid, started_at=started_at)
    try:
        if kind == 'yolo_model':
            _drive_yolo(data_root, tag_label, out_dir, pid, started_at)
        else:
            _drive_clip(data_root, tag_label, out_dir, pid, started_at)
    except Exception as exc:
        write_status(out_dir, status='failed', pid=pid, started_at=started_at,
                     error=str(exc), failed_at=int(time.time()))
        _log(out_dir, f'FAILED: {exc}')
        raise


def _require_worker(client, out_dir):
    if not client.is_available():
        raise RuntimeError(
            'Worker unavailable — start/configure the media worker to train '
            '(this build offloads training to the worker with no local fallback).')


def _drive_yolo(data_root, tag_label, out_dir, pid, started_at):
    from media_manager.database import Database
    from media_manager.manual_db import ManualDB
    from PIL import Image as PILImage

    client = _client(data_root)
    _require_worker(client, out_dir)

    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    rows = manual.get_spatial_tags_for_label(tag_label)
    if len(rows) < MIN_BOXES:
        raise ValueError(f'Need at least {MIN_BOXES} labeled bounding boxes for this tag; have {len(rows)}.')

    checksums = {r[0] for r in rows}
    file_rows = {r['checksum']: r for r in db.get_files_by_checksums(list(checksums))}

    boxes_by_checksum = {}
    for checksum, x1, y1, x2, y2, image_width, image_height in rows:
        boxes_by_checksum.setdefault(checksum, []).append((x1, y1, x2, y2, image_width, image_height))

    slug = slugify(tag_label)
    job_id = client.train_create('yolo_model', tag_label, slug, epochs=None)
    _active.update(client=client, job_id=job_id)
    _log(out_dir, f'created remote job {job_id}; uploading downscaled images')

    batch, n_images, n_boxes, n_skipped = [], 0, 0, 0
    for checksum in sorted(boxes_by_checksum):
        file_row = file_rows.get(checksum)
        if file_row is None:
            n_skipped += 1
            continue
        src_path = os.path.join(data_root, file_row['path'])
        try:
            with PILImage.open(src_path) as im:
                img = im.convert('RGB')
                img.load()
        except Exception:
            n_skipped += 1  # missing / undecodable — skip rather than fail the whole run
            continue
        image_bytes = _downscale_jpeg(img)
        norm_boxes = []
        for x1, y1, x2, y2, bw, bh in boxes_by_checksum[checksum]:
            norm_boxes.append([(x1 + x2) / 2.0 / bw, (y1 + y2) / 2.0 / bh,
                               (x2 - x1) / bw, (y2 - y1) / bh])
        batch.append({'name': os.path.basename(src_path), 'image': image_bytes, 'boxes': norm_boxes})
        n_images += 1
        n_boxes += len(norm_boxes)
        if len(batch) >= _YOLO_UPLOAD_BATCH:
            client.train_add(job_id, 'yolo_model', batch)
            _log(out_dir, f'uploaded {n_images} images...')
            batch = []
    if batch:
        client.train_add(job_id, 'yolo_model', batch)

    if n_boxes < MIN_BOXES:
        client.train_cancel(job_id)
        raise ValueError(
            f'Only {n_boxes} of {len(rows)} labeled boxes are still on disk '
            f'(some tagged photos may have been removed/moved) — need at least {MIN_BOXES}.')

    _log(out_dir, f'uploaded {n_images} images / {n_boxes} boxes ({n_skipped} skipped); starting training')
    client.train_run(job_id)
    _poll_until_done(client, job_id, out_dir, pid, started_at, kind='yolo_model',
                     done_extra={'n_boxes': len(rows), 'n_images': n_images})


def _drive_clip(data_root, tag_label, out_dir, pid, started_at):
    from media_manager.database import Database
    from media_manager.manual_db import ManualDB
    from PIL import Image as PILImage

    client = _client(data_root)
    _require_worker(client, out_dir)

    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    slug = slugify(tag_label)
    job_id = client.train_create('clip_model', tag_label, slug, epochs=None)
    _active.update(client=client, job_id=job_id)
    _log(out_dir, f'created remote job {job_id}; uploading examples')

    # Whole-image examples ship as their already-stored ~2 KB CLIP embeddings —
    # no image files, no model needed on either side for these.
    polarities = manual.get_whole_image_tag_polarities(tag_label)
    embedding_by_checksum = {row[3]: row[2] for row in db.get_all_embeddings()}
    batch, n_pos, n_neg = [], 0, 0
    for checksum, polarity in polarities.items():
        emb = embedding_by_checksum.get(checksum)
        if emb is None:
            continue
        label = 1 if polarity == 'positive' else 0
        batch.append({'label': label, 'embedding': emb})
        n_pos += label
        n_neg += (1 - label)
        if len(batch) >= _CLIP_UPLOAD_BATCH:
            client.train_add(job_id, 'clip_model', batch)
            batch = []

    # Region examples must be cropped from the original (the crop's OWN embedding
    # is more precise than the whole-image one). Crop + downscale on the host and
    # ship the small JPEG; the worker embeds it with its CLIP model.
    region_rows = manual.get_all_spatial_tags_for_label(tag_label)
    n_region, n_region_skipped = 0, 0
    if region_rows:
        checksums = {r[0] for r in region_rows}
        file_rows = {r['checksum']: r for r in db.get_files_by_checksums(list(checksums))}
        for checksum, x1, y1, x2, y2, _iw, _ih, polarity in region_rows:
            file_row = file_rows.get(checksum)
            if file_row is None:
                n_region_skipped += 1
                continue
            abs_path = os.path.join(data_root, file_row['path'])
            try:
                with PILImage.open(abs_path) as im:
                    crop = im.convert('RGB').crop((x1, y1, x2, y2))
                    crop.load()
            except Exception:
                n_region_skipped += 1
                continue
            label = 1 if polarity == 'positive' else 0
            batch.append({'label': label, 'image': _downscale_jpeg(crop)})
            n_pos += label
            n_neg += (1 - label)
            n_region += 1
            if len(batch) >= _CLIP_UPLOAD_BATCH:
                client.train_add(job_id, 'clip_model', batch)
                batch = []
    if batch:
        client.train_add(job_id, 'clip_model', batch)

    if n_pos < MIN_EXAMPLES_PER_CLASS or n_neg < MIN_EXAMPLES_PER_CLASS:
        client.train_cancel(job_id)
        raise ValueError(
            f'Need at least {MIN_EXAMPLES_PER_CLASS} confirmed and {MIN_EXAMPLES_PER_CLASS} rejected '
            f'examples (whole-image or region) with an embedded/loadable photo; '
            f'have {n_pos} confirmed, {n_neg} rejected.')

    _log(out_dir, f'uploaded {n_pos} pos / {n_neg} neg (regions: {n_region}, skipped {n_region_skipped}); starting training')
    client.train_run(job_id)
    _poll_until_done(client, job_id, out_dir, pid, started_at, kind='clip_model',
                     done_extra={'n_region_examples': n_region, 'n_region_skipped': n_region_skipped})


def _poll_until_done(client, job_id, out_dir, pid, started_at, kind, done_extra):
    consecutive_errors = 0
    while True:
        time.sleep(_POLL_INTERVAL)
        try:
            st = client.train_status(job_id)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            _log(out_dir, f'status poll error ({exc}) [{consecutive_errors}/{_MAX_POLL_ERRORS}]')
            if consecutive_errors >= _MAX_POLL_ERRORS:
                raise RuntimeError(f'Lost contact with the worker during training: {exc}')
            continue

        _worker_tail[0] = st.get('log_tail') or ''
        _flush_log(out_dir)
        status = st.get('status')
        if status == 'running':
            write_status(out_dir, status='running', pid=pid, started_at=started_at,
                         current_epoch=st.get('current_epoch', 0),
                         total_epochs=st.get('total_epochs', 0))
        elif status == 'done':
            _finalize_done(client, job_id, out_dir, kind, st.get('metrics') or {}, done_extra)
            return
        elif status in ('failed', 'cancelled'):
            write_status(out_dir, status='failed', pid=pid, started_at=started_at,
                         error=st.get('error') or f'worker training {status}',
                         failed_at=int(time.time()))
            raise RuntimeError(st.get('error') or f'worker training {status}')


def _finalize_done(client, job_id, out_dir, kind, metrics, done_extra):
    if kind == 'clip_model':
        # Pull the small weights.npz back so score_all runs locally on the host.
        art = client.train_status(job_id, want_artifact=True)
        blob = art.get('artifact')
        if not blob:
            raise RuntimeError('worker reported CLIP training done but returned no weights artifact')
        with open(os.path.join(out_dir, 'weights.npz'), 'wb') as f:
            f.write(blob)
        write_status(
            out_dir, status='done', trained_at=int(time.time()),
            n_positive=metrics.get('n_positive'), n_negative=metrics.get('n_negative'),
            cv_accuracy=metrics.get('cv_accuracy'),
            n_region_examples=done_extra.get('n_region_examples', 0),
            n_region_skipped=done_extra.get('n_region_skipped', 0),
            backend='worker')
    else:
        # YOLO: the checkpoint stays on the worker (served by tag_detect); nothing
        # to pull. checkpoint='worker' tells the host inference path to go remote.
        write_status(
            out_dir, status='done', trained_at=int(time.time()),
            n_boxes=done_extra.get('n_boxes'), n_images=done_extra.get('n_images'),
            held_out_map=metrics.get('held_out_map'),
            checkpoint='worker', backend='worker')
    _log(out_dir, f'training complete ({kind}); metrics={metrics}')


def _sigterm(signum, frame):
    """web.py's cancel route kills this driver with SIGTERM (and writes the
    'failed: Cancelled' status itself). Relay the cancel to the worker so the
    remote job actually stops, then exit."""
    client, job_id = _active.get('client'), _active.get('job_id')
    if client is not None and job_id is not None:
        try:
            client.train_cancel(job_id)
        except Exception:
            pass
    sys.exit(143)


def main():
    parser = argparse.ArgumentParser(description='Drive per-tag training on the remote worker.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--kind', required=True, choices=['clip_model', 'yolo_model'])
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    train_for_tag(args.data_root, args.tag, args.kind)


if __name__ == '__main__':
    main()
