"""Per-tag YOLO-World fine-tune — trains on this tag's bbox-labeled examples
(see manual_db.add_spatial_tag) starting from the same cached zero-shot
checkpoint detector.py already uses, producing a checkpoint specialized for
just this one class.

Meant to be run as a standalone subprocess (see web.py's train-yolo route),
never imported into the running web app's own process — real training here
can take minutes and must never share the app's GIL. Run directly:

    python -m media_manager.yolo_tag_classifier --data-root /path/to/repo --tag "monstera plant"

Output lands in .media/tag_classifiers/{slug}/yolo_model/ — see
clip_tag_classifier.classifier_dir for the shared path convention and
write_status for the {running,done,failed} status contract.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

from media_manager.clip_tag_classifier import classifier_dir, slugify, write_status

MIN_BOXES = 10
EPOCHS = 30


class TrainingCancelled(Exception):
    """Raised inside the epoch callback to abort an in-progress ultralytics run
    (used by the remote worker's cancel path; the local trainer passes no
    cancel_cb, so it never fires there)."""


def train_from_dataset(data_yaml_path, out_dir, epochs=EPOCHS, progress_cb=None, cancel_cb=None):
    """Run the ultralytics YOLO-World fine-tune over a PREBUILT dataset dir
    (images/{train,val} + labels/{train,val}, referenced by data_yaml_path),
    copy the resulting best.pt into out_dir, and return
    ``{'held_out_map': float|None, 'best_path': str}``.

    Shared verbatim by the local subprocess trainer and the remote worker so the
    training logic exists once. ``progress_cb(current_epoch, total_epochs)`` is
    called at each epoch end; ``cancel_cb() -> bool``, when it returns True,
    aborts the run (raises :class:`TrainingCancelled`). Both are optional — the
    local trainer passes only progress_cb."""
    from media_manager.detector import WEIGHTS_DIR
    from ultralytics import YOLOWorld

    os.makedirs(out_dir, exist_ok=True)
    base_weights = os.path.join(WEIGHTS_DIR, 'yolov8s-worldv2.pt')
    model = YOLOWorld(base_weights)

    # trainer.epoch is 0-indexed, hence the +1. The cancel check runs at each
    # epoch boundary — ultralytics has no finer cooperative-stop hook, so a
    # cancel takes effect at the next epoch end (a known, documented wrinkle).
    def _on_epoch_end(trainer):
        if cancel_cb is not None and cancel_cb():
            raise TrainingCancelled('training cancelled')
        if progress_cb is not None:
            progress_cb(trainer.epoch + 1, epochs)
    model.add_callback('on_train_epoch_end', _on_epoch_end)

    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640,
                          project=out_dir, name='run', exist_ok=True)

    run_dir = getattr(results, 'save_dir', os.path.join(out_dir, 'run'))
    best_src = os.path.join(run_dir, 'weights', 'best.pt')
    best_dst = os.path.join(out_dir, 'best.pt')
    if not os.path.isfile(best_src):
        raise RuntimeError(f'Training finished but no checkpoint was produced at {best_src}.')
    shutil.copyfile(best_src, best_dst)

    held_out_map = None
    try:
        held_out_map = float(results.results_dict.get('metrics/mAP50-95(B)'))
    except Exception:
        pass  # metric extraction is best-effort — a missing/renamed key shouldn't fail an otherwise-successful run
    return {'held_out_map': held_out_map, 'best_path': best_dst}


def train_for_tag(data_root, tag_label):
    """Core trainable entry point. Never raises: any failure is caught and
    recorded in metadata.json as status='failed' — see clip_tag_classifier's
    train_for_tag for the identical reasoning."""
    out_dir = classifier_dir(data_root, tag_label, 'yolo_model')
    os.makedirs(out_dir, exist_ok=True)
    pid = os.getpid()
    started_at = int(time.time())
    write_status(out_dir, status='running', pid=pid, started_at=started_at)

    try:
        _train(data_root, tag_label, out_dir, pid, started_at)
    except Exception as exc:
        write_status(out_dir, status='failed', pid=pid, started_at=started_at,
                     error=str(exc), failed_at=int(time.time()))
        raise


def _train(data_root, tag_label, out_dir, pid, started_at):
    from media_manager.database import Database
    from media_manager.manual_db import ManualDB

    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    rows = manual.get_spatial_tags_for_label(tag_label)
    if len(rows) < MIN_BOXES:
        raise ValueError(f'Need at least {MIN_BOXES} labeled bounding boxes for this tag; have {len(rows)}.')

    checksums = {r[0] for r in rows}
    file_rows = {r['checksum']: r for r in db.get_files_by_checksums(list(checksums))}

    # One example per SOURCE IMAGE (not per box) — a photo with multiple boxes
    # for this tag needs all of them in the same label file, one line each,
    # which is the standard YOLO-detect layout.
    boxes_by_checksum = {}
    for checksum, x1, y1, x2, y2, image_width, image_height in rows:
        boxes_by_checksum.setdefault(checksum, []).append((x1, y1, x2, y2, image_width, image_height))

    # Existing-on-disk, actually-decodable examples only, checksums in a stable
    # order so the 80/20 split below is deterministic across runs. A file can
    # exist and even open fine in PIL but still fail here — cv2 (what
    # ultralytics actually decodes with at train time) doesn't support every
    # format PIL does (e.g. some AVIF files saved under a .jpg extension), and
    # ultralytics reports that failure as a confusing "Image Not Found"
    # mid-training rather than a clean skip — so validate with the SAME
    # decoder ultralytics uses, up front.
    import cv2

    examples = []  # [(checksum, src_path, boxes), ...]
    for checksum in sorted(boxes_by_checksum):
        file_row = file_rows.get(checksum)
        if file_row is None:
            continue  # tagged photo no longer tracked — skip rather than fail the whole run
        src_path = os.path.join(data_root, file_row['path'])
        if not os.path.isfile(src_path) or cv2.imread(src_path) is None:
            continue  # missing, corrupt, or a format cv2 can't decode — skip rather than fail the whole run
        examples.append((checksum, src_path, boxes_by_checksum[checksum]))

    n_boxes_available = sum(len(boxes) for _checksum, _src, boxes in examples)
    if n_boxes_available < MIN_BOXES:
        raise ValueError(
            f'Only {n_boxes_available} of {len(rows)} labeled boxes are still on disk '
            f'(some tagged photos may have been removed/moved) — need at least {MIN_BOXES}.'
        )

    # 80/20 train/val split by IMAGE (not by box), so a multi-box image
    # doesn't leak between train and val.
    import random
    rng = random.Random(0)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.2))
    val_examples, train_examples = shuffled[:split], shuffled[split:]
    if not train_examples:
        train_examples, val_examples = val_examples, train_examples

    # Standard ultralytics YOLO-detect dataset layout: images/{train,val}/ +
    # labels/{train,val}/, one label file per image with the same stem — the
    # well-tested convention (a flat images/ dir + train.txt/val.txt list
    # files pointing into it is also nominally supported, but hit an
    # unresolved ultralytics path-resolution bug during manual testing here;
    # this layout is what ultralytics' own examples/tests actually use).
    scratch_dir = tempfile.mkdtemp(prefix=f'yolo_tag_{slugify(tag_label)}_')

    def write_split(split_name, split_examples):
        images_dir = os.path.join(scratch_dir, 'images', split_name)
        labels_dir = os.path.join(scratch_dir, 'labels', split_name)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        for i, (_checksum, src_path, boxes) in enumerate(split_examples):
            ext = os.path.splitext(src_path)[1] or '.jpg'
            name = f'{i:05d}'
            shutil.copyfile(src_path, os.path.join(images_dir, name + ext))
            label_lines = []
            for x1, y1, x2, y2, image_width, image_height in boxes:
                cx = (x1 + x2) / 2.0 / image_width
                cy = (y1 + y2) / 2.0 / image_height
                w = (x2 - x1) / image_width
                h = (y2 - y1) / image_height
                label_lines.append(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
            with open(os.path.join(labels_dir, name + '.txt'), 'w') as f:
                f.write('\n'.join(label_lines) + '\n')

    write_split('train', train_examples)
    write_split('val', val_examples or train_examples)

    data_yaml_path = os.path.join(scratch_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(
            f"path: {scratch_dir}\ntrain: images/train\nval: images/val\n"
            f"nc: 1\nnames: ['{tag_label}']\n"
        )

    # Surfaces real progress (not just "running" + elapsed time) for a job that
    # can take minutes. Re-supplies pid/started_at on every write since
    # write_status overwrites the whole file rather than merging.
    def _progress(current_epoch, total_epochs):
        write_status(
            out_dir, status='running', pid=pid, started_at=started_at,
            current_epoch=current_epoch, total_epochs=total_epochs,
        )

    res = train_from_dataset(data_yaml_path, out_dir, epochs=EPOCHS, progress_cb=_progress)

    write_status(
        out_dir, status='done', trained_at=int(time.time()),
        n_boxes=len(rows), n_images=len(examples), held_out_map=res['held_out_map'],
        checkpoint='best.pt',
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a per-tag YOLO-World checkpoint.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    out_dir = classifier_dir(args.data_root, args.tag, 'yolo_model')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')
    # buffering=1 (line-buffered): a redirected file defaults to fully
    # block-buffered, meaning the train-log viewer would show nothing until
    # the buffer happened to fill or the process exited — useless for
    # watching a run that's still in progress, which is the whole point.
    with open(log_path, 'w', buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        train_for_tag(args.data_root, args.tag)


if __name__ == '__main__':
    main()
