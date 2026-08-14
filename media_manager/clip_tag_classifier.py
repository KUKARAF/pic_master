"""Per-tag CLIP-embedding linear classifier — trains on whole-image tag
confirmations/rejections already in manual.db, scores every photo's already-
computed CLIP embedding (see indexer.py) with a single learned weight vector.

Meant to be run as a standalone subprocess (see web.py's train-clip route),
never imported into the running web app's own process — a multi-second-plus
training run must never share the app's GIL. Run directly:

    python -m media_manager.clip_tag_classifier --data-root /path/to/repo --tag "monstera plant"

Output lands in .media/tag_classifiers/{slug}/clip_model/ — see write_status
for the exact files and the {running,done,failed} status contract every
caller (web.py's status routes, tags.html/search.html) reads back.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

MIN_EXAMPLES_PER_CLASS = 5


def slugify(label):
    """Filesystem-safe directory name for a tag label — 'Monstera Plant' ->
    'monstera_plant'. Shared shape expected by web.py's status/train routes,
    which must derive the same slug from the same label independently."""
    slug = re.sub(r'[^a-z0-9]+', '_', label.strip().lower()).strip('_')
    return slug or 'tag'


def classifier_dir(data_root, tag_label, kind):
    """kind is 'clip_model' or 'yolo_model' — shared with yolo_tag_classifier.py
    and web.py so every caller agrees on the exact same path for a given tag."""
    return os.path.join(os.path.abspath(data_root), '.media', 'tag_classifiers', slugify(tag_label), kind)


def _atomic_write_json(path, data):
    """Write-to-temp-then-rename so a reader (web.py's status route, possibly
    running in a different process) never sees a half-written metadata.json —
    os.rename is atomic on POSIX."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def write_status(out_dir, **fields):
    _atomic_write_json(os.path.join(out_dir, 'metadata.json'), fields)


def train_for_tag(data_root, tag_label):
    """Core trainable entry point — also called directly by tests/other code,
    not just the CLI below. Never raises: any failure is caught and recorded
    in metadata.json as status='failed' so a subprocess crash always leaves a
    readable status behind instead of silently vanishing."""
    out_dir = classifier_dir(data_root, tag_label, 'clip_model')
    os.makedirs(out_dir, exist_ok=True)
    pid = os.getpid()
    started_at = int(time.time())
    write_status(out_dir, status='running', pid=pid, started_at=started_at)

    try:
        _train(data_root, tag_label, out_dir)
    except Exception as exc:
        write_status(out_dir, status='failed', pid=pid, started_at=started_at,
                     error=str(exc), failed_at=int(time.time()))
        raise


def _train(data_root, tag_label, out_dir):
    import torch

    # Imported here (not at module load) so importing this module for its
    # slugify/classifier_dir helpers — e.g. from web.py, in-process — never
    # pulls in torch/manual_db/database just to compute a path.
    from media_manager.database import Database
    from media_manager.manual_db import ManualDB

    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    polarities = manual.get_whole_image_tag_polarities(tag_label)
    embedding_by_checksum = {row[3]: row[2] for row in db.get_all_embeddings()}

    xs, ys = [], []
    for checksum, polarity in polarities.items():
        emb_bytes = embedding_by_checksum.get(checksum)
        if emb_bytes is None:
            continue  # no embedding yet for this photo — skip rather than fail the whole run
        xs.append(np.frombuffer(emb_bytes, dtype=np.float32))
        ys.append(1.0 if polarity == 'positive' else 0.0)

    # Region-level examples (see manual_db.add_spatial_tag/get_all_spatial_tags_for_label)
    # — a cropped region's OWN embedding, not the whole photo's. Strictly more
    # precise than a whole-image polarity when the same photo contains both
    # the target and a look-alike (e.g. a monstera AND a dog in one shot) —
    # a hard-negative region crop (polarity='negative') is exactly the
    # capability that was missing before this. Only loads CLIP (real model
    # weights, real memory) if there's actually at least one region to crop.
    region_rows = manual.get_all_spatial_tags_for_label(tag_label)
    if region_rows:
        n_region, n_skipped_region = _add_region_examples(db, data_root, region_rows, xs, ys)
    else:
        n_region, n_skipped_region = 0, 0

    n_positive = sum(1 for y in ys if y == 1.0)
    n_negative = sum(1 for y in ys if y == 0.0)
    if n_positive < MIN_EXAMPLES_PER_CLASS or n_negative < MIN_EXAMPLES_PER_CLASS:
        raise ValueError(
            f'Need at least {MIN_EXAMPLES_PER_CLASS} confirmed and {MIN_EXAMPLES_PER_CLASS} rejected '
            f'examples (whole-image or region) with an embedded/loadable photo; '
            f'have {n_positive} confirmed, {n_negative} rejected.'
        )

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    # Stratified 80/20 split so both classes appear in the held-out set —
    # a plain random split can accidentally put all of one tiny class into
    # training only, making the reported accuracy meaningless. Falls back to
    # 5-fold CV (averaged accuracy) when there's too little data for a clean
    # single split to mean anything.
    rng = np.random.default_rng(0)
    pos_idx = rng.permutation(np.where(y == 1.0)[0])
    neg_idx = rng.permutation(np.where(y == 0.0)[0])

    if len(pos_idx) >= 10 and len(neg_idx) >= 10:
        pos_split = max(1, int(len(pos_idx) * 0.2))
        neg_split = max(1, int(len(neg_idx) * 0.2))
        val_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
        train_idx = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])
        held_out_w, held_out_b = _fit_logistic(X[train_idx], y[train_idx])
        cv_accuracy = _accuracy(X[val_idx], y[val_idx], held_out_w, held_out_b)
        # The held-out split model is only for measuring cv_accuracy — the
        # artifact actually saved is retrained on every example, so the 20%
        # held out for evaluation isn't simply thrown away in the deployed model.
        w, b = _fit_logistic(X, y)
    else:
        # 5-fold (or fewer, if a class is smaller than 5) CV accuracy, then a
        # final model trained on everything for the artifact that actually
        # gets saved.
        k = min(5, len(pos_idx), len(neg_idx))
        pos_folds = np.array_split(pos_idx, k)
        neg_folds = np.array_split(neg_idx, k)
        accuracies = []
        for i in range(k):
            val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
            train_idx = np.concatenate(
                [f for j, f in enumerate(pos_folds) if j != i] + [f for j, f in enumerate(neg_folds) if j != i]
            )
            fw, fb = _fit_logistic(X[train_idx], y[train_idx])
            accuracies.append(_accuracy(X[val_idx], y[val_idx], fw, fb))
        cv_accuracy = float(np.mean(accuracies))
        w, b = _fit_logistic(X, y)

    np.savez(os.path.join(out_dir, 'weights.npz'), w=w, b=np.float32(b), embedding_model='ViT-B-32/openai')
    write_status(
        out_dir, status='done', trained_at=int(time.time()),
        n_positive=n_positive, n_negative=n_negative, cv_accuracy=round(cv_accuracy, 4),
        n_region_examples=n_region, n_region_skipped=n_skipped_region,
    )


def _add_region_examples(db, data_root, region_rows, xs, ys):
    """Crops each region row to its own image and embeds JUST that crop (not
    the whole photo) via CLIPIndexer.embed_pil_images — appends to xs/ys in
    place. Returns (n_added, n_skipped). Only instantiates the real CLIP
    model (real weights, real memory) when there's at least one region row to
    process — most tags will have none, and this function is a no-op call
    away from that cost in that case (see its caller)."""
    from PIL import Image as PILImage
    from media_manager.indexer import CLIPIndexer

    checksums = {row[0] for row in region_rows}
    file_rows = {r['checksum']: r for r in db.get_files_by_checksums(list(checksums))}

    crops, crop_labels = [], []
    n_skipped = 0
    for checksum, x1, y1, x2, y2, _image_width, _image_height, polarity in region_rows:
        file_row = file_rows.get(checksum)
        if file_row is None:
            n_skipped += 1
            continue
        abs_path = os.path.join(data_root, file_row['path'])
        try:
            with PILImage.open(abs_path) as img:
                crop = img.convert('RGB').crop((x1, y1, x2, y2))
                crop.load()  # force decode while the file is still open
        except Exception:
            n_skipped += 1
            continue
        crops.append(crop)
        crop_labels.append(1.0 if polarity == 'positive' else 0.0)

    if not crops:
        return 0, n_skipped

    indexer = CLIPIndexer()
    embeddings = indexer.embed_pil_images(crops)
    for emb, label in zip(embeddings, crop_labels):
        xs.append(emb.astype(np.float32))
        ys.append(label)
    return len(crops), n_skipped


def _fit_logistic(X, y, epochs=200, lr=0.01):
    """Plain torch logistic regression — a single Linear(512, 1) layer. Data
    this small (tens to low hundreds of examples) doesn't need more than a
    full-batch Adam loop; no new dependency, torch is already required for
    CLIP embedding (see indexer.py)."""
    import torch

    torch.manual_seed(0)
    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)
    layer = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(layer(x_t), y_t)
        loss.backward()
        optimizer.step()
    w = layer.weight.detach().numpy().ravel().astype(np.float32)
    b = float(layer.bias.detach().numpy()[0])
    return w, b


def _accuracy(X, y, w, b):
    scores = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    preds = (scores >= 0.5).astype(np.float32)
    return float((preds == y).mean())


def score_all(data_root, tag_label, embeddings):
    """{checksum: score} for a batch of (checksum, embedding_bytes) pairs,
    using this tag's already-trained weights — a single vectorized matmul,
    no torch needed at inference time. Returns {} if this tag has no done
    CLIP model yet. Used by web.py's classifier-ranked suggestion endpoint."""
    weights_path = os.path.join(classifier_dir(data_root, tag_label, 'clip_model'), 'weights.npz')
    if not os.path.isfile(weights_path):
        return {}
    data = np.load(weights_path)
    w, b = data['w'], float(data['b'])
    result = {}
    for checksum, emb_bytes in embeddings:
        vec = np.frombuffer(emb_bytes, dtype=np.float32)
        result[checksum] = float(1.0 / (1.0 + np.exp(-(vec @ w + b))))
    return result


def main():
    parser = argparse.ArgumentParser(description='Train a per-tag CLIP linear classifier.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    out_dir = classifier_dir(args.data_root, args.tag, 'clip_model')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')
    # buffering=1 (line-buffered) — see yolo_tag_classifier.py's identical note.
    with open(log_path, 'w', buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        train_for_tag(args.data_root, args.tag)


if __name__ == '__main__':
    main()
