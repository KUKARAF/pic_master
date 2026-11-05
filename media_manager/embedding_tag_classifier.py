"""Per-tag embedding linear classifier — trains on whole-image tag
confirmations/rejections already in manual.db, scores every photo's already-
computed embedding with a single learned weight vector.

The embedding SPACE is selectable (see SPACES): 'siglip' (SigLIP 2, 1152-d,
the default) or 'dino' (DINOv2-L, 1024-d, image-only).  Each space has its own
artifact directory (.media/tag_classifiers/{slug}/{siglip,dino}_model/) and
weights.npz records the space + embedding model id it was trained on, so an
artifact is never silently applied to vectors from a different space: scoring
raises NeedsRetrain on any mismatch, and weights_state() reports 'needs_retrain'
for the UI.

The old CLIP (ViT-B/32) space is gone.  A leftover clip_model/ directory from
before is never read or scored (LEGACY_CLIP_DIR_KIND is only used so the UI can
say "retrain in SigLIP 2"); unstamped weights.npz files are treated as CLIP-era
and raise NeedsRetrain in any current space.

Meant to be run as a standalone subprocess (see web.py's train routes),
never imported into the running web app's own process — a multi-second-plus
training run must never share the app's GIL. Run directly:

    python -m media_manager.embedding_tag_classifier --data-root /path/to/repo --tag "monstera plant" [--space siglip|dino]

Output lands in .media/tag_classifiers/{slug}/{siglip,dino}_model/ — see write_status
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

DEFAULT_SPACE = 'siglip'

# space -> artifact dir kind, UI label, nominal dim (for messages only: the binding
# check is always against the actual vectors being scored, so env-overridden model
# variants and test fakes work), and the module-level default model id builder.
SPACES = {
    'siglip': {'dir_kind': 'siglip_model', 'label': 'SigLIP 2', 'dim': 1152, 'legacy': False},
    'dino': {'dir_kind': 'dino_model', 'label': 'DINOv2', 'dim': 1024, 'legacy': False},
}
# Pre-SigLIP artifact directory kind. Never trained, read or scored any more; the UI only
# looks for it to tell the user to retrain in SigLIP 2.
LEGACY_CLIP_DIR_KIND = 'clip_model'
CLASSIFIER_KINDS = tuple(v['dir_kind'] for v in SPACES.values()) + ('yolo_model',)


class NeedsRetrain(ValueError):
    """A stored classifier was trained in a different embedding space/model (or
    dimension) than the vectors it is being applied to.  Never apply it — retrain."""


def check_space(space):
    if space not in SPACES:
        raise ValueError(f'Unknown embedding space {space!r}; expected one of {tuple(SPACES)}')
    return space


def space_kind(space):
    """Artifact directory kind for a space ('siglip' -> 'siglip_model')."""
    return SPACES[check_space(space)]['dir_kind']


def current_embedding_model(space):
    """Model id the space's embeddings are produced with right now (env-aware, no
    torch import — mirrors embedders.SigLIP2Embedder/DINOv2Embedder.model_id)."""
    check_space(space)
    if space == 'siglip':
        return (f"{os.environ.get('MEDIA_SIGLIP_MODEL', 'ViT-SO400M-14-SigLIP2')}/"
                f"{os.environ.get('MEDIA_SIGLIP_PRETRAINED', 'webli')}")
    variant = os.environ.get('MEDIA_DINO_MODEL', 'large').lower()
    hub = {'small': 'dinov2_vits14_reg', 'base': 'dinov2_vitb14_reg', 'large': 'dinov2_vitl14_reg'}
    return f"{hub.get(variant, variant)}/cls"


def all_embeddings_for_space(db, space):
    """[(file_id, path, embedding_bytes, checksum)] of primary embeddings in `space`."""
    check_space(space)
    return getattr(db, f'get_all_{space}_embeddings')()


def _make_embedder(space):
    """Image embedder for region crops (GPU-only; raises on a machine without one)."""
    if space == 'siglip':
        from media_manager.embedders import SigLIP2Embedder  # hook-allow: lazy — pulls torch/open_clip
        return SigLIP2Embedder()
    from media_manager.embedders import DINOv2Embedder  # hook-allow: lazy — pulls torch
    return DINOv2Embedder()


def slugify(label):
    """Filesystem-safe directory name for a tag label — 'Monstera Plant' ->
    'monstera_plant'. Shared shape expected by web.py's status/train routes,
    which must derive the same slug from the same label independently."""
    slug = re.sub(r'[^a-z0-9]+', '_', label.strip().lower()).strip('_')
    return slug or 'tag'


def classifier_dir(data_root, tag_label, kind):
    """kind is one of CLASSIFIER_KINDS ('siglip_model', 'dino_model' or
    'yolo_model') — shared with yolo_tag_classifier.py
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


def train_for_tag(data_root, tag_label, space=DEFAULT_SPACE):
    """Core trainable entry point — also called directly by tests/other code,
    not just the CLI below. Never raises: any failure is caught and recorded
    in metadata.json as status='failed' so a subprocess crash always leaves a
    readable status behind instead of silently vanishing."""
    out_dir = classifier_dir(data_root, tag_label, space_kind(space))
    os.makedirs(out_dir, exist_ok=True)
    pid = os.getpid()
    started_at = int(time.time())
    write_status(out_dir, status='running', pid=pid, started_at=started_at)

    try:
        _train(data_root, tag_label, out_dir, space)
    except Exception as exc:  # hook-allow: broad — boundary records failure then re-raises
        write_status(out_dir, status='failed', pid=pid, started_at=started_at,
                     error=str(exc), failed_at=int(time.time()))
        raise


def _train(data_root, tag_label, out_dir, space=DEFAULT_SPACE):
    import torch  # hook-allow: lazy — heavy dep (torch)

    # Imported here (not at module load) so importing this module for its
    # slugify/classifier_dir helpers — e.g. from web.py, in-process — never
    # pulls in torch/manual_db/database just to compute a path.
    from media_manager.database import Database  # hook-allow: lazy — deliberately deferred, see comment above
    from media_manager.manual_db import ManualDB  # hook-allow: lazy — deliberately deferred, see comment above

    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))

    polarities = manual.get_whole_image_tag_polarities(tag_label)
    embedding_by_checksum = {row[3]: row[2] for row in all_embeddings_for_space(db, space)}

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
    # capability that was missing before this. Only loads the embedder (real model
    # weights, real memory) if there's actually at least one region to crop.
    region_rows = manual.get_all_spatial_tags_for_label(tag_label)
    if region_rows:
        n_region, n_skipped_region = _add_region_examples(db, data_root, region_rows, xs, ys, space)
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

    dims = {x.shape[0] for x in xs}
    if len(dims) != 1:
        raise NeedsRetrain(f'Mixed embedding dimensions {sorted(dims)} among the {space} training examples '
                           f'(the {SPACES[space]["label"]} index changed model mid-library) — re-embed, then retrain.')
    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    w, b, cv_accuracy = fit_from_examples(X, y)

    save_weights(os.path.join(out_dir, 'weights.npz'), w, b, space)
    write_status(
        out_dir, status='done', trained_at=int(time.time()), space=space,
        embedding_model=current_embedding_model(space), dim=int(w.shape[0]),
        n_positive=n_positive, n_negative=n_negative, cv_accuracy=round(cv_accuracy, 4),
        n_region_examples=n_region, n_region_skipped=n_skipped_region,
    )


def _add_region_examples(db, data_root, region_rows, xs, ys, space=DEFAULT_SPACE):
    """Crops each region row to its own image and embeds JUST that crop (not
    the whole photo) via the space's embedder.embed_pil_images — appends to xs/ys in
    place. Returns (n_added, n_skipped). Only instantiates the real
    model (real weights, real memory) when there's at least one region row to
    process — most tags will have none, and this function is a no-op call
    away from that cost in that case (see its caller)."""
    from PIL import Image as PILImage  # hook-allow: lazy — PIL only needed when regions exist

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
        except Exception:  # hook-allow: broad — per-region decode resilience
            n_skipped += 1
            continue
        crops.append(crop)
        crop_labels.append(1.0 if polarity == 'positive' else 0.0)

    if not crops:
        return 0, n_skipped

    embeddings = _make_embedder(space).embed_pil_images(crops)
    for emb, label in zip(embeddings, crop_labels):
        xs.append(emb.astype(np.float32))
        ys.append(label)
    return len(crops), n_skipped


def fit_from_examples(X, y):
    """Train the per-tag logistic model from stacked embeddings ``X`` [N, D] and
    labels ``y`` [N] (1.0 positive / 0.0 negative). Returns ``(w, b, cv_accuracy)``.
    Shared verbatim by the local subprocess trainer and the remote worker so the
    fit/evaluation logic lives once.

    Stratified 80/20 split so both classes appear in the held-out set — a plain
    random split can accidentally put all of one tiny class into training only,
    making the reported accuracy meaningless. Falls back to 5-fold CV (averaged
    accuracy) when there's too little data for a clean single split. The
    cv_accuracy model is only for measuring; the returned (w, b) is always
    retrained on every example so the 20% held out for evaluation isn't thrown
    away in the deployed artifact."""
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
        w, b = _fit_logistic(X, y)
    else:
        # 5-fold (or fewer, if a class is smaller than 5) CV accuracy, then a
        # final model trained on everything for the artifact that gets saved.
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

    return w, b, cv_accuracy


def _fit_logistic(X, y, epochs=200, lr=0.01):
    """Plain torch logistic regression — a single Linear(D, 1) layer (D = the space's dim). Data
    this small (tens to low hundreds of examples) doesn't need more than a
    full-batch Adam loop; no new dependency, torch is already required for
    embedding."""
    import torch  # hook-allow: lazy — heavy dep (torch)

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


def save_weights(path, w, b, space):
    """weights.npz stamped with the space + embedding model id + dim it was trained on."""
    np.savez(path, w=w, b=np.float32(b), space=space, dim=np.int64(w.shape[0]),
             embedding_model=current_embedding_model(space))


def _load_checked(data_root, tag_label, space):
    """(w, b) or None if no artifact exists; raises NeedsRetrain on a space/model mismatch
    (an artifact without a space stamp predates SigLIP/DINO — it was CLIP — so it is never valid)."""
    check_space(space)
    weights_path = os.path.join(classifier_dir(data_root, tag_label, space_kind(space)), 'weights.npz')
    if not os.path.isfile(weights_path):
        return None
    data = np.load(weights_path)
    stamped = str(data['embedding_model']) if 'embedding_model' in data.files else None
    stamped_space = str(data['space']) if 'space' in data.files else 'clip'   # unstamped == CLIP era
    if stamped_space != space or (stamped is not None and stamped != current_embedding_model(space)):
        raise NeedsRetrain(
            f'"{tag_label}" classifier was trained on {stamped or "an unknown model"} ({stamped_space}), '
            f'but the current {SPACES[space]["label"]} model is {current_embedding_model(space)} — retrain it.')
    return data['w'].astype(np.float32), float(data['b'])


def weights_state(data_root, tag_label, space=DEFAULT_SPACE):
    """'missing' | 'ok' | 'needs_retrain' for this tag's artifact in `space` (no scoring)."""
    try:
        return 'missing' if _load_checked(data_root, tag_label, space) is None else 'ok'
    except NeedsRetrain:
        return 'needs_retrain'


def score_all(data_root, tag_label, embeddings, space=DEFAULT_SPACE):
    """{checksum: score} for a batch of (checksum, embedding_bytes) pairs in `space`,
    using this tag's already-trained weights — a single vectorized matmul,
    no torch needed at inference time. Returns {} if this tag has no done
    model in that space yet. Raises NeedsRetrain if the artifact was trained on
    a different model/dimension than the vectors passed in (never a silent
    wrong-dimension matmul). Used by web.py's classifier-ranked suggestion endpoint."""
    loaded = _load_checked(data_root, tag_label, space)
    if loaded is None:
        return {}
    w, b = loaded
    result = {}
    for checksum, emb_bytes in embeddings:
        vec = np.frombuffer(emb_bytes, dtype=np.float32)
        if vec.shape[0] != w.shape[0]:
            raise NeedsRetrain(
                f'"{tag_label}" classifier has dim {w.shape[0]} but {SPACES[space]["label"]} embeddings '
                f'have dim {vec.shape[0]} — retrain it.')
        result[checksum] = float(1.0 / (1.0 + np.exp(-(vec @ w + b))))
    return result


def score_matrix(data_root, tag_label, matrix, space=DEFAULT_SPACE):
    """Vectorized sigmoid scores float32[N] for an [N, D] embedding matrix, or None if
    no artifact; NeedsRetrain on any model/dim mismatch."""
    loaded = _load_checked(data_root, tag_label, space)
    if loaded is None:
        return None
    w, b = loaded
    if matrix.shape[0] and matrix.shape[1] != w.shape[0]:
        raise NeedsRetrain(
            f'"{tag_label}" classifier has dim {w.shape[0]} but {SPACES[space]["label"]} embeddings '
            f'have dim {matrix.shape[1]} — retrain it.')
    return (1.0 / (1.0 + np.exp(-(matrix @ w + b)))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a per-tag embedding linear classifier.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--space', default=DEFAULT_SPACE, choices=list(SPACES))
    args = parser.parse_args()

    out_dir = classifier_dir(args.data_root, args.tag, space_kind(args.space))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')
    # buffering=1 (line-buffered) — see yolo_tag_classifier.py's identical note.
    with open(log_path, 'w', buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        train_for_tag(args.data_root, args.tag, args.space)


if __name__ == '__main__':
    main()
