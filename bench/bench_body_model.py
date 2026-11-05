"""Body re-ID benchmark: DINOv2-L vs SigLIP 2 on person crops. RUN ON THE GPU MACHINE (bigboy);
the embedders are GPU-only.

Ground truth comes from your own library: every named face (manual.db) whose photo has a stored
person box containing that face (body_index.match_face_to_body) yields one labeled body crop
(identity = the face's name). Each crop is then used as a query against all OTHER crops from
OTHER photos; a hit is a crop of the same identity. Reports, per model: Recall@1, Recall@5, mAP,
and the mean cosine of same- vs different-identity pairs (to pick MEDIA_BODY_THRESHOLD).
Photos of the same identity taken in the same event/outfit are easy; this does not separate
outfit changes, so also eyeball /find-person/<name>/body after flipping MEDIA_BODY_MODEL.

    cd <repo>/media_manager
    .venv/bin/python -I bench/bench_body_model.py --data-root /mnt/raid/pics --max-crops 1500 \
        [--json bench/results/body_model.json]
Then set MEDIA_BODY_MODEL=dino|siglip on the web server and rerun the Bodies stage.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from media_manager import body_index  # noqa: E402
from media_manager.database import Database  # noqa: E402
from media_manager.manual_db import ManualDB  # noqa: E402


def collect(db, manual, data_root, max_crops, min_per_identity=2):
    """[(photo_checksum, identity, PIL crop)] from named faces matched to stored person boxes."""
    by_id = {}
    for checksum, identity, fbox in manual.get_named_face_boxes():
        frow = db.get_file_by_checksum(checksum)
        if frow is None:
            continue
        boxes = db.get_person_detections_for_file(
            frow['id'], min_conf=body_index.MIN_PERSON_CONFIDENCE, class_names=body_index.PERSON_LIKE_CLASSES)
        bbox = body_index.match_face_to_body(fbox, boxes)
        if bbox is not None:
            by_id.setdefault(identity, []).append((checksum, frow['path'], bbox))
    items = []
    for identity, rows in by_id.items():
        if len(rows) < min_per_identity:
            continue
        for checksum, path, bbox in rows:
            absp = os.path.join(data_root, path)
            if os.path.isfile(absp):
                pairs = body_index.crop_bodies(absp, [bbox])
                if pairs:
                    items.append((checksum, identity, pairs[0][1]))
    rng = np.random.default_rng(0)
    if len(items) > max_crops:
        items = [items[i] for i in rng.choice(len(items), max_crops, replace=False)]
    return items


def evaluate(emb, ids, photos):
    ids, photos = np.asarray(ids), np.asarray(photos)
    sim = emb @ emb.T
    r1 = r5 = aps = n = 0
    same, diff = [], []
    for i in range(len(ids)):
        valid = photos != photos[i]                     # other photos only
        rel = (ids == ids[i]) & valid
        if not rel.any():
            continue
        s = np.where(valid, sim[i], -np.inf)
        order = np.argsort(-s)[:valid.sum()]
        hits = rel[order]
        r1 += hits[0]
        r5 += hits[:5].any()
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps += (prec * hits).sum() / hits.sum()
        n += 1
        same.extend(sim[i][rel]); diff.extend(sim[i][valid & ~rel])
    return {'queries': n, 'R@1': r1 / n, 'R@5': r5 / n, 'mAP': aps / n,
            'cos_same_mean': float(np.mean(same)), 'cos_diff_mean': float(np.mean(diff)),
            'cos_diff_p99': float(np.percentile(diff, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--media-dir', default=None, help='dir holding media.db/manual.db (default <data-root>/.media)')
    ap.add_argument('--max-crops', type=int, default=1500)
    ap.add_argument('--json')
    a = ap.parse_args()
    media_dir = a.media_dir or os.path.join(a.data_root, '.media')
    db = Database(os.path.join(media_dir, 'media.db'))
    manual = ManualDB(os.path.join(media_dir, 'manual.db'))
    items = collect(db, manual, a.data_root, a.max_crops)
    print(f'{len(items)} labeled body crops, {len({i[1] for i in items})} identities')
    if len(items) < 10:
        sys.exit('not enough labeled bodies (need named faces whose photos have a stored person box)')
    from media_manager.embedders import DINOv2Embedder, SigLIP2Embedder
    out = {}
    for name, cls in (('dino', DINOv2Embedder), ('siglip', SigLIP2Embedder)):
        emb = cls().embed_pil_images([c for _, _, c in items])
        out[name] = evaluate(np.asarray(emb, np.float32), [i[1] for i in items], [i[0] for i in items])
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
