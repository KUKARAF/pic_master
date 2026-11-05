"""Stage 2 of the 'place' location model: lazy person-masked re-embedding (no library re-scan).

Stage 1 (location_place.py) ranks from stored data only. Here the top MASK_TOPK candidates it
produced, plus the location's own photos, are embedded AGAIN with their person regions filled
by the mean colour of the rest of the frame, so SigLIP 2 / DINOv2 see only the place. The
masked candidates are scored against the masked members' centroid and the result is z-matched
into the stage-1 scale and blended (MASK_BLEND) with it.

Recipe (px1): pixel masking, not DINOv2 patch-token pooling. Reason: the masked image goes
through the SAME embedders and preprocessing as the stored vectors, so masked vectors live in
the same space as the library matrices. That lets photos that need no mask (scanned, no
person boxes) reuse their stored whole-image vector, and lets both models (SigLIP 2 and DINOv2)
be masked by one mechanism. Patch-token averaging would produce a different descriptor (mean
patch token != CLS) that is incomparable with every stored vector and would force re-embedding
the clean photos too.

Per photo (location_place.file_regions), status:
  clean      scanned, no person boxes (or < MASK_MIN_COVERAGE of the frame): stored vector reused
  masked     person boxes cover MASK_MIN_COVERAGE..MASK_MAX_COVERAGE of the frame: re-embedded
  covered    people fill more than MASK_MAX_COVERAGE: no usable background; keeps its stage-1
             score (and is not used as a member)
  unscanned  never scanned for persons (no detections / faces rows): NOT masked, NOT assumed
             clean; keeps its stage-1 score. (Simplest robust choice: running the object
             detector lazily would add a second worker round trip per candidate; the stage-1
             report and the UI tell the user how much of the library still needs the scan.)
  video      not an image: keeps its stage-1 score
Failure behaviour: an old worker (no embed_masked ops) or an unreachable one switches stage 2 off
for a back-off period with a visible reason; the request never waits longer than the budget and
the stage-1 ranking is always returned.
"""
import hashlib
import io
import os
import threading
import time
from collections import deque, namedtuple

import numpy as np

from . import location_place

RECIPE = 'px1'                        # bump when the masking recipe changes (cache key part)
MASK_TOPK = int(os.environ.get('MEDIA_LOC_MASK_TOPK', 200))
MASK_BUDGET_S = float(os.environ.get('MEDIA_LOC_MASK_BUDGET_S', 12))
MASK_MIN_COVERAGE = 0.03              # below this the mask changes nothing: use the stored vector
MASK_MAX_COVERAGE = 0.5               # above this there is no background to speak of
MASK_BLEND = 0.7                      # final = 0.7 * masked (z-matched) + 0.3 * stage 1
MASK_MEMBERS_MAX = 30                 # members re-embedded (the most representative ones)
MASK_MIN_SUBSET = 5                   # fewer scored candidates than this: no z-matching, no blend
MASK_CHUNK = int(os.environ.get('MEDIA_LOC_MASK_CHUNK', 32))
MASK_DECODE_LONG = 896                # masking happens at (at most) this long side
MASK_NEGATIVES_MAX = 50
FAIL_TTL_S = 600.0                    # a per-image failure is not retried for this long
BACKOFF_S = {'worker_outdated': 300.0, 'worker_unavailable': 60.0}
WEIGHTS = {'dino': 0.6, 'siglip': 0.4}    # same as web.LOCATION_FUSION_WEIGHTS

Plan = namedtuple('Plan', 'status boxes hash coverage')   # boxes: normalised [x1,y1,x2,y2] lists
Job = namedtuple('Job', 'fid hash path boxes coverage')


class MaskUnavailable(Exception):
    """Stage 2 cannot run right now. `kind` = 'worker_outdated' | 'worker_unavailable'."""

    def __init__(self, kind, message=''):
        super().__init__(message or kind)
        self.kind = kind


# ================================================================================ pixels

def mask_pil(img, boxes_norm):
    """(PIL RGB image with the boxes filled by the mean colour of the unmasked pixels, coverage).
    Boxes are normalised [x1, y1, x2, y2] in 0..1 (upright image). Coverage = masked pixel
    fraction. When nothing is left unmasked the image is returned unchanged with coverage 1.0."""
    from PIL import Image  # hook-allow: lazy -- keeps module import light
    a = np.asarray(img.convert('RGB'))
    h, w = a.shape[:2]
    m = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes_norm:
        xa, xb = int(np.floor(max(0.0, x1) * w)), int(np.ceil(min(1.0, x2) * w))
        ya, yb = int(np.floor(max(0.0, y1) * h)), int(np.ceil(min(1.0, y2) * h))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    cov = float(m.mean())
    if cov <= 0.0 or cov >= 1.0:
        return img.convert('RGB'), cov
    out = a.copy()
    out[m] = np.round(a[~m].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    return Image.fromarray(out), cov


def decode_masked(src, boxes_norm, long_side=MASK_DECODE_LONG):
    """Decode `src` (path or file object): first frame, EXIF-upright, RGB, shrunk to at most
    `long_side`, then masked. Returns (PIL image, coverage). Torch-free (runs on the worker)."""
    from PIL import Image, ImageOps  # hook-allow: lazy -- keeps module import light
    img = Image.open(src)
    try:
        img.seek(0)
    except EOFError:
        pass
    if img.format == 'JPEG':
        img.draft(None, (long_side, long_side))
    img = ImageOps.exif_transpose(img).convert('RGB')
    s = long_side / float(max(img.size))
    if s < 1.0:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return mask_pil(img, boxes_norm)


def normalize_regions(regions, w, h):
    return [[round(max(0.0, x1 / w), 4), round(max(0.0, y1 / h), 4),
             round(min(1.0, x2 / w), 4), round(min(1.0, y2 / h), 4)] for x1, y1, x2, y2 in regions]


def box_hash(boxes_norm):
    return hashlib.md5(repr(sorted(map(tuple, boxes_norm))).encode()).hexdigest()[:16]


def make_plans(db, file_ids):
    """{file_id: Plan} for the given files (see the module docstring for the statuses)."""
    plans = {}
    for fid, info in location_place.file_regions(db, file_ids).items():
        if not info['scanned']:
            plans[fid] = Plan('unscanned', [], '', 0.0)
            continue
        w, h = info['wh']
        regions = info['regions']
        cov = location_place.union_fraction(regions, w, h) if regions and w > 0 else 0.0
        if cov < MASK_MIN_COVERAGE:
            plans[fid] = Plan('clean', [], '', cov)
        elif cov > MASK_MAX_COVERAGE:
            plans[fid] = Plan('covered', [], '', cov)
        else:
            boxes = normalize_regions(regions, w, h)
            plans[fid] = Plan('masked', boxes, box_hash(boxes), cov)
    return plans


# ================================================================================ service

class MaskService:
    """Background embedder of masked images with a persistent cache (db.place_masked_embeddings).

    `embed_chunk(items)` -> (pairs, errors, models): items = [(abs_path, boxes_norm)], pairs[i] =
    (siglip, dino) float32 vectors or None, errors[i] = message or None, models = (siglip_model,
    dino_model) or None. It raises MaskUnavailable when the worker is old / unreachable.
    One daemon thread (one DB connection for its whole life) drains the queue, so a request that
    stops waiting at its budget leaves the remaining work running: it doubles as the prefetch
    that makes the next swipe-stack refill instant."""

    def __init__(self, db, embed_chunk, recipe=RECIPE, chunk=None, clock=time.monotonic):
        self.db, self.embed_chunk, self.recipe = db, embed_chunk, recipe
        self.chunk = chunk or MASK_CHUNK
        self.clock = clock
        self.cv = threading.Condition()
        self.queue = deque()
        self.pending = {}                  # (fid, hash) -> Job   (queued or in flight)
        self.failed = {}                   # (fid, hash) -> (ts, message)
        self.state = {'kind': None, 'until': 0.0, 'message': ''}
        self.thread = None
        self.stats = {'embedded': 0, 'batches': 0}

    # -- availability ---------------------------------------------------------------------
    def status(self):
        """(ok, kind, message): ok False while a back-off from a worker failure is running."""
        with self.cv:
            if self.state['kind'] and self.clock() < self.state['until']:
                return False, self.state['kind'], self.state['message']
            return True, None, ''

    def _trip(self, exc):
        with self.cv:
            self.state = {'kind': exc.kind, 'until': self.clock() + BACKOFF_S.get(exc.kind, 60.0),
                          'message': str(exc)}
            self.queue.clear()
            self.pending.clear()
            self.cv.notify_all()

    # -- queue ----------------------------------------------------------------------------
    def request(self, jobs, wait_s, front=False):
        """Queue the jobs not already pending / recently failed and wait up to `wait_s` for them.
        Returns how many of the requested jobs are still pending when it returns."""
        deadline = self.clock() + max(0.0, wait_s)
        with self.cv:
            keys = []
            fresh = []
            for j in jobs:
                k = (j.fid, j.hash)
                bad = self.failed.get(k)
                if bad and self.clock() - bad[0] < FAIL_TTL_S:
                    continue
                keys.append(k)
                if k not in self.pending:
                    self.pending[k] = j
                    fresh.append(j)
            if front:
                self.queue.extendleft(reversed(fresh))
            else:
                self.queue.extend(fresh)
            if fresh:
                self._ensure_thread()
                self.cv.notify_all()
            while any(k in self.pending for k in keys):
                left = deadline - self.clock()
                if left <= 0:
                    break
                self.cv.wait(left)
            return sum(1 for k in keys if k in self.pending)

    def _ensure_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name='place-mask', daemon=True)
            self.thread.start()

    def _run(self):
        while True:
            with self.cv:
                while not self.queue:
                    if not self.cv.wait(60.0) and not self.queue:
                        self.thread = None     # idle: exit (and release this thread's DB connection)
                        self._close_db()
                        return
                batch = [self.queue.popleft() for _ in range(min(self.chunk, len(self.queue)))]
            self._process(batch)

    def _close_db(self):
        close = getattr(self.db, 'close', None)
        if close is not None:
            close()

    def _process(self, batch):
        try:
            pairs, errors, models = self.embed_chunk([(j.path, j.boxes) for j in batch])
        except MaskUnavailable as exc:
            self._trip(exc)
            return
        except Exception as exc:  # hook-allow: broad -- the background thread must survive anything
            self._trip(MaskUnavailable('worker_unavailable', f'{type(exc).__name__}: {exc}'))
            return
        rows, now = [], self.clock()
        sm, dm = (models or (None, None))
        with self.cv:
            for j, p, e in zip(batch, pairs, errors):
                if p is None:
                    self.failed[(j.fid, j.hash)] = (now, e or 'no embedding returned')
                else:
                    rows.append((j.fid, j.hash, p[0], p[1], sm, dm, j.coverage))
        if rows:
            try:
                self.db.put_masked_embeddings(rows, self.recipe)
            except Exception as exc:  # hook-allow: broad -- cache write failure must not kill the thread
                print(f'[place-mask] cache write failed: {exc}', flush=True)
                with self.cv:
                    for j in batch:
                        self.failed[(j.fid, j.hash)] = (now, f'cache write failed: {exc}')
        with self.cv:
            for j in batch:
                self.pending.pop((j.fid, j.hash), None)
            self.stats['embedded'] += len(rows)
            self.stats['batches'] += 1
            self.cv.notify_all()


# ================================================================================ scoring

def _stored(db, space, ids):
    """{file_id: float32 vector} of the stored whole-image vectors of `ids` in a space."""
    ids = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
    fids, _cs, mat = getattr(db, f'get_{space}_embeddings_matrix')()
    if mat.shape[0] == 0 or len(ids) == 0:
        return {}
    idx = np.flatnonzero(np.isin(fids, ids))
    return {int(fids[i]): np.ascontiguousarray(mat[i], dtype=np.float32) for i in idx}


def _models_ok(db, entry):
    cur = db.current_space_models()
    return all(c is None or m is None or c == m for c, m in zip(cur, (entry[2], entry[3])))


def blend(stage1, masked_cos, alpha=MASK_BLEND):
    """z-match + blend. stage1 = {key: stage-1 score}, masked_cos = {key: fused masked
    similarity}. The masked scores of the scored subset are standardised and mapped onto the
    subset's stage-1 mean/std, then blended: alpha * masked + (1 - alpha) * stage 1. Keys without
    a masked score keep their stage-1 score. Returns {key: final} for every key in stage1."""
    out = dict(stage1)
    keys = [k for k in masked_cos if k in stage1]
    if len(keys) < MASK_MIN_SUBSET:
        return out
    m = np.array([masked_cos[k] for k in keys], np.float64)
    s = np.array([stage1[k] for k in keys], np.float64)
    ms = m.std()
    if ms < 1e-9:
        return out
    mapped = s.mean() + (m - m.mean()) / ms * max(s.std(), 1e-6)
    for k, mp, s1 in zip(keys, mapped, s):
        out[k] = float(alpha * mp + (1 - alpha) * s1)
    return out


def run_stage2(db, service, window, member_ids, negative_ids, stage1, data_root, budget_s=None,
               is_image=lambda path: True):
    """Re-score `window` ([(file_id, rel_path)] best first) with person-masked vectors.

    stage1 {file_id: stage-1 score}. Returns (final {file_id: score}, detail {file_id: dict},
    info dict). `final` equals `stage1` for everything that could not be re-scored. Never raises
    for worker problems: they come back in info['state'] ('worker_outdated' | 'worker_unavailable'
    | 'no_members' | 'ok' | 'partial')."""
    budget_s = MASK_BUDGET_S if budget_s is None else budget_s
    t0 = time.monotonic()
    info = {'state': 'ok', 'recipe': RECIPE, 'window': len(window), 'masked': 0, 'clean': 0,
            'covered': 0, 'unscanned': 0, 'skipped': 0, 'pending': 0, 'budget_s': budget_s}
    detail = {}
    ok, kind, msg = service.status()
    if not ok:
        info.update(state=kind, message=msg)
        return dict(stage1), detail, info
    members = list(member_ids)
    if len(members) > MASK_MEMBERS_MAX:
        members = _representative(db, members, MASK_MEMBERS_MAX)
    win = [(fid, p) for fid, p in window if is_image(p)]
    info['skipped'] = len(window) - len(win)
    ids = list(dict.fromkeys(members + [fid for fid, _ in win]))
    plans = make_plans(db, ids)
    paths = dict(win)
    paths.update({r['id']: r['path'] for r in db.get_files_by_ids(members)} if members else {})
    for fid, _p in win:
        pl = plans[fid]
        detail[fid] = {'stage2': pl.status}
        if pl.status in ('masked', 'clean', 'covered', 'unscanned'):
            info[pl.status] += 1

    def cached():
        got = db.get_masked_embeddings([f for f, p in plans.items() if p.status == 'masked'], RECIPE)
        return {k: v for k, v in got.items() if _models_ok(db, v)}

    have = cached()

    def missing(fids):
        return [Job(f, plans[f].hash, os.path.join(data_root, paths[f]), plans[f].boxes, plans[f].coverage)
                for f in fids if plans[f].status == 'masked' and (f, plans[f].hash) not in have
                and f in paths]
    member_jobs = missing(members)
    cand_jobs = missing([f for f, _ in win])
    pending = 0
    if member_jobs or cand_jobs:
        pending = service.request(member_jobs, budget_s, front=True)
        left = max(0.0, budget_s - (time.monotonic() - t0))
        pending += service.request(cand_jobs, left)
        have = cached()
        ok, kind, msg = service.status()
        if not ok:
            info.update(state=kind, message=msg)
            return dict(stage1), detail, info
    info['pending'] = sum(1 for j in member_jobs + cand_jobs if (j.fid, j.hash) not in have)
    if info['pending']:
        info['state'] = 'partial'

    # masked members' centroid per space (clean / unscanned members use their stored vector;
    # covered or still-pending members are left out)
    from media_manager.similarity import adjusted_centroid, mean_normalized_centroid
    centroids, n_members = {}, 0
    for space, idx in (('siglip', 0), ('dino', 1)):
        stored = _stored(db, space, [m for m in members if plans[m].status in ('clean', 'unscanned')])
        vecs = []
        for m in members:
            pl = plans[m]
            if pl.status == 'masked' and (m, pl.hash) in have:
                vecs.append(have[(m, pl.hash)][idx].astype(np.float32).tobytes())
            elif pl.status in ('clean', 'unscanned') and m in stored:
                vecs.append(stored[m].tobytes())
        n_members = max(n_members, len(vecs))
        if not vecs:
            continue
        neg_ids = list(negative_ids or ())[:MASK_NEGATIVES_MAX]
        neg = [v.tobytes() for v in _stored(db, space, neg_ids).values()] if neg_ids else []
        centroids[space] = adjusted_centroid(vecs, neg) if neg else mean_normalized_centroid(vecs)
    info['members_used'] = n_members
    if not centroids:
        info['state'] = 'no_members'
        return dict(stage1), detail, info

    cands = [f for f, _ in win if plans[f].status in ('masked', 'clean')]
    stored_c = {sp: _stored(db, sp, [f for f in cands if plans[f].status == 'clean']) for sp in centroids}
    cos = {}
    for fid in cands:
        pl = plans[fid]
        parts, wsum = 0.0, 0.0
        for space, c in centroids.items():
            idx = 0 if space == 'siglip' else 1
            if pl.status == 'masked':
                e = have.get((fid, pl.hash))
                v = e[idx] if e is not None else None
            else:
                v = stored_c[space].get(fid)
            if v is None:
                parts = None
                break
            parts += WEIGHTS[space] * float(np.dot(v, c))
            wsum += WEIGHTS[space]
        if parts is not None and wsum:
            cos[fid] = parts / wsum
        elif pl.status == 'masked':
            detail[fid]['stage2'] = 'pending'
    final = blend(stage1, cos)
    for fid, c in cos.items():
        detail[fid]['masked'] = round(c, 3)
    info['scored'] = len(cos)
    return final, detail, info


def _representative(db, member_ids, n):
    """The n members closest to the DINOv2 centroid of all of them (stable fallback: first n)."""
    vecs = _stored(db, 'dino', member_ids)
    if len(vecs) <= n:
        return sorted(member_ids)[:n]
    ids = sorted(vecs)
    m = np.stack([vecs[i] for i in ids])
    c = m.mean(axis=0)
    order = np.argsort(-(m @ c))
    keep = [ids[i] for i in order[:n]]
    rest = [i for i in sorted(member_ids) if i not in vecs]
    return (keep + rest)[:n]
