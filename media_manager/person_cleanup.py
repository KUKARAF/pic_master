"""Anchor-based person cleanup: pure helpers (no DB, no web).

The idea: the user picks one or more clear, definitely-correct faces of a person (the
ANCHORS); every other face assigned to that person is scored by cosine similarity to
the anchors' normalized centroid and put in a band:

    KEEP       score >= keep_threshold     (close to the anchor)
    REMOVE     score <  remove_threshold   (far from the anchor; bulk-removable)
    UNCERTAIN  in between                  (reviewed one by one, tinder style)

Only faces that are provably machine-assigned (see `eligibility`) may ever land in
REMOVE or UNCERTAIN; a face a human set is never a candidate at all.
"""
import numpy as np

# ArcFace (buffalo_l, 512-d, L2-normalized) cosine: genuine same-person pairs mostly sit
# >= 0.45 even across pose/lighting, impostor pairs cluster around 0.0-0.2. 0.25 / 0.45
# leave a deliberately wide uncertain band so the bulk step only ever removes faces that
# are clearly a different person.
DEFAULT_KEEP = 0.45
DEFAULT_REMOVE = 0.25

HIST_LO, HIST_HI, HIST_STEP = -0.2, 1.0, 0.05


def clamp_thresholds(keep, remove):
    """Valid (keep, remove) with remove <= keep, both within [-1, 1]."""
    keep = max(-1.0, min(1.0, float(keep)))
    remove = max(-1.0, min(1.0, float(remove)))
    return keep, min(remove, keep)


def band_for(score, keep, remove):
    if score >= keep:
        return 'keep'
    if score < remove:
        return 'remove'
    return 'uncertain'


def eligibility(row):
    """Who may the cleanup touch?  Returns 'auto', 'legacy' or 'protected'.

    The `faces.origin` column records how a person-assignment came to be:
      'auto'   -- promoted by the auto-matcher (find_matching_identity), no human involved
      'human'  -- a human named/confirmed/assigned it
      NULL     -- written before the column existed: origin unknown ('legacy')
    Hand-drawn boxes (bbox_source != 'auto'), faces without a media.db source to return
    to, and anything that carries a human fingerprint (reviewed_ok, favorite, picked as
    profile picture) are 'protected' regardless. Legacy rows are only candidates when
    the caller explicitly opts in (and never when they have a human fingerprint or a
    manual rotation)."""
    keys = row.keys()
    if (row['bbox_source'] != 'auto' or row['source_face_id'] is None
            or row['rejected'] or row['identity'] is None):
        return 'protected'
    if row['reviewed_ok'] or row['favorite'] or ('profile_at' in keys and row['profile_at']):
        return 'protected'
    origin = row['origin'] if 'origin' in keys else None
    if origin == 'auto':
        return 'auto'
    if origin is None and not (row['rotation'] if 'rotation' in keys else 0):
        return 'legacy'
    return 'protected'


def quality_score(det_score, width, height, angle_deg):
    """0..1 'how clearly visible is this face' from what the DB already has: the
    detector's confidence, the face box size in pixels (bigger = more detail), and how
    upright it is (|roll| near 0). Missing pieces fall back to neutral values."""
    det = 0.5 if det_score is None else max(0.0, min(1.0, float(det_score)))
    side = max(0.0, min(float(width or 0), float(height or 0)))
    size = min(1.0, side / 200.0)
    upright = max(0.0, 1.0 - abs(float(angle_deg or 0.0)) / 45.0)
    return 0.45 * det + 0.35 * size + 0.20 * upright


def unit_centroid(vectors):
    """Normalized mean of unit vectors; None when the mean degenerates to ~0."""
    c = np.mean(np.stack(vectors), axis=0)
    n = float(np.linalg.norm(c))
    if n < 1e-6:
        return None
    return (c / n).astype(np.float32)


def histogram(scores, lo=HIST_LO, hi=HIST_HI, step=HIST_STEP):
    """[{lo, hi, count}] with fixed-width bins; out-of-range scores clip into the end bins."""
    n = int(round((hi - lo) / step))
    counts = [0] * n
    for s in scores:
        i = int((s - lo) // step)
        counts[max(0, min(n - 1, i))] += 1
    return [{'lo': round(lo + i * step, 4), 'hi': round(lo + (i + 1) * step, 4), 'count': c}
            for i, c in enumerate(counts)]


def parse_embedding(blob, dim=None):
    """float32 unit vector from a stored blob, or None when unusable (empty, bad length,
    wrong width, non-finite, zero norm)."""
    if not blob or len(blob) % 4:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and v.shape[0] != dim:
        return None
    if not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return (v / n).astype(np.float32)
