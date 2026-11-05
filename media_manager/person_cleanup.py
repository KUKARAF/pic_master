"""Anchor-based person cleanup: pure helpers (no DB, no web).

The idea: the user picks one or more clear, definitely-correct faces of a person (the
ANCHORS); every other face assigned to that person is scored by cosine similarity to
the anchors' normalized centroid and put in a band:

    KEEP       score >= keep_threshold     (close to the anchor)
    REMOVE     score <  remove_threshold   (far from the anchor; bulk-removable)
    UNCERTAIN  in between                  (reviewed one by one, tinder style)

ANCHORS are the person's trust-3 (hand-set) faces. Only automatic faces (trust 1) may
ever land in REMOVE; swipe-approved faces (trust 2) are reviewed one by one on opt-in
and never bulk-removed; a face a human set (trust 3) is never a candidate at all.
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


# ---- Trust hierarchy for named faces (`faces.trust`, kept consistent with `faces.origin`) ----
#   3 MANUAL / ANCHOR  set by hand: typed/picked name, hand-drawn box, picked as cleanup
#                      anchor, favorite, profile picture.        origin='human'
#   2 APPROVED         approved through swipe-style review (face/identity swipe, cleanup
#                      swipe 'keep', bulk confirm-above-threshold). origin='human'
#   1 AUTO             written by the auto-matcher (find_matching_identity) or a legacy row
#                      whose origin was never recorded.            origin='auto'
#   0 UNKNOWN          unnamed / not yet assigned (never a reference or a candidate).
# Only a user action raises or lowers a face's trust; nothing demotes automatically.
TRUST_UNKNOWN, TRUST_AUTO, TRUST_APPROVED, TRUST_MANUAL = 0, 1, 2, 3
TRUST_LABELS = {3: 'anchor', 2: 'approved', 1: 'auto', 0: 'unknown'}
TRUST_TITLES = {
    3: 'Set by hand: always an anchor, never touched by cleanup',
    2: 'Approved by swipe: protected from automatic removal',
    1: 'Automatic match (older faces without a manual mark count as automatic)',
    0: 'Unknown',
}
# Matcher reference weights (rank among references, see ManualDB.get_trusted_face_matrix).
TRUST_MATCH_WEIGHTS = {TRUST_MANUAL: 1.0, TRUST_APPROVED: 0.7}


def derive_trust(origin, reviewed_ok=0, favorite=0, profile_at=None, bbox_source='auto', identity='x'):
    """Trust implied by the pre-trust columns. The one-time migration in manual_db mirrors
    this in SQL (tests keep the two in step); also the fallback for rows lacking a value."""
    if identity is None:
        return TRUST_UNKNOWN
    if favorite or profile_at or bbox_source != 'auto':
        return TRUST_MANUAL
    if reviewed_ok:
        return TRUST_APPROVED
    if origin == 'human':
        return TRUST_MANUAL
    return TRUST_AUTO


def face_trust(row):
    """THE place a face row's trust tier is read: the stored `trust` when set, otherwise
    derived from the older columns."""
    keys = row.keys()
    t = row['trust'] if 'trust' in keys else None
    if t:
        return int(t)
    g = lambda k, d=None: row[k] if k in keys else d
    return derive_trust(g('origin'), g('reviewed_ok', 0), g('favorite', 0), g('profile_at'),
                        g('bbox_source', 'auto'), g('identity', 'x'))


def eligibility(row):
    """Who may the cleanup touch?  Returns 'auto', 'approved' or 'protected'.

      'auto'       trust 1: removable (bulk or swipe).
      'approved'   trust 2: never auto-removed; only reviewed one by one when the user
                   opts in (swipe queue), see ManualDB.cleanup_remove_faces.
      'protected'  trust 3 (anything set by hand), plus rows with no media.db source to
                   return to, hand-drawn boxes, rejected / unassigned rows.
    Older faces whose origin was never recorded were migrated to trust 1."""
    if (row['bbox_source'] != 'auto' or row['source_face_id'] is None
            or row['rejected'] or row['identity'] is None):
        return 'protected'
    t = face_trust(row)
    if t >= TRUST_MANUAL:
        return 'protected'
    if t == TRUST_APPROVED:
        return 'approved'
    return 'auto'


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
