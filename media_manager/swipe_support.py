"""Shared helpers for every swipe-based suggestion stream (faces, sets,
categories, tags) — the generic "buffer/exclude/bias" mechanics factored out of
the original face-suggestion implementation so each feature's own ranking
function doesn't reimplement this part. Ranking/scoring itself stays per-feature
(faces score against named identities, sets/categories/tags score against a
CLIP centroid via similarity.py) — this module only handles what's identical
across all of them."""


def parse_exclude(exclude_str):
    """Parse a comma-joined 'exclude' query param into a set of strings. Empty
    entries (from a leading/trailing/doubled comma, or an empty string) are
    dropped rather than producing a stray ''."""
    if not exclude_str:
        return set()
    return {r for r in exclude_str.split(',') if r}


def bias_reorder(candidates, key_fn, bias_key, bias_action):
    """Reorder a list of (already scored) candidates so the most recent
    confirm/reject decision steers what's suggested next — the same "sort tweak,
    not a new ranking model" idea the face-suggestion feature used first.

    candidates: any list; key_fn(candidate) extracts the value to compare
        against bias_key (e.g. an identity name, a file id).
    bias_key: the confirmed/rejected item's key, or None/falsy for no bias.
    bias_action: 'confirm' (push matching-key candidates to the front) or
        'reject' (push them to the back). Any other value, or a missing
        bias_key, leaves the input order untouched (caller is expected to have
        already sorted by score descending).

    A feature with no meaningful bias dimension (e.g. Sets and Categories, where
    every candidate already scores against the one same centroid) simply never
    calls this with a bias_key — it's optional per feature, not mandatory.
    """
    if not bias_key or bias_action not in ('confirm', 'reject'):
        return candidates
    matches_first = bias_action == 'confirm'
    return sorted(
        candidates,
        key=lambda c: 0 if (key_fn(c) == bias_key) == matches_first else 1
    )
