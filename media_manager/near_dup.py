"""Near-duplicate keeper-selection + pre-classification (Phase 2).

Pure logic — no DB access. Grouping is done by the external **czkawka** CLI (see
`czkawka.py`); the web layer resolves each czkawka group to `items` and hands them
here to pick a keeper and pre-label the group into one of the cases the human confirms.

An `item` is a dict with:
    file_id       int
    width,height  int|None   (ORIGINAL pixels — picks the keeper)
    size          int|None   (bytes — keeper tiebreak)
    taken_at      int|None   (unix seconds; separates a burst from a copy)
    broken        bool       (damaged)
    is_still      bool       (a captured video still, not a standalone photo)
    parent        str|None   (source-video checksum, when is_still)
"""

# Tunable thresholds. Conservative on purpose — a human confirms.
H_DUP = 4        # czkawka image `difference` <= this reads as "the same pixels"
BURST_SECS = 3   # capture-time gap under which near frames read as a burst, not a copy
RATIO_DIFF = 1.2 # resolution/size ratio above which two copies "differ" in quality
MAX_GROUP = 12   # drop any czkawka group bigger than this as noise rather than present it


def _megapixels(it):
    w, h = it.get('width'), it.get('height')
    return (w * h) if (w and h) else 0


def _pick_keeper(group_items):
    """Highest resolution, then largest bytes, then not-broken, then lowest id."""
    return sorted(
        group_items,
        key=lambda it: (_megapixels(it), it.get('size') or 0, 0 if it['broken'] else 1,
                        -it['file_id']),
        reverse=True,
    )[0]['file_id']


def _resolution_or_size_varies(group_items):
    mps = [_megapixels(it) for it in group_items if _megapixels(it)]
    if len(mps) >= 2 and max(mps) >= min(mps) * RATIO_DIFF:
        return True
    sizes = [it.get('size') for it in group_items if it.get('size')]
    return len(sizes) >= 2 and max(sizes) >= min(sizes) * RATIO_DIFF


def _time_spread(group_items):
    ts = [it['taken_at'] for it in group_items if it.get('taken_at')]
    return (max(ts) - min(ts)) if len(ts) == len(group_items) and len(ts) >= 2 else None


def classify(group_items, worst_diff=None):
    """Pre-label a group. Returns {label, action, keeper, reason}.

    `worst_diff` is the largest pairwise dissimilarity within the group (czkawka's
    image `difference`, 0 = identical). It replaces the old pHash `_max_pair_hamming`
    as the "same pixels" signal for the lower_quality_copy gate; None disables that gate.

    label/action:
      damaged_twin      -> merge  (keep the healthy, highest-res copy)
      lower_quality_copy-> merge  (keep the highest-res copy)
      screenshot        -> link   (a still of a video + a standalone image; keep both)
      burst             -> keep   (shots seconds apart; keep all)
      review            -> review (uncertain; human decides)
    """
    stills = [it for it in group_items if it['is_still']]
    photos = [it for it in group_items if not it['is_still']]

    # A captured video still together with a standalone image => screenshot of a video.
    if stills and photos:
        return {'label': 'screenshot', 'action': 'link', 'keeper': None,
                'reason': 'a video frame matches a standalone image'}

    # All captured stills (different videos) — not a user-facing dup; leave for review.
    if stills and not photos:
        return {'label': 'review', 'action': 'review', 'keeper': _pick_keeper(group_items),
                'reason': 'captured video frames matched across videos'}

    broken = [it for it in group_items if it['broken']]
    healthy = [it for it in group_items if not it['broken']]
    if broken and healthy:
        return {'label': 'damaged_twin', 'action': 'merge', 'keeper': _pick_keeper(healthy),
                'reason': 'one copy is damaged, another is intact'}

    worst = worst_diff  # czkawka difference (0 = identical); None → skip the same-pixels gate
    spread = _time_spread(group_items)

    if worst is not None and worst <= H_DUP and _resolution_or_size_varies(group_items):
        return {'label': 'lower_quality_copy', 'action': 'merge', 'keeper': _pick_keeper(group_items),
                'reason': 'same image at different resolution/size'}

    if spread is not None and spread <= BURST_SECS and not _resolution_or_size_varies(group_items):
        return {'label': 'burst', 'action': 'keep', 'keeper': None,
                'reason': f'shots taken within {spread}s of each other'}

    return {'label': 'review', 'action': 'review', 'keeper': _pick_keeper(group_items),
            'reason': 'looks similar but the case is ambiguous'}
