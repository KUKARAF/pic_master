"""Near-duplicate grouping + pre-classification (Phase 2).

Pure logic — no DB access. The web layer assembles `items` from the phashes/files
tables and hands them here; this module clusters them into candidate near-duplicate
groups and pre-labels each group into one of the cases the human then confirms.

An `item` is a dict with:
    file_id       int
    phash         int   (64-bit; from the stored 8-byte BLOB)
    width,height  int|None   (ORIGINAL pixels — picks the keeper)
    size          int|None   (bytes — keeper tiebreak)
    taken_at      int|None   (unix seconds; separates a burst from a copy)
    broken        bool       (damaged)
    is_still      bool       (a captured video still, not a standalone photo)
    parent        str|None   (source-video checksum, when is_still)

Grouping uses banded (multi-index) bucketing over the 64-bit pHash so it's ~linear
instead of O(n^2), then verifies each candidate pair by full Hamming distance and
unions them. Two sampled stills of the *same* video are never grouped (they're
intentional samples, not duplicates).
"""

# Tunable thresholds (64-bit Hamming). Conservative on purpose — a human confirms.
H_DUP = 4        # <= this: the same pixels (re-encode/resize/damage)
H_NEAR = 10      # <= this: near — same scene / burst / possible dup
BURST_SECS = 3   # capture-time gap under which near frames read as a burst, not a copy
RATIO_DIFF = 1.2 # resolution/size ratio above which two copies "differ" in quality
CROSS_THRESH = 0.92  # CLIP cosine for the screenshot↔video-still wide net (Phase 3)

_BANDS = 4
_BAND_BITS = 16
_BAND_MASK = (1 << _BAND_BITS) - 1


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())


def _is_degenerate(phash: int) -> bool:
    """All-zero / all-one hashes come from solid-colour or featureless frames and would
    form a giant false hub — skip them from grouping."""
    return phash == 0 or phash == 0xFFFFFFFFFFFFFFFF


def group(items, max_hamming=H_NEAR, blocked_pairs=None):
    """Cluster `items` into candidate near-dup groups (lists of file_ids, size >= 2).

    Banded bucketing over the pHash yields candidate pairs sharing any 16-bit band;
    each is verified by full Hamming <= max_hamming before union. Same-parent stills
    are never unioned, and pairs the user marked "not a duplicate" (blocked_pairs, a set
    of frozenset({checksum_a, checksum_b})) are skipped so they never regroup. Singletons
    are dropped."""
    blocked_pairs = blocked_pairs or set()
    by_id = {it['file_id']: it for it in items}
    uf = _UnionFind()
    for it in items:                        # every id is a node so singletons resolve
        uf.find(it['file_id'])

    buckets = [dict() for _ in range(_BANDS)]
    for it in items:
        ph = it['phash']
        if _is_degenerate(ph):
            continue
        for bi in range(_BANDS):
            key = (ph >> (bi * _BAND_BITS)) & _BAND_MASK
            buckets[bi].setdefault(key, []).append(it['file_id'])

    seen_pairs = set()
    for band in buckets:
        for members in band.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    pair = (a, b) if a < b else (b, a)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    ia, ib = by_id[a], by_id[b]
                    # never treat two sampled frames of the same video as duplicates
                    if ia['is_still'] and ib['is_still'] and ia.get('parent') \
                       and ia.get('parent') == ib.get('parent'):
                        continue
                    if blocked_pairs and frozenset((ia.get('checksum'), ib.get('checksum'))) in blocked_pairs:
                        continue  # user marked these "not a duplicate"
                    if hamming(ia['phash'], ib['phash']) <= max_hamming:
                        uf.union(a, b)

    return [g for g in uf.groups() if len(g) >= 2]


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


def _max_pair_hamming(group_items):
    hs = [it['phash'] for it in group_items]
    worst = 0
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            worst = max(worst, hamming(hs[i], hs[j]))
    return worst


def _resolution_or_size_varies(group_items):
    mps = [_megapixels(it) for it in group_items if _megapixels(it)]
    if len(mps) >= 2 and max(mps) >= min(mps) * RATIO_DIFF:
        return True
    sizes = [it.get('size') for it in group_items if it.get('size')]
    return len(sizes) >= 2 and max(sizes) >= min(sizes) * RATIO_DIFF


def _time_spread(group_items):
    ts = [it['taken_at'] for it in group_items if it.get('taken_at')]
    return (max(ts) - min(ts)) if len(ts) == len(group_items) and len(ts) >= 2 else None


def classify(group_items):
    """Pre-label a group. Returns {label, action, keeper, reason}.

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

    worst = _max_pair_hamming(group_items)
    spread = _time_spread(group_items)

    if worst <= H_DUP and _resolution_or_size_varies(group_items):
        return {'label': 'lower_quality_copy', 'action': 'merge', 'keeper': _pick_keeper(group_items),
                'reason': 'same image at different resolution/size'}

    if spread is not None and spread <= BURST_SECS and not _resolution_or_size_varies(group_items):
        return {'label': 'burst', 'action': 'keep', 'keeper': None,
                'reason': f'shots taken within {spread}s of each other'}

    return {'label': 'review', 'action': 'review', 'keeper': _pick_keeper(group_items),
            'reason': 'looks similar but the case is ambiguous'}
