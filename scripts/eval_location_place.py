#!/usr/bin/env python3
"""Offline evaluation of the location-suggestion variants on YOUR library.

    python scripts/eval_location_place.py --data-root /path/with/.media [--min-members 8]
        [--holdout 0.2] [--seed 0] [--max-locations 50] [--variants siglip dino combined place]
        [--json OUT.json]

What it does, per location that has >= --min-members photos with SigLIP 2 + DINOv2 embeddings:
  1. hides --holdout (20%) of the location's photos (the "held-out" set, ground truth),
  2. builds the suggestion ranking from the remaining members exactly as the web page does
     (same z-score fusion; `place` adds location_place's people/geo corrections), over the
     whole library with the remaining members removed,
  3. scores where the held-out photos landed.
Columns of the printed table (means over evaluated locations):
  P@20      share of the top 20 suggestions that are held-out photos of the same location
            (strict: other photos genuinely taken there but not filed there count as misses,
            so read it as a relative number between variants, not an absolute accuracy)
  R@100     share of the held-out photos found in the top 100
  bait@20 / bait@100
            the "same person, different place" probe. Bait = photos filed under a DIFFERENT
            location that show an identity who is also in this location's remaining members
            (named faces / whole-photo assignments). Count of bait photos inside the top 20 /
            100 suggestions; LOWER IS BETTER. A variant that wins P@20 but also fills the top
            with bait is matching people, not the place. Locations with no recurring identity
            have no probe and are left out of the bait columns (n_probe says how many had one).
Variants are clip-free: siglip, dino, combined (0.6 DINOv2 + 0.4 SigLIP 2) and place (combined +
people-aware correction). The stage-2 masked re-embedding needs the GPU worker and is not
evaluated here.

It opens media.db / manual.db the same way the web app does (same migrations); it never
writes embeddings or labels. Prefer running it on a copy or while the server is stopped.
Needs the SigLIP 2 + DINOv2 embeddings and, for `place` to differ from `combined`, the face
scan / object index (the report prints how many photos are person-scanned).
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from media_manager import location_place as lp  # noqa: E402

VARIANTS = {
    'siglip': {'siglip': 1.0},
    'dino': {'dino': 1.0},
    'combined': None,        # default fusion weights
    'place': None,
}


def _rank_variant(db, manual, variant, train_ids, train_cs, location):
    weights = VARIANTS[variant]
    fids, cks, rank, display, _cov = lp.fused_rank_numpy(db, train_ids, None, weights)
    if rank.shape[0] == 0:
        return fids, rank
    if variant == 'place':
        rank, _why, _info = lp.apply_place_correction(db, manual, rank, display, fids, cks,
                                                      train_ids, train_cs, location=location)
    rank = np.array(rank, dtype=np.float32, copy=True)
    rank[np.isin(fids, np.asarray(train_ids, dtype=np.int64))] = -np.inf     # members are never suggested
    return fids, rank


def _top(fids, rank, k):
    k = min(k, rank.shape[0])
    if k == 0:
        return []
    idx = np.argpartition(-rank, k - 1)[:k]
    idx = idx[np.argsort(-rank[idx], kind='stable')]
    return [int(fids[i]) for i in idx if np.isfinite(rank[i])]


def _bait_ids(db, manual, loc_id, train_cs, other_members):
    """File ids filed under other locations that show an identity recurring in `train_cs`."""
    prof = lp.identity_profile(train_cs, lp._names(manual, train_cs))
    if not prof:
        return set()
    cs_list = list(other_members)
    names = lp._names(manual, cs_list)
    hit = [cs for cs in cs_list if any(n in prof for n in names.get(cs, ()))]
    return {int(r['id']) for r in db.get_files_by_checksums(hit)}


def evaluate(db, manual, variants=('siglip', 'dino', 'combined', 'place'), min_members=8, holdout=0.2,
             seed=0, max_locations=None, k_prec=20, k_rec=100, log=None):
    """{'per_location': [...], 'summary': {variant: {...}}, 'scan': {...}}. Deterministic for a seed."""
    rng = random.Random(seed)
    embedded = set()
    for space in ('siglip', 'dino'):
        ids, _cs, _m = getattr(db, f'get_{space}_embeddings_matrix')()
        embedded = set(ids.tolist()) if not embedded else embedded & set(ids.tolist())
    members = {}                      # location id -> {checksum: file id}
    for row in manual.list_locations():
        cs = manual.get_checksums_for_location(row['id'])
        rows = [r for r in db.get_files_by_checksums(list(cs)) if int(r['id']) in embedded]
        if len(rows) >= min_members:
            members[row['id']] = {r['checksum']: int(r['id']) for r in rows}
    loc_ids = sorted(members)
    if max_locations and len(loc_ids) > max_locations:
        loc_ids = sorted(rng.sample(loc_ids, max_locations))
    all_members = {r['id']: set(manual.get_checksums_for_location(r['id'])) for r in manual.list_locations()}
    per = []
    for lid in loc_ids:
        cs_ids = sorted(members[lid].items())
        rng.shuffle(cs_ids)
        n_hold = max(1, int(round(len(cs_ids) * holdout)))
        held, train = cs_ids[:n_hold], cs_ids[n_hold:]
        train_cs = [c for c, _ in train]
        train_ids = [i for _, i in train]
        held_ids = {i for _, i in held}
        location = manual.get_location(lid)
        others = set().union(*[all_members[o] for o in all_members if o != lid]) - all_members[lid]
        bait = _bait_ids(db, manual, lid, train_cs, others)
        entry = {'location': lid, 'name': location['name'], 'members': len(cs_ids),
                 'held_out': len(held_ids), 'bait_pool': len(bait), 'variants': {}}
        for v in variants:
            fids, rank = _rank_variant(db, manual, v, train_ids, train_cs, location)
            top = _top(fids, rank, max(k_prec, k_rec))
            t20, t100 = top[:k_prec], top[:k_rec]
            entry['variants'][v] = {
                'p_at_20': len(held_ids & set(t20)) / float(k_prec),
                'r_at_100': len(held_ids & set(t100)) / float(len(held_ids)),
                'bait_at_20': len(bait & set(t20)) if bait else None,
                'bait_at_100': len(bait & set(t100)) if bait else None,
            }
        per.append(entry)
        if log:
            log(f"  {entry['name'][:30]:30s} n={entry['members']:4d} bait_pool={len(bait):4d} " + ' '.join(
                f"{v}:P{entry['variants'][v]['p_at_20']:.2f}" for v in variants))
    summary = {}
    for v in variants:
        def mean(key, probe=False):
            vals = [e['variants'][v][key] for e in per if e['variants'][v][key] is not None]
            return float(np.mean(vals)) if vals else None
        summary[v] = {'locations': len(per), 'p_at_20': mean('p_at_20'), 'r_at_100': mean('r_at_100'),
                      'bait_at_20': mean('bait_at_20'), 'bait_at_100': mean('bait_at_100'),
                      'n_probe': sum(1 for e in per if e['variants'][v]['bait_at_20'] is not None)}
    store = lp.store_for(db).get(db)
    scan = {'library_person_scanned': int(store.scanned.sum()), 'library_files': int(store.total)}
    return {'per_location': per, 'summary': summary, 'scan': scan}


def format_table(result, variants):
    f = lambda x, p=3: '   -  ' if x is None else f'{x:.{p}f}'
    lines = [f"{'variant':10s} {'locs':>5s} {'P@20':>7s} {'R@100':>7s} {'bait@20':>8s} {'bait@100':>9s} {'n_probe':>8s}"]
    for v in variants:
        s = result['summary'][v]
        lines.append(f"{v:10s} {s['locations']:5d} {f(s['p_at_20']):>7s} {f(s['r_at_100']):>7s} "
                     f"{f(s['bait_at_20'], 2):>8s} {f(s['bait_at_100'], 2):>9s} {s['n_probe']:8d}")
    sc = result['scan']
    lines.append(f"person-scanned photos: {sc['library_person_scanned']} of {sc['library_files']} "
                 f"({100.0 * sc['library_person_scanned'] / max(sc['library_files'], 1):.0f}%)")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default='.', help='directory containing .media/ (default: cwd)')
    ap.add_argument('--min-members', type=int, default=8)
    ap.add_argument('--holdout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-locations', type=int, default=None)
    ap.add_argument('--variants', nargs='+', default=['siglip', 'dino', 'combined', 'place'],
                    choices=sorted(VARIANTS))
    ap.add_argument('--json', default=None, help='also write the full result here')
    ap.add_argument('-q', '--quiet', action='store_true', help='no per-location lines')
    a = ap.parse_args(argv)
    from media_manager.database import Database
    from media_manager.manual_db import ManualDB
    media = os.path.join(os.path.abspath(a.data_root), '.media')
    if not os.path.isfile(os.path.join(media, 'media.db')):
        ap.error(f'no media.db under {media}')
    db = Database(os.path.join(media, 'media.db'))
    manual = ManualDB(os.path.join(media, 'manual.db'))
    res = evaluate(db, manual, tuple(a.variants), a.min_members, a.holdout, a.seed, a.max_locations,
                   log=None if a.quiet else print)
    print()
    print(format_table(res, a.variants))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
