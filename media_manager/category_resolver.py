"""Resolves a file's effective category: a manual.db override (including an
explicit "no category" decision) always wins over a media.db ML auto-match —
mirrors the negated-tag/promoted-face precedent pattern (see manual_db.py's
get_negated_labels / promote_auto_face docstrings). Lives in its own module,
not web.py and not media.py, since both the CLI (media.py's
MediaManager.match_categories, to skip manually-decided files) and web.py need
this, and media.py must not import web.py."""


def resolve_category_for_file(manual, db, file_id, checksum):
    """Single-file resolution for the photo page.
    Returns {'name': str|None, 'source': 'manual'|'auto'|None, 'score': float|None}."""
    override = manual.get_category_override(checksum)
    if override is not None:
        return {'name': override['name'], 'source': 'manual', 'score': None}
    match = db.get_file_category_match(file_id)
    if match is not None:
        return {'name': match['category_name'], 'source': 'auto', 'score': match['score']}
    return {'name': None, 'source': None, 'score': None}


def resolve_categories_for_checksums(manual, db, file_checksum_pairs):
    """Batched resolution for grid views (mirrors _enrich_rows's batching idiom).
    file_checksum_pairs: [(file_id, checksum), ...].
    Returns {checksum: {'name','source','score'}} — a checksum resolving to
    "explicitly uncategorized" or "no decision + no auto match" is simply absent
    from the result; callers should treat a missing key as uncategorized."""
    if not file_checksum_pairs:
        return {}
    checksums = [c for _fid, c in file_checksum_pairs]
    overrides = manual.get_category_overrides_for_checksums(checksums)

    result = {}
    unresolved_file_ids = []
    for file_id, checksum in file_checksum_pairs:
        if checksum in overrides:
            name = overrides[checksum]['name']
            if name is not None:
                result[checksum] = {'name': name, 'source': 'manual', 'score': None}
            # else: explicit "no category" — leave absent from result, like uncategorized
        else:
            unresolved_file_ids.append((file_id, checksum))

    if unresolved_file_ids:
        matches = db.get_file_category_matches_for_files([fid for fid, _cs in unresolved_file_ids])
        for file_id, checksum in unresolved_file_ids:
            match = matches.get(file_id)
            if match is not None:
                category_name, score, _model = match
                result[checksum] = {'name': category_name, 'source': 'auto', 'score': score}
    return result


def get_resolved_checksums_for_category(manual, db, category_id, category_name, limit=500):
    """Checksums currently resolving to this category: manual assignments plus
    unsuppressed auto-matches (i.e. excluding any checksum with an override
    pointing elsewhere, or an explicit 'no category' override)."""
    manual_checksums = set(manual.get_example_checksums_for_category(category_id, limit=limit))

    auto_checksums = set()
    for _file_id, checksum, name, _score in db.get_all_file_category_matches():
        if name == category_name:
            auto_checksums.add(checksum)
    auto_checksums -= manual_checksums

    if auto_checksums:
        overridden = set(manual.get_category_overrides_for_checksums(list(auto_checksums)).keys())
        auto_checksums -= overridden

    return list(manual_checksums | auto_checksums)[:limit]


def get_category_counts(manual, db):
    """{category_name: resolved_count} across the whole library — manual
    assignments plus unsuppressed auto-matches. Used for the navbar dropdown."""
    counts = {}
    for row in manual.list_categories():
        counts[row['name']] = row['image_count']

    auto_matches = db.get_all_file_category_matches()
    if not auto_matches:
        return counts

    # Any checksum with a manual decision at all — assigned elsewhere, or an
    # explicit "no category" — must not also be counted via its (stale/
    # superseded) auto-match row.
    overrides = manual.get_category_overrides_for_checksums([checksum for _fid, checksum, _n, _s in auto_matches])

    for _file_id, checksum, name, _score in auto_matches:
        if checksum in overrides:
            continue
        counts[name] = counts.get(name, 0) + 1

    return counts
