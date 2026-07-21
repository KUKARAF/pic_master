"""Resolves a file's effective categories: manual.db's manual category
assignments (file_categories) union media.db's ML auto-matches
(file_category_matches), except an auto-match for a category the file has been
explicitly rejected for (manual.db's file_category_exclusions — the category
suggestion swipe stream's reject action, one row per (checksum, category_id)
pair, mirroring file_set_exclusions) is suppressed. Categories are
multi-valued: a file can carry any number of manual assignments and
independently clear threshold for any number of auto-matches at once, so
every function here returns a *list* of categories per file, not a single
best one. Lives in its own module, not web.py and not media.py, since both
the CLI (media.py's MediaManager.match_categories, to skip files already
excluded) and web.py need this, and media.py must not import web.py."""


def resolve_categories_for_file(manual, db, file_id, checksum):
    """Single-file resolution for the photo page.
    Returns [{'id': int|None, 'name': str, 'source': 'manual'|'auto', 'score': float|None}, ...].
    `id` is always present for a manual entry (needed to remove it); an auto
    entry's `id` is looked up too (cheap here — single-file, not batched)."""
    manual_categories = manual.get_categories_for_checksum(checksum)
    result = [{'id': c['id'], 'name': c['name'], 'source': 'manual', 'score': None} for c in manual_categories]
    seen_names = {c['name'] for c in result}
    excluded_category_ids = manual.get_category_exclusions_for_checksums([checksum]).get(checksum, set())
    for category_name, score, _model in db.get_file_category_matches(file_id):
        if category_name in seen_names:
            continue  # already has this one manually — don't list it twice
        cat_row = manual.find_category(category_name)
        cat_id = cat_row['id'] if cat_row else None
        if cat_id is not None and cat_id in excluded_category_ids:
            continue  # explicitly rejected for this specific category
        result.append({'id': cat_id, 'name': category_name, 'source': 'auto', 'score': score})
    return result


def resolve_categories_for_checksums(manual, db, file_checksum_pairs):
    """Batched resolution for grid views (mirrors _enrich_rows's batching idiom).
    file_checksum_pairs: [(file_id, checksum), ...].
    Returns {checksum: [{'name','source','score'}, ...]} — a checksum with no
    categories at all is simply absent from the result; callers should treat a
    missing key as an empty list."""
    if not file_checksum_pairs:
        return {}
    checksums = [c for _fid, c in file_checksum_pairs]
    manual_map = manual.get_categories_for_checksums(checksums)
    exclusions_map = manual.get_category_exclusions_for_checksums(checksums)

    result = {}
    for checksum, categories in manual_map.items():
        result[checksum] = [{'name': c['name'], 'source': 'manual', 'score': None} for c in categories]

    matches_by_file = db.get_file_category_matches_for_files([fid for fid, _cs in file_checksum_pairs])
    # category_name -> id, resolved once for the whole batch rather than per row.
    name_to_id = {row['name']: row['id'] for row in manual.list_categories()}
    for file_id, checksum in file_checksum_pairs:
        matches = matches_by_file.get(file_id)
        if not matches:
            continue
        existing = result.setdefault(checksum, [])
        seen_names = {c['name'] for c in existing}
        excluded_ids = exclusions_map.get(checksum, set())
        for category_name, score, _model in matches:
            if category_name in seen_names:
                continue
            if name_to_id.get(category_name) in excluded_ids:
                continue
            existing.append({'name': category_name, 'source': 'auto', 'score': score})

    return {checksum: cats for checksum, cats in result.items() if cats}


def get_resolved_checksums_for_category(manual, db, category_id, category_name, limit=500):
    """Checksums currently resolving to this category: manual assignments plus
    unsuppressed auto-matches (i.e. excluding checksums explicitly rejected for
    this specific category)."""
    manual_checksums = set(manual.get_example_checksums_for_category(category_id, limit=limit))

    auto_checksums = set()
    for _file_id, checksum, name, _score in db.get_all_file_category_matches():
        if name == category_name:
            auto_checksums.add(checksum)
    auto_checksums -= manual_checksums

    if auto_checksums:
        auto_checksums -= manual.get_excluded_checksums_for_category(category_id)

    return list(manual_checksums | auto_checksums)[:limit]


def get_category_counts(manual, db):
    """{category_name: resolved_count} across the whole library — manual
    assignments plus unsuppressed auto-matches. Used for the navbar dropdown.
    A file with both a manual assignment and an auto-match for the *same*
    category only counts once."""
    counts = {}
    manual_by_category = {}
    for row in manual.list_categories():
        counts[row['name']] = row['image_count']
        manual_by_category[row['name']] = set(manual.get_example_checksums_for_category(row['id'], limit=100000))

    auto_matches = db.get_all_file_category_matches()
    if not auto_matches:
        return counts

    excluded_by_category = {
        row['name']: manual.get_excluded_checksums_for_category(row['id'])
        for row in manual.list_categories()
    }

    for _file_id, checksum, name, _score in auto_matches:
        if checksum in manual_by_category.get(name, ()):
            continue
        if checksum in excluded_by_category.get(name, ()):
            continue
        counts[name] = counts.get(name, 0) + 1

    return counts
