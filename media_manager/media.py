#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import os
from media_manager import MediaManager

def _ensure_self_signed_cert(media_dir, host):
    """Return (certfile, keyfile), generating a self-signed cert via the
    system openssl binary if one isn't already cached in .media/.

    Returns None if the openssl binary is unavailable, so the caller can
    fall back to plain HTTP rather than failing to start."""
    certfile = os.path.join(media_dir, 'https_cert.pem')
    keyfile = os.path.join(media_dir, 'https_key.pem')
    if os.path.exists(certfile) and os.path.exists(keyfile):
        return certfile, keyfile

    if not shutil.which('openssl'):
        print("WARNING: HTTPS requires the 'openssl' command-line tool, which "
              "was not found on PATH. Falling back to plain HTTP. "
              "Install openssl or pass --http to silence this warning.",
              file=sys.stderr)
        return None

    print("Generating self-signed HTTPS certificate (first run)...")
    subj = f'/CN={host}'
    cmd = [
        'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
        '-keyout', keyfile, '-out', certfile,
        '-days', '3650', '-nodes', '-subj', subj,
        '-addext', f'subjectAltName=DNS:{host},IP:{host}' if host.replace('.', '').isdigit()
                   else f'subjectAltName=DNS:{host}',
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.chmod(keyfile, 0o600)
    return certfile, keyfile


def main():
    parser = argparse.ArgumentParser(description='Media manager - like git for your media files')
    parser.add_argument('--strict', action='store_true', default=True,
                        help='Exit with error if repo is not initialised (default: true)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # media init - create .media folder
    init = sub.add_parser('init', help='Initialize a media repository')

    # media add <path> - scan directory
    add = sub.add_parser('add', help='Scan a directory for media files')
    add.add_argument('path', help='Directory to scan')
    add.add_argument('-r', '--recursive', action='store_true', default=True,
                     help='Scan recursively')
    add.add_argument('--reindex', action='store_true',
                     help='Force reprocessing (clear detections/faces/embedding) for already-tracked files found in this scan')

    # media hash <path> - show stored hash
    hash_cmd = sub.add_parser('hash', help='Print the stored hash for a file or directory')
    hash_cmd.add_argument('path', help='File or directory')
    hash_cmd.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='List hashes for all files under the directory')

    # media mv <src> <dst> - retired (kept as a parser so old scripts get a clear
    # message instead of "unknown command", since content-hash identity means a plain
    # 'media add' after a move re-links everything automatically)
    mv = sub.add_parser('mv', help='(retired) content is tracked by hash now — just run media add')
    mv.add_argument('src', help='Old path (relative to repo root)')
    mv.add_argument('dst', help='New path (relative to repo root)')

    # media rm <path> - untrack a file or directory (like `git rm --cached`): removes
    # it from the index (media.db) so it stops showing up anywhere, but never touches
    # the file on disk and never touches manual.db's curated data (tags/faces/sets/
    # etc, keyed by content checksum) — re-running `media add` on the same content
    # later reattaches all of that automatically.
    rm_cmd = sub.add_parser('rm', help='Untrack a file or directory (index only — never deletes from disk)')
    rm_cmd.add_argument('path', help='Tracked file or directory path (relative to repo root) to untrack')

    # media commit [path] [--with-full-ml] - scan, optionally followed by the full
    # ML pipeline (metadata, YOLO index, CLIP embed, face detect) in one go
    commit = sub.add_parser('commit', help='Scan a directory, optionally running the full ML pipeline too')
    commit.add_argument('path', nargs='?', default='.', help='Directory to process (default: .)')
    commit.add_argument('-r', '--recursive', action='store_true', default=True,
                         help='Scan recursively')
    commit.add_argument('--with-full-ml', action='store_true',
                         help='Also run metadata, YOLO indexing, CLIP embedding, and face detection')
    commit.add_argument('--reindex', action='store_true',
                         help='Force reprocessing (clear detections/faces/embedding) for already-tracked files found in this scan')

    # media status - show tracked paths that are no longer found on disk
    status = sub.add_parser('status', help='Show tracked paths missing from disk (moved or deleted)')
    status.add_argument('--limit', type=int, default=100,
                         help='Number of files to list')

    # media ls - list all files
    ls = sub.add_parser('ls', help='List all tracked files')
    ls.add_argument('--limit', type=int, default=100,
                     help='Number of files to list')

    # media duplicates - list content seen at more than one path
    duplicates = sub.add_parser('duplicates', help='List content that exists at more than one path')
    duplicates.add_argument('--limit', type=int, default=200,
                             help='Number of duplicate groups to list')

    # media count - count files in repo
    count = sub.add_parser('count', help='Count tracked files')
    count.add_argument('--limit',  type=int, default=None,
                        help='Maximum rows to count')

    # media find_broken - discover corrupted images/videos
    find_broken = sub.add_parser('find_broken', help='Find and mark corrupted images/videos')
    find_broken.add_argument('path', nargs='?', default='.',
                             help='Directory to scan (relative to repo root)')
    find_broken.add_argument('-j', '--jobs', type=int, default=8,
                              help='Concurrent workers (default: 8)')

    # media index [path] - YOLO-World detect objects in images
    index_cmd = sub.add_parser('index', help='Detect objects in images using YOLO-World')
    index_cmd.add_argument('path', nargs='?', default='.',
                           help='Directory to index (relative to repo root, default: .)')
    index_cmd.add_argument('--batch-size', type=int, default=32,
                           help='Images per batch (default: 32)')
    index_cmd.add_argument('--model-size', choices=['s', 'm', 'l', 'x'], default='s',
                           help='YOLO-World model size (default: s)')
    index_cmd.add_argument('--conf', type=float, default=0.15,
                           help='Detection confidence threshold (default: 0.15)')
    index_cmd.add_argument('--reindex', action='store_true',
                           help='Clear existing detections and re-run on all images')

    # media search <query> - search by detected object class (YOLO-World)
    search_cmd = sub.add_parser('search', help='Search images by detected object class (YOLO-World)')
    search_cmd.add_argument('query', help='Text query to search for')
    search_cmd.add_argument('--limit', type=int, default=20,
                            help='Number of results to return (default: 20)')

    # media embed [path] - build CLIP embeddings for "find similar"
    embed_cmd = sub.add_parser('embed', help='Build CLIP embeddings for visual similarity search')
    embed_cmd.add_argument('path', nargs='?', default='.',
                           help='Directory to embed (relative to repo root, default: .)')
    embed_cmd.add_argument('--batch-size', type=int, default=32,
                           help='Images per batch (default: 32)')

    # media faces [path] - detect and embed faces
    faces_cmd = sub.add_parser('faces', help='Detect and embed faces in images using InsightFace')
    faces_cmd.add_argument('path', nargs='?', default='.',
                           help='Directory to scan (relative to repo root, default: .)')
    faces_cmd.add_argument('--batch-size', type=int, default=16,
                           help='Images per batch (default: 16)')
    faces_cmd.add_argument('--model', choices=['buffalo_l', 'buffalo_sc'], default='buffalo_l',
                           help='InsightFace model pack (default: buffalo_l)')
    faces_cmd.add_argument('--thresh', type=float, default=0.5,
                           help='Detection confidence threshold (default: 0.5)')

    # media bodies [path] - crop + embed person boxes for find-by-body search
    bodies_cmd = sub.add_parser('bodies', help='Crop and embed person boxes for find-by-body search')
    bodies_cmd.add_argument('path', nargs='?', default='.',
                            help='Directory to scan (relative to repo root, default: .)')
    bodies_cmd.add_argument('--batch-size', type=int, default=16,
                            help='Images per progress update (default: 16)')

    # media metadata [path] - read EXIF capture time + GPS coordinates
    metadata_cmd = sub.add_parser('metadata', help='Read EXIF capture time and GPS coordinates for images')
    metadata_cmd.add_argument('path', nargs='?', default='.',
                              help='Directory to check (relative to repo root, default: .)')

    # media geo fetch-cities - download the offline GeoNames city database (CC-BY 4.0)
    # into media.db, so photos can be reverse-geocoded to a nearest-city NAME instead
    # of raw coordinates (see geonames.py). One-time setup; no runtime network calls.
    geo_cmd = sub.add_parser('geo', help='Offline city database (GeoNames) for reverse-geocoding photos to city names')
    geo_sub = geo_cmd.add_subparsers(dest='geo_cmd', required=True)
    geo_sub.add_parser('fetch-cities',
                       help='Download the GeoNames cities15000 dump (~26k cities, CC-BY 4.0) into media.db')


    # media who <image> - find who appears in an image
    who_cmd = sub.add_parser('who', help='Find which people appear in an image')
    who_cmd.add_argument('image', help='Path to image file')
    who_cmd.add_argument('--thresh', type=float, default=0.4,
                         help='Similarity threshold (default: 0.4)')
    who_cmd.add_argument('--limit', type=int, default=10,
                         help='Number of results (default: 10)')

    # media set <create|ls|assign> - manage image sets (name + studio)
    set_cmd = sub.add_parser('set', help='Manage image sets (named collections, e.g. a studio shoot)')
    set_sub = set_cmd.add_subparsers(dest='set_cmd', required=True)

    set_create = set_sub.add_parser('create', help='Create a set')
    set_create.add_argument('name', help='Set name')
    set_create.add_argument('--studio', default=None, help='Studio name (optional)')

    set_ls = set_sub.add_parser('ls', help='List all sets')

    set_assign = set_sub.add_parser('assign', help='Assign a file to a set (creates the set if needed)')
    set_assign.add_argument('path', help='File path (relative to repo root)')
    set_assign.add_argument('name', help='Set name')
    set_assign.add_argument('--studio', default=None, help='Studio name (optional)')

    set_files = set_sub.add_parser('files', help='List files in a set')
    set_files.add_argument('name', help='Set name')
    set_files.add_argument('--studio', default=None, help='Studio name (optional)')
    set_files.add_argument('--limit', type=int, default=200, help='Number of files to list')

    # media dir2set <path> [<path> ...] - the common "this whole folder is one
    # set" workflow (a studio shoot, an event, a trip) in one command:
    # scans/imports each folder like `media add`, creates a set named after
    # its own name (override with --name, only valid for a single folder),
    # then adds every currently-tracked photo under it to that set — same
    # matching db.find_files_under_folder already does for the web UI's "add
    # folder to set". nargs='+' means a shell glob like `dir2set folder/*`
    # (expanded by the shell into one argument per subfolder before this ever
    # runs) creates one set per subfolder in a single invocation.
    dir2set = sub.add_parser('dir2set', help="Import folder(s), each into its own set named after it")
    dir2set.add_argument('path', nargs='+', help="Directory (or directories, e.g. via a shell glob) to import — "
                                                  "set name defaults to each folder's own name")
    dir2set.add_argument('--name', default=None, help="Set name (default: the folder's own name; only valid with a single path)")
    dir2set.add_argument('--studio', default=None, help='Studio name (optional)')
    dir2set.add_argument('--reindex', action='store_true',
                          help='Force reprocessing (clear detections/faces/embedding) for already-tracked files found in this scan')

    # media category <create|ls|assign|clear|files|match> - single-value,
    # ML-assisted classification (e.g. Outside/Convention/Anime)
    category_cmd = sub.add_parser('category', help='Manage image categories (multi-value, ML-assisted classification)')
    category_sub = category_cmd.add_subparsers(dest='category_cmd', required=True)

    category_create = category_sub.add_parser('create', help='Create a category')
    category_create.add_argument('name', help='Category name')
    category_create.add_argument('--temperature', type=float, default=0.75,
                                  help='Similarity threshold 0.0-1.0 for auto-matching (default: 0.75)')

    category_ls = category_sub.add_parser('ls', help='List all categories')

    category_assign = category_sub.add_parser(
        'assign', help='Manually assign a file to a category (creates the category if needed)')
    category_assign.add_argument('path', help='File path (relative to repo root)')
    category_assign.add_argument('name', help='Category name')

    category_clear = category_sub.add_parser(
        'clear',
        help="Remove one category from a file (a file can have several; other "
             "categories it has are untouched)")
    category_clear.add_argument('path', help='File path (relative to repo root)')
    category_clear.add_argument('name', help='Category name to remove')

    category_files = category_sub.add_parser('files', help='List files resolved to a category (manual + auto-matched)')
    category_files.add_argument('name', help='Category name')
    category_files.add_argument('--limit', type=int, default=200, help='Number of files to list')

    category_match = category_sub.add_parser('match', help='Run CLIP-similarity auto-matching against category examples')
    category_match.add_argument('path', nargs='?', default='.', help='Path to scope matching to (default: whole repo)')
    category_match.add_argument('--dry-run', action='store_true', help='Report matches without writing to the DB')

    # media age-setup - build the isolated MiVOLO venv (machine-level, not per-repo)
    age_setup = sub.add_parser(
        'age-setup',
        help='Set up age/gender estimation (installs MiVOLO into its own isolated venv)')
    age_setup.add_argument('--dest', default=None,
                           help='Venv directory (default: ~/.local/share/media_manager/age-venv)')
    age_setup.add_argument('--force', action='store_true',
                           help='Delete and recreate the venv if it already exists')

    # media migrate - deliberate, backed-up, verified schema migrations (currently:
    # promoting tags to a real id'd table) — never run silently/automatically the
    # way simple additive schema changes elsewhere in this app are.
    migrate_cmd = sub.add_parser(
        'migrate',
        help='Check for and apply pending database schema migrations (dry-run by default)')
    migrate_cmd.add_argument('--apply', action='store_true',
                             help='Actually perform the migration (backs up manual.db first). '
                                  'Without this flag, only reports what would change.')

    # media web - launch the FastAPI gallery server
    web_cmd = sub.add_parser('web', help='Start the web gallery UI at https://localhost:8000')
    web_cmd.add_argument('host_pos', nargs='?', metavar='host', default=None,
                         help='Host to bind to, e.g. 192.168.1.231 to expose on the local '
                              'network (shorthand for --host)')
    web_cmd.add_argument('--host', default='127.0.0.1',
                         help='Host to bind to (default: 127.0.0.1)')
    web_cmd.add_argument('--port', type=int, default=8000,
                         help='Port to listen on (default: 8000)')
    web_cmd.add_argument('--http', action='store_true',
                         help='Serve over plain HTTP. By default the gallery is served over '
                              'HTTPS using a self-signed cert (auto-generated via the openssl '
                              'binary, cached in .media/).')
    web_cmd.add_argument('--workers', type=int, default=1,
                         help='Number of worker processes (default: 1). Each extra process '
                              'independently holds its own ML models / embedding matrices, so '
                              'RAM multiplies by worker count — the top cause of web-server OOM. '
                              'Raise it only if you offload ML to a `media worker` (so the web '
                              'process stays model-free) and need more request concurrency.')

    worker_cmd = sub.add_parser('worker',
                                help='Run the Reticulum media worker server (heavy ML offload target)')
    worker_cmd.add_argument('--identity-file', default=None,
                            help='Path to the RNS identity file (default: '
                                 '~/.config/media_manager/worker_identity)')
    worker_cmd.add_argument('--config-dir', default=None,
                            help='RNS config directory (default: RNS default)')
    worker_cmd.add_argument('--announce-interval', type=int, default=300,
                            help='Seconds between destination announces (default: 300)')
    worker_cmd.add_argument('--preload', action='store_true',
                            help='Load all ML models up front instead of lazily on first request')

    worker_connect_cmd = sub.add_parser('worker-connect',
                                        help='Point this host at a media worker to offload heavy ML')
    worker_connect_cmd.add_argument('address', help='Worker RNS destination hash (hex)')
    worker_connect_cmd.add_argument('--disable', action='store_true',
                                    help='Save the address but leave offloading disabled')

    args = parser.parse_args()

    # strict-check helper
    def _ensure_repo():
        # `worker` is the remote-offload server: it runs on a device that has no
        # media library of its own, so it must never require a .media/ repo.
        if args.cmd == 'worker':
            return
        if args.strict and not os.path.isdir('.media'):
            print("ERROR: no media repo found (.media/ missing)", file=sys.stderr)
            sys.exit(1)

    # ---- command dispatch -------------------------------------------------
    if args.cmd == 'init':
        if os.path.isdir('.media'):
            print("ERROR: media repository already exists", file=sys.stderr)
            return 1
        os.makedirs('.media', exist_ok=True)
        # initialise empty db file inside .media/
        from media_manager.database import Database
        Database(os.path.join('.media', 'media.db')).close()
        print("Initialized media repository")
        return 0

    if args.cmd == 'age-setup':
        # Machine-level setup (like init, runs outside any media repo). MiVOLO can't be
        # a pip extra of this package — its pinned old ultralytics/timm conflict with
        # the main app's detector/indexer, so it gets its own venv + subprocess bridge.
        from media_manager.age_estimator import setup_age_venv
        try:
            setup_age_venv(dest=args.dest, force=args.force)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: age-setup failed while running: {' '.join(map(str, exc.cmd))}",
                  file=sys.stderr)
            return 1
        return 0

    # every other command first checks that repo exists when --strict is on
    _ensure_repo()

    if args.cmd == 'migrate':
        # Deliberately does NOT go through ManualDB/MediaManager — ManualDB's
        # _create_tables() fail-loud-guards against any pending migration, so a
        # ManualDB can't even be constructed until one's been applied. Operates on
        # the raw manual.db path directly; see media_manager/migrations/ for the
        # list of migrations this checks/runs, in order.
        from media_manager.manual_db import run_migrations
        manual_db_path = os.path.join('.media', 'manual.db')
        try:
            reports = run_migrations(manual_db_path, apply=args.apply)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        for line in reports:
            print(line)
        return 0

    if args.cmd == 'add':
        m = MediaManager()
        # Content is the identity now (see database.py) — hashing happens inline with
        # the scan, so a moved/renamed file is already re-linked to its existing
        # tags/faces/sets by the time this returns. No separate move-tracking pass.
        count, dup_count, already_indexed = m.start_scan(args.path, recursive=args.recursive, reindex=args.reindex)
        print("Scan done,", count, "files")
        if already_indexed:
            if args.reindex:
                print(f"{already_indexed} already-tracked file(s) marked for reindexing.")
            else:
                print(f"{already_indexed} file(s) already indexed. If you want to force reindex run with --reindex")
        m.close()
        if dup_count:
            print(f"WARNING: {dup_count} duplicate file(s) found and NOT tracked — see duplicates.txt", file=sys.stderr)
        return 0

    elif args.cmd == 'hash':
        m = MediaManager()
        files = m.get_hash_list(args.path, recursive=args.recursive)
        if not files:
            print("ERROR: no hashes found", file=sys.stderr)
            sys.exit(1)
        for path, digest in files:
            print(f"{path}\t{digest}")
        m.close()
        return 0

    elif args.cmd == 'mv':
        print("'media mv' is retired — content is tracked by hash now, not path, so "
              "a plain 'media add' after moving/renaming a file re-links it to its "
              "existing tags/faces/sets automatically. Nothing to record.", file=sys.stderr)
        return 0

    elif args.cmd == 'rm':
        m = MediaManager()
        paths_removed, files_removed = m.db.remove_paths_under(args.path)
        m.close()
        if paths_removed == 0:
            print(f"No tracked file or directory found at '{args.path}'.", file=sys.stderr)
            return 1
        print(f"Untracked {paths_removed} path(s) ({files_removed} file(s) fully removed from the index).")
        return 0

    elif args.cmd == 'commit':
        m = MediaManager()
        count, dup_count, already_indexed = m.start_scan(args.path, recursive=args.recursive, reindex=args.reindex)
        step_label = "[1/5] Scan" if args.with_full_ml else "Scan"
        print(f"{step_label} done, {count} files")
        if already_indexed:
            if args.reindex:
                print(f"{already_indexed} already-tracked file(s) marked for reindexing.")
            else:
                print(f"{already_indexed} file(s) already indexed. If you want to force reindex run with --reindex")
        if dup_count:
            print(f"WARNING: {dup_count} duplicate file(s) found and NOT tracked — see duplicates.txt", file=sys.stderr)

        if not args.with_full_ml:
            m.close()
            return 0

        checked, found = m.extract_metadata(args.path)
        print(f"[2/5] Metadata: checked {checked} images, found EXIF data in {found}")

        indexed, failed = m.index_files(args.path)
        print(f"[3/5] Indexed: detected objects in {indexed} images, {failed} failed")

        embedded, embed_failed = m.embed_files(args.path)
        print(f"[4/5] Embedded: {embedded} images, {embed_failed} failed")

        face_indexed, face_count, face_failed = m.detect_faces(args.path)
        print(f"[5/5] Faces: processed {face_indexed} images, found {face_count} faces, {face_failed} failed")

        m.close()
        return 0

    elif args.cmd == 'status':
        m = MediaManager()
        stale = m.get_stale_paths(limit=args.limit)
        for path, status, detail in stale:
            if status == 'moved':
                print(f"moved\t{path} -> {detail}")
            else:
                print(f"missing\t{path}")
        m.close()
        return 0

    elif args.cmd == 'ls':
        m = MediaManager()
        for file_id, path, size, checksum in m.list_files(limit=args.limit):
            print(f"{path}\t[{size} bytes]\t[{checksum}]")
        m.close()
        return 0

    elif args.cmd == 'duplicates':
        m = MediaManager()
        for file_id, path_count in m.find_duplicates(limit=args.limit):
            paths = [p for p, _ in m.get_paths_for_file(file_id)]
            print(f"file_id={file_id} ({path_count} copies):")
            for p in paths:
                print(f"    {p}")
        m.close()
        return 0

    elif args.cmd == 'count':
        m = MediaManager()
        total = m.count_files(limit=args.limit)
        print(total)
        m.close()
        return 0

    elif args.cmd == 'find_broken':
        m = MediaManager()
        broken = m.find_broken(args.path, max_workers=args.jobs)
        print(f"Found {broken} broken files")
        m.close()
        return 0

    elif args.cmd == 'index':
        m = MediaManager()
        if args.reindex:
            m.db.conn.execute('DELETE FROM detections')
            m.db.conn.commit()
            print("Cleared existing detections.")
        indexed, failed = m.index_files(args.path, model_size=args.model_size, conf_threshold=args.conf)
        print(f"Done: detected objects in {indexed} images, {failed} failed")
        m.close()
        return 0

    elif args.cmd == 'search':
        m = MediaManager()
        results = m.search(args.query, limit=args.limit)
        for path, score in results:
            print(f"{score:.4f}\t{path}")
        m.close()
        return 0

    elif args.cmd == 'embed':
        m = MediaManager()
        indexed, failed = m.embed_files(args.path, batch_size=args.batch_size)
        print(f"Done: embedded {indexed} images, {failed} failed")
        m.close()
        return 0

    elif args.cmd == 'set':
        m = MediaManager()
        if args.set_cmd == 'create':
            set_id = m.manual.create_set(args.name, args.studio)
            print(f"Set '{args.name}'" + (f" ({args.studio})" if args.studio else "") + f" — id {set_id}")

        elif args.set_cmd == 'ls':
            sets = m.manual.list_sets()
            if not sets:
                print("No sets yet. Create one with: media set create <name> [--studio STUDIO]")
            for row in sets:
                studio = f" ({row['studio']})" if row['studio'] else ""
                print(f"{row['id']}\t{row['name']}{studio}\t{row['image_count']} images")

        elif args.set_cmd == 'assign':
            file_row = m.db.get_file_by_path(args.path)
            if file_row is None:
                print(f"ERROR: '{args.path}' not tracked — run 'media add' first", file=sys.stderr)
                m.close()
                sys.exit(1)
            set_id = m.manual.create_set(args.name, args.studio)
            m.manual.assign_file_to_set(file_row['checksum'], set_id)
            print(f"Assigned '{args.path}' to set '{args.name}'" + (f" ({args.studio})" if args.studio else ""))

        elif args.set_cmd == 'files':
            row = m.manual.find_set(args.name, args.studio)
            if row is None:
                print(f"ERROR: no set named '{args.name}'" + (f" ({args.studio})" if args.studio else ""), file=sys.stderr)
                m.close()
                sys.exit(1)
            for checksum in m.manual.get_files_by_set(row['id'], limit=args.limit):
                file_row = m.db.get_file_by_checksum(checksum)
                if file_row is not None:
                    print(file_row['path'])
        m.close()
        return 0

    elif args.cmd == 'dir2set':
        if args.name and len(args.path) > 1:
            print("ERROR: --name only makes sense with a single folder — drop it to name each "
                  "set after its own folder, or run this once per folder", file=sys.stderr)
            sys.exit(1)

        m = MediaManager()
        exit_code = 0
        for path in args.path:
            abs_path = os.path.join(m.data_root, path)
            if not os.path.isdir(abs_path):
                print(f"ERROR: '{path}' is not a directory", file=sys.stderr)
                exit_code = 1
                continue
            # Paths are stored relative to data_root with '/' separators (see
            # fast_scan.py) — find_files_under_folder needs that same form, not
            # whatever form the path was typed in (absolute, './'-relative, etc).
            rel_path = os.path.relpath(abs_path, m.data_root).replace(os.sep, '/').strip('/')
            set_name = args.name or os.path.basename(rel_path)
            if not set_name:
                print(f"ERROR: could not derive a set name from '{path}' — pass --name explicitly", file=sys.stderr)
                exit_code = 1
                continue

            count, dup_count, already_indexed = m.start_scan(path, recursive=True, reindex=args.reindex)
            print(f"Scan done, {count} files")
            if already_indexed:
                if args.reindex:
                    print(f"{already_indexed} already-tracked file(s) marked for reindexing.")
                else:
                    print(f"{already_indexed} file(s) already indexed. If you want to force reindex run with --reindex")
            if dup_count:
                print(f"WARNING: {dup_count} duplicate file(s) found and NOT tracked — see duplicates.txt", file=sys.stderr)

            set_id = m.manual.create_set(set_name, args.studio)
            matches = m.db.find_files_under_folder(rel_path)
            existing = set(m.manual.get_files_by_set(set_id, limit=1000000))
            added = 0
            for row in matches:
                checksum = row['checksum']
                if checksum in existing:
                    continue
                m.manual.assign_file_to_set(checksum, set_id)
                existing.add(checksum)
                added += 1
            studio_suffix = f" ({args.studio})" if args.studio else ""
            print(f"Set '{set_name}'{studio_suffix} — id {set_id}: added {added} of {len(matches)} matched "
                  f"photo(s) to the set ({len(matches) - added} already there).")
        m.close()
        return exit_code

    elif args.cmd == 'category':
        m = MediaManager()
        if args.category_cmd == 'create':
            category_id = m.manual.create_category(args.name, args.temperature)
            cat = m.manual.get_category(category_id)
            print(f"Category '{args.name}' (temperature {cat['temperature']}) — id {category_id}")

        elif args.category_cmd == 'ls':
            categories = m.manual.list_categories()
            if not categories:
                print("No categories yet. Create one with: media category create <name> [--temperature 0.0-1.0]")
            for row in categories:
                print(f"{row['id']}\t{row['name']}\ttemp={row['temperature']}\t{row['image_count']} files")

        elif args.category_cmd == 'assign':
            file_row = m.db.get_file_by_path(args.path)
            if file_row is None:
                print(f"ERROR: '{args.path}' not tracked — run 'media add' first", file=sys.stderr)
                m.close()
                sys.exit(1)
            category_id = m.manual.create_category(args.name)
            m.manual.add_file_category(file_row['checksum'], category_id)
            print(f"Assigned '{args.path}' to category '{args.name}'")

        elif args.category_cmd == 'clear':
            file_row = m.db.get_file_by_path(args.path)
            if file_row is None:
                print(f"ERROR: '{args.path}' not tracked — run 'media add' first", file=sys.stderr)
                m.close()
                sys.exit(1)
            cat_row = m.manual.find_category(args.name)
            if cat_row is None:
                print(f"ERROR: no category named '{args.name}'", file=sys.stderr)
                m.close()
                sys.exit(1)
            m.manual.remove_file_category(file_row['checksum'], cat_row['id'])
            print(f"Removed category '{args.name}' from '{args.path}'")

        elif args.category_cmd == 'files':
            row = m.manual.find_category(args.name)
            if row is None:
                print(f"ERROR: no category named '{args.name}'", file=sys.stderr)
                m.close()
                sys.exit(1)
            from media_manager.category_resolver import get_resolved_checksums_for_category
            checksums = get_resolved_checksums_for_category(m.manual, m.db, row['id'], row['name'], limit=args.limit)
            for checksum in checksums:
                file_row = m.db.get_file_by_checksum(checksum)
                if file_row is not None:
                    print(file_row['path'])

        elif args.category_cmd == 'match':
            matched, skipped, considered = m.match_categories(args.path, dry_run=args.dry_run)
            verb = "Would match" if args.dry_run else "Matched"
            print(f"{verb} {matched} file(s); skipped {skipped} with a manual override; considered {considered} total.")
        m.close()
        return 0

    elif args.cmd == 'faces':
        m = MediaManager()
        indexed, face_count, failed = m.detect_faces(
            args.path,
            batch_size=args.batch_size,
            model_name=args.model,
            det_thresh=args.thresh,
        )
        print(f"Done: processed {indexed} images, found {face_count} faces, {failed} failed")
        m.close()
        return 0

    elif args.cmd == 'bodies':
        m = MediaManager()
        indexed, failed = m.index_bodies(args.path, batch_size=args.batch_size)
        print(f"Done: body-indexed {indexed} images, {failed} failed")
        m.close()
        return 0

    elif args.cmd == 'metadata':
        m = MediaManager()
        checked, found = m.extract_metadata(args.path)
        print(f"Done: checked {checked} images, found EXIF data in {found}")
        m.close()
        return 0

    elif args.cmd == 'geo':
        if args.geo_cmd == 'fetch-cities':
            from media_manager import geonames
            m = MediaManager()
            try:
                count = geonames.fetch_cities(m.db)
                print(f"Loaded {count} cities. Now run the '🏙 Match cities' bulk action "
                      "(⚡ menu) or `media metadata` to label photos with their nearest city.")
            finally:
                m.close()
        return 0

    elif args.cmd == 'who':
        m = MediaManager()
        results = m.search_by_face_image(args.image, limit=args.limit, similarity_threshold=args.thresh)
        if not results:
            print("No matching faces found.")
        else:
            for r in results:
                print(f"{r['score']:.3f}\t{r['path']}")
        m.close()
        return 0

    elif args.cmd == 'web':
        _ensure_repo()
        data_root = os.path.abspath('.')
        try:
            import uvicorn
        except ImportError:
            print("ERROR: uvicorn is not installed. Run: pip install 'uvicorn[standard]'",
                  file=sys.stderr)
            return 1
        try:
            from media_manager.web import create_app
        except ImportError as exc:
            print(f"ERROR: could not import web module: {exc}", file=sys.stderr)
            return 1
        try:
            app = create_app(data_root)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        host = args.host_pos or args.host
        ssl_kwargs = {}
        scheme = 'http'
        if not args.http:
            media_dir = os.path.join(data_root, '.media')
            cert = _ensure_self_signed_cert(media_dir, host)
            if cert is not None:
                certfile, keyfile = cert
                ssl_kwargs = {'ssl_certfile': certfile, 'ssl_keyfile': keyfile}
                scheme = 'https'
        print(f"Starting media gallery at {scheme}://{host}:{args.port}/")
        if args.workers > 1:
            # uvicorn's multi-worker mode spawns separate processes that each
            # need to independently construct the app, so it needs an import
            # string/factory rather than the already-built `app` object above
            # (kept and used above only to surface a clean pending-migration
            # error before committing to spawning workers — see create_app's
            # RuntimeError check). data_root travels to each worker via env
            # var since create_app_from_env is called with no arguments.
            os.environ['MEDIA_WEB_DATA_ROOT'] = data_root
            print(f"Running with {args.workers} worker processes.")
            uvicorn.run('media_manager.web:create_app_from_env', factory=True,
                        host=host, port=args.port, workers=args.workers, **ssl_kwargs)
        else:
            uvicorn.run(app, host=host, port=args.port, **ssl_kwargs)
        return 0

    elif args.cmd == 'worker':
        from .worker_server import run as worker_run
        worker_run(identity_file=args.identity_file, config_dir=args.config_dir,
                   announce_interval=args.announce_interval, preload=args.preload)
        return 0

    elif args.cmd == 'worker-connect':
        from . import worker_config
        # Validate the address is well-formed before saving (raises loudly on
        # bad input — no silent acceptance of a garbage hash).
        worker_config.address_hash_bytes(args.address)
        m = MediaManager()
        path = worker_config.save(m.data_root, args.address, enabled=not args.disable)
        state = 'disabled' if args.disable else 'enabled'
        print(f"Saved worker config to {path} (address={args.address}, offloading {state})")
        return 0

    else:
        print("ERROR: unknown command group", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
