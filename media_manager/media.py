#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from media_manager import MediaManager

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

    # media age-setup - build the isolated MiVOLO venv (machine-level, not per-repo)
    age_setup = sub.add_parser(
        'age-setup',
        help='Set up age/gender estimation (installs MiVOLO into its own isolated venv)')
    age_setup.add_argument('--dest', default=None,
                           help='Venv directory (default: ~/.local/share/media_manager/age-venv)')
    age_setup.add_argument('--force', action='store_true',
                           help='Delete and recreate the venv if it already exists')

    # media web - launch the FastAPI gallery server
    web_cmd = sub.add_parser('web', help='Start the web gallery UI at localhost:8000')
    web_cmd.add_argument('host_pos', nargs='?', metavar='host', default=None,
                         help='Host to bind to, e.g. 192.168.1.231 to expose on the local '
                              'network (shorthand for --host)')
    web_cmd.add_argument('--host', default='127.0.0.1',
                         help='Host to bind to (default: 127.0.0.1)')
    web_cmd.add_argument('--port', type=int, default=8000,
                         help='Port to listen on (default: 8000)')

    args = parser.parse_args()

    # strict-check helper
    def _ensure_repo():
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
        print(f"Starting media gallery at http://{host}:{args.port}/")
        uvicorn.run(app, host=host, port=args.port)
        return 0

    else:
        print("ERROR: unknown command group", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
