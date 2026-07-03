#!/usr/bin/env python3
import argparse
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

    # media hash <path> - show stored hash
    hash_cmd = sub.add_parser('hash', help='Print the stored hash for a file or directory')
    hash_cmd.add_argument('path', help='File or directory')
    hash_cmd.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='List hashes for all files under the directory')

    # media mv <src> <dst>
    mv = sub.add_parser('mv', help='Record a file move (rename) in the database')
    mv.add_argument('src', help='Old path (relative to repo root)')
    mv.add_argument('dst', help='New path (relative to repo root)')

    # media commit - hash unhashed files
    commit = sub.add_parser('commit', help='Create hashes for unscanned files')
    commit.add_argument('--batch-size', type=int, default=100,
                         help='Number of files to hash at once')

    # media status - show unhashed files and detect moved files
    status = sub.add_parser('status', help='Show files without hashes and detect moved files')
    status.add_argument('--limit', type=int, default=100,
                         help='Number of files to list')

    # media ls - list all files
    ls = sub.add_parser('ls', help='List all tracked files')
    ls.add_argument('--limit', type=int, default=100,
                     help='Number of files to list')
    ls.add_argument('--hashed', action='store_true',
                   help='Only show hashed files')
    ls.add_argument('--unhashed', action='store_true',
                     help='Only show unhashed files')

    # media hashes - show files with hashes
    hashes = sub.add_parser('hashes', help='Show files with hashes')
    hashes.add_argument('--limit', type=int, default=100,
                         help='Number of files to list')

    # media count - count files in repo
    count = sub.add_parser('count', help='Count tracked files matching criteria')
    count.add_argument('--hashed',   action='store_true',
                        help='Count only hashed files')
    count.add_argument('--unhashed', action='store_true',
                        help='Count only unhashed files')
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

    # media search <query> - semantic image search
    search_cmd = sub.add_parser('search', help='Search images by text using CLIP')
    search_cmd.add_argument('query', help='Text query to search for')
    search_cmd.add_argument('--limit', type=int, default=20,
                            help='Number of results to return (default: 20)')

    # media web - launch the FastAPI gallery server
    web_cmd = sub.add_parser('web', help='Start the web gallery UI at localhost:8000')
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

    # every other command first checks that repo exists when --strict is on
    _ensure_repo()

    if args.cmd == 'add':
        m = MediaManager()
        count = m.start_scan(args.path, recursive=args.recursive)
        # Also update any moved files
        moved_count = m.update_moved_files()
        if moved_count > 0:
            print(f"Scan done, {count} files (updated {moved_count} moved files)")
        else:
            print("Scan done,", count, "files")
        m.close()
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
        m = MediaManager()
        ok = m.record_move(args.src, args.dst)
        if not ok:
            print(f"ERROR: '{args.src}' not tracked", file=sys.stderr)
            sys.exit(1)
        print(f"Recorded move: {args.src} → {args.dst}")
        m.close()
        return 0

    elif args.cmd == 'commit':
        m = MediaManager()
        processed = m.hash_files(batch_size=args.batch_size)
        # Also update any moved files
        moved_count = m.update_moved_files()
        if moved_count > 0:
            print(f"Processed {processed} files (updated {moved_count} moved files)")
        else:
            print(f"Processed {processed} files")
        m.close()
        return 0

    elif args.cmd == 'status':
        m = MediaManager()
        # 1. unhashed list
        files = m.get_unhashed_files(limit=args.limit)
        for file_info in files:
            print(f"unhashed\t{file_info[1]}")
        # 2. moved candidates
        moved = m.find_moved_candidates(limit=args.limit)
        for old, new, chksum in moved:
            print(f"moved\t{old} -> {new}")
        m.close()
        return 0

    elif args.cmd == 'ls':
        m = MediaManager()
        for file_info in m.list_files(
                limit=args.limit,
                hashed_only=args.hashed,
                unhashed_only=args.unhashed):
            path, size, checksum = file_info[1], file_info[2], file_info[4]
            hash_status = "✓" if checksum else "✗"
            print(f"{path}\t[{size} bytes]\t[{hash_status}]")
        m.close()
        return 0

    elif args.cmd == 'hashes':
        m = MediaManager()
        for file_info in m.get_hashed_files(limit=args.limit):
            print(f"{file_info[1]}\t{file_info[4]}")
        m.close()
        return 0

    elif args.cmd == 'count':
        m = MediaManager()
        total = m.count_files(hashed_only=args.hashed,
                              unhashed_only=args.unhashed,
                              limit=args.limit)
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
        print(f"Starting media gallery at http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    else:
        print("ERROR: unknown command group", file=sys.stderr)
        return 1


if __name__ == '__main__':
    main()
