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
        from .database import Database
        Database(os.path.join('.media', 'media.db')).close()
        print("Initialized media repository")
        return 0

    # every other command first checks that repo exists when --strict is on
    _ensure_repo()

    if args.cmd == 'add':
        m = MediaManager()
        count = m.start_scan(args.path, recursive=args.recursive)
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
        print(f"Processed {processed} files")
        m.close()
        return 0

    elif args.cmd == 'status':
        m = MediaManager()
        for file_info in m.get_unhashed_files(limit=args.limit):
            print(f"unhashed\t{file_info[1]}")
        for old, new, chksum in m.find_moved_candidates(limit=args.limit):
            print(f"moved\t{old} → {new}")
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

    else:
        print("ERROR: unknown command group", file=sys.stderr)
        return 1


if __name__ == '__main__':
    main()
