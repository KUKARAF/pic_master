#!/usr/bin/env python3
import argparse
import sys
import os
from media_manager import MediaManager

def main():
    parser = argparse.ArgumentParser(description='Media manager - like git for your media files')
    sub = parser.add_subparsers(dest='cmd')
    
    # media init - create .media folder
    init = sub.add_parser('init', help='Initialize a media repository')
    
    # media add <path> - scan directory
    add = sub.add_parser('add', help='Scan a directory for media files')
    add.add_argument('path', help='Directory to scan')
    add.add_argument('-r', '--recursive', action='store_true', default=True, help='Scan recursively')

    # media hash <path> - show stored hash
    hash_cmd = sub.add_parser('hash', help='Print the stored hash for a file or directory')
    hash_cmd.add_argument('path', help='File or directory')
    hash_cmd.add_argument('-r', '--recursive', action='store_true', default=False, help='List hashes for all files under the directory')

    # media mv <src> <dst>
    mv = sub.add_parser('mv', help='Record a file move (rename) in the database')
    mv.add_argument('src', help='Old path (relative to repo root)')
    mv.add_argument('dst', help='New path (relative to repo root)')

    # media commit - hash unhashed files
    commit = sub.add_parser('commit', help='Create hashes for unscanned files')
    commit.add_argument('--batch-size', type=int, default=100, help='Number of files to hash at once')

    # media status - show unhashed files and detect moved files
    status = sub.add_parser('status', help='Show files without hashes and detect moved files')
    status.add_argument('--limit', type=int, default=100, help='Number of files to list')

    # media ls - list all files
    ls = sub.add_parser('ls', help='List all tracked files')
    ls.add_argument('--limit', type=int, default=100, help='Number of files to list')
    ls.add_argument('--hashed', action='store_true', help='Only show hashed files')
    ls.add_argument('--unhashed', action='store_true', help='Only show unhashed files')

    # media hashes - show files with hashes
    hashes = sub.add_parser('hashes', help='Show files with hashes')
    hashes.add_argument('--limit', type=int, default=100, help='Number of files to list')

    args = parser.parse_args()

    if args.cmd == 'init':
        media_dir = os.path.join(os.getcwd(), '.media')
        if os.path.exists(media_dir):
            print("Media repository already exists")
            return 1
        
        os.makedirs(media_dir, exist_ok=True)
        db_path = os.path.join(media_dir, 'media.db')
        
        # Initialize the database
        temp_manager = MediaManager(db_path=db_path)
        temp_manager.close()
        
        print(f"Initialized media repository in {os.getcwd()}")
        return 0
        
    elif args.cmd == 'add':
        m = MediaManager()
        try:
            count = m.start_scan(args.path, recursive=args.recursive)
            print(f"Scan done, {count} files")
        finally:
            m.close()
    elif args.cmd == 'hash':
        m = MediaManager()
        try:
            files = m.get_hash_list(args.path, recursive=args.recursive)
            if not files:
                print("No hashes found", file=sys.stderr)
                return 1
            for path, digest in files:
                print(f"{path}\t{digest}")
        finally:
            m.close()
    elif args.cmd == 'mv':
        m = MediaManager