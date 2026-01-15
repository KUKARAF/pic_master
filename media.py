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

    # media commit - hash unhashed files
    commit = sub.add_parser('commit', help='Create hashes for unscanned files')
    commit.add_argument('--batch-size', type=int, default=100, help='Number of files to hash at once')

    # media status - show unhashed files
    status = sub.add_parser('status', help='Show files without hashes')
    status.add_argument('--limit', type=int, default=100, help='Number of files to list')

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
            m.start_scan(args.path, recursive=args.recursive)
            print("Scan done")
        finally:
            m.close()
    elif args.cmd == 'commit':
        m = MediaManager()
        try:
            processed = m.hash_files(batch_size=args.batch_size)
            print(f"Processed {processed} files")
        finally:
            m.close()
    elif args.cmd == 'status':
        m = MediaManager()
        try:
            files = m.get_unhashed_files(limit=args.limit)
            for file_info in files:
                print(file_info[1])
        finally:
            m.close()
    else:
        print("Available commands: init, add, commit, status")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
