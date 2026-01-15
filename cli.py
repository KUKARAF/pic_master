#!/usr/bin/env python3
import argparse
import sys
import os
from media_manager import MediaManager

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    
    scan = sub.add_parser('scan')
    scan.add_argument('path')
    scan.add_argument('-r', '--recursive', action='store_true', default=True)

    unhashed = sub.add_parser('unhashed')
    unhashed.add_argument('--limit', type=int, default=100, help='Number of files to list')

    args = parser.parse_args()

    if args.cmd == 'scan':
        m = MediaManager()
        try:
            m.start_scan(args.path, recursive=args.recursive)
            print("Scan done")
        finally:
            m.close()
    elif args.cmd == 'unhashed':
        m = MediaManager()
        try:
            files = m.get_unhashed_files(limit=args.limit)
            for file_info in files:
                print(file_info[1])
        finally:
            m.close()
    else:
        print("Only 'scan' and 'unhashed' supported")
        return 1

if __name__ == '__main__':
    sys.exit(main())
