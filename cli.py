#!/usr/bin/env python3
import argparse
import sys
from media_manager import MediaManager

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    scan = sub.add_parser('scan')
    scan.add_argument('path')
    scan.add_argument('-r', '--recursive', action='store_true', default=True)
    args = parser.parse_args()

    if args.cmd != 'scan':
        print("Only 'scan' supported")
        return 1

    m = MediaManager()
    try:
        m.start_scan(args.path, recursive=args.recursive)
        print("Scan done")
    finally:
        m.close()

if __name__ == '__main__':
    sys.exit(main())
