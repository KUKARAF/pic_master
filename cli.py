#!/usr/bin/env python3
"""
CLI interface for the MediaManager
"""
import argparse
import os
import sys
from media_manager import MediaManager

def parse_args():
    parser = argparse.ArgumentParser(description='Media Manager CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory for media files')
    scan_parser.add_argument('path', help='Path to scan (relative to ./data/)')
    scan_parser.add_argument('-r', '--recursive', action='store_true', 
                           help='Scan recursively (default: True)')
    scan_parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                           help='Scan only the immediate directory')
    scan_parser.set_defaults(recursive=True)

    # Hash command
    hash_parser = subparsers.add_parser('hash', help='Hash unprocessed files')
    hash_parser.add_argument('--batch-size', type=int, default=100,
                           help='Number of files to hash in one batch')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify file integrity')
    verify_parser.add_argument('path', help='Path to verify (relative to ./data/)')

    # Info command
    info_parser = subparsers.add_parser('info', help='Get file info')
    info_parser.add_argument('path', help='Path to get info about (relative to ./data/)')

    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use -h for help.")
        return 1

    # Initialize MediaManager
    manager = MediaManager()

    try:
        if args.command == 'scan':
            print(f"Scanning: {args.path}")
            success = manager.start_scan(args.path, recursive=args.recursive)
            if success:
                print("Scan completed successfully")
            else:
                print("Scan failed")
                return 1

        elif args.command == 'hash':
            print(f"Processing files in batches of {args.batch_size}...")
            processed = manager.hash_files(batch_size=args.batch_size)
            print(f"Processed {processed} files")

        elif args.command == 'verify':
            print(f"Verifying: {args.path}")
            if manager.verify_integrity(args.path):
                print("File integrity verified")
            else:
                print("File integrity check failed")
                return 1

        elif args.command == 'info':
            info = manager.get_file_info(args.path)
            if info:
                print(f"Path: {info['path']}")
                print(f"Size: {info['size']} bytes")
                print(f"Modified: {info['modified_time']}")
                print(f"Checksum: {info['checksum'] or 'Not computed'}")
                print(f"Last hashed: {info['last_hashed'] or 'Never'}")
            else:
                print("File not found")
                return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        manager.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())
