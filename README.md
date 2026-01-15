# Media Manager

A Python-based media management system for scanning directories and generating file hashes.

## Features

- Fast file discovery without immediate hashing
- On-demand and batch hashing capabilities
- Scan session tracking with pause/resume functionality
- File integrity verification
- Support for multiple hash algorithms

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from media_manager import MediaManager

manager = MediaManager()
manager.start_scan("/path/to/media")
manager.hash_files()
```

## Project Structure

```
media_manager/
├── __init__.py
├── media_manager.py      # Main MediaManager class
├── scanner.py            # FileScanner class (discovers files, no hashing)
├── hasher.py             # FileHasher class (generates hashes on demand)
├── database.py           # Database schema and operations
└── models.py             # Data models (File, ScanSession, etc.)
```

## Development

See [TODO.md](TODO.md) for the development roadmap and planned features.
