# Media Manager TODO

## Phase 1: Folder Scan
- Import all files and folders into database
- Implement fast file discovery without hashing
- Track scan sessions for pause/resume functionality
- Store file paths with NULL checksums initially

## Phase 2: Hashing
- Implement on-demand hashing for individual files
- Add batch hashing for files with NULL checksums
- Support incremental hashing operations
- Generate xxhash for file integrity verification

## Implementation Tasks

### Scanner Module
1. Create FileScanner class with methods:
   - scan_directory(path): Recursively discover files
   - store_file_metadata(path, session_id): Store file info in database
   - create_scan_session(): Start a new scanning session
   - resume_scan_session(session_id): Continue from previous session
   - get_scan_progress(session_id): Check scanning progress

### Hasher Module
1. Create FileHasher class with methods:
   - get_xxhash(path): Generate hash for a single file
   - update_file_hash(file_id, hash): Update database record
   - process_null_hashes(batch_size=100): Process files without hashes
   - verify_file_integrity(file_id): Compare stored vs computed hash

### Database Module
1. Design database schema:
   - Files table: id, path, size, modified_time, checksum, scan_session_id
   - ScanSessions table: id, start_time, end_time, status, root_path
   - Implement CRUD operations for both tables
   - Add indexes for performance optimization

### MediaManager Main Interface
1. Create unified interface:
   - start_scan(path): Begin scanning a directory
   - pause_scan(): Pause current scanning session
   - resume_scan(): Resume paused session
   - hash_files(batch_size=100): Start hashing process
   - get_file_info(path): Retrieve file metadata and hash
   - verify_integrity(path): Verify file integrity

### Additional Features
- Support for multiple hash algorithms (xxhash, md5, sha256)
- Progress reporting for long operations
- Error handling and logging
- Configuration management
- Command-line interface
- Web interface (optional)

## Development Notes
- Use SQLite for initial database implementation
- Focus on performance for large media collections
- Implement proper error recovery mechanisms
- Add comprehensive logging
- Write unit tests for each module
