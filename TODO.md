# Media Manager TODO

## Phase 1: Folder Scan
- Import all files and folders into database
- Implement fast file discovery without hashing
- Store file paths with NULL checksums initially
- Record unix timestamp `last_hashed` when a hash is generated (initially NULL)

## Phase 2: Hashing
- Implement on-demand hashing for individual files
- Add batch hashing for files with NULL checksums
- Support incremental hashing operations
- Generate xxhash for file integrity verification
- Update `last_hashed` timestamp with current unix time after hashing

## Implementation Tasks

### Scanner Module
1. Create FileScanner class with methods:
   - scan_directory(path): Recursively discover files
   - store_file_metadata(path): Store file info in database (size, modified_time, checksum=NULL, last_hashed=NULL)

### Hasher Module
1. Create FileHasher class with methods:
   - get_xxhash(path): Generate hash for a single file
   - update_file_hash(file_id, hash): Update database record and set last_hashed to current unix timestamp
   - process_null_hashes(batch_size=100): Process files without hashes
   - verify_file_integrity(file_id): Compare stored vs computed hash

### Database Module
1. Design database schema:
   - Files table: id, path, size, modified_time, checksum, last_hashed (integer unix timestamp)
   - Implement CRUD operations for the files table
   - Add indexes for performance optimization (path, checksum, last_hashed)

### MediaManager Main Interface
1. Create unified interface:
   - start_scan(path): Begin scanning a directory
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
