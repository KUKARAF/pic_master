"""Find-only driver for the czkawka CLI (near-duplicate + broken-file detection).

We NEVER let czkawka delete or move anything — it only *lists*; our own reversible
Trash / mark-broken does everything. Every invocation is asserted free of delete/move
flags (`_FORBIDDEN`). Requires ``czkawka_cli`` >= 12 on PATH — ``info()`` returns None
otherwise, which the web layer surfaces as "czkawka not found, deduplication will not
be available". The similar-VIDEOS scan additionally needs ffmpeg/ffprobe; the broken-file
scan validates images/PDF/archive/music by content only (not video streams).

Verified against czkawka v12.x: scan subcommands are list-only by default; JSON is
written to a file (``-C``), never stdout; a scan that FINDS items exits 11 (not an
error) unless ``-W`` is passed.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

# Flags that make czkawka mutate the filesystem — MUST never appear in a command.
_FORBIDDEN = {'-D', '--delete-method', '--delete-files', '-y', '--move-to-trash'}


def info():
    """{'path','version','major'} for czkawka_cli on PATH, or None if not installed.
    `version` is the bare semver (e.g. '12.0.0'); `major` its integer major component."""
    path = shutil.which('czkawka_cli')
    if not path:
        return None
    try:
        out = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return None
    m = re.search(r'(\d+)\.(\d+)\.(\d+)', out)
    version = m.group(0) if m else (out.strip().splitlines() or [''])[0]
    return {'path': path, 'version': version, 'major': int(m.group(1)) if m else None}


def has_ffmpeg():
    """Similar-videos + broken-video checks need both ffmpeg and ffprobe on PATH."""
    return bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))


def _run(subcmd, directories, extra):
    """Run one FIND-ONLY czkawka scan; return the parsed JSON. Raises on a real failure
    (rc not in {0, 11}) or if any mutating flag slipped into `extra` (belt-and-braces)."""
    nfo = info()
    if nfo is None:
        raise RuntimeError('czkawka_cli not found')
    bad = _FORBIDDEN.intersection(extra)
    if bad:
        raise AssertionError(f'refusing to run czkawka with mutating flag(s): {sorted(bad)}')
    dirs = [str(d) for d in directories]
    fd, out_path = tempfile.mkstemp(suffix='.json', prefix='czkawka_')
    os.close(fd)
    try:
        # -W: don't exit non-zero when items are found. -N: suppress the stdout results
        # (we read the JSON). -C: compact JSON to a file (JSON never goes to stdout).
        cmd = [nfo['path'], subcmd, '-d', *dirs, '-W', '-N', '-C', out_path, *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode not in (0, 11):   # 11 = "items found", not an error
            raise RuntimeError(f'czkawka {subcmd} failed (rc={r.returncode}): '
                               f'{(r.stderr or r.stdout or "").strip()[:500]}')
        if os.path.getsize(out_path) == 0:
            return []
        with open(out_path) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _extract_groups(data):
    """Normalize any czkawka similarity JSON shape into a list of groups, each a list of
    member entry dicts ({path,width,height,size,difference,...}). Handles the normal
    Vec<Vec<entry>>, the reference-mode Vec<[ref,[members]]>, and dup's size-nested
    Vec<Vec<Vec<entry>>> — by recursively collecting any list whose items are entries."""
    groups = []

    def is_entry(e):
        return isinstance(e, dict) and 'path' in e

    def walk(node):
        if not isinstance(node, list) or not node:
            return
        if all(is_entry(e) for e in node):                    # a group of entries
            groups.append(node)
            return
        if len(node) == 2 and is_entry(node[0]) and isinstance(node[1], list):
            groups.append([node[0]] + [e for e in node[1] if is_entry(e)])  # reference mode
            return
        for child in node:
            walk(child)

    walk(data or [])
    return groups


# czkawka's default --minimal-file-size is 16384 (8192 for video): it SILENTLY skips any
# file under that. A photo library has plenty of small images (thumbnails, screenshots,
# heavily-compressed pics), so we drop the floor to 1 byte — better a little extra work
# than invisibly missing near-duplicates. (0 is rejected: "must be at least 1 byte".)
_MIN_SIZE = '1'


def similar_images(directories, max_difference=6, hash_size=16, hash_alg='Gradient'):
    """Groups of visually-similar images. `max_difference` is a 0..40 hamming tolerance
    (2 ≈ near-identical, 15 ≈ loose) at `hash_size` 8/16/32/64."""
    return _extract_groups(_run('image', directories,
                                ['-s', str(max_difference), '-c', str(hash_size),
                                 '-g', hash_alg, '-m', _MIN_SIZE]))


def similar_videos(directories, tolerance=10, scan_duration=10):
    """Groups of visually-similar videos (needs ffmpeg). [] when ffmpeg is absent."""
    if not has_ffmpeg():
        return []
    return _extract_groups(_run('video', directories,
                                ['-t', str(tolerance), '-A', str(scan_duration), '-m', _MIN_SIZE]))


def exact_duplicates(directories):
    """Groups of byte-identical files (BLAKE3). Note: our content-addressed storage
    already collapses identical bytes to one file, so most of these map to a single
    file_id downstream — kept for completeness."""
    return _extract_groups(_run('dup', directories, ['-s', 'HASH', '-t', 'BLAKE3', '-m', _MIN_SIZE]))


def broken_files(directories, allowed_extensions='IMAGE'):
    """FLAT list of {path, type_of_file, error_string, size, ...} for corrupt/undecodable
    files. czkawka's `broken` validates images, PDFs, archives and music by content; it does
    NOT decode video streams, so corrupt videos are out of scope here (the on-decode-failure
    marking + `media find_broken` cover those). `allowed_extensions` limits the scan (czkawka
    macro, IMAGE by default); pass a falsy value to check every type czkawka supports."""
    extra = ['-x', allowed_extensions] if allowed_extensions else []
    data = _run('broken', directories, extra)
    return list(data or [])
